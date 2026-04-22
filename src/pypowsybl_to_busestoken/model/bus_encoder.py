"""
BusEncoder — Graph Transformer Encoder for Power Grid Bus Embeddings.

Architecture:
    InputProjection  : 17 → 512
    BusEncoderLayer  × 6 :
        PreNorm(RMSNorm) + EdgeGatedAttention(8 heads, d=512) + GatedResidual
        PreNorm(RMSNorm) + SwiGLU FFN(d=512 → 2752 → 512)   + GatedResidual
    SIGReg regularizer (train-time only, optional)

Design choices:
    - RMSNorm instead of LayerNorm: preserves zero semantics for transit buses (p_net=0)
      and voltage-level heterogeneity — no mean-centering bias.

    - EdgeGatedAttention (enriched K/V + score bias):
        Branch features (r, x, tap_ratio, …) enter the attention in two ways:
          (1) Enrichment of K and V vectors: k_e += edge_to_k(e),  v_e += edge_to_v(e)
              → branch type changes WHAT information is extracted from the neighbour bus,
                not just whether it is attended.
          (2) Scalar bias per head: score += MLP(e)
              → global routing signal, independent of query/key content.
        Both signals are complementary and cheap to compute.

    - GatedResidual instead of plain x + h:
        gate = σ( Linear([x_new, x_old, x_new − x_old]) )
        output = x_new * gate + x_old * (1 − gate)
        → each bus learns how much to absorb from the sub-layer update;
          low-signal transit buses can naturally gate down, while highly
          connected buses can absorb stronger updates.
        Initialised with gate ≈ 0.88 (near standard residual) so training
        starts stable and specialises progressively.

    - SwiGLU FFN: gating handles conditional sparsity (transit buses with zero
      active injection produce near-zero gate → nearly silent FFN path).

    - SIGReg: projects final embeddings onto M random directions and penalises
      deviation from standard normality — prevents representation collapse without
      contrastive pairs (single-snapshot forward pass).

Inputs (per snapshot):
    x          : (N, 17)    bus features (float32)
    edge_index : (2, E)     int64, PyG convention [src, dst]
    edge_attr  : (E, 20)    branch / relation features (float32)

Output:
    z_bus      : (N, 512)   bus embeddings (float32)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (no mean centering).

    Equivalent to LayerNorm without the mean-subtraction step, which preserves
    the sign and magnitude semantics of zero-valued features (transit buses).
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    FFN(x) = Linear_out( SiLU(Linear_gate(x)) ⊙ Linear_val(x) )

    d_hidden is set to ⌊8/3 · d_model⌋ rounded to nearest 64 so the
    parameter count is comparable to a plain 4× FFN while gaining the gating
    mechanism.
    """

    def __init__(self, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        if d_hidden is None:
            # 8/3 * d_model, rounded to nearest 64 — matches LLaMA convention
            d_hidden = int(math.ceil((8 / 3 * d_model) / 64) * 64)
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.val = nn.Linear(d_model, d_hidden, bias=False)
        self.proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.gate(x)) * self.val(x))


class GatedResidual(nn.Module):
    """Adaptive gated residual connection (from Dwivedi & Bresson, 2021).

    Instead of the fixed update  x = x + h,  learns per-token how much to
    absorb from the sub-layer output vs. retain the previous state:

        gate_input = [ x_new ∥ x_old ∥ x_new − x_old ]   # (3d,)
        gate        = σ( Linear(gate_input) )               # scalar ∈ (0, 1)
        output      = x_new × gate  +  x_old × (1 − gate)

    The  x_new − x_old  term gives the gate a direct signal of *how large* the
    proposed update is: a large delta → gate opens wider; no change → gate
    suppresses the update.

    Initialisation: gate ≈ 0.88 (bias = +2.0) so training starts near-standard
    residual behaviour and specialises gradually.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d * 3, 1, bias=True)
        # Near-identity initialisation: gate starts at sigmoid(2) ≈ 0.88
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 2.0)

    def forward(self, x_new: torch.Tensor, x_old: torch.Tensor) -> torch.Tensor:
        # x_new, x_old : (N, d)
        gate_input = torch.cat([x_new, x_old, x_new - x_old], dim=-1)  # (N, 3d)
        gate = torch.sigmoid(self.proj(gate_input))                      # (N, 1)
        return x_new * gate + x_old * (1.0 - gate)                      # (N, d)


# ---------------------------------------------------------------------------
# Edge-gated Sparse Multi-head Attention  (enriched K/V + score bias)
# ---------------------------------------------------------------------------


class EdgeGatedAttention(nn.Module):
    """Sparse multi-head attention with branch-feature enrichment of K, V and scores.

    For each directed edge (src → dst):

        k_e  = W_k(x[src])  +  edge_to_k(edge_attr)   # enriched key
        v_e  = W_v(x[src])  +  edge_to_v(edge_attr)   # enriched value
        score = (q_dst · k_e) / √d_head  +  gate_bias  # bias per head

    The K/V enrichment lets branch features change *what information is read*
    from the source bus (not just how much weight to give it).  The scalar gate
    bias adds a content-independent routing signal.

    Sparse softmax is computed per destination node using scatter_reduce /
    scatter_add_ — pure PyTorch, no torch_scatter dependency.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_edge: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.dropout_p = dropout

        # Node projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Edge → enriched K and V (branch modifies *what* is extracted from src)
        self.edge_to_k = nn.Linear(d_edge, d_model, bias=False)
        self.edge_to_v = nn.Linear(d_edge, d_model, bias=False)

        # Edge → scalar score bias per head (content-independent routing)
        d_gate_hidden = max(d_edge * 2, 64)
        self.edge_gate = nn.Sequential(
            nn.Linear(d_edge, d_gate_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(d_gate_hidden, n_heads, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,           # (N, d_model)
        edge_index: torch.Tensor,  # (2, E)  int64
        edge_attr: torch.Tensor,   # (E, d_edge)
    ) -> torch.Tensor:             # (N, d_model)
        N = x.size(0)
        E = edge_attr.size(0)
        H = self.n_heads
        D = self.d_head
        src = edge_index[0]        # (E,)
        dst = edge_index[1]        # (E,)

        # ── Node projections ──────────────────────────────────────────────
        q = self.W_q(x).view(N, H, D)   # (N, H, D)
        k = self.W_k(x).view(N, H, D)
        v = self.W_v(x).view(N, H, D)

        # ── Gather per-edge node representations ─────────────────────────
        q_e = q[dst]               # (E, H, D)  — query from destination bus
        k_e = k[src]               # (E, H, D)  — key   from source bus
        v_e = v[src]               # (E, H, D)  — value from source bus

        # ── Edge K/V enrichment: branch modifies key and value space ─────
        # edge_to_k/v: (E, d_model) → reshape to (E, H, D)
        k_e = k_e + self.edge_to_k(edge_attr).view(E, H, D)
        v_e = v_e + self.edge_to_v(edge_attr).view(E, H, D)

        # ── Attention scores = content (enriched K) + routing bias ───────
        scores = (q_e * k_e).sum(-1) / self.scale   # (E, H)  content
        scores = scores + self.edge_gate(edge_attr)  # (E, H)  + routing bias

        # ── Sparse softmax per destination node ───────────────────────────
        idx_dst = dst.unsqueeze(1).expand(-1, H)     # (E, H)

        # Step 1: max per (dst, head) for numerical stability
        max_scores = torch.full(
            (N, H), float("-inf"), dtype=scores.dtype, device=scores.device
        )
        max_scores = max_scores.scatter_reduce(
            0, idx_dst, scores, reduce="amax", include_self=True
        )                                            # (N, H)

        # Step 2: exponentiate
        exp_s = torch.exp(scores - max_scores[dst])  # (E, H)

        # Step 3: sum exp per dst
        sum_exp = torch.zeros(N, H, dtype=scores.dtype, device=scores.device)
        sum_exp.scatter_add_(0, idx_dst, exp_s)      # (N, H)

        # Step 4: normalise
        att = exp_s / (sum_exp[dst] + 1e-9)          # (E, H)

        if self.training and self.dropout_p > 0:
            att = F.dropout(att, p=self.dropout_p)

        # ── Weighted aggregation of enriched values ───────────────────────
        weighted_v = att.unsqueeze(-1) * v_e         # (E, H, D)
        out = torch.zeros(N, H, D, dtype=x.dtype, device=x.device)
        idx_dst_3d = dst.view(-1, 1, 1).expand(-1, H, D)
        out.scatter_add_(0, idx_dst_3d, weighted_v)  # (N, H, D)

        # ── Merge heads and output projection ─────────────────────────────
        return self.W_o(out.reshape(N, self.d_model))


# ---------------------------------------------------------------------------
# BusEncoderLayer
# ---------------------------------------------------------------------------


class BusEncoderLayer(nn.Module):
    """One transformer layer for the power grid bus encoder.

    Sub-layer 1: PreNorm(RMSNorm) + EdgeGatedAttention + GatedResidual
    Sub-layer 2: PreNorm(RMSNorm) + SwiGLU FFN         + GatedResidual
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_edge: int = 20,
        d_ffn_hidden: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)
        self.attn = EdgeGatedAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_edge=d_edge,
            dropout=dropout,
        )
        self.ffn = SwiGLU(d_model=d_model, d_hidden=d_ffn_hidden)
        self.gate_attn = GatedResidual(d_model)
        self.gate_ffn = GatedResidual(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (N, d_model)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, d_edge)
    ) -> torch.Tensor:
        # ── Sub-layer 1: edge-gated attention + gated residual ────────────
        h = self.attn(self.norm_attn(x), edge_index, edge_attr)
        h = self.dropout(h)
        x = self.gate_attn(x + h, x)   # gate between (x+delta) and x

        # ── Sub-layer 2: SwiGLU FFN + gated residual ──────────────────────
        h = self.ffn(self.norm_ffn(x))
        h = self.dropout(h)
        x = self.gate_ffn(x + h, x)

        return x


# ---------------------------------------------------------------------------
# SIGReg — Sketched-Isotropic-Gaussian Regularizer
# ---------------------------------------------------------------------------


class SIGReg(nn.Module):
    """Sketched-Isotropic-Gaussian Regularizer.

    Projects the embedding matrix Z ∈ ℝ^{N×d} onto M random unit-norm
    directions and penalises deviations from standard normality (μ=0, σ=1):

        y_m  = Z @ ω_m                          ω_m ~ 𝒩(0, I),  ‖ω_m‖ = 1
        loss = mean_m [ μ(y_m)²  +  (σ(y_m) − 1)² ]

    For Z ~ 𝒩(0, I):  y_m = Z @ ω_m ~ 𝒩(0, 1)  →  loss ≈ 0.
    For collapsed Z = 0:  σ = 0  →  loss = 1.0.

    No learnable parameters — ω is a fixed buffer.
    """

    def __init__(self, d_model: int, M: int = 64) -> None:
        super().__init__()
        self.M = M
        omega = torch.randn(d_model, M)
        omega = F.normalize(omega, dim=0)   # unit-norm columns
        self.register_buffer("omega", omega)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (N, d_model) — bus embeddings after final encoder layer.
        Returns:
            Scalar loss term λ * sigreg added to total training loss.
        """
        y = z @ self.omega                           # (N, M)
        mu = y.mean(0)                               # (M,)
        sigma = y.std(0, unbiased=False)             # (M,)
        return mu.pow(2).mean() + (sigma - 1.0).pow(2).mean()


# ---------------------------------------------------------------------------
# BusEncoder  (full 6-layer stack)
# ---------------------------------------------------------------------------


class BusEncoder(nn.Module):
    """Full Graph Transformer bus encoder.

    Projects raw bus features (17-dim) to d_model=512, applies 6 stacked
    BusEncoderLayer with edge-gated attention and gated residual connections,
    and (at training time) computes the SIGReg anti-collapse regularisation.

    Args:
        d_model    : embedding dimension (default 512)
        n_heads    : attention heads (default 8, d_head = 64)
        n_layers   : number of BusEncoderLayer stacked (default 6)
        d_bus_in   : raw bus feature dimension (default 17)
        d_edge_in  : raw edge feature dimension (default 20)
        dropout    : dropout probability (default 0.0)
        sigreg_M   : number of random projections for SIGReg (default 64)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_bus_in: int = 17,
        d_edge_in: int = 20,
        dropout: float = 0.0,
        sigreg_M: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_bus_in, d_model, bias=False)
        self.input_norm = RMSNorm(d_model)

        self.layers = nn.ModuleList(
            [
                BusEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_edge=d_edge_in,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.output_norm = RMSNorm(d_model)
        self.sigreg = SIGReg(d_model=d_model, M=sigreg_M)

        self._init_weights()

    def _init_weights(self) -> None:
        """Scaled init for residual-branch projections (1/√n_layers).

        GatedResidual is already near-identity at construction (bias=+2).
        edge_to_k / edge_to_v get small Xavier init so they start as minor
        corrections to the node keys/values.
        """
        n = sum(1 for m in self.modules() if isinstance(m, BusEncoderLayer))
        scale = 1.0 / math.sqrt(n) if n > 0 else 1.0

        for module in self.modules():
            if isinstance(module, EdgeGatedAttention):
                nn.init.xavier_uniform_(module.W_o.weight, gain=scale)
                nn.init.xavier_uniform_(module.edge_to_k.weight, gain=0.1)
                nn.init.xavier_uniform_(module.edge_to_v.weight, gain=0.1)
            elif isinstance(module, SwiGLU):
                nn.init.xavier_uniform_(module.proj.weight, gain=scale)

    def forward(
        self,
        x: torch.Tensor,            # (N, d_bus_in)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, d_edge_in)
        return_sigreg: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Returns:
            z_bus       : (N, d_model) — bus embeddings
            sigreg_loss : scalar or None
                          (only non-None when return_sigreg=True AND model.training)
        """
        x = self.input_norm(self.input_proj(x))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        z_bus = self.output_norm(x)

        sigreg_loss = None
        if return_sigreg and self.training:
            sigreg_loss = self.sigreg(z_bus)

        return z_bus, sigreg_loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
