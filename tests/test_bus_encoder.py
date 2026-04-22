"""
Tests for BusEncoder — Graph Transformer architecture.

Coverage:
    - RMSNorm       : shape, no mean-centering, zero-input, gradient
    - SwiGLU        : shape, hidden-dim rounding, gradient
    - GatedResidual : gate bounds (0,1), near-identity init, suppression of
                      zero-delta, gradient flow
    - EdgeGatedAttention : shape (mini + large synthetic), sparse softmax sums-to-1
                           (with K-enrichment), gradient, heads assertion
    - BusEncoderLayer   : shape, output changes input, gradient
    - SIGReg            : scalar output, buffer not param, ~0 on N(0,I),
                          large on collapsed, gradient
    - BusEncoder        : forward pass (large synthetic + mini), sigreg train/eval,
                          param count, gradient end-to-end, no NaN/Inf,
                          timing, determinism, different graphs → different output,
                          edge enrichment changes output vs. no enrichment
"""

import math
import time

import pytest
import torch
import torch.nn as nn

from pypowsybl_to_busestoken.model import (
    BusEncoder,
    BusEncoderLayer,
    EdgeGatedAttention,
    GatedResidual,
    RMSNorm,
    SIGReg,
    SwiGLU,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_random_graph(N: int, E: int, d_bus: int = 17, d_edge: int = 20):
    """Generate a random directed graph with package feature shapes."""
    x = torch.randn(N, d_bus)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    E = min(E, src.size(0))
    edge_index = torch.stack([src[:E], dst[:E]], dim=0)
    edge_attr = torch.randn(E, d_edge)
    return x, edge_index, edge_attr


# ─── RMSNorm ─────────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(512)
        assert norm(torch.randn(100, 512)).shape == (100, 512)

    def test_no_mean_centering(self):
        """All-positive input → output stays positive (no zero-centering)."""
        norm = RMSNorm(512)
        y = norm(torch.ones(1000, 512) * 5.0)
        assert y.mean().item() > 0.1

    def test_zero_input_stays_near_zero(self):
        """Transit bus with all-zero features → output ≈ 0."""
        norm = RMSNorm(512)
        nn.init.ones_(norm.weight)
        y = norm(torch.zeros(1, 512))
        assert y.abs().max().item() < 1e-2

    def test_gradient_flows(self):
        norm = RMSNorm(64)
        x = torch.randn(10, 64, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None


# ─── SwiGLU ──────────────────────────────────────────────────────────────────

class TestSwiGLU:
    def test_output_shape(self):
        assert SwiGLU(512)(torch.randn(100, 512)).shape == (100, 512)

    def test_hidden_dim_rounded(self):
        ffn = SwiGLU(512)
        expected = int(math.ceil((8 / 3 * 512) / 64) * 64)
        assert ffn.gate.out_features == expected

    def test_gradient_flows(self):
        x = torch.randn(10, 64, requires_grad=True)
        SwiGLU(64)(x).sum().backward()
        assert x.grad is not None


# ─── GatedResidual ────────────────────────────────────────────────────────────

class TestGatedResidual:
    def test_output_shape(self):
        gr = GatedResidual(64)
        x = torch.randn(20, 64)
        assert gr(x, x).shape == (20, 64)

    def test_gate_in_zero_one(self):
        """Gate values must be strictly in (0, 1) — sigmoid output."""
        gr = GatedResidual(64)
        x_new = torch.randn(100, 64)
        x_old = torch.randn(100, 64)
        gate_input = torch.cat([x_new, x_old, x_new - x_old], dim=-1)
        gate = torch.sigmoid(gr.proj(gate_input))
        assert gate.min().item() > 0.0
        assert gate.max().item() < 1.0

    def test_near_identity_init(self):
        """At initialisation (bias=+2), gate ≈ 0.88 → output ≈ x_new × 0.88 + x_old × 0.12.
        The output should be much closer to x_new than to x_old."""
        gr = GatedResidual(64)
        x_new = torch.ones(10, 64) * 10.0   # clearly different from x_old
        x_old = torch.zeros(10, 64)
        out = gr(x_new, x_old)
        # expected output ≈ 8.8; should be well above 5 (midpoint) and below 10
        assert out.mean().item() > 5.0
        assert out.mean().item() < 10.0

    def test_zero_delta_is_suppressed(self):
        """If x_new == x_old (no update), output must equal x_old exactly."""
        gr = GatedResidual(32)
        x = torch.randn(10, 32)
        out = gr(x, x)
        # gate(x, x, x-x=0) → any gate value → x*g + x*(1-g) = x
        assert torch.allclose(out, x, atol=1e-5)

    def test_gradient_flows(self):
        gr = GatedResidual(64)
        x_new = torch.randn(10, 64, requires_grad=True)
        x_old = torch.randn(10, 64, requires_grad=True)
        gr(x_new, x_old).sum().backward()
        assert x_new.grad is not None
        assert x_old.grad is not None


# ─── EdgeGatedAttention ───────────────────────────────────────────────────────

class TestEdgeGatedAttention:
    def test_output_shape_mini(self):
        attn = EdgeGatedAttention(d_model=64, n_heads=4, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=20, E=40, d_bus=64)
        assert attn(x, edge_index, edge_attr).shape == (20, 64)

    def test_output_shape_large_grid(self):
        """Large-grid synthetic graph.
        EdgeGatedAttention receives d_model-dim inputs (post projection)."""
        attn = EdgeGatedAttention(d_model=512, n_heads=8, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=4096, E=6000, d_bus=512)
        with torch.no_grad():
            out = attn(x, edge_index, edge_attr)
        assert out.shape == (4096, 512)

    def test_sparse_softmax_sums_to_one(self):
        """For each dst node, attention weights summed over its neighbours = 1.
        Manual replication includes K-enrichment from edge features."""
        d_model, n_heads, d_edge = 32, 4, 8
        attn = EdgeGatedAttention(d_model=d_model, n_heads=n_heads, d_edge=d_edge)
        N, E = 15, 30
        x, edge_index, edge_attr = make_random_graph(N=N, E=E, d_bus=d_model, d_edge=d_edge)
        src, dst = edge_index[0], edge_index[1]
        E_actual = src.size(0)
        H = n_heads
        D = d_model // n_heads

        with torch.no_grad():
            q = attn.W_q(x).view(N, H, D)
            k = attn.W_k(x).view(N, H, D)

            q_e = q[dst]
            k_e = k[src]

            # Apply K-enrichment (new in v2)
            k_e = k_e + attn.edge_to_k(edge_attr).view(E_actual, H, D)

            scores = (q_e * k_e).sum(-1) / math.sqrt(D)
            scores = scores + attn.edge_gate(edge_attr)   # + routing bias

            idx_dst = dst.unsqueeze(1).expand(-1, H)
            max_s = torch.full((N, H), float("-inf"))
            max_s = max_s.scatter_reduce(0, idx_dst, scores, reduce="amax", include_self=True)
            exp_s = torch.exp(scores - max_s[dst])
            sum_e = torch.zeros(N, H).scatter_add(0, idx_dst, exp_s)
            att = exp_s / (sum_e[dst] + 1e-9)

            for node in dst.unique():
                mask = dst == node
                sums = att[mask].sum(0)
                assert torch.allclose(sums, torch.ones(H), atol=1e-4), (
                    f"Node {node.item()}: att sums = {sums.tolist()}"
                )

    def test_edge_enrichment_changes_output(self):
        """K/V enrichment from edge features must change the output vs.
        zero-edge-attr baseline (edge features carry information)."""
        attn = EdgeGatedAttention(d_model=64, n_heads=4, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=30, E=60, d_bus=64)
        with torch.no_grad():
            out_enriched = attn(x, edge_index, edge_attr)
            out_zero_edge = attn(x, edge_index, torch.zeros_like(edge_attr))
        assert not torch.allclose(out_enriched, out_zero_edge)

    def test_gradient_flows(self):
        attn = EdgeGatedAttention(d_model=64, n_heads=4, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=30, E=60, d_bus=64)
        x.requires_grad_(True)
        attn(x, edge_index, edge_attr).sum().backward()
        assert x.grad is not None

    def test_heads_assertion(self):
        with pytest.raises(AssertionError):
            EdgeGatedAttention(d_model=512, n_heads=7, d_edge=20)


# ─── BusEncoderLayer ─────────────────────────────────────────────────────────

class TestBusEncoderLayer:
    def test_output_shape(self):
        layer = BusEncoderLayer(d_model=64, n_heads=4, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=50, E=100, d_bus=64)
        assert layer(x, edge_index, edge_attr).shape == (50, 64)

    def test_output_differs_from_input(self):
        layer = BusEncoderLayer(d_model=64, n_heads=4, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=20, E=40, d_bus=64)
        assert not torch.allclose(layer(x, edge_index, edge_attr), x)

    def test_gated_residual_used(self):
        """With GatedResidual, output should NOT be exactly x + attn(norm(x))
        (i.e. the gate is doing something beyond plain addition)."""
        layer = BusEncoderLayer(d_model=64, n_heads=4, d_edge=20)
        layer.eval()
        x, edge_index, edge_attr = make_random_graph(N=20, E=40, d_bus=64)
        with torch.no_grad():
            # Compute what plain residual would give
            h_attn = layer.attn(layer.norm_attn(x), edge_index, edge_attr)
            plain_residual = x + h_attn
            # Actual layer output (with GatedResidual)
            actual = layer(x, edge_index, edge_attr)
        # GatedResidual blends x_new and x_old, so actual ≠ plain_residual
        assert not torch.allclose(actual, plain_residual, atol=1e-5)

    def test_gradient_flows(self):
        layer = BusEncoderLayer(d_model=64, n_heads=4, d_edge=20)
        x, edge_index, edge_attr = make_random_graph(N=20, E=40, d_bus=64)
        x.requires_grad_(True)
        layer(x, edge_index, edge_attr).sum().backward()
        assert x.grad is not None


# ─── SIGReg ──────────────────────────────────────────────────────────────────

class TestSIGReg:
    def test_output_is_scalar(self):
        sigreg = SIGReg(d_model=512, M=64)
        assert sigreg(torch.randn(4096, 512)).shape == ()

    def test_buffer_not_parameter(self):
        sigreg = SIGReg(d_model=128, M=32)
        assert "omega" not in {n for n, _ in sigreg.named_parameters()}

    def test_loss_near_zero_for_standard_normal(self):
        """z ~ N(0, I_d): projected y_m ~ N(0, 1) → loss ≈ 0."""
        sigreg = SIGReg(d_model=512, M=64)
        loss = sigreg(torch.randn(10_000, 512))
        assert loss.item() < 0.1, f"SIGReg on N(0,I): {loss.item():.4f}"

    def test_loss_large_for_collapsed(self):
        """z = 0: σ = 0 → (σ−1)² = 1 → loss ≈ 1."""
        sigreg = SIGReg(d_model=512, M=64)
        loss = sigreg(torch.zeros(1000, 512))
        assert loss.item() > 0.5

    def test_gradient_flows(self):
        sigreg = SIGReg(d_model=64, M=16)
        z = torch.randn(100, 64, requires_grad=True)
        sigreg(z).backward()
        assert z.grad is not None


# ─── BusEncoder (full stack) ─────────────────────────────────────────────────

class TestBusEncoder:
    def _make_encoder(self, **kwargs) -> BusEncoder:
        defaults = dict(d_model=512, n_heads=8, n_layers=6, d_bus_in=17, d_edge_in=20)
        defaults.update(kwargs)
        return BusEncoder(**defaults)

    def test_output_shape_large_grid(self):
        """Full forward pass on a large synthetic graph."""
        model = self._make_encoder()
        x, edge_index, edge_attr = make_random_graph(N=4096, E=6000)
        with torch.no_grad():
            z, _ = model(x, edge_index, edge_attr)
        assert z.shape == (4096, 512)
        assert z.dtype == torch.float32

    def test_output_shape_mini(self):
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        x, edge_index, edge_attr = make_random_graph(N=10, E=20)
        z, _ = model(x, edge_index, edge_attr)
        assert z.shape == (10, 64)

    def test_sigreg_returned_in_train_mode(self):
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        model.train()
        x, edge_index, edge_attr = make_random_graph(N=50, E=100)
        z, sigreg = model(x, edge_index, edge_attr, return_sigreg=True)
        assert sigreg is not None and sigreg.shape == ()

    def test_sigreg_none_in_eval_mode(self):
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        model.eval()
        x, edge_index, edge_attr = make_random_graph(N=50, E=100)
        with torch.no_grad():
            _, sigreg = model(x, edge_index, edge_attr, return_sigreg=True)
        assert sigreg is None

    def test_sigreg_false_always_none(self):
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        model.train()
        x, edge_index, edge_attr = make_random_graph(N=50, E=100)
        _, sigreg = model(x, edge_index, edge_attr, return_sigreg=False)
        assert sigreg is None

    def test_parameter_count_range(self):
        """d_model=512, 8 heads, 6 layers with GatedResidual + edge K/V enrichment.
        GatedResidual adds ~18K params; edge_to_k/v add ~123K → total ~19-22M."""
        model = self._make_encoder()
        n = model.count_parameters()
        print(f"\nBusEncoder total parameters: {n:,}")
        assert 10_000_000 < n < 60_000_000, f"Unexpected param count: {n:,}"

    def test_gradient_flows_end_to_end(self):
        """Gradient must reach input through 6 layers + GatedResidual gates."""
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=6)
        model.train()
        x, edge_index, edge_attr = make_random_graph(N=30, E=60)
        x.requires_grad_(True)
        z, sigreg = model(x, edge_index, edge_attr, return_sigreg=True)
        (z.sum() + sigreg).backward()
        assert x.grad is not None
        assert x.grad.abs().max().item() > 0

    def test_no_nan_in_output(self):
        model = self._make_encoder()
        x, edge_index, edge_attr = make_random_graph(N=4096, E=6000)
        with torch.no_grad():
            z, _ = model(x, edge_index, edge_attr)
        assert not torch.isnan(z).any(), "NaN in BusEncoder output"
        assert not torch.isinf(z).any(), "Inf in BusEncoder output"

    def test_forward_timing_large_grid(self):
        """Single large-grid forward pass must complete in < 30s on CPU."""
        model = self._make_encoder()
        x, edge_index, edge_attr = make_random_graph(N=4096, E=6000)
        with torch.no_grad():
            t0 = time.perf_counter()
            model(x, edge_index, edge_attr)
            elapsed = time.perf_counter() - t0
        print(f"\nForward pass (N=4096, E=6000): {elapsed:.3f}s")
        assert elapsed < 30.0, f"Too slow: {elapsed:.1f}s"

    def test_deterministic_in_eval_mode(self):
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        model.eval()
        x, edge_index, edge_attr = make_random_graph(N=50, E=100)
        with torch.no_grad():
            z1, _ = model(x, edge_index, edge_attr)
            z2, _ = model(x, edge_index, edge_attr)
        assert torch.allclose(z1, z2)

    def test_different_graphs_give_different_embeddings(self):
        """Different topology → different z_bus."""
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        model.eval()
        x = torch.randn(20, 17)
        _, ei1, ea1 = make_random_graph(N=20, E=30)
        _, ei2, ea2 = make_random_graph(N=20, E=30)
        with torch.no_grad():
            z1, _ = model(x, ei1, ea1)
            z2, _ = model(x, ei2, ea2)
        assert not torch.allclose(z1, z2)

    def test_edge_features_affect_output(self):
        """Non-zero edge features must give different output than zero-edge-attr.
        Validates that edge_to_k / edge_to_v / edge_gate all have an effect."""
        model = self._make_encoder(d_model=64, n_heads=4, n_layers=2)
        model.eval()
        x, edge_index, edge_attr = make_random_graph(N=30, E=60)
        with torch.no_grad():
            z_enriched, _ = model(x, edge_index, edge_attr)
            z_zero_edge, _ = model(x, edge_index, torch.zeros_like(edge_attr))
        assert not torch.allclose(z_enriched, z_zero_edge)
