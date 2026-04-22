# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""BusesToken dataclass: the output of BusesTokenConverter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Token feature columns (buses) — order is the feature schema contract.
#
# Sign conventions
# ----------------
# gen_p, gen_q, bat_p, bat_q : GENERATOR convention (positive = injecting into network).
#   pypowsybl uses load convention (gen.p < 0 for production); these are negated
#   at extraction time so a generating bus always has gen_p > 0.
# load_p, load_q             : LOAD convention (positive = consuming from network).
# shunt_q                    : pypowsybl load convention (negative = capacitor
#   injecting reactive; positive = reactor absorbing reactive).
# p_net = gen_p − load_p + bat_p          →  standard nodal injection P_i (MW)
# q_net = gen_q − load_q − shunt_q + bat_q →  standard nodal injection Q_i (MVAr)
#   Positive p_net / q_net = bus injects into the network (generator bus).
#   Negative p_net / q_net = bus draws from the network (load bus).
#   This matches the standard AC power-flow equations:
#   P_i = Σ_{j∈N(i)} V_i V_j (G_ij cosθ_ij + B_ij sinθ_ij)
TOKEN_NUMERICAL_COLS = [
    "v_mag",             # kV  — voltage magnitude (AC state variable from SE/OLF)
    "v_angle",           # deg — voltage phase angle (AC state variable; slack = 0)
    "nominal_v",         # kV  — rated voltage of the voltage level
    "is_main_component", # 0/1 — belongs to main synchronous island
    "gen_p",             # MW   — total active generation (generator conv., >0 = producing)
    "gen_q",             # MVAr — total reactive generation (generator conv., >0 = injecting Q)
    "load_p",            # MW   — total active load (load conv., >0 = consuming)
    "load_q",            # MVAr — total reactive load (load conv., >0 = consuming Q)
    "shunt_q",           # MVAr — shunt reactive power (load conv.; <0 = capacitor injecting)
    "bat_p",             # MW   — total battery net injection (generator conv., >0 = discharging)
    "bat_q",             # MVAr — total battery reactive injection (generator conv.)
    "p_net",             # MW   — nodal injection P_i = gen_p − load_p + bat_p (>0 = source)
    "q_net",             # MVAr — nodal injection Q_i = gen_q − load_q − shunt_q + bat_q
    "n_gens",            # int  — number of generators at this bus
    "n_loads",           # int  — number of loads
    "n_shunts",          # int  — number of shunt compensators
    "n_batteries",       # int  — number of batteries
]

# Relation feature columns (branches) — order is the feature schema contract.
#
# tap_rho   : off-nominal transformation ratio ρ for 2WT (1.0 for lines).
#   Tap changers regulate voltage by shifting ρ away from 1.0.
#   The effective Y_ij in Y_bus scales as 1/ρ², making this feature critical
#   for the model to learn the correct admittance between bus pairs.
#   When transformer_voltage_control_on=True in OLF, ρ reflects the actual
#   tap position chosen by the voltage regulator.
# tap_alpha : phase-shift angle α (degrees) for phase-shifting transformers (PST).
#   0.0 for standard ratio tap changers and for lines.
RELATION_NUMERICAL_COLS = [
    "r",            # pu  — series resistance (defines G_ij = r/(r²+x²))
    "x",            # pu  — series reactance  (defines B_ij = −x/(r²+x²))
    "g1",           # pu  — shunt conductance side 1 (line charging; NaN for 2WT)
    "b1",           # pu  — shunt susceptance side 1 (line charging; NaN for 2WT)
    "g2",           # pu  — shunt conductance side 2 (line charging; NaN for 2WT)
    "b2",           # pu  — shunt susceptance side 2 (line charging; NaN for 2WT)
    "tap_rho",      # pu  — off-nominal tap ratio ρ (1.0 for lines; ≠1.0 for regulated 2WT)
    "tap_alpha",    # deg — phase-shift angle α (0.0 unless PST)
    "p1",           # MW   — active power flow at side 1 (from AC state)
    "q1",           # MVAr — reactive power at side 1
    "i1",           # A    — current magnitude at side 1
    "p2",           # MW   — active power flow at side 2
    "q2",           # MVAr — reactive power at side 2
    "i2",           # A    — current magnitude at side 2
    "limit1",       # A    — permanent current limit side 1 (IST, NaN if undefined)
    "limit2",       # A    — permanent current limit side 2 (IST, NaN if undefined)
    "base_rho",     # pu   — loading ratio = max(i1/limit1, i2/limit2) at N-0
    "is_line",      # 0/1  — 1 if LINE
    "is_2wt",       # 0/1  — 1 if 2-winding transformer
    "is_self_loop", # 0/1  — 1 if bus1_id == bus2_id (bus-bar coupling, maintenance)
]


@dataclass
class BusesToken:
    """
    Bus Token representation of an operational power grid snapshot.

    Each active bus is a **token** with a unique ID (bus_id), analogous to
    word tokens in NLP.  Relations between tokens (branches) define the sparse
    attention mask for Graph Transformer models and carry physical edge features.

    All feature values reflect the actual operating state at snapshot time —
    not nominal values — which is the key distinction from topology-only
    representations (e.g. H2MG).

    Attributes
    ----------
    bus_df : pd.DataFrame
        Index = bus_id (str) — the token vocabulary key.
        Numerical columns (see TOKEN_NUMERICAL_COLS for full schema):
          v_mag, v_angle            — AC state variables from SE/OLF
          nominal_v, is_main_component
          gen_p, gen_q              — generation in generator convention (>0 = producing)
          load_p, load_q            — consumption in load convention (>0 = consuming)
          shunt_q                   — pypowsybl load convention (<0 = capacitor)
          bat_p, bat_q              — battery in generator convention (>0 = discharging)
          p_net = gen_p−load_p+bat_p     — standard nodal injection P_i
          q_net = gen_q−load_q−shunt_q+bat_q — standard nodal injection Q_i
          n_gens, n_loads, n_shunts, n_batteries
        Metadata columns: voltage_level_id, substation_id, connected_component.
    relation_df : pd.DataFrame
        Index = branch_id (str).
        Numerical columns (see RELATION_NUMERICAL_COLS for full schema):
          r, x                 — series impedance (pu) → Y_ij = 1/(r+jx)
          g1, b1, g2, b2       — shunt admittance per side (line charging; NaN for 2WT)
          tap_rho              — off-nominal transformation ratio ρ (1.0 for lines)
          tap_alpha            — phase-shift angle α in degrees (0.0 unless PST)
          p1, q1, i1           — AC operating state at side 1
          p2, q2, i2           — AC operating state at side 2
          limit1, limit2       — IST permanent current limits (A; NaN if undefined)
          base_rho             — loading ratio max(i1/limit1, i2/limit2)
          is_line, is_2wt, is_self_loop
        Identifier columns: bus1_id, bus2_id, branch_kind ('LINE' | '2WT').
    snapshot_id : str or None
        Identifier of the snapshot (e.g. filename stem).
    """

    bus_df: pd.DataFrame
    relation_df: pd.DataFrame
    snapshot_id: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        """Number of active buses (tokens)."""
        return len(self.bus_df)

    @property
    def n_relations(self) -> int:
        """Number of active branches (relations between tokens)."""
        return len(self.relation_df)

    # ------------------------------------------------------------------
    # Feature matrices
    # ------------------------------------------------------------------

    @property
    def token_feature_names(self) -> list[str]:
        """Ordered list of numerical token (bus) feature column names."""
        return [c for c in TOKEN_NUMERICAL_COLS if c in self.bus_df.columns]

    @property
    def relation_feature_names(self) -> list[str]:
        """Ordered list of numerical relation (branch) feature column names."""
        return [c for c in RELATION_NUMERICAL_COLS if c in self.relation_df.columns]

    @property
    def token_features(self) -> np.ndarray:
        """Float32 array (n_tokens, n_token_features). NaN filled with 0."""
        return self.bus_df[self.token_feature_names].fillna(0.0).astype(np.float32).values

    @property
    def relation_features(self) -> np.ndarray:
        """Float32 array (n_relations, n_relation_features). NaN filled with 0."""
        return self.relation_df[self.relation_feature_names].fillna(0.0).astype(np.float32).values

    @property
    def relation_index(self) -> np.ndarray:
        """
        Int64 array of shape (2, n_relations) mapping branch endpoints to bus
        indices (positions in ``bus_df``).

        Compatible with PyTorch Geometric convention:
        ``relation_index[0]`` = source token index,
        ``relation_index[1]`` = destination token index.

        This is the sparse attention mask: only token pairs connected by a
        physical branch attend to each other in the Graph Transformer.
        """
        bus_to_idx: dict[str, int] = {b: i for i, b in enumerate(self.bus_df.index)}
        src = self.relation_df["bus1_id"].map(bus_to_idx).fillna(-1).astype(np.int64).values
        dst = self.relation_df["bus2_id"].map(bus_to_idx).fillna(-1).astype(np.int64).values
        return np.stack([src, dst], axis=0)

    # ------------------------------------------------------------------
    # Optional conversions
    # ------------------------------------------------------------------

    def to_networkx(self):
        """
        Convert to a ``networkx.MultiDiGraph``.
        Nodes carry all ``bus_df`` columns; edges carry all ``relation_df`` columns.
        Requires ``networkx``.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("networkx is required: pip install networkx") from e

        G = nx.MultiDiGraph()
        for bus_id, row in self.bus_df.iterrows():
            G.add_node(bus_id, **row.to_dict())
        for branch_id, row in self.relation_df.iterrows():
            G.add_edge(row["bus1_id"], row["bus2_id"], key=branch_id, **row.to_dict())
        return G

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        snap = self.snapshot_id or "?"
        return (
            f"BusesToken(snapshot={snap!r}, "
            f"n_tokens={self.n_tokens}, n_relations={self.n_relations}, "
            f"token_features={len(self.token_feature_names)}, "
            f"relation_features={len(self.relation_feature_names)})"
        )
