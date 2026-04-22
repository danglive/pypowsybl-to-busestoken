# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Relation feature extraction: active branches with limits and loading ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypowsybl.network as pn

# PyPowSyBl encodes "no limit defined" as this sentinel value.
_LIMIT_INF: float = 1.7976931348623157e308


def _active_branch(df: pd.DataFrame, active_bus_ids: frozenset[str]) -> pd.DataFrame:
    """
    Keep branch elements (line / 2WT) where:
      1. Both terminals are connected (connected1=True AND connected2=True).
      2. Both bus IDs are non-empty strings belonging to active buses.

    The empty-string check is necessary because PyPowSyBl returns bus1_id=''
    for a terminal whose circuit breaker is open, and '' ∉ active_bus_ids but
    isin() would treat it as a valid lookup if not caught explicitly.
    """
    mask = (
        (df["connected1"] == True)
        & (df["connected2"] == True)
        & df["bus1_id"].notna()
        & df["bus2_id"].notna()
        & (df["bus1_id"] != "")
        & (df["bus2_id"] != "")
        & df["bus1_id"].isin(active_bus_ids)
        & df["bus2_id"].isin(active_bus_ids)
    )
    return df[mask].copy()


def _permanent_limits(network: pn.Network) -> tuple[pd.Series, pd.Series]:
    """
    Extract per-side permanent current limits (IST) from operational limits.

    Returns two Series (limit_side1, limit_side2), indexed by element_id.
    The conservative value for each side is kept when multiple groups define
    the same permanent limit for the same element/side.
    The sentinel value 1.797...e+308 (= "no limit") is replaced with NaN.

    Returns
    -------
    limit1 : pd.Series  — permanent limit on side ONE, index=element_id
    limit2 : pd.Series  — permanent limit on side TWO, index=element_id
    """
    ol = network.get_operational_limits().reset_index()
    perm = ol[
        (ol["acceptable_duration"] == -1)
        & (ol["type"] == "CURRENT")
    ].copy()

    def _side_limits(side: str) -> pd.Series:
        s = (
            perm[perm["side"] == side]
            .groupby("element_id")["value"]
            .min()  # conservative: take the most restrictive limit
        )
        s = s.replace(_LIMIT_INF, np.nan)
        return s

    return _side_limits("ONE").rename("limit1"), _side_limits("TWO").rename("limit2")


def build_relation_df(network: pn.Network) -> pd.DataFrame:
    """
    Build the relation feature DataFrame for the operational state.

    Each row corresponds to one active branch (line or 2-winding transformer)
    with both terminals energised.  Features include:

    Topology:
        bus1_id, bus2_id   — endpoint bus IDs (post-topology processing)
        branch_kind        — 'LINE' or '2WT'
        is_line, is_2wt    — one-hot encoding of branch_kind
        is_self_loop       — 1 if bus1_id == bus2_id (rare after bus-splitting)

    Static electrical parameters:
        r, x               — series resistance and reactance (pu)
                             Y_ij = 1/(r+jx) is the branch admittance in Y_bus
        g1, b1, g2, b2     — shunt admittance at each terminal (pu); NaN for 2WT
                             (line charging susceptance; 2WT has single magnetising
                              admittance not split per side)
        tap_rho            — off-nominal transformation ratio ρ (1.0 for lines)
                             For regulated 2WT, ρ ≠ 1.0 reflects the tap-changer
                             position chosen by OLF voltage control.
                             Effective Y_ij in Y_bus scales as 1/ρ² for 2WT.
        tap_alpha          — phase-shift angle α in degrees (0.0 for lines and
                             standard ratio tap changers; non-zero for PSTs)

    Operating state (from load flow):
        p1, q1, i1         — active power (MW), reactive (MVAr), current (A) at side 1
        p2, q2, i2         — same at side 2

    Thermal limits (IST — Intensité de Seuil Thermique):
        limit1, limit2     — permanent current limits sides 1/2 (A); NaN if undefined
        base_rho           — loading ratio = max(i1/limit1, i2/limit2); NaN if no limit

    Notes
    -----
    * ``v_angle`` in PyPowSyBl is in **degrees**.
    * ``base_rho >= 1.0`` means the branch is already overloaded at N-0.
    * Parallel edges (same bus pair) are kept — this is a MultiGraph.
    * Self-loops (bus1_id == bus2_id) can arise from bus-bar splitting during
      maintenance; they are kept and flagged with ``is_self_loop = 1``.

    Parameters
    ----------
    network : pn.Network
        PyPowSyBl network after AC load flow.

    Returns
    -------
    pd.DataFrame
        Index = branch_id (str).
    """
    buses = network.get_buses()
    active_ids = frozenset(
        buses.index[buses["v_mag"].notna() & buses["v_angle"].notna()]
    )

    limit1, limit2 = _permanent_limits(network)

    # ── Lines ─────────────────────────────────────────────────────────────
    lines_raw = _active_branch(network.get_lines(), active_ids)
    lines = lines_raw[["bus1_id", "bus2_id", "r", "x",
                        "g1", "b1", "g2", "b2",
                        "p1", "q1", "i1", "p2", "q2", "i2"]].copy()
    lines["branch_kind"] = "LINE"
    lines["is_line"] = 1
    lines["is_2wt"] = 0
    # Lines have no transformation: ρ = 1 (no off-nominal ratio), α = 0 (no phase shift).
    lines["tap_rho"]   = 1.0
    lines["tap_alpha"] = 0.0

    # ── 2-Winding Transformers ────────────────────────────────────────────
    # TWT schema: single magnetising admittance columns 'g' and 'b' (not split).
    # We expose them as NaN for the per-side columns to keep a uniform schema.
    #
    # tap_rho  : off-nominal transformation ratio ρ after tap-changer adjustment.
    #   When transformer_voltage_control_on=True in OLF, ρ reflects the actual
    #   tap position selected by the voltage regulator.  Affects Y_ij as 1/ρ².
    #   pypowsybl exposes 'rho' directly in get_2_windings_transformers() after
    #   load flow; falls back to 1.0 (neutral tap) if column is absent.
    # tap_alpha: phase-shift angle α in degrees for phase-shifting transformers (PST).
    #   0.0 for standard ratio tap changers.
    #   pypowsybl exposes 'alpha' in get_2_windings_transformers() after load flow.
    # Fetch 2WT with all_attributes=True to obtain 'rho' and 'alpha' — columns
    # computed by OLF and exposed only via this flag.  'rho' is the effective
    # off-nominal transformation ratio after tap-changer regulation; 'alpha' is
    # the phase-shift angle for phase-shifting transformers (PSTs).
    twts_raw = _active_branch(
        network.get_2_windings_transformers(all_attributes=True), active_ids
    )
    twts = twts_raw[["bus1_id", "bus2_id", "r", "x",
                     "p1", "q1", "i1", "p2", "q2", "i2"]].copy()
    # g1/b1/g2/b2 not split for 2WT (single magnetising admittance): set to NaN.
    twts[["g1", "b1", "g2", "b2"]] = np.nan
    twts["branch_kind"] = "2WT"
    twts["is_line"] = 0
    twts["is_2wt"] = 1
    # Tap ratio ρ: present in all_attributes; fallback to 1.0 (neutral) if absent.
    twts["tap_rho"] = (
        twts_raw["rho"].values
        if "rho" in twts_raw.columns
        else 1.0
    )
    # Phase shift α: present for PSTs in all_attributes; 0.0 for standard 2WT.
    twts["tap_alpha"] = (
        twts_raw["alpha"].values
        if "alpha" in twts_raw.columns
        else 0.0
    )

    # ── Concatenate ───────────────────────────────────────────────────────
    relation_df = pd.concat([lines, twts], axis=0)
    relation_df.index.name = "branch_id"

    # ── Attach thermal limits ─────────────────────────────────────────────
    relation_df = relation_df.join(limit1, how="left")
    relation_df = relation_df.join(limit2, how="left")

    # ── Loading ratio (base_rho) ──────────────────────────────────────────
    # NaN if neither side has a defined limit.
    rho1 = relation_df["i1"] / relation_df["limit1"]
    rho2 = relation_df["i2"] / relation_df["limit2"]
    relation_df["base_rho"] = pd.concat([rho1, rho2], axis=1).max(axis=1)

    # ── Self-loop flag ────────────────────────────────────────────────────
    relation_df["is_self_loop"] = (relation_df["bus1_id"] == relation_df["bus2_id"]).astype(int)

    return relation_df
