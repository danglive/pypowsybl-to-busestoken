# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Bus token feature extraction: bus-level aggregation of injections."""

from __future__ import annotations

import pandas as pd
import pypowsybl.network as pn


def _active_bus_ids(network: pn.Network) -> frozenset[str]:
    """
    Return the set of bus IDs that are electrically solved by the load flow.

    A bus is "active" if and only if both v_mag and v_angle are non-NaN after
    the AC load flow, meaning Newton-Raphson converged for that connected
    component.  Buses that belong to electrical islands not connected to a
    slack bus will have NaN state and are excluded.
    """
    buses = network.get_buses()
    mask = buses["v_mag"].notna() & buses["v_angle"].notna()
    return frozenset(buses.index[mask])


def _filter_injection(df: pd.DataFrame, active_bus_ids: frozenset[str]) -> pd.DataFrame:
    """
    Keep injection elements (gen/load/shunt/battery) that are:
      1. connected=True (circuit breaker closed)
      2. bus_id is a non-empty string belonging to an active bus

    Note: bus_id can be an empty string '' for disconnected elements — must
    filter explicitly, as '' is truthy in pandas isin checks against a set
    that may or may not contain it.
    """
    mask = (
        (df["connected"] == True)
        & df["bus_id"].notna()
        & (df["bus_id"] != "")
        & df["bus_id"].isin(active_bus_ids)
    )
    return df[mask].copy()


def build_bus_df(network: pn.Network) -> pd.DataFrame:
    """
    Build the bus token DataFrame for the operational state.

    Each row corresponds to one active bus (electrical island).  Features are
    grouped into three semantic layers:

    Layer 1 — Electrical state (from load flow):
        v_mag (kV), v_angle (deg)

    Layer 2 — Aggregated injections (summed over all devices at the bus):
        gen_p, gen_q   — active/reactive generation in generator convention
                         (positive = producing/injecting; negated from pypowsybl
                          load convention where gen.p < 0 for production)
        load_p, load_q — active/reactive demand in load convention
                         (positive = consuming)
        shunt_q        — shunt reactive power in pypowsybl load convention
                         (negative = capacitor injecting reactive into network;
                          positive = reactor absorbing reactive)
        bat_p, bat_q   — battery in generator convention
                         (positive = discharging = injecting into network)
        p_net = gen_p − load_p + bat_p   (standard nodal injection P_i, MW)
        q_net = gen_q − load_q − shunt_q + bat_q  (standard Q_i, MVAr)
        n_gens, n_loads, n_shunts, n_batteries — device counts

    Layer 3 — Topology/metadata:
        nominal_v (kV), substation_id, voltage_level_id,
        connected_component, is_main_component

    Parameters
    ----------
    network : pn.Network
        PyPowSyBl network **after** AC load flow has been run.

    Returns
    -------
    pd.DataFrame
        Index = bus_id (str).  All injection columns default to 0 when a bus
        has no devices of a given type.  NaN is preserved only for v_mag and
        v_angle if a bus is somehow included despite not being solved (should
        not occur with default filtering).
    """
    active_ids = _active_bus_ids(network)

    # ── Base: active buses ────────────────────────────────────────────────
    buses = network.get_buses()
    buses = buses[buses.index.isin(active_ids)].copy()
    buses.index.name = "bus_id"
    buses["is_main_component"] = (buses["connected_component"] == 0).astype(float)

    # ── Voltage level metadata ────────────────────────────────────────────
    vls = (
        network.get_voltage_levels()
        .reset_index()
        .rename(columns={"id": "voltage_level_id"})
        [["voltage_level_id", "substation_id", "nominal_v"]]
    )
    bus_df = buses.reset_index().rename(columns={"id": "bus_id"})
    bus_df = bus_df.merge(vls, on="voltage_level_id", how="left").set_index("bus_id")

    # ── Injection aggregation helpers ─────────────────────────────────────
    def _sum_by_bus(df: pd.DataFrame, cols: list[str], rename: dict[str, str]) -> pd.DataFrame:
        return df.groupby("bus_id")[cols].sum().rename(columns=rename)

    def _count_by_bus(df: pd.DataFrame, name: str) -> pd.Series:
        return df.groupby("bus_id").size().rename(name)

    # ── Generators ───────────────────────────────────────────────────────
    # pypowsybl load convention: gen.p < 0 when producing, gen.q < 0 when injecting reactive.
    # We negate here to obtain generator convention: gen_p > 0 = producing MW,
    # gen_q > 0 = injecting MVAr.  This aligns with the standard AC nodal injection
    # P_i = P_gen_i − P_load_i (positive = net injection into the network).
    gens = _filter_injection(network.get_generators(), active_ids)
    gen_sums = _sum_by_bus(gens, ["p", "q"], {"p": "gen_p", "q": "gen_q"})
    gen_sums["gen_p"] = -gen_sums["gen_p"]   # load → generator convention
    gen_sums["gen_q"] = -gen_sums["gen_q"]   # load → generator convention
    gen_count = _count_by_bus(gens, "n_gens")

    # ── Loads ─────────────────────────────────────────────────────────────
    # load.p > 0 and load.q > 0 for consumption — kept as-is (load convention).
    loads = _filter_injection(network.get_loads(), active_ids)
    load_sums = _sum_by_bus(loads, ["p", "q"], {"p": "load_p", "q": "load_q"})
    load_count = _count_by_bus(loads, "n_loads")

    # ── Shunts ────────────────────────────────────────────────────────────
    # shunt.q is in pypowsybl load convention:
    #   shunt_q < 0  →  capacitor bank (generating reactive, injecting into network)
    #   shunt_q > 0  →  reactor (absorbing reactive, consuming from network)
    # Kept in load convention; q_net uses (−shunt_q) to get the injection contribution.
    shunts = _filter_injection(network.get_shunt_compensators(), active_ids)
    shunt_sums = _sum_by_bus(shunts, ["q"], {"q": "shunt_q"})
    shunt_count = _count_by_bus(shunts, "n_shunts")

    # ── Batteries ─────────────────────────────────────────────────────────
    # pypowsybl load convention: bat.p < 0 when discharging (injecting into network).
    # Negate to generator convention: bat_p > 0 = discharging = net injection.
    batteries = _filter_injection(network.get_batteries(), active_ids)
    bat_sums = _sum_by_bus(batteries, ["p", "q"], {"p": "bat_p", "q": "bat_q"})
    bat_sums["bat_p"] = -bat_sums["bat_p"]   # load → generator convention
    bat_sums["bat_q"] = -bat_sums["bat_q"]   # load → generator convention
    bat_count = _count_by_bus(batteries, "n_batteries")

    # ── Join all aggregations ─────────────────────────────────────────────
    for agg in [gen_sums, gen_count, load_sums, load_count,
                shunt_sums, shunt_count, bat_sums, bat_count]:
        bus_df = bus_df.join(agg, how="left")

    # Fill NaN counts → 0 (buses with no devices of that type)
    count_cols = ["n_gens", "n_loads", "n_shunts", "n_batteries"]
    bus_df[count_cols] = bus_df[count_cols].fillna(0).astype(int)

    # Fill NaN injection sums → 0
    inj_cols = ["gen_p", "gen_q", "load_p", "load_q", "shunt_q", "bat_p", "bat_q"]
    bus_df[inj_cols] = bus_df[inj_cols].fillna(0.0)

    # ── Net nodal injection ───────────────────────────────────────────────
    # Standard AC power-flow convention (positive = net injection into network):
    #   P_i = P_gen_i − P_load_i + P_bat_discharge_i
    #   Q_i = Q_gen_i − Q_load_i − Q_shunt_i(load_conv) + Q_bat_i
    #
    # All "source" terms (gen_p, bat_p, gen_q, bat_q) are already in generator
    # convention (positive = injecting) after the negation above.
    # load_p, load_q remain in load convention (positive = consuming).
    # shunt_q is in pypowsybl load convention: subtract to get Q_shunt injected
    # (capacitor: shunt_q < 0 → −shunt_q > 0 adds reactive injection ✓).
    bus_df["p_net"] = bus_df["gen_p"] - bus_df["load_p"] + bus_df["bat_p"]
    bus_df["q_net"] = bus_df["gen_q"] - bus_df["load_q"] - bus_df["shunt_q"] + bus_df["bat_q"]

    return bus_df
