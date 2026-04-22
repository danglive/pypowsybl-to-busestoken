# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Integration tests for pypowsybl-to-busestoken.

Run with:
    pytest tests/test_converter.py -v
"""

from __future__ import annotations

import numpy as np
import pypowsybl.loadflow as lf
import pypowsybl.network as pn
import pytest

import pypowsybl_to_busestoken as pb
from pypowsybl_to_busestoken._token import TOKEN_NUMERICAL_COLS, RELATION_NUMERICAL_COLS
from pypowsybl_to_busestoken.ready_to_use import ACLoadFlowBusesTokenConverter, RTE_OLF_PARAMS

IEEE_NETWORK_FACTORIES = [
    ("ieee9", pn.create_ieee9),
    ("ieee14", pn.create_ieee14),
    ("ieee30", pn.create_ieee30),
    ("ieee57", pn.create_ieee57),
    ("ieee118", pn.create_ieee118),
    ("ieee300", pn.create_ieee300),
]


# ===========================================================================
# Helpers
# ===========================================================================

def _assert_token_invariants(token: pb.BusesToken, label: str) -> None:
    """Common invariant checks applied to every BusesToken."""
    assert token.n_tokens > 0, f"{label}: no tokens"
    assert token.n_relations > 0, f"{label}: no relations"

    # bus_df index must be unique strings
    assert token.bus_df.index.is_unique, f"{label}: duplicate bus_ids in bus_df"
    assert token.bus_df.index.dtype == object, f"{label}: bus index is not str"

    # relation_df index must be unique strings
    assert token.relation_df.index.is_unique, f"{label}: duplicate branch_ids in relation_df"
    assert token.relation_df.index.dtype == object, f"{label}: relation index is not str"

    # relation endpoints must reference known buses (no empty strings after filtering)
    active_buses = set(token.bus_df.index)
    bad_src = token.relation_df.loc[~token.relation_df["bus1_id"].isin(active_buses), "bus1_id"]
    bad_dst = token.relation_df.loc[~token.relation_df["bus2_id"].isin(active_buses), "bus2_id"]
    assert len(bad_src) == 0, f"{label}: {len(bad_src)} relations with unknown bus1_id"
    assert len(bad_dst) == 0, f"{label}: {len(bad_dst)} relations with unknown bus2_id"

    # feature matrices
    X_token = token.token_features
    X_rel = token.relation_features
    assert X_token.shape == (token.n_tokens, len(token.token_feature_names))
    assert X_rel.shape == (token.n_relations, len(token.relation_feature_names))
    assert X_token.dtype == np.float32
    assert X_rel.dtype == np.float32
    assert not np.isnan(X_token).any(), f"{label}: NaN in token_features after fillna"
    assert not np.isnan(X_rel).any(), f"{label}: NaN in relation_features after fillna"

    # relation_index
    ri = token.relation_index
    assert ri.shape == (2, token.n_relations)
    assert ri.dtype == np.int64
    assert (ri >= 0).all(), f"{label}: negative index in relation_index"
    assert (ri < token.n_tokens).all(), f"{label}: relation_index out of range"

    # v_mag must be positive for all active buses
    assert (token.bus_df["v_mag"] > 0).all(), f"{label}: non-positive v_mag"

    # base_rho must be >= 0 where defined
    rho = token.relation_df["base_rho"].dropna()
    assert (rho >= 0).all(), f"{label}: negative base_rho"

    # branch_kind must only contain LINE or 2WT
    kinds = set(token.relation_df["branch_kind"].unique())
    assert kinds <= {"LINE", "2WT"}, f"{label}: unexpected branch_kind {kinds}"

    print(token)


# ===========================================================================
# PyPowSyBl built-in IEEE networks
# ===========================================================================

class TestBuiltInIEEENetworks:

    @pytest.mark.parametrize("case_name, factory", IEEE_NETWORK_FACTORIES, ids=[n for n, _ in IEEE_NETWORK_FACTORIES])
    def test_conversion_all_available_ieee_networks(self, case_name, factory):
        """Validate conversion on all standard IEEE networks bundled by PyPowSyBl."""
        network = factory()
        converter = pb.BusesTokenConverter(run_lf=True)
        token = converter(network, snapshot_id=case_name)

        _assert_token_invariants(token, case_name)
        assert token.snapshot_id == case_name

    @pytest.mark.parametrize("case_name, factory", IEEE_NETWORK_FACTORIES, ids=[n for n, _ in IEEE_NETWORK_FACTORIES])
    def test_ready_to_use_converter_all_available_ieee_networks(self, case_name, factory):
        """The robust OpenLoadFlow preset should also convert every built-in IEEE case."""
        network = factory()
        converter = ACLoadFlowBusesTokenConverter()
        token = converter(network, snapshot_id=f"{case_name}_preset")

        _assert_token_invariants(token, f"{case_name}_preset")
        assert token.snapshot_id == f"{case_name}_preset"


# ===========================================================================
# IEEE-14 (small, fast, always available via pypowsybl)
# ===========================================================================

class TestIEEE14:

    def setup_method(self):
        self.network = pn.create_ieee14()
        # Run AC load flow manually so we can pass run_lf=False
        results = lf.run_ac(self.network)
        assert results[0].status.name == "CONVERGED", "IEEE-14 LF did not converge"

    def test_basic_conversion(self):
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network, snapshot_id="ieee14")
        _assert_token_invariants(token, "IEEE-14")
        assert token.snapshot_id == "ieee14"

    def test_repr_contains_key_info(self):
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        r = repr(token)
        assert "BusesToken" in r
        assert "n_tokens" in r
        assert "n_relations" in r

    def test_bus_df_required_columns(self):
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        for col in ["v_mag", "v_angle", "nominal_v", "p_net", "q_net"]:
            assert col in token.bus_df.columns, f"Missing bus column: {col}"

    def test_relation_df_required_columns(self):
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        for col in ["bus1_id", "bus2_id", "branch_kind", "r", "x", "p1", "p2", "i1", "i2"]:
            assert col in token.relation_df.columns, f"Missing relation column: {col}"

    def test_token_features_schema(self):
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        # token_feature_names must be a subset of TOKEN_NUMERICAL_COLS
        assert set(token.token_feature_names) <= set(TOKEN_NUMERICAL_COLS)

    def test_relation_features_schema(self):
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        assert set(token.relation_feature_names) <= set(RELATION_NUMERICAL_COLS)

    def test_run_lf_true(self):
        """Converter should run load flow internally and still produce a valid token."""
        network = pn.create_ieee14()
        converter = pb.BusesTokenConverter(run_lf=True)
        token = converter(network, snapshot_id="ieee14_with_lf")
        _assert_token_invariants(token, "IEEE-14 (run_lf=True)")

    def test_to_networkx(self):
        pytest.importorskip("networkx")
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        G = token.to_networkx()
        import networkx as nx
        assert isinstance(G, nx.MultiDiGraph)
        assert G.number_of_nodes() == token.n_tokens
        assert G.number_of_edges() == token.n_relations

    def test_net_injection_consistency(self):
        """p_net = gen_p - load_p + bat_p (by construction)."""
        converter = pb.BusesTokenConverter(run_lf=False)
        token = converter(self.network)
        df = token.bus_df
        expected_p_net = df["gen_p"] - df["load_p"] + df["bat_p"]
        np.testing.assert_allclose(
            df["p_net"].values, expected_p_net.values, atol=1e-4,
            err_msg="p_net != gen_p - load_p + bat_p"
        )


# ===========================================================================
# IEEE-300 (medium, stresses relation filtering logic)
# ===========================================================================

class TestIEEE300:

    def test_basic_conversion(self):
        network = pn.create_ieee300()
        converter = pb.BusesTokenConverter(run_lf=True)
        token = converter(network, snapshot_id="ieee300")
        _assert_token_invariants(token, "IEEE-300")
        assert token.n_tokens >= 100  # sanity: not a degenerate token

    def test_no_empty_bus_ids_in_relations(self):
        """Relation filtering must remove all disconnected terminals (bus_id == '')."""
        network = pn.create_ieee300()
        converter = pb.BusesTokenConverter(run_lf=True)
        token = converter(network)
        assert not (token.relation_df["bus1_id"] == "").any(), "empty bus1_id in relation_df"
        assert not (token.relation_df["bus2_id"] == "").any(), "empty bus2_id in relation_df"

    def test_2wt_shunt_cols_are_nan(self):
        """2-winding transformers must have NaN for g1/b1/g2/b2 (schema compatibility)."""
        network = pn.create_ieee300()
        converter = pb.BusesTokenConverter(run_lf=True)
        token = converter(network)
        twt = token.relation_df[token.relation_df["branch_kind"] == "2WT"]
        if len(twt) > 0:
            for col in ["g1", "b1", "g2", "b2"]:
                if col in twt.columns:
                    assert twt[col].isna().all(), f"2WT row has non-NaN {col}"

    def test_is_line_is_2wt_flags(self):
        """is_line and is_2wt must be mutually exclusive and cover all rows."""
        network = pn.create_ieee300()
        converter = pb.BusesTokenConverter(run_lf=True)
        token = converter(network)
        df = token.relation_df
        assert ((df["is_line"] + df["is_2wt"]) == 1).all(), "is_line + is_2wt != 1"


# ===========================================================================
# Ready-to-use: ACLoadFlowBusesTokenConverter
# ===========================================================================

class TestACLoadFlowConverter:

    def test_ieee14_with_ready_to_use(self):
        converter = ACLoadFlowBusesTokenConverter()
        token = converter(pn.create_ieee14(), snapshot_id="ieee14_rtc")
        _assert_token_invariants(token, "IEEE-14 (ACLoadFlowBusesTokenConverter)")

    def test_rte_olf_params_attributes(self):
        from pypowsybl_to_busestoken._compat import _HAS_NEW_COMPONENT_MODE

        assert RTE_OLF_PARAMS.voltage_init_mode == lf.VoltageInitMode.DC_VALUES
        assert RTE_OLF_PARAMS.transformer_voltage_control_on is True
        assert RTE_OLF_PARAMS.use_reactive_limits is True

        # Check component-mode using the correct API for the installed version.
        if _HAS_NEW_COMPONENT_MODE:
            import pypowsybl._pypowsybl as _pp
            assert RTE_OLF_PARAMS.component_mode == _pp.ComponentMode.MAIN_CONNECTED
        else:
            assert RTE_OLF_PARAMS.connected_component_mode == lf.ConnectedComponentMode.MAIN

    def test_from_module_import(self):
        """Top-level package must expose BusesTokenConverter and BusesToken."""
        assert hasattr(pb, "BusesTokenConverter")
        assert hasattr(pb, "BusesToken")
        assert hasattr(pb.ready_to_use, "ACLoadFlowBusesTokenConverter")
        assert hasattr(pb.ready_to_use, "RTE_OLF_PARAMS")
