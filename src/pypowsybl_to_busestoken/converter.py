# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""BusesTokenConverter: main entry point for the conversion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pypowsybl.loadflow as lf
import pypowsybl.network as pn

from ._buses import build_bus_df
from ._relations import build_relation_df
from ._token import BusesToken


class BusesTokenConverter:
    """
    Convert a PyPowSyBl network (loaded from a XIIDM/CGMES/... file) into a
    :class:`BusesToken` — a sparse bus-token representation of the operational state.

    The conversion pipeline is:

    1. (Optional) Run AC load flow with the provided ``lf_params`` and
       ``provider``.  If ``run_lf=False`` the network must already carry valid
       ``v_mag`` / ``v_angle`` / ``i1`` values.
    2. Build :func:`build_bus_df` — active buses with electrical-state and
       aggregated injection features.
    3. Build :func:`build_relation_df` — active branches with electrical-parameter,
       operating-state, and thermal-limit features.
    4. Wrap into :class:`BusesToken`.

    Parameters
    ----------
    lf_params : lf.Parameters or None
        Load flow parameters.  If *None*, the default parameters of the chosen
        provider are used.  For production use consider passing a pre-configured
        :class:`lf.Parameters` instance (see ``ready_to_use.ACLoadFlowBusesTokenConverter``
        for a robust OpenLoadFlow preset).
    provider : str
        Name of the load flow provider.  Default ``'OpenLoadFlow'``.  Must match
        a provider registered in the pypowsybl installation.
    run_lf : bool
        Whether to run AC load flow before building the token representation.
        Set to *False* if the network was already solved externally (e.g. inside
        a batch loop where you want to reuse a solved network object).

    Examples
    --------
    Basic usage::

        from pypowsybl_to_busestoken import BusesTokenConverter

        converter = BusesTokenConverter()
        token = converter.from_file("network.xiidm")
        print(token)
        # BusesToken(snapshot='network', n_tokens=..., n_relations=..., ...)

    Advanced usage with a pre-solved network::

        import pypowsybl.network as pn
        import pypowsybl.loadflow as lf

        network = pn.load("snapshot.xiidm")
        lf.run_ac(network, provider="OpenLoadFlow")

        converter = BusesTokenConverter(run_lf=False)
        token = converter(network, snapshot_id="snapshot")
    """

    def __init__(
        self,
        lf_params: Optional[lf.Parameters] = None,
        provider: str = "OpenLoadFlow",
        run_lf: bool = True,
    ) -> None:
        self.lf_params = lf_params
        self.provider = provider
        self.run_lf = run_lf

    def __call__(
        self,
        network: pn.Network,
        snapshot_id: Optional[str] = None,
    ) -> BusesToken:
        """
        Convert a loaded (and optionally pre-solved) network into a BusesToken.

        Parameters
        ----------
        network : pn.Network
            PyPowSyBl network object.  Modified in-place when ``run_lf=True``.
        snapshot_id : str or None
            Optional label stored in :attr:`BusesToken.snapshot_id`.

        Returns
        -------
        BusesToken

        Raises
        ------
        RuntimeError
            If ``run_lf=True`` and the load flow does not converge (status !=
            CONVERGED).  A non-converged load flow yields unreliable ``v_mag``
            / ``v_angle`` values and should not be used as a reference state.
        """
        if self.run_lf:
            results = lf.run_ac(network, parameters=self.lf_params, provider=self.provider)
            status = results[0].status.name
            if status != "CONVERGED":
                raise RuntimeError(
                    f"AC load flow did not converge (status={status}). "
                    "The resulting state is unreliable as a reference. "
                    "Check your network data or load flow parameters."
                )

        bus_df = build_bus_df(network)
        relation_df = build_relation_df(network)

        return BusesToken(
            bus_df=bus_df,
            relation_df=relation_df,
            snapshot_id=snapshot_id,
        )

    def from_file(
        self,
        path: str | Path,
        snapshot_id: Optional[str] = None,
    ) -> BusesToken:
        """
        Load a network file and convert it to a BusesToken in one call.

        Parameters
        ----------
        path : str or Path
            Path to the network file (XIIDM, CGMES, UCTE, ...).  All formats
            supported by ``pypowsybl.network.load()`` are accepted.
        snapshot_id : str or None
            If *None*, defaults to the file stem (e.g. ``'network'``
            for ``'network.xiidm'``).

        Returns
        -------
        BusesToken
        """
        path = Path(path)
        if snapshot_id is None:
            # Strip all suffixes: e.g. 'file.arc.xz' → 'file'
            snapshot_id = path.name.split(".")[0]

        network = pn.load(str(path))
        return self(network, snapshot_id=snapshot_id)
