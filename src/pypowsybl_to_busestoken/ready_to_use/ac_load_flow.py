# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Pre-configured converter for large-grid OpenLoadFlow use.

The default pypowsybl load-flow parameters (NR with flat start) often stall on
large meshed transmission networks (MAX_ITERATION_REACHED).
:class:`ACLoadFlowBusesTokenConverter` ships with conservative settings that are
often useful for operational transmission-grid snapshots:

- ``voltage_init_mode = DC_VALUES``  — warm-start voltages from a DC solution,
  dramatically reduces NR iterations on stressed or heavily-loaded networks.
- ``transformer_voltage_control_on = True`` — models tap-changer regulation.
- ``use_reactive_limits = True`` — respects generator Q limits (PQ switching).
- ``connected_component_mode = MAIN`` — only solves the largest synchronous
  island, skipping isolated micro-components that would fail individually.

These settings are provided as a practical starting point and can be overridden
for a specific study or operator configuration.
"""

from __future__ import annotations

import pypowsybl.loadflow as lf

from pypowsybl_to_busestoken._compat import make_component_mode_kwarg
from pypowsybl_to_busestoken.converter import BusesTokenConverter

#: Recommended load-flow parameters for large meshed transmission networks.
#: Built with :func:`~pypowsybl_to_busestoken._compat.make_component_mode_kwarg`
#: so the component-mode argument is correct for any installed pypowsybl version.
RTE_OLF_PARAMS = lf.Parameters(
    voltage_init_mode=lf.VoltageInitMode.DC_VALUES,
    transformer_voltage_control_on=True,
    use_reactive_limits=True,
    **make_component_mode_kwarg(main_only=True),
)


class ACLoadFlowBusesTokenConverter(BusesTokenConverter):
    """
    :class:`BusesTokenConverter` pre-configured with robust OpenLoadFlow parameters.

    Produces a :class:`~pypowsybl_to_busestoken.BusesToken` from any pypowsybl-
    compatible network file.  The AC load flow is run automatically with the
    :data:`RTE_OLF_PARAMS` settings.

    Parameters
    ----------
    provider : str
        Load-flow provider name.  Defaults to ``'OpenLoadFlow'``.
    run_lf : bool
        Whether to run AC load flow before building the token representation.
        Defaults to ``True``.  Set to ``False`` when the network was solved
        externally.

    Examples
    --------
    Simplest usage — load a snapshot and get a BusesToken::

        from pypowsybl_to_busestoken.ready_to_use import ACLoadFlowBusesTokenConverter

        converter = ACLoadFlowBusesTokenConverter()
        token = converter.from_file("network.xiidm")
        print(token)
        # BusesToken(snapshot='network', n_tokens=..., n_relations=..., ...)

    Use the pre-configured params directly::

        import pypowsybl.network as pn
        from pypowsybl_to_busestoken.ready_to_use import ACLoadFlowBusesTokenConverter, RTE_OLF_PARAMS
        import pypowsybl.loadflow as lf

        network = pn.load("snapshot.xiidm")
        lf.run_ac(network, parameters=RTE_OLF_PARAMS, provider="OpenLoadFlow")

        converter = ACLoadFlowBusesTokenConverter(run_lf=False)
        token = converter(network, snapshot_id="snapshot")
    """

    def __init__(
        self,
        provider: str = "OpenLoadFlow",
        run_lf: bool = True,
    ) -> None:
        super().__init__(
            lf_params=RTE_OLF_PARAMS,
            provider=provider,
            run_lf=run_lf,
        )
