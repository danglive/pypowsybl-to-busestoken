# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Compatibility helpers for pypowsybl API changes across versions.

pypowsybl renamed the connected-component parameter in lf.Parameters:
  - Old API: connected_component_mode = ConnectedComponentMode.MAIN  (ALL | MAIN)
  - New API: component_mode           = ComponentMode.MAIN_CONNECTED  (ALL_CONNECTED | MAIN_CONNECTED | MAIN_SYNCHRONOUS)

Both parameter names coexist in the transition version but the old one is
deprecated.  This module detects which API is available at import time and
exposes a single canonical helper so the rest of the package never needs to
branch on the pypowsybl version.
"""

from __future__ import annotations

import inspect

import pypowsybl.loadflow as lf

# ---------------------------------------------------------------------------
# Detect available API once at import time
# ---------------------------------------------------------------------------

_LF_PARAMS_KEYS: frozenset[str] = frozenset(
    inspect.signature(lf.Parameters.__init__).parameters
)

_HAS_NEW_COMPONENT_MODE: bool = "component_mode" in _LF_PARAMS_KEYS


def make_component_mode_kwarg(*, main_only: bool = True) -> dict:
    """
    Return the correct ``lf.Parameters`` keyword argument for component-mode
    filtering, adaptive to the installed pypowsybl version.

    Parameters
    ----------
    main_only : bool
        If ``True`` (default), restrict the load flow to the **main** connected
        component — equivalent to the old ``ConnectedComponentMode.MAIN``.
        If ``False``, solve **all** connected components.

    Returns
    -------
    dict
        A one-item dict ready to be unpacked into ``lf.Parameters(**...)``.
        Old API → ``{"connected_component_mode": ConnectedComponentMode.MAIN}``
        New API → ``{"component_mode": ComponentMode.MAIN_CONNECTED}``

    Examples
    --------
    ::

        params = lf.Parameters(
            voltage_init_mode=lf.VoltageInitMode.DC_VALUES,
            **make_component_mode_kwarg(main_only=True),
        )
    """
    if _HAS_NEW_COMPONENT_MODE:
        # New API: pypowsybl introduced ComponentMode with finer granularity.
        # MAIN_CONNECTED   ≈ old MAIN  (largest synchronously-connected island)
        # ALL_CONNECTED    ≈ old ALL
        # MAIN_SYNCHRONOUS = synchronous island only (stricter, new option)
        import pypowsybl._pypowsybl as _pp

        value = _pp.ComponentMode.MAIN_CONNECTED if main_only else _pp.ComponentMode.ALL_CONNECTED
        return {"component_mode": value}
    else:
        # Old API: ConnectedComponentMode (MAIN | ALL)
        value = lf.ConnectedComponentMode.MAIN if main_only else lf.ConnectedComponentMode.ALL
        return {"connected_component_mode": value}
