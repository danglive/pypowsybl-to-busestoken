# PyPowSyBl to BusesToken

Convert solved PyPowSyBl networks into bus-token representations for graph-based
power-grid machine learning.

The package turns an operational network state into:

- **bus tokens** carrying voltage state, aggregated injections, nominal voltage,
  and connected-component metadata;
- **branch relations** carrying active topology, electrical parameters, operating
  flows, permanent current limits, and N-0 loading ratio;
- **sparse relation indices** compatible with graph neural networks and sparse
  attention models.

No operational dataset is included in this repository. Tests use public IEEE
networks created by PyPowSyBl or synthetic tensors.

## Installation

```shell
pip install -e ".[dev]"
```

## Quick Start

```python
import pypowsybl.network as pn
from pypowsybl_to_busestoken import BusesTokenConverter

network = pn.create_ieee14()
converter = BusesTokenConverter(run_lf=True)
token = converter(network, snapshot_id="ieee14")

print(token)
# BusesToken(snapshot='ieee14', n_tokens=..., n_relations=..., ...)

x_bus = token.token_features
edge_attr = token.relation_features
edge_index = token.relation_index
```

For a network file supported by PyPowSyBl:

```python
from pypowsybl_to_busestoken import BusesTokenConverter

converter = BusesTokenConverter(run_lf=True, provider="OpenLoadFlow")
token = converter.from_file("network.xiidm")
```

## Feature Schema

Bus token features include:

```text
v_mag, v_angle, nominal_v, is_main_component,
gen_p, gen_q, load_p, load_q, shunt_q, bat_p, bat_q,
p_net, q_net,
n_gens, n_loads, n_shunts, n_batteries
```

Branch relation features include:

```text
r, x, g1, b1, g2, b2, tap_rho, tap_alpha,
p1, q1, i1, p2, q2, i2,
limit1, limit2, base_rho,
is_line, is_2wt, is_self_loop
```

`base_rho` is derived as:

```text
base_rho = max(i1 / limit1, i2 / limit2)
```

where `limit1` and `limit2` are permanent current limits when available.

## Tests

```shell
python3 -m pytest tests -q
```

Some load-flow tests may need the local PyPowSyBl/OpenLoadFlow installation to
be able to write its temporary files.

## Design Principles

The converter is deliberately close to power-system semantics:

- disconnected terminals and unsolved buses are filtered out;
- active bus states come from the solved AC load-flow state;
- branch relations preserve parallel branches and self-loop flags;
- injections are aggregated at bus level with explicit sign conventions;
- permanent current limits are exposed so downstream models can reason about
  loading and security margins.

This package is the data interface between power-grid snapshots and ML models;
it does not include private grid snapshots, generated labels, or training
artifacts.
