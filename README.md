# PowerFactory Admittance Matrix Library

A Python library for extracting admittance matrices from DIgSILENT PowerFactory networks.

## Notice

⚠️ **This library is under active development.**

**PowerFactory compatibility:** This library has been tested with DIgSILENT PowerFactory 2024 SP4A only. Other PowerFactory versions have not yet been validated.

The library currently supports extraction of multiple PowerFactory network elements, however some components require further refinement—particularly the proper handling of voltage tap settings for 2-winding and 3-winding transformers.

If you encounter any issues or would like to request new functionality, please [open an issue](https://github.com/ULFE-LPEE/PowerFactory-Admittance-Matrix/issues) on GitHub or contact the developer directly at martin.valencic@fe.uni-lj.si.

## Features

- Extract load flow and stability admittance matrices from PowerFactory
- Kron reduction to active-source internal nodes
- Power distribution ratio calculations for generator trip scenarios

## Installation

### From GitHub

```bash
pip install git+https://github.com/ULFE-LPEE/PowerFactory-Admittance-Matrix.git
```

To update to the latest version:

```bash
pip install --upgrade git+https://github.com/ULFE-LPEE/PowerFactory-Admittance-Matrix.git
```

### Local development

The project uses [uv](https://docs.astral.sh/uv/) to manage its development
environment. Python 3.12 is selected by `.python-version`; the package itself
supports Python 3.10 and newer. The PowerFactory Python module is supplied by a
licensed PowerFactory installation and is not available from PyPI. Copy
`.env.example` to `.env` and set `POWERFACTORY_PYTHON_PATH` to the API directory
whose Python version matches the uv environment. The local `.env` is ignored by
Git. An environment variable with the same name overrides the `.env` value.

From the repository root, run:

```powershell
uv sync --extra speed
uv run --extra speed python -c "import src; print(src.__version__)"
```

`uv sync` installs the library in editable mode, so changes to its Python source
are available on the next Python process. Commit `uv.lock` with dependency
updates, and use `uv sync --locked` when reproducing a recorded environment.
The `speed` extra installs SciPy for sparse Kron reduction and branch-flow solves;
without it, the library uses NumPy's dense solver. Development tools are in
the `dev` dependency group, which uv includes by default.
For the example notebook, also install the `notebook` group and select this
project's uv environment as its Jupyter kernel.

```powershell
uv sync --extra speed --group notebook
```

The importable package is now `src`, containing `network`, `matrices`,
`outage_analysis`, `adapters`, and `utils`. `from src import Network` resolves
to the standalone model. Existing `admittance_matrix` imports need to be
updated in consuming projects.

## Quick Start

```python
from src import Network, connect
from src.adapters.powerfactory import extract_network, extract_operating_point

app = connect("Lokalizacija\\11_bus_radial_system", show=True)
network: Network = extract_network(app, base_mva=100.0)
operating_point = extract_operating_point(app, network)

passive_y = network.get_passive_y_matrix()
stability_y = network.get_stability_y_matrix(operating_point)
print(passive_y.shape, stability_y.shape)
```

`get_passive_y_matrix()` contains branches, transformers, and passive filters.
`get_load_flow_y_matrix(operating_point)` also includes constant-impedance loads;
the two matrices therefore need not be identical. The augmented stability
matrix keeps the library's source-internal-first node order.

## Module Structure

The new `src.network.Network` is a PowerFactory-independent model. It accepts
the existing branch, shunt, and three-winding-transformer elements and builds
the existing load-flow and stability matrices. The adapter can construct it
from an active PowerFactory application:

```python
from src.adapters.powerfactory import extract_network

network = extract_network(app, base_mva=100.0)
load_flow_y = network.get_load_flow_y_matrix()
passive_y = network.get_passive_y_matrix()
print(network.generator_names, network.line_names)
```

The element hierarchy is `Element` (name and optional ID), then `Bus`,
two-terminal `Branch`, and one-terminal `Shunt`. Equipment types such as
`LineBranch` and `GeneratorShunt` implement their own electrical models.
`Network.branches` and `Network.shunts` are the stored equipment collections;
`Network.lines`, `Network.generators`, `Network.loads`, and
`Network.voltage_sources` are read-only views of them. Add equipment through
the `add_branch()` and `add_shunt()` methods so connected buses are registered too.

For a solved operating point, choose one of the three outage formulations:

```python
from src.adapters.powerfactory import extract_operating_point
from src.outage_analysis import AngleUpdateFormulation, ReducedMatrixFormulation, SynchroCoefficients

operating_point = extract_operating_point(app, network)
reduced_y = network.get_stability_y_matrix(operating_point)
internal_voltages = network.get_internal_voltage_vector(operating_point)
for formulation in (SynchroCoefficients, AngleUpdateFormulation, ReducedMatrixFormulation):
    ratios_by_source = formulation(network, operating_point).calculate("SG 12")
    print(formulation.__name__, ratios_by_source)
```

All three return a dictionary of dimensionless ratios in
`operating_point.source_names` order, including zero for the tripped source.
`SynchroCoefficients` is the original Mode 0 (prefault coefficients),
`AngleUpdateFormulation` is Mode 1 (outaged admittance and angle update), and
`ReducedMatrixFormulation` is Mode 2 (prefault/postfault internal power change).
The notebook selects Mode 1 by default. None of these ratios is an absolute
active-power change in MW. Mode 2 also exposes modeled initial changes through
`power_changes_mw(outaged_source)`.

```
src/
├── adapters/
│   └── powerfactory/     # PowerFactory-specific code
│       ├── extractor.py  # Equipment extraction and unit-aware attributes
│       ├── load_flow.py  # Load-flow execution and operating-point extraction
│       ├── naming.py     # Bus naming utilities
├── network/              # PowerFactory-independent electrical model
│   ├── elements.py       # Branches, transformers, shunts, and tap changers
│   ├── network.py        # Standalone Network class
│   ├── operating_point.py # Solved bus and source data
│   └── topology.py       # Closed-switch bus merging
├── matrices/
│   ├── passive.py        # Passive and load-flow physical-bus matrices
│   ├── stability.py      # Stability bus, extended, and reduced matrices
│   └── reducer.py        # Source-node extension and Kron reduction
├── outage_analysis/
│   ├── synchronizing.py  # Mode 0: SynchroCoefficients
│   ├── angle_update.py   # Mode 1: AngleUpdateFormulation
│   ├── reduced_matrix.py # Mode 2: ReducedMatrixFormulation
│   ├── branch_flows.py   # Directional branch active-power changes
│   └── __init__.py       # Public formulation classes
└── utils/
    ├── connection.py     # PowerFactory API path, import, and project connection
    ├── rms_simulation.py # Dynamic outage simulation and result extraction
    └── helpers.py        # Utility functions
```

## Grid-forming static generators

The PowerFactory adapter includes an energized ElmGenstat as a responding
source when its simulation model uses the voltage-source representation
(iSimModel 0 or 2) and its composite model has a slot named "Virtual impedance".
Other ElmGenstat units are skipped by this initial-response model.

The simplified VSM model uses only the composite block's virtual impedance;
converter short-circuit parameters uk and Pcu are not added. **The block
parameters are currently assumed to be per unit on the network MVA base and
the terminal nominal-voltage base.** Verify that assumption for each
PowerFactory composite model before interpreting numerical results. The source
internal voltage is initialized from the solved load-flow P, Q, and bus voltage
through E = V + Z conj(S / V).

Modeled static generators appear in Network.static_generator_names and
Network.stability_source_names and respond to synchronous-generator outages
in all three formulations. The calculate_all() methods still enumerate
synchronous-generator outages; an individual static-generator outage can be
calculated by name. These formulations hold surviving internal voltages fixed
and do not reproduce VSM controller dynamics. Compare their initial response
with RMS simulation before drawing conclusions.

For comparisons, use network.generator_names for the tripped synchronous
machines and network.stability_source_names for responding sources,
including the modeled VSM units. RMS monitors must also record ElmGenstat
active power if their responses are to appear in comparison plots.

## Logging

By default, the library produces no console output. To enable logging:

```python
import logging
logging.getLogger("src").setLevel(logging.WARNING)
```

For detailed debug output:

```python
logging.getLogger("src").setLevel(logging.INFO)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this library in academic work, please cite it using [CITATION.cff](CITATION.cff).
