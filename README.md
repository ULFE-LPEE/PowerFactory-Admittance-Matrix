# PowerFactory Admittance Matrix Library

A Python library for extracting admittance matrices from DIgSILENT PowerFactory networks.

## Notice

⚠️ **This library is under active development.**

The library currently supports extraction of multiple PowerFactory network elements, however some components require further refinement—particularly the proper handling of voltage tap settings for 2-winding and 3-winding transformers.

If you encounter any issues or would like to request new functionality, please [open an issue](https://github.com/ULFE-LPEE/PowerFactory-Admittance-Matrix/issues) on GitHub or contact the developer directly at martin.valencic@fe.uni-lj.si.

## Features

- Extract load flow and stability admittance matrices from PowerFactory
- Kron reduction to generator internal buses
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
licensed PowerFactory installation and is not available from PyPI. Before
importing this library, make the PowerFactory Python API directory available to
the chosen interpreter through `PYTHONPATH`. Its Python version must match the
interpreter in the uv environment. Do not add a machine-specific API path to
the repository.

From the repository root, run:

```powershell
uv sync --extra speed
uv run --extra speed python -c "import admittance_matrix; print(admittance_matrix.__version__)"
```

`uv sync` installs the library in editable mode, so changes to its Python source
are available on the next Python process. Commit `uv.lock` with dependency
updates, and use `uv sync --locked` when reproducing a recorded environment.
The `speed` extra installs SciPy for the optional sparse Kron-reduction path;
without it, the library uses NumPy's dense solver. Development tools are in
the `dev` dependency group, which uv includes by default.

## Quick Start

```python
import powerfactory as pf

from admittance_matrix import Network
from admittance_matrix.utils import init_project
import pandas as pd

# Connect to PowerFactory
app = pf.GetApplicationExt()
init_project(app, "Lokalizacija\\11_bus_radial_system") # Enter your PF project path here

# Initialize network and build matrices
net = Network(app, base_mva=100.0)
net.build_matrices()

# Access the matrices
Y_loadflow = net.Y_lf_matrix       # Load flow admittance matrix
Y_stability = net.Y_stab_matrix    # Stability admittance matrix (with generator reactances)

print(f"Load flow Y-matrix shape: {Y_loadflow.shape}")
print(f"Stability Y-matrix shape: {Y_stability.shape}")

# Display with bus names as index and columns
pd.DataFrame(Y_loadflow, index=net.bus_names, columns=net.bus_names)
```

## Module Structure

```
admittance_matrix/
├── adapters/
│   └── powerfactory/     # PowerFactory-specific code
│       ├── extractor.py  # Network element extraction
│       ├── loadflow.py   # Load flow execution & results
│       ├── naming.py     # Bus naming utilities
│       └── results.py    # Result dataclasses
├── core/
│   ├── elements.py       # BranchElement, ShuntElement classes
│   ├── network.py        # High-level Network wrapper
│   └── reductionEngine.py # Network reduction engine
├── matrices/
│   ├── builder.py        # build_admittance_matrix()
│   ├── reducer.py        # Kron reduction functions
│   ├── analysis.py       # Power distribution ratio calculations
│   └── topology.py       # Used for network simplification
└── utils/
    └── helpers.py        # Utility functions
```

## Logging

By default, the library produces no console output. To enable logging:

```python
import logging
logging.getLogger("admittance_matrix").setLevel(logging.WARNING)
```

For detailed debug output:

```python
logging.getLogger("admittance_matrix").setLevel(logging.INFO)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this library in academic work, please cite it using [CITATION.cff](CITATION.cff).
