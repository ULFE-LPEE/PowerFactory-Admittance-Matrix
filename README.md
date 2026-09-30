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
The `speed` extra installs SciPy for the optional sparse Kron-reduction path;
without it, the library uses NumPy's dense solver. Development tools are in
the `dev` dependency group, which uv includes by default.
For the example notebook, also install the `notebook` group and select this
project's uv environment as its Jupyter kernel.

```powershell
uv sync --extra speed --group notebook
```

The importable package is now `src`, containing `core`, `matrices`, `adapters`,
and `utils`. This is a breaking import change; existing `admittance_matrix`
imports need to be updated in consuming projects.

## Quick Start

```python
from src import Network, connect

import pandas as pd

# Connect to PowerFactory
app = connect("Lokalizacija\\11_bus_radial_system", show=True)

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
src/
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
    ├── connection.py     # PowerFactory API path, import, and project connection
    └── helpers.py        # Utility functions
```

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
