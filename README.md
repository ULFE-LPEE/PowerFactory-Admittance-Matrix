# PowerFactory Admittance Matrix Library

A Python library for extracting admittance matrices from DIgSILENT PowerFactory networks.

## Notice

⚠️ **This library is under active development.**

The library currently supports extraction of multiple PowerFactory network elements, however some components require further refinement—particularly the proper handling of voltage tap settings for 2-winding and 3-winding transformers.

If you encounter any issues or would like to request new functionality, please [open an issue](https://github.com/valenciaga1231/PF_Admittance_Matrix/issues) on GitHub or contact the developer directly.

## Features

- Extract load flow and stability admittance matrices from PowerFactory
- Kron reduction to generator internal buses
- Power distribution ratio calculations for generator trip scenarios

## Installation

### From GitHub

```bash
pip install git+https://github.com/valenciaga1231/PF_Admittance_Matrix.git
```

To update to the latest version:

```bash
pip install --upgrade git+https://github.com/valenciaga1231/PF_Admittance_Matrix.git
```

### Local Development

```bash
pip install -e .
```

## Quick Start

```python
import sys
sys.path.insert(0, r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP4A\Python\3.12")
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
Y_loadflow = net.Y_lf       # Load flow admittance matrix
Y_stability = net.Y_stab    # Stability admittance matrix (with generator reactances)

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
│   └── network.py        # High-level Network wrapper
├── matrices/
│   ├── builder.py        # build_admittance_matrix()
│   ├── reducer.py        # Kron reduction functions
│   ├── analysis.py       # Power distribution ratio calculations
│   └── diagnostics.py    # Network diagnostics
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
