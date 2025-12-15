# PowerFactory Admittance Matrix Library

A Python library for extracting admittance matrices from DIgSILENT PowerFactory networks.

## Features

- Extract load flow and stability admittance matrices from PowerFactory
- Kron reduction to generator internal buses
- Power distribution ratio calculations for generator trip scenarios

## Installation

### From GitHub

```bash
pip install git+https://github.com/valenciaga1231/PF_Admittance_Matrix.git
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
init_project(app, "Lokalizacija\\11_bus_radial_system")

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
├── core/
│   ├── elements.py       # BranchElement, ShuntElement classes
│   └── network.py        # High-level Network wrapper
├── matrices/
│   ├── builder.py        # build_admittance_matrix()
│   ├── reducer.py        # Kron reduction functions
│   └── analysis.py       # Power distribution ratio calculations
├── powerflow/
│   ├── results.py        # Result dataclasses
│   ├── extractor.py      # Network element extraction
│   └── solver.py         # Load flow interface
└── utils/
    └── helpers.py        # Utility functions
```

## Requirements

- Python 3.10+
- NumPy
- DIgSILENT PowerFactory
