# Examples

This folder contains Jupyter notebooks demonstrating how to use the `admittance_matrix` library.

## Prerequisites

1. **PowerFactory** must be installed
2. **Python packages**: `numpy`, `pandas`, `matplotlib`

## Network Options

When creating a `Network` object, you can use the following options:

```python
from admittance_matrix import Network

# Basic usage
net = Network(app, base_mva=100.0)

# With topology simplification (merges buses connected by closed switches)
net = Network(app, base_mva=100.0, simplify_topology=True)
```

## Notebooks

### 01_load_flow_admittance_matrix.ipynb

Demonstrates the basic workflow:

- Connect to PowerFactory
- Import a PFD file and activate project (or manually open desired network in PF)
- Extract network and build admittance matrix
- View load flow Y-matrix as DataFrame
- Run load flow calculation
- Display load flow results (voltage magnitude and angle)

### 02_stability_admittance_matrix_with_power_distribution.ipynb

Demonstrates stability analysis with power distribution ratios:

- Connect to PowerFactory and import PFD file
- Build network and run load flow
- Reduce admittance matrix to generator internal buses (Kron reduction)
- Calculate power distribution ratios for a generator trip scenario
- Bar chart visualization of power redistribution
