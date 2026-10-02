# Examples

This folder contains Jupyter notebooks demonstrating how to use the `src` library package.

## Prerequisites

1. **PowerFactory** must be installed
2. **Python packages**: `numpy`, `pandas`, `matplotlib`

## Network Options

Connect to an existing PowerFactory project and extract a standalone network:

```python
from src import connect
from src.adapters.powerfactory import extract_network, extract_operating_point

app = connect("Radial System", show=True)
network = extract_network(app, base_mva=100.0)
operating_point = extract_operating_point(app, network)

# Optional topology simplification merges buses connected by closed switches.
network = extract_network(app, base_mva=100.0, merge_closed_switches=True)
```

## Notebooks

### 01_load_flow_admittance_matrix.ipynb

Demonstrates the basic workflow:

- Connect to PowerFactory
- Connect to an existing PowerFactory project
- Extract network and build admittance matrix
- View load flow Y-matrix as DataFrame
- Read a solved operating point
- Display load flow results (voltage magnitude and angle)

### 02_stability_admittance_matrix_with_power_distribution.ipynb

Demonstrates stability analysis with power distribution ratios:

- Connect to an existing PowerFactory project
- Build the reduced stability matrix from the standalone network
- Select one of the three outage formulation classes
- Calculate source response ratios for a generator trip scenario
- Bar chart visualization of power redistribution
