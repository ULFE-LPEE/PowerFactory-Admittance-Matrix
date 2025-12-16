# Examples

This folder contains Jupyter notebooks demonstrating how to use the `admittance_matrix` library.

## Prerequisites

1. **PowerFactory** must be running with an active project and study case
2. **Python packages**: `numpy`, `pandas`, `matplotlib`

## Notebooks

### 01_load_flow_admittance_matrix.ipynb

Demonstrates the basic workflow:

- Connect to PowerFactory
- Extract network elements (lines, transformers, generators, loads)
- Build the admittance matrix
- Run load flow calculation
- View Y-matrix as DataFrame
- Visualize sparsity pattern

### 02_power_distribution_ratios.ipynb

Demonstrates power redistribution analysis:

- Build network and run load flow
- View generator data
- Apply Kron reduction to generator internal buses
- Calculate power distribution ratios for a generator trip
- Bar chart visualization
- Comparison heatmap for multiple generator trips
