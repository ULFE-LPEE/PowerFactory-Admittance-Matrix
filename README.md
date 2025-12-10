# PowerFactory Admittance Matrix Library

A Python library for building admittance matrices from DIgSILENT PowerFactory networks.

## Features

- **Network Element Extraction**: Extract lines, switches, generators, and loads using cubicle-based connectivity
- **Admittance Matrix Building**: Build Y-matrices in per-unit on system base (100 MVA default)
- **Matrix Types**: Support for load flow, stability, and stability-full matrix types
- **Kron Reduction**: Reduce to generator internal buses for transient stability analysis
- **Power Distribution Ratios**: Calculate synchronizing power coefficients for generator trip scenarios

## Installation

```bash
# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Quick Start

### Using the High-Level Network Class

```python
import sys
sys.path.insert(0, r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP4A\Python\3.12")
import powerfactory as pf

from admittance_matrix import Network

# Connect to PowerFactory
app = pf.GetApplication()
app.ActivateProject("User\\MyProject")

# Initialize network
net = Network(app, base_mva=100.0)

# Build matrices and run load flow
net.build_matrices()
net.run_load_flow()

# Reduce to generator internal buses
net.reduce_to_generators()

# Get generator data
for name in net.gen_names:
    gen = net.get_generator(name)
    print(f"{name}: E' = {gen.internal_voltage_mag:.4f}∠{gen.internal_voltage_angle:.2f}°")

# Calculate power distribution ratios
ratios = net.calculate_power_ratios("G1")
for name, ratio in zip(net.gen_names, ratios):
    print(f"{name}: {ratio*100:.1f}%")
```

### Using Individual Functions

```python
from admittance_matrix import (
    get_network_elements,
    get_unique_buses,
    build_admittance_matrix,
    MatrixType,
    reduce_to_generator_internal_buses,
    run_load_flow,
    get_load_flow_results,
    get_generator_data_from_pf,
    calculate_power_distribution_ratios,
)

# Extract network elements
branches, shunts = get_network_elements(app)
bus_names = get_unique_buses(branches, shunts)

# Build matrices
Y_lf, bus_idx = build_admittance_matrix(
    branches, shunts, bus_names,
    matrix_type=MatrixType.LOAD_FLOW,
    base_mva=100.0
)

Y_stab, _ = build_admittance_matrix(
    branches, shunts, bus_names,
    matrix_type=MatrixType.STABILITY,
    base_mva=100.0
)

# Reduce to generator internal buses
Y_reduced, gen_names = reduce_to_generator_internal_buses(Y_stab, bus_idx, shunts)

# Run load flow and get results
run_load_flow(app)
lf_results = get_load_flow_results(app)
gen_data = get_generator_data_from_pf(app, shunts, lf_results)

# Calculate power distribution ratios
ratios, _ = calculate_power_distribution_ratios(Y_reduced, gen_data, "G1")
```

## Module Structure

```
admittance_matrix/
├── __init__.py           # Main exports
├── core/
│   ├── elements.py       # BranchElement, ShuntElement classes
│   └── network.py        # High-level Network wrapper
├── matrices/
│   ├── builder.py        # build_admittance_matrix()
│   ├── reducer.py        # kron_reduction(), reduce_to_generator_internal_buses()
│   └── analysis.py       # calculate_power_distribution_ratios()
├── powerflow/
│   ├── results.py        # BusResult, GeneratorResult classes
│   ├── extractor.py      # get_network_elements()
│   └── solver.py         # run_load_flow(), get_load_flow_results()
└── utils/
    └── helpers.py        # Utility functions
```

## Requirements

- Python 3.10+
- NumPy
- DIgSILENT PowerFactory (for network extraction)

## Theory

### Per-Unit Conversion

All admittances are converted to per-unit on the system base:

```
Y_pu = Y_siemens × (V²_base / S_base)
```

### Generator Internal Voltage

Generator internal voltage E' behind sub-transient reactance X''d:

```
E' = V + jX''d × (S*/V*)
```

### Kron Reduction

The Kron reduction formula for eliminating buses:

```
Y_red = Y_AA - Y_AB × inv(Y_BB) × Y_BA
```

### Power Distribution Ratios

Synchronizing power coefficient between generators:

```
K_ij = E_i × E_j × (B_ij × cos(δ_i - δ_j) - G_ij × sin(δ_i - δ_j))
```

## License

MIT License
