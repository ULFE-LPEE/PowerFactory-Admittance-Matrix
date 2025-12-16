"""
PowerFactory Admittance Matrix Library
======================================

A Python library for building admittance matrices from DIgSILENT PowerFactory networks.

Features:
- Extract network elements (lines, switches, generators, loads) using cubicle connectivity
- Build admittance matrices in per-unit on system base
- Support for load flow and stability analysis matrix types
- Kron reduction to generator internal buses
- Power distribution ratio calculations

Quick Start
-----------

Using the high-level Network class:

    from admittance_matrix import Network
    
    # Initialize network from PowerFactory
    net = Network(app, base_mva=100.0)
    
    # Build matrices and run load flow
    net.build_matrices()
    net.run_load_flow()
    
    # Reduce to generator internal buses
    net.reduce_to_generators()
    
    # Calculate power distribution ratios
    ratios = net.calculate_power_ratios("G1")

Using individual functions:

    from admittance_matrix import (
        get_network_elements,
        build_admittance_matrix,
        MatrixType,
        kron_reduction,
    )
    
    # Extract elements
    branches, shunts = get_network_elements(app)
    
    # Build matrix
    Y, bus_idx = build_admittance_matrix(branches, shunts, bus_names)
"""

__version__ = "0.1.1"
__author__ = "PowerFactory User"

# Core classes
from .core import (
    Network,
    BranchElement,
    LineBranch,
    SwitchBranch,
    TransformerBranch,
    Transformer3WBranch,
    ShuntElement,
    LoadShunt,
    GeneratorShunt,
    ExternalGridShunt,
    VoltageSourceShunt,
)

# Matrix functions
from .matrices import (
    MatrixType,
    build_admittance_matrix,
    get_unique_buses,
    get_generator_buses,
    kron_reduction,
    reduce_to_generator_internal_buses,
    calculate_power_distribution_ratios,
)

# PowerFactory adapter functions (new canonical location)
from .adapters.powerfactory import (
    BusResult,
    GeneratorResult,
    VoltageSourceResult,
    ExternalGridResult,
    get_bus_full_name,
    get_network_elements,
    run_load_flow,
    get_load_flow_results,
    get_generator_data_from_pf,
    get_voltage_source_data_from_pf,
    get_external_grid_data_from_pf,
)

# Utilities
from .utils import (
    init_project,
    import_pfd_file,
)

__all__ = [
    # Version
    '__version__',
    
    # Core classes
    'Network',
    'BranchElement',
    'LineBranch',
    'SwitchBranch',
    'TransformerBranch',
    'Transformer3WBranch',
    'ShuntElement',
    'LoadShunt',
    'GeneratorShunt',
    'ExternalGridShunt',
    'VoltageSourceShunt',
    
    # Matrix types and functions
    'MatrixType',
    'build_admittance_matrix',
    'get_unique_buses',
    'get_generator_buses',
    'kron_reduction',
    'reduce_to_generator_internal_buses',
    'calculate_power_distribution_ratios',
    
    # Result classes
    'BusResult',
    'GeneratorResult',
    'VoltageSourceResult',
    'ExternalGridResult',
    
    # PowerFactory adapter functions
    'get_bus_full_name',
    'get_network_elements',
    'run_load_flow',
    'get_load_flow_results',
    'get_generator_data_from_pf',
    'get_voltage_source_data_from_pf',
    'get_external_grid_data_from_pf',
    
    # Utilities
    'init_project',
    'import_pfd_file',
]
