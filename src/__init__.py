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

    from src import Network
    
    # Initialize network from PowerFactory
    net = Network(app, base_mva=100.0)
    
    # Build matrices and run load flow
    net.build_matrices()
    net.run_load_flow()
    
    # Reduce to generator nodes
    DIST_GEN = "SG 11"  # <-- Enter generator name here
    MODE = 1
    net.reduce_to_generators(outage_source_name=DIST_GEN, MODE=MODE)

    # Calculate power distribution ratios (returns ratios and matching gen names)
    ratios, sources_name_order, sources_types = net.calculate_power_ratios(DIST_GEN, MODE)

Logging
-------
This library uses Python's standard logging module. By default, no output is shown.
To enable logging:

    import logging
    logging.getLogger("src").setLevel(logging.INFO)
    
For detailed debug output:

    logging.getLogger("src").setLevel(logging.DEBUG)
"""

import logging

from .utils.connection import load_powerfactory

# The existing modules import PowerFactory at module load time. Configure its
# API path before importing those modules below.
load_powerfactory()

__version__ = "0.2.0.dev0"

# Configure library logging (NullHandler prevents "No handler found" warnings)
logging.getLogger(__name__).addHandler(logging.NullHandler())
__author__ = "LPEE"

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
    PVSystemShunt,
    ExternalGridShunt,
    VoltageSourceShunt,
)

# Matrix functions
from .matrices import (
    MatrixType,
    build_admittance_matrix,
    perform_kron_reduction,
    calculate_power_distribution_ratios,
    calculate_power_distribution_ratios_from_reduced_column,
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
    connect,
    init_project,
    import_pfd_file,
)

__all__ = [
    # Version
    '__version__',
    'load_powerfactory',
    
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
    'PVSystemShunt',
    'ExternalGridShunt',
    'VoltageSourceShunt',
    
    # Matrix types and functions
    'MatrixType',
    'build_admittance_matrix',
    'perform_kron_reduction',
    'calculate_power_distribution_ratios',
    'calculate_power_distribution_ratios_from_reduced_column',
    
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
    'connect',
    'init_project',
    'import_pfd_file',
]
