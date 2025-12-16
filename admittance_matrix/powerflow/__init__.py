"""
PowerFactory load flow and network extraction functions.

DEPRECATED: This module is a compatibility shim. Import from 
admittance_matrix.adapters.powerfactory instead.
"""

import warnings

warnings.warn(
    "admittance_matrix.powerflow is deprecated. "
    "Use admittance_matrix.adapters.powerfactory instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backwards compatibility
from ..adapters.powerfactory import (
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

__all__ = [
    'BusResult',
    'GeneratorResult',
    'VoltageSourceResult',
    'ExternalGridResult',
    'get_bus_full_name',
    'get_network_elements',
    'run_load_flow',
    'get_load_flow_results',
    'get_generator_data_from_pf',
    'get_voltage_source_data_from_pf',
    'get_external_grid_data_from_pf',
]
