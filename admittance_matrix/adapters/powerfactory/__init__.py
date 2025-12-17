"""
PowerFactory adapter module.

This package contains all PowerFactory-specific code:
- Network element extraction (extractor.py)
- Load flow execution and result reading (loadflow.py)
- Result data classes (results.py)
- Bus naming utilities (naming.py)
"""

from .naming import get_bus_full_name

from .results import (
    BusResult,
    GeneratorResult,
    VoltageSourceResult,
    ExternalGridResult,
)

from .extractor import (
    get_network_elements,
    get_main_bus_names,
)

from .loadflow import (
    run_load_flow,
    get_load_flow_results,
    get_generator_data_from_pf,
    get_voltage_source_data_from_pf,
    get_external_grid_data_from_pf,
)

__all__ = [
    # Naming
    'get_bus_full_name',
    # Results
    'BusResult',
    'GeneratorResult',
    'VoltageSourceResult',
    'ExternalGridResult',
    # Extraction
    'get_network_elements',
    # Load flow
    'run_load_flow',
    'get_load_flow_results',
    'get_generator_data_from_pf',
    'get_voltage_source_data_from_pf',
    'get_external_grid_data_from_pf',
]
