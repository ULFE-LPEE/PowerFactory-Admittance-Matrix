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

    from src import Network, connect
    from src.adapters.powerfactory import extract_network, extract_operating_point
    from src.outage_analysis import SynchroCoefficients

    app = connect("PowerFactory project", show=True)
    network = extract_network(app)
    operating_point = extract_operating_point(app, network)
    result = SynchroCoefficients(network, operating_point).calculate("SG 11")

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
from typing import TYPE_CHECKING, Any

from .adapters.powerfactory import (
    BusResult,
    ExternalGridResult,
    GeneratorResult,
    VoltageSourceResult,
    extract_network,
    extract_operating_point,
    get_bus_full_name,
    get_external_grid_data_from_pf,
    get_generator_data_from_pf,
    get_load_flow_results,
    get_voltage_source_data_from_pf,
    run_load_flow,
)
from .matrices import perform_kron_reduction
from .network import (
    Branch,
    Bus,
    Element,
    ExternalGridShunt,
    GeneratorShunt,
    LineBranch,
    LoadShunt,
    Network,
    OperatingPoint,
    PVSystemShunt,
    Shunt,
    SourceShunt,
    SwitchBranch,
    Transformer3WBranch,
    TransformerBranch,
    VoltageSourceShunt,
)
from .utils import connect, load_powerfactory

if TYPE_CHECKING:
    from .utils.helpers import import_pfd_file, init_project

__version__ = "0.2.0.dev0"

# Configure library logging (NullHandler prevents "No handler found" warnings)
logging.getLogger(__name__).addHandler(logging.NullHandler())
__author__ = "LPEE"


def __getattr__(name: str) -> Any:
    """Load helpers that require PowerFactory only when requested."""
    if name in {"init_project", "import_pfd_file"}:
        from . import utils

        value = getattr(utils, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    "load_powerfactory",
    # Standalone network classes
    "Network",
    "OperatingPoint",
    "Element",
    "Bus",
    "Branch",
    "LineBranch",
    "SwitchBranch",
    "TransformerBranch",
    "Transformer3WBranch",
    "Shunt",
    "SourceShunt",
    "LoadShunt",
    "GeneratorShunt",
    "PVSystemShunt",
    "ExternalGridShunt",
    "VoltageSourceShunt",
    # Matrix types and functions
    "perform_kron_reduction",
    # Result classes
    "BusResult",
    "GeneratorResult",
    "VoltageSourceResult",
    "ExternalGridResult",
    # PowerFactory adapter functions
    "get_bus_full_name",
    "extract_network",
    "extract_operating_point",
    "run_load_flow",
    "get_load_flow_results",
    "get_generator_data_from_pf",
    "get_voltage_source_data_from_pf",
    "get_external_grid_data_from_pf",
    # Utilities
    "connect",
    "init_project",
    "import_pfd_file",
]
