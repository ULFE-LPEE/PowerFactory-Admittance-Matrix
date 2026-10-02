"""Connection and PowerFactory-specific utility functions."""

from typing import TYPE_CHECKING, Any

from .connection import connect, get_powerfactory_python_path, load_powerfactory
from .rms_simulation import (
    MonitorSpec,
    event_power_changes,
    save_rms_results,
    simulate_generator_outage,
)

if TYPE_CHECKING:
    from .helpers import (
        get_simulation_data,
        get_simulation_data_with_loads,
        import_pfd_file,
        init_project,
        obtain_rms_results,
        obtain_rms_results_with_loads,
    )

_HELPER_NAMES = {
    "init_project",
    "import_pfd_file",
    "get_simulation_data",
    "get_simulation_data_with_loads",
    "obtain_rms_results_with_loads",
    "obtain_rms_results",
}


def __getattr__(name: str) -> Any:
    """Load helpers after the PowerFactory API has been configured."""
    if name in _HELPER_NAMES:
        load_powerfactory()
        from . import helpers

        return getattr(helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MonitorSpec",
    "event_power_changes",
    "save_rms_results",
    "simulate_generator_outage",
    "connect",
    "get_powerfactory_python_path",
    "load_powerfactory",
    "init_project",
    "import_pfd_file",
    "get_simulation_data",
    "get_simulation_data_with_loads",
    "obtain_rms_results_with_loads",
    "obtain_rms_results",
]
