"""
Utility functions for PowerFactory operations.
"""

from .connection import connect, get_powerfactory_python_path, load_powerfactory

_HELPER_NAMES = {
    'init_project',
    'import_pfd_file',
    'get_simulation_data',
    'get_simulation_data_with_loads',
    'obtain_rms_results_with_loads',
    'obtain_rms_results',
}


def __getattr__(name: str):
    """Load helpers after the PowerFactory API has been configured."""
    if name in _HELPER_NAMES:
        from . import helpers

        return getattr(helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'connect',
    'get_powerfactory_python_path',
    'load_powerfactory',
    'init_project',
    'import_pfd_file',
    'get_simulation_data',
    'get_simulation_data_with_loads',
    'obtain_rms_results_with_loads',
    'obtain_rms_results'
]
