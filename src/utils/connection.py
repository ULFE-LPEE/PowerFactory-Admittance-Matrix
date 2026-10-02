"""Connection helpers for the PowerFactory application."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from dotenv import find_dotenv, load_dotenv

if TYPE_CHECKING:
    import powerfactory as pf

POWERFACTORY_PATH_ENV = "POWERFACTORY_PYTHON_PATH"


class _PowerFactoryModule(Protocol):
    def GetApplicationExt(self) -> pf.Application | None: ...


def get_powerfactory_python_path() -> Path:
    """Return the configured PowerFactory Python API directory."""
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file)

    configured_path = os.getenv(POWERFACTORY_PATH_ENV)
    if not configured_path:
        raise RuntimeError(f"{POWERFACTORY_PATH_ENV} is not configured. Create a .env file based on .env.example.")

    api_path = Path(configured_path).expanduser().resolve()
    if not api_path.is_dir():
        raise RuntimeError(f"PowerFactory Python API path does not exist: {api_path}")
    return api_path


def load_powerfactory() -> ModuleType:
    """Import the PowerFactory API from the configured directory."""
    api_path = get_powerfactory_python_path()
    path_string = str(api_path)

    if path_string not in sys.path:
        sys.path.insert(0, path_string)

    try:
        import powerfactory as pf

        return pf

    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"Could not import the PowerFactory Python API from {api_path}. Check that the PowerFactory API and Python versions match."
        ) from error


def connect(project_name: str, *, show: bool = False) -> pf.Application:
    """Open PowerFactory and activate the requested project.

    PowerFactory returns zero from ``ActivateProject`` on success.
    """
    powerfactory = cast(_PowerFactoryModule, load_powerfactory())
    application = powerfactory.GetApplicationExt()

    if application is None:
        raise RuntimeError("PowerFactory did not return an Application instance.")

    if application.ActivateProject(project_name):
        raise RuntimeError(f"Could not activate PowerFactory project: {project_name}")

    if show:
        application.Show()

    return application
