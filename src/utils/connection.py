"""Connection helpers for the PowerFactory application."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import find_dotenv, load_dotenv

POWERFACTORY_PATH_ENV = "POWERFACTORY_PYTHON_PATH"


def get_powerfactory_python_path() -> Path:
    """Return the configured PowerFactory Python API directory."""
    env_file = find_dotenv(usecwd=True)
    if not env_file:
        repository_env = Path(__file__).resolve().parents[2] / ".env"
        if repository_env.is_file():
            env_file = str(repository_env)
    if env_file:
        load_dotenv(env_file, override=False)

    configured_path = os.environ.get(POWERFACTORY_PATH_ENV)
    if not configured_path:
        raise RuntimeError(
            f"{POWERFACTORY_PATH_ENV} is not configured. "
            "Create a .env file based on .env.example."
        )

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
        return importlib.import_module("powerfactory")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"Could not import the PowerFactory Python API from {api_path}. "
            "Check that the PowerFactory API and Python versions match."
        ) from error


def connect(project_name: str, *, show: bool = False) -> Any:
    """Open PowerFactory and activate the requested project.

    PowerFactory returns zero from ``ActivateProject`` on success.
    """
    application = load_powerfactory().GetApplicationExt()
    if application is None:
        raise RuntimeError("PowerFactory did not return an Application instance.")

    if application.ActivateProject(project_name):
        raise RuntimeError(f"Could not activate PowerFactory project: {project_name}")

    if show:
        application.Show()

    return application
