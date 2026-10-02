"""Stable names for PowerFactory terminals."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import powerfactory as pf


def get_bus_full_name(terminal: pf.DataObject) -> str:
    """Include substation and bay context when PowerFactory provides it."""
    terminal_name = terminal.GetAttribute("loc_name")
    try:
        substation = terminal.GetAttribute("cpSubstat")
        if substation is None:
            return terminal_name

        substation_name = substation.GetAttribute("loc_name")
        parent = terminal.GetParent()
        if parent.GetClassName() == "ElmBay":
            bay_name = parent.GetAttribute("loc_name")
            return f"Sub_{substation_name}_Bay_{bay_name}_Term_{terminal_name}"
        return f"{substation_name}_{terminal_name}"
    except Exception:
        return terminal_name
