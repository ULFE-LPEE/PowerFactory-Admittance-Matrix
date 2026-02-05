"""
PowerFactory bus naming utilities.

This module provides functions to generate consistent bus names
from PowerFactory terminal objects.
"""

def get_bus_full_name(terminal) -> str:
    """
    Get the full bus name including substation and bay context when available.

    Formats:
        - "SubstationName_BusName" when the terminal is not in a bay.
        - "Sub_SubstationName_Bay_BayName_Term_TerminalName" when the terminal
          is within a bay (ElmBay).
        - "BusName" when no substation can be resolved.

    Args:
        terminal: PowerFactory terminal object (ElmTerm).

    Returns:
        Full bus name with substation and bay prefixes where applicable.
    """
    try:
        # Get substation name
        substatName = terminal.GetAttribute("cpSubstat").loc_name
        parentClassName = terminal.GetParent().GetClassName()

        # If terminal is in a bay, include bay name
        if (parentClassName == 'ElmBay'):
            return f"Sub_{substatName}_Bay_{ terminal.GetParent().loc_name}_Term_{terminal.loc_name}"
        # Else just substation + terminal
        return f"{substatName}_{terminal.loc_name}"
    
    except Exception:
        # If any issue occurs (e.g., no substation), return just terminal name
        return terminal.loc_name
