"""
PowerFactory bus naming utilities.

This module provides functions to generate consistent bus names
from PowerFactory terminal objects.
"""


def get_bus_full_name(terminal) -> str:
    """
    Get the full bus name including substation prefix.
    
    Format: "SubstationName_BusName" or just "BusName" if no substation.
    
    Args:
        terminal: PowerFactory terminal object (ElmTerm)
        
    Returns:
        Full bus name with substation prefix
    """
    try:
        # Get the parent folder which is typically the substation (ElmSubstat)
        parent = terminal.GetParent()
        if parent is not None and hasattr(parent, 'loc_name'):
            # Check if parent is a substation or site
            class_name = parent.GetClassName() if hasattr(parent, 'GetClassName') else ""
            if class_name in ('ElmSubstat', 'ElmSite', 'ElmTrfstat'):
                return f"{parent.loc_name}_{terminal.loc_name}"
        # Fallback to just the terminal name
        return terminal.loc_name
    except Exception:
        return terminal.loc_name
