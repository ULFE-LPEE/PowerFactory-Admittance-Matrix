"""
Admittance matrix construction.

This module provides functions for building Y-matrices from network elements.
"""

import numpy as np
from enum import Enum

from ..core.elements import BranchElement, ShuntElement, GeneratorShunt, LoadShunt


class MatrixType(Enum):
    """Type of admittance matrix to build."""
    LOAD_FLOW = "load_flow"           # Network only (no shunt admittances)
    STABILITY = "stability"            # Network + loads (for Kron reduction)
    STABILITY_FULL = "stability_full"  # Network + loads + generators on diagonal


def get_unique_buses(
    branches: list[BranchElement], 
    shunts: list[ShuntElement]
) -> list[str]:
    """Extract unique bus names from branches and shunts."""
    buses = set()
    
    for b in branches:
        buses.add(b.from_bus_name)
        buses.add(b.to_bus_name)
    
    for s in shunts:
        buses.add(s.bus_name)
    
    return sorted(list(buses))


def build_admittance_matrix(
    branches: list[BranchElement],
    shunts: list[ShuntElement],
    bus_names: list[str],
    matrix_type: MatrixType = MatrixType.LOAD_FLOW,
    base_mva: float = 100.0
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build the admittance (Y) matrix from branch and shunt elements.
    
    Args:
        branches: List of branch elements (lines, switches)
        shunts: List of shunt elements (generators, loads)
        bus_names: List of unique bus names
        matrix_type: Type of matrix to build
        base_mva: System base power in MVA (default 100 MVA)
        
    Returns:
        Tuple of (Y_matrix, bus_index_map)
    """
    n = len(bus_names)
    bus_idx = {name: i for i, name in enumerate(bus_names)}
    
    # Initialize Y-matrix as complex
    Y = np.zeros((n, n), dtype=complex)
    
    # Process branch elements
    for branch in branches:
        i = bus_idx[branch.from_bus_name]
        j = bus_idx[branch.to_bus_name]
        
        # Get Y-matrix entries from branch (in per-unit on system base)
        Yii, Yjj, Yij, Yji = branch.get_y_matrix_entries(base_mva)
        
        # Add to matrix
        Y[i, i] += Yii
        Y[j, j] += Yjj
        Y[i, j] += Yij
        Y[j, i] += Yji
    
    # Process shunt elements (add to diagonal)
    if matrix_type == MatrixType.STABILITY:
        # Add only loads (generators will be added as internal buses separately)
        for shunt in shunts:
            if type(shunt).__name__ == 'LoadShunt':
                i = bus_idx[shunt.bus_name]
                Y[i, i] += shunt.get_admittance_pu(base_mva)
    
    elif matrix_type == MatrixType.STABILITY_FULL:
        # Add all shunts (loads + generators) directly to diagonal
        for shunt in shunts:
            i = bus_idx[shunt.bus_name]
            Y[i, i] += shunt.get_admittance_pu(base_mva)
    
    return Y, bus_idx


def get_generator_buses(shunts: list[ShuntElement]) -> list[str]:
    """Extract bus names where generators are connected."""
    return sorted(list(set(
        s.bus_name for s in shunts if type(s).__name__ == 'GeneratorShunt'
    )))
