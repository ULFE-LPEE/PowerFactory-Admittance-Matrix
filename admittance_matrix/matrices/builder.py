"""
Admittance matrix construction.

This module provides functions for building Y-matrices from network elements.
"""

import numpy as np
from enum import Enum

from ..core.elements import BranchElement, ShuntElement, Transformer3WBranch


class MatrixType(Enum):
    """Type of admittance matrix to build."""
    LOAD_FLOW = "load_flow"           # Network only (no shunt admittances)
    STABILITY = "stability"            # Network + loads + generators

def build_admittance_matrix(
    branches: list[BranchElement],
    shunts: list[ShuntElement],
    bus_names: list[str],
    matrix_type: MatrixType = MatrixType.LOAD_FLOW,
    base_mva: float = 100.0,
    transformers_3w: list[Transformer3WBranch] | None = None,
    exclude_source_name: str | None = None,  # Add this parameter
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build the admittance (Y) matrix from branch and shunt elements.
    
    Args:
        branches: List of branch elements (lines, switches, 2W transformers)
        shunts: List of shunt elements (generators, loads)
        bus_names: List of unique bus names (including virtual star nodes for 3W trafos)
        matrix_type: Type of matrix to build
        base_mva: System base power in MVA (default 100 MVA)
        transformers_3w: List of 3-winding transformers (optional)
        exclude_source_name: Name of source to exclude from admittance matrix (optional)
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
    
    # Process 3-winding transformers
    # Use the local 3x3 admittance matrix directly (star model with Kron reduction)
    if transformers_3w:
        for t3w in transformers_3w:
            # Get the complete 3x3 local admittance matrix
            local_matrix, local_bus_names = t3w.get_local_admittance_matrix()
            
            # Map local bus indices to global indices
            local_to_global = [bus_idx[name] for name in local_bus_names]
            
            # Add local matrix entries to global Y matrix
            for local_i in range(3):
                global_i = local_to_global[local_i]
                for local_j in range(3):
                    global_j = local_to_global[local_j]
                    Y[global_i, global_j] += local_matrix[local_i][local_j]

    # Add shunt filters (passive network elements - always included in Y-matrix)
    for shunt in shunts:
        if type(shunt).__name__ == 'ShuntFilterShunt':
            i = bus_idx[shunt.bus_name]
            Y[i, i] += shunt.get_admittance_pu(base_mva)
    
    # Process other shunt elements based on matrix type
    if matrix_type == MatrixType.STABILITY:
        # Add loads (generators/sources will be added as internal buses separately)
        for shunt in shunts:
            if exclude_source_name is not None and shunt.name == exclude_source_name:
                print("Excluding source admittance for:", exclude_source_name) #! Dev
                continue  # Exclude specified source admittance
            i = bus_idx[shunt.bus_name]
            Y[i, i] += shunt.get_admittance_pu(base_mva)

    return Y, bus_idx