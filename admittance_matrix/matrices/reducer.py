"""
Kron reduction and matrix reduction utilities.

This module provides functions for reducing Y-matrices to specific buses,
including generator internal bus reduction for stability analysis.
"""

import logging

import numpy as np
from ..core.elements import ShuntElement, GeneratorShunt, VoltageSourceShunt, ExternalGridShunt

logger = logging.getLogger(__name__)


def perform_kron_reduction(
    Y: np.ndarray, 
    indices_to_keep: list[int],
) -> np.ndarray:
    """
    Apply Kron reduction to eliminate buses and retain only specified indices.
    
    Uses the formula: Y_red = Y_AA - Y_AB @ inv(Y_BB) @ Y_BA
    Where A = indices to keep, B = indices to eliminate.
    
    Args:
        Y: Full admittance matrix
        indices_to_keep: List of integer indices to retain
        
    Returns:
        Reduced Y-matrix at specified indices
    """
    n = Y.shape[0]
    all_indices = set(range(n))
    indices_to_eliminate = sorted(all_indices - set(indices_to_keep))

    if not indices_to_eliminate:
        # Nothing to eliminate, just return submatrix
        return Y[np.ix_(indices_to_keep, indices_to_keep)]
    
    # Extract submatrices
    Y_AA = Y[np.ix_(indices_to_keep, indices_to_keep)]
    Y_AB = Y[np.ix_(indices_to_keep, indices_to_eliminate)]
    Y_BA = Y[np.ix_(indices_to_eliminate, indices_to_keep)]
    Y_BB = Y[np.ix_(indices_to_eliminate, indices_to_eliminate)]

    # Apply Kron reduction: Y_red = Y_AA - Y_AB @ inv(Y_BB) @ Y_BA
    Y_reduced = Y_AA - Y_AB @ np.linalg.inv(Y_BB) @ Y_BA
    
    return Y_reduced

def extend_matrix_to_generator_internal_nodes(
    Y_bus: np.ndarray,                                          # Stability Y-matrix (generator and load admittances included)
    bus_idx: dict[str, int],                                    # Bus name to index mapping
    sources: list[ShuntElement],                                 # List of shunt elements (to extract generators)
    base_mva: float = 100.0,
):
    # =============== Obtain sources data required for extended matrix (bus indices and admittances) ================
    # Combine all sources
    all_sources: list[ShuntElement] = []
    source_names: list[str] = []
    source_types: list[str] = []
    
    for source in sources:
        all_sources.append(source)
        source_names.append(source.name)
        if isinstance(source, GeneratorShunt):
            source_types.append('generator')
        elif isinstance(source, VoltageSourceShunt):
            source_types.append('voltage_source')
        elif isinstance(source, ExternalGridShunt):
            source_types.append('external_grid')
        else:
            source_types.append('unknown')

    n_sources = len(all_sources)
    n_bus = len(bus_idx)
    
    # Get source data
    source_bus_indices = [bus_idx[s.bus_name] for s in all_sources]
    source_admittances = np.array([s.get_admittance_pu(base_mva) for s in all_sources], dtype=complex)

    # =============== Now build extended Y-matrix that includes internal generator nodes ================
    '''
    Y_extended = | K   L |
                 | L^T M |
    K is a submatrix includes connection to the internal nodes of sources
    M is the original Y_bus including source admittances
    L is the connection between internal nodes and network buses

    Y_extended = | Y_gen   -Y_gen  |
                 | -Y_gen   Y_stab'|
    ''' 
    # Define submatrices
    M = Y_bus.copy()
    K = np.diag(source_admittances)
    L = np.zeros((n_sources, n_bus), dtype=complex)
    for i, bus_i in enumerate(source_bus_indices):
        L[i, bus_i] = -source_admittances[i]

    # Assemble extended matrix from submatrices
    Y_extended = np.block([
        [K,     L],
        [L.T,   M]
    ])

    return Y_extended
