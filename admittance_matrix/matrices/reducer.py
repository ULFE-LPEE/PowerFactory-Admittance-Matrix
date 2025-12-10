"""
Kron reduction and matrix reduction utilities.

This module provides functions for reducing Y-matrices to specific buses,
including generator internal bus reduction for stability analysis.
"""

import numpy as np
from ..core.elements import ShuntElement, GeneratorShunt


def kron_reduction(
    Y: np.ndarray, 
    bus_idx: dict[str, int], 
    buses_to_keep: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Apply Kron reduction to eliminate buses and retain only specified buses.
    
    Uses the formula: Y_red = Y_AA - Y_AB @ inv(Y_BB) @ Y_BA
    Where A = buses to keep, B = buses to eliminate.
    
    Args:
        Y: Full admittance matrix
        bus_idx: Mapping of bus names to indices in Y
        buses_to_keep: List of bus names to retain (e.g., generator buses)
        
    Returns:
        Tuple of (reduced Y-matrix, new bus index map for kept buses)
    """
    # Get indices for buses to keep (A) and eliminate (B)
    all_buses = set(bus_idx.keys())
    buses_to_eliminate = all_buses - set(buses_to_keep)
    
    idx_A = [bus_idx[bus] for bus in buses_to_keep]
    idx_B = [bus_idx[bus] for bus in buses_to_eliminate]
    
    # Extract submatrices
    Y_AA = Y[np.ix_(idx_A, idx_A)]
    Y_AB = Y[np.ix_(idx_A, idx_B)]
    Y_BA = Y[np.ix_(idx_B, idx_A)]
    Y_BB = Y[np.ix_(idx_B, idx_B)]
    
    # Apply Kron reduction: Y_red = Y_AA - Y_AB @ inv(Y_BB) @ Y_BA
    Y_BB_inv = np.linalg.inv(Y_BB)
    Y_reduced = Y_AA - Y_AB @ Y_BB_inv @ Y_BA
    
    # Create new bus index map for reduced matrix
    new_bus_idx = {bus: i for i, bus in enumerate(buses_to_keep)}
    
    return Y_reduced, new_bus_idx


def _get_generators(shunts: list[ShuntElement]) -> list:
    """Extract generator elements from shunts list."""
    return [s for s in shunts if type(s).__name__ == 'GeneratorShunt']


def reduce_to_generator_internal_buses(
    Y_stab: np.ndarray,
    bus_idx: dict[str, int],
    shunts: list[ShuntElement],
    base_mva: float = 100.0
) -> tuple[np.ndarray, list[str]]:
    """
    Reduce stability Y-matrix to generator internal buses only.
    
    Creates internal generator nodes (E' behind X''d) and applies Kron reduction
    to eliminate all network buses. This is the standard approach for 
    multi-machine transient stability analysis.
    
    The extended matrix structure before reduction:
        Y_ext = | Y_gen   -Y_gen  |
                | -Y_gen   Y_stab'|
    
    Where Y_stab' = Y_stab with generator admittances added to diagonals.
    
    Args:
        Y_stab: Stability Y-matrix (network + loads, built with MatrixType.STABILITY)
        bus_idx: Bus name to index mapping
        shunts: List of shunt elements (to extract generators)
        base_mva: System base power in MVA
        
    Returns:
        Tuple of (reduced Y-matrix, generator internal bus names)
    """
    generators = _get_generators(shunts)
    n_gen = len(generators)
    n_bus = len(bus_idx)
    
    if n_gen == 0:
        raise ValueError("No generators found in shunts list")
    
    # Get generator data
    gen_names = [g.name for g in generators]
    gen_bus_indices = [bus_idx[g.bus_name] for g in generators]
    gen_admittances = np.array([g.get_admittance_pu(base_mva) for g in generators], dtype=complex)
    
    # === Build extended Y-matrix ===
    # Copy Y_stab and add generator admittances to their bus diagonals
    Y_network = Y_stab.copy()
    for i, gen in enumerate(generators):
        bus_i = gen_bus_indices[i]
        Y_network[bus_i, bus_i] += gen_admittances[i]
    
    # Generator internal bus block (diagonal)
    Y_gen = np.diag(gen_admittances)
    
    # Connection block (internal nodes to network buses)
    Y_conn = np.zeros((n_gen, n_bus), dtype=complex)
    for i, bus_i in enumerate(gen_bus_indices):
        Y_conn[i, bus_i] = -gen_admittances[i]
    
    # Assemble extended matrix: [gen_internal | network_buses]
    Y_ext = np.block([
        [Y_gen,    Y_conn],
        [Y_conn.T, Y_network]
    ])
    
    # === Apply Kron reduction ===
    # Keep first n_gen rows/cols (generator internal buses), eliminate rest
    Y_AA = Y_ext[:n_gen, :n_gen]
    Y_AB = Y_ext[:n_gen, n_gen:]
    Y_BA = Y_ext[n_gen:, :n_gen]
    Y_BB = Y_ext[n_gen:, n_gen:]
    
    Y_reduced = Y_AA - Y_AB @ np.linalg.inv(Y_BB) @ Y_BA
    
    return Y_reduced, gen_names
