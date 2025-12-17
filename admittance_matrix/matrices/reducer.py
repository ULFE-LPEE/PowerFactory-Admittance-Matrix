"""
Kron reduction and matrix reduction utilities.

This module provides functions for reducing Y-matrices to specific buses,
including generator internal bus reduction for stability analysis.
"""

import logging

import numpy as np
from ..core.elements import ShuntElement, GeneratorShunt, VoltageSourceShunt, ExternalGridShunt

logger = logging.getLogger(__name__)


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
    return [s for s in shunts if isinstance(s, GeneratorShunt)]


def _get_voltage_sources(shunts: list[ShuntElement]) -> list:
    """Extract voltage source elements from shunts list."""
    return [s for s in shunts if isinstance(s, VoltageSourceShunt)]


def _get_external_grids(shunts: list[ShuntElement]) -> list:
    """Extract external grid elements from shunts list."""
    return [s for s in shunts if isinstance(s, ExternalGridShunt)]


def reduce_to_generator_internal_buses(
    Y_stab: np.ndarray,
    bus_idx: dict[str, int],
    shunts: list[ShuntElement],
    base_mva: float = 100.0,
    include_voltage_sources: bool = True,
    include_external_grids: bool = False
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Reduce stability Y-matrix to generator internal buses only.
    
    Creates internal generator nodes (E' behind X''d) and applies Kron reduction
    to eliminate all network buses. This is the standard approach for 
    multi-machine transient stability analysis.
    
    Optionally includes voltage sources and external grids as additional 
    "source" elements that participate in power redistribution.
    
    The extended matrix structure before reduction:
        Y_ext = | Y_sources   -Y_sources  |
                | -Y_sources   Y_stab'    |
    
    Where Y_stab' = Y_stab with source admittances added to diagonals.
    
    Args:
        Y_stab: Stability Y-matrix (network + loads, built with MatrixType.STABILITY)
        bus_idx: Bus name to index mapping
        shunts: List of shunt elements (to extract generators and sources)
        base_mva: System base power in MVA
        include_voltage_sources: If True, include AC voltage sources in reduction
        include_external_grids: If True, include external grids in reduction
        
    Returns:
        Tuple of:
        - Y_reduced: Reduced admittance matrix at source internal buses
        - source_names: List of source names (generators, voltage sources, etc.)
        - source_types: List of source types ('generator', 'voltage_source', 'external_grid')
    """
    generators = _get_generators(shunts)
    voltage_sources = _get_voltage_sources(shunts) if include_voltage_sources else []
    external_grids = _get_external_grids(shunts) if include_external_grids else []
    
    # Combine all sources
    all_sources = []
    source_names = []
    source_types = []
    
    for gen in generators:
        all_sources.append(gen)
        source_names.append(gen.name)
        source_types.append('generator')
    
    for vs in voltage_sources:
        all_sources.append(vs)
        source_names.append(vs.name)
        source_types.append('voltage_source')
    
    for xg in external_grids:
        all_sources.append(xg)
        source_names.append(xg.name)
        source_types.append('external_grid')
    
    n_sources = len(all_sources)
    n_bus = len(bus_idx)
    
    if n_sources == 0:
        raise ValueError("No generators or sources found in shunts list")
    
    # Get source data
    source_bus_indices = [bus_idx[s.bus_name] for s in all_sources]
    source_admittances = np.array([s.get_admittance_pu(base_mva) for s in all_sources], dtype=complex)

    logger.info(f"Source reduction: {len(generators)} generators, {len(voltage_sources)} voltage sources, "
                f"{len(external_grids)} external grids = {n_sources} total")
    
    # === Build extended Y-matrix ===
    # Copy Y_stab and add source admittances to their bus diagonals
    Y_network = Y_stab.copy()
    for i, source in enumerate(all_sources):
        bus_i = source_bus_indices[i]
        Y_network[bus_i, bus_i] += source_admittances[i]
    
    # Source internal bus block (diagonal)
    Y_source_diag = np.diag(source_admittances)
    
    # Connection block (internal nodes to network buses)
    Y_conn = np.zeros((n_sources, n_bus), dtype=complex)
    for i, bus_i in enumerate(source_bus_indices):
        Y_conn[i, bus_i] = -source_admittances[i]
    
    # Assemble extended matrix: [source_internal | network_buses]
    Y_ext = np.block([
        [Y_source_diag, Y_conn],
        [Y_conn.T,      Y_network]
    ])
    
    # === Apply Kron reduction ===
    # Keep first n_sources rows/cols (source internal buses), eliminate rest
    Y_AA = Y_ext[:n_sources, :n_sources]
    Y_AB = Y_ext[:n_sources, n_sources:]
    Y_BA = Y_ext[n_sources:, :n_sources]
    Y_BB = Y_ext[n_sources:, n_sources:]
    
    Y_reduced = Y_AA - Y_AB @ np.linalg.inv(Y_BB) @ Y_BA
    
    return Y_reduced, source_names, source_types
