"""
Power system analysis functions.

This module provides functions for analyzing the reduced Y-matrix,
including power distribution ratio calculations.
"""

import numpy as np
from typing import Union
from ..adapters.powerfactory import GeneratorResult, VoltageSourceResult

# Type alias for source data
SourceData = Union[GeneratorResult, VoltageSourceResult]


def calculate_power_distribution_ratios(
    Y_reduced: np.ndarray,
    source_data: list[SourceData],
    disturbance_source_name: str
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Calculate power distribution ratios based on synchronizing power coefficients.
    
    When a source (generator/voltage source) trips, this calculates how the lost 
    power is distributed among the remaining sources based on their synchronizing 
    power coefficients.
    
    The synchronizing power coefficient between sources i and the disturbance
    source is:
        K_ij = E_i × E_dist × (B_ij × cos(δ_i - δ_dist) - G_ij × sin(δ_i - δ_dist))
    
    Args:
        Y_reduced: Reduced Y-matrix (source internal buses only)
        source_data: List of GeneratorResult or VoltageSourceResult objects
        disturbance_source_name: Name of the source that trips
        
    Returns:
        Tuple of (ratios array, source names in order, source types in order)
    """
    # Extract source names and types
    source_names = [s.name for s in source_data]
    source_types = [s.source_type for s in source_data]
    
    if disturbance_source_name not in source_names:
        raise ValueError(f"Source '{disturbance_source_name}' not found. Available: {source_names}")
    
    dist_idx = source_names.index(disturbance_source_name)
    
    # Build E magnitude and angle vectors
    E_abs = np.array([s.internal_voltage_mag for s in source_data]).reshape(-1, 1)
    E_angle = np.array([np.radians(s.internal_voltage_angle) for s in source_data]).reshape(-1, 1)
    
    # Extract B and G from reduced Y-matrix
    B_K = np.imag(Y_reduced)
    G_K = np.real(Y_reduced)
    
    n = len(source_data)
    
    # Calculate synchronizing power coefficients
    # K_ij = E_i * E_dist * (B_ij * cos(δ_i - δ_dist) - G_ij * sin(δ_i - δ_dist))
    angle_diff = E_angle - np.ones((n, 1)) * E_angle[dist_idx]

    K = (np.ones((n, n)) * E_abs[dist_idx] * E_abs * 
         (B_K * np.cos(angle_diff) - G_K * np.sin(angle_diff)))
    
    # Replace NaN values with zero
    K = np.nan_to_num(K, nan=0.0)
    
    # Select the column at disturbance bus
    K_col = K[:, dist_idx].copy()
    
    # Set disturbance source's contribution to zero
    K_col[dist_idx] = 0
    
    # Calculate power distribution ratios
    total_K = np.sum(K_col)
    if total_K != 0:
        ratios = K_col / total_K
    else:
        ratios = np.zeros_like(K_col)
    
    return ratios, source_names, source_types
