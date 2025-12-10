"""
Power system analysis functions.

This module provides functions for analyzing the reduced Y-matrix,
including power distribution ratio calculations.
"""

import numpy as np


def calculate_power_distribution_ratios(
    Y_reduced: np.ndarray,
    gen_data: list,
    disturbance_gen_name: str
) -> tuple[np.ndarray, list[str]]:
    """
    Calculate power distribution ratios based on synchronizing power coefficients.
    
    When a generator trips, this calculates how the lost power is distributed
    among the remaining generators based on their synchronizing power coefficients.
    
    The synchronizing power coefficient between generators i and the disturbance
    generator is:
        K_ij = E_i × E_dist × (B_ij × cos(δ_i - δ_dist) - G_ij × sin(δ_i - δ_dist))
    
    Args:
        Y_reduced: Reduced Y-matrix (generator internal buses only)
        gen_data: List of GeneratorResult objects from get_generator_data()
        disturbance_gen_name: Name of the generator that trips
        
    Returns:
        Tuple of (ratios array, generator names in order)
    """
    # Extract generator names and find disturbance index
    gen_names = [g.name for g in gen_data]
    
    if disturbance_gen_name not in gen_names:
        raise ValueError(f"Generator '{disturbance_gen_name}' not found. Available: {gen_names}")
    
    dist_idx = gen_names.index(disturbance_gen_name)
    
    # Build E magnitude and angle vectors
    E_abs = np.array([g.internal_voltage_mag for g in gen_data]).reshape(-1, 1)
    E_angle = np.array([np.radians(g.internal_voltage_angle) for g in gen_data]).reshape(-1, 1)
    
    # Extract B and G from reduced Y-matrix
    B_K = np.imag(Y_reduced)
    G_K = np.real(Y_reduced)
    
    n = len(gen_data)
    
    # Calculate synchronizing power coefficients
    # K_ij = E_i * E_dist * (B_ij * cos(δ_i - δ_dist) - G_ij * sin(δ_i - δ_dist))
    angle_diff = E_angle - np.ones((n, 1)) * E_angle[dist_idx]
    
    K = (np.ones((n, n)) * E_abs[dist_idx] * E_abs * 
         (B_K * np.cos(angle_diff) - G_K * np.sin(angle_diff)))
    
    # Replace NaN values with zero
    K = np.nan_to_num(K, nan=0.0)
    
    # Select the column at disturbance bus
    K_col = K[:, dist_idx].copy()
    
    # Set disturbance generator's contribution to zero
    K_col[dist_idx] = 0
    
    # Calculate power distribution ratios
    total_K = np.sum(K_col)
    if total_K != 0:
        ratios = K_col / total_K
    else:
        ratios = np.zeros_like(K_col)
    
    return ratios, gen_names
