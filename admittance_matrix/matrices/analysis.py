"""
Power system analysis functions.

This module provides functions for analyzing the reduced Y-matrix,
including power distribution ratio calculations.
"""

import numpy as np
from ..adapters.powerfactory import GeneratorResult, VoltageSourceResult, ExternalGridResult

def calculate_power_distribution_ratios(
    Y_reduced: np.ndarray,
    source_data: list[GeneratorResult | VoltageSourceResult | ExternalGridResult],
    disturbance_source_name: str,
    dist_angle_mode: str = "terminal_current",   # "internal_E" | "terminal_current" | "terminal_voltage"
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Calculate power distribution ratios based on synchronizing power coefficients.

    When a source (generator/voltage source) trips, this calculates how the lost
    power is distributed among the remaining sources based on their synchronizing
    power coefficients.

    We only need the disturbance column K_{:,d}:
        K_{i,d} = E_i * E_d * ( B_{i,d} * cos(δ_i - δ_d) - G_{i,d} * sin(δ_i - δ_d) )

    Args:
        Y_reduced: Reduced Y-matrix (source internal buses only), shape (n, n)
        source_data: List of GeneratorResult / VoltageSourceResult / ExternalGridResult objects
        disturbance_source_name: Name of the source that trips
        dist_angle_mode:
            - "internal_E": use internal EMF angle from source_data[dist].internal_voltage_angle
            - "terminal_current": δ_d := angle(I) = angle(V) - angle(S) with S=P+jQ
            - "terminal_voltage": δ_d := angle(V)

    Returns:
        (ratios, source_names_in_order, source_types_in_order)
        - ratios is shape (n,)
    """
    # Extract source names and types
    source_names = [s.name for s in source_data]
    source_types = [s.source_type for s in source_data]

    if disturbance_source_name not in source_names:
        raise ValueError(f"Source '{disturbance_source_name}' not found. Available: {source_names}")

    dist_idx = source_names.index(disturbance_source_name)

    n = len(source_data)
    if Y_reduced.shape != (n, n):
        raise ValueError(f"Y_reduced shape {Y_reduced.shape} does not match number of sources {n}.")

    # Build E magnitude and angle vectors as 1-D arrays (shape (n,))
    E_abs = np.array([np.abs(s.internal_voltage) for s in source_data], dtype=float)
    E_angle = np.array([np.angle(s.internal_voltage) for s in source_data], dtype=float)

    # Extract B and G from reduced Y-matrix
    B_K = np.imag(Y_reduced)
    G_K = np.real(Y_reduced)

    # --- Choose disturbance angle δ_d ---
    if dist_angle_mode == "internal_E":
        deltad = float(E_angle[dist_idx])
        print(
            f"dist_angle_mode=internal_E: δd = internal voltage angle = {np.degrees(deltad):.2f} deg")  #! Dev

    elif dist_angle_mode == "terminal_current":
        # δ_d := angle(I) = angle(V) - angle(S), with S = P + jQ # The same as terminal current angle prefault
        # NOTE: requires terminal voltage phasor and P/Q in pu
        P_PU = np.array([s.p_pu for s in source_data], dtype=float)
        Q_PU = np.array([s.q_pu for s in source_data], dtype=float)

        pf_angle = float(np.arctan2(Q_PU[dist_idx], P_PU[dist_idx]))  # angle(S) also the current angle.
        terminal_voltage_angle = float(np.angle(source_data[dist_idx].terminal_voltage))
        deltad = terminal_voltage_angle - pf_angle
        E_angle[dist_idx] = deltad

        print(
            f"dist_angle_mode=terminal_current: "
            f"δd = angle(V) - angle(S) = {np.degrees(terminal_voltage_angle):.2f} "
            f"- {np.degrees(pf_angle):.2f} = {np.degrees(deltad):.2f} deg"
        )  #! Dev

    elif dist_angle_mode == "terminal_voltage":
        # δ_d := angle(V)
        deltad = float(np.angle(source_data[dist_idx].terminal_voltage))
        print(
            f"dist_angle_mode=terminal_voltage: δd = angle(V) = {np.degrees(deltad):.2f} deg"
        )  #! Dev

    else:
        raise ValueError(
            f"Unknown dist_angle_mode='{dist_angle_mode}'. "
            "Use 'internal_E', 'terminal_current', or 'terminal_voltage'."
        )

    # Scalars for disturbance source
    Ed = float(E_abs[dist_idx])

    # Disturbance column of Y parts
    Bid = B_K[:, dist_idx]   # (n,)
    Gid = G_K[:, dist_idx]   # (n,)

    # Angle diffs δ_i - δ_d
    d = E_angle - deltad     # (n,)
    # print(f"Angle diffs δ_i - δ_d (deg) = {np.degrees(d)}")  #! Dev

    # K_{i,d}
    K_col = E_abs * Ed * (Bid * np.cos(d) - Gid * np.sin(d))
    # print(f"K_col = {K_col}")  #! Dev

    # Replace NaN values with zero
    K_col = np.nan_to_num(K_col, nan=0.0)

    # Set disturbance source's contribution to zero
    K_col[dist_idx] = 0.0

    # Calculate power distribution ratios
    total_K = float(np.sum(K_col))
    print(f"Total K (excluding disturbance source) = {total_K}")  #! Dev

    if total_K != 0.0:
        ratios = K_col / total_K
    else:
        ratios = np.zeros_like(K_col)

    return ratios, source_names, source_types

def calculate_power_distribution_ratios_prefault_postfault(
        Yred0_3x3: np.ndarray,   
        Yred1_2x2: np.ndarray,
        E_abs: np.ndarray,
        E_angle: np.ndarray,
        dist_idx: int,
        keep_idx: list[int],
        sbase_mva: float = 100.0,
    ):
        """
        Compute t=0+ electrical redistribution shares among remaining generators,
        using:
        - prefault Yred0_3x3 for baseline P0
        - post-trip Yred1_2x2 for P1 of remaining machines

        No Kron reduction is done here. You provide Yred1_2x2.
        """
        # Internal EMFs as phasors (prefault)
        E0 = E_abs * np.exp(1j * E_angle)  # (3,)

        # --- Prefault internal currents and powers (all 3 machines)
        I0 = Yred0_3x3 @ E0
        S0 = E0 * np.conj(I0)
        P0 = np.real(S0)

        # --- Post-trip: remaining internal EMFs (assumed unchanged at t=0+)
        E1 = E0[keep_idx]
        I1 = Yred1_2x2 @ E1
        S1 = E1 * np.conj(I1)
        P1 = np.real(S1)

        # --- ΔP for remaining machines
        dP_keep = P1 - P0[keep_idx]

        # Shares among remaining machines
        total = float(np.sum(dP_keep))
        if abs(total) > 1e-12:
            ratios_keep = dP_keep / total
        else:
            ratios_keep = np.zeros_like(dP_keep)

        # Expand to length 3 with 0 for tripped machine
        ratios = np.zeros(len(E_abs), dtype=float)
        for k, idx in enumerate(keep_idx):
            ratios[idx] = float(ratios_keep[k])
        ratios[dist_idx] = 0.0

        debug = {
            "P0_pu": P0,
            "P0_MW": P0 * sbase_mva,
            "P1_keep_pu": P1,
            "P1_keep_MW": P1 * sbase_mva,
            "dP_keep_pu": dP_keep,
            "dP_keep_MW": dP_keep * sbase_mva,
            "ratios_keep": ratios_keep,
            "ratios_full": ratios,
            "sum_ratios_keep": float(np.sum(ratios_keep)),
            "sum_ratios_full": float(np.sum(ratios)),
        }

        return ratios, debug
