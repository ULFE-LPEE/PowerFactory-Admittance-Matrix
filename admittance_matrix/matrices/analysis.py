"""
Power system analysis functions.

This module provides functions for analyzing the reduced Y-matrix,
including power distribution ratio calculations.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..adapters.powerfactory import GeneratorResult, VoltageSourceResult, ExternalGridResult

SourceResult = GeneratorResult | VoltageSourceResult | ExternalGridResult


def calculate_power_distribution_ratios(
    Y_reduced: npt.NDArray[np.complex128],
    source_data: list[SourceResult],
    disturbance_source_name: str,
    dist_angle_mode: str = "terminal_current",   # "internal_E" | "terminal_current"
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """
    Calculate power distribution ratios based on synchronizing power coefficients.

    When a source trips, this calculates how the lost power is distributed among
    the remaining sources.
    """
    source_names = [s.name for s in source_data]
    if disturbance_source_name not in source_names:
        raise ValueError(f"Source '{disturbance_source_name}' not found. Available: {source_names}")

    dist_idx = source_names.index(disturbance_source_name)
    n = len(source_data)
    if Y_reduced.shape != (n, n):
        raise ValueError(f"Y_reduced shape {Y_reduced.shape} does not match number of sources {n}.")

    return calculate_power_distribution_ratios_from_reduced_column(
        Y_reduced[:, dist_idx],
        source_data,
        disturbance_source_name,
        dist_angle_mode=dist_angle_mode,
    )


def calculate_power_distribution_ratios_from_reduced_column(
    Y_reduced_column: npt.NDArray[np.complex128],
    source_data: list[SourceResult],
    disturbance_source_name: str,
    dist_angle_mode: str = "terminal_current",
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Calculate power distribution ratios from one reduced Y-matrix column."""
    source_names = [s.name for s in source_data]
    source_types = [s.source_type for s in source_data]

    if disturbance_source_name not in source_names:
        raise ValueError(f"Source '{disturbance_source_name}' not found. Available: {source_names}")

    dist_idx = source_names.index(disturbance_source_name)
    n = len(source_data)
    if Y_reduced_column.shape != (n,):
        raise ValueError(f"Y_reduced_column shape {Y_reduced_column.shape} does not match number of sources {n}.")

    E_abs = np.array([np.abs(s.internal_voltage) for s in source_data], dtype=float)
    E_angle = np.array([np.angle(s.internal_voltage) for s in source_data], dtype=float)

    B_col = np.imag(Y_reduced_column)
    G_col = np.real(Y_reduced_column)

    if dist_angle_mode == "internal_E":
        deltad = float(E_angle[dist_idx])
    elif dist_angle_mode == "terminal_current":
        p_pu = np.array([s.p_pu for s in source_data], dtype=float)
        q_pu = np.array([s.q_pu for s in source_data], dtype=float)

        pf_angle = float(np.arctan2(q_pu[dist_idx], p_pu[dist_idx]))
        terminal_voltage_angle = float(np.angle(source_data[dist_idx].terminal_voltage))
        deltad = terminal_voltage_angle - pf_angle
        E_angle[dist_idx] = deltad
    else:
        raise ValueError(
            f"Unknown dist_angle_mode='{dist_angle_mode}'. "
            "Use 'internal_E' or 'terminal_current'."
        )

    Ed = float(E_abs[dist_idx])
    angle_diff = E_angle - deltad
    k_col = E_abs * Ed * (B_col * np.cos(angle_diff) - G_col * np.sin(angle_diff))
    k_col = np.nan_to_num(k_col, nan=0.0)
    k_col[dist_idx] = 0.0

    total_k = float(np.sum(k_col))
    if total_k != 0.0:
        ratios = k_col / total_k
    else:
        ratios = np.zeros_like(k_col)

    return ratios, source_names, source_types


def calculate_power_distribution_ratios_prefault_postfault(
        Y_red_before: np.ndarray,
        Y_red_after: np.ndarray,
        E_abs: np.ndarray,
        E_angle: np.ndarray,
        dist_idx: int,
        keep_idx: list[int],
        sbase_mva: float = 100.0,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """
        Compute t=0+ electrical redistribution shares among remaining generators,
        using:
        - prefault  Y_red_before for baseline P0
        - post-trip Y_red_after for P1 of remaining machines

        No Kron reduction is done here. You provide Y_red_after.
        """
        # Internal EMFs as phasors (prefault)
        E0 = E_abs * np.exp(1j * E_angle)
        E1 = E0[keep_idx]
        I1 = Y_red_after @ E1

        return calculate_power_distribution_ratios_from_postfault_currents(
            Y_red_before=Y_red_before,
            I_after_keep=I1,
            E_abs=E_abs,
            E_angle=E_angle,
            dist_idx=dist_idx,
            keep_idx=keep_idx,
            sbase_mva=sbase_mva,
        )


def calculate_power_distribution_ratios_from_postfault_currents(
        Y_red_before: np.ndarray,
        I_after_keep: np.ndarray,
        E_abs: np.ndarray,
        E_angle: np.ndarray,
        dist_idx: int,
        keep_idx: list[int],
        sbase_mva: float = 100.0,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """
        Compute t=0+ redistribution shares when post-trip currents are already known.
        """
        # Internal EMFs as phasors (prefault)
        E0 = E_abs * np.exp(1j * E_angle)

        # --- Prefault internal currents and powers (all machines)
        I0 = Y_red_before @ E0
        S0 = E0 * np.conj(I0)
        P0 = np.real(S0)

        # --- Post-trip: remaining internal EMFs (assumed unchanged at t=0+)
        E1 = E0[keep_idx]
        if I_after_keep.shape != E1.shape:
            raise ValueError(
                f"I_after_keep shape {I_after_keep.shape} does not match remaining source count {E1.shape}."
            )

        I1 = I_after_keep
        S1 = E1 * np.conj(I1)
        P1 = np.real(S1)

        # --- Delta P for remaining machines
        dP_keep = P1 - P0[keep_idx]

        # Shares among remaining machines
        total = float(np.sum(dP_keep))
        if abs(total) > 1e-12:
            ratios_keep = dP_keep / total
        else:
            ratios_keep = np.zeros_like(dP_keep)

        # Expand to length of E_abs with 0 for tripped machine
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
