"""Directional line-power changes from prefault and postfault network matrices.

Uses the same fixed internal-source-voltage approximation as the reduced
matrix outage formulation. The physical bus voltages satisfy
``Y_bb V_b = -Y_bs E_s`` in each network state. For a branch from bus i to j,
``I_ij = Y_ii V_i + Y_ij V_j`` and ``P_ij = Re(V_i conj(I_ij)) S_base``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..matrices.reducer import solve_admittance_system

if TYPE_CHECKING:
    from ..network import Network, OperatingPoint
    from ..network.elements import Branch


@dataclass(frozen=True, slots=True)
class BranchActivePowerChange:
    """Modeled MW flow at the stored ``from_bus_name`` branch terminal."""

    branch_name: str
    from_bus_name: str
    to_bus_name: str
    prefault_mw: float
    postfault_mw: float

    @property
    def change_mw(self) -> float:
        """Return postfault minus prefault from-terminal active power."""
        return self.postfault_mw - self.prefault_mw


def calculate_branch_power_changes(
    network: Network,
    operating_point: OperatingPoint,
    *,
    outaged_source: str,
    branch_names: Sequence[str],
) -> dict[str, BranchActivePowerChange]:
    """Calculate named branch responses to opening a source internal port.

    Source internal voltage phasors stay fixed at their prefault values.
    Constant-impedance load admittances are updated from the solved operating
    point in both states. This is the instantaneous scalar network model, not
    an RMS trajectory or a settled postfault load flow.
    """
    source_names = network.stability_source_names
    if outaged_source not in source_names:
        raise ValueError(f"Unknown outaged source: {outaged_source}")
    requested = tuple(branch_names)
    if len(requested) != len(set(requested)):
        raise ValueError("Requested branch names must be unique.")
    selected: dict[str, Branch] = {}
    for branch in network.branches:
        if branch.name in requested:
            if branch.name in selected:
                raise ValueError(f"More than one network branch is named {branch.name!r}.")
            selected[branch.name] = branch
    missing = set(requested) - selected.keys()
    if missing:
        raise ValueError(f"Network branches not found: {sorted(missing)}")
    if not requested:
        return {}

    prefault_e = network.get_internal_voltage_vector(operating_point)
    prefault_y = network.get_extended_stability_y_matrix(operating_point)
    prefault_v = _physical_bus_voltages(prefault_y, prefault_e, len(network.buses))

    surviving_indices = [index for index, name in enumerate(source_names) if name != outaged_source]
    postfault_y = network.get_extended_stability_y_matrix(operating_point, excluded_sources={outaged_source})
    postfault_v = _physical_bus_voltages(postfault_y, prefault_e[surviving_indices], len(network.buses))
    bus_indices = {name: index for index, name in enumerate(network.bus_names)}
    return {
        name: BranchActivePowerChange(
            branch_name=name,
            from_bus_name=selected[name].from_bus_name,
            to_bus_name=selected[name].to_bus_name,
            prefault_mw=_from_terminal_power_mw(selected[name], prefault_v, bus_indices, network.base_mva),
            postfault_mw=_from_terminal_power_mw(selected[name], postfault_v, bus_indices, network.base_mva),
        )
        for name in requested
    }


def calculate_all_branch_power_changes(
    network: Network,
    operating_point: OperatingPoint,
    *,
    branch_names: Sequence[str],
) -> dict[str, dict[str, BranchActivePowerChange]]:
    """Calculate selected branch flows for every generator outage in one solve.

    Let A be the prefault physical-bus admittance matrix, V its solved bus
    voltages, y_g the outaged source admittance, E_g its internal voltage,
    and e_k the unit vector of its terminal bus. Opening source g removes
    y_g e_k e_k.T from A and y_g E_g e_k from the current injection. With
    u_k = A^-1 e_k, the exact postfault voltage for this same fixed-E model is

        V_g = V + y_g (V_k - E_g) u_k / (1 - y_g u_k[k]).

    All u_k and V are obtained with one shared matrix factorization. Branch
    powers use the same from-terminal calculation as the single-outage path.
    """
    generators = network.generators
    if not generators:
        raise ValueError("At least one generator is required for outage analysis.")

    requested = tuple(branch_names)
    if len(requested) != len(set(requested)):
        raise ValueError("Requested branch names must be unique.")
    selected: dict[str, Branch] = {}
    for branch in network.branches:
        if branch.name in requested:
            if branch.name in selected:
                raise ValueError(f"More than one network branch is named {branch.name!r}.")
            selected[branch.name] = branch
    missing = set(requested) - selected.keys()
    if missing:
        raise ValueError(f"Network branches not found: {sorted(missing)}")
    if not requested:
        return {generator.name: {} for generator in generators}

    source_names = network.stability_source_names
    if source_names != operating_point.source_names:
        raise ValueError("Operating-point source order must match network source order.")
    e = network.get_internal_voltage_vector(operating_point)
    extended_y = network.get_extended_stability_y_matrix(operating_point)
    source_count = len(source_names)
    bus_count = len(network.buses)
    bus_y = extended_y[source_count:, source_count:]
    source_to_bus_y = extended_y[source_count:, :source_count]
    bus_indices = {name: index for index, name in enumerate(network.bus_names)}
    source_indices = {name: index for index, name in enumerate(source_names)}
    generator_bus_indices = {generator.name: bus_indices[generator.bus_name] for generator in generators}
    unique_bus_indices = tuple(dict.fromkeys(generator_bus_indices.values()))

    # The first right-hand side gives prefault V. The others give A^-1 e_k
    # for every distinct generator terminal bus.
    rhs = np.zeros((bus_count, 1 + len(unique_bus_indices)), dtype=np.complex128)
    rhs[:, 0] = -(source_to_bus_y @ e)
    for column, bus_index in enumerate(unique_bus_indices, start=1):
        rhs[bus_index, column] = 1.0
    try:
        solved = solve_admittance_system(bus_y, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("Physical-bus admittance block is singular.") from error
    prefault_v = solved[:, 0]
    sensitivities = {bus_index: solved[:, column] for column, bus_index in enumerate(unique_bus_indices, start=1)}
    prefault_p = {
        name: _from_terminal_power_mw(selected[name], prefault_v, bus_indices, network.base_mva) for name in requested
    }

    results = {}
    for generator in generators:
        g = source_indices[generator.name]
        k = generator_bus_indices[generator.name]
        y_g = extended_y[g, g]
        u_k = sensitivities[k]
        denominator = 1 - y_g * u_k[k]
        if np.isclose(denominator, 0.0):
            raise ValueError(f"Postfault bus matrix is singular for {generator.name}.")
        postfault_v = prefault_v + y_g * (prefault_v[k] - e[g]) * u_k / denominator
        results[generator.name] = {
            name: BranchActivePowerChange(
                branch_name=name,
                from_bus_name=selected[name].from_bus_name,
                to_bus_name=selected[name].to_bus_name,
                prefault_mw=prefault_p[name],
                postfault_mw=_from_terminal_power_mw(selected[name], postfault_v, bus_indices, network.base_mva),
            )
            for name in requested
        }
    return results


def _physical_bus_voltages(
    extended_y: npt.NDArray[np.complex128],
    internal_e: npt.NDArray[np.complex128],
    bus_count: int,
) -> npt.NDArray[np.complex128]:
    """Back-substitute buses from a source-first extended admittance matrix."""
    source_count = len(internal_e)
    if extended_y.shape != (source_count + bus_count,) * 2:
        raise ValueError("Extended matrix size does not match sources and buses.")
    bus_y = extended_y[source_count:, source_count:]
    source_to_bus_y = extended_y[source_count:, :source_count]
    try:
        return solve_admittance_system(bus_y, -(source_to_bus_y @ internal_e))
    except np.linalg.LinAlgError as error:
        raise ValueError("Physical-bus admittance block is singular.") from error


def _from_terminal_power_mw(
    branch: Branch,
    bus_v: npt.NDArray[np.complex128],
    bus_indices: dict[str, int],
    base_mva: float,
) -> float:
    from_v = bus_v[bus_indices[branch.from_bus_name]]
    to_v = bus_v[bus_indices[branch.to_bus_name]]
    y_from_from, _, y_from_to, _ = branch.get_y_matrix_entries(base_mva)
    current = y_from_from * from_v + y_from_to * to_v
    return float((from_v * current.conjugate()).real * base_mva)
