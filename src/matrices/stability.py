"""Physical-bus, extended, and reduced classical stability matrices."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..network.elements import (
    ExternalGridShunt,
    GeneratorShunt,
    PVSystemShunt,
    SourceShunt,
    StaticGeneratorShunt,
    VoltageSourceShunt,
)
from .passive import build_load_flow_y_matrix
from .reducer import extend_matrix_to_generator_internal_nodes, perform_kron_reduction

if TYPE_CHECKING:
    from ..network import Network
    from ..network.operating_point import OperatingPoint


def build_stability_bus_y_matrix(
    network: Network,
    operating_point: OperatingPoint,
    *,
    excluded_sources: Collection[str] | None = None,
) -> npt.NDArray[np.complex128]:
    """Add source and PV impedances to the load-flow physical-bus matrix."""
    sources = _included_sources(network, operating_point, excluded_sources)
    return _stability_bus_y(network, operating_point, sources)


def build_extended_stability_y_matrix(
    network: Network,
    operating_point: OperatingPoint,
    *,
    excluded_sources: Collection[str] | None = None,
) -> npt.NDArray[np.complex128]:
    """Prepend source internal nodes to the physical-bus stability matrix.

    Excluded sources contribute neither an internal node nor a bus-diagonal
    admittance. Constant-impedance loads use solved bus voltages.
    """
    sources = _included_sources(network, operating_point, excluded_sources)
    return _extended_stability_y(network, operating_point, sources)


def build_stability_y_matrix(
    network: Network,
    operating_point: OperatingPoint,
    *,
    excluded_sources: Collection[str] | None = None,
) -> npt.NDArray[np.complex128]:
    """Kron-reduce physical buses, retaining selected source internal nodes."""
    sources = _included_sources(network, operating_point, excluded_sources)
    extended = _extended_stability_y(network, operating_point, sources)
    return perform_kron_reduction(extended, list(range(len(sources))))


def build_internal_voltage_vector(
    network: Network,
    operating_point: OperatingPoint,
    *,
    excluded_sources: Collection[str] | None = None,
) -> npt.NDArray[np.complex128]:
    """Return source internal voltages in reduced-matrix order."""
    included = _included_sources(network, operating_point, excluded_sources)
    names = {source.name for source in included}
    return np.asarray(
        [source.internal_voltage for source in operating_point.sources if source.name in names],
        dtype=np.complex128,
    )


def _stability_bus_y(
    network: Network,
    operating_point: OperatingPoint,
    sources: list[SourceShunt],
) -> npt.NDArray[np.complex128]:
    """Add only included active-source impedances and all PV impedances."""
    y = build_load_flow_y_matrix(network, operating_point)
    bus_idx = {name: index for index, name in enumerate(network.bus_names)}
    for source in sources:
        i = bus_idx[source.bus_name]
        y[i, i] += source.get_admittance_pu(network.base_mva)

    # PV impedances affect the bus matrix but have no internal source node.
    for shunt in network.shunts:
        if isinstance(shunt, PVSystemShunt):
            i = bus_idx[shunt.bus_name]
            y[i, i] += shunt.get_admittance_pu(network.base_mva)

    return y


def _extended_stability_y(
    network: Network,
    operating_point: OperatingPoint,
    sources: list[SourceShunt],
) -> npt.NDArray[np.complex128]:
    y_bus = _stability_bus_y(network, operating_point, sources)
    bus_idx = {name: index for index, name in enumerate(network.bus_names)}
    return extend_matrix_to_generator_internal_nodes(
        Y_bus=y_bus,
        bus_idx=bus_idx,
        sources=sources,
        base_mva=network.base_mva,
    )


def _included_sources(
    network: Network,
    operating_point: OperatingPoint,
    excluded_sources: Collection[str] | None,
) -> list[SourceShunt]:
    if not np.isclose(network.base_mva, operating_point.base_mva):
        raise ValueError("Operating-point and network base MVA must match.")
    sources = network.stability_sources

    if network.stability_source_names != operating_point.source_names:
        raise ValueError("Operating-point source order must match network source order.")
    expected_types = {
        GeneratorShunt: "generator",
        StaticGeneratorShunt: "static_generator",
        VoltageSourceShunt: "voltage_source",
        ExternalGridShunt: "external_grid",
    }
    for source, result in zip(sources, operating_point.sources):
        if result.source_type != expected_types[type(source)]:
            raise ValueError(f"Source type does not match network element: {source.name}")
    excluded = frozenset(excluded_sources or ())
    unknown = excluded.difference(operating_point.source_names)
    if unknown:
        raise ValueError(f"Unknown excluded sources: {sorted(unknown)}")
    included = [source for source in sources if source.name not in excluded]
    if not included:
        raise ValueError("At least one source is required for stability reduction.")
    return included
