"""Electrical network assembled from library elements, without a data-source dependency."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..matrices import (
    build_extended_stability_y_matrix,
    build_internal_voltage_vector,
    build_load_flow_y_matrix,
    build_passive_y_matrix,
    build_stability_bus_y_matrix,
    build_stability_y_matrix,
)
from .elements import (
    Branch,
    Bus,
    ExternalGridShunt,
    GeneratorShunt,
    LineBranch,
    LoadShunt,
    Shunt,
    SourceShunt,
    Transformer3WBranch,
    VoltageSourceShunt,
)
from .operating_point import OperatingPoint


@dataclass(kw_only=True)
class Network:
    """A network model whose elements may come from PowerFactory or another source.

    Networks assembled with ``add_branch`` and related methods use bus
    registration order. Pre-populated element lists use sorted bus names.
    Three-winding transformer star nodes are represented internally by their
    element's local admittance matrix.
    """

    base_mva: float = 100.0
    branches: list[Branch] = field(default_factory=list)
    shunts: list[Shunt] = field(default_factory=list)
    transformers_3w: list[Transformer3WBranch] = field(default_factory=list)
    buses: dict[str, Bus] = field(default_factory=dict)
    bus_mapping: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.base_mva <= 0:
            raise ValueError("Network base MVA must be greater than zero.")
        for name in self._collect_bus_names():
            self.register_bus(name)

    @property
    def bus_names(self) -> tuple[str, ...]:
        """Return bus names in admittance-matrix order."""
        return tuple(self.buses)

    @property
    def branch_names(self) -> tuple[str, ...]:
        """Return two-terminal equipment names in network order."""
        return tuple(branch.name for branch in self.branches)

    @property
    def lines(self) -> tuple[LineBranch, ...]:
        """Return transmission lines, excluding switches and transformers."""
        return tuple(branch for branch in self.branches if isinstance(branch, LineBranch))

    @property
    def line_names(self) -> tuple[str, ...]:
        """Return line names in network order."""
        return tuple(line.name for line in self.lines)

    @property
    def shunt_names(self) -> tuple[str, ...]:
        """Return one-terminal equipment names in network order."""
        return tuple(shunt.name for shunt in self.shunts)

    @property
    def generators(self) -> tuple[GeneratorShunt, ...]:
        """Return synchronous generators from the canonical shunt collection."""
        return tuple(shunt for shunt in self.shunts if isinstance(shunt, GeneratorShunt))

    @property
    def load_names(self) -> tuple[str, ...]:
        """Return load names in network order."""
        return tuple(load.name for load in self.loads)

    @property
    def voltage_sources(self) -> tuple[VoltageSourceShunt | ExternalGridShunt, ...]:
        """Return AC voltage sources and external-grid equivalents."""
        return tuple(shunt for shunt in self.shunts if isinstance(shunt, (VoltageSourceShunt, ExternalGridShunt)))

    @property
    def voltage_source_names(self) -> tuple[str, ...]:
        """Return AC voltage source and external-grid names."""
        return tuple(source.name for source in self.voltage_sources)

    @property
    def loads(self) -> tuple[LoadShunt, ...]:
        """Return loads from the canonical shunt collection."""
        return tuple(shunt for shunt in self.shunts if isinstance(shunt, LoadShunt))

    @property
    def transformer_3w_names(self) -> tuple[str, ...]:
        """Return three-winding transformer names in network order."""
        return tuple(transformer.name for transformer in self.transformers_3w)

    def register_bus(self, name: str) -> None:
        """Include a bus, including one that has no connected elements yet."""
        if not name:
            raise ValueError("Bus name must not be empty.")
        if name not in self.buses:
            self.buses[name] = Bus(name=name)

    def add_branch(self, branch: Branch) -> None:
        """Add a two-terminal branch and its endpoint buses."""
        self.register_bus(branch.from_bus_name)
        self.register_bus(branch.to_bus_name)
        self.branches.append(branch)

    def add_shunt(self, shunt: Shunt) -> None:
        """Add a one-terminal element and its bus."""
        if isinstance(shunt, SourceShunt):
            if shunt.name in self.stability_source_names:
                raise ValueError(f"Duplicate active source name: {shunt.name}")
        self.register_bus(shunt.bus_name)
        self.shunts.append(shunt)

    def add_three_winding_transformer(self, transformer: Transformer3WBranch) -> None:
        """Add a three-winding transformer and its terminal buses."""
        self.register_bus(transformer.hv_bus_name)
        self.register_bus(transformer.mv_bus_name)
        self.register_bus(transformer.lv_bus_name)
        self.transformers_3w.append(transformer)

    @property
    def stability_sources(
        self,
    ) -> tuple[GeneratorShunt | VoltageSourceShunt | ExternalGridShunt, ...]:
        """Return sources in the PowerFactory adapter's operating-point order."""
        return self.generators + self.voltage_sources

    @property
    def generator_names(self) -> tuple[str, ...]:
        """Return synchronous-generator names."""
        return tuple(source.name for source in self.generators)

    @property
    def stability_source_names(self) -> tuple[str, ...]:
        """Return source names in the default reduced-matrix order."""
        return tuple(source.name for source in self.stability_sources)

    @property
    def stability_node_names(self) -> tuple[str, ...]:
        """Return augmented-matrix names: source internals, then buses."""
        internals = tuple(f"{name} internal" for name in self.stability_source_names)
        return internals + tuple(self.bus_names)

    def resolve_bus_name(self, bus_name: str) -> str:
        """Resolve a pre-merge bus name to its retained network name."""
        return (self.bus_mapping or {}).get(bus_name, bus_name)

    def get_passive_y_matrix(self) -> npt.NDArray[np.complex128]:
        """Delegate branch, transformer, and filter Y-bus construction."""
        return build_passive_y_matrix(self)

    def get_load_flow_y_matrix(
        self,
        operating_point: OperatingPoint | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Return the physical-bus matrix with constant-impedance loads."""
        return build_load_flow_y_matrix(self, operating_point)

    def get_stability_bus_y_matrix(
        self,
        operating_point: OperatingPoint,
        *,
        excluded_sources: Collection[str] | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Return the physical-bus matrix with source impedances."""
        return build_stability_bus_y_matrix(self, operating_point, excluded_sources=excluded_sources)

    def get_stability_y_matrix(
        self,
        operating_point: OperatingPoint,
        *,
        excluded_sources: Collection[str] | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Return the Kron-reduced classical stability admittance matrix."""
        return build_stability_y_matrix(self, operating_point, excluded_sources=excluded_sources)

    def get_extended_stability_y_matrix(
        self,
        operating_point: OperatingPoint,
        *,
        excluded_sources: Collection[str] | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Return the augmented matrix before physical-bus reduction."""
        return build_extended_stability_y_matrix(self, operating_point, excluded_sources=excluded_sources)

    def get_internal_voltage_vector(
        self,
        operating_point: OperatingPoint,
        *,
        excluded_sources: Collection[str] | None = None,
    ) -> npt.NDArray[np.complex128]:
        """Return internal source voltages in reduced-matrix order."""
        return build_internal_voltage_vector(self, operating_point, excluded_sources=excluded_sources)

    def _collect_bus_names(self) -> list[str]:
        names: set[str] = set()
        for branch in self.branches:
            names.update((branch.from_bus_name, branch.to_bus_name))
        for shunt in self.shunts:
            names.add(shunt.bus_name)
        for transformer in self.transformers_3w:
            names.update(
                (
                    transformer.hv_bus_name,
                    transformer.mv_bus_name,
                    transformer.lv_bus_name,
                )
            )
        return sorted(names)
