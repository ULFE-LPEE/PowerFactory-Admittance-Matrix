"""Mode 1: source-admittance removal and disturbance-angle update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..matrices import extend_matrix_to_generator_internal_nodes, perform_kron_reduction
from ..network import Network, OperatingPoint


@dataclass(frozen=True, slots=True, kw_only=True)
class AngleUpdateResult:
    """Mode 1 assumed angles, signed coefficients, and outage ratios."""

    outaged_source: str
    disturbance_angle_rad: float
    source_angles_rad: dict[str, float]
    coefficients_mw_per_rad: dict[str, float]
    redistribution_ratios: dict[str, float]

    @property
    def source_names(self) -> tuple[str, ...]:
        """Surviving sources in reduced-matrix order."""
        return tuple(self.coefficients_mw_per_rad)

    @property
    def total_coefficient_mw_per_rad(self) -> float:
        """Signed sum used to normalize the coefficients."""
        return sum(self.coefficients_mw_per_rad.values())


@dataclass(slots=True)
class AngleUpdateFormulation:
    """Calculate the original Mode 1 signed redistribution ratios.

    Remove the outaged source admittance from the bus block, retain all
    internal nodes, and Kron-reduce the physical buses. Approximate the
    disturbance angle as ``arg(V_g) - atan2(Q_g, P_g)`` while surviving
    sources keep their prefault internal angles. Apply the Mode 0 coefficient
    equation to the resulting reduced matrix. This is not a dynamic solution.
    """

    network: Network
    operating_point: OperatingPoint

    def calculate(self, outaged_source: str) -> dict[str, float]:
        """Return Mode 1 ratios, including zero for the outaged source."""
        ratios = self.calculate_result(outaged_source).redistribution_ratios
        return {name: 0.0 if name == outaged_source else ratios[name] for name in self.operating_point.source_names}

    def calculate_result(self, outaged_source: str) -> AngleUpdateResult:
        """Return angles, coefficients, and ratios for one source outage."""
        names, y_bus, bus_idx, e = self._prefault_state()
        if outaged_source not in names:
            raise ValueError(f"Unknown outaged source: {outaged_source}")
        return self._result(outaged_source, names, y_bus, bus_idx, e)

    def calculate_all(self) -> dict[str, AngleUpdateResult]:
        """Analyze every generator outage, reusing the prefault bus matrix."""
        if not self.network.generators:
            raise ValueError("At least one generator is required for outage analysis.")
        names, y_bus, bus_idx, e = self._prefault_state()
        return {
            generator.name: self._result(generator.name, names, y_bus, bus_idx, e)
            for generator in self.network.generators
        }

    def _prefault_state(
        self,
    ) -> tuple[
        tuple[str, ...],
        npt.NDArray[np.complex128],
        dict[str, int],
        npt.NDArray[np.complex128],
    ]:
        names = self.network.stability_source_names
        if names != self.operating_point.source_names:
            raise ValueError("Operating-point source order must match network source order.")
        if not names:
            raise ValueError("At least one source is required for outage analysis.")
        y_bus = self.network.get_stability_bus_y_matrix(self.operating_point)
        bus_idx = {name: index for index, name in enumerate(self.network.bus_names)}
        e = self.network.get_internal_voltage_vector(self.operating_point)
        if e.shape != (len(names),):
            raise ValueError("Source voltage shape does not match source names.")
        return names, y_bus, bus_idx, e

    def _result(
        self,
        outaged_source: str,
        names: tuple[str, ...],
        prefault_y_bus: npt.NDArray[np.complex128],
        bus_idx: dict[str, int],
        e: npt.NDArray[np.complex128],
    ) -> AngleUpdateResult:
        d = names.index(outaged_source)
        source = self.network.stability_sources[d]
        y_bus = prefault_y_bus.copy()
        bus = bus_idx[source.bus_name]
        y_bus[bus, bus] -= source.get_admittance_pu(self.network.base_mva)

        # Mode 1 retains the outaged internal node in the augmented matrix.
        extended_y = extend_matrix_to_generator_internal_nodes(
            Y_bus=y_bus,
            bus_idx=bus_idx,
            sources=list(self.network.stability_sources),
            base_mva=self.network.base_mva,
        )
        y = perform_kron_reduction(extended_y, list(range(len(names))))

        angles = np.angle(e)
        result = self.operating_point.sources[d]
        disturbance_angle = np.angle(result.terminal_voltage) - np.arctan2(result.q_pu, result.p_pu)
        angles[d] = disturbance_angle
        delta = angles - disturbance_angle
        k = np.abs(e) * np.abs(e[d]) * (y[:, d].imag * np.cos(delta) - y[:, d].real * np.sin(delta))
        k = np.nan_to_num(k, nan=0.0)
        k[d] = 0.0
        coefficients = {name: float(k[i] * self.network.base_mva) for i, name in enumerate(names) if i != d}
        total = sum(coefficients.values())
        ratios = {name: value / total if total != 0.0 else 0.0 for name, value in coefficients.items()}
        return AngleUpdateResult(
            outaged_source=outaged_source,
            disturbance_angle_rad=float(disturbance_angle),
            source_angles_rad={name: float(angles[i]) for i, name in enumerate(names)},
            coefficients_mw_per_rad=coefficients,
            redistribution_ratios=ratios,
        )
