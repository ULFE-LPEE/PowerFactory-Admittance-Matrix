"""Mode 0: prefault synchronizing-power coefficient formulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..network import Network, OperatingPoint


@dataclass(frozen=True, slots=True, kw_only=True)
class SynchronizingResult:
    """Signed coupling coefficients and ratios for one source outage."""

    outaged_source: str
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
class SynchroCoefficients:
    """Distribute a source outage using prefault synchronizing coefficients.

    For internal phasors E and reduced admittance Y = G + jB,
    ``K_id = |E_i||E_d|[B_id cos(delta_i-delta_d)
    - G_id sin(delta_i-delta_d)]``. The outaged source's *column* is used,
    preserving the original Mode 0 convention when Y is asymmetric.
    This is an initial, fixed-voltage approximation.
    """

    network: Network
    operating_point: OperatingPoint

    @property
    def pairwise_pu_per_rad(self) -> npt.NDArray[np.float64]:
        """Return all K_id values, with a zero diagonal."""
        names = self.network.stability_source_names
        if names != self.operating_point.source_names:
            raise ValueError("Operating-point source order must match network source order.")
        y = self.network.get_stability_y_matrix(self.operating_point)
        e = self.network.get_internal_voltage_vector(self.operating_point)
        if y.shape != (len(names), len(names)) or e.shape != (len(names),):
            raise ValueError("Reduced matrix or source voltage shape does not match source names.")
        angles = np.angle(e)
        delta = angles[:, None] - angles[None, :]
        k = np.outer(np.abs(e), np.abs(e)) * (y.imag * np.cos(delta) - y.real * np.sin(delta))
        k = np.nan_to_num(k, nan=0.0)
        np.fill_diagonal(k, 0.0)
        return k

    @property
    def pairwise_mw_per_rad(self) -> npt.NDArray[np.float64]:
        """Return all synchronizing coefficients in MW/rad."""
        return self.pairwise_pu_per_rad * self.network.base_mva

    @property
    def angle_jacobian_pu_per_rad(self) -> npt.NDArray[np.float64]:
        """Return the angle-coupling matrix with zero row sums."""
        k = self.pairwise_pu_per_rad
        jacobian = -k.copy()
        np.fill_diagonal(jacobian, np.sum(k, axis=1))
        return jacobian

    @property
    def angle_jacobian_mw_per_rad(self) -> npt.NDArray[np.float64]:
        """Return the angle-coupling matrix in MW/rad."""
        return self.angle_jacobian_pu_per_rad * self.network.base_mva

    def coefficients_for_outage(self, outaged_source: str) -> dict[str, float]:
        """Return each surviving source's coupling to the outaged source."""
        return self.calculate_result(outaged_source).coefficients_mw_per_rad

    def calculate(self, outaged_source: str) -> dict[str, float]:
        """Return Mode 0 ratios, including zero for the outaged source."""
        ratios = self.calculate_result(outaged_source).redistribution_ratios
        return {name: 0.0 if name == outaged_source else ratios[name] for name in self.operating_point.source_names}

    def calculate_result(self, outaged_source: str) -> SynchronizingResult:
        """Return coefficients and ratios for one source outage."""
        names = self.operating_point.source_names
        if outaged_source not in names:
            raise ValueError(f"Unknown outaged source: {outaged_source}")
        return self._result(outaged_source, names, self.pairwise_mw_per_rad)

    def calculate_all(self) -> dict[str, SynchronizingResult]:
        """Analyze every synchronous-generator outage from one coefficient matrix."""
        if not self.network.generators:
            raise ValueError("At least one generator is required for outage analysis.")
        names = self.operating_point.source_names
        k = self.pairwise_mw_per_rad
        return {generator.name: self._result(generator.name, names, k) for generator in self.network.generators}

    def _result(
        self,
        outaged_source: str,
        names: tuple[str, ...],
        k: npt.NDArray[np.float64],
    ) -> SynchronizingResult:
        d = names.index(outaged_source)
        coefficients = {name: float(k[i, d]) for i, name in enumerate(names) if i != d}
        total = sum(coefficients.values())
        ratios = {name: value / total if total != 0.0 else 0.0 for name, value in coefficients.items()}
        return SynchronizingResult(
            outaged_source=outaged_source,
            coefficients_mw_per_rad=coefficients,
            redistribution_ratios=ratios,
        )
