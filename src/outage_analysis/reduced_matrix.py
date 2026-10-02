"""Initial active-power redistribution from a reduced stability matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..network import Network, OperatingPoint
from .branch_flows import BranchActivePowerChange, calculate_all_branch_power_changes, calculate_branch_power_changes


@dataclass(frozen=True, slots=True, kw_only=True)
class ReducedMatrixResult:
    """Modeled source powers immediately after one source is disconnected.

    Powers are evaluated at source internal nodes. Surviving internal-voltage
    phasors are fixed at their prefault values, so these are initial modeled
    responses rather than dynamic or measured responses.
    """

    outaged_source: str
    prefault_power_mw: dict[str, float]
    postfault_power_mw: dict[str, float]
    power_change_mw: dict[str, float]
    redistribution_ratios: dict[str, float]
    lost_source_power_mw: float

    @property
    def source_names(self) -> tuple[str, ...]:
        """Surviving sources in reduced-matrix order."""
        return tuple(self.prefault_power_mw)

    @property
    def total_power_change_mw(self) -> float:
        """Sum of the surviving sources' modeled power changes."""
        return sum(self.power_change_mw.values())

    @property
    def lost_power_fractions(self) -> dict[str, float]:
        """Changes divided by the outaged source's prefault internal power."""
        if np.isclose(self.lost_source_power_mw, 0.0):
            raise ValueError(f"Prefault internal power of {self.outaged_source} is zero.")
        return {name: delta_p / self.lost_source_power_mw for name, delta_p in self.power_change_mw.items()}


@dataclass(slots=True)
class ReducedMatrixFormulation:
    """Calculate initial redistribution after disconnecting a source.

    For prefault reduced admittance Y and internal phasors E, I = Y E.
    Eliminating source port g gives ``delta I_R = -Y_Rg I_g / Y_gg`` for
    surviving ports R. This Schur complement matches rebuilding the
    postfault matrix when E is held fixed. Source power is Re(E conj(I)).
    """

    network: Network
    operating_point: OperatingPoint

    def calculate(self, outaged_source: str) -> dict[str, float]:
        """Return Mode 2 ratios, including zero for the outaged source."""
        ratios = self.calculate_result(outaged_source).redistribution_ratios
        return {name: 0.0 if name == outaged_source else ratios[name] for name in self.operating_point.source_names}

    def calculate_result(self, outaged_source: str) -> ReducedMatrixResult:
        """Return powers and ratios for one generator or other active source."""
        names, y, e, i, p0 = self._prefault_state()
        if outaged_source not in names:
            raise ValueError(f"Unknown outaged source: {outaged_source}")
        return self._result(outaged_source, names, y, e, i, p0)

    def calculate_all(self) -> dict[str, ReducedMatrixResult]:
        """Analyze every synchronous-generator outage from one reduction."""
        if not self.network.generators:
            raise ValueError("At least one generator is required for outage analysis.")
        names, y, e, i, p0 = self._prefault_state()
        return {
            generator.name: self._result(generator.name, names, y, e, i, p0) for generator in self.network.generators
        }

    def power_changes_mw(self, outaged_source: str) -> dict[str, float]:
        """Return the surviving sources' modeled active-power changes."""
        return self.calculate_result(outaged_source).power_change_mw

    def calculate_branch_flows(
        self, outaged_source: str, branch_names: list[str] | tuple[str, ...]
    ) -> dict[str, BranchActivePowerChange]:
        """Return modeled MW changes at the selected branches' from terminals."""
        return calculate_branch_power_changes(
            self.network,
            self.operating_point,
            outaged_source=outaged_source,
            branch_names=branch_names,
        )

    def calculate_all_branch_flows(
        self, branch_names: list[str] | tuple[str, ...]
    ) -> dict[str, dict[str, BranchActivePowerChange]]:
        """Return selected from-terminal branch flows for every generator outage."""
        return calculate_all_branch_power_changes(
            self.network,
            self.operating_point,
            branch_names=branch_names,
        )

    def _prefault_state(
        self,
    ) -> tuple[
        tuple[str, ...],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        npt.NDArray[np.float64],
    ]:
        names = self.network.stability_source_names
        if names != self.operating_point.source_names:
            raise ValueError("Operating-point source order must match network source order.")
        y = self.network.get_stability_y_matrix(self.operating_point)
        e = self.network.get_internal_voltage_vector(self.operating_point)
        if y.shape != (len(names), len(names)) or e.shape != (len(names),):
            raise ValueError("Reduced matrix or source voltage shape does not match source names.")
        i = y @ e
        p0 = (e * i.conjugate()).real * self.network.base_mva
        return names, y, e, i, p0

    def _result(
        self,
        outaged_source: str,
        names: tuple[str, ...],
        y: npt.NDArray[np.complex128],
        e: npt.NDArray[np.complex128],
        i: npt.NDArray[np.complex128],
        p0: npt.NDArray[np.float64],
    ) -> ReducedMatrixResult:
        g = names.index(outaged_source)
        if len(names) == 1:
            raise ValueError("At least one source must survive the outage.")
        if np.isclose(y[g, g], 0.0):
            raise ValueError(f"Reduced self-admittance is zero for {outaged_source}.")

        # Open port g: delta I_R = -Y_Rg I_g / Y_gg.
        delta_i = -y[:, g] * i[g] / y[g, g]
        delta_p = (e * delta_i.conjugate()).real * self.network.base_mva
        prefault = {name: float(p0[k]) for k, name in enumerate(names) if k != g}
        changes = {name: float(delta_p[k]) for k, name in enumerate(names) if k != g}
        postfault = {name: prefault[name] + changes[name] for name in prefault}
        total = sum(changes.values())
        tolerance_mw = 1e-12 * self.network.base_mva
        ratios = {name: change / total if abs(total) > tolerance_mw else 0.0 for name, change in changes.items()}
        return ReducedMatrixResult(
            outaged_source=outaged_source,
            prefault_power_mw=prefault,
            postfault_power_mw=postfault,
            power_change_mw=changes,
            redistribution_ratios=ratios,
            lost_source_power_mw=float(p0[g]),
        )
