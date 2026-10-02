"""
Load-flow and source data for a solved network operating point.

This module contains result data structures for:
- Bus load flow results
- Generator operating data and internal voltage calculations
- Voltage source results
- External grid results
"""

from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass
class BusResult:
    """Load flow results for a busbar."""

    name: str
    voltage_pu: float
    angle_deg: float
    voltage_kv: float

    @property
    def voltage_complex(self) -> complex:
        """Return voltage as complex phasor (p.u.)"""
        angle_rad = math.radians(self.angle_deg)
        return self.voltage_pu * complex(math.cos(angle_rad), math.sin(angle_rad))


@dataclass
class GeneratorResult:
    """Generator data with terminal voltage, impedance, and internal voltage."""

    name: str
    bus_name: str
    terminal_voltage: complex  # Terminal voltage as complex phasor (p.u.)
    impedance_pu: complex  # Impedance on system base (p.u.)
    p_pu: float  # Active power on the network base (p.u.)
    q_pu: float  # Reactive power on the network base (p.u.)
    internal_voltage: complex  # Internal voltage E' behind X''d (p.u.)
    rated_mva: float
    rated_kv: float
    source_type: str = "generator"  # Source type identifier


@dataclass
class VoltageSourceResult:
    """Active voltage-source data, including modeled grid-forming converters."""

    name: str
    bus_name: str
    terminal_voltage: complex  # Terminal voltage as complex phasor (p.u.)
    impedance_pu: complex  # Internal impedance on system base (p.u.)
    p_pu: float  # Active power on system base (p.u.)
    q_pu: float  # Reactive power on system base (p.u.)
    internal_voltage: complex  # Internal voltage (behind impedance)
    internal_voltage_mag: float  # |E| magnitude (p.u.)
    internal_voltage_angle: float  # E angle (degrees)
    source_type: str = "voltage_source"  # Source type identifier


@dataclass
class ExternalGridResult:
    """External grid data with terminal voltage and internal voltage."""

    name: str
    bus_name: str
    terminal_voltage: complex  # Terminal voltage as complex phasor (p.u.)
    impedance_pu: complex  # Internal impedance on system base (p.u.)
    p_pu: float  # Active power on system base (p.u.)
    q_pu: float  # Reactive power on system base (p.u.)
    internal_voltage: complex  # Internal voltage (behind impedance)
    internal_voltage_mag: float  # |E| magnitude (p.u.)
    internal_voltage_angle: float  # E angle (degrees)
    source_type: str = "external_grid"  # Source type identifier


SourceResult = GeneratorResult | VoltageSourceResult | ExternalGridResult


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """PowerFactory-independent snapshot of one solved load-flow condition.

    ``sources`` preserves the order used by source-internal admittance matrices.
    The source P and Q values are per unit on ``base_mva``.
    """

    base_mva: float
    buses: Mapping[str, BusResult]
    sources: tuple[SourceResult, ...]

    def __post_init__(self) -> None:
        if self.base_mva <= 0:
            raise ValueError("Operating-point base MVA must be greater than zero.")
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Operating-point source names must be unique.")

    @property
    def source_names(self) -> tuple[str, ...]:
        """Return source names in reduced-matrix order."""
        return tuple(source.name for source in self.sources)

    @property
    def source_types(self) -> tuple[str, ...]:
        """Return source kinds in the same reduced-matrix order."""
        return tuple(source.source_type for source in self.sources)
