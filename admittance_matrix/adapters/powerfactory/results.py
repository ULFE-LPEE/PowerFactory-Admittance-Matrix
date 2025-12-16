"""
Load flow and generator result classes.

This module contains result data structures for:
- Bus load flow results
- Generator operating data and internal voltage calculations
- Voltage source results
- External grid results
"""

from dataclasses import dataclass
import cmath
import math


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
    voltage: complex          # Terminal voltage as complex phasor (p.u.)
    xdss_pu: float           # Sub-transient reactance on generator base (p.u.)
    impedance_pu: complex    # Impedance on system base (p.u.)
    p_pu: float              # Active power on generator base (p.u.)
    q_pu: float              # Reactive power on generator base (p.u.)
    internal_voltage: complex # Internal voltage E' behind X''d (p.u.)
    internal_voltage_mag: float  # |E'| magnitude (p.u.)
    internal_voltage_angle: float  # E' angle (degrees)
    rated_mva: float
    rated_kv: float
    zone: str = 'Unknown'     # Zone name from cpZone attribute
    source_type: str = 'generator'  # Source type identifier


@dataclass
class VoltageSourceResult:
    """Voltage source data with terminal voltage and internal voltage."""
    name: str
    bus_name: str
    voltage: complex          # Terminal voltage as complex phasor (p.u.)
    impedance_pu: complex     # Internal impedance on system base (p.u.)
    internal_voltage: complex # Internal voltage (behind impedance)
    internal_voltage_mag: float  # |E| magnitude (p.u.)
    internal_voltage_angle: float  # E angle (degrees)
    source_type: str = 'voltage_source'  # Source type identifier


@dataclass
class ExternalGridResult:
    """External grid data with terminal voltage and internal voltage."""
    name: str
    bus_name: str
    voltage: complex          # Terminal voltage as complex phasor (p.u.)
    impedance_pu: complex     # Internal impedance on system base (p.u.)
    internal_voltage: complex # Internal voltage (behind impedance)
    internal_voltage_mag: float  # |E| magnitude (p.u.)
    internal_voltage_angle: float  # E angle (degrees)
    source_type: str = 'external_grid'  # Source type identifier
