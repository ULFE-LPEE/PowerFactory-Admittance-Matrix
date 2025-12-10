"""
Element definitions for power system network components.

This module contains the base classes and implementations for:
- Branch elements (lines, switches)
- Shunt elements (loads, generators)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import powerfactory as pf


@dataclass
class BranchElement(ABC):
    """Abstract base class for two-terminal elements."""
    pf_object: pf.DataObject
    name: str
    from_bus_name: str
    to_bus_name: str
    voltage_kv: float
    admittance: complex = field(init=False)
    shunt_admittance: complex = field(default=complex(0, 0))
    
    @property
    def impedance(self) -> complex:
        """Return impedance Z = 1/Y (in Ohms)"""
        if self.admittance == 0:
            return complex(float('inf'), float('inf'))
        return 1 / self.admittance
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        Y_pu = Y_siemens * Z_base = Y_siemens * (V_base^2 / S_base)
        """
        if self.voltage_kv > 0:
            z_base = (self.voltage_kv ** 2) / base_mva
            return self.admittance * z_base
        return self.admittance
    
    def get_y_matrix_entries(self, base_mva: float | None = None) -> tuple[complex, complex, complex, complex]:
        """
        Return Y-matrix contributions: (Yii, Yjj, Yij, Yji)
        For symmetric elements: Yii = Yjj = y, Yij = Yji = -y
        """
        y = self.get_admittance_pu(base_mva) if base_mva else self.admittance
        return (y, y, -y, -y)


@dataclass
class LineBranch(BranchElement):
    """Transmission/distribution line element."""
    resistance_ohm: float = 0.0
    reactance_ohm: float = 0.0
    susceptance_us: float = 0.0  # Total susceptance in µS
    
    def __post_init__(self):
        # Calculate series admittance
        if self.resistance_ohm == 0 and self.reactance_ohm == 0:
            self.admittance = complex(0, 0)
        else:
            self.admittance = 1 / complex(self.resistance_ohm, self.reactance_ohm)
        
        # Calculate shunt admittance (B/2 at each end)
        self.shunt_admittance = complex(0, self.susceptance_us * 1e-6 / 2)
    
    def get_y_matrix_entries(self, base_mva: float | None = None) -> tuple[complex, complex, complex, complex]:
        """Include shunt admittance (pi-model)"""
        if base_mva and self.voltage_kv > 0:
            z_base = (self.voltage_kv ** 2) / base_mva
            y = self.admittance * z_base
            y_shunt = self.shunt_admittance * z_base
        else:
            y = self.admittance
            y_shunt = self.shunt_admittance
        # Diagonal includes series + shunt, off-diagonal is just series
        return (y + y_shunt, y + y_shunt, -y, -y)


@dataclass
class SwitchBranch(BranchElement):
    """Switch/Coupler element."""
    is_closed: bool = True
    
    def __post_init__(self):
        if self.is_closed:
            # Closed switch = very high admittance (1 micro-ohm)
            self.admittance = complex(1e6, 0)
        else:
            self.admittance = complex(0, 0)


@dataclass
class TransformerBranch(BranchElement):
    """
    Two-winding transformer element.
    
    Uses the standard transformer pi-model with off-nominal tap ratio.
    The model accounts for:
    - Series impedance (R + jX) on transformer base
    - Tap ratio (t) - ratio of actual voltage to nominal voltage on HV side
    - Magnetizing admittance (optional)
    
    Y-matrix for transformer between buses i (HV) and j (LV):
        Y_ii = y / t²
        Y_jj = y
        Y_ij = Y_ji = -y / t
    
    Where y = 1/(R + jX) in per-unit on system base.
    """
    rated_power_mva: float = 0.0
    hv_kv: float = 0.0  # HV side rated voltage
    lv_kv: float = 0.0  # LV side rated voltage
    resistance_pu: float = 0.0  # R on transformer base (p.u.)
    reactance_pu: float = 0.0   # X on transformer base (p.u.)
    tap_ratio: float = 1.0      # Tap ratio (actual/nominal on HV side)
    tap_side: int = 0           # 0 = HV side, 1 = LV side
    magnetizing_admittance: complex = field(default=complex(0, 0))  # Y_m in p.u.
    
    def __post_init__(self):
        # Calculate series admittance in per-unit on transformer base
        if self.resistance_pu == 0 and self.reactance_pu == 0:
            self.admittance = complex(0, 0)
        else:
            z_pu_trafo = complex(self.resistance_pu, self.reactance_pu)
            self.admittance = 1 / z_pu_trafo  # Y in p.u. on transformer base
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get series admittance in per-unit on system base.
        
        Conversion: Y_sys = Y_trafo * (S_trafo / S_sys)
        """
        if self.rated_power_mva > 0:
            return self.admittance * (self.rated_power_mva / base_mva)
        return self.admittance
    
    def get_y_matrix_entries(self, base_mva: float | None = None) -> tuple[complex, complex, complex, complex]:
        """
        Return Y-matrix contributions for transformer with tap ratio.
        
        For tap on HV side (from_bus):
            Y_ii = y / t²    (HV side)
            Y_jj = y         (LV side)  
            Y_ij = Y_ji = -y / t
            
        For tap on LV side (to_bus):
            Y_ii = y         (HV side)
            Y_jj = y / t²    (LV side)
            Y_ij = Y_ji = -y / t
        """
        y = self.get_admittance_pu(base_mva) if base_mva else self.admittance
        t = self.tap_ratio
        
        if self.tap_side == 0:  # Tap on HV (from) side
            Yii = y / (t * t)
            Yjj = y
            Yij = -y / t
        else:  # Tap on LV (to) side
            Yii = y
            Yjj = y / (t * t)
            Yij = -y / t
        
        return (Yii, Yjj, Yij, Yij)


@dataclass
class ShuntElement(ABC):
    """Abstract base class for single-terminal elements."""
    name: str
    bus_name: str
    voltage_kv: float
    admittance: complex = field(init=False)
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        Y_pu = Y_siemens * Z_base = Y_siemens * (V_base^2 / S_base)
        """
        if self.voltage_kv > 0:
            z_base = (self.voltage_kv ** 2) / base_mva
            return self.admittance * z_base
        return self.admittance


@dataclass
class LoadShunt(ShuntElement):
    """Load element - constant impedance model."""
    p_mw: float = 0.0
    q_mvar: float = 0.0
    
    def __post_init__(self):
        # Load: P + jQ -> Y = (P - jQ) / |V|^2
        if self.voltage_kv > 0:
            self.admittance = complex(self.p_mw, -self.q_mvar) / (self.voltage_kv ** 2)
        else:
            self.admittance = complex(0, 0)


@dataclass
class GeneratorShunt(ShuntElement):
    """Synchronous generator - transient/sub-transient reactance model."""
    rated_power_mva: float = 0.0
    rated_voltage_kv: float = 0.0
    xdss_pu: float = 0.0  # Sub-transient reactance on generator base
    
    def __post_init__(self):
        """Calculate generator admittance behind sub-transient reactance."""
        if self.xdss_pu > 0 and self.rated_power_mva > 0 and self.rated_voltage_kv > 0:
            # Calculate impedance in ohms
            z_base = (self.rated_voltage_kv ** 2) / self.rated_power_mva
            x_ohms = self.xdss_pu * z_base
            # Y = 1 / (jX) = -j/X
            self.admittance = complex(0, -1 / x_ohms)
        else:
            self.admittance = complex(0, 0)
