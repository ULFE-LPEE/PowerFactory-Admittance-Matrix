"""
Element definitions for power system network components.

This module contains the base classes and implementations for:
- Branch elements (lines, switches)
- Shunt elements (loads, generators)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import powerfactory as pf


class ShuntFilterType(Enum):
    """Shunt filter/capacitor layout types matching PowerFactory ElmShnt."""
    R_L_C = 0       # Series R-L with parallel C
    R_L = 1         # Series R-L (reactor only)
    C = 2           # Capacitor only
    R_L_C_Rp = 3    # Series R-L-C with parallel R
    R_L_C1_C2_Rp = 4  # High-pass filter: Series R-L-C1 with parallel C2 and Rp


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
    n_parallel: int = 1  # Number of parallel systems (nlnum in PowerFactory)
    
    def __post_init__(self):
        # Calculate series admittance for a single line
        if self.resistance_ohm == 0 and self.reactance_ohm == 0:
            single_admittance = complex(0, 0)
        else:
            single_admittance = 1 / complex(self.resistance_ohm, self.reactance_ohm)
        
        # Multiply by number of parallel systems (parallel admittances add up)
        self.admittance = single_admittance * self.n_parallel
        
        # Calculate shunt admittance (B/2 at each end), also scaled by n_parallel
        self.shunt_admittance = complex(0, self.susceptance_us * 1e-6 / 2) * self.n_parallel
    
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
    - Number of parallel transformers (ntnum)
    
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
    n_parallel: int = 1  # Number of parallel transformers (ntnum in PowerFactory)
    
    def __post_init__(self):
        # Calculate series admittance in per-unit on transformer base for single transformer
        if self.resistance_pu == 0 and self.reactance_pu == 0:
            single_admittance = complex(0, 0)
        else:
            z_pu_trafo = complex(self.resistance_pu, self.reactance_pu)
            single_admittance = 1 / z_pu_trafo  # Y in p.u. on transformer base
        
        # Multiply by number of parallel transformers (parallel admittances add up)
        self.admittance = single_admittance * self.n_parallel
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get series admittance in per-unit on system base.
        
        Conversion: Y_sys = Y_trafo * (S_trafo / S_sys)
        Note: rated_power_mva is for a single transformer, n_parallel already applied to admittance
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
class CommonImpedanceBranch(BranchElement):
    """
    Common impedance element (ElmZpu).
    
    Models a general two-port impedance element used for:
    - Simplified transformer models
    - Series reactors
    - Equivalent network impedances
    
    Accepts R and X values in Ohms (typically obtained from PowerFactory's
    GetImpedance() method in the extractor), then converts to per-unit
    on the system base.
    
    Y-matrix contributions:
        Y_ii = Y_jj = y
        Y_ij = Y_ji = -y
    
    Where y = 1/Z in per-unit on system base.
    """
    resistance_ohm: float = 0.0  # Resistance in Ohms
    reactance_ohm: float = 0.0   # Reactance in Ohms
    hv_kv: float = 0.0           # High voltage side rated voltage (kV)
    lv_kv: float = 0.0           # Low voltage side rated voltage (kV)
    rated_power_mva: float = 0.0 # Rated power (MVA)
    
    def __post_init__(self):
        """Calculate admittance from R and X values in Ohms."""
        # Avoid division by zero - use small values for zero R or X
        R_ohm = self.resistance_ohm if self.resistance_ohm != 0 else 1e-12
        X_ohm = self.reactance_ohm if self.reactance_ohm != 0 else 1e-12
        Z_ohm = complex(R_ohm, X_ohm)

        self.admittance = 1 / Z_ohm
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        
        Note: The admittance is already calculated in __post_init__ on the
        system base (_base_mva). If a different base is requested, we need
        to rescale.
        """
        Z_base = (self.hv_kv ** 2) / base_mva
        if Z_base == 0:
            Z_base = 1e-12
        Z_pu = 1 / self.admittance / Z_base
        return (1 / Z_pu)
    
    def get_y_matrix_entries(self, base_mva: float | None = None) -> tuple[complex, complex, complex, complex]:
        """
        Return Y-matrix contributions for common impedance.
        
        For a simple series impedance:
            Y_ii = y
            Y_jj = y
            Y_ij = Y_ji = -y
        """
        y = self.get_admittance_pu(base_mva) if base_mva else self.admittance
        return (y, y, -y, -y)


@dataclass
class SeriesReactorBranch(BranchElement):
    """
    Series reactor element (ElmSind).
    
    Models a series reactor (inductor) between two buses, typically used for:
    - Current limiting reactors
    - Fault current reduction
    - Power flow control
    
    Accepts R and X values in Ohms (typically obtained from PowerFactory's
    GetImpedance() method in the extractor), then converts to per-unit
    on the system base.
    
    Y-matrix contributions:
        Y_ii = Y_jj = y
        Y_ij = Y_ji = -y
    
    Where y = 1/Z in per-unit on system base.
    """
    resistance_ohm: float = 0.0  # Resistance in Ohms
    reactance_ohm: float = 0.0   # Reactance in Ohms
    rated_power_mva: float = 0.0 # Rated power (MVA) - used for base conversion
    
    def __post_init__(self):
        """Calculate admittance from R and X values in Ohms."""
        if self.voltage_kv > 0:
            # Avoid division by zero - use small values for zero R or X
            R_ohm = self.resistance_ohm if self.resistance_ohm != 0 else 1e-12
            X_ohm = self.reactance_ohm if self.reactance_ohm != 0 else 1e-12
            Z_ohm = complex(R_ohm, X_ohm)

            self.admittance = 1 / Z_ohm
        else:
            self.admittance = complex(0, 0)
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        
        Note: The admittance is already calculated in __post_init__ on the
        system base (_base_mva). If a different base is requested, we need
        to rescale.
        """
        Z_base = (self.voltage_kv ** 2) / base_mva
        if Z_base == 0:
            Z_base = 1e-12
        Z_pu = 1 / self.admittance / Z_base
        return (1 / Z_pu)
    
    def get_y_matrix_entries(self, base_mva: float | None = None) -> tuple[complex, complex, complex, complex]:
        """
        Return Y-matrix contributions for series reactor.
        
        For a simple series impedance:
            Y_ii = y
            Y_jj = y
            Y_ij = Y_ji = -y
        """
        y = self.get_admittance_pu(base_mva) if base_mva else self.admittance
        return (y, y, -y, -y)


@dataclass
class Transformer3WBranch:
    """
    Three-winding transformer element.
    
    Uses PowerFactory's GetZpu() method to retrieve pair impedances directly
    and builds a 3×3 nodal admittance matrix (HV, MV, LV order).
    
    The pair impedances are:
    - Z_HM (pair 0): HV-MV impedance (LV open)
    - Z_ML (pair 1): MV-LV impedance (HV open)  
    - Z_LH (pair 2): LV-HV impedance (MV open)
    
    Local node order: [HV, MV, LV]
    """
    pf_object: pf.DataObject  # ElmTr3
    name: str
    hv_bus_name: str
    mv_bus_name: str
    lv_bus_name: str
    base_mva: float = 100.0  # System base MVA
    n_parallel: int = 1  # Number of parallel transformers
    
    # Optional: store rated values for reference
    rated_power_mva: float = 0.0
    hv_kv: float = 0.0
    mv_kv: float = 0.0
    lv_kv: float = 0.0
    
    def _get_tap_z_dependent_side(self) -> int:
        """
        Side whose tap affects the impedance (0=HV, 1=MV, 2=LV, or -1 if none).
        """
        return self.pf_object.GetTapZDependentSide()
    
    def _get_tap_position_for_Z(self) -> int:
        """
        Tap position to use in GetZpu: tap of the Z-dependent side.
        If none is defined, fall back to HV side tap (0).
        """
        z_side = self._get_tap_z_dependent_side()
        if z_side < 0:
            z_side = 0  # fallback: HV side
        return self.pf_object.NTap(z_side)
    
    def _get_Z_pair(self, pair_index: int) -> complex:
        """
        Get pair impedance in pu on SYSTEM base between:
            pair_index = 0 → HV-MV
                         1 → MV-LV
                         2 → LV-HV
        """
        tap_pos = self._get_tap_position_for_Z()
        # systembase = 1 → pu on system base (e.g. 100 MVA)
        rpu, xpu = self.pf_object.GetZpu(tap_pos, pair_index, 1)
        return complex(rpu, xpu)
    
    def Z_hm(self) -> complex:
        """Z between HV and MV in pu on system base."""
        return self._get_Z_pair(0)
    
    def Z_ml(self) -> complex:
        """Z between MV and LV in pu on system base."""
        return self._get_Z_pair(1)
    
    def Z_lh(self) -> complex:
        """Z between LV and HV in pu on system base."""
        return self._get_Z_pair(2)
    
    def _safe_inv(self, Z: complex, eps: float = 1e-12) -> complex:
        """Avoid infinities in admittance; treat |Z|<eps as open (0 admittance)."""
        if abs(Z) < eps:
            return complex(0.0, 0.0)
        return 1.0 / Z
    
    def Y_hm(self) -> complex:
        """Admittance HV-MV."""
        return self._safe_inv(self.Z_hm()) * self.n_parallel
    
    def Y_ml(self) -> complex:
        """Admittance MV-LV."""
        return self._safe_inv(self.Z_ml()) * self.n_parallel
    
    def Y_lh(self) -> complex:
        """Admittance LV-HV."""
        return self._safe_inv(self.Z_lh()) * self.n_parallel
    
    def get_local_admittance_matrix(self) -> tuple[list[list[complex]], list[str]]:
        """
        Returns 3×3 pu nodal admittance matrix in order [HV, MV, LV].
        
        Returns:
            Tuple of (3x3 matrix as nested list, [hv_bus, mv_bus, lv_bus])
        """
        Y_ab = self.Y_hm()  # HV-MV
        Y_bc = self.Y_ml()  # MV-LV
        Y_ca = self.Y_lh()  # LV-HV
        
        # Build 3x3 matrix
        # Off-diagonals
        Y_01 = -Y_ab  # HV-MV
        Y_12 = -Y_bc  # MV-LV
        Y_02 = -Y_ca  # HV-LV
        
        # Diagonals
        Y_00 = Y_ab + Y_ca  # HV
        Y_11 = Y_ab + Y_bc  # MV
        Y_22 = Y_bc + Y_ca  # LV
        
        matrix = [
            [Y_00, Y_01, Y_02],
            [Y_01, Y_11, Y_12],
            [Y_02, Y_12, Y_22]
        ]
        
        bus_names = [self.hv_bus_name, self.mv_bus_name, self.lv_bus_name]
        
        return matrix, bus_names
    
    def get_y_matrix_contributions(self, base_mva: float = 100.0) -> dict:
        """
        Get Y-matrix contributions for the 3-winding transformer.
        
        Returns a dictionary with entries for each bus pair.
        Format: {(bus_i, bus_j): (Yii_contrib, Yjj_contrib, Yij, Yji)}
        
        Note: base_mva parameter is kept for API compatibility but the
        impedances are already on system base from GetZpu().
        """
        Y_ab = self.Y_hm()  # HV-MV
        Y_bc = self.Y_ml()  # MV-LV  
        Y_ca = self.Y_lh()  # LV-HV
        
        contributions = {
            # HV-MV pair
            (self.hv_bus_name, self.mv_bus_name): (Y_ab, Y_ab, -Y_ab, -Y_ab),
            # MV-LV pair
            (self.mv_bus_name, self.lv_bus_name): (Y_bc, Y_bc, -Y_bc, -Y_bc),
            # LV-HV pair (note: LV first, HV second to maintain consistency)
            (self.lv_bus_name, self.hv_bus_name): (Y_ca, Y_ca, -Y_ca, -Y_ca),
        }
        
        return contributions
    
    @property
    def bus_names(self) -> list[str]:
        """List of connected bus names [HV, MV, LV]."""
        return [self.hv_bus_name, self.mv_bus_name, self.lv_bus_name]


@dataclass
class ShuntElement(ABC):
    """Abstract base class for single-terminal elements."""
    pf_object: pf.DataObject
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
    """
    Load element - constant impedance model.
    
    For stability analysis, use update_admittance_with_lf_voltage() after
    running load flow to recalculate admittance using actual bus voltage.
    """
    p_mw: float = 0.0
    q_mvar: float = 0.0
    lf_voltage_kv: float = field(default=0.0, repr=False)  # Load flow voltage (set after LF)
    
    def __post_init__(self):
        # Load: P + jQ -> Y = (P - jQ) / |V|^2
        # Initially use nominal voltage
        if self.voltage_kv > 0:
            self.admittance = complex(self.p_mw, -self.q_mvar) / (self.voltage_kv ** 2)
        else:
            self.admittance = complex(0, 0)
    
    def update_admittance_with_lf_voltage(self) -> None:
        """
        Recalculate admittance using load flow voltage.
        
        Call this after running load flow to get accurate constant impedance
        model for stability analysis. Uses lf_voltage_kv if set, otherwise
        falls back to nominal voltage_kv.
        """
        # Use LF voltage if available, otherwise nominal
        v_kv = self.lf_voltage_kv if self.lf_voltage_kv > 0 else self.voltage_kv
        
        if v_kv > 0:
            self.admittance = complex(self.p_mw, -self.q_mvar) / (v_kv ** 2)
        else:
            self.admittance = complex(0, 0)
    
    def set_lf_voltage(self, voltage_kv: float) -> None:
        """Set the load flow voltage and recalculate admittance."""
        self.lf_voltage_kv = voltage_kv
        self.update_admittance_with_lf_voltage()


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
            self.admittance = complex(0, -1 / x_ohms)
        else:
            self.admittance = complex(0, 0)


@dataclass
class ExternalGridShunt(ShuntElement):
    """
    External grid element (network equivalent).
    
    Models an external grid as a shunt admittance based on short-circuit power.
    """
    s_sc_mva: float = 0.0     # Short-circuit power in MVA
    c_factor: float = 1.0     # Voltage factor for short-circuit calculation
    r_x_ratio: float = 0.1    # R/X ratio
    
    def __post_init__(self):
        """Calculate grid admittance from short-circuit parameters."""
        if self.s_sc_mva > 0 and self.voltage_kv > 0:
            # Short-circuit impedance magnitude
            z_sc = (self.voltage_kv ** 2) / self.s_sc_mva
            
            # Calculate R and X components from R/X ratio
            # R/X = ratio => R = X * ratio
            # |Z|² = R² + X² => X = |Z| / sqrt(1 + ratio²)
            x_sc = z_sc / ((1 + self.r_x_ratio ** 2) ** 0.5) * self.c_factor
            r_sc = x_sc * self.r_x_ratio
            
            # Impedance and admittance
            z_complex = complex(r_sc, x_sc)
            self.admittance = 1 / z_complex
        else:
            self.admittance = complex(0, 0)


@dataclass
class VoltageSourceShunt(ShuntElement):
    """
    AC voltage source element.
    
    Models an AC voltage source with specified R and X values.
    """
    resistance_ohm: float = 0.0
    reactance_ohm: float = 0.0
    
    def __post_init__(self):
        """Calculate admittance from R and X parameters."""
        # Avoid division by zero
        r = self.resistance_ohm if self.resistance_ohm != 0 else 1e-12
        x = self.reactance_ohm if self.reactance_ohm != 0 else 1e-12
        
        z_complex = complex(r, x)
        self.admittance = 1 / z_complex


@dataclass
class ShuntFilterShunt(ShuntElement):
    """
    Shunt filter/capacitor element (ElmShnt).
    
    Models various shunt filter configurations:
    - R_L_C: Series R-L with parallel C (tuned filter)
    - R_L: Series R-L (shunt reactor)
    - C: Capacitor bank
    - R_L_C_Rp: Series R-L-C with parallel R (damped filter)
    - R_L_C1_C2_Rp: High-pass filter with series R-L-C1 and parallel C2, Rp
    
    All parameters are stored in base units (Ohms, µS, µF) and converted
    to per-unit in get_admittance_pu().
    """
    filter_type: ShuntFilterType = ShuntFilterType.C
    
    # Active steps
    n_cap: float = 1.0   # Number of active capacitor steps
    n_rea: float = 1.0   # Number of active reactor steps
    
    # R-L-C parameters (per step, in base units)
    bcap_us: float = 0.0     # Capacitor susceptance per step [µS]
    xrea_ohm: float = 0.0    # Reactor reactance per step [Ohm]
    rrea_ohm: float = 0.0    # Reactor resistance per step [Ohm]
    gparac_us: float = 0.0   # Parallel conductance per step [µS]
    
    # High-pass filter parameters
    c1_uf: float = 0.0       # Series capacitor C1 [µF]
    c2_uf: float = 0.0       # Parallel capacitor C2 [µF]
    rpara_ohm: float = 0.0   # Parallel resistance [Ohm]
    
    # System frequency
    f_sys: float = 50.0      # System frequency [Hz]
    
    def __post_init__(self):
        """Calculate admittance in Siemens based on filter type and parameters."""
        self.admittance = self._calculate_admittance_siemens()
    
    def _calculate_admittance_siemens(self) -> complex:
        """Calculate total admittance in Siemens."""
        omega_sys = 2 * np.pi * self.f_sys
        
        if self.filter_type == ShuntFilterType.R_L_C:
            # Series R-L with parallel C
            # Reactor branch (R + jX)
            if self.rrea_ohm == 0 and self.xrea_ohm == 0:
                Y_rea_step = complex(0, 0)
            else:
                denom = self.rrea_ohm**2 + self.xrea_ohm**2
                Y_rea_step = complex(self.rrea_ohm, -self.xrea_ohm) / denom
            Y_rea_total = Y_rea_step * self.n_rea
            
            # Capacitor branch (jB)
            Y_cap_total = 1j * (self.bcap_us * 1e-6) * self.n_cap
        
        elif self.filter_type == ShuntFilterType.R_L:
            # Series R-L (reactor only)
            denom = self.rrea_ohm**2 + self.xrea_ohm**2
            if denom == 0:
                Y_step = complex(0, 0)
            else:
                Y_step = complex(self.rrea_ohm, -self.xrea_ohm) / denom
            return Y_step * self.n_rea
        
        elif self.filter_type == ShuntFilterType.C:
            # Capacitor bank with optional parallel conductance
            Y_step = (self.gparac_us * 1e-6) + 1j * (self.bcap_us * 1e-6)
            return Y_step * self.n_cap
        
        elif self.filter_type == ShuntFilterType.R_L_C_Rp:
            # Damped filter: Series R-L-C with parallel Rp
            # Similar to R_L_C but with additional parallel resistance
            if self.rrea_ohm == 0 and self.xrea_ohm == 0:
                Y_rea_step = complex(0, 0)
            else:
                denom = self.rrea_ohm**2 + self.xrea_ohm**2
                Y_rea_step = complex(self.rrea_ohm, -self.xrea_ohm) / denom
            Y_rea_total = Y_rea_step * self.n_rea
            
            Y_cap_total = 1j * (self.bcap_us * 1e-6) * self.n_cap
            
            # Parallel resistance
            Y_p = 1 / self.rpara_ohm if self.rpara_ohm != 0 else complex(0, 0)
            
            return Y_rea_total + Y_cap_total + Y_p
        
        elif self.filter_type == ShuntFilterType.R_L_C1_C2_Rp:
            # High-pass filter: Series R-L-C1 with parallel C2 and Rp
            B_C1 = omega_sys * self.c1_uf * 1e-6
            B_C2 = omega_sys * self.c2_uf * 1e-6
            
            # Branch 1: Series R-L-C1
            if B_C1 == 0:
                Z_C1 = complex(0, 0)
            else:
                Z_C1 = -1j / B_C1
            
            Z_branch = complex(self.rrea_ohm, self.xrea_ohm) + Z_C1
            if abs(Z_branch) < 1e-12:
                Y_branch = complex(0, 0)
            else:
                Y_branch = 1 / Z_branch
            
            # Branch 2: Parallel resistor Rp
            Y_p = 1 / self.rpara_ohm if self.rpara_ohm != 0 else complex(0, 0)
            
            # Branch 3: Parallel capacitor C2
            Y_c2 = 1j * B_C2
            
            # Total admittance (all branches in parallel)
            Y_total = Y_branch + Y_p + Y_c2
            
            # Apply capacitor steps (filters usually switch as a unit)
            return Y_total * self.n_cap
        
        else:
            return complex(0, 0)
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        Y_pu = Y_siemens * Z_base = Y_siemens * (V_base^2 / S_base)
        """
        if self.voltage_kv > 0:
            z_base = (self.voltage_kv ** 2) / base_mva
            return self.admittance * z_base
        return self.admittance
