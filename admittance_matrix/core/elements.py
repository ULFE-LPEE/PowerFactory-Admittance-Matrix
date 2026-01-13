"""
Element definitions for power system network components.

This module contains the base classes and implementations for:
- Branch elements (lines, switches)
- Shunt elements (loads, generators)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np


class TapChangerType(Enum):
    """Tap changer type enumeration matching PowerFactory tapchtype."""
    RATIO_ASYM = 0      # Ratio/Asymmetric phase shifter
    IDEAL_PHASE = 1     # Ideal phase shifter
    SYM_PHASE = 2       # Symmetric phase shifter


class LoadModelType(Enum):
    """Load model type enumeration."""
    CONSTANT_IMPEDANCE = 0  # Z = const, P,Q vary with V^2
    CONSTANT_POWER = 1      # P,Q = const, Z varies with V^2


@dataclass
class TapChanger(ABC):
    """
    Abstract base class for transformer tap changers.
    
    Attributes:
        tap_side: Tap side (0 = HV, 1 = LV)
        nntap0: Neutral tap position
        ntpmn: Minimum tap position
        ntpmx: Maximum tap position
    """
    tap_side: int = 0
    nntap0: int = 0
    ntpmn: int = 0
    ntpmx: int = 0
    
    @abstractmethod
    def get_complex_tap_ratio(self, tap_pos: int) -> complex:
        """
        Calculate the complex tap ratio for a given tap position.
        
        Args:
            tap_pos: Current tap position
            
        Returns:
            Complex tap ratio t = |t| * exp(j * phi)
        """
        pass
    
    @property
    @abstractmethod
    def tap_type(self) -> TapChangerType:
        """Return the tap changer type."""
        pass


@dataclass
class RatioAsymTapChanger(TapChanger):
    """
    Ratio/Asymmetric phase shifter tap changer.
    
    Used for transformers with voltage magnitude control and 
    optional asymmetric phase shift.
    
    Attributes:
        dutap: Additional voltage per tap in %
        phitr: Phase angle of du in degrees
    """
    dutap: float = 0.0
    phitr: float = 0.0
    
    @property
    def tap_type(self) -> TapChangerType:
        return TapChangerType.RATIO_ASYM
    
    def get_complex_tap_ratio(self, tap_pos: int) -> complex:
        """
        Calculate complex tap ratio for ratio/asymmetric phase shifter.
        
        The voltage change per tap has both magnitude (dutap) and angle (phitr):
        du = dutap * exp(j * phitr)
        t = 1 + n * du / 100
        
        Where n = tap_pos - nntap0 (deviation from neutral)
        """
        n = tap_pos - self.nntap0
        
        # Convert phitr to radians
        phi_rad = math.radians(self.phitr)
        
        # Complex voltage change per tap
        du_complex = (self.dutap / 100.0) * complex(math.cos(phi_rad), math.sin(phi_rad))
        
        # Complex tap ratio
        tap_ratio = 1.0 + n * du_complex
        
        return tap_ratio


@dataclass
class IdealPhaseTapChanger(TapChanger):
    """
    Ideal phase shifter tap changer.
    
    Used for pure phase shifting transformers with no magnitude change.
    
    Attributes:
        dphitap: Additional phase angle per tap in degrees
    """
    dphitap: float = 0.0
    
    @property
    def tap_type(self) -> TapChangerType:
        return TapChangerType.IDEAL_PHASE
    
    def get_complex_tap_ratio(self, tap_pos: int) -> complex:
        """
        Calculate complex tap ratio for ideal phase shifter.
        
        Pure phase shift with unity magnitude:
        t = exp(j * n * dphitap)
        
        Where n = tap_pos - nntap0 (deviation from neutral)
        """
        n = tap_pos - self.nntap0
        
        # Total phase shift in radians
        phi_rad = math.radians(n * self.dphitap)
        
        # Complex tap ratio with unity magnitude
        tap_ratio = complex(math.cos(phi_rad), math.sin(phi_rad))
        
        return tap_ratio


@dataclass
class SymPhaseTapChanger(TapChanger):
    """
    Symmetric phase shifter tap changer.
    
    Used for transformers with symmetric phase shifting capability.
    Similar to ratio/asymmetric but with symmetric voltage change.
    
    Attributes:
        dutap: Additional voltage per tap in %
        phitr: Phase angle of du in degrees
    """
    dutap: float = 0.0
    phitr: float = 0.0
    
    @property
    def tap_type(self) -> TapChangerType:
        return TapChangerType.SYM_PHASE
    
    def get_complex_tap_ratio(self, tap_pos: int) -> complex:
        """
        Calculate complex tap ratio for symmetric phase shifter.
        
        The voltage change per tap has both magnitude (dutap) and angle (phitr):
        du = dutap * exp(j * phitr)
        t = 1 + n * du / 100
        
        Where n = tap_pos - nntap0 (deviation from neutral)
        """
        n = tap_pos - self.nntap0
        
        # Convert phitr to radians
        phi_rad = math.radians(self.phitr)
        
        # Complex voltage change per tap
        du_complex = (self.dutap / 100.0) * complex(math.cos(phi_rad), math.sin(phi_rad))
        
        # Complex tap ratio
        tap_ratio = 1.0 + n * du_complex
        
        return tap_ratio

@dataclass
class BranchElement(ABC):
    """Abstract base class for two-terminal elements."""
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
        # Calculate series admittance for a single line
        # Use small values to avoid numerical instabilities (near-zero impedance)
        r = self.resistance_ohm if self.resistance_ohm != 0 else 1e-12
        x = self.reactance_ohm if self.reactance_ohm != 0 else 1e-12
        self.admittance = 1 / complex(r, x)
        
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
    - Complex tap ratio (t) - includes magnitude and phase shift from tap changer
    - Magnetizing admittance (optional)
    - Number of parallel transformers (ntnum)
    
    Y-matrix for transformer between buses i (HV) and j (LV) with complex tap ratio:
        Y_ii = y / |t|²
        Y_jj = y
        Y_ij = -y / t*  (conjugate of t)
        Y_ji = -y / t
    
    Where y = 1/(R + jX) in per-unit on system base, t is complex tap ratio.
    """
    rated_power_mva: float = 0.0
    hv_kv: float = 0.0  # HV side rated voltage
    lv_kv: float = 0.0  # LV side rated voltage
    resistance_pu: float = 0.0  # R on transformer base (p.u.)
    reactance_pu: float = 0.0   # X on transformer base (p.u.)
    tap_changer: TapChanger | None = None  # Tap changer object
    tap_pos: int = 0             # Current tap position
    magnetizing_admittance: complex = field(default=complex(0, 0))  # Y_m in p.u.
    n_parallel: int = 1  # Number of parallel transformers (ntnum in PowerFactory)
    
    @property
    def tap_ratio(self) -> complex:
        """Get the complex tap ratio from the tap changer."""
        if self.tap_changer is not None:
            return self.tap_changer.get_complex_tap_ratio(self.tap_pos)
        return complex(1.0, 0.0)
    
    @property
    def tap_side(self) -> int:
        """Get the tap side from the tap changer."""
        if self.tap_changer is not None:
            return self.tap_changer.tap_side
        return 0
    
    def __post_init__(self):
        # Calculate series admittance in per-unit on transformer base for single transformer
        # Use small values to avoid numerical instabilities
        r = self.resistance_pu if self.resistance_pu != 0 else 1e-12
        x = self.reactance_pu if self.reactance_pu != 0 else 1e-12
        z_pu_trafo = complex(r, x)
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
        Return Y-matrix contributions for transformer with complex tap ratio.
        
        For tap on HV side (from_bus) with complex tap ratio t:
            Y_ii = y / |t|²    (HV side)
            Y_jj = y           (LV side)  
            Y_ij = -y / t*     (conjugate of t)
            Y_ji = -y / t
            
        For tap on LV side (to_bus) with complex tap ratio t:
            Y_ii = y           (HV side)
            Y_jj = y / |t|²    (LV side)
            Y_ij = -y / t*
            Y_ji = -y / t
        """
        y = self.get_admittance_pu(base_mva) if base_mva else self.admittance
        t = self.tap_ratio  # Complex tap ratio
        t_conj = t.conjugate()
        t_mag_sq = abs(t) ** 2
        
        if self.tap_side == 0:  # Tap on HV (from) side
            Yii = y / t_mag_sq
            Yjj = y
            Yij = -y / t_conj
            Yji = -y / t
        else:  # Tap on LV (to) side
            Yii = y
            Yjj = y / t_mag_sq
            Yij = -y / t_conj
            Yji = -y / t
        
        return (Yii, Yjj, Yij, Yji)


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
            # Use small values to avoid numerical instabilities
            R_ohm = self.resistance_ohm if self.resistance_ohm != 0 else 1e-12
            X_ohm = self.reactance_ohm if self.reactance_ohm != 0 else 1e-12
            Z_ohm = complex(R_ohm, X_ohm)
            self.admittance = 1 / Z_ohm
        else:
            self.admittance = complex(1e-12, 1e-12)
    
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
    
    Calculates pair impedances from short-circuit test data (uk, ukr percentages).
    
    The pair impedances from short-circuit tests (with 3rd winding open) are:
    - Z_HM (uktr3_h): HV-MV impedance (LV open), referred to min(S_H, S_M)
    - Z_ML (uktr3_m): MV-LV impedance (HV open), referred to min(S_M, S_L)
    - Z_LH (uktr3_l): LV-HV impedance (MV open), referred to min(S_L, S_H)
    
    These are converted to star (Y) equivalent impedances:
    - Z_H = (Z_HM + Z_LH - Z_ML) / 2  (HV winding to internal node)
    - Z_M = (Z_HM + Z_ML - Z_LH) / 2  (MV winding to internal node)
    - Z_L = (Z_ML + Z_LH - Z_HM) / 2  (LV winding to internal node)
    
    The 3x3 admittance matrix is obtained by Kron-reducing the internal star node.
    
    Tap ratio effects (magnitude and phase shift) are applied on each winding side.
    
    Local node order: [HV, MV, LV]
    """
    name: str
    hv_bus_name: str
    mv_bus_name: str
    lv_bus_name: str
    base_mva: float = 100.0  # System base MVA
    n_parallel: int = 1  # Number of parallel transformers
    
    # Rated powers for each winding (MVA)
    rated_power_hv_mva: float = 0.0
    rated_power_mv_mva: float = 0.0
    rated_power_lv_mva: float = 0.0
    
    # Rated voltages for each winding (kV)
    hv_kv: float = 0.0
    mv_kv: float = 0.0
    lv_kv: float = 0.0
    
    # Short-circuit voltages (uk) in % for each pair
    # uktr3_h: HV-MV pair (referred to min(S_H, S_M))
    # uktr3_m: MV-LV pair (referred to min(S_M, S_L))
    # uktr3_l: LV-HV pair (referred to min(S_L, S_H))
    uk_hm_percent: float = 0.0
    uk_ml_percent: float = 0.0
    uk_lh_percent: float = 0.0
    
    # Real parts of short-circuit voltages (ukr) in % for each pair
    ukr_hm_percent: float = 0.0
    ukr_ml_percent: float = 0.0
    ukr_lh_percent: float = 0.0
    
    # Tap changers for each winding (only HV typically used)
    tap_changer_hv: TapChanger | None = None
    tap_pos_hv: int = 0
    tap_changer_mv: TapChanger | None = None
    tap_pos_mv: int = 0
    tap_changer_lv: TapChanger | None = None
    tap_pos_lv: int = 0
    
    @property
    def tap_ratio_hv(self) -> complex:
        """Get the complex tap ratio for HV side."""
        if self.tap_changer_hv is not None:
            return self.tap_changer_hv.get_complex_tap_ratio(self.tap_pos_hv)
        return complex(1.0, 0.0)
    
    @property
    def tap_ratio_mv(self) -> complex:
        """Get the complex tap ratio for MV side."""
        if self.tap_changer_mv is not None:
            return self.tap_changer_mv.get_complex_tap_ratio(self.tap_pos_mv)
        return complex(1.0, 0.0)
    
    @property
    def tap_ratio_lv(self) -> complex:
        """Get the complex tap ratio for LV side."""
        if self.tap_changer_lv is not None:
            return self.tap_changer_lv.get_complex_tap_ratio(self.tap_pos_lv)
        return complex(1.0, 0.0)
    
    def _calculate_pair_impedance_pu(self, uk_percent: float, ukr_percent: float, 
                                      s_ref_mva: float) -> complex:
        """
        Calculate pair impedance in pu on system base from short-circuit test data.
        
        Args:
            uk_percent: Short-circuit voltage magnitude in %
            ukr_percent: Resistive part of short-circuit voltage in %
            s_ref_mva: Reference power (min of two winding ratings) in MVA
            
        Returns:
            Complex impedance Z = R + jX in pu on system base
        """
        if s_ref_mva <= 0 or uk_percent <= 0:
            return complex(1e-12, 1e-12)  # Return small value to avoid singularities
        
        # Calculate R and X in pu on the reference (transformer) base
        r_pu_ref = ukr_percent / 100.0
        
        # X = sqrt(uk² - ukr²) / 100
        uk_sq = uk_percent ** 2
        ukr_sq = ukr_percent ** 2
        if uk_sq > ukr_sq:
            x_pu_ref = math.sqrt(uk_sq - ukr_sq) / 100.0
        else:
            x_pu_ref = uk_percent / 100.0  # Fallback: assume X ≈ uk if ukr > uk
        
        # Convert from transformer base to system base
        # Z_pu conversion: Z_pu_new = Z_pu_old * (S_new / S_old)
        # Where S_old = s_ref_mva (transformer base) and S_new = self.base_mva (system base)
        z_pu_ref = complex(r_pu_ref, x_pu_ref)
        z_pu_sys = z_pu_ref * (self.base_mva / s_ref_mva)
        
        return z_pu_sys
    
    def Z_hm(self) -> complex:
        """Z between HV and MV in pu on system base (LV open)."""
        s_ref = min(self.rated_power_hv_mva, self.rated_power_mv_mva)
        return self._calculate_pair_impedance_pu(
            self.uk_hm_percent, self.ukr_hm_percent, s_ref
        )
    
    def Z_ml(self) -> complex:
        """Z between MV and LV in pu on system base (HV open)."""
        s_ref = min(self.rated_power_mv_mva, self.rated_power_lv_mva)
        return self._calculate_pair_impedance_pu(
            self.uk_ml_percent, self.ukr_ml_percent, s_ref
        )
    
    def Z_lh(self) -> complex:
        """Z between LV and HV in pu on system base (MV open)."""
        s_ref = min(self.rated_power_lv_mva, self.rated_power_hv_mva)
        return self._calculate_pair_impedance_pu(
            self.uk_lh_percent, self.ukr_lh_percent, s_ref
        )
    
    def get_star_impedances(self) -> tuple[complex, complex, complex]:
        """
        Convert pair impedances to star (Y) equivalent impedances.
        
        The measured pair impedances satisfy:
            Z_hm = Z_H + Z_M
            Z_ml = Z_M + Z_L
            Z_lh = Z_L + Z_H
        
        Solving for star impedances:
            Z_H = (Z_hm + Z_lh - Z_ml) / 2
            Z_M = (Z_hm + Z_ml - Z_lh) / 2
            Z_L = (Z_ml + Z_lh - Z_hm) / 2
        
        Returns:
            Tuple of (Z_H, Z_M, Z_L) in pu on system base
        """
        Z_hm = self.Z_hm()
        Z_ml = self.Z_ml()
        Z_lh = self.Z_lh()
        
        Z_H = (Z_hm + Z_lh - Z_ml) / 2.0
        Z_M = (Z_hm + Z_ml - Z_lh) / 2.0
        Z_L = (Z_ml + Z_lh - Z_hm) / 2.0
        
        return Z_H, Z_M, Z_L
    
    def _safe_inv(self, Z: complex, eps: float = 1e-12, max_admittance: float = 1e6) -> complex:
        """
        Safely invert impedance to admittance.
        
        If |Z| < eps (near short circuit), cap admittance to max_admittance.
        If |Z| is very large (near open circuit), admittance approaches 0.
        
        Args:
            Z: Impedance to invert
            eps: Threshold for near-zero impedance
            max_admittance: Maximum admittance magnitude to return
            
        Returns:
            Admittance Y = 1/Z, capped if necessary
        """
        if abs(Z) < eps:
            # Near short circuit - cap to large admittance
            return complex(max_admittance, 0.0)
        Y = 1.0 / Z
        if abs(Y) > max_admittance:
            # Cap magnitude while preserving angle
            return Y * (max_admittance / abs(Y))
        return Y
    
    def get_star_admittances(self) -> tuple[complex, complex, complex]:
        """
        Get star admittances from star impedances, scaled by n_parallel.
        
        Returns:
            Tuple of (Y_H, Y_M, Y_L) in pu on system base
        """
        Z_H, Z_M, Z_L = self.get_star_impedances()
        
        Y_H = self._safe_inv(Z_H) * self.n_parallel
        Y_M = self._safe_inv(Z_M) * self.n_parallel
        Y_L = self._safe_inv(Z_L) * self.n_parallel
        
        return Y_H, Y_M, Y_L
    
    def get_local_admittance_matrix(self) -> tuple[list[list[complex]], list[str]]:
        """
        Returns 3×3 pu nodal admittance matrix in order [HV, MV, LV].
        
        Uses star-to-Kron-reduction to obtain the 3-port equivalent:
        
        For star admittances y_H, y_M, y_L connected to internal node,
        with y_s = y_H + y_M + y_L:
        
        Off-diagonal: Y_ij = -(y_i * y_j) / y_s
        Diagonal: Y_ii = y_i - (y_i^2) / y_s = y_i * (1 - y_i/y_s)
        
        Then applies complex tap ratios for each winding:
        - For winding i with tap t_i, the admittance terms are modified as:
          Y_ii → Y_ii / |t_i|^2
          Y_ij → Y_ij / (t_i* · t_j)  (conjugate of t_i times t_j)
        
        Returns:
            Tuple of (3x3 matrix as nested list, [hv_bus, mv_bus, lv_bus])
        """
        y_H, y_M, y_L = self.get_star_admittances()
        y_s = y_H + y_M + y_L
        
        # Avoid division by zero if all admittances are zero
        if abs(y_s) < 1e-12:
            matrix = [
                [complex(0, 0), complex(0, 0), complex(0, 0)],
                [complex(0, 0), complex(0, 0), complex(0, 0)],
                [complex(0, 0), complex(0, 0), complex(0, 0)]
            ]
            return matrix, [self.hv_bus_name, self.mv_bus_name, self.lv_bus_name]
        
        # Kron reduction: Y_ii = y_i * (y_s - y_i) / y_s = y_i - y_i^2/y_s
        # Off-diagonal: Y_ij = -y_i * y_j / y_s
        Y_00 = y_H - (y_H * y_H) / y_s  # HV diagonal
        Y_11 = y_M - (y_M * y_M) / y_s  # MV diagonal
        Y_22 = y_L - (y_L * y_L) / y_s  # LV diagonal
        
        Y_01_base = -(y_H * y_M) / y_s  # HV-MV off-diagonal
        Y_02_base = -(y_H * y_L) / y_s  # HV-LV off-diagonal
        Y_12_base = -(y_M * y_L) / y_s  # MV-LV off-diagonal
        
        # Apply tap ratios
        t_H = self.tap_ratio_hv
        t_M = self.tap_ratio_mv
        t_L = self.tap_ratio_lv
        
        t_H_conj = t_H.conjugate()
        t_M_conj = t_M.conjugate()
        t_L_conj = t_L.conjugate()
        
        t_H_mag_sq = abs(t_H) ** 2
        t_M_mag_sq = abs(t_M) ** 2
        t_L_mag_sq = abs(t_L) ** 2
        
        # Diagonal elements: Y_ii / |t_i|^2
        Y_00 = Y_00 / t_H_mag_sq
        Y_11 = Y_11 / t_M_mag_sq
        Y_22 = Y_22 / t_L_mag_sq
        
        # Off-diagonal elements: Y_ij / (t_i* · t_j) and Y_ji / (t_j* · t_i)
        Y_01 = Y_01_base / (t_H_conj * t_M)
        Y_10 = Y_01_base / (t_M_conj * t_H)
        
        Y_02 = Y_02_base / (t_H_conj * t_L)
        Y_20 = Y_02_base / (t_L_conj * t_H)
        
        Y_12 = Y_12_base / (t_M_conj * t_L)
        Y_21 = Y_12_base / (t_L_conj * t_M)
        
        matrix = [
            [Y_00, Y_01, Y_02],
            [Y_10, Y_11, Y_12],
            [Y_20, Y_21, Y_22]
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
        matrix, bus_names = self.get_local_admittance_matrix()
        
        # Extract contributions from the 3x3 matrix
        # Each pair contributes to both diagonals and off-diagonals
        contributions = {}
        
        # For a proper 3-port, we return the full matrix entries
        # This is different from simple 2-port branches
        for i in range(3):
            for j in range(3):
                if i != j:
                    bus_i = bus_names[i]
                    bus_j = bus_names[j]
                    if (bus_i, bus_j) not in contributions:
                        # Store: (Yii contribution, Yjj contribution, Yij, Yji)
                        contributions[(bus_i, bus_j)] = (
                            matrix[i][i],  # Not used directly, but for reference
                            matrix[j][j],  # Not used directly, but for reference
                            matrix[i][j],
                            matrix[j][i]
                        )
        
        return contributions
    
    @property
    def bus_names(self) -> list[str]:
        """List of connected bus names [HV, MV, LV]."""
        return [self.hv_bus_name, self.mv_bus_name, self.lv_bus_name]


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
    """
    Load element - supports constant impedance and constant power models.
    
    Load Model Types:
    - CONSTANT_IMPEDANCE: Z = const, P and Q vary with V^2
      For stability analysis, use update_admittance_with_lf_voltage() after
      running load flow to recalculate admittance using actual bus voltage.
    - CONSTANT_POWER: P and Q remain constant regardless of voltage.
      The admittance is calculated at nominal voltage and does not change.
    """
    p_mw: float = 0.0
    q_mvar: float = 0.0
    lf_voltage_kv: float = field(default=1.0, repr=False)  # Load flow voltage (set after LF)
    load_model: LoadModelType = LoadModelType.CONSTANT_IMPEDANCE
    
    def __post_init__(self):
        # Load: P + jQ -> Y = (P - jQ) / |V|^2
        # Initially use nominal voltage
        if self.voltage_kv > 0:
            self.admittance = complex(self.p_mw, -self.q_mvar) / (1 ** 2)
        else:
            self.admittance = complex(1e-12, 1e-12)
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        Y_pu = Y_siemens * Z_base = Y_siemens * (V_base^2 / S_base)
        
        For CONSTANT_IMPEDANCE: Uses load flow voltage if available.
        For CONSTANT_POWER: Always uses nominal voltage (admittance doesn't change with V).
        """
        return self.admittance / base_mva
    
    def update_admittance_with_lf_voltage(self) -> None:
        """
        Recalculate admittance using load flow voltage.
        
        For CONSTANT_IMPEDANCE model:
            Call this after running load flow to get accurate constant impedance
            model for stability analysis. Uses lf_voltage_kv if set, otherwise
            falls back to nominal voltage_kv.
        
        For CONSTANT_POWER model:
            This method has no effect - admittance is always calculated at nominal
            voltage since P and Q are constant regardless of actual voltage.
        """
        # For constant power model, admittance doesn't change with voltage
        if self.load_model == LoadModelType.CONSTANT_POWER:
            return
        
        # Constant impedance: recalculate based on LF voltage
        v_kv = self.lf_voltage_kv if self.lf_voltage_kv > 0 else self.voltage_kv

        if v_kv > 0:
            self.admittance = complex(self.p_mw, -self.q_mvar) / (v_kv ** 2)
        else:
            self.admittance = complex(1e-12, 1e-12)
    
    def set_lf_voltage(self, voltage_kv: float) -> None:
        """Set the load flow voltage and recalculate admittance (if constant impedance model)."""
        self.lf_voltage_kv = voltage_kv
        self.update_admittance_with_lf_voltage()


@dataclass
class GeneratorShunt(ShuntElement):
    """Synchronous generator - transient/sub-transient reactance model."""
    rated_power_mva: float = 0.0
    rated_voltage_kv: float = 0.0
    z_pu: float = 0.0  # Sub-transient reactance on generator base
    
    def __post_init__(self):
        """Calculate generator admittance behind sub-transient reactance."""
        # Calculate impedance in ohms
        z_base = (self.rated_voltage_kv ** 2) / self.rated_power_mva
        z_ohms = self.z_pu * z_base
        self.admittance = 1 / z_ohms


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
            # Use small non-zero value to avoid numerical instabilities in matrix reduction
            self.admittance = complex(1e-12, -1e-12)


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

class ShuntFilterType(Enum):
    """Shunt filter/capacitor layout types matching PowerFactory ElmShnt."""
    R_L_C = 0       # Series R-L-C (tuned filter)
    R_L = 1         # Series R-L (reactor only)
    C = 2           # Capacitor only
    R_L_C_Rp = 3    # Series R-L-C with parallel R (damped filter)
    R_L_C1_C2_Rp = 4  # High-pass filter: Series R-L-C1 with parallel C2 and Rp


@dataclass
class ShuntFilterShunt(ShuntElement):
    """
    Shunt filter/capacitor element (ElmShnt).
    
    For steady-state power flow and stability analysis, the filter is modeled
    using its actual reactive power output (Qact) which PowerFactory calculates
    based on the current operating point (voltage, frequency, active steps).
    
    This simplifies the model while maintaining accuracy because:
    - Qact already accounts for all active steps
    - Qact reflects the actual operating point at system frequency
    - For Y-matrix purposes, we need the equivalent susceptance at fundamental frequency
    
    Admittance calculation:
        Y = (P - jQ) / V²  [Siemens]
        For purely reactive: Y = -jQ / V² = jB
        
    Sign convention (generator convention):
        Q > 0: Capacitive (generating reactive power, positive susceptance)
        Q < 0: Inductive (absorbing reactive power, negative susceptance)
    
    Supported filter types:
    - Type 0 (R_L_C): Series R-L-C tuned filter
    - Type 1 (R_L): Series R-L shunt reactor
    - Type 2 (C): Capacitor bank
    - Type 3 (R_L_C_Rp): Damped filter
    - Type 4 (R_L_C1_C2_Rp): High-pass filter
    """
    filter_type: ShuntFilterType = ShuntFilterType.C
    
    # Actual reactive power output (from PowerFactory)
    q_mvar: float = 0.0  # Actual reactive power in Mvar (positive = capacitive)
    p_mw: float = 0.0    # Actual active power in MW (losses, usually small)
    
    # Controller parameters (for reference)
    ncapx: int = 1       # Maximum number of capacitor steps
    ncapa: int = 1       # Active number of capacitor steps
    nreax: int = 1       # Maximum number of reactor steps  
    nreaa: int = 1       # Active number of reactor steps
    
    # Design parameters (for reference/detailed modeling)
    qtotn_mvar: float = 0.0  # Rated reactive power per step [Mvar] (type 0)
    qrean_mvar: float = 0.0  # Rated reactive power L per step [Mvar] (type 1)
    fres_hz: float = 0.0     # Resonant frequency [Hz]
    quality_factor: float = 0.0  # Quality factor at resonant/nominal frequency
    
    # Layout parameters per step (for detailed modeling if needed)
    bcap_us: float = 0.0     # Capacitor susceptance per step [µS]
    xrea_ohm: float = 0.0    # Reactor reactance per step [Ohm]
    rrea_ohm: float = 0.0    # Reactor resistance per step [Ohm]
    
    def __post_init__(self):
        """
        Calculate admittance from actual power output.
        
        Y = (P - jQ) / V²  [Siemens]
        """
        if self.voltage_kv > 0:
            v_sq = self.voltage_kv ** 2  # kV² = MW/S = Mvar/S
            # Complex power S = P + jQ, admittance Y = S* / |V|² = (P - jQ) / V²
            self.admittance = complex(self.p_mw, -self.q_mvar) / v_sq
        else:
            self.admittance = complex(0, 0)
    
    def get_admittance_pu(self, base_mva: float = 100.0) -> complex:
        """
        Get admittance in per-unit on system base.
        
        Since admittance is already calculated in Siemens from MW/Mvar and kV,
        we just need to scale to per-unit:
        Y_pu = Y_siemens * Z_base = Y_siemens * (V_base² / S_base)
        """
        if self.voltage_kv > 0:
            z_base = (self.voltage_kv ** 2) / base_mva
            return self.admittance * z_base
        return self.admittance
    
    def get_detailed_admittance_siemens(self) -> complex:
        """
        Calculate admittance from detailed layout parameters (R, L, C).
        
        Use this method if you need the impedance-based model instead of
        the power-based model. This is more accurate for frequency-dependent
        analysis but requires all layout parameters to be available.
        
        Returns:
            Complex admittance in Siemens
        """
        if self.filter_type == ShuntFilterType.R_L_C:
            # Series R-L-C tuned filter
            # At system frequency: Z = R + j(X_L - X_C)
            R = self.rrea_ohm * self.ncapa
            X_L = self.xrea_ohm * self.ncapa
            
            # X_C = 1/B_C where B_C is susceptance
            B_C = self.bcap_us * 1e-6 * self.ncapa
            X_C = 1 / B_C if B_C > 0 else 0.0
            
            Z_total = complex(R, X_L - X_C)
            if abs(Z_total) < 1e-12:
                return complex(0, 0)
            return 1 / Z_total
        
        elif self.filter_type == ShuntFilterType.R_L:
            # Series R-L reactor
            R = self.rrea_ohm * self.nreaa
            X = self.xrea_ohm * self.nreaa
            
            if R == 0 and X == 0:
                return complex(0, 0)
            
            Z = complex(R, X)
            return 1 / Z
        
        elif self.filter_type == ShuntFilterType.C:
            # Pure capacitor bank
            B = self.bcap_us * 1e-6 * self.ncapa
            return complex(0, B)
        
        else:
            # For other types, fall back to power-based calculation
            return self.admittance
