"""
PowerFactory network element extraction.

This module provides functions to extract network elements from PowerFactory
using cubicle-based connectivity.
"""

from ..core.elements import (
    BranchElement, ShuntElement,
    LineBranch, SwitchBranch, TransformerBranch,
    LoadShunt, GeneratorShunt
)


def get_network_elements(app) -> tuple[list[BranchElement], list[ShuntElement]]:
    """
    Extract branch and shunt elements from the active PowerFactory network.
    
    Uses cubicle-based connectivity to determine terminal connections.
    
    Args:
        app: PowerFactory application instance
        
    Returns:
        Tuple of (branches, shunts) lists
    """
    branches: list[BranchElement] = []
    shunts: list[ShuntElement] = []

    # --- Branch elements: Lines (ElmLne) ---
    pf_lines = app.GetCalcRelevantObjects("*.ElmLne", 0, 0, 0)
    for line in pf_lines:
        if line.outserv == 1:
            continue
        
        from_bus = line.GetCubicle(0).cterm
        to_bus = line.GetCubicle(1).cterm
        
        # Extract line parameters
        R = line.R1  # Total resistance (ohms)
        X = line.X1  # Total reactance (ohms)
        B = line.B1  # Total susceptance (µS)
        
        branches.append(LineBranch(
            pf_object=line,
            name=line.loc_name,
            from_bus_name=from_bus.loc_name,
            to_bus_name=to_bus.loc_name,
            voltage_kv=from_bus.uknom,
            resistance_ohm=R,
            reactance_ohm=X,
            susceptance_us=B
        ))

    # --- Branch elements: Switches/Couplers (ElmCoup) ---
    pf_switches = app.GetCalcRelevantObjects("*.ElmCoup", 0, 0, 0)
    for switch in pf_switches:
        if switch.outserv == 1:
            continue
        
        from_bus = switch.GetCubicle(0).cterm
        to_bus = switch.GetCubicle(1).cterm
        is_closed = not (hasattr(switch, 'on_off') and switch.on_off == 0)
        
        branches.append(SwitchBranch(
            pf_object=switch,
            name=switch.loc_name,
            from_bus_name=from_bus.loc_name,
            to_bus_name=to_bus.loc_name,
            voltage_kv=from_bus.uknom,
            is_closed=is_closed
        ))

    # --- Branch elements: Two-winding Transformers (ElmTr2) ---
    pf_trafos = app.GetCalcRelevantObjects("*.ElmTr2", 0, 0, 0)
    for trafo in pf_trafos:
        if trafo.outserv == 1:
            continue
        
        # Get terminals via cubicles (HV = cubicle 0, LV = cubicle 1)
        hv_bus = trafo.GetCubicle(0).cterm
        lv_bus = trafo.GetCubicle(1).cterm
        
        # Get transformer type data
        pf_type = trafo.GetAttribute("typ_id")
        if pf_type is None:
            continue
        
        # Rated values from type
        rated_mva = pf_type.strn if hasattr(pf_type, 'strn') else 0.0
        hv_kv = pf_type.utrn_h if hasattr(pf_type, 'utrn_h') else 0.0
        lv_kv = pf_type.utrn_l if hasattr(pf_type, 'utrn_l') else 0.0
        
        # Impedance from type (uk = short-circuit voltage %, ur = resistive part %)
        uk_percent = pf_type.uktr if hasattr(pf_type, 'uktr') else 0.0
        ur_percent = pf_type.uktrr if hasattr(pf_type, 'uktrr') else 0.0
        
        # Convert to per-unit on transformer base
        x_pu = (uk_percent / 100.0)  # Approximate: X ≈ uk for small R
        r_pu = (ur_percent / 100.0)
        # More accurate: X = sqrt(uk² - ur²)
        if uk_percent > ur_percent:
            x_pu = ((uk_percent ** 2 - ur_percent ** 2) ** 0.5) / 100.0
        
        # Get tap position and calculate tap ratio
        # nntap = current tap position, ntpmn/ntpmx = min/max tap positions
        # dutap = voltage change per tap step (%)
        tap_pos = trafo.nntap if hasattr(trafo, 'nntap') else 0
        tap_neutral = pf_type.ntpm if hasattr(pf_type, 'ntpm') else 0
        dutap = pf_type.dutap if hasattr(pf_type, 'dutap') else 0.0
        
        # Tap ratio: t = 1 + (tap_pos - tap_neutral) * dutap / 100
        tap_ratio = 1.0 + (tap_pos - tap_neutral) * dutap / 100.0
        
        # Determine tap side (0 = HV, 1 = LV)
        # In PowerFactory, tap_side attribute or default to HV
        tap_side = pf_type.tap_side if hasattr(pf_type, 'tap_side') else 0
        
        branches.append(TransformerBranch(
            pf_object=trafo,
            name=trafo.loc_name,
            from_bus_name=hv_bus.loc_name,
            to_bus_name=lv_bus.loc_name,
            voltage_kv=hv_kv,  # Use HV side as reference
            rated_power_mva=rated_mva,
            hv_kv=hv_kv,
            lv_kv=lv_kv,
            resistance_pu=r_pu,
            reactance_pu=x_pu,
            tap_ratio=tap_ratio,
            tap_side=tap_side
        ))

    # --- Shunt elements: Synchronous Generators (ElmSym) ---
    pf_gens = app.GetCalcRelevantObjects("*.ElmSym", 0, 0, 0)
    for gen in pf_gens:
        if gen.outserv == 1:
            continue
        
        bus = gen.GetCubicle(0).cterm
        pf_type = gen.GetAttribute("typ_id")
        
        rated_mva = pf_type.sgn if pf_type and hasattr(pf_type, 'sgn') else 0.0
        rated_kv = pf_type.ugn if pf_type and hasattr(pf_type, 'ugn') else 0.0
        xdss = pf_type.xdss if pf_type and hasattr(pf_type, 'xdss') else 0.0
        
        shunts.append(GeneratorShunt(
            name=gen.loc_name,
            bus_name=bus.loc_name,
            voltage_kv=bus.uknom,
            rated_power_mva=rated_mva,
            rated_voltage_kv=rated_kv,
            xdss_pu=xdss
        ))

    # --- Shunt elements: Loads (ElmLod) ---
    pf_loads = app.GetCalcRelevantObjects("*.ElmLod", 0, 0, 0)
    for load in pf_loads:
        if load.outserv == 1:
            continue
        
        bus = load.GetCubicle(0).cterm
        
        shunts.append(LoadShunt(
            name=load.loc_name,
            bus_name=bus.loc_name,
            voltage_kv=bus.uknom,
            p_mw=load.plini,
            q_mvar=load.qlini
        ))

    return branches, shunts
