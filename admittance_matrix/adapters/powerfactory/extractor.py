"""
PowerFactory network element extraction.

This module provides functions to extract network elements from PowerFactory
using cubicle-based connectivity.
"""

import logging
from typing import List
import powerfactory as pf

from .naming import get_bus_full_name
from ...core.elements import (
    BranchElement, ShuntElement,
    LineBranch, SwitchBranch, TransformerBranch, Transformer3WBranch,
    CommonImpedanceBranch, SeriesReactorBranch,
    LoadShunt, GeneratorShunt, ExternalGridShunt, VoltageSourceShunt,
    ShuntFilterShunt, ShuntFilterType
)

logger = logging.getLogger(__name__)


def get_network_elements(app) -> tuple[list[BranchElement], list[ShuntElement], list[Transformer3WBranch]]:
    """
    Extract branch and shunt elements from the active PowerFactory network.
    
    Uses cubicle-based connectivity to determine terminal connections.
    
    Args:
        app: PowerFactory application instance
        
    Returns:
        Tuple of (branches, shunts, transformers_3w) lists
    """
    branches: list[BranchElement] = []
    shunts: list[ShuntElement] = []
    transformers_3w: list[Transformer3WBranch] = []

    # --- Branch elements: Lines (ElmLne) ---
    pf_lines: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmLne", 0, 0, 0)
    for line in pf_lines:
        if line.outserv == 1:
            continue
        
        # Check if element is energized
        if line.IsEnergized() != 1:
            continue
        
        try:
            cub0 = line.GetCubicle(0)
            cub1 = line.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Line '{line.loc_name}': Missing cubicle(s), skipping")
                continue
            from_bus = cub0.cterm
            to_bus = cub1.cterm
            if from_bus is None or to_bus is None:
                logger.info(f" Line '{line.loc_name}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Line '{line.loc_name}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Line '{line.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Extract line parameters
        R = line.R1  # Total resistance (ohms)
        X = line.X1  # Total reactance (ohms)
        B = line.B1  # Total susceptance (µS)
        n_parallel = getattr(line, 'nlnum', 1) or 1  # Number of parallel systems
        
        branches.append(LineBranch(
            pf_object=line,
            name=line.loc_name,
            from_bus_name=get_bus_full_name(from_bus),
            to_bus_name=get_bus_full_name(to_bus),
            voltage_kv=from_bus.uknom,
            resistance_ohm=R,
            reactance_ohm=X,
            susceptance_us=B,
            n_parallel=1 # I think we read this from the R, X and B directly
        ))

    # --- Branch elements: Switches/Couplers (ElmCoup) ---
    pf_switches: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmCoup", 0, 0, 0)
    for switch in pf_switches:
        if switch.outserv == 1:
            continue
        
        # Check if element is energized
        if switch.IsEnergized() != 1:
            continue
        
        try:
            cub0 = switch.GetCubicle(0)
            cub1 = switch.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Switch '{switch.loc_name}': Missing cubicle(s), skipping")
                continue
            from_bus = cub0.cterm
            to_bus = cub1.cterm
            if from_bus is None or to_bus is None:
                logger.info(f" Switch '{switch.loc_name}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                continue
        except Exception as e:
            logger.warning(f" Switch '{switch.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        is_closed = not (hasattr(switch, 'on_off') and switch.on_off == 0)
        
        branches.append(SwitchBranch(
            pf_object=switch,
            name=switch.loc_name,
            from_bus_name=get_bus_full_name(from_bus),
            to_bus_name=get_bus_full_name(to_bus),
            voltage_kv=from_bus.uknom,
            is_closed=is_closed
        ))

    # --- Branch elements: Two-winding Transformers (ElmTr2) ---
    pf_trafos: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmTr2", 0, 0, 0)
    for trafo in pf_trafos:
        if trafo.outserv == 1:
            continue
        
        # Check if element is energized
        if trafo.IsEnergized() != 1:
            continue
        
        # Get terminals via cubicles (HV = cubicle 0, LV = cubicle 1)
        try:
            cub0 = trafo.GetCubicle(0)
            cub1 = trafo.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Transformer '{trafo.loc_name}': Missing cubicle(s), skipping")
                continue
            hv_bus = cub0.cterm
            lv_bus = cub1.cterm
            if hv_bus is None or lv_bus is None:
                logger.info(f" Transformer '{trafo.loc_name}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if hv_bus.IsEnergized() != 1 or lv_bus.IsEnergized() != 1:
                logger.info(f" Transformer '{trafo.loc_name}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Transformer '{trafo.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get transformer type data
        pf_type = trafo.GetAttribute("typ_id")
        if pf_type is None:
            logger.warning(f" Transformer '{trafo.loc_name}': No type data (typ_id is None), skipping")
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
        tap_pos = trafo.nntap if hasattr(trafo, 'nntap') else 0
        tap_neutral = pf_type.ntpm if hasattr(pf_type, 'ntpm') else 0
        dutap = pf_type.dutap if hasattr(pf_type, 'dutap') else 0.0
        
        # Tap ratio: t = 1 + (tap_pos - tap_neutral) * dutap / 100
        tap_ratio = 1.0 + (tap_pos - tap_neutral) * dutap / 100.0
        
        # Determine tap side (0 = HV, 1 = LV)
        tap_side = pf_type.tap_side if hasattr(pf_type, 'tap_side') else 0
        
        # Number of parallel transformers
        n_parallel = getattr(trafo, 'ntnum', 1) or 1
        
        branches.append(TransformerBranch(
            pf_object=trafo,
            name=trafo.loc_name,
            from_bus_name=get_bus_full_name(hv_bus),
            to_bus_name=get_bus_full_name(lv_bus),
            voltage_kv=hv_kv,  # Use HV side as reference
            rated_power_mva=rated_mva,
            hv_kv=hv_kv,
            lv_kv=lv_kv,
            resistance_pu=r_pu,
            reactance_pu=x_pu,
            tap_ratio=tap_ratio,
            tap_side=tap_side,
            n_parallel=n_parallel
        ))

    # --- Branch elements: Common Impedances (ElmZpu) ---
    pf_zpu: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmZpu", 0, 0, 0)
    for zpu in pf_zpu:
        if zpu.outserv == 1:
            continue
        
        # Check if element is energized
        if zpu.IsEnergized() != 1:
            continue
        
        # Get terminals via cubicles (side 0 and side 1)
        try:
            cub0 = zpu.GetCubicle(0)
            cub1 = zpu.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Common Impedance '{zpu.loc_name}': Missing cubicle(s), skipping")
                continue
            from_bus = cub0.cterm
            to_bus = cub1.cterm
            if from_bus is None or to_bus is None:
                logger.info(f" Common Impedance '{zpu.loc_name}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Common Impedance '{zpu.loc_name}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Common Impedance '{zpu.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get voltage levels from terminals
        hv_kv = from_bus.uknom
        lv_kv = to_bus.uknom
        
        # Get impedance from PowerFactory (returns [R, X] in Ohms at specified voltage)
        imp_pf = zpu.GetImpedance(hv_kv)
        if (imp_pf[0] == 1):
            logger.warning(f" Common Impedance '{zpu.loc_name}': Error obtaining impedance, skipping")
            continue
        R_ohm = imp_pf[1]
        X_ohm = imp_pf[2]
        
        # Get rated power (if available)
        rated_mva = getattr(zpu, 'Sn', 0.0) or 0.0
        
        branches.append(CommonImpedanceBranch(
            pf_object=zpu,
            name=zpu.loc_name,
            from_bus_name=get_bus_full_name(from_bus),
            to_bus_name=get_bus_full_name(to_bus),
            voltage_kv=hv_kv,
            resistance_ohm=R_ohm,
            reactance_ohm=X_ohm,
            hv_kv=hv_kv,
            lv_kv=lv_kv,
            rated_power_mva=rated_mva,
        ))

    # --- Branch elements: Series Reactors (ElmSind) ---
    pf_sind: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmSind", 0, 0, 0)
    for sind in pf_sind:
        if sind.outserv == 1:
            continue
        
        # Check if element is energized
        if sind.IsEnergized() != 1:
            continue
        
        # Get terminals via cubicles (side 0 and side 1)
        try:
            cub0 = sind.GetCubicle(0)
            cub1 = sind.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Series Reactor '{sind.loc_name}': Missing cubicle(s), skipping")
                continue
            from_bus = cub0.cterm
            to_bus = cub1.cterm
            if from_bus is None or to_bus is None:
                logger.info(f" Series Reactor '{sind.loc_name}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Series Reactor '{sind.loc_name}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Series Reactor '{sind.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get voltage level from terminal
        voltage_kv = from_bus.uknom
        
        # Get impedance from PowerFactory
        imp_pf = sind.GetImpedance(voltage_kv)
        if (imp_pf[0] == 1):
            logger.warning(f" Series Reactor '{sind.loc_name}': Error obtaining impedance, skipping")
            continue
        R_ohm = imp_pf[1]
        X_ohm = imp_pf[2]
        
        # Get rated power (if available)
        rated_mva = getattr(sind, 'Sn', 0.0) or 0.0
        
        branches.append(SeriesReactorBranch(
            pf_object=sind,
            name=sind.loc_name,
            from_bus_name=get_bus_full_name(from_bus),
            to_bus_name=get_bus_full_name(to_bus),
            voltage_kv=voltage_kv,
            resistance_ohm=R_ohm,
            reactance_ohm=X_ohm,
            rated_power_mva=rated_mva,
        ))

    # --- Shunt elements: Synchronous Generators (ElmSym) ---
    pf_gens: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmSym", 0, 0, 0)
    for gen in pf_gens:
        if gen.outserv == 1:
            continue
        
        # Check if element is energized
        if gen.IsEnergized() != 1:
            continue
        
        try:
            cub0 = gen.GetCubicle(0)
            if cub0 is None:
                logger.info(f" Generator '{gen.loc_name}': Missing cubicle, skipping")
                continue
            bus = cub0.cterm
            if bus is None:
                logger.info(f" Generator '{gen.loc_name}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" Generator '{gen.loc_name}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Generator '{gen.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        pf_type = gen.GetAttribute("typ_id")
        
        rated_mva = pf_type.sgn if pf_type and hasattr(pf_type, 'sgn') else 0.0
        rated_kv = pf_type.ugn if pf_type and hasattr(pf_type, 'ugn') else 0.0
        xdss = pf_type.xdss if pf_type and hasattr(pf_type, 'xdss') else 0.0
        
        shunts.append(GeneratorShunt(
            pf_object=gen,
            name=gen.loc_name,
            bus_name=get_bus_full_name(bus),
            voltage_kv=bus.uknom,
            rated_power_mva=rated_mva,
            rated_voltage_kv=rated_kv,
            xdss_pu=xdss
        ))

    # --- Shunt elements: Loads (ElmLod) ---
    pf_loads: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmLod", 0, 0, 0)
    for load in pf_loads:
        if load.outserv == 1:
            continue
        
        # Check if element is energized
        if load.IsEnergized() != 1:
            continue
        
        try:
            cub0 = load.GetCubicle(0)
            if cub0 is None:
                logger.info(f" Load '{load.loc_name}': Missing cubicle, skipping")
                continue
            bus = cub0.cterm
            if bus is None:
                logger.info(f" Load '{load.loc_name}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" Load '{load.loc_name}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Load '{load.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        shunts.append(LoadShunt(
            pf_object=load,
            name=load.loc_name,
            bus_name=get_bus_full_name(bus),
            voltage_kv=bus.uknom,
            p_mw=load.plini*load.scale0,
            q_mvar=load.qlini*load.scale0
        ))

    # --- Shunt elements: External Grids (ElmXnet) ---
    pf_xnets: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmXnet", 0, 0, 0)
    for xnet in pf_xnets:
        if xnet.outserv == 1:
            continue
        
        # Check if element is energized
        if xnet.IsEnergized() != 1:
            continue
        
        try:
            cub0 = xnet.GetCubicle(0)
            if cub0 is None:
                logger.info(f" External Grid '{xnet.loc_name}': Missing cubicle, skipping")
                continue
            bus = cub0.cterm
            if bus is None:
                logger.info(f" External Grid '{xnet.loc_name}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" External Grid '{xnet.loc_name}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" External Grid '{xnet.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get short-circuit parameters
        s_sc = xnet.snss if hasattr(xnet, 'snss') else 0.0
        c_factor = xnet.cfac if hasattr(xnet, 'cfac') else 1.0
        r_x_ratio = xnet.rntxn if hasattr(xnet, 'rntxn') else 0.1
        
        shunts.append(ExternalGridShunt(
            pf_object=xnet,
            name=xnet.loc_name,
            bus_name=get_bus_full_name(bus),
            voltage_kv=bus.uknom,
            s_sc_mva=s_sc,
            c_factor=c_factor,
            r_x_ratio=r_x_ratio
        ))

    # --- Shunt elements: AC Voltage Sources (ElmVac) ---
    pf_vacs: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmVac", 0, 0, 0)
    for vac in pf_vacs:
        if vac.outserv == 1:
            continue
        
        # Check if element is energized
        if vac.IsEnergized() != 1:
            continue
        
        try:
            cub0 = vac.GetCubicle(0)
            if cub0 is None:
                logger.info(f" AC Voltage Source '{vac.loc_name}': Missing cubicle, skipping")
                continue
            bus = cub0.cterm
            if bus is None:
                logger.info(f" AC Voltage Source '{vac.loc_name}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" AC Voltage Source '{vac.loc_name}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" AC Voltage Source '{vac.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get R and X values (in ohms)
        r_ohm = vac.R1 if hasattr(vac, 'R1') else 0.0
        x_ohm = vac.X1 if hasattr(vac, 'X1') else 0.0
        
        shunts.append(VoltageSourceShunt(
            pf_object=vac,
            name=vac.loc_name,
            bus_name=get_bus_full_name(bus),
            voltage_kv=bus.uknom,
            resistance_ohm=r_ohm,
            reactance_ohm=x_ohm
        ))

    # --- Shunt elements: Shunt Filters/Capacitors (ElmShnt) ---
    pf_shunts: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmShnt", 0, 0, 0)
    for shnt in pf_shunts:
        if shnt.outserv == 1:
            continue
        
        # Check if element is energized
        if shnt.IsEnergized() != 1:
            continue
        
        try:
            cub0 = shnt.GetCubicle(0)
            if cub0 is None:
                logger.info(f" Shunt Filter '{shnt.loc_name}': Missing cubicle, skipping")
                continue
            bus = cub0.cterm
            if bus is None:
                logger.info(f" Shunt Filter '{shnt.loc_name}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" Shunt Filter '{shnt.loc_name}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" Shunt Filter '{shnt.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get filter type
        shtype = getattr(shnt, 'shtype', 2) or 2
        try:
            filter_type = ShuntFilterType(shtype)
        except ValueError:
            filter_type = ShuntFilterType.C
        
        # Get active steps
        try:
            n_cap = shnt.GetAttribute("ncap_a") or 1.0
        except Exception:
            n_cap = 1.0
        
        try:
            n_rea = shnt.GetAttribute("nrea_a") or 1.0
        except Exception:
            n_rea = 1.0
        
        # Get R-L-C parameters (per step)
        bcap_us = getattr(shnt, 'bcap', 0.0) or 0.0
        xrea_ohm = getattr(shnt, 'xrea', 0.0) or 0.0
        rrea_ohm = getattr(shnt, 'rrea', 0.0) or 0.0
        gparac_us = getattr(shnt, 'gparac', 0.0) or 0.0

        # High-pass filter parameters
        c1_uf = getattr(shnt, 'c1', 0.0) or 0.0
        c2_uf = getattr(shnt, 'c2', 0.0) or 0.0
        rpara_ohm = getattr(shnt, 'rpara', 0.0) or 0.0
        
        shunts.append(ShuntFilterShunt(
            pf_object=shnt,
            name=shnt.loc_name,
            bus_name=get_bus_full_name(bus),
            voltage_kv=bus.uknom,
            filter_type=filter_type,
            n_cap=n_cap,
            n_rea=n_rea,
            bcap_us=bcap_us,
            xrea_ohm=xrea_ohm,
            rrea_ohm=rrea_ohm,
            gparac_us=gparac_us,
            c1_uf=c1_uf,
            c2_uf=c2_uf,
            rpara_ohm=rpara_ohm,
            f_sys=50.0
        ))

    # --- Three-winding Transformers (ElmTr3) ---
    pf_trafos_3w: List[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmTr3", 0, 0, 0)
    for trafo in pf_trafos_3w:
        if trafo.outserv == 1:
            continue
        
        # Check if element is energized
        if trafo.IsEnergized() != 1:
            continue
        
        # Get terminals via cubicles (HV = 0, MV = 1, LV = 2)
        try:
            cub0 = trafo.GetCubicle(0)
            cub1 = trafo.GetCubicle(1)
            cub2 = trafo.GetCubicle(2)
            if cub0 is None or cub1 is None or cub2 is None:
                logger.info(f" 3W Transformer '{trafo.loc_name}': Missing cubicle(s), skipping")
                continue
            hv_bus = cub0.cterm
            mv_bus = cub1.cterm
            lv_bus = cub2.cterm
            if hv_bus is None or mv_bus is None or lv_bus is None:
                logger.info(f" 3W Transformer '{trafo.loc_name}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if hv_bus.IsEnergized() != 1 or mv_bus.IsEnergized() != 1 or lv_bus.IsEnergized() != 1:
                logger.info(f" 3W Transformer '{trafo.loc_name}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" 3W Transformer '{trafo.loc_name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue
        
        # Get transformer type data
        pf_type = trafo.GetAttribute("typ_id")
        rated_mva = 0.0
        hv_kv = 0.0
        mv_kv = 0.0
        lv_kv = 0.0
        if pf_type is not None:
            rated_mva = pf_type.strn3_h if hasattr(pf_type, 'strn3_h') else 0.0
            hv_kv = pf_type.utrn3_h if hasattr(pf_type, 'utrn3_h') else 0.0
            mv_kv = pf_type.utrn3_m if hasattr(pf_type, 'utrn3_m') else 0.0
            lv_kv = pf_type.utrn3_l if hasattr(pf_type, 'utrn3_l') else 0.0
        
        # Number of parallel transformers
        n_parallel = getattr(trafo, 'ntnum', 1) or 1
        
        transformers_3w.append(Transformer3WBranch(
            pf_object=trafo,
            name=trafo.loc_name,
            hv_bus_name=get_bus_full_name(hv_bus),
            mv_bus_name=get_bus_full_name(mv_bus),
            lv_bus_name=get_bus_full_name(lv_bus),
            base_mva=100.0,
            n_parallel=n_parallel,
            rated_power_mva=rated_mva,
            hv_kv=hv_kv,
            mv_kv=mv_kv,
            lv_kv=lv_kv
        ))

    return branches, shunts, transformers_3w
