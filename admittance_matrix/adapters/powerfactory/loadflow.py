"""
PowerFactory load flow execution and result extraction.

This module provides functions to run load flow calculations and
retrieve results from PowerFactory.
"""

import cmath
import logging
import powerfactory as pf
from collections.abc import Mapping, Sequence

from .naming import get_bus_full_name
from .results import BusResult, GeneratorResult, VoltageSourceResult, ExternalGridResult
from ...core.elements import ShuntElement, GeneratorShunt, VoltageSourceShunt, ExternalGridShunt

logger = logging.getLogger(__name__)

def run_load_flow(app) -> bool:
    """
    Execute load flow calculation in PowerFactory.
    
    Args:
        app: PowerFactory application instance
        
    Returns:
        True if load flow converged, False otherwise
    """
    ldf = app.GetFromStudyCase("ComLdf")
    if ldf is None:
        raise RuntimeError("Could not get load flow command from study case")
    
    err = ldf.Execute()
    return err == 0

def get_load_flow_results(app: pf.Application) -> dict[str, BusResult]:
    """
    Get load flow results for all busbars in the network.
    
    Requires a load flow calculation to have been executed.
    
    Args:
        app: PowerFactory application instance
        
    Returns:
        Dictionary mapping bus names to BusResult objects
    """
    results: dict[str, BusResult] = {}
    pf_busbars: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmTerm", 0, 0, 0)

    for bus in pf_busbars:
        if bus.GetAttribute("outserv") == 1:
            continue
        try:
            v_pu = bus.GetAttribute("m:u")      # Voltage magnitude in p.u.
            angle = bus.GetAttribute("m:phiu")  # Voltage angle in degrees
        except Exception as e:
            logger.info(f" Failed to get load flow results for bus {bus.GetAttribute('loc_name')}: {e}")
            continue
        
        if v_pu is None or angle is None:
            logger.info(f" Load flow results not available for bus {bus.GetAttribute('loc_name')}")
            continue
        
        v_kv = bus.GetAttribute("uknom") * v_pu             # Voltage in kV
        
        results[get_bus_full_name(bus)] = BusResult(
            name=get_bus_full_name(bus),
            voltage_pu=v_pu,
            angle_deg=angle,
            voltage_kv=v_kv
        )
    
    return results

def get_generator_data_from_pf(
    app: pf.Application,
    syn_gens: Sequence[GeneratorShunt],
    lf_results: Mapping[str, BusResult],
    BASE_MVA: float = 100.0
) -> list[GeneratorResult]:
    """
    Get generator data including P/Q from PowerFactory load flow results.
    
    This function directly reads P and Q from PowerFactory objects.
    
    Args:
        app: PowerFactory application instance
        syn_gens: Sequence of synchronous generator shunt elements
        lf_results: Load flow results from get_load_flow_results()
        BASE_MVA: System base power in MVA
        
    Returns:
        List of GeneratorResult objects
    """
    # # First initialize dynamic data (only)
    # oInit = app.GetFromStudyCase('ComInc') # Get initial condition calculation object
    # oInit.Execute() # type: ignore

    # Initialize results list
    results: list[GeneratorResult] = []
    
    # Get all ElmSym DataObjects from PF to extract data
    pf_gens: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmSym", 0, 0, 0)
    gen_pf_map: dict[str, pf.DataObject] = {gen.GetAttribute("loc_name"): gen for gen in pf_gens}

    for gen in syn_gens:
        # Get load flow result for this generator's bus
        bus_result = lf_results.get(gen.bus_name)
        if bus_result is None:
            logger.warning(f" No load flow result for bus {gen.bus_name}, skipping generator {gen.name}")
            continue

        # Get PowerFactory object for this generator
        pf_gen = gen_pf_map.get(gen.name)
        if pf_gen is None:
            logger.warning(f" PowerFactory object not found for generator {gen.name}")
            continue

        # Read terminal voltage and power from load flow results
        voltage_pu = bus_result.voltage_complex 
        P_MW = pf_gen.GetAttribute("m:P:bus1")
        Q_MVAR = pf_gen.GetAttribute("m:Q:bus1")

        # Convert P and Q to per-unit on system base
        P_PU = P_MW / BASE_MVA
        Q_PU = Q_MVAR / BASE_MVA
        S_PU = complex(P_PU, Q_PU)

        # Convert generator impedance to system base
        Z_PU_SYS = gen.z_pu * (BASE_MVA / gen.rated_power_mva)

        # ========================= Calculate generators internal voltage =========================
        E0 = voltage_pu + Z_PU_SYS * (S_PU.conjugate() / voltage_pu.conjugate())
        # print(f" {gen.name}: |E'| = {abs(E0):.4f} pu, angle = {cmath.phase(E0) * 180 / cmath.pi:.2f} deg") #! Dev

        # # Collect power factory calculations #TODO
        # internalAngle = pf_gen.GetAttribute("s:firel")
        # internalMag = pf_gen.GetAttribute("s:ve")
        # E0 = cmath.rect(internalMag, internalAngle * cmath.pi / 180)
        # print(f" {gen.name}: |E'| = {abs(E0):.4f} pu, angle = {cmath.phase(E0) * 180 / cmath.pi:.2f} deg")

        results.append(GeneratorResult(
            name=gen.name,
            bus_name=gen.bus_name,
            terminal_voltage=voltage_pu,
            impedance_pu=Z_PU_SYS,
            p_pu=P_PU,
            q_pu=Q_PU,
            internal_voltage=E0,
            rated_mva=gen.rated_power_mva,
            rated_kv=gen.rated_voltage_kv,
        ))
    
    logger.debug("Number of generators extracted: %d", len(results))
    return results

def get_voltage_source_data_from_pf(
    app: pf.Application,
    v_sources: Sequence[VoltageSourceShunt],
    lf_results: Mapping[str, BusResult],
    BASE_MVA: float = 100.0
) -> list[VoltageSourceResult]:
    """
    Get voltage source data including internal voltage from PowerFactory load flow results.
    
    For voltage sources, the internal voltage is calculated similarly to generators:
    E = V + Z × I, where I = (S*/V*)
    
    Args:
        app: PowerFactory application instance
        v_sources: List of voltage source shunt elements
        lf_results: Load flow results from get_load_flow_results()
        BASE_MVA: System base power in MVA
        
    Returns:
        List of VoltageSourceResult objects
    """
    results: list[VoltageSourceResult] = []
    
    # Get all AC voltage sources from PowerFactory
    pf_vacs = app.GetCalcRelevantObjects("*.ElmVac", 0, 0, 0)
    vac_pf_map = {vac.loc_name: vac for vac in pf_vacs}
    
    for src in v_sources:
        # Get load flow result for this voltage source's bus
        bus_result = lf_results.get(src.bus_name)
        if bus_result is None:
            logger.warning(f" No load flow result for bus {src.bus_name}, skipping voltage source {src.name}")
            continue
        
        # Get PowerFactory object for this voltage source
        pf_vac = vac_pf_map.get(src.name)
        if pf_vac is None:
            logger.warning(f" PowerFactory object not found for voltage source {src.name}")
            continue
        
        # Read terminal V and voltage source P, Q from LF results
        voltage_pu = bus_result.voltage_complex
        P_MW = pf_vac.GetAttribute("m:P:bus1")
        Q_MVAR = pf_vac.GetAttribute("m:Q:bus1")

        # Convert P and Q to per-unit on system base
        P_PU = P_MW / BASE_MVA
        Q_PU = Q_MVAR / BASE_MVA
        S_PU = complex(P_PU, Q_PU)
        
        # Calculate impedance on system base
        Z_BASE = (src.voltage_kv ** 2) / BASE_MVA
        Z_ohm = complex(src.resistance_ohm, src.reactance_ohm)
        Z_PU_SYS = Z_ohm / Z_BASE if Z_BASE > 0 else complex(0, 0)
        # TODO: Pomojem moram tule dati impdeanco na system base ne na Z_base kot je tule zgoraj...
        
        # Calculate internal voltage
        if abs(voltage_pu) > 0 and abs(Z_PU_SYS) > 0:
            i_pu = (S_PU.conjugate() / voltage_pu.conjugate())
            internal_v = voltage_pu + Z_PU_SYS * i_pu
        else:
            # If no impedance, internal voltage = terminal voltage
            internal_v = voltage_pu
        
        internal_v_mag = abs(internal_v)
        internal_v_angle = cmath.phase(internal_v) * 180 / cmath.pi

        results.append(VoltageSourceResult(
            name=src.name,
            bus_name=src.bus_name,
            terminal_voltage=voltage_pu,
            impedance_pu=Z_PU_SYS,
            p_pu=P_PU,
            q_pu=Q_PU,
            internal_voltage=internal_v,
            internal_voltage_mag=internal_v_mag,
            internal_voltage_angle=internal_v_angle
        ))
    
    logger.debug("Number of voltage sources extracted:", len(results))
    return results

def get_external_grid_data_from_pf(
    app: pf.Application,
    xnets: Sequence[ExternalGridShunt],
    lf_results: Mapping[str, BusResult],
    BASE_MVA: float = 100.0
) -> list[ExternalGridResult]:
    """
    Get external grid data including internal voltage from PowerFactory load flow results.
    
    For external grids, the internal voltage is calculated similarly to generators:
    E = V + Z × I, where I = (S*/V*)
    
    Args:
        app: PowerFactory application instance
        xnets: List of external grid elements (to extract external grids)
        lf_results: Load flow results from get_load_flow_results()
        BASE_MVA: System base power in MVA
        
    Returns:
        List of ExternalGridResult objects
    """
    results: list[ExternalGridResult] = []
    
    # Get all external grids from PowerFactory
    pf_xnets = app.GetCalcRelevantObjects("*.ElmXnet", 0, 0, 0)
    xnet_pf_map = {xnet.loc_name: xnet for xnet in pf_xnets}
    
    for xnet in xnets:
        if not isinstance(xnet, ExternalGridShunt):
            continue
        
        bus_result = lf_results.get(xnet.bus_name)
        if bus_result is None:
            logger.warning(f" No load flow result for bus {xnet.bus_name}, skipping external grid {xnet.name}")
            continue
        
        voltage = bus_result.voltage_complex
        
        # Get PowerFactory object for this external grid
        pf_xnet = xnet_pf_map.get(xnet.name)
        if pf_xnet is None:
            logger.warning(f" PowerFactory object not found for external grid {xnet.name}")
            continue
        
        # Get P and Q from load flow results
        p_mw = pf_xnet.GetAttribute("m:P:bus1") or 0.0
        q_mvar = pf_xnet.GetAttribute("m:Q:bus1") or 0.0
        
        # Calculate impedance on system base from short-circuit data
        if xnet.s_sc_mva > 0 and xnet.voltage_kv > 0:
            z_sc = (xnet.voltage_kv ** 2) / xnet.s_sc_mva
            x_sc = z_sc / ((1 + xnet.r_x_ratio ** 2) ** 0.5) * xnet.c_factor
            r_sc = x_sc * xnet.r_x_ratio
            z_ohm = complex(r_sc, x_sc)
            z_base = (xnet.voltage_kv ** 2) / BASE_MVA
            z_pu_sys = z_ohm / z_base if z_base > 0 else complex(0, 0)
        else:
            z_pu_sys = complex(0, 0)
        
        # Calculate internal voltage: E = V + Z × (S*/V*)
        if abs(voltage) > 0 and abs(z_pu_sys) > 0:
            s_pu = complex(p_mw / BASE_MVA, q_mvar / BASE_MVA)
            i_pu = (s_pu.conjugate() / voltage.conjugate())
            internal_v = voltage + z_pu_sys * i_pu
        else:
            # If no impedance, internal voltage = terminal voltage
            internal_v = voltage
        
        internal_v_mag = abs(internal_v)
        internal_v_angle = cmath.phase(internal_v) * 180 / cmath.pi
        
        # # Set internal voltage magnitude and angle from terminal voltage
        # # (for voltage sources without impedance or as fallback)
        # internal_v_mag = abs(voltage)
        # internal_v_angle = cmath.phase(voltage) * 180 / cmath.pi
        results.append(ExternalGridResult(
            name=xnet.name,
            bus_name=xnet.bus_name,
            terminal_voltage=voltage,
            impedance_pu=z_pu_sys,
            p_pu=p_mw / BASE_MVA,
            q_pu=q_mvar / BASE_MVA,
            internal_voltage=internal_v,
            internal_voltage_mag=internal_v_mag,
            internal_voltage_angle=internal_v_angle
        ))
    
    logger.debug("Number of external grids extracted:", len(results))
    return results
