"""
PowerFactory load flow execution and result extraction.

This module provides functions to run load flow calculations and
retrieve results from PowerFactory.
"""

import cmath
import logging

from .naming import get_bus_full_name
from .results import BusResult, GeneratorResult, VoltageSourceResult, ExternalGridResult
from ...core.elements import ShuntElement, GeneratorShunt, VoltageSourceShunt, ExternalGridShunt

logger = logging.getLogger(__name__)


def _calculate_internal_voltage(
    terminal_voltage: complex,
    p_pu: float,
    q_pu: float,
    xdss_pu: float
) -> tuple[complex, float, float]:
    """
    Calculate generator internal voltage E' behind sub-transient reactance.
    
    E' = V + jX''d × (S*/V*)
    
    Args:
        terminal_voltage: Complex terminal voltage (p.u.)
        p_pu: Active power on generator base (p.u.)
        q_pu: Reactive power on generator base (p.u.)
        xdss_pu: Sub-transient reactance on generator base (p.u.)
        
    Returns:
        Tuple of (E' complex, |E'|, angle in degrees)
    """
    if abs(terminal_voltage) == 0:
        return complex(0, 0), 0.0, 0.0
    
    s_pu = complex(p_pu, q_pu)
    z_pu = complex(0, xdss_pu)
    
    internal_voltage = terminal_voltage + z_pu * (s_pu.conjugate() / terminal_voltage.conjugate())
    magnitude = abs(internal_voltage)
    angle_deg = cmath.phase(internal_voltage) * 180 / cmath.pi
    
    return internal_voltage, magnitude, angle_deg


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


def get_load_flow_results(app) -> dict[str, BusResult]:
    """
    Get load flow results for all busbars in the network.
    
    Requires a load flow calculation to have been executed.
    
    Args:
        app: PowerFactory application instance
        
    Returns:
        Dictionary mapping bus names to BusResult objects
    """
    results: dict[str, BusResult] = {}
    
    pf_busbars = app.GetCalcRelevantObjects("*.ElmTerm", 0, 0, 0)
    for bus in pf_busbars:
        if bus.outserv == 1:
            continue
        
        try:
            v_pu = bus.GetAttribute("m:u")      # Voltage magnitude in p.u.
            angle = bus.GetAttribute("m:phiu")  # Voltage angle in degrees
        except Exception as e:
            logger.info(f" Failed to get load flow results for bus {bus.loc_name}: {e}")
            continue
        
        if v_pu is None or angle is None:
            logger.info(f" Load flow results not available for bus {bus.loc_name}")
            continue
        
        v_kv = bus.uknom * v_pu             # Voltage in kV
        
        results[get_bus_full_name(bus)] = BusResult(
            name=get_bus_full_name(bus),
            voltage_pu=v_pu,
            angle_deg=angle,
            voltage_kv=v_kv
        )
    
    return results


def get_generator_data_from_pf(
    app,
    shunts: list[ShuntElement],
    lf_results: dict[str, BusResult],
    base_mva: float = 100.0
) -> list[GeneratorResult]:
    """
    Get generator data including P/Q from PowerFactory load flow results.
    
    This function directly reads P and Q from PowerFactory objects.
    
    Args:
        app: PowerFactory application instance
        shunts: List of shunt elements (to extract generators)
        lf_results: Load flow results from get_load_flow_results()
        base_mva: System base power in MVA
        
    Returns:
        List of GeneratorResult objects
    """
    results: list[GeneratorResult] = []
    
    # Get all generators from PowerFactory
    pf_gens = app.GetCalcRelevantObjects("*.ElmSym", 0, 0, 0)
    gen_pf_map = {gen.loc_name: gen for gen in pf_gens}

    logger.debug(len(pf_gens))
    
    for s in shunts:
        if type(s).__name__ != 'GeneratorShunt':
            continue
        
        bus_result = lf_results.get(s.bus_name)
        if bus_result is None:
            logger.warning(f" No load flow result for bus {s.bus_name}, skipping generator {s.name}")
            continue
        
        voltage = bus_result.voltage_complex
        
        # Get PowerFactory object for this generator
        pf_gen = gen_pf_map.get(s.name)
        if pf_gen is None:
            continue
        
        # Get P and Q from load flow results
        p_mw = pf_gen.GetAttribute("m:P:bus1")
        q_mvar = pf_gen.GetAttribute("m:Q:bus1")
        
        if hasattr(s, 'rated_power_mva') and s.rated_power_mva > 0:
            p_pu = p_mw / s.rated_power_mva
            q_pu = q_mvar / s.rated_power_mva
            z_pu_sys = complex(0, s.xdss_pu * base_mva / s.rated_power_mva)
        else:
            p_pu = 0.0
            q_pu = 0.0
            z_pu_sys = complex(0, 0)
        
        internal_v, internal_v_mag, internal_v_angle = _calculate_internal_voltage(
            voltage, p_pu, q_pu, s.xdss_pu
        )
        
        # Get zone from cpZone attribute
        zone_name = 'Unknown'
        try:
            zone_obj = pf_gen.GetAttribute('cpZone')
            if zone_obj is not None:
                zone_name = zone_obj.loc_name
        except Exception:
            pass
        
        results.append(GeneratorResult(
            name=s.name,
            bus_name=s.bus_name,
            voltage=voltage,
            xdss_pu=s.xdss_pu,
            impedance_pu=z_pu_sys,
            p_pu=p_pu,
            q_pu=q_pu,
            internal_voltage=internal_v,
            internal_voltage_mag=internal_v_mag,
            internal_voltage_angle=internal_v_angle,
            rated_mva=s.rated_power_mva,
            rated_kv=s.rated_voltage_kv,
            zone=zone_name
        ))
    
    logger.debug("Number of generators extracted:", len(results))
    return results


def get_voltage_source_data_from_pf(
    app,
    shunts: list[ShuntElement],
    lf_results: dict[str, BusResult],
    base_mva: float = 100.0
) -> list[VoltageSourceResult]:
    """
    Get voltage source data including internal voltage from PowerFactory load flow results.
    
    For voltage sources, the internal voltage is calculated similarly to generators:
    E = V + Z × I, where I = (S*/V*)
    
    Args:
        app: PowerFactory application instance
        shunts: List of shunt elements (to extract voltage sources)
        lf_results: Load flow results from get_load_flow_results()
        base_mva: System base power in MVA
        
    Returns:
        List of VoltageSourceResult objects
    """
    results: list[VoltageSourceResult] = []
    
    # Get all AC voltage sources from PowerFactory
    pf_vacs = app.GetCalcRelevantObjects("*.ElmVac", 0, 0, 0)
    vac_pf_map = {vac.loc_name: vac for vac in pf_vacs}
    
    for s in shunts:
        if not isinstance(s, VoltageSourceShunt):
            continue
        
        bus_result = lf_results.get(s.bus_name)
        if bus_result is None:
            logger.warning(f" No load flow result for bus {s.bus_name}, skipping voltage source {s.name}")
            continue
        
        voltage = bus_result.voltage_complex
        
        # Get PowerFactory object for this voltage source
        pf_vac = vac_pf_map.get(s.name)
        if pf_vac is None:
            logger.warning(f" PowerFactory object not found for voltage source {s.name}")
            continue
        
        # Get P and Q from load flow results
        p_mw = pf_vac.GetAttribute("m:P:bus1") or 0.0
        q_mvar = pf_vac.GetAttribute("m:Q:bus1") or 0.0
        
        # Calculate impedance on system base
        z_base = (s.voltage_kv ** 2) / base_mva
        z_ohm = complex(s.resistance_ohm, s.reactance_ohm)
        z_pu_sys = z_ohm / z_base if z_base > 0 else complex(0, 0)
        
        # Calculate internal voltage: E = V + Z × (S*/V*)
        if abs(voltage) > 0 and abs(z_pu_sys) > 0:
            s_pu = complex(p_mw / base_mva, q_mvar / base_mva)
            i_pu = (s_pu.conjugate() / voltage.conjugate())
            internal_v = voltage + z_pu_sys * i_pu
        else:
            # If no impedance, internal voltage = terminal voltage
            internal_v = voltage
        
        internal_v_mag = abs(internal_v)
        internal_v_angle = cmath.phase(internal_v) * 180 / cmath.pi
        
        results.append(VoltageSourceResult(
            name=s.name,
            bus_name=s.bus_name,
            voltage=voltage,
            impedance_pu=z_pu_sys,
            internal_voltage=internal_v,
            internal_voltage_mag=internal_v_mag,
            internal_voltage_angle=internal_v_angle
        ))
    
    logger.debug("Number of voltage sources extracted:", len(results))
    return results


def get_external_grid_data_from_pf(
    app,
    shunts: list[ShuntElement],
    lf_results: dict[str, BusResult],
    base_mva: float = 100.0
) -> list[ExternalGridResult]:
    """
    Get external grid data including internal voltage from PowerFactory load flow results.
    
    For external grids, the internal voltage is calculated similarly to generators:
    E = V + Z × I, where I = (S*/V*)
    
    Args:
        app: PowerFactory application instance
        shunts: List of shunt elements (to extract external grids)
        lf_results: Load flow results from get_load_flow_results()
        base_mva: System base power in MVA
        
    Returns:
        List of ExternalGridResult objects
    """
    results: list[ExternalGridResult] = []
    
    # Get all external grids from PowerFactory
    pf_xnets = app.GetCalcRelevantObjects("*.ElmXnet", 0, 0, 0)
    xnet_pf_map = {xnet.loc_name: xnet for xnet in pf_xnets}
    
    for s in shunts:
        if not isinstance(s, ExternalGridShunt):
            continue
        
        bus_result = lf_results.get(s.bus_name)
        if bus_result is None:
            logger.warning(f" No load flow result for bus {s.bus_name}, skipping external grid {s.name}")
            continue
        
        voltage = bus_result.voltage_complex
        
        # Get PowerFactory object for this external grid
        pf_xnet = xnet_pf_map.get(s.name)
        if pf_xnet is None:
            logger.warning(f" PowerFactory object not found for external grid {s.name}")
            continue
        
        # Get P and Q from load flow results
        p_mw = pf_xnet.GetAttribute("m:P:bus1") or 0.0
        q_mvar = pf_xnet.GetAttribute("m:Q:bus1") or 0.0
        
        # Calculate impedance on system base from short-circuit data
        if s.s_sc_mva > 0 and s.voltage_kv > 0:
            z_sc = (s.voltage_kv ** 2) / s.s_sc_mva
            x_sc = z_sc / ((1 + s.r_x_ratio ** 2) ** 0.5) * s.c_factor
            r_sc = x_sc * s.r_x_ratio
            z_ohm = complex(r_sc, x_sc)
            z_base = (s.voltage_kv ** 2) / base_mva
            z_pu_sys = z_ohm / z_base if z_base > 0 else complex(0, 0)
        else:
            z_pu_sys = complex(0, 0)
        
        # Calculate internal voltage: E = V + Z × (S*/V*)
        if abs(voltage) > 0 and abs(z_pu_sys) > 0:
            s_pu = complex(p_mw / base_mva, q_mvar / base_mva)
            i_pu = (s_pu.conjugate() / voltage.conjugate())
            internal_v = voltage + z_pu_sys * i_pu
        else:
            # If no impedance, internal voltage = terminal voltage
            internal_v = voltage
        
        internal_v_mag = abs(internal_v)
        internal_v_angle = cmath.phase(internal_v) * 180 / cmath.pi
        
        results.append(ExternalGridResult(
            name=s.name,
            bus_name=s.bus_name,
            voltage=voltage,
            impedance_pu=z_pu_sys,
            internal_voltage=internal_v,
            internal_voltage_mag=internal_v_mag,
            internal_voltage_angle=internal_v_angle
        ))
    
    logger.debug("Number of external grids extracted:", len(results))
    return results
