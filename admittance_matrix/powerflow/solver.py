"""
PowerFactory load flow execution and result extraction.

This module provides functions to run load flow calculations and
retrieve results from PowerFactory.
"""

from .results import BusResult, GeneratorResult, calculate_internal_voltage
from ..core.elements import ShuntElement, GeneratorShunt


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
        
        v_pu = bus.GetAttribute("m:u")      # Voltage magnitude in p.u.
        angle = bus.GetAttribute("m:phiu")  # Voltage angle in degrees
        v_kv = bus.uknom * v_pu             # Voltage in kV
        
        results[bus.loc_name] = BusResult(
            name=bus.loc_name,
            voltage_pu=v_pu,
            angle_deg=angle,
            voltage_kv=v_kv
        )
    
    return results


def get_generator_data(
    shunts: list[ShuntElement],
    lf_results: dict[str, BusResult],
    base_mva: float = 100.0
) -> list[GeneratorResult]:
    """
    Get generator terminal voltages, impedances, and internal voltages.
    
    Args:
        shunts: List of shunt elements (to extract generators)
        lf_results: Load flow results from get_load_flow_results()
        base_mva: System base power in MVA
        
    Returns:
        List of GeneratorResult objects
    """
    results: list[GeneratorResult] = []
    
    for s in shunts:
        # Use type name check to handle autoreload issues
        if type(s).__name__ != 'GeneratorShunt':
            continue
        
        # Get terminal voltage from load flow results
        bus_result = lf_results.get(s.bus_name)
        if bus_result is None:
            print(f"Warning: No load flow result for bus {s.bus_name}, skipping generator {s.name}")
            continue
        
        voltage = bus_result.voltage_complex
        
        # Get P and Q - need to access the pf_object if available
        # For the refactored version, we need to store these values differently
        # This function will need the PowerFactory objects or the values passed in
        p_pu = 0.0
        q_pu = 0.0
        
        if hasattr(s, 'rated_power_mva') and s.rated_power_mva > 0:
            z_pu_gen = complex(0, s.xdss_pu)
            z_pu_sys = complex(0, s.xdss_pu * base_mva / s.rated_power_mva)
        else:
            z_pu_gen = complex(0, 0)
            z_pu_sys = complex(0, 0)
        
        # Calculate internal voltage
        internal_v, internal_v_mag, internal_v_angle = calculate_internal_voltage(
            voltage, p_pu, q_pu, s.xdss_pu
        )
        
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
            rated_kv=s.rated_voltage_kv
        ))
    
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
    
    for s in shunts:
        if type(s).__name__ != 'GeneratorShunt':
            continue
        
        bus_result = lf_results.get(s.bus_name)
        if bus_result is None:
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
        
        internal_v, internal_v_mag, internal_v_angle = calculate_internal_voltage(
            voltage, p_pu, q_pu, s.xdss_pu
        )
        
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
            rated_kv=s.rated_voltage_kv
        ))
    
    return results
