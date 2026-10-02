"""Extract a solved PowerFactory operating point and load-flow results."""

from __future__ import annotations

import cmath
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ...network import Network
from ...network.elements import ExternalGridShunt, GeneratorShunt, VoltageSourceShunt
from ...network.operating_point import (
    BusResult,
    ExternalGridResult,
    GeneratorResult,
    OperatingPoint,
    VoltageSourceResult,
)
from .naming import get_bus_full_name

if TYPE_CHECKING:
    import powerfactory as pf

logger = logging.getLogger(__name__)


def extract_operating_point(
    app: pf.Application,
    network: Network,
    *,
    run_load_flow_first: bool = True,
) -> OperatingPoint:
    """Read a solved PowerFactory operating point for a standalone network."""
    app.Hide()
    try:
        if run_load_flow_first and not run_load_flow(app):
            raise RuntimeError("PowerFactory load flow not executed. Check PowerFactory Output Window.")

        buses = get_load_flow_results(app)
        generators = network.generators
        voltage_sources = [source for source in network.voltage_sources if isinstance(source, VoltageSourceShunt)]
        external_grids = [source for source in network.voltage_sources if isinstance(source, ExternalGridShunt)]

        sources = (
            get_generator_data_from_pf(app, generators, buses, network.base_mva)
            + get_voltage_source_data_from_pf(app, voltage_sources, buses, network.base_mva)
            + get_external_grid_data_from_pf(app, external_grids, buses, network.base_mva)
        )
        results_by_name = {source.name: source for source in sources}
        missing = set(network.stability_source_names) - results_by_name.keys()
        if missing:
            raise ValueError(f"Missing load-flow results for sources: {sorted(missing)}")
        ordered_sources = tuple(results_by_name[name] for name in network.stability_source_names)
    finally:
        app.Show()

    return OperatingPoint(
        base_mva=network.base_mva,
        buses=buses,
        sources=ordered_sources,
    )


def get_load_flow_results(app: pf.Application) -> dict[str, BusResult]:
    """Read bus voltages from an already solved PowerFactory load flow."""
    results: dict[str, BusResult] = {}
    pf_busbars: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmTerm", 0, 0, 0)

    for bus in pf_busbars:
        if bus.GetAttribute("outserv") == 1:
            continue
        try:
            v_pu = bus.GetAttribute("m:u")  # Voltage magnitude in p.u.
            angle = bus.GetAttribute("m:phiu")  # Voltage angle in degrees
        except Exception as error:
            logger.info(
                "Failed to read load flow for bus %s: %s",
                bus.GetAttribute("loc_name"),
                error,
            )
            continue

        if v_pu is None or angle is None:
            logger.info("Load flow results unavailable for bus %s", bus.GetAttribute("loc_name"))
            continue

        v_kv = bus.GetAttribute("uknom") * v_pu  # Voltage in kV

        bus_name = get_bus_full_name(bus)
        results[bus_name] = BusResult(
            name=bus_name,
            voltage_pu=v_pu,
            angle_deg=angle,
            voltage_kv=v_kv,
        )

    return results


def get_generator_data_from_pf(
    app: pf.Application,
    syn_gens: Sequence[GeneratorShunt],
    lf_results: Mapping[str, BusResult],
    base_mva: float = 100.0,
) -> list[GeneratorResult]:
    """Read generator P/Q and calculate internal voltage on the system base.

    Uses the classical source relation ``E = V + Z * conj(S / V)``.
    """
    results: list[GeneratorResult] = []

    pf_gens: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmSym", 0, 0, 0)
    gen_pf_map: dict[str, pf.DataObject] = {gen.GetAttribute("loc_name"): gen for gen in pf_gens}

    for gen in syn_gens:
        # Get load flow result for this generator's bus
        bus_result = lf_results.get(gen.bus_name)
        if bus_result is None:
            logger.warning(
                "No load flow result for bus %s; skipping generator %s",
                gen.bus_name,
                gen.name,
            )
            continue

        # Get PowerFactory object for this generator
        pf_gen = gen_pf_map.get(gen.name)
        if pf_gen is None:
            logger.warning("PowerFactory object not found for generator %s", gen.name)
            continue

        # Read terminal voltage and power from load flow results
        voltage_pu = bus_result.voltage_complex
        p_mw = pf_gen.GetAttribute("m:P:bus1")
        q_mvar = pf_gen.GetAttribute("m:Q:bus1")

        # Convert P and Q to per-unit on system base
        p_pu = p_mw / base_mva
        q_pu = q_mvar / base_mva
        s_pu = complex(p_pu, q_pu)

        # Convert generator impedance to system base
        z_pu_system = gen.z_pu * (base_mva / gen.rated_power_mva)

        internal_voltage = voltage_pu + z_pu_system * (s_pu.conjugate() / voltage_pu.conjugate())

        results.append(
            GeneratorResult(
                name=gen.name,
                bus_name=gen.bus_name,
                terminal_voltage=voltage_pu,
                impedance_pu=z_pu_system,
                p_pu=p_pu,
                q_pu=q_pu,
                internal_voltage=internal_voltage,
                rated_mva=gen.rated_power_mva,
                rated_kv=gen.rated_voltage_kv,
            )
        )

    logger.debug("Number of generators extracted: %d", len(results))
    return results


def get_voltage_source_data_from_pf(
    app: pf.Application,
    v_sources: Sequence[VoltageSourceShunt],
    lf_results: Mapping[str, BusResult],
    base_mva: float = 100.0,
) -> list[VoltageSourceResult]:
    """Read AC-source P/Q and calculate internal voltage behind its impedance.

    Convert the source impedance with ``Z_base = V_base**2 / S_base`` and
    calculate ``E = V + Z * conj(S / V)`` on the system base.
    """
    results: list[VoltageSourceResult] = []

    # Get all AC voltage sources from PowerFactory
    pf_vacs = app.GetCalcRelevantObjects("*.ElmVac", 0, 0, 0)
    vac_pf_map = {vac.loc_name: vac for vac in pf_vacs}

    for src in v_sources:
        # Get load flow result for this voltage source's bus
        bus_result = lf_results.get(src.bus_name)
        if bus_result is None:
            logger.warning(
                "No load flow result for bus %s; skipping voltage source %s",
                src.bus_name,
                src.name,
            )
            continue

        # Get PowerFactory object for this voltage source
        pf_vac = vac_pf_map.get(src.name)
        if pf_vac is None:
            logger.warning("PowerFactory object not found for voltage source %s", src.name)
            continue

        # Read terminal V and voltage source P, Q from LF results
        voltage_pu = bus_result.voltage_complex
        p_mw = pf_vac.GetAttribute("m:P:bus1")
        q_mvar = pf_vac.GetAttribute("m:Q:bus1")

        # Convert P and Q to per-unit on system base
        p_pu = p_mw / base_mva
        q_pu = q_mvar / base_mva
        s_pu = complex(p_pu, q_pu)

        # Calculate impedance on system base
        z_base = (src.voltage_kv**2) / base_mva
        z_ohm = complex(src.resistance_ohm, src.reactance_ohm)
        z_pu_system = z_ohm / z_base if z_base > 0 else complex(0, 0)

        # Calculate internal voltage
        if abs(voltage_pu) > 0 and abs(z_pu_system) > 0:
            i_pu = s_pu.conjugate() / voltage_pu.conjugate()
            internal_v = voltage_pu + z_pu_system * i_pu
        else:
            # If no impedance, internal voltage = terminal voltage
            internal_v = voltage_pu

        internal_v_mag = abs(internal_v)
        internal_v_angle = cmath.phase(internal_v) * 180 / cmath.pi

        results.append(
            VoltageSourceResult(
                name=src.name,
                bus_name=src.bus_name,
                terminal_voltage=voltage_pu,
                impedance_pu=z_pu_system,
                p_pu=p_pu,
                q_pu=q_pu,
                internal_voltage=internal_v,
                internal_voltage_mag=internal_v_mag,
                internal_voltage_angle=internal_v_angle,
            )
        )

    logger.debug("Number of voltage sources extracted: %d", len(results))
    return results


def get_external_grid_data_from_pf(
    app: pf.Application,
    xnets: Sequence[ExternalGridShunt],
    lf_results: Mapping[str, BusResult],
    base_mva: float = 100.0,
) -> list[ExternalGridResult]:
    """Read external-grid P/Q and calculate internal voltage.

    The existing short-circuit impedance model is converted to the system base,
    then ``E = V + Z * conj(S / V)`` is evaluated when impedance is nonzero.
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
            logger.warning(
                "No load flow result for bus %s; skipping external grid %s",
                xnet.bus_name,
                xnet.name,
            )
            continue

        voltage = bus_result.voltage_complex

        # Get PowerFactory object for this external grid
        pf_xnet = xnet_pf_map.get(xnet.name)
        if pf_xnet is None:
            logger.warning("PowerFactory object not found for external grid %s", xnet.name)
            continue

        # Get P and Q from load flow results
        p_mw = pf_xnet.GetAttribute("m:P:bus1") or 0.0
        q_mvar = pf_xnet.GetAttribute("m:Q:bus1") or 0.0

        # Calculate impedance on system base from short-circuit data
        if xnet.s_sc_mva > 0 and xnet.voltage_kv > 0:
            z_sc = (xnet.voltage_kv**2) / xnet.s_sc_mva
            x_sc = z_sc / ((1 + xnet.r_x_ratio**2) ** 0.5) * xnet.c_factor
            r_sc = x_sc * xnet.r_x_ratio
            z_ohm = complex(r_sc, x_sc)
            z_base = (xnet.voltage_kv**2) / base_mva
            z_pu_sys = z_ohm / z_base if z_base > 0 else complex(0, 0)
        else:
            z_pu_sys = complex(0, 0)

        # Calculate internal voltage: E = V + Z × (S*/V*)
        if abs(voltage) > 0 and abs(z_pu_sys) > 0:
            s_pu = complex(p_mw / base_mva, q_mvar / base_mva)
            i_pu = s_pu.conjugate() / voltage.conjugate()
            internal_v = voltage + z_pu_sys * i_pu
        else:
            # If no impedance, internal voltage = terminal voltage
            internal_v = voltage

        internal_v_mag = abs(internal_v)
        internal_v_angle = cmath.phase(internal_v) * 180 / cmath.pi

        results.append(
            ExternalGridResult(
                name=xnet.name,
                bus_name=xnet.bus_name,
                terminal_voltage=voltage,
                impedance_pu=z_pu_sys,
                p_pu=p_mw / base_mva,
                q_pu=q_mvar / base_mva,
                internal_voltage=internal_v,
                internal_voltage_mag=internal_v_mag,
                internal_voltage_angle=internal_v_angle,
            )
        )

    logger.debug("Number of external grids extracted:", len(results))
    return results


def run_load_flow(app: pf.Application) -> bool:
    """Run the active study case's load flow and report convergence."""
    ldf = app.GetFromStudyCase("ComLdf")
    if ldf is None:
        raise RuntimeError("Could not get load flow command from study case")

    err = ldf.Execute()
    return err == 0
