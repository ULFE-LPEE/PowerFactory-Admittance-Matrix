"""Extract a standalone network and equipment from PowerFactory."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from ...network import Network
from ...network.elements import (
    CommonImpedanceBranch,
    ExternalGridShunt,
    GeneratorShunt,
    IdealPhaseTapChanger,
    LineBranch,
    LoadModelType,
    LoadShunt,
    PVSystemShunt,
    RatioAsymTapChanger,
    SeriesReactorBranch,
    ShuntFilterShunt,
    ShuntFilterType,
    SwitchBranch,
    SymPhaseTapChanger,
    TapChanger,
    Transformer3WBranch,
    TransformerBranch,
    VoltageSourceShunt,
)
from ...network.topology import merge_closed_switch_buses
from .naming import get_bus_full_name

if TYPE_CHECKING:
    import powerfactory as pf

logger = logging.getLogger(__name__)


# Unit conversions

VOLTAGE_TO_KV = {"kV": 1.0, "V": 1e-3}
POWER_TO_MVA = {"MVA": 1.0, "kVA": 1e-3, "VA": 1e-6}
POWER_TO_MW = {"MW": 1.0, "kW": 1e-3, "W": 1e-6}
POWER_TO_MVAR = {"Mvar": 1.0, "MVar": 1.0, "MVAr": 1.0, "kvar": 1e-3, "var": 1e-6}
POWER_TO_KW = {"MW": 1e3, "kW": 1.0, "W": 1e-3}
PERCENT_TO_PERCENT = {"%": 1.0, "p.u.": 100.0, "pu": 100.0}
PERCENT_OR_PER_UNIT = {"%": 1e-2, "p.u.": 1.0, "pu": 1.0}
LENGTH_TO_KM = {"km": 1.0, "m": 1e-3}
RESISTANCE_TO_OHM = {
    "Ohm": 1.0,
    "ohm": 1.0,
    "Ω": 1.0,
    "mOhm": 1e-3,
    "mΩ": 1e-3,
    "kOhm": 1e3,
}
RESISTANCE_PER_LENGTH_TO_OHM_PER_KM = {"Ohm/km": 1.0, "Ohm/m": 1e3}
ADMITTANCE_PER_LENGTH_TO_S_PER_KM = {
    "S/km": 1.0,
    "mS/km": 1e-3,
    "uS/km": 1e-6,
    "µS/km": 1e-6,
    "μS/km": 1e-6,
    "nS/km": 1e-9,
}
SUSCEPTANCE_TO_US = {"S": 1e6, "mS": 1e3, "uS": 1.0, "µS": 1.0, "μS": 1.0, "nS": 1e-3}
FREQUENCY_TO_HZ = {"Hz": 1.0, "kHz": 1e3}
ANGLE_TO_DEG = {"deg": 1.0, "°": 1.0, "rad": 180.0 / math.pi}


# Network assembly


def extract_network(
    app: pf.Application,
    *,
    base_mva: float = 100.0,
    merge_closed_switches: bool = False,
) -> Network:
    """Build a standalone network from calculation-relevant PowerFactory elements.

    Closed switches can be contracted after all elements have been added.
    """
    network = Network(base_mva=base_mva)
    app.Hide()

    try:
        # Extract branches
        for line in extract_lines(app):
            network.add_branch(line)
        for switch in extract_switches(app):
            network.add_branch(switch)
        for transformer in extract_two_winding_transformers(app):
            network.add_branch(transformer)
        for impedance in extract_common_impedances(app):
            network.add_branch(impedance)
        for reactor in extract_series_reactors(app):
            network.add_branch(reactor)

        # Extract shunt elements
        for generator in extract_generators(app):
            network.add_shunt(generator)
        for load in extract_loads(app):
            network.add_shunt(load)
        for source in extract_external_grids(app):
            network.add_shunt(source)
        for source in extract_voltage_sources(app):
            network.add_shunt(source)
        for pv_system in extract_pv_systems(app):
            network.add_shunt(pv_system)
        for shunt_filter in extract_shunt_filters(app):
            network.add_shunt(shunt_filter)

        # Extract three-winding transformers
        for transformer in extract_three_winding_transformers(app):
            network.add_three_winding_transformer(transformer)

        main_buses = get_main_bus_names(app) if merge_closed_switches else None
    finally:
        app.Show()

    if merge_closed_switches:
        return merge_closed_switch_buses(network, main_buses=main_buses)
    return network


# Lines, switches, and series equipment


def extract_lines(app: pf.Application) -> list[LineBranch]:
    """Read line type values and convert them to total branch R, X and B.

    Series impedance is divided by the number of parallel circuits; shunt
    susceptance is multiplied by it. The existing pi-model then halves B.
    """
    elements: list[LineBranch] = []
    pf_lines: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmLne", 0, 0, 0)
    for line in pf_lines:
        if line.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if line.IsEnergized() != 1:
            continue

        try:
            # Get cubicles
            cub0 = line.GetCubicle(0)
            cub1 = line.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Line '{line.GetAttribute('loc_name')}': Missing cubicle(s), skipping")
                continue

            # Check if cubicles are closed
            cub0_status = cub0.IsClosed()  # type: ignore
            cub1_status = cub1.IsClosed()  # type: ignore

            if cub0_status != 1 or cub1_status != 1:
                logger.info(f" Line '{line.GetAttribute('loc_name')}': One or both cubicles are open, skipping")
                continue

            # Get terminals
            from_bus = cub0.GetAttribute("cterm")
            to_bus = cub1.GetAttribute("cterm")
            if from_bus is None or to_bus is None:
                logger.info(f" Line '{line.GetAttribute('loc_name')}': Missing terminal(s) (cterm is None), skipping")
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Line '{line.GetAttribute('loc_name')}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Line '{line.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        line_type = line.GetAttribute("typ_id")
        if line_type is None:
            logger.warning("Line %s has no assigned type; skipping", line.GetAttribute("loc_name"))
            continue

        line_length_km = read_attribute(line, "dline", LENGTH_TO_KM)
        parallel_circuits = int(line.GetAttribute("nlnum") or 0)
        if parallel_circuits <= 0:
            logger.warning(
                "Line %s has an invalid parallel-circuit count; skipping",
                line.GetAttribute("loc_name"),
            )
            continue

        r_ohm_per_km = read_attribute(line_type, "rline", RESISTANCE_PER_LENGTH_TO_OHM_PER_KM)
        x_ohm_per_km = read_attribute(line_type, "xline", RESISTANCE_PER_LENGTH_TO_OHM_PER_KM)
        b_siemens_per_km = read_attribute(line_type, "bline", ADMITTANCE_PER_LENGTH_TO_S_PER_KM)

        resistance_ohm = r_ohm_per_km * line_length_km / parallel_circuits
        reactance_ohm = x_ohm_per_km * line_length_km / parallel_circuits
        susceptance_us = b_siemens_per_km * line_length_km * parallel_circuits * 1e6

        elements.append(
            LineBranch(
                name=line.GetAttribute("loc_name"),
                from_bus_name=get_bus_full_name(from_bus),
                to_bus_name=get_bus_full_name(to_bus),
                voltage_kv=read_attribute(from_bus, "uknom", VOLTAGE_TO_KV),
                resistance_ohm=resistance_ohm,
                reactance_ohm=reactance_ohm,
                susceptance_us=susceptance_us,
            )
        )

    return elements


def extract_switches(app: pf.Application) -> list[SwitchBranch]:
    """Extract active SwitchBranch objects from PowerFactory."""
    elements: list[SwitchBranch] = []
    pf_switches: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmCoup", 0, 0, 0)
    for switch in pf_switches:
        if switch.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if switch.IsEnergized() != 1:
            continue

        try:
            # Get cubicles
            cub0 = switch.GetCubicle(0)
            cub1 = switch.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Switch '{switch.GetAttribute('loc_name')}': Missing cubicle(s), skipping")
                continue

            # Check if cubicles are closed
            cub0_status = cub0.IsClosed()  # type: ignore
            cub1_status = cub1.IsClosed()  # type: ignore

            if cub0_status != 1 or cub1_status != 1:
                logger.info(f" Switch '{switch.GetAttribute('loc_name')}': One or both cubicles are open, skipping")
                continue

            # Get terminals via cubicles
            from_bus = cub0.GetAttribute("cterm")
            to_bus = cub1.GetAttribute("cterm")
            if from_bus is None or to_bus is None:
                logger.info(
                    f" Switch '{switch.GetAttribute('loc_name')}': Missing terminal(s) (cterm is None), skipping"
                )
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Switch '{switch.GetAttribute('loc_name')}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Switch '{switch.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        is_closed = bool(read_optional_value(switch, "on_off", default=1))

        elements.append(
            SwitchBranch(
                name=switch.GetAttribute("loc_name"),
                from_bus_name=get_bus_full_name(from_bus),
                to_bus_name=get_bus_full_name(to_bus),
                voltage_kv=read_attribute(from_bus, "uknom", VOLTAGE_TO_KV),
                is_closed=is_closed,
            )
        )

    return elements


def extract_common_impedances(app: pf.Application) -> list[CommonImpedanceBranch]:
    """Extract active CommonImpedanceBranch objects from PowerFactory."""
    elements: list[CommonImpedanceBranch] = []
    pf_zpu: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmZpu", 0, 0, 0)
    for zpu in pf_zpu:
        if zpu.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if zpu.IsEnergized() != 1:
            continue

        # Get terminals via cubicles (side 0 and side 1)
        try:
            cub0 = zpu.GetCubicle(0)
            cub1 = zpu.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Common Impedance '{zpu.GetAttribute('loc_name')}': Missing cubicle(s), skipping")
                continue

            # Check if cubicles are closed
            cub0_status = cub0.IsClosed()  # type: ignore
            cub1_status = cub1.IsClosed()  # type: ignore

            if cub0_status != 1 or cub1_status != 1:
                logger.info(
                    f" Common Impedance '{zpu.GetAttribute('loc_name')}': One or both cubicles are open, skipping"
                )
                continue

            from_bus = cub0.GetAttribute("cterm")
            to_bus = cub1.GetAttribute("cterm")
            if from_bus is None or to_bus is None:
                logger.info(
                    f" Common Impedance '{zpu.GetAttribute('loc_name')}': Missing terminal(s) (cterm is None), skipping"
                )
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Common Impedance '{zpu.GetAttribute('loc_name')}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Common Impedance '{zpu.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get voltage levels from terminals
        hv_kv = read_attribute(from_bus, "uknom", VOLTAGE_TO_KV)
        lv_kv = read_attribute(to_bus, "uknom", VOLTAGE_TO_KV)

        # GetImpedance returns [status, R, X] in ohms at the requested voltage.
        imp_pf = zpu.GetImpedance(hv_kv)
        if imp_pf[0] == 1:
            logger.warning(f" Common Impedance '{zpu.GetAttribute('loc_name')}': Error obtaining impedance, skipping")
            continue
        R_ohm = imp_pf[1]
        X_ohm = imp_pf[2]

        # Get rated power (if available)
        rated_mva = read_optional_attribute(zpu, "Sn", POWER_TO_MVA)

        elements.append(
            CommonImpedanceBranch(
                name=zpu.GetAttribute("loc_name"),
                from_bus_name=get_bus_full_name(from_bus),
                to_bus_name=get_bus_full_name(to_bus),
                voltage_kv=hv_kv,
                resistance_ohm=R_ohm,
                reactance_ohm=X_ohm,
                hv_kv=hv_kv,
                lv_kv=lv_kv,
                rated_power_mva=rated_mva,
            )
        )

    return elements


def extract_series_reactors(app: pf.Application) -> list[SeriesReactorBranch]:
    """Extract active SeriesReactorBranch objects from PowerFactory."""
    elements: list[SeriesReactorBranch] = []
    pf_sind: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmSind", 0, 0, 0)
    for sind in pf_sind:
        if sind.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if sind.IsEnergized() != 1:
            continue

        # Get terminals via cubicles (side 0 and side 1)
        try:
            cub0 = sind.GetCubicle(0)
            cub1 = sind.GetCubicle(1)
            if cub0 is None or cub1 is None:
                logger.info(f" Series Reactor '{sind.GetAttribute('loc_name')}': Missing cubicle(s), skipping")
                continue

            # Check if cubicles are closed
            cub0_status = cub0.IsClosed()  # type: ignore
            cub1_status = cub1.IsClosed()  # type: ignore

            if cub0_status != 1 or cub1_status != 1:
                logger.info(
                    f" Series Reactor '{sind.GetAttribute('loc_name')}': One or both cubicles are open, skipping"
                )
                continue

            from_bus = cub0.GetAttribute("cterm")
            to_bus = cub1.GetAttribute("cterm")
            if from_bus is None or to_bus is None:
                logger.info(
                    f" Series Reactor '{sind.GetAttribute('loc_name')}': Missing terminal(s) (cterm is None), skipping"
                )
                continue
            # Check if buses are energized
            if from_bus.IsEnergized() != 1 or to_bus.IsEnergized() != 1:
                logger.info(f" Series Reactor '{sind.GetAttribute('loc_name')}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Series Reactor '{sind.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get voltage level from terminal
        voltage_kv = read_attribute(from_bus, "uknom", VOLTAGE_TO_KV)

        # GetImpedance returns [status, R, X] in ohms at the requested voltage.
        imp_pf = sind.GetImpedance(voltage_kv)
        if imp_pf[0] == 1:
            logger.warning(f" Series Reactor '{sind.GetAttribute('loc_name')}': Error obtaining impedance, skipping")
            continue
        R_ohm = imp_pf[1]
        X_ohm = imp_pf[2]

        # Get rated power (if available)
        rated_mva = read_optional_attribute(sind, "Sn", POWER_TO_MVA)

        elements.append(
            SeriesReactorBranch(
                name=sind.GetAttribute("loc_name"),
                from_bus_name=get_bus_full_name(from_bus),
                to_bus_name=get_bus_full_name(to_bus),
                voltage_kv=voltage_kv,
                resistance_ohm=R_ohm,
                reactance_ohm=X_ohm,
                rated_power_mva=rated_mva,
            )
        )

    return elements


# Transformers


def extract_two_winding_transformers(PFapp: pf.Application) -> list[TransformerBranch]:
    """Extract active TransformerBranch objects from PowerFactory."""

    all_transformers: list[pf.DataObject] = PFapp.GetCalcRelevantObjects("*.ElmTr2", 0, 0, 0)
    extracted_transformers: list[TransformerBranch] = []

    for trafo in all_transformers:
        if trafo.GetAttribute("outserv") == 1:
            continue
        if trafo.IsEnergized() != 1:
            continue

        name = trafo.GetAttribute("loc_name")
        try:
            hv_cubicle = trafo.GetCubicle(0)
            lv_cubicle = trafo.GetCubicle(1)
            if hv_cubicle is None or lv_cubicle is None:
                logger.info(f" 2W-Trafo '{name}': Missing cubicle(s), skipping")
                continue
            if hv_cubicle.IsClosed() != 1 or lv_cubicle.IsClosed() != 1:  # type: ignore
                logger.info(f" 2W-Trafo '{name}': One or both cubicles are open, skipping")
                continue

            hv_terminal: pf.DataObject = hv_cubicle.GetAttribute("cterm")
            lv_terminal: pf.DataObject = lv_cubicle.GetAttribute("cterm")
            if hv_terminal is None or lv_terminal is None:
                logger.info(f" 2-W Trafo '{name}': Missing terminal(s) (cterm is None), skipping")
                continue
            if hv_terminal.IsEnergized() != 1 or lv_terminal.IsEnergized() != 1:
                logger.info(f" 2-W Trafo '{name}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(f" 2-W Trafo '{name}': Failed to get cubicle/terminal - {type(e).__name__}: {e}")
            continue

        # Get transformer type data
        pf_type = trafo.GetAttribute("typ_id")
        if pf_type is None:
            logger.warning(f" 2-W Trafo '{name}': No type data (typ_id is None), skipping")
            continue

        # Rated values from type
        rated_mva = read_attribute(pf_type, "strn", POWER_TO_MVA)
        hv_kv = read_attribute(pf_type, "utrn_h", VOLTAGE_TO_KV)
        lv_kv = read_attribute(pf_type, "utrn_l", VOLTAGE_TO_KV)

        # Impedance from type (uk = short-circuit voltage %, ur = resistive part %)
        uk_percent = read_attribute(pf_type, "uktr", PERCENT_TO_PERCENT)
        ur_percent = read_optional_attribute(pf_type, "uktrr", PERCENT_TO_PERCENT)

        # Convert to per-unit on transformer base
        x_pu = uk_percent / 100.0  # Approximate: X ≈ uk for small R
        r_pu = ur_percent / 100.0
        # More accurate: X = sqrt(uk² - ur²)
        if uk_percent > ur_percent:
            x_pu = ((uk_percent**2 - ur_percent**2) ** 0.5) / 100.0

        # Get tap changer parameters
        tap_pos = int(read_optional_value(trafo, "nntap"))
        tap_side = int(read_optional_value(pf_type, "tap_side"))
        nntap0 = int(read_optional_value(pf_type, "nntap0"))
        ntpmn = int(read_optional_value(pf_type, "ntpmn"))
        ntpmx = int(read_optional_value(pf_type, "ntpmx"))

        # Get tap changer type (0 = Ratio/Asym, 1 = Ideal phase, 2 = Sym phase)
        tapchtype = int(read_optional_value(pf_type, "tapchtype"))

        # Create appropriate tap changer based on type
        tap_changer: TapChanger
        if tapchtype == 1:  # Ideal phase shifter
            dphitap = read_optional_attribute(pf_type, "dphitap", ANGLE_TO_DEG)
            tap_changer = IdealPhaseTapChanger(
                tap_side=tap_side,
                nntap0=nntap0,
                ntpmn=ntpmn,
                ntpmx=ntpmx,
                dphitap=dphitap,
            )
        elif tapchtype == 2:  # Symmetric phase shifter
            dutap = read_optional_attribute(pf_type, "dutap", PERCENT_TO_PERCENT)
            phitr = read_optional_attribute(pf_type, "phitr", ANGLE_TO_DEG)
            tap_changer = SymPhaseTapChanger(
                tap_side=tap_side,
                nntap0=nntap0,
                ntpmn=ntpmn,
                ntpmx=ntpmx,
                dutap=dutap,
                phitr=phitr,
            )
        else:  # tapchtype == 0: Ratio/Asymmetric phase shifter (default)
            dutap = read_optional_attribute(pf_type, "dutap", PERCENT_TO_PERCENT)
            phitr = read_optional_attribute(pf_type, "phitr", ANGLE_TO_DEG)
            tap_changer = RatioAsymTapChanger(
                tap_side=tap_side,
                nntap0=nntap0,
                ntpmn=ntpmn,
                ntpmx=ntpmx,
                dutap=dutap,
                phitr=phitr,
            )

        # Get vector group phase shift
        # Phase shift in degrees = nt2ag * 30°
        nt2ag = read_optional_value(pf_type, "nt2ag")
        vector_group_shift_deg = nt2ag * 30

        # Number of parallel transformers
        n_parallel = int(read_optional_value(trafo, "ntnum", default=1))
        extracted_transformers.append(
            TransformerBranch(
                name=trafo.GetAttribute("loc_name"),
                from_bus_name=get_bus_full_name(hv_terminal),
                to_bus_name=get_bus_full_name(lv_terminal),
                voltage_kv=hv_kv,  # Use HV side as reference
                rated_power_mva=rated_mva,
                hv_kv=hv_kv,
                lv_kv=lv_kv,
                resistance_pu=r_pu,
                reactance_pu=x_pu,
                tap_changer=tap_changer,
                tap_pos=tap_pos,
                vector_group_shift_deg=vector_group_shift_deg,
                n_parallel=n_parallel,
            )
        )

    return extracted_transformers


def extract_three_winding_transformers(
    app: pf.Application,
) -> list[Transformer3WBranch]:
    """Extract active Transformer3WBranch objects from PowerFactory."""
    elements: list[Transformer3WBranch] = []
    pf_trafos_3w: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmTr3", 0, 0, 0)
    for trafo in pf_trafos_3w:
        if trafo.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if trafo.IsEnergized() != 1:
            continue

        # Get terminals via cubicles (HV = 0, MV = 1, LV = 2)
        try:
            cub0 = trafo.GetAttribute("bushv")
            cub1 = trafo.GetAttribute("busmv")
            cub2 = trafo.GetAttribute("buslv")
            if cub0 is None or cub1 is None or cub2 is None:
                logger.info(f" 3W Transformer '{trafo.GetAttribute('loc_name')}': Missing cubicle(s), skipping")
                continue

            hv_bus = cub0.GetAttribute("cterm")
            mv_bus = cub1.GetAttribute("cterm")
            lv_bus = cub2.GetAttribute("cterm")

            if hv_bus is None or mv_bus is None or lv_bus is None:
                logger.info(
                    f" 3W Transformer '{trafo.GetAttribute('loc_name')}': Missing terminal(s) (cterm is None), skipping"
                )
                continue
            # # Check if buses are energized
            if hv_bus.IsEnergized() != 1 and mv_bus.IsEnergized() != 1 and lv_bus.IsEnergized() != 1:
                logger.info(f" 3W Transformer '{trafo.GetAttribute('loc_name')}': Bus(es) de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" 3W Transformer '{trafo.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get transformer type data
        pf_type = trafo.GetAttribute("typ_id")

        # Initialize default values
        rated_power_hv = 0.0
        rated_power_mv = 0.0
        rated_power_lv = 0.0
        hv_kv = 0.0
        mv_kv = 0.0
        lv_kv = 0.0
        uk_hm = 0.0
        uk_ml = 0.0
        uk_lh = 0.0
        ukr_hm = 0.0
        ukr_ml = 0.0
        ukr_lh = 0.0

        if pf_type is not None:
            # Rated powers for each winding (MVA)
            rated_power_hv = read_optional_attribute(pf_type, "strn3_h", POWER_TO_MVA)
            rated_power_mv = read_optional_attribute(pf_type, "strn3_m", POWER_TO_MVA)
            rated_power_lv = read_optional_attribute(pf_type, "strn3_l", POWER_TO_MVA)

            # Rated voltages for each winding (kV)
            hv_kv = read_optional_attribute(pf_type, "utrn3_h", VOLTAGE_TO_KV)
            mv_kv = read_optional_attribute(pf_type, "utrn3_m", VOLTAGE_TO_KV)
            lv_kv = read_optional_attribute(pf_type, "utrn3_l", VOLTAGE_TO_KV)

            # Short-circuit voltages (uk) in % for each pair
            # uktr3_h: HV-MV pair, uktr3_m: MV-LV pair, uktr3_l: LV-HV pair
            uk_hm = read_optional_attribute(pf_type, "uktr3_h", PERCENT_TO_PERCENT)
            uk_ml = read_optional_attribute(pf_type, "uktr3_m", PERCENT_TO_PERCENT)
            uk_lh = read_optional_attribute(pf_type, "uktr3_l", PERCENT_TO_PERCENT)

            # Real parts of short-circuit voltages (ukr) in % for each pair
            ukr_hm = read_optional_attribute(pf_type, "uktrr3_h", PERCENT_TO_PERCENT)
            ukr_ml = read_optional_attribute(pf_type, "uktrr3_m", PERCENT_TO_PERCENT)
            ukr_lh = read_optional_attribute(pf_type, "uktrr3_l", PERCENT_TO_PERCENT)

        # Get HV side tap changer parameters
        tap_changer_hv: TapChanger | None = None
        tap_pos_hv = 0
        if pf_type is not None:
            # Get tap position from transformer element
            tap_pos_hv = int(read_optional_value(trafo, "n3tap_h"))

            # Get tap changer parameters from type
            du3tp_h = read_optional_attribute(pf_type, "du3tp_h", PERCENT_TO_PERCENT)
            ph3tr_h = read_optional_attribute(pf_type, "ph3tr_h", ANGLE_TO_DEG)
            n3tp0_h = int(read_optional_value(pf_type, "n3tp0_h"))
            n3tmn_h = int(read_optional_value(pf_type, "n3tmn_h"))
            n3tmx_h = int(read_optional_value(pf_type, "n3tmx_h"))

            # Create RatioAsymTapChanger for HV side
            tap_changer_hv = RatioAsymTapChanger(
                tap_side=0,  # HV side
                nntap0=n3tp0_h,
                ntpmn=n3tmn_h,
                ntpmx=n3tmx_h,
                dutap=du3tp_h,
                phitr=ph3tr_h,
            )

        # Get vector group phase shifts
        # Phase shift in degrees = nt3ag * 30°
        nt3ag_h = read_optional_value(pf_type, "nt3ag_h") if pf_type else 0
        nt3ag_m = read_optional_value(pf_type, "nt3ag_m") if pf_type else 0
        nt3ag_l = read_optional_value(pf_type, "nt3ag_l") if pf_type else 0

        vector_group_shift_deg_hv = nt3ag_h * 30
        vector_group_shift_deg_mv = nt3ag_m * 30
        vector_group_shift_deg_lv = nt3ag_l * 30

        # Number of parallel transformers
        n_parallel = int(read_optional_value(trafo, "ntnum", default=1))

        elements.append(
            Transformer3WBranch(
                name=trafo.GetAttribute("loc_name"),
                hv_bus_name=get_bus_full_name(hv_bus),
                mv_bus_name=get_bus_full_name(mv_bus),
                lv_bus_name=get_bus_full_name(lv_bus),
                base_mva=100.0,
                n_parallel=n_parallel,
                rated_power_hv_mva=rated_power_hv,
                rated_power_mv_mva=rated_power_mv,
                rated_power_lv_mva=rated_power_lv,
                hv_kv=hv_kv,
                mv_kv=mv_kv,
                lv_kv=lv_kv,
                uk_hm_percent=uk_hm,
                uk_ml_percent=uk_ml,
                uk_lh_percent=uk_lh,
                ukr_hm_percent=ukr_hm,
                ukr_ml_percent=ukr_ml,
                ukr_lh_percent=ukr_lh,
                tap_changer_hv=tap_changer_hv,
                tap_pos_hv=tap_pos_hv,
                vector_group_shift_deg_hv=vector_group_shift_deg_hv,
                vector_group_shift_deg_mv=vector_group_shift_deg_mv,
                vector_group_shift_deg_lv=vector_group_shift_deg_lv,
            )
        )

    return elements


# Loads, sources, and shunt equipment


def extract_generators(app: pf.Application) -> list[GeneratorShunt]:
    """Extract active GeneratorShunt objects from PowerFactory."""
    elements: list[GeneratorShunt] = []
    pf_gens: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmSym", 0, 0, 0)
    for gen in pf_gens:
        if gen.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if gen.IsEnergized() != 1:
            continue

        try:
            cub0 = gen.GetCubicle(0)
            if cub0 is None:
                logger.info(f" Generator '{gen.GetAttribute('loc_name')}': Missing cubicle, skipping")
                continue

            # Check if cubicle is closed
            cub0_status = cub0.IsClosed()  # type: ignore
            if cub0_status != 1:
                logger.info(f" Generator '{gen.GetAttribute('loc_name')}': Cubicle is open, skipping")
                continue

            bus = cub0.GetAttribute("cterm")
            if bus is None:
                logger.info(f" Generator '{gen.GetAttribute('loc_name')}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" Generator '{gen.GetAttribute('loc_name')}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Generator '{gen.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        pf_type = gen.GetAttribute("typ_id")
        if pf_type is None:
            logger.warning(
                "Generator %s has no assigned type; skipping",
                gen.GetAttribute("loc_name"),
            )
            continue

        rated_mva = read_attribute(pf_type, "sgn", POWER_TO_MVA)
        rated_kv = read_attribute(pf_type, "ugn", VOLTAGE_TO_KV)
        n_parallel = int(read_optional_value(gen, "ngnum", default=1))

        # Read generator model in PF
        model = pf_type.GetAttribute("model_inp")
        if model == "cls":
            # Classical model
            rstr = read_optional_attribute(pf_type, "rstr", PERCENT_OR_PER_UNIT)
            xstr = read_attribute(pf_type, "xstr", PERCENT_OR_PER_UNIT)
            z_pu = complex(rstr, xstr)
        elif model == "det":
            # Standard model
            rstr = read_optional_attribute(pf_type, "rstr", PERCENT_OR_PER_UNIT)
            xdss = read_attribute(pf_type, "xdss", PERCENT_OR_PER_UNIT)
            xqss = read_attribute(pf_type, "xqss", PERCENT_OR_PER_UNIT)
            # Preserve the existing d-axis subtransient approximation.
            z_pu = complex(rstr, (xdss + xqss) / 2)
        else:
            # Default to classical model
            rstr = read_optional_attribute(pf_type, "rstr", PERCENT_OR_PER_UNIT)
            xstr = read_attribute(pf_type, "xstr", PERCENT_OR_PER_UNIT)
            z_pu = complex(rstr, xstr)
            logger.info(
                f" Generator '{gen.GetAttribute('loc_name')}': Unknown model '{model}', defaulting to classical model"
            )

        # Get generator Zone name
        zone = gen.GetAttribute("cpZone")
        zone_name = zone.GetAttribute("loc_name") if zone is not None else "None"

        # Get generator Grid name
        grid = gen.GetAttribute("cpGrid")
        grid_name = grid.GetAttribute("loc_name") if grid is not None else "None"

        elements.append(
            GeneratorShunt(
                name=gen.GetAttribute("loc_name"),
                bus_name=get_bus_full_name(bus),
                voltage_kv=read_attribute(bus, "uknom", VOLTAGE_TO_KV),
                rated_power_mva=rated_mva,
                rated_voltage_kv=rated_kv,
                z_pu=z_pu,
                n_parallel=n_parallel,
                zone=zone_name,
                grid_code=grid_name,
            )
        )

    return elements


def extract_loads(app: pf.Application) -> list[LoadShunt]:
    """Extract active LoadShunt objects from PowerFactory."""
    elements: list[LoadShunt] = []
    pf_loads: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmLod", 0, 0, 0)
    for load in pf_loads:
        if load.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if load.IsEnergized() != 1:
            continue

        try:
            cub0 = load.GetCubicle(0)
            load_name = load.GetAttribute("loc_name")
            if cub0 is None:
                logger.info(f" Load '{load_name}': Missing cubicle, skipping")
                continue

            cub0_status = cub0.IsClosed()  # type: ignore
            if cub0_status != 1:
                logger.info(f" Load '{load_name}': Cubicle is open, skipping")
                continue

            bus = cub0.GetAttribute("cterm")
            if bus is None:
                logger.info(f" Load '{load_name}': Missing terminal (cterm is None), skipping")
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" Load '{load_name}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Load '{load.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get load dynamic simulation model (# TODO: Add more load models)
        ldtype = load.GetAttribute("typ_id")
        if ldtype is not None:
            # Check for constant impedance load model
            lodst = read_optional_value(ldtype, "lodst")
            if lodst == 100:
                load_model = LoadModelType.CONSTANT_IMPEDANCE
            else:
                load_model = LoadModelType.CONSTANT_POWER
        else:
            load_model = LoadModelType.CONSTANT_IMPEDANCE  # Default to constant impedance

        elements.append(
            LoadShunt(
                name=load_name,
                bus_name=get_bus_full_name(bus),
                voltage_kv=read_attribute(bus, "uknom", VOLTAGE_TO_KV),
                p_mw=read_attribute(load, "plini", POWER_TO_MW) * read_optional_value(load, "scale0", default=1),
                q_mvar=read_attribute(load, "qlini", POWER_TO_MVAR) * read_optional_value(load, "scale0", default=1),
                load_model=load_model,
            )
        )

    return elements


def extract_external_grids(app: pf.Application) -> list[ExternalGridShunt]:
    """Extract active ExternalGridShunt objects from PowerFactory."""
    elements: list[ExternalGridShunt] = []
    pf_xnets: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmXnet", 0, 0, 0)
    for xnet in pf_xnets:
        if xnet.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if xnet.IsEnergized() != 1:
            continue

        try:
            cub0 = xnet.GetCubicle(0)
            if cub0 is None:
                logger.info(f" External Grid '{xnet.GetAttribute('loc_name')}': Missing cubicle, skipping")
                continue
            bus = cub0.GetAttribute("cterm")
            if bus is None:
                logger.info(
                    f" External Grid '{xnet.GetAttribute('loc_name')}': Missing terminal (cterm is None), skipping"
                )
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" External Grid '{xnet.GetAttribute('loc_name')}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" External Grid '{xnet.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get short-circuit parameters
        s_sc = read_optional_attribute(xnet, "snss", POWER_TO_MVA)
        c_factor = read_optional_value(xnet, "cfac", default=1)
        r_x_ratio = read_optional_value(xnet, "rntxn", default=0.1)

        elements.append(
            ExternalGridShunt(
                name=xnet.GetAttribute("loc_name"),
                bus_name=get_bus_full_name(bus),
                voltage_kv=read_attribute(bus, "uknom", VOLTAGE_TO_KV),
                s_sc_mva=s_sc,
                c_factor=c_factor,
                r_x_ratio=r_x_ratio,
            )
        )

    return elements


def extract_voltage_sources(app: pf.Application) -> list[VoltageSourceShunt]:
    """Extract active VoltageSourceShunt objects from PowerFactory."""
    elements: list[VoltageSourceShunt] = []
    pf_vacs: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmVac", 0, 0, 0)
    for vac in pf_vacs:
        if vac.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if vac.IsEnergized() != 1:
            continue

        try:
            cub0 = vac.GetCubicle(0)
            if cub0 is None:
                logger.info(f" AC Voltage Source '{vac.GetAttribute('loc_name')}': Missing cubicle, skipping")
                continue
            bus = cub0.GetAttribute("cterm")
            if bus is None:
                logger.info(
                    f" AC Voltage Source '{vac.GetAttribute('loc_name')}': Missing terminal (cterm is None), skipping"
                )
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" AC Voltage Source '{vac.GetAttribute('loc_name')}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" AC Voltage Source '{vac.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get R and X values (in ohms)
        r_ohm = read_optional_attribute(vac, "R1", RESISTANCE_TO_OHM)
        x_ohm = read_optional_attribute(vac, "X1", RESISTANCE_TO_OHM)

        elements.append(
            VoltageSourceShunt(
                name=vac.GetAttribute("loc_name"),
                bus_name=get_bus_full_name(bus),
                voltage_kv=read_attribute(bus, "uknom", VOLTAGE_TO_KV),
                resistance_ohm=r_ohm,
                reactance_ohm=x_ohm,
            )
        )

    return elements


def extract_pv_systems(app: pf.Application) -> list[PVSystemShunt]:
    """Extract active PVSystemShunt objects from PowerFactory."""
    elements: list[PVSystemShunt] = []
    pf_pvsys: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmPvsys", 0, 0, 0)
    for pvsys in pf_pvsys:
        if pvsys.GetAttribute("outserv") == 1:
            continue

        if pvsys.IsEnergized() != 1:
            continue

        try:
            cub0 = pvsys.GetCubicle(0)
            if cub0 is None:
                logger.info(f" PV System '{pvsys.GetAttribute('loc_name')}': Missing cubicle, skipping")
                continue

            cub0_status = cub0.IsClosed()  # type: ignore
            if cub0_status != 1:
                logger.info(f" PV System '{pvsys.GetAttribute('loc_name')}': Cubicle is open, skipping")
                continue

            bus = cub0.GetAttribute("cterm")
            if bus is None:
                logger.info(
                    f" PV System '{pvsys.GetAttribute('loc_name')}': Missing terminal (cterm is None), skipping"
                )
                continue
            if bus.IsEnergized() != 1:
                logger.info(f" PV System '{pvsys.GetAttribute('loc_name')}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" PV System '{pvsys.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        rated_mva = read_optional_attribute(pvsys, "sgn", POWER_TO_MVA)
        rated_kv = read_attribute(bus, "uknom", VOLTAGE_TO_KV)
        uk_percent = read_optional_attribute(pvsys, "uk", PERCENT_TO_PERCENT)
        try:
            copper_losses_kw = read_attribute(pvsys, "Pcu", POWER_TO_KW)
        except (AttributeError, TypeError):
            copper_losses_kw = read_optional_attribute(pvsys, "pcu", POWER_TO_KW)

        if rated_mva <= 0 or rated_kv <= 0 or uk_percent <= 0:
            logger.info(
                f" PV System '{pvsys.GetAttribute('loc_name')}': Missing rated power/voltage or short-circuit impedance, skipping"
            )
            continue

        elements.append(
            PVSystemShunt(
                name=pvsys.GetAttribute("loc_name"),
                bus_name=get_bus_full_name(bus),
                voltage_kv=read_attribute(bus, "uknom", VOLTAGE_TO_KV),
                rated_power_mva=rated_mva,
                rated_voltage_kv=rated_kv,
                uk_percent=uk_percent,
                copper_losses_kw=copper_losses_kw,
            )
        )

    return elements


def extract_shunt_filters(app: pf.Application) -> list[ShuntFilterShunt]:
    """Extract active ShuntFilterShunt objects from PowerFactory."""
    elements: list[ShuntFilterShunt] = []
    pf_shunts: list[pf.DataObject] = app.GetCalcRelevantObjects("*.ElmShnt", 0, 0, 0)
    for shnt in pf_shunts:
        if shnt.GetAttribute("outserv") == 1:
            continue

        # Check if element is energized
        if shnt.IsEnergized() != 1:
            continue

        try:
            cub0 = shnt.GetCubicle(0)
            if cub0 is None:
                logger.info(f" Shunt Filter '{shnt.GetAttribute('loc_name')}': Missing cubicle, skipping")
                continue

                # Check if cubicle is closed
            cub0_status = cub0.IsClosed()  # type: ignore
            if cub0_status != 1:
                logger.info(f" Shunt Filter '{shnt.GetAttribute('loc_name')}': Cubicle is open, skipping")
                continue

            bus = cub0.GetAttribute("cterm")
            if bus is None:
                logger.info(
                    f" Shunt Filter '{shnt.GetAttribute('loc_name')}': Missing terminal (cterm is None), skipping"
                )
                continue
            # Check if bus is energized
            if bus.IsEnergized() != 1:
                logger.info(f" Shunt Filter '{shnt.GetAttribute('loc_name')}': Bus de-energized, skipping")
                continue
        except Exception as e:
            logger.warning(
                f" Shunt Filter '{shnt.GetAttribute('loc_name')}': Failed to get cubicle/terminal - {type(e).__name__}: {e}"
            )
            continue

        # Get filter type
        shtype = int(read_optional_value(shnt, "shtype", default=2))
        try:
            filter_type = ShuntFilterType(shtype)
        except ValueError:
            filter_type = ShuntFilterType.C

        # Get actual power output (this is what matters for Y-matrix)
        # Qact is the actual reactive power at current operating point
        q_mvar = read_optional_attribute(shnt, "Qact", POWER_TO_MVAR)
        p_mw = read_optional_attribute(shnt, "Pact", POWER_TO_MW)

        # Controller parameters
        ncapx = int(read_optional_value(shnt, "ncapx", default=1))
        ncapa = int(read_optional_value(shnt, "ncapa", default=1))
        nreax = int(read_optional_value(shnt, "nreax", default=1))
        nreaa = int(read_optional_value(shnt, "nreaa", default=1))

        # Design parameters (for reference)
        qtotn_mvar = read_optional_attribute(shnt, "qtotn", POWER_TO_MVAR)
        qrean_mvar = read_optional_attribute(shnt, "qrean", POWER_TO_MVAR)
        fres_hz = read_optional_attribute(shnt, "fres", FREQUENCY_TO_HZ)

        # Quality factor - different attribute names for different types
        if filter_type == ShuntFilterType.R_L_C:
            quality_factor = read_optional_value(shnt, "greaf0")
        elif filter_type == ShuntFilterType.R_L:
            quality_factor = read_optional_value(shnt, "grea")
        else:
            quality_factor = 0.0

        # Layout parameters per step (for detailed modeling if needed)
        bcap_us = read_optional_attribute(shnt, "bcap", SUSCEPTANCE_TO_US)
        xrea_ohm = read_optional_attribute(shnt, "xrea", RESISTANCE_TO_OHM)
        rrea_ohm = read_optional_attribute(shnt, "rrea", RESISTANCE_TO_OHM)

        elements.append(
            ShuntFilterShunt(
                name=shnt.GetAttribute("loc_name"),
                bus_name=get_bus_full_name(bus),
                voltage_kv=read_attribute(bus, "uknom", VOLTAGE_TO_KV),
                filter_type=filter_type,
                q_mvar=q_mvar,
                p_mw=p_mw,
                ncapx=ncapx,
                ncapa=ncapa,
                nreax=nreax,
                nreaa=nreaa,
                qtotn_mvar=qtotn_mvar,
                qrean_mvar=qrean_mvar,
                fres_hz=fres_hz,
                quality_factor=quality_factor,
                bcap_us=bcap_us,
                xrea_ohm=xrea_ohm,
                rrea_ohm=rrea_ohm,
            )
        )

    return elements


# Bus selection


def get_main_bus_names(app: pf.Application) -> set[str]:
    """
    Get names of main busbars (terminals with iUsage == 0) from PowerFactory.

    In PowerFactory, iUsage indicates terminal usage type:
    - 0: Busbar (main busbar)
    - 1: Junction node
    - 2: Internal node

    Args:
        app: PowerFactory application instance

    Returns:
        Set of bus names that are main busbars
    """
    main_buses: set[str] = set()

    # Get all terminals
    terminals = app.GetCalcRelevantObjects("*.ElmTerm", 0, 0, 0)

    for term in terminals:
        # Skip out-of-service or de-energized terminals
        if term.GetAttribute("outserv") == 1:
            continue
        if term.IsEnergized() != 1:
            continue

        # Check iUsage - 0 means main busbar
        usage = read_optional_value(term, "iUsage", default=1)
        if usage == 0:
            main_buses.add(get_bus_full_name(term))

    return main_buses


# Unit-aware attribute readers


def read_attribute(
    obj: pf.DataObject,
    attribute: str,
    conversions: dict[str, float],
) -> float:
    """Read a PowerFactory numeric attribute in the requested SI-derived unit."""
    value = float(obj.GetAttribute(attribute))
    unit = obj.GetAttributeUnit(attribute)
    try:
        factor = conversions[unit]
    except KeyError:
        raise ValueError(f"Unsupported unit for {attribute}: {unit!r}") from None
    return value * factor


def read_optional_attribute(
    obj: pf.DataObject,
    attribute: str,
    conversions: dict[str, float],
    *,
    default: float = 0.0,
) -> float:
    """Read an optional unit-bearing attribute without hiding unknown units."""
    try:
        value = obj.GetAttribute(attribute)
    except AttributeError:
        return default
    if value is None:
        return default
    return read_attribute(obj, attribute, conversions)


def read_optional_value(obj: pf.DataObject, attribute: str, *, default: float = 0.0) -> float:
    """Read an optional dimensionless numeric attribute."""
    try:
        value = obj.GetAttribute(attribute)
    except AttributeError:
        return default
    return default if value is None else float(value)
