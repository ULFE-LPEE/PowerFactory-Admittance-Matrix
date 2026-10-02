"""Numerical regression checks for PowerFactory-independent outage methods."""

import numpy as np

from src.network import Network, OperatingPoint
from src.network.elements import ExternalGridShunt, GeneratorShunt, LineBranch
from src.network.operating_point import BusResult, ExternalGridResult, GeneratorResult
from src.outage_analysis import ReducedMatrixFormulation


def example_formulation() -> ReducedMatrixFormulation:
    """Three-bus network with two generators sharing a terminal bus."""
    network = Network(base_mva=100.0)
    for name, start, end, resistance, reactance in (
        ("AB", "A", "B", 1.1, 5.0),
        ("BC", "B", "C", 1.3, 6.0),
        ("CA", "C", "A", 1.8, 7.0),
    ):
        network.add_branch(
            LineBranch(
                name=name,
                from_bus_name=start,
                to_bus_name=end,
                voltage_kv=110.0,
                resistance_ohm=resistance,
                reactance_ohm=reactance,
            )
        )

    generators = (("G1", "A", 1.05 + 0.03j), ("G2", "B", 1.02 - 0.01j), ("G3", "B", 1.01 + 0.02j))
    for name, bus, _ in generators:
        network.add_shunt(
            GeneratorShunt(
                name=name,
                bus_name=bus,
                voltage_kv=110.0,
                rated_power_mva=100.0,
                rated_voltage_kv=110.0,
                z_pu=0.01 + 0.2j,
            )
        )
    network.add_shunt(ExternalGridShunt(name="Grid", bus_name="C", voltage_kv=110.0, s_sc_mva=1000.0))

    sources = tuple(
        GeneratorResult(
            name=name,
            bus_name=bus,
            terminal_voltage=1.0 + 0.0j,
            impedance_pu=0.01 + 0.2j,
            p_pu=0.0,
            q_pu=0.0,
            internal_voltage=voltage,
            rated_mva=100.0,
            rated_kv=110.0,
        )
        for name, bus, voltage in generators
    ) + (
        ExternalGridResult(
            name="Grid",
            bus_name="C",
            terminal_voltage=1.0 + 0.0j,
            impedance_pu=0.01 + 0.1j,
            p_pu=0.0,
            q_pu=0.0,
            internal_voltage=1.0 + 0.0j,
            internal_voltage_mag=1.0,
            internal_voltage_angle=0.0,
        ),
    )
    operating_point = OperatingPoint(
        base_mva=100.0,
        buses={name: BusResult(name, 1.0, 0.0, 110.0) for name in ("A", "B", "C")},
        sources=sources,
    )
    return ReducedMatrixFormulation(network, operating_point)


def test_calculate_all_matches_single_generator_outages() -> None:
    formulation = example_formulation()
    all_results = formulation.calculate_all()

    assert tuple(all_results) == formulation.network.generator_names
    for name in formulation.network.generator_names:
        individual = formulation.calculate_result(name)
        assert all_results[name].source_names == individual.source_names
        np.testing.assert_allclose(
            list(all_results[name].power_change_mw.values()),
            list(individual.power_change_mw.values()),
            rtol=1e-12,
            atol=1e-12,
        )


def test_batch_branch_flows_match_single_outage_solves() -> None:
    formulation = example_formulation()
    branches = ("AB", "BC", "CA")
    batch = formulation.calculate_all_branch_flows(branches)

    assert tuple(batch) == formulation.network.generator_names
    for generator in formulation.network.generator_names:
        individual = formulation.calculate_branch_flows(generator, branches)
        for branch in branches:
            np.testing.assert_allclose(
                [batch[generator][branch].prefault_mw, batch[generator][branch].postfault_mw],
                [individual[branch].prefault_mw, individual[branch].postfault_mw],
                rtol=1e-10,
                atol=1e-9,
            )
