"""Static-generator source modeling without a PowerFactory installation."""

import math

import numpy as np
import pytest

from src.adapters.powerfactory import extractor
from src.adapters.powerfactory.load_flow import get_static_generator_data_from_pf
from src.network import OperatingPoint
from src.network.elements import StaticGeneratorShunt
from src.network.operating_point import BusResult, VoltageSourceResult
from src.outage_analysis import AngleUpdateFormulation, ReducedMatrixFormulation, SynchroCoefficients

from test_outage_analysis import example_formulation


def test_static_generator_impedance_uses_only_virtual_impedance() -> None:
    source = StaticGeneratorShunt(
        name="VSM",
        bus_name="C",
        voltage_kv=110.0,
        virtual_impedance_pu=0.01 + 0.03j,
        virtual_impedance_base_mva=100.0,
    )
    expected_z_ohm = (0.01 + 0.03j) * 110.0**2 / 100.0
    np.testing.assert_allclose(source.admittance, 1 / expected_z_ohm)
    np.testing.assert_allclose(source.get_admittance_pu(100.0), 1 / (0.01 + 0.03j))

    with pytest.raises(ValueError, match="cannot be zero"):
        StaticGeneratorShunt(
            name="Zero impedance",
            bus_name="C",
            voltage_kv=110.0,
            virtual_impedance_pu=0j,
            virtual_impedance_base_mva=100.0,
        )


def test_static_generator_responds_in_all_formulations() -> None:
    original = example_formulation()
    network = original.network
    source = StaticGeneratorShunt(
        name="VSM",
        bus_name="C",
        voltage_kv=110.0,
        virtual_impedance_pu=0.02j,
        virtual_impedance_base_mva=100.0,
    )
    network.add_shunt(source)
    operating_point = OperatingPoint(
        base_mva=100.0,
        buses=original.operating_point.buses,
        sources=original.operating_point.sources[:3]
        + (
            VoltageSourceResult(
                name="VSM",
                bus_name="C",
                terminal_voltage=1 + 0j,
                impedance_pu=1 / source.get_admittance_pu(100.0),
                p_pu=0.2,
                q_pu=0.0,
                internal_voltage=1.01 + 0.04j,
                internal_voltage_mag=abs(1.01 + 0.04j),
                internal_voltage_angle=math.degrees(np.angle(1.01 + 0.04j)),
                source_type="static_generator",
            ),
        )
        + original.operating_point.sources[3:],
    )
    assert network.static_generator_names == ("VSM",)
    assert network.stability_source_names == operating_point.source_names
    assert "VSM" in ReducedMatrixFormulation(network, operating_point).calculate_result("G1").power_change_mw
    for method in (SynchroCoefficients, AngleUpdateFormulation, ReducedMatrixFormulation):
        model = method(network, operating_point)
        assert np.isfinite(model.calculate("G1")["VSM"])
        assert np.isfinite(model.calculate("VSM")["G1"])
    reduced_without_vsm = network.get_stability_y_matrix(operating_point, excluded_sources={"VSM"})
    assert reduced_without_vsm.shape == (len(network.stability_sources) - 1,) * 2
    flows = ReducedMatrixFormulation(network, operating_point).calculate_all_branch_flows(("AB",))
    assert set(flows) == set(network.generator_names)


class FakeObject:
    def __init__(self, attributes: dict[str, object], units: dict[str, str] | None = None):
        self.attributes = attributes
        self.units = units or {}

    def GetAttribute(self, name: str) -> object:
        if name not in self.attributes:
            raise AttributeError(name)
        return self.attributes[name]

    def GetAttributeUnit(self, name: str) -> str:
        return self.units[name]

    def IsEnergized(self) -> int:
        return 1

    def IsClosed(self) -> int:
        return 1

    def GetCubicle(self, index: int) -> "FakeObject":
        assert index == 0
        return self.attributes["cubicle"]


class FakeApp:
    def __init__(self, generators: list[FakeObject]):
        self.generators = generators

    def GetCalcRelevantObjects(self, pattern: str, *flags: int) -> list[FakeObject]:
        assert pattern == "*.ElmGenstat"
        return self.generators


def test_powerfactory_adapter_extracts_only_vsm_and_initializes_voltage(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = FakeObject({"uknom": 110.0}, {"uknom": "kV"})
    cubicle = FakeObject({"cterm": bus})
    block = FakeObject({"loc_name": "Virtual impedance"})
    parameters = FakeObject({"params": [0.01, 0.03]})
    composite = FakeObject({"pblk": [block], "pelm": [parameters]})
    units = {
        "m:P:bus1": "MW",
        "m:Q:bus1": "Mvar",
    }
    vsm = FakeObject(
        {
            "loc_name": "VSM",
            "outserv": 0,
            "iSimModel": 2,
            "cubicle": cubicle,
            "c_pmod": composite,
            "m:P:bus1": 20.0,
            "m:Q:bus1": 3.0,
        },
        units,
    )
    other = FakeObject(
        {"loc_name": "Other", "outserv": 0, "iSimModel": 1, "cubicle": cubicle, "c_pmod": composite},
        units,
    )
    app = FakeApp([vsm, other])
    monkeypatch.setattr(extractor, "get_bus_full_name", lambda _: "C")

    sources = extractor.extract_static_generators(app, base_mva=100.0)
    assert [source.name for source in sources] == ["VSM"]
    assert sources[0].virtual_impedance_pu == 0.01 + 0.03j

    results = get_static_generator_data_from_pf(app, sources, {"C": BusResult("C", 1.0, 0.0, 110.0)})
    assert len(results) == 1
    result = results[0]
    assert result.source_type == "static_generator"
    expected_e = 1 + (1 / sources[0].get_admittance_pu(100.0)) * (0.2 - 0.03j)
    np.testing.assert_allclose(result.internal_voltage, expected_e)
