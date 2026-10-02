"""Passive and load-flow admittance matrices on physical network buses."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..network.elements import LoadModelType, LoadShunt, Shunt, ShuntFilterShunt

if TYPE_CHECKING:
    from ..network import Network, OperatingPoint


def build_passive_y_matrix(network: Network) -> npt.NDArray[np.complex128]:
    """Include branches, transformers, and passive filters, but no loads or sources."""
    filters = [shunt for shunt in network.shunts if isinstance(shunt, ShuntFilterShunt)]
    return _assemble_physical_bus_y(network, filters)


def build_load_flow_y_matrix(
    network: Network,
    operating_point: OperatingPoint | None = None,
) -> npt.NDArray[np.complex128]:
    """Add constant-impedance loads to the passive physical-bus matrix.

    With a solved operating point, each load uses that bus's voltage on a copy
    of the element. The network's stored load parameters are not changed.
    """
    if operating_point is not None and not np.isclose(network.base_mva, operating_point.base_mva):
        raise ValueError("Operating-point and network base MVA must match.")

    shunts: list[Shunt] = []
    for shunt in network.shunts:
        if isinstance(shunt, ShuntFilterShunt):
            shunts.append(shunt)
        elif isinstance(shunt, LoadShunt) and shunt.load_model == LoadModelType.CONSTANT_IMPEDANCE:
            load = copy(shunt)
            if operating_point is not None:
                bus = operating_point.buses.get(load.bus_name)
                if bus is not None:
                    load.set_lf_voltage(bus.voltage_pu)
            shunts.append(load)

    return _assemble_physical_bus_y(network, shunts)


def _assemble_physical_bus_y(
    network: Network,
    shunts: Sequence[Shunt],
) -> npt.NDArray[np.complex128]:
    """Stamp branch, three-winding-transformer, and selected shunt terms."""
    bus_idx = {name: index for index, name in enumerate(network.bus_names)}
    y = np.zeros((len(bus_idx), len(bus_idx)), dtype=np.complex128)

    for branch in network.branches:
        i = bus_idx[branch.from_bus_name]
        j = bus_idx[branch.to_bus_name]
        y_ii, y_jj, y_ij, y_ji = branch.get_y_matrix_entries(network.base_mva)
        y[i, i] += y_ii
        y[j, j] += y_jj
        y[i, j] += y_ij
        y[j, i] += y_ji

    for transformer in network.transformers_3w:
        local_y, local_buses = transformer.get_local_admittance_matrix()
        indices = np.asarray([bus_idx[name] for name in local_buses], dtype=int)
        y[np.ix_(indices, indices)] += np.asarray(local_y, dtype=np.complex128)

    for shunt in shunts:
        i = bus_idx[shunt.bus_name]
        y[i, i] += shunt.get_admittance_pu(network.base_mva)

    return y
