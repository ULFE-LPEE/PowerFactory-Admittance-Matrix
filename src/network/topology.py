"""Contract bus sections connected by closed switches."""

from __future__ import annotations

import logging
from copy import copy
from typing import TYPE_CHECKING

from .elements import SwitchBranch

if TYPE_CHECKING:
    from .network import Network

logger = logging.getLogger(__name__)


class _DisjointSet:
    """Track connected buses while preferring named main busbars."""

    def __init__(self, preferred: set[str]) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        self.preferred = preferred

    def find(self, name: str) -> str:
        if name not in self.parent:
            self.parent[name] = name
            self.rank[name] = 0
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])
        return self.parent[name]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return

        if first_root in self.preferred and second_root not in self.preferred:
            self.parent[second_root] = first_root
        elif second_root in self.preferred and first_root not in self.preferred:
            self.parent[first_root] = second_root
        else:
            if self.rank[first_root] < self.rank[second_root]:
                first_root, second_root = second_root, first_root
            self.parent[second_root] = first_root
            if self.rank[first_root] == self.rank[second_root]:
                self.rank[first_root] += 1


def merge_closed_switch_buses(
    network: Network,
    *,
    main_buses: set[str] | None = None,
) -> Network:
    """Return a copy of *network* with closed-switch bus sections merged.

    Main busbars are preferred as representatives. Closed switches and branches
    whose endpoints collapse to one bus are omitted, as in the existing model.
    Original names remain available through ``Network.resolve_bus_name``.
    """
    from .network import Network

    disjoint_set = _DisjointSet(main_buses or set())
    closed_switches = [branch for branch in network.branches if isinstance(branch, SwitchBranch) and branch.is_closed]
    for switch in closed_switches:
        disjoint_set.union(switch.from_bus_name, switch.to_bus_name)

    mapping = {name: disjoint_set.find(name) for name in network.bus_names}
    aliases = {original: mapping.get(current, current) for original, current in (network.bus_mapping or {}).items()}
    aliases.update(mapping)
    merged = Network(base_mva=network.base_mva, bus_mapping=aliases)

    # Keep the source network's bus order and any buses without equipment.
    for name in network.bus_names:
        merged.register_bus(mapping[name])

    closed_switch_ids = {id(switch) for switch in closed_switches}
    for branch in network.branches:
        if id(branch) in closed_switch_ids:
            continue
        remapped = copy(branch)
        remapped.from_bus_name = mapping[branch.from_bus_name]
        remapped.to_bus_name = mapping[branch.to_bus_name]
        if remapped.from_bus_name != remapped.to_bus_name:
            merged.add_branch(remapped)

    for shunt in network.shunts:
        remapped = copy(shunt)
        remapped.bus_name = mapping[shunt.bus_name]
        merged.add_shunt(remapped)

    for transformer in network.transformers_3w:
        remapped = copy(transformer)
        remapped.hv_bus_name = mapping[transformer.hv_bus_name]
        remapped.mv_bus_name = mapping[transformer.mv_bus_name]
        remapped.lv_bus_name = mapping[transformer.lv_bus_name]
        merged.add_three_winding_transformer(remapped)

    logger.info(
        "Merged %d closed switches: %d buses became %d",
        len(closed_switches),
        len(network.bus_names),
        len(merged.bus_names),
    )
    return merged
