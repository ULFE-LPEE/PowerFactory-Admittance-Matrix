"""
Topology simplification for power networks.

This module provides functions to simplify network topology by merging
buses connected by closed switches (zero-impedance connections).
"""

import logging

from ..core.elements import BranchElement, ShuntElement, SwitchBranch, Transformer3WBranch

logger = logging.getLogger(__name__)


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure with preferred node support.
    
    When preferred nodes are specified, they will be chosen as representatives
    over non-preferred nodes during union operations.
    """
    
    def __init__(self, preferred: set[str] | None = None):
        """
        Initialize Union-Find.
        
        Args:
            preferred: Set of node names that should be preferred as representatives.
                      These are typically main busbars (iUsage == 0 in PowerFactory).
        """
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        self.preferred: set[str] = preferred or set()
    
    def find(self, x: str) -> str:
        """Find representative of set containing x (with path compression)."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str) -> None:
        """
        Union the sets containing x and y.
        
        Preferred nodes (main busbars) are always chosen as representatives
        over non-preferred nodes. Among preferred or non-preferred nodes,
        union by rank is used.
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # Prefer main busbars as representatives
        px_preferred = px in self.preferred
        py_preferred = py in self.preferred
        
        if px_preferred and not py_preferred:
            # px is preferred, make it the root
            self.parent[py] = px
        elif py_preferred and not px_preferred:
            # py is preferred, make it the root
            self.parent[px] = py
        else:
            # Both preferred or both not preferred: use rank
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1


def simplify_topology(
    branches: list[BranchElement],
    shunts: list[ShuntElement],
    transformers_3w: list[Transformer3WBranch],
    main_buses: set[str] | None = None
) -> tuple[list[BranchElement], list[ShuntElement], list[Transformer3WBranch], dict[str, str]]:
    """
    Simplify network topology by merging buses connected by closed switches.
    
    This function:
    1. Finds all closed switches (SwitchBranch with is_closed=True)
    2. Uses Union-Find to merge buses connected by closed switches
    3. Rewrites all branch/shunt bus names to use representative names
    4. Removes the closed switches from the branch list
    
    When main_buses is provided, main busbars (iUsage == 0 in PowerFactory)
    are preferred as representatives during merging, preserving their names.
    
    Args:
        branches: List of BranchElement objects
        shunts: List of ShuntElement objects  
        transformers_3w: List of Transformer3WBranch objects
        main_buses: Set of bus names that are main busbars (preferred as representatives)
        
    Returns:
        Tuple of (filtered_branches, shunts, transformers_3w, bus_mapping)
        where bus_mapping is dict[original_name] -> representative_name
    """
    uf = UnionFind(preferred=main_buses)
    
    # Step 1: Union buses connected by closed switches
    closed_switches: list[SwitchBranch] = []
    for branch in branches:
        if isinstance(branch, SwitchBranch) and branch.is_closed:
            uf.union(branch.from_bus_name, branch.to_bus_name)
            closed_switches.append(branch)
    
    # Step 2: Build mapping from original bus name to representative
    all_buses: set[str] = set()
    for b in branches:
        all_buses.add(b.from_bus_name)
        all_buses.add(b.to_bus_name)
    for s in shunts:
        all_buses.add(s.bus_name)
    for t in transformers_3w:
        all_buses.add(t.hv_bus_name)
        all_buses.add(t.mv_bus_name)
        all_buses.add(t.lv_bus_name)
    
    bus_mapping: dict[str, str] = {bus: uf.find(bus) for bus in all_buses}
    
    # Step 3: Rewrite bus names in branches (and remove closed switches)
    filtered_branches: list[BranchElement] = []
    for branch in branches:
        if branch in closed_switches:
            continue  # Drop closed switches
        
        # Update bus names to representative
        branch.from_bus_name = bus_mapping[branch.from_bus_name]
        branch.to_bus_name = bus_mapping[branch.to_bus_name]
        
        # Skip self-loops (both ends now same bus)
        if branch.from_bus_name == branch.to_bus_name:
            continue
            
        filtered_branches.append(branch)
    
    # Step 4: Rewrite bus names in shunts
    for shunt in shunts:
        shunt.bus_name = bus_mapping[shunt.bus_name]
    
    # Step 5: Rewrite bus names in 3-winding transformers
    for trafo in transformers_3w:
        trafo.hv_bus_name = bus_mapping[trafo.hv_bus_name]
        trafo.mv_bus_name = bus_mapping[trafo.mv_bus_name]
        trafo.lv_bus_name = bus_mapping[trafo.lv_bus_name]
    
    # Count stats
    n_original_buses = len(all_buses)
    n_merged_buses = len(set(bus_mapping.values()))
    n_switches_removed = len(closed_switches)
    
    logger.info(f"Topology Simplification: {n_switches_removed} switches removed, "
                f"{n_original_buses} → {n_merged_buses} buses ({n_original_buses - n_merged_buses} eliminated)")
    
    return filtered_branches, shunts, transformers_3w, bus_mapping
