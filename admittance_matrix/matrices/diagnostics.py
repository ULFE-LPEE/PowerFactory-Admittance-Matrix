"""
Network diagnostics for admittance matrix analysis.

This module provides functions to diagnose issues with admittance matrices,
including detecting islands, isolated buses, and singular matrix problems.
"""

import numpy as np
from collections import deque


def find_connected_components(branches: list, bus_names: list[str]) -> list[set[str]]:
    """
    Find connected components (islands) in the network using BFS.
    
    Args:
        branches: List of branch elements with from_bus_name and to_bus_name
        bus_names: List of all bus names in the network
        
    Returns:
        List of sets, each set containing bus names in that connected component.
        If len(result) > 1, the network has islands.
    """
    # Build adjacency list
    adjacency = {bus: set() for bus in bus_names}
    
    for branch in branches:
        from_bus = branch.from_bus_name
        to_bus = branch.to_bus_name
        
        # Only add connection if branch has finite impedance (i.e., not an open switch)
        z = branch.impedance
        if z.real != float('inf') and z.imag != float('inf'):
            if from_bus in adjacency and to_bus in adjacency:
                adjacency[from_bus].add(to_bus)
                adjacency[to_bus].add(from_bus)
    
    # BFS to find connected components
    visited = set()
    components = []
    
    for start_bus in bus_names:
        if start_bus in visited:
            continue
            
        # BFS from this bus
        component = set()
        queue = deque([start_bus])
        
        while queue:
            bus = queue.popleft()
            if bus in visited:
                continue
            visited.add(bus)
            component.add(bus)
            
            for neighbor in adjacency[bus]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        if component:
            components.append(component)
    
    return components


def find_isolated_buses(Y: np.ndarray, bus_idx: dict[str, int], tol: float = 1e-12) -> list[str]:
    """
    Find buses with no off-diagonal connections (only self-admittance or zero row).
    
    Args:
        Y: Admittance matrix
        bus_idx: Dictionary mapping bus name to matrix index
        tol: Tolerance for considering a value as zero
        
    Returns:
        List of bus names that are isolated (no branch connections)
    """
    idx_to_bus = {v: k for k, v in bus_idx.items()}
    isolated = []
    
    n = Y.shape[0]
    for i in range(n):
        # Sum of absolute off-diagonal elements in row i
        off_diag_sum = np.sum(np.abs(Y[i, :])) - np.abs(Y[i, i])
        if off_diag_sum < tol:
            isolated.append(idx_to_bus[i])
    
    return isolated


def find_zero_rows_cols(Y: np.ndarray, bus_idx: dict[str, int], tol: float = 1e-12) -> dict:
    """
    Find rows and columns that are entirely zero (completely disconnected buses).
    
    Args:
        Y: Admittance matrix
        bus_idx: Dictionary mapping bus name to matrix index
        tol: Tolerance for considering a value as zero
        
    Returns:
        Dictionary with 'zero_rows' and 'zero_cols' lists of bus names
    """
    idx_to_bus = {v: k for k, v in bus_idx.items()}
    
    zero_rows = []
    zero_cols = []
    
    n = Y.shape[0]
    for i in range(n):
        if np.sum(np.abs(Y[i, :])) < tol:
            zero_rows.append(idx_to_bus[i])
        if np.sum(np.abs(Y[:, i])) < tol:
            zero_cols.append(idx_to_bus[i])
    
    return {'zero_rows': zero_rows, 'zero_cols': zero_cols}


def check_matrix_health(Y: np.ndarray, name: str = "Y") -> dict:
    """
    Check the health of an admittance matrix.
    
    Args:
        Y: Admittance matrix
        name: Name for display purposes
        
    Returns:
        Dictionary with diagnostic information
    """
    n = Y.shape[0]
    
    # Compute SVD for rank and condition number
    try:
        U, s, Vh = np.linalg.svd(Y)
        rank = np.sum(s > 1e-10)
        cond_number = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
        min_singular = s[-1]
        near_zero_sv = np.sum(s < 1e-10)
    except Exception as e:
        rank = None
        cond_number = None
        min_singular = None
        near_zero_sv = None
    
    # Check if matrix is singular
    try:
        det = np.linalg.det(Y)
        is_singular = np.abs(det) < 1e-10
    except:
        det = None
        is_singular = True
    
    return {
        'name': name,
        'size': n,
        'rank': rank,
        'rank_deficiency': n - rank if rank else None,
        'condition_number': cond_number,
        'min_singular_value': min_singular,
        'near_zero_singular_values': near_zero_sv,
        'determinant': det,
        'is_singular': is_singular
    }


def find_ungrounded_buses(Y_stab: np.ndarray, Y_lf: np.ndarray, 
                          bus_idx: dict[str, int], tol: float = 1e-12) -> list[str]:
    """
    Find buses with no shunt admittance (no grounding from loads).
    
    A bus is grounded if Y_stab diagonal differs from Y_lf diagonal,
    indicating a load shunt was added.
    
    Args:
        Y_stab: Stability admittance matrix (with loads)
        Y_lf: Load flow admittance matrix (network only)
        bus_idx: Dictionary mapping bus name to matrix index
        tol: Tolerance for considering values equal
        
    Returns:
        List of bus names with no shunt grounding
    """
    idx_to_bus = {v: k for k, v in bus_idx.items()}
    ungrounded = []
    
    for bus, i in bus_idx.items():
        # Compare diagonals - if same, no load was added at this bus
        diff = np.abs(Y_stab[i, i] - Y_lf[i, i])
        if diff < tol:
            ungrounded.append(bus)
    
    return ungrounded


def validate_generator_buses(shunts: list, branches: list, bus_names: list[str]) -> dict:
    """
    Validate that generator buses are properly connected.
    
    Args:
        shunts: List of shunt elements
        branches: List of branch elements
        bus_names: List of all bus names
        
    Returns:
        Dictionary with validation results
    """
    from ..core.elements import GeneratorShunt
    
    # Get generator buses
    gen_buses = set()
    for s in shunts:
        if isinstance(s, GeneratorShunt):
            gen_buses.add(s.bus_name)
    
    # Get all buses connected by branches
    branch_buses = set()
    for b in branches:
        branch_buses.add(b.from_bus_name)
        branch_buses.add(b.to_bus_name)
    
    # Check which generator buses are not in branch connections
    gen_not_in_branches = gen_buses - branch_buses
    
    # Check which generator buses are not in bus_names
    gen_not_in_bus_list = gen_buses - set(bus_names)
    
    # Find connected components
    components = find_connected_components(branches, bus_names)
    
    # Identify which component each generator is in
    gen_components = {}
    for gen_bus in gen_buses:
        for i, comp in enumerate(components):
            if gen_bus in comp:
                gen_components[gen_bus] = i
                break
        else:
            gen_components[gen_bus] = None  # Not found in any component
    
    return {
        'generator_buses': list(gen_buses),
        'generators_not_in_branches': list(gen_not_in_branches),
        'generators_not_in_bus_list': list(gen_not_in_bus_list),
        'generator_components': gen_components,
        'n_components': len(components),
        'generators_in_different_islands': len(set(gen_components.values())) > 1
    }


def diagnose_network(branches: list, shunts: list, bus_names: list[str],
                     bus_idx: dict[str, int], Y_lf: np.ndarray = None, 
                     Y_stab: np.ndarray = None) -> dict:
    """
    Comprehensive network diagnostics.
    
    Args:
        branches: List of branch elements
        shunts: List of shunt elements
        bus_names: List of bus names
        bus_idx: Bus name to index mapping
        Y_lf: Load flow admittance matrix (optional)
        Y_stab: Stability admittance matrix (optional)
        
    Returns:
        Dictionary with all diagnostic results
    """
    results = {
        'n_buses': len(bus_names),
        'n_branches': len(branches),
        'n_shunts': len(shunts),
    }
    
    # Find connected components
    components = find_connected_components(branches, bus_names)
    results['n_islands'] = len(components)
    results['islands'] = [list(c) for c in components]
    results['has_islands'] = len(components) > 1
    
    # Validate generator buses
    gen_validation = validate_generator_buses(shunts, branches, bus_names)
    results['generator_validation'] = gen_validation
    
    # Matrix diagnostics
    if Y_lf is not None:
        results['Y_lf_health'] = check_matrix_health(Y_lf, "Y_lf")
        results['Y_lf_isolated_buses'] = find_isolated_buses(Y_lf, bus_idx)
        results['Y_lf_zero_rows_cols'] = find_zero_rows_cols(Y_lf, bus_idx)
    
    if Y_stab is not None:
        results['Y_stab_health'] = check_matrix_health(Y_stab, "Y_stab")
        results['Y_stab_isolated_buses'] = find_isolated_buses(Y_stab, bus_idx)
        results['Y_stab_zero_rows_cols'] = find_zero_rows_cols(Y_stab, bus_idx)
    
    if Y_lf is not None and Y_stab is not None:
        results['ungrounded_buses'] = find_ungrounded_buses(Y_stab, Y_lf, bus_idx)
    
    return results


def print_diagnostics(diag: dict) -> None:
    """
    Print diagnostic results in a readable format.
    
    Args:
        diag: Dictionary from diagnose_network()
    """
    print("=" * 60)
    print("NETWORK DIAGNOSTICS")
    print("=" * 60)
    
    print(f"\n📊 Network Size:")
    print(f"   Buses: {diag['n_buses']}")
    print(f"   Branches: {diag['n_branches']}")
    print(f"   Shunts: {diag['n_shunts']}")
    
    print(f"\n🏝️  Connectivity (Islands):")
    print(f"   Number of islands: {diag['n_islands']}")
    if diag['has_islands']:
        print("   ⚠️  WARNING: Network has multiple islands!")
        for i, island in enumerate(diag['islands']):
            print(f"   Island {i+1} ({len(island)} buses): {island[:5]}{'...' if len(island) > 5 else ''}")
    else:
        print("   ✅ Network is fully connected")
    
    gen_val = diag['generator_validation']
    print(f"\n⚡ Generator Validation:")
    print(f"   Generator buses: {gen_val['generator_buses']}")
    if gen_val['generators_not_in_branches']:
        print(f"   ⚠️  Generators NOT in any branch: {gen_val['generators_not_in_branches']}")
    if gen_val['generators_not_in_bus_list']:
        print(f"   ⚠️  Generators NOT in bus list: {gen_val['generators_not_in_bus_list']}")
    if gen_val['generators_in_different_islands']:
        print(f"   ⚠️  Generators are in different islands!")
        print(f"   Generator components: {gen_val['generator_components']}")
    
    if 'Y_lf_health' in diag:
        health = diag['Y_lf_health']
        print(f"\n📈 Y_lf Matrix Health:")
        print(f"   Size: {health['size']}x{health['size']}")
        print(f"   Rank: {health['rank']} (deficiency: {health['rank_deficiency']})")
        print(f"   Condition number: {health['condition_number']:.2e}" if health['condition_number'] else "   Condition number: N/A")
        print(f"   Singular: {'⚠️  YES' if health['is_singular'] else '✅ NO'}")
        
        if diag['Y_lf_isolated_buses']:
            print(f"   ⚠️  Isolated buses: {diag['Y_lf_isolated_buses']}")
    
    if 'Y_stab_health' in diag:
        health = diag['Y_stab_health']
        print(f"\n📈 Y_stab Matrix Health:")
        print(f"   Size: {health['size']}x{health['size']}")
        print(f"   Rank: {health['rank']} (deficiency: {health['rank_deficiency']})")
        print(f"   Condition number: {health['condition_number']:.2e}" if health['condition_number'] else "   Condition number: N/A")
        print(f"   Singular: {'⚠️  YES' if health['is_singular'] else '✅ NO'}")
        
        if diag['Y_stab_isolated_buses']:
            print(f"   ⚠️  Isolated buses: {diag['Y_stab_isolated_buses']}")
    
    if 'ungrounded_buses' in diag and diag['ungrounded_buses']:
        print(f"\n🔌 Ungrounded Buses (no load shunt):")
        print(f"   {diag['ungrounded_buses']}")
    
    print("\n" + "=" * 60)
