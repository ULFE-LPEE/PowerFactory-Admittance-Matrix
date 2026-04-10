"""
Admittance matrix building and reduction functions.
"""

from .builder import (
    MatrixBuildResult,
    MatrixType,
    build_admittance_matrix,
    build_admittance_matrices,
)

from .reducer import (
    perform_kron_reduction,
    extend_matrix_to_generator_internal_nodes,
)

from .analysis import (
    calculate_power_distribution_ratios,
    calculate_power_distribution_ratios_prefault_postfault,
)

from .topology import (
    simplify_topology,
)

__all__ = [
    # Builder
    'MatrixBuildResult',
    'MatrixType',
    'build_admittance_matrix',
    'build_admittance_matrices',

    # Reducer
    'perform_kron_reduction',
    'extend_matrix_to_generator_internal_nodes',

    # Analysis (Power distribution ratios)
    'calculate_power_distribution_ratios',
    'calculate_power_distribution_ratios_prefault_postfault',

    # Topology
    'simplify_topology',
]
