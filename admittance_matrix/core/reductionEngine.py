'''
This should be used only for advanced modelling when analyzing outage of generator via synchronizing power coefficient method or similiar methods.
Used in Network.calculate_power_ratios() method.
'''
from admittance_matrix.core.network import BranchElement, ShuntElement, Transformer3WBranch
from admittance_matrix.core.elements import GeneratorShunt, VoltageSourceShunt, ExternalGridShunt
import numpy as np
import numpy.typing as npt
from admittance_matrix.matrices.builder import build_admittance_matrix, MatrixType
from admittance_matrix.matrices.reducer import extend_matrix_to_generator_internal_nodes, perform_kron_reduction

def perform_reduction_mode1(
        bus_names: list[str],
        branches: list[BranchElement],
        branches_3w_traformers: list[Transformer3WBranch],
        shunts: list[ShuntElement],
        sources: list[GeneratorShunt | VoltageSourceShunt | ExternalGridShunt],
        BASE_MVA: float,
        excluded_source_name: str | None = None,
    ) -> npt.NDArray[np.complex128]:

    # Build admittance matrix
    Y_matrix, bus_idx = build_admittance_matrix(
        bus_names=bus_names, branches=branches, shunts=shunts,
        matrix_type=MatrixType.STABILITY,
        base_mva=BASE_MVA,
        transformers_3w=branches_3w_traformers,
        exclude_source_name=excluded_source_name  # Exclude generator admittance
    )

    # Get extended matrix with internal generator nodes (EXTENDED MATRIX MODIFIED)
    Y_extended = extend_matrix_to_generator_internal_nodes(
        Y_bus=Y_matrix,
        bus_idx=bus_idx,
        sources=sources,
        base_mva=BASE_MVA,
    )

    # Reduce to only internal generator buses
    n_sources = len(sources)
    indices_to_keep = list(range(n_sources))
    Y_reduced = perform_kron_reduction(Y_extended, indices_to_keep)

    return Y_reduced

def perform_reduction_mode2(
        bus_names: list[str],
        branches: list[BranchElement],
        branches_3w_traformers: list[Transformer3WBranch],
        shunts: list[ShuntElement],
        filtered_sources: list[GeneratorShunt | VoltageSourceShunt | ExternalGridShunt],
        BASE_MVA: float,
        excluded_source_name: str | None = None,
    ) -> np.ndarray:

    # Build admittance matrix
    Y_matrix, bus_idx = build_admittance_matrix(
        bus_names=bus_names, branches=branches, shunts=shunts,
        matrix_type=MatrixType.STABILITY,
        base_mva=BASE_MVA,
        transformers_3w=branches_3w_traformers,
        exclude_source_name=excluded_source_name  # Exclude generator admittance
    )

    # Get extended matrix with internal generator nodes (EXTENDED MATRIX)
    Y_extended = extend_matrix_to_generator_internal_nodes(
        Y_bus=Y_matrix,
        bus_idx=bus_idx,
        sources=filtered_sources,
        base_mva=BASE_MVA,
    )

    # Reduce to only internal generator buses
    n_sources = len(filtered_sources)
    indices_to_keep = list(range(n_sources))
    Y_reduced = perform_kron_reduction(Y_extended, indices_to_keep)

    return Y_reduced