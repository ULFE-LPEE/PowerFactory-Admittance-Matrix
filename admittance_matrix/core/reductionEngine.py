'''
This should be used only for advanced modelling when analyzing outage of generator via synchronizing power coefficient method or similiar methods.
Used in Network.calculate_power_ratios() method.
'''
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from admittance_matrix.core.elements import (
    BranchElement,
    GeneratorShunt,
    ShuntElement,
    Transformer3WBranch,
    VoltageSourceShunt,
    ExternalGridShunt,
)
from admittance_matrix.matrices.builder import build_admittance_matrix, MatrixType
from admittance_matrix.matrices.reducer import extend_matrix_to_generator_internal_nodes, perform_kron_reduction


@dataclass(slots=True)
class Mode1OutageReduction:
    """Reusable Mode 1 outage-reduction data built from pre-fault matrices."""

    Y_stab: npt.NDArray[np.complex128]
    bus_idx: dict[str, int]
    sources: list[GeneratorShunt | VoltageSourceShunt | ExternalGridShunt]
    base_mva: float
    source_names: list[str]
    source_bus_indices: npt.NDArray[np.int_]
    source_admittances: npt.NDArray[np.complex128]
    bus_position_by_index: dict[int, int]
    y_reduced: npt.NDArray[np.complex128]
    l_inv_bus_columns: npt.NDArray[np.complex128]
    l_inv_t_bus_columns: npt.NDArray[np.complex128]

    @classmethod
    def from_prefault_matrices(
        cls,
        Y_stab: npt.NDArray[np.complex128],
        Y_reduced: npt.NDArray[np.complex128],
        bus_idx: dict[str, int],
        sources: list[GeneratorShunt | VoltageSourceShunt | ExternalGridShunt],
        base_mva: float,
    ) -> "Mode1OutageReduction":
        """
        Create reusable Mode 1 outage-reduction data.

        A Mode 1 outage removes one source admittance from one bus diagonal.
        That is a rank-one change of the bus block in the Kron reduction, so
        every outage can reuse the base matrix solves computed here.
        """
        source_names = [source.name for source in sources]
        source_bus_indices = np.array([bus_idx[source.bus_name] for source in sources], dtype=int)
        source_admittances = np.array([source.get_admittance_pu(base_mva) for source in sources], dtype=np.complex128)
        unique_bus_indices = np.array(sorted(set(int(idx) for idx in source_bus_indices)), dtype=int)
        bus_position_by_index = {int(bus_i): pos for pos, bus_i in enumerate(unique_bus_indices)}

        bus_selector = np.zeros((Y_stab.shape[0], len(unique_bus_indices)), dtype=np.complex128)
        for pos, bus_i in enumerate(unique_bus_indices):
            bus_selector[int(bus_i), pos] = 1.0

        try:
            inv_bus_columns = np.linalg.solve(Y_stab, bus_selector)
            inv_t_bus_columns = np.linalg.solve(Y_stab.T, bus_selector)
        except np.linalg.LinAlgError as exc:
            raise Mode1OutageUpdateError("base stability matrix solve failed") from exc

        L = np.zeros((len(sources), Y_stab.shape[0]), dtype=np.complex128)
        for source_i, bus_i in enumerate(source_bus_indices):
            L[source_i, int(bus_i)] = -source_admittances[source_i]

        return cls(
            Y_stab=Y_stab,
            bus_idx=bus_idx,
            sources=sources,
            base_mva=base_mva,
            source_names=source_names,
            source_bus_indices=source_bus_indices,
            source_admittances=source_admittances,
            bus_position_by_index=bus_position_by_index,
            y_reduced=Y_reduced,
            l_inv_bus_columns=L @ inv_bus_columns,
            l_inv_t_bus_columns=L @ inv_t_bus_columns,
        )

    def reduced_column_after_outage(
        self,
        disturbance_source_name: str,
        denominator_tolerance: float = 1e-10,
    ) -> npt.NDArray[np.complex128]:
        """
        Return one Mode 1 outage column without rebuilding or re-reducing.

        This is algebraically equivalent to removing the source admittance from
        Y_stab and doing a full Kron reduction, but only updates the one reduced
        matrix column needed by the power-ratio calculation.
        """
        if disturbance_source_name not in self.source_names:
            raise ValueError(f"Source '{disturbance_source_name}' not found. Available: {self.source_names}")

        dist_idx = self.source_names.index(disturbance_source_name)
        bus_i = int(self.source_bus_indices[dist_idx])
        bus_pos = self.bus_position_by_index[bus_i]
        source_admittance = self.source_admittances[dist_idx]

        if source_admittance == 0:
            raise Mode1OutageUpdateError(f"source '{disturbance_source_name}' has zero internal admittance")

        base_inverse_at_bus = self.l_inv_bus_columns[dist_idx, bus_pos] / (-source_admittance)
        denominator = 1.0 - source_admittance * base_inverse_at_bus
        if abs(denominator) < denominator_tolerance:
            raise Mode1OutageUpdateError(
                f"rank-one update denominator is near zero for '{disturbance_source_name}'"
            )

        factor = source_admittance / denominator
        return (
            self.y_reduced[:, dist_idx]
            - factor * self.l_inv_bus_columns[:, bus_pos] * self.l_inv_t_bus_columns[dist_idx, bus_pos]
        )

    def reduced_matrix_after_outage_direct(
        self,
        disturbance_source_name: str,
    ) -> npt.NDArray[np.complex128]:
        """
        Return the full reduced matrix after one source outage.

        This method applies the outage directly to a copy of Y_stab and then
        performs normal matrix extension and Kron reduction.
        """
        return _reduce_after_source_outage(
            Y_stab=self.Y_stab,
            bus_idx=self.bus_idx,
            sources=self.sources,
            base_mva=self.base_mva,
            excluded_source_name=disturbance_source_name,
        )


class Mode1OutageUpdateError(Exception):
    """Raised when the rank-one outage update should use direct reduction."""


def _reduce_after_source_outage(
    Y_stab: npt.NDArray[np.complex128],
    bus_idx: dict[str, int],
    sources: list[GeneratorShunt | VoltageSourceShunt | ExternalGridShunt],
    base_mva: float,
    excluded_source_name: str,
) -> npt.NDArray[np.complex128]:
    source_to_exclude = next(
        (source for source in sources if source.name == excluded_source_name),
        None,
    )
    if source_to_exclude is None:
        source_names = [source.name for source in sources]
        raise ValueError(f"Source '{excluded_source_name}' not found. Available: {source_names}")

    Y_matrix = Y_stab.copy()
    bus_i = bus_idx[source_to_exclude.bus_name]
    Y_matrix[bus_i, bus_i] -= source_to_exclude.get_admittance_pu(base_mva)

    Y_extended = extend_matrix_to_generator_internal_nodes(
        Y_bus=Y_matrix,
        bus_idx=bus_idx,
        sources=sources,
        base_mva=base_mva,
    )

    indices_to_keep = list(range(len(sources)))
    return perform_kron_reduction(Y_extended, indices_to_keep)

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
