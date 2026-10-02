"""Source-node extension, linear solves, and Kron reduction."""

import numpy as np
import numpy.typing as npt
from ..network.elements import SourceShunt

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover - optional speed-up dependency
    sp = None
    spla = None


def solve_admittance_system(
    y: npt.NDArray[np.complex128],
    rhs: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Solve Y x = rhs, using SciPy for sufficiently sparse matrices."""
    if sp is not None and spla is not None and y.size:
        density = np.count_nonzero(y) / y.size
        if density <= 0.15:
            solution = np.asarray(spla.spsolve(sp.csc_matrix(y), rhs), dtype=np.complex128)
            if rhs.ndim == 2 and solution.ndim == 1:
                solution = solution.reshape(-1, 1)
            if not np.isfinite(solution).all():
                raise np.linalg.LinAlgError("Admittance matrix is singular.")
            return solution

    return np.linalg.solve(y, rhs)


def perform_kron_reduction(
    Y: npt.NDArray[np.complex128],
    indices_to_keep: list[int],
) -> npt.NDArray[np.complex128]:
    """Eliminate all nodes except ``indices_to_keep`` using a Schur complement."""
    n = Y.shape[0]
    indices_to_eliminate = sorted(set(range(n)) - set(indices_to_keep))

    if not indices_to_eliminate:
        return Y[np.ix_(indices_to_keep, indices_to_keep)]

    Y_AA = Y[np.ix_(indices_to_keep, indices_to_keep)]
    Y_AB = Y[np.ix_(indices_to_keep, indices_to_eliminate)]
    Y_BA = Y[np.ix_(indices_to_eliminate, indices_to_keep)]
    Y_BB = Y[np.ix_(indices_to_eliminate, indices_to_eliminate)]

    solution = solve_admittance_system(Y_BB, Y_BA)
    return Y_AA - Y_AB @ solution


def extend_matrix_to_generator_internal_nodes(
    Y_bus: npt.NDArray[np.complex128],
    bus_idx: dict[str, int],
    sources: list[SourceShunt],
    base_mva: float = 100.0,
) -> npt.NDArray[np.complex128]:
    """Prepend source-internal nodes to the physical-bus stability matrix.

    ``Y_bus`` already includes each source admittance on its terminal-bus
    diagonal. The augmented blocks are ``[[K, L], [L.T, Y_bus]]``, where
    ``K = diag(y_source)`` and ``L[source, bus] = -y_source``. This retains
    the established source-first ordering and modelling convention.
    """
    n_sources = len(sources)
    n_bus = len(bus_idx)
    source_admittances = np.asarray([source.get_admittance_pu(base_mva) for source in sources], dtype=np.complex128)
    source_to_bus = np.zeros((n_sources, n_bus), dtype=np.complex128)
    for index, source in enumerate(sources):
        source_to_bus[index, bus_idx[source.bus_name]] = -source_admittances[index]

    return np.block(
        [
            [np.diag(source_admittances), source_to_bus],
            [source_to_bus.T, Y_bus],
        ]
    )
