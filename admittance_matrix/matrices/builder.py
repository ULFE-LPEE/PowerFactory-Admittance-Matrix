"""
Admittance matrix construction.

This module provides functions for building Y-matrices from network elements.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..core.elements import BranchElement, ExternalGridShunt, GeneratorShunt, LoadModelType, LoadShunt, ShuntElement, ShuntFilterShunt, Transformer3WBranch, VoltageSourceShunt

logger = logging.getLogger(__name__)

AdmittanceMatrix = npt.NDArray[np.complex128]

@dataclass(slots=True)
class MatrixBuildResult:
    """Pair of admittance matrices that share the same bus index mapping.
        - y_lf: Load flow matrix (network only)
        - y_stab: Stability matrix (includes loads and generators)
        - bus_idx: Mapping of bus names to matrix indices
    """
    y_lf: AdmittanceMatrix
    y_stab: AdmittanceMatrix
    bus_idx: dict[str, int]

class MatrixType(Enum):
    """Type of admittance matrix to build."""
    LOAD_FLOW = "load_flow"           # Network only (no shunt admittances)
    STABILITY = "stability"            # Network + loads + generators

def _build_common_admittance_matrix(
    bus_names: Sequence[str],
    branches: Sequence[BranchElement],
    shunts: Sequence[ShuntElement],
    transformers_3w: Sequence[Transformer3WBranch] | None,
    base_mva: float,
) -> tuple[AdmittanceMatrix, dict[str, int]]:
    """Build the shared matrix terms used by both load-flow and stability matrices."""

    n = len(bus_names)
    bus_idx: dict[str, int] = {name: i for i, name in enumerate(bus_names)}
    matrix: AdmittanceMatrix = np.zeros((n, n), dtype=np.complex128)

    # Add all branch elements
    for branch in branches:
        i = bus_idx[branch.from_bus_name]
        j = bus_idx[branch.to_bus_name]
        Yii, Yjj, Yij, Yji = branch.get_y_matrix_entries(base_mva)

        matrix[i, i] += Yii
        matrix[j, j] += Yjj
        matrix[i, j] += Yij
        matrix[j, i] += Yji

    # Add 3-winding transformer contributions
    if transformers_3w:
        for transformer in transformers_3w:
            local_matrix, local_bus_names = transformer.get_local_admittance_matrix()
            indices = np.asarray([bus_idx[name] for name in local_bus_names], dtype=int)
            matrix[np.ix_(indices, indices)] += np.asarray(local_matrix, dtype=np.complex128)

    # Add passive shunt filters and constant impedance loads
    for shunt in shunts:
        if isinstance(shunt, ShuntFilterShunt):
            i = bus_idx[shunt.bus_name]
            matrix[i, i] += shunt.get_admittance_pu(base_mva)
        if isinstance(shunt, LoadShunt):
            if shunt.load_model == LoadModelType.CONSTANT_IMPEDANCE:
                i = bus_idx[shunt.bus_name]
                matrix[i, i] += shunt.get_admittance_pu(base_mva)

    return matrix, bus_idx

def _add_stability_shunts(
    matrix: AdmittanceMatrix,
    bus_idx: dict[str, int],
    shunts: Sequence[ShuntElement],
    base_mva: float,
    exclude_source_name: str | None,
) -> None:
    """Add stability-only shunt contributions in place. Currently adds only sources
    
    Args:
        matrix: Admittance matrix to modify
        bus_idx: Mapping of bus names to matrix indices
        shunts: List of shunt elements
        base_mva: System base power in MVA
        exclude_source_name: Name of source to exclude from admittance matrix (optional for some analyses)
    """
    for shunt in shunts:
        if exclude_source_name is not None and shunt.name == exclude_source_name:
            logger.debug("Excluding source admittance for %s", exclude_source_name)
            print(f"Excluding source admittance for {exclude_source_name}")
            continue

        if isinstance(shunt, (GeneratorShunt, VoltageSourceShunt, ExternalGridShunt)):
            i = bus_idx[shunt.bus_name]
            matrix[i, i] += shunt.get_admittance_pu(base_mva)

def build_admittance_matrices(
    *,
    bus_names: Sequence[str],
    branches: Sequence[BranchElement],
    shunts: Sequence[ShuntElement],
    transformers_3w: Sequence[Transformer3WBranch] | None = None,
    base_mva: float = 100.0,
    exclude_source_name: str | None = None,
) -> MatrixBuildResult:
    """Build the load-flow and stability matrices in one shared pass.
    
    Args:

        bus_names: List of unique bus names (including virtual star nodes for 3W trafos)
        branches: List of branch elements (lines, switches, 2W transformers)
        shunts: List of shunt elements (generators, loads)
        transformers_3w: List of 3-winding transformers (optional)
        base_mva: System base power in MVA (default 100 MVA)
        exclude_source_name: Name of source to exclude from admittance matrix (optional for some analyses)
    Returns:        
        MatrixBuildResult containing both matrices and bus index mapping
    """
    common_matrix, bus_idx = _build_common_admittance_matrix(
        bus_names,
        branches,
        shunts,
        transformers_3w,
        base_mva,
    )
    y_lf = common_matrix.copy()
    y_stab = common_matrix.copy()
    _add_stability_shunts(y_stab, bus_idx, shunts, base_mva, exclude_source_name)
    return MatrixBuildResult(y_lf=y_lf, y_stab=y_stab, bus_idx=bus_idx)

def build_admittance_matrix(
    *,
    bus_names: Sequence[str],
    branches: Sequence[BranchElement],
    shunts: Sequence[ShuntElement],
    transformers_3w: Sequence[Transformer3WBranch] | None = None,
    matrix_type: MatrixType = MatrixType.LOAD_FLOW,
    base_mva: float = 100.0,
    exclude_source_name: str | None = None,
) -> tuple[npt.NDArray[np.complex128], dict[str, int]]:
    """
    Build the admittance (Y) matrix from branch and shunt elements.
    
    Args:
        bus_names: List of unique bus names (including virtual star nodes for 3W trafos)
        branches: List of branch elements (lines, switches, 2W transformers)
        shunts: List of shunt elements (generators, loads)
        transformers_3w: List of 3-winding transformers (optional)
        matrix_type: Type of matrix to build
        base_mva: System base power in MVA (default 100 MVA)
        exclude_source_name: Name of source to exclude from admittance matrix (optional)
    Returns:
        Tuple of (Y_matrix, bus_index_map)
    """
    common_matrix, bus_idx = _build_common_admittance_matrix(
        bus_names,
        branches,
        shunts,
        transformers_3w,
        base_mva,
    )

    if matrix_type == MatrixType.LOAD_FLOW:
        return common_matrix, bus_idx

    matrix = common_matrix.copy()
    _add_stability_shunts(matrix, bus_idx, shunts, base_mva, exclude_source_name)
    return matrix, bus_idx