"""
Network wrapper class for PowerFactory admittance matrix operations.

This module provides a high-level Network class that encapsulates all
the functionality of the admittance_matrix library.
"""

import numpy as np

from ..matrices.builder import build_admittance_matrix, get_unique_buses, MatrixType
from ..matrices.reducer import reduce_to_generator_internal_buses
from ..matrices.analysis import calculate_power_distribution_ratios
from ..powerflow.extractor import get_network_elements
from ..powerflow.solver import run_load_flow, get_load_flow_results, get_generator_data_from_pf
from ..powerflow.results import GeneratorResult


class Network:
    """
    High-level wrapper for PowerFactory network analysis.
    
    This class provides a convenient interface for:
    - Extracting network elements from PowerFactory
    - Building admittance matrices
    - Running load flow calculations
    - Reducing to generator internal buses
    - Calculating power distribution ratios
    """
    
    def __init__(self, app, base_mva: float = 100.0):
        """
        Initialize the Network from a PowerFactory application.
        
        Args:
            app: PowerFactory application instance (already connected and with active project)
            base_mva: System base power in MVA (default 100)
        """
        self.app = app
        self.base_mva = base_mva
        
        # Network data
        self.branches = []
        self.shunts = []
        self.bus_names = []
        
        # Matrices
        self.Y_lf = None  # Load flow Y-matrix
        self.Y_stab = None  # Stability Y-matrix (with loads)
        self.Y_reduced = None  # Reduced to generator internal buses
        self.bus_idx = None
        self.gen_names = []
        
        # Results
        self.lf_results = None
        self.gen_data = None
        
        # Extract network elements
        self._extract_network()
    
    def _extract_network(self):
        """Extract network elements from PowerFactory."""
        self.branches, self.shunts = get_network_elements(self.app)
        self.bus_names = get_unique_buses(self.branches, self.shunts)
    
    def build_matrices(self, include_generators: bool = False) -> None:
        """
        Build admittance matrices from the network elements.
        
        Args:
            include_generators: If True, include generator admittances in diagonal
        """
        # Build load flow matrix (network only)
        self.Y_lf, self.bus_idx = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=MatrixType.LOAD_FLOW,
            base_mva=self.base_mva
        )
        
        # Build stability matrix (with loads)
        matrix_type = MatrixType.STABILITY_FULL if include_generators else MatrixType.STABILITY
        self.Y_stab, _ = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=matrix_type,
            base_mva=self.base_mva
        )
    
    def run_load_flow(self) -> bool:
        """
        Execute load flow calculation in PowerFactory.
        
        Returns:
            True if load flow converged, False otherwise
        """
        success = run_load_flow(self.app)
        if success:
            self.lf_results = get_load_flow_results(self.app)
            self.gen_data = get_generator_data_from_pf(
                self.app, self.shunts, self.lf_results, self.base_mva
            )
            # Update gen_names to match gen_data order (used for plotting)
            self._gen_data_names = [g.name for g in self.gen_data]
        return success
    
    def reduce_to_generators(self) -> None:
        """
        Apply Kron reduction to obtain generator internal bus matrix.
        
        The Y_stab matrix must be built first using build_matrices().
        """
        if self.Y_stab is None:
            raise RuntimeError("Must call build_matrices() first")
        
        self.Y_reduced, self.gen_names = reduce_to_generator_internal_buses(
            self.Y_stab, self.bus_idx, self.shunts, self.base_mva
        )
    
    def calculate_power_ratios(self, disturbance_gen_name: str) -> tuple[np.ndarray, list[str]]:
        """
        Calculate power distribution ratios for a generator trip.
        
        Args:
            disturbance_gen_name: Name of the generator that trips
            
        Returns:
            Tuple of (ratios array, generator names in matching order)
        """
        if self.Y_reduced is None:
            raise RuntimeError("Must call reduce_to_generators() first")
        if self.gen_data is None:
            raise RuntimeError("Must call run_load_flow() first")
        
        ratios, gen_names_order = calculate_power_distribution_ratios(
            self.Y_reduced, self.gen_data, disturbance_gen_name
        )
        return ratios, gen_names_order
    
    def get_generator(self, name: str) -> GeneratorResult:
        """
        Get generator data by name.
        
        Args:
            name: Generator name
            
        Returns:
            GeneratorResult object
        """
        if self.gen_data is None:
            raise RuntimeError("Must call run_load_flow() first")
        
        for g in self.gen_data:
            if g.name == name:
                return g
        
        raise KeyError(f"Generator '{name}' not found")
    
    @property
    def n_buses(self) -> int:
        """Number of buses in the network."""
        return len(self.bus_names)
    
    @property
    def n_generators(self) -> int:
        """Number of generators in the network."""
        return len([s for s in self.shunts if type(s).__name__ == 'GeneratorShunt'])
    
    @property
    def n_loads(self) -> int:
        """Number of loads in the network."""
        return len([s for s in self.shunts if type(s).__name__ == 'LoadShunt'])
    
    @property
    def n_lines(self) -> int:
        """Number of lines in the network."""
        return len([b for b in self.branches if type(b).__name__ == 'LineBranch'])
    
    @property
    def n_transformers(self) -> int:
        """Number of transformers in the network."""
        return len([b for b in self.branches if type(b).__name__ == 'TransformerBranch'])
    
    @property
    def n_switches(self) -> int:
        """Number of switches/couplers in the network."""
        return len([b for b in self.branches if type(b).__name__ == 'SwitchBranch'])
