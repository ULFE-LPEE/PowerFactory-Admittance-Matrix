"""
Network wrapper class for PowerFactory admittance matrix operations.

This module provides a high-level Network class that encapsulates all
the functionality of the admittance_matrix library.
"""

import numpy as np

from ..matrices.builder import build_admittance_matrix, get_unique_buses, MatrixType
from ..matrices.reducer import reduce_to_generator_internal_buses
from ..matrices.analysis import calculate_power_distribution_ratios
from ..matrices.diagnostics import diagnose_network, print_diagnostics
from ..powerflow.extractor import get_network_elements
from ..powerflow.solver import run_load_flow, get_load_flow_results, get_generator_data_from_pf, get_voltage_source_data_from_pf
from ..powerflow.results import GeneratorResult, VoltageSourceResult


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
        self.transformers_3w = []  # 3-winding transformers
        self.bus_names = []
        
        # Matrices
        self.Y_lf = None  # Load flow Y-matrix
        self.Y_stab = None  # Stability Y-matrix (with loads)
        self.Y_reduced = None  # Reduced to generator internal buses
        self.bus_idx = None
        self.gen_names = []  # Names of sources in reduced matrix
        self.source_types = []  # Types: 'generator', 'voltage_source', 'external_grid'
        
        # Results
        self.lf_results = None
        self.gen_data = None  # Generator results
        self.vs_data = None   # Voltage source results
        self.source_data = None  # Combined source data for analysis
        
        # Extract network elements
        self._extract_network()
    
    def _extract_network(self):
        """Extract network elements from PowerFactory."""
        self.branches, self.shunts, self.transformers_3w = get_network_elements(self.app)
        self.bus_names = get_unique_buses(self.branches, self.shunts, self.transformers_3w)
    
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
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w
        )
        
        # Build stability matrix (with loads)
        matrix_type = MatrixType.STABILITY_FULL if include_generators else MatrixType.STABILITY
        self.Y_stab, _ = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=matrix_type,
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w
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
            self.vs_data = get_voltage_source_data_from_pf(
                self.app, self.shunts, self.lf_results, self.base_mva
            )
            # Update gen_names to match gen_data order (used for plotting)
            self._gen_data_names = [g.name for g in self.gen_data]
        return success
    
    def reduce_to_generators(
        self,
        include_voltage_sources: bool = True,
        include_external_grids: bool = False
    ) -> None:
        """
        Apply Kron reduction to obtain generator internal bus matrix.
        
        Optionally includes voltage sources and external grids as additional 
        sources that participate in power redistribution.
        
        The Y_stab matrix must be built first using build_matrices().
        
        Args:
            include_voltage_sources: If True, include AC voltage sources in reduction
            include_external_grids: If True, include external grids in reduction
        """
        if self.Y_stab is None:
            raise RuntimeError("Must call build_matrices() first")
        
        self.Y_reduced, self.gen_names, self.source_types = reduce_to_generator_internal_buses(
            self.Y_stab, self.bus_idx, self.shunts, self.base_mva,
            include_voltage_sources=include_voltage_sources,
            include_external_grids=include_external_grids
        )
    
    def calculate_power_ratios(self, disturbance_source_name: str) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Calculate power distribution ratios for a source (generator/voltage source) trip.
        
        Args:
            disturbance_source_name: Name of the source that trips
            
        Returns:
            Tuple of (ratios array, source names in order, source types in order)
        """
        if self.Y_reduced is None:
            raise RuntimeError("Must call reduce_to_generators() first")
        if self.gen_data is None:
            raise RuntimeError("Must call run_load_flow() first")
        
        # Build combined source data matching the order in gen_names
        # gen_names contains all sources in the order they were added to Y_reduced
        self._build_source_data()

        ratios, source_names_order, source_types_order = calculate_power_distribution_ratios(
            self.Y_reduced, self.source_data, disturbance_source_name
        )
        return ratios, source_names_order, source_types_order
    
    def calculate_all_power_ratios(
        self,
        outage_generators: list[str] = None,
        normalize: bool = True,
        verbose: bool = False
    ) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """
        Calculate power distribution ratios matrix for multiple generator outages.
        
        Each row corresponds to one generator outage, each column corresponds to
        a source (generator or voltage source) receiving power.
        
        Args:
            outage_generators: List of generator names to trip. If None, all 
                              synchronous generators will be used.
            normalize: If True, normalize each row to sum to 100%
            verbose: If True, print progress every 10 generators
            
        Returns:
            Tuple of:
                - ratios_matrix: 2D numpy array (n_outages × n_sources)
                - outage_names: List of generator names that were tripped (row labels)
                - source_names: List of all source names (column labels)
                - source_types: List of source types (column types)
        """
        if self.Y_reduced is None:
            raise RuntimeError("Must call reduce_to_generators() first")
        if self.gen_data is None:
            raise RuntimeError("Must call run_load_flow() first")
        
        # Default to all synchronous generators if not specified
        if outage_generators is None:
            outage_generators = [
                name for name, stype in zip(self.gen_names, self.source_types) 
                if stype == 'generator'
            ]
        
        # Build source data once
        self._build_source_data()
        
        all_ratios = []
        valid_outages = []
        
        for _, gen_name in enumerate(outage_generators):
            try:
                ratios_i, source_names, source_types = calculate_power_distribution_ratios(
                    self.Y_reduced, self.source_data, gen_name
                )
                
                if normalize:
                    ratio_sum = np.sum(ratios_i)
                    if ratio_sum > 0:
                        ratios_i = (ratios_i / ratio_sum) * 100
                
                all_ratios.append(ratios_i)
                valid_outages.append(gen_name)
                
            except Exception as e:
                if verbose:
                    print(f"Skipping {gen_name}: {e}")
        
        ratios_matrix = np.array(all_ratios)
        
        if verbose:
            print(f"\nRatios matrix shape: {ratios_matrix.shape}")
            print(f"  Rows (outages): {len(valid_outages)}")
            print(f"  Columns (sources): {len(source_names)}")
        
        return ratios_matrix, valid_outages, source_names, source_types
    
    def _build_source_data(self) -> None:
        """
        Build combined source data list matching the order in gen_names.
        
        This combines generator and voltage source data in the same order
        as they appear in the reduced Y-matrix.
        """
        # Create lookup dictionaries
        gen_lookup = {g.name: g for g in self.gen_data} if self.gen_data else {}
        vs_lookup = {v.name: v for v in self.vs_data} if self.vs_data else {}
        
        self.source_data = []
        for name, stype in zip(self.gen_names, self.source_types):
            if stype == 'generator' and name in gen_lookup:
                self.source_data.append(gen_lookup[name])
            elif stype == 'voltage_source' and name in vs_lookup:
                self.source_data.append(vs_lookup[name])
            else:
                print(f"Warning: Source '{name}' (type: {stype}) not found in load flow data")
    
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
    
    def get_zone(self, source_name: str) -> str:
        """
        Get the zone for a source (generator or voltage source).
        
        Args:
            source_name: Name of the source
            
        Returns:
            Zone name string, or 'Unknown' if not found
        """
        if self.gen_data is None:
            raise RuntimeError("Must call run_load_flow() first")
        
        # Check generators first
        for g in self.gen_data:
            if g.name == source_name:
                return g.zone
        
        # Voltage sources don't have zones, return 'Unknown'
        if self.vs_data:
            for v in self.vs_data:
                if v.name == source_name:
                    return 'Unknown'
        
        return 'Unknown'
    
    @property
    def gen_zones(self) -> list[str]:
        """
        Get zones for all sources in gen_names order.
        
        Returns:
            List of zone names matching gen_names order
        """
        if not self.gen_names:
            return []
        return [self.get_zone(name) for name in self.gen_names]
    
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
        """Number of 2-winding transformers in the network."""
        return len([b for b in self.branches if type(b).__name__ == 'TransformerBranch'])
    
    @property
    def n_transformers_3w(self) -> int:
        """Number of 3-winding transformers in the network."""
        return len(self.transformers_3w)
    
    @property
    def n_switches(self) -> int:
        """Number of switches/couplers in the network."""
        return len([b for b in self.branches if type(b).__name__ == 'SwitchBranch'])
    
    @property
    def n_external_grids(self) -> int:
        """Number of external grids in the network."""
        return len([s for s in self.shunts if type(s).__name__ == 'ExternalGridShunt'])
    
    @property
    def n_voltage_sources(self) -> int:
        """Number of AC voltage sources in the network."""
        return len([s for s in self.shunts if type(s).__name__ == 'VoltageSourceShunt'])
    
    def diagnose(self, print_results: bool = True) -> dict:
        """
        Run comprehensive network diagnostics to identify potential issues.
        
        This is useful for debugging singular matrix errors during Kron reduction.
        
        Args:
            print_results: If True, print formatted diagnostic report
            
        Returns:
            Dictionary with all diagnostic results
        """
        diag = diagnose_network(
            branches=self.branches,
            shunts=self.shunts,
            bus_names=self.bus_names,
            bus_idx=self.bus_idx,
            Y_lf=self.Y_lf,
            Y_stab=self.Y_stab
        )
        
        if print_results:
            print_diagnostics(diag)
        
        return diag
