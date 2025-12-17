"""
Network wrapper class for PowerFactory admittance matrix operations.

This module provides a high-level Network class that encapsulates all
the functionality of the admittance_matrix library.
"""

import logging

import numpy as np
import powerfactory as pf

from ..matrices.builder import build_admittance_matrix, get_unique_buses, MatrixType
from ..matrices.reducer import reduce_to_generator_internal_buses
from ..matrices.analysis import calculate_power_distribution_ratios
from ..matrices.diagnostics import diagnose_network, print_diagnostics
from ..matrices.topology import simplify_topology
from ..adapters.powerfactory import get_network_elements
from ..adapters.powerfactory import run_load_flow, get_load_flow_results, get_generator_data_from_pf, get_voltage_source_data_from_pf, get_external_grid_data_from_pf
from ..adapters.powerfactory import GeneratorResult, VoltageSourceResult, ExternalGridResult

logger = logging.getLogger(__name__)
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
    
    def __init__(self, app, base_mva: float = 100.0, simplify_topology: bool = False, verbose: bool = True):
        """
        Initialize the Network from a PowerFactory application.
        
        Args:
            app: PowerFactory application instance (already connected and with active project)
            base_mva: System base power in MVA (default 100)
            simplify_topology: If True, merge buses connected by closed switches (reduces bus count)
            verbose: If True, print extraction summary to console (default True)
        """
        self.app: pf.Application = app
        self._hide()

        # Network data
        self.base_mva = base_mva
        self.simplify_topology = simplify_topology
        self.verbose = verbose
        self.bus_mapping = None  # Mapping from original to merged bus names
        self.branches = []
        self.shunts = []
        self.transformers_3w = []  # 3-winding transformers
        self.bus_names = []
        
        # Matrices (use properties Y_lf_matrix, Y_stab_matrix, Y_reduced_matrix for access)
        self._Y_lf: np.ndarray | None = None
        self._Y_stab: np.ndarray | None = None
        self._Y_reduced: np.ndarray | None = None
        self.bus_idx = None
        self.gen_names = []  # Names of sources in reduced matrix
        self.source_types = []  # Types: 'generator', 'voltage_source', 'external_grid'
        
        # Results
        self.lf_results = None
        self.gen_data = None  # Generator results
        self.vs_data = None   # Voltage source results
        self.xnet_data = None # External grid results
        self.source_data = None  # Combined source data for analysis
        
        # Extract network elements
        self._extract_network()
        self._build_matrices()
        self._show()

    def _extract_network(self):
        """Extract network elements from PowerFactory."""
        self.branches, self.shunts, self.transformers_3w = get_network_elements(self.app)
        
        # Print extraction summary before simplification
        if self.verbose:
            self._print_network_summary("Network extracted:")
        
        # Optionally merge buses connected by closed switches
        if self.simplify_topology:
            n_buses_before = len(get_unique_buses(self.branches, self.shunts, self.transformers_3w))
            self.branches, self.shunts, self.transformers_3w, self.bus_mapping = simplify_topology(
                self.branches, self.shunts, self.transformers_3w
            )
            n_buses_after = len(get_unique_buses(self.branches, self.shunts, self.transformers_3w))
            if self.verbose:
                print(f"Topology simplified: {n_buses_before} to {n_buses_after} buses ({n_buses_before - n_buses_after} eliminated)")
                self._print_network_summary("Network after simplification:")
        
        self.bus_names = get_unique_buses(self.branches, self.shunts, self.transformers_3w)
    
    def _update_load_admittances_with_lf_voltage(self) -> None:
        """
        Update load admittances using actual load flow bus voltages.
        
        For constant impedance load modeling in stability analysis,
        the admittance should be calculated using the actual operating
        voltage rather than the rated voltage.
        
        Requires lf_results to be populated (call run_load_flow first).
        """
        from ..core.elements import LoadShunt
        
        if self.lf_results is None:
            return
        
        for shunt in self.shunts:
            if isinstance(shunt, LoadShunt):
                # Get the load flow voltage for this bus
                bus_name = shunt.bus_name
                if bus_name in self.lf_results:
                    lf_voltage_kv = self.lf_results[bus_name].voltage_kv
                    shunt.set_lf_voltage(lf_voltage_kv)
    
    def _build_matrices(self, include_generators: bool = False) -> None:
        """
        Build admittance matrices from the network elements.
        
        Args:
            include_generators: If True, include generator admittances in diagonal
        """
        # Build load flow matrix (network only)
        self._Y_lf, self.bus_idx = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=MatrixType.LOAD_FLOW,
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w
        )
        
        # Build stability matrix (with loads)
        matrix_type = MatrixType.STABILITY_FULL if include_generators else MatrixType.STABILITY
        self._Y_stab, _ = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=matrix_type,
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w
        )
    
    def run_load_flow(self) -> bool:
        """
        Execute load flow calculation in PowerFactory.
        
        After successful load flow, updates load admittances with actual
        bus voltages and rebuilds the stability matrix for accurate
        constant impedance modeling.
        
        Returns:
            True if load flow converged, False otherwise
        """
        self._hide()
        success = run_load_flow(self.app)
        if success:
            self.lf_results = get_load_flow_results(self.app)
            self.gen_data = get_generator_data_from_pf(
                self.app, self.shunts, self.lf_results, self.base_mva
            )
            self.vs_data = get_voltage_source_data_from_pf(
                self.app, self.shunts, self.lf_results, self.base_mva
            )
            self.xnet_data = get_external_grid_data_from_pf(
                self.app, self.shunts, self.lf_results, self.base_mva
            )
            # Update gen_names to match gen_data order (used for plotting)
            self._gen_data_names = [g.name for g in self.gen_data]
            
            # Update load admittances with actual load flow voltages
            self._update_load_admittances_with_lf_voltage()
            
            # Rebuild stability matrix with updated load admittances
            self._build_matrices()

        self._show()
        return success
    
    def reduce_to_generators(
        self,
        include_voltage_sources: bool = True,
        include_external_grids: bool = True
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
        self._hide()
        if self._Y_stab is None:
            raise RuntimeError("Must call build_matrices() first")
        
        self._Y_reduced, self.gen_names, self.source_types = reduce_to_generator_internal_buses(
            self._Y_stab, self.bus_idx, self.shunts, self.base_mva,
            include_voltage_sources=include_voltage_sources,
            include_external_grids=include_external_grids
        )

        self._show()
    
    def calculate_power_ratios(self, disturbance_source_name: str) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Calculate power distribution ratios for a source (generator/voltage source) trip.
        
        Args:
            disturbance_source_name: Name of the source that trips
            
        Returns:
            Tuple of (ratios array, source names in order, source types in order)
        """
        self._hide()
        if self._Y_reduced is None:
            raise RuntimeError("Must call reduce_to_generators() first")
        if self.gen_data is None:
            raise RuntimeError("Must call run_load_flow() first")
        
        # Build combined source data matching the order in gen_names
        # gen_names contains all sources in the order they were added to Y_reduced
        self._build_source_data()

        ratios, source_names_order, source_types_order = calculate_power_distribution_ratios(
            self._Y_reduced, self.source_data, disturbance_source_name
        )
        self._show()
        return ratios, source_names_order, source_types_order
    
    def calculate_all_power_ratios(
        self,
        outage_generators: list[str] = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """
        Calculate power distribution ratios matrix for multiple generator outages.
        
        Each row corresponds to one generator outage, each column corresponds to
        a source (generator or voltage source) receiving power.
        
        Args:
            outage_generators: List of generator names to trip. If None, all 
                              synchronous generators will be used.
            normalize: If True, normalize each row to sum to 100%
            
        Returns:
            Tuple of:
                - ratios_matrix: 2D numpy array (n_outages × n_sources)
                - outage_names: List of generator names that were tripped (row labels)
                - source_names: List of all source names (column labels)
                - source_types: List of source types (column types)
        """
        self._hide()
        if self._Y_reduced is None:
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
                    self._Y_reduced, self.source_data, gen_name
                )
                
                if normalize:
                    ratio_sum = np.sum(ratios_i)
                    if ratio_sum > 0:
                        ratios_i = (ratios_i / ratio_sum) * 100
                
                all_ratios.append(ratios_i)
                valid_outages.append(gen_name)
                
            except Exception as e:
                logger.warning(f"Skipping {gen_name}: {e}")
        
        ratios_matrix = np.array(all_ratios)
        
        self._show()
        return ratios_matrix, valid_outages, source_names, source_types
    
    def _build_source_data(self) -> None:
        """
        Build combined source data list matching the order in gen_names.
        
        This combines generator, voltage source, and external grid data in the same order
        as they appear in the reduced Y-matrix.
        """
        # Create lookup dictionaries
        gen_lookup = {g.name: g for g in self.gen_data} if self.gen_data else {}
        vs_lookup = {v.name: v for v in self.vs_data} if self.vs_data else {}
        xnet_lookup = {x.name: x for x in self.xnet_data} if self.xnet_data else {}
        
        self.source_data = []
        for name, stype in zip(self.gen_names, self.source_types):
            if stype == 'generator' and name in gen_lookup:
                self.source_data.append(gen_lookup[name])
            elif stype == 'voltage_source' and name in vs_lookup:
                self.source_data.append(vs_lookup[name])
            elif stype == 'external_grid' and name in xnet_lookup:
                self.source_data.append(xnet_lookup[name])
            else:
                logger.warning(f"Source '{name}' (type: {stype}) not found in load flow data")
    
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
    
    def _print_network_summary(self, title: str = "Network summary:") -> None:
        """Print a summary of network elements to console."""
        n_lines = len([b for b in self.branches if type(b).__name__ == 'LineBranch'])
        n_trafos = len([b for b in self.branches if type(b).__name__ == 'TransformerBranch'])
        n_trafos_3w = len(self.transformers_3w)
        n_switches = len([b for b in self.branches if type(b).__name__ == 'SwitchBranch'])
        n_zpu = len([b for b in self.branches if type(b).__name__ == 'CommonImpedanceBranch'])
        n_sind = len([b for b in self.branches if type(b).__name__ == 'SeriesReactorBranch'])
        n_gens = len([s for s in self.shunts if type(s).__name__ == 'GeneratorShunt'])
        n_loads = len([s for s in self.shunts if type(s).__name__ == 'LoadShunt'])
        n_xnets = len([s for s in self.shunts if type(s).__name__ == 'ExternalGridShunt'])
        n_vacs = len([s for s in self.shunts if type(s).__name__ == 'VoltageSourceShunt'])
        n_shunts = len([s for s in self.shunts if type(s).__name__ == 'ShuntFilterShunt'])
        n_buses = len(get_unique_buses(self.branches, self.shunts, self.transformers_3w))
        
        print(f"{title}")
        if n_lines > 0:
            print(f"  Lines:              {n_lines}")
        if n_trafos > 0:
            print(f"  Transformers (2W):  {n_trafos}")
        if n_trafos_3w > 0:
            print(f"  Transformers (3W):  {n_trafos_3w}")
        if n_switches > 0:
            print(f"  Switches:           {n_switches}")
        if n_zpu > 0:
            print(f"  Common impedances:  {n_zpu}")
        if n_sind > 0:
            print(f"  Series reactors:    {n_sind}")
        if n_gens > 0:
            print(f"  Generators:         {n_gens}")
        if n_loads > 0:
            print(f"  Loads:              {n_loads}")
        if n_xnets > 0:
            print(f"  External grids:     {n_xnets}")
        if n_vacs > 0:
            print(f"  Voltage sources:    {n_vacs}")
        if n_shunts > 0:
            print(f"  Shunt filters:      {n_shunts}")
        print(f"  Buses:              {n_buses}")
    
    def _hide(self) -> None:
        """Hide the PowerFactory application window."""
        if self.app is not None:
            self.app.Hide()
    
    def _show(self) -> None:
        """Show the PowerFactory application window."""
        if self.app is not None:
            self.app.Show()
    
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
    def Y_lf_matrix(self) -> np.ndarray:
        """Get load flow Y-matrix. Raises if not built yet."""
        if self._Y_lf is None:
            raise RuntimeError("Y_lf not built - call _build_matrices() first")
        return self._Y_lf
    
    @property
    def Y_stab_matrix(self) -> np.ndarray:
        """Get stability Y-matrix (with loads). Raises if not built yet."""
        if self._Y_stab is None:
            raise RuntimeError("Y_stab not built - call _build_matrices() first")
        return self._Y_stab
    
    @property
    def Y_reduced_matrix(self) -> np.ndarray:
        """Get reduced Y-matrix (generator internal buses). Raises if not built yet."""
        if self._Y_reduced is None:
            raise RuntimeError("Y_reduced not built - call reduce_to_generators() first")
        return self._Y_reduced
    
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
            Y_lf=self._Y_lf,
            Y_stab=self._Y_stab
        )
        
        if print_results:
            print_diagnostics(diag)
        
        return diag
