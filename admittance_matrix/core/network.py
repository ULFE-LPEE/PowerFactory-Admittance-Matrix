"""
Network wrapper class for PowerFactory admittance matrix operations.

This module provides a high-level Network class that encapsulates all
the functionality of the admittance_matrix library.
"""

import logging

import numpy as np
import pandas as pd
import powerfactory as pf
from typing import Literal

from ..matrices.builder import build_admittance_matrix, MatrixType
from ..matrices.reducer import extend_matrix_to_generator_internal_nodes, perform_kron_reduction, perform_kron_reduction_on_busbars
from ..matrices.analysis import calculate_power_distribution_ratios, calculate_power_distribution_ratios_prefault_postfault
from ..matrices.topology import simplify_topology
from ..adapters.powerfactory import get_network_elements, get_main_bus_names
from ..adapters.powerfactory import run_load_flow, get_load_flow_results, get_generator_data_from_pf, get_voltage_source_data_from_pf, get_external_grid_data_from_pf
from ..adapters.powerfactory import GeneratorResult, VoltageSourceResult, ExternalGridResult
from ..adapters.powerfactory.results import BusResult, GeneratorResult, VoltageSourceResult, ExternalGridResult
from .elements import BranchElement, ShuntElement, Transformer3WBranch, GeneratorShunt

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
        self._hide() # Hide PowerFactory window during operations

        # ========================= Initialize Network data =========================
        # Network data
        self.base_mva = base_mva
        self.simplify_topology = simplify_topology
        self.verbose = verbose
        self.bus_mapping = None  # Mapping from original to merged bus names

        # All Network elements
        self.branches: list[BranchElement] = [] # All branch elements
        self.shunts: list[ShuntElement] = [] # All shunt elements
        self.transformers_3w: list[Transformer3WBranch] = []  # 3-winding transformers
        self.bus_names: list[str] = []
        
        # Matrices
        self._Y_lf: np.ndarray | None = None        # Admittance matrix for load flow (network only)
        self._Y_stab: np.ndarray | None = None      # Admittance matrix including loads and generators
        self._Y_reduced: np.ndarray | None = None   # Reduced to generator internal buses
        self.bus_idx: dict[str, int] | None = None                         # Mapping of bus names to indices in Y matrices
        self.gen_names = []                         # Names of sources in reduced matrix
        self.source_types = []                      # Types: 'generator', 'voltage_source', 'external_grid'
        
        # Results
        self.lf_results: dict[str, BusResult] | None = None
        self.gen_data: list[GeneratorResult] | None = None                  # Generator results from loadflow
        self.vs_data: list[VoltageSourceResult] | None = None               # Voltage source results from loadflow
        self.xnet_data: list[ExternalGridResult] | None = None              # External grid results from loadflow
        self.source_data: list[GeneratorResult | VoltageSourceResult | ExternalGridResult] | None = None         # Combined source data for analysis
        
        # ========================= Extract network elements and build admittance matrices =========================
        self._extract_network()
        self._build_matrices()

        self._show() # Show PowerFactory window after operations

    def _extract_network(self):
        """Extract network elements from PowerFactory."""
        self.branches, self.shunts, self.transformers_3w = get_network_elements(self.app)
        
        # Print extraction summary before simplification
        if self.verbose:
            self._print_network_summary("Network extracted:")
        
        # Optionally merge buses connected by closed switches
        if self.simplify_topology:
            n_buses_before = len(self._get_unique_buses(self.branches, self.shunts, self.transformers_3w))
            
            # Get main busbars to preserve their names during merging
            main_buses = get_main_bus_names(self.app)
            
            self.branches, self.shunts, self.transformers_3w, self.bus_mapping = simplify_topology(
                self.branches, self.shunts, self.transformers_3w, main_buses=main_buses
            )
            n_buses_after = len(self._get_unique_buses(self.branches, self.shunts, self.transformers_3w))
            if self.verbose:
                print(f"Topology simplified: {n_buses_before} to {n_buses_after} buses ({n_buses_before - n_buses_after} eliminated)")
                self._print_network_summary("Network after simplification:")
        
        self.bus_names = self._get_unique_buses(self.branches, self.shunts, self.transformers_3w)
    
    def _build_matrices(self) -> None:
        """
        Build admittance matrices from the network elements.
        """
        # Build load flow matrix (network only)
        self._Y_lf, self.bus_idx = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=MatrixType.LOAD_FLOW,
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w
        )
        
        # Build stability matrix (with loads and generators)
        self._Y_stab, _ = build_admittance_matrix(
            self.branches, self.shunts, self.bus_names,
            matrix_type=MatrixType.STABILITY,
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
            # Get load flow results for all buses
            self.lf_results = get_load_flow_results(self.app)

            # Obtain LF results for ElmSyn, ElmGenStat, ExtGrid
            self.gen_data = get_generator_data_from_pf(self.app, self.shunts, self.lf_results, self.base_mva)
            self.vs_data = get_voltage_source_data_from_pf(self.app, self.shunts, self.lf_results, self.base_mva)
            self.xnet_data = get_external_grid_data_from_pf(self.app, self.shunts, self.lf_results, self.base_mva)

            # Update gen_names and source_types to include all source names from load flow data
            self.gen_names = [g.name for g in self.gen_data]
            self.source_types = ['generator'] * len(self.gen_data)

            self.gen_names.extend([v.name for v in self.vs_data])
            self.source_types.extend(['voltage_source'] * len(self.vs_data))

            self.gen_names.extend([x.name for x in self.xnet_data])
            self.source_types.extend(['external_grid'] * len(self.xnet_data))

            # Update load admittances with actual load flow voltages
            self._update_load_admittances_with_lf_voltage()
            
            # Rebuild matrix with updated load admittances for accurate modeling of constant impedance loads
            self._build_matrices()

            # Build source data once
            self._build_source_data()

        self._show()
        return success
    
    def reduce_to_generators(
        self,
        outage_source_name: str | None = None,
        MODE: Literal[0, 1, 2] = 1,
    ) -> np.ndarray:
        """
        Apply Kron reduction to obtain generator internal bus matrix.
        
        Optionally includes voltage sources and external grids as additional 
        sources that participate in power redistribution.
        
        The Y_stab matrix must be built first using build_matrices().
        
        Args:
            include_voltage_sources: If True, include AC voltage sources in reduction
            include_external_grids: If True, include external grids in reduction
            outage_source_name: If provided, exclude this source's admittance from network
        """
        if self._Y_stab is None:
            raise RuntimeError("Must call build_matrices() first")
        if self.bus_idx is None:
            raise RuntimeError("bus_idx is not initialized")
        
        # ========================= Get full extended matrix for whole network =========================
        filtered_sources = self.shunts
        filtered_sources = [s for s in self.shunts if (isinstance(s, GeneratorShunt))]
        filtered_sources = self._get_all_sources()

        # Get extended matrix with internal generator nodes (FULL EXTENDED MATRIX)
        self._Y_extended = extend_matrix_to_generator_internal_nodes(
            Y_bus=self._Y_stab,
            bus_idx=self.bus_idx,
            sources=filtered_sources,
            base_mva=self.base_mva,
        )

        n_sources = len(filtered_sources)
        # Reduce to only internal generator buses (indices 0 to n_sources-1)
        indices_to_keep = list(range(n_sources))
        self._Y_reduced = perform_kron_reduction(self._Y_extended, indices_to_keep)

        # ========================= MODE 1: Get extended matrix for missing outaged generator admittance in M sub-matrix =========================
        if MODE == 1:
            Y_stab_mode1, self.bus_idx = build_admittance_matrix(
                self.branches, self.shunts, self.bus_names,
                matrix_type=MatrixType.STABILITY,
                base_mva=self.base_mva,
                transformers_3w=self.transformers_3w,
                exclude_source_name=outage_source_name  # Exclude generator admittance
            )
            #** This is development feature to test the method
            filtered_sources = self.shunts
            filtered_sources = [s for s in self.shunts if (isinstance(s, GeneratorShunt))]
            filtered_sources = self._get_all_sources()

            # Get extended matrix with internal generator nodes (EXTENDED MATRIX MODIFIED)
            self._Y_extended_mode1 = extend_matrix_to_generator_internal_nodes(
                Y_bus=Y_stab_mode1,
                bus_idx=self.bus_idx,
                sources=filtered_sources,
                base_mva=self.base_mva,
            )

            n_sources = len(filtered_sources)
            # Reduce to only internal generator buses (indices 0 to n_sources-1)
            indices_to_keep = list(range(n_sources))
            self._Y_reduced_mode1 = perform_kron_reduction(self._Y_extended_mode1, indices_to_keep)

        # ========================= MODE 2: Get extended matrix to generator internal nodes completly without outaged generator node =========================
        if MODE == 2:
            Y_stab_mode2, self.bus_idx = build_admittance_matrix(
                self.branches, self.shunts, self.bus_names,
                matrix_type=MatrixType.STABILITY,
                base_mva=self.base_mva,
                transformers_3w=self.transformers_3w,
                exclude_source_name=outage_source_name  # Exclude generator admittance
            )
            filtered_sources = self.shunts
            filtered_sources = [
                s for s in self.shunts 
                if (isinstance(s, GeneratorShunt) and s.name != outage_source_name)
            ]
            filtered_sources = self._get_all_sources(name_to_exclude=outage_source_name)

            # Get extended matrix with internal generator nodes (EXTENDED MATRIX WITHOUT OUTAGED GENERATOR)
            self._Y_extended_mode2 = extend_matrix_to_generator_internal_nodes(
                Y_bus=Y_stab_mode2,
                bus_idx=self.bus_idx,
                sources=filtered_sources,
                base_mva=self.base_mva,
            )

            n_sources = len(filtered_sources)
            # Reduce to only internal generator buses (indices 0 to n_sources-1)
            indices_to_keep = list(range(n_sources))
            self._Y_reduced_mode2 = perform_kron_reduction(self._Y_extended_mode2, indices_to_keep)

        # Return reduced matrix
        return self._Y_reduced

    def get_generator_busbar_distances(self, include_gen_Y: bool = False) -> pd.DataFrame:
        """
        Get electrical distances between generator busbars.

        Builds a stability Y-matrix, reduces it to generator busbars, and returns a
        distance matrix indexed by generator names.

        Args:
            include_gen_Y: If True, include generator shunt
                admittances in the stability matrix. Voltage sources and
                external grids are always excluded.
        """
        from ..core.elements import VoltageSourceShunt, ExternalGridShunt

        if self.bus_idx is None:
            raise RuntimeError("bus_idx is not initialized")

        gen_shunts = [s for s in self.shunts if isinstance(s, GeneratorShunt)]
        if not gen_shunts:
            raise RuntimeError("No GeneratorShunt elements found in the network")

        # Build stability matrix, optionally including generator admittances
        if include_gen_Y:
            non_source_shunts = [
                s for s in self.shunts
                if not isinstance(s, (VoltageSourceShunt, ExternalGridShunt))
            ]
        else:
            non_source_shunts = [
                s for s in self.shunts
                if not isinstance(s, (GeneratorShunt, VoltageSourceShunt, ExternalGridShunt))
            ]

        Y_stab_no_sources, bus_idx_local = build_admittance_matrix(
            self.branches,
            non_source_shunts,
            self.bus_names,
            matrix_type=MatrixType.STABILITY,
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w,
        )

        gen_names = [g.name for g in gen_shunts]
        gen_bus_indices = [bus_idx_local[g.bus_name] for g in gen_shunts]
        unique_bus_indices = sorted(set(gen_bus_indices))
        Y_reduced = perform_kron_reduction_on_busbars(Y_stab_no_sources, unique_bus_indices)

        # Electrical distance derived directly from the reduced matrix
        distance_unique = np.abs(Y_reduced)

        pos_map = {bus_idx: i for i, bus_idx in enumerate(unique_bus_indices)}
        positions = [pos_map[idx] for idx in gen_bus_indices]
        distance_full = distance_unique[np.ix_(positions, positions)]

        return pd.DataFrame(distance_full, index=gen_names, columns=gen_names)
    
    def _get_all_sources(self, name_to_exclude: str | None = None) -> list[ShuntElement]:
        """Get all source shunt elements (generators, voltage sources, external grids)."""
        from ..core.elements import VoltageSourceShunt, ExternalGridShunt
        
        sources = []
        if name_to_exclude is not None:
            for shunt in self.shunts:
                if shunt.name == name_to_exclude:
                    continue
                if isinstance(shunt, (GeneratorShunt, VoltageSourceShunt, ExternalGridShunt)):
                    sources.append(shunt)
        else:
            for shunt in self.shunts:
                if isinstance(shunt, (GeneratorShunt, VoltageSourceShunt, ExternalGridShunt)):
                    sources.append(shunt)
                    
        return sources
    
    def calculate_power_ratios(self, disturbance_source_name: str, MODE: Literal[0, 1, 2] = 1) -> tuple[np.ndarray, list[str], list[str]]:
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
        if self.source_data is None:
            raise RuntimeError("Source data is not available. Ensure run_load_flow() has been called and source data is built.")

        # ============== MODE 0: Calculation of power ratios using internal voltage angle as disturbance angle ===============
        if MODE == 0:
            ratios, source_names_order, source_types_order = calculate_power_distribution_ratios(
                self._Y_reduced, self.source_data, disturbance_source_name, dist_angle_mode="internal_E"
            )

        # ============== MODE 1: Calculation of power ratios via missing generator admittance in M submatrix ===============
        elif MODE == 1:
            ratios, source_names_order, source_types_order = calculate_power_distribution_ratios(
                self._Y_reduced_mode1, self.source_data, disturbance_source_name, dist_angle_mode="terminal_current"
            )

        # ============== MODE 2: Calculation of power ratios using pre-fault and post-fault admittance matrices ===============
        else:
            E_abs = np.array([np.abs(s.internal_voltage) for s in self.source_data], dtype=float).flatten()
            E_angle = np.array([np.angle(s.internal_voltage) for s in self.source_data], dtype=float).flatten()
            print(len(E_abs), len(E_angle))

            source_names_order = [s.name for s in self.source_data]
            source_types_order = [s.source_type for s in self.source_data]

            # Find the index of the disturbance source
            dist_idx = source_names_order.index(disturbance_source_name) if disturbance_source_name in source_names_order else None
            if dist_idx is None:
                raise ValueError(f"Disturbance source '{disturbance_source_name}' not found in source names")
            
            # Get all indices except the disturbance source
            n_sources = len(source_names_order)
            keep_idx = [i for i in range(n_sources) if i != dist_idx]
            
            ratios, _ = calculate_power_distribution_ratios_prefault_postfault(
                        self._Y_reduced, self._Y_reduced_mode2, E_abs, E_angle, 
                        dist_idx=dist_idx, keep_idx=keep_idx
            )

        self._show()
        return ratios, source_names_order, source_types_order
    
    def calculate_all_power_ratios(
        self,
        outage_generators: list[str] | None = None,
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
        
        # Build source data once
        self._build_source_data()
        if self.source_data is None:
            raise RuntimeError("Source data is not available. Ensure run_load_flow() has been called and source data is built.")
        
        # Default to all synchronous generators if not specified
        if outage_generators is None:
            outage_generators = [
                name for name, stype in zip(self.gen_names, self.source_types) 
                if stype == 'generator'
            ]

        all_ratios = []
        valid_outages = []
        source_names = []
        source_types = []
        
        for _, gen_name in enumerate(outage_generators):
            try:
                # TODO: We might need to fix function call below because it takes very long time to compute in a loop...
                self.reduce_to_generators(outage_source_name=gen_name)

                ratios_i, source_names, source_types = calculate_power_distribution_ratios(
                    self._Y_reduced, self.source_data, gen_name, dist_angle_mode="terminal_current"
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
    
    def get_zone(self, source_name: str) -> str | None:
        """
        Get the zone for a source (generator or voltage source).
        
        Args:
            source_name: Name of the source
            
        Returns:
            Zone name string, or None if not found
        """
        # Find the source in shunts list
        for shunt in self.shunts:
            if shunt.name == source_name:
                return shunt.zone
        
        logger.warning(f"Zone not found for source: {source_name}")
        return None
    
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
                    lf_voltage_pu = self.lf_results[bus_name].voltage_pu
                    shunt.set_lf_voltage(lf_voltage_pu)
    
    def update_load_admittances_with_post_disturbance_voltage(self, load_voltages: dict[str, float]) -> None:
        """
        Update load admittances using post-disturbance voltages from RMS simulation.
        
        This allows recalculating power distribution ratios using the voltage
        profile that exists after a generator trip, rather than the pre-disturbance
        load flow voltages.
        
        Args:
            load_voltages: Dictionary mapping load names to their voltage (kV) 
                          at the disturbance time from RMS simulation results.
        """
        from ..core.elements import LoadShunt, GeneratorShunt
        
        updated_count = 0
        for shunt in self.shunts:
            if isinstance(shunt, LoadShunt):
                if shunt.name in load_voltages:
                    voltage_kv = load_voltages[shunt.name]
                    shunt.set_lf_voltage(voltage_kv)
                    updated_count += 1

        if self.verbose:
            print(f"Updated {updated_count} load admittances with post-disturbance voltages")
        
        # Rebuild matrices with updated load admittances
        self._build_matrices()
        
        # Re-run Kron reduction if it was done before
        if self._Y_reduced is not None:
            self.reduce_to_generators()
    
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
        n_buses = len(self._get_unique_buses(self.branches, self.shunts, self.transformers_3w))
        
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

    @staticmethod
    def _get_unique_buses(
        branches: list[BranchElement],
        shunts: list[ShuntElement],
        transformers_3w: list[Transformer3WBranch] | None = None,
    ) -> list[str]:
        """Extract unique bus names from branches, shunts, and 3-winding transformers."""
        buses = set()

        for b in branches:
            buses.add(b.from_bus_name)
            buses.add(b.to_bus_name)

        for s in shunts:
            buses.add(s.bus_name)

        # Add 3-winding transformer buses (HV, MV, LV - no virtual star node needed)
        if transformers_3w:
            for t3w in transformers_3w:
                buses.add(t3w.hv_bus_name)
                buses.add(t3w.mv_bus_name)
                buses.add(t3w.lv_bus_name)

        return sorted(list(buses))
    
    def _show(self) -> None:
        """Show the PowerFactory application window."""
        if self.app is not None:
            self.app.Show()
    
    @property
    def gen_zones(self) -> list[str | None]:
        """
        Get zones for all sources in gen_names order.
        
        Returns:
            List of zone names matching gen_names order (None if zone not found)
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
            raise RuntimeError("self._Y_lf not built - call _build_matrices() first")
        return self._Y_lf
    
    @property
    def Y_stab_matrix(self) -> np.ndarray:
        """Get stability Y-matrix (with loads). Raises if not built yet."""
        if self._Y_stab is None:
            raise RuntimeError("self._Y_stab not built - call _build_matrices() first")
        return self._Y_stab
    
    @property
    def Y_reduced_matrix(self) -> np.ndarray:
        """Get reduced Y-matrix (generator internal buses). Raises if not built yet."""
        if self._Y_reduced is None:
            raise RuntimeError("self._Y_reduced not built - call reduce_to_generators() first")
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
