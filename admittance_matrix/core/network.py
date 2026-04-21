"""
Network wrapper class for PowerFactory admittance matrix operations.

This module provides a high-level Network class that encapsulates all
the functionality of the admittance_matrix library.
"""

import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import powerfactory as pf
from typing import Literal

from ..matrices.builder import build_admittance_matrix, build_admittance_matrices, MatrixBuildResult, MatrixType
from ..matrices.reducer import extend_matrix_to_generator_internal_nodes, perform_kron_reduction, perform_kron_reduction_on_busbars
from ..matrices.analysis import calculate_power_distribution_ratios, calculate_power_distribution_ratios_prefault_postfault
from ..matrices.topology import simplify_topology
from ..adapters.powerfactory import get_network_elements, get_main_bus_names
from ..adapters.powerfactory import run_load_flow, get_load_flow_results, get_generator_data_from_pf, get_voltage_source_data_from_pf, get_external_grid_data_from_pf
from ..adapters.powerfactory import GeneratorResult, VoltageSourceResult, ExternalGridResult
from ..adapters.powerfactory.results import BusResult, GeneratorResult, VoltageSourceResult, ExternalGridResult
from .elements import BranchElement, ShuntElement, Transformer3WBranch, GeneratorShunt, VoltageSourceShunt, ExternalGridShunt
from .reductionEngine import perform_reduction_mode1, perform_reduction_mode2

logger = logging.getLogger(__name__)
SourceShunt = GeneratorShunt | VoltageSourceShunt | ExternalGridShunt

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

        # Network data
        self.base_mva = base_mva
        self.simplify_topology = simplify_topology
        self.verbose = verbose

        # Initialize all network data structures to empty/None
        self._init_state()

        try:
            self._hide()
            self._initialize_network_model()
        finally:
            self._show()

    def _init_state(self) -> None:
        # =============== Initialize Network element states ===============
        self.branches:          list[BranchElement] = []                # All branch elements
        self.shunts:            list[ShuntElement] = []                 # All shunt elements
        self.transformers_3w:   list[Transformer3WBranch] = []          # 3-winding transformers
        self.bus_names:         list[str] = []                          # List of unique bus names

        # =============== Bus mapping for simplified topology (original bus name to merged bus name) ===============
        self.bus_mapping:   dict[str, str] | None = None

        # =============== Admittance matrices and related data ===============
        # Y matrices result
        self.Y_result: MatrixBuildResult | None = None

        # Compatibility
        self._Y_lf:     npt.NDArray[np.complex128] | None = None      # Admittance matrix for load flow (network only)
        self._Y_stab:   npt.NDArray[np.complex128] | None = None      # Admittance matrix including loads and generators
        self._bus_idx:  dict[str, int] | None = None                  # Mapping of bus names to indices in Y matrices
        self._Y_reduced: npt.NDArray[np.complex128] | None = None   # Reduced to generator internal buses

        # =============== Load-Flow Snapshot ===============
        # Busbars Load-Flow results
        self.lf_results:    dict[str, BusResult] | None = None
        
        # SyncGen, VoltageSource, ExternalGrid data from load flow results
        self.gen_data:      list[GeneratorResult] | None = None
        self.vs_data:       list[VoltageSourceResult] | None = None
        self.xnet_data:     list[ExternalGridResult] | None = None

        # Combined data
        self.source_data:   list[GeneratorResult | VoltageSourceResult | ExternalGridResult] | None = None
        self.source_names:  list[str] = []
        self.source_types:  list[str] = []

    def _initialize_network_model(self) -> None:
        self._extract_network()
        self._build_matrices()

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
        matrices: MatrixBuildResult = build_admittance_matrices(
            bus_names=self.bus_names,
            branches=self.branches,
            shunts=self.shunts,
            base_mva=self.base_mva,
            transformers_3w=self.transformers_3w,
        )

        # Save for compatibility
        self._Y_lf = matrices.y_lf
        self._Y_stab = matrices.y_stab
        self._bus_idx = matrices.bus_idx

        # Build reduced matrix to generator internal nodes
        self._Y_reduced = self._reduce_network_to_internal_generator_nodes()
    
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

        try:
            success = run_load_flow(self.app)
            if success:
                # ============= Get load flow results for all buses =============
                self.lf_results = get_load_flow_results(self.app)

                # ============= Get load flow results of sources and build source data =============
                self.syn_gens    = [s for s in self.shunts if isinstance(s, GeneratorShunt)]
                self.v_sources   = [s for s in self.shunts if isinstance(s, VoltageSourceShunt)]
                self.xnets       = [s for s in self.shunts if isinstance(s, ExternalGridShunt)]

                # Obtain LF results for dynamic sources used in the reduction workflow
                self.gen_data   =   get_generator_data_from_pf(self.app, self.syn_gens, self.lf_results, self.base_mva)
                self.vs_data    =   get_voltage_source_data_from_pf(self.app, self.v_sources, self.lf_results, self.base_mva)
                self.xnet_data  =   get_external_grid_data_from_pf(self.app, self.xnets, self.lf_results, self.base_mva)

                # Update source_names and source_types to include all source names from load flow data
                self.source_names = [g.name for g in self.gen_data]
                self.source_types = ['generator'] * len(self.gen_data)

                self.source_names.extend([v.name for v in self.vs_data])
                self.source_types.extend(['voltage_source'] * len(self.vs_data))

                self.source_names.extend([x.name for x in self.xnet_data])
                self.source_types.extend(['external_grid'] * len(self.xnet_data))

                # Combined source data
                self.source_data = self.gen_data + self.vs_data + self.xnet_data

                # ============= Update load admittances with actual load flow voltages =============
                self._update_load_admittances_with_lf_voltage()
                
                # ============= Rebuild matrices with updated load admittances =============
                self._build_matrices()
        finally:
            self._show()
        return success
    
    def _reduce_network_to_internal_generator_nodes(self) -> npt.NDArray[np.complex128]:
        if self._Y_stab is None:
            raise RuntimeError("Must call build_matrices() first")
        if self._bus_idx is None:
            raise RuntimeError("bus_idx is not initialized")
        
        filtered_sources = self._get_all_sources()

        # Get extended matrix with internal generator nodes (FULL EXTENDED MATRIX)
        self._Y_extended = extend_matrix_to_generator_internal_nodes(
            Y_bus=self._Y_stab,
            bus_idx=self._bus_idx,
            sources=filtered_sources,
            base_mva=self.base_mva,
        )

        n_sources = len(filtered_sources)
        # Reduce to only internal generator buses (indices 0 to n_sources-1)
        indices_to_keep = list(range(n_sources))
        Y_reduced = perform_kron_reduction(self._Y_extended, indices_to_keep)

        return Y_reduced
    
    def _get_all_sources(self, name_to_exclude: str | None = None) -> list[SourceShunt]:
        """Get all source shunt elements (generators, voltage sources, external grids)."""
        source_types = (GeneratorShunt, VoltageSourceShunt, ExternalGridShunt)
        sources: list[SourceShunt] = []

        for shunt in self.shunts:
            if name_to_exclude is not None and shunt.name == name_to_exclude:
                continue

            if isinstance(shunt, source_types):
                sources.append(shunt)

        return sources
    
    def calculate_power_ratios(self, disturbance_source_name: str, MODE: Literal[0, 1, 2] = 1) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
        """
        Calculate power distribution ratios for a source (generator/voltage source) trip.
        
        Args:
            disturbance_source_name: Name of the source that trips
            
        Returns:
            Tuple of (ratios array, source names in order, source types in order)
        """
        # self._hide()
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
            filtered_sources = self._get_all_sources()
            Y_mode1 = perform_reduction_mode1(
                bus_names=self.bus_names,
                branches=self.branches,
                branches_3w_traformers=self.transformers_3w,
                shunts=self.shunts,
                sources=filtered_sources,
                BASE_MVA=self.base_mva,
                excluded_source_name=disturbance_source_name,
            )
            ratios, source_names_order, source_types_order = calculate_power_distribution_ratios(
                Y_mode1, self.source_data, disturbance_source_name, dist_angle_mode="terminal_current"
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

            Y_mode2 = perform_reduction_mode2(
                bus_names=self.bus_names,
                branches=self.branches,
                branches_3w_traformers=self.transformers_3w,
                shunts=self.shunts,
                filtered_sources=self._get_all_sources(name_to_exclude=disturbance_source_name),
                BASE_MVA=self.base_mva,
                excluded_source_name=disturbance_source_name,
            )
            
            # Get all indices except the disturbance source
            n_sources = len(source_names_order)
            keep_idx = [i for i in range(n_sources) if i != dist_idx]
            
            ratios, _ = calculate_power_distribution_ratios_prefault_postfault(
                        self._Y_reduced, Y_mode2, E_abs, E_angle, 
                        dist_idx=dist_idx, keep_idx=keep_idx
            )
        return ratios, source_names_order, source_types_order
    
    def calculate_all_power_ratios(
        self,
        outage_generators: list[str] | None = None,
        MODE: Literal[0, 1, 2] = 1,
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
        if self.source_data is None:
            raise RuntimeError("Source data is not available. Ensure run_load_flow() has been called and source data is built.")
        
        # Default to all synchronous generators if not specified
        if outage_generators is None:
            outage_generators = [
                name for name, stype in zip(self.source_names, self.source_types) 
                if stype == 'generator'
            ]

        all_ratios:     list[npt.NDArray[np.float64]] = []
        valid_outages:  list[str] = []
        source_names:   list[str] = []
        source_types:   list[str] = []
        
        for _, gen_name in enumerate(outage_generators):
            try:
                ratios_i, source_names, source_types = self.calculate_power_ratios(gen_name, MODE=MODE)
                
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
                if shunt.zone is not None:
                    return shunt.zone
                else:
                    return "None"
        
        logger.warning(f"Zone not found for source: {source_name}")
        return "None"
    
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
            raise RuntimeError("lf_results is not available. Ensure run_load_flow() has been called and lf_results is populated.")
        
        for shunt in self.shunts:
            if isinstance(shunt, LoadShunt):
                bus_name = shunt.bus_name
                # Get the load flow voltage for this bus
                if bus_name in self.lf_results:
                    lf_voltage_pu = self.lf_results[bus_name].voltage_pu
                    # Set LF voltage in the shunt element
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
        from ..core.elements import LoadShunt
        print(load_voltages)
        
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

        if self._bus_idx is None:
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
            bus_names=self.bus_names,
            branches=self.branches,
            shunts=non_source_shunts,
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
        n_pvsys = len([s for s in self.shunts if type(s).__name__ == 'PVSystemShunt'])
        n_genstats = len([s for s in self.shunts if type(s).__name__ == 'StaticGeneratorShunt'])
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
        if n_pvsys > 0:
            print(f"  PV systems:         {n_pvsys}")
        if n_genstats > 0:
            print(f"  Static generators:  {n_genstats}")
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

    @property
    def gen_zones(self) -> list[str | None]:
        """
        Get zones for all sources in source_names order.
        
        Returns:
            List of zone names matching source_names order (None if zone not found)
        """
        if not self.source_names:
            return []
        return [self.get_zone(name) for name in self.source_names]
    
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

    @property
    def n_pv_systems(self) -> int:
        """Number of PV systems in the network."""
        return len([s for s in self.shunts if type(s).__name__ == 'PVSystemShunt'])

    @property
    def n_static_generators(self) -> int:
        """Number of static generators in the network."""
        return len([s for s in self.shunts if type(s).__name__ == 'StaticGeneratorShunt'])
