"""
Utility functions for PowerFactory operations.
"""
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
import os
import time
import powerfactory as pf
from powerfactory import DataObject

logger = logging.getLogger(__name__)

def init_project(app: pf.Application, project_path: str) -> bool:
    """
    Initialize a PowerFactory project.
    
    Args:
        app: PowerFactory application instance
        project_path: Full path to the project (e.g., "User\\ProjectName")
        
    Returns:
        True if successful, False otherwise
    """
    err = app.ActivateProject(project_path)
    return err == 0

def import_pfd_file(app: pf.Application, pfd_file_project: str, pfd_project_name: str) -> pf.DataObject:
    # Get current user folder
    current_user_folder = app.GetCurrentUser()
    if not current_user_folder:
        raise ValueError("Current user folder not found. Please ensure you are logged in to PowerFactory.")
    
    # Create new folder in the current user folder
    new_folder = current_user_folder.CreateObject("IntFolder", "Y Bus Example")

    # Get and set the ComImp command
    ComImp = app.GetFromStudyCase("ComPfdImport")
    ComImp.SetAttribute("g_file", pfd_file_project)
    ComImp.SetAttribute("g_target", new_folder)
    
    project_folder_name = new_folder.GetAttribute("loc_name")
    project_path = project_folder_name + "\\" + pfd_project_name
    ComImp.Execute() # type: ignore

    # Activate the project
    app.ActivateProject(project_path)
    project = app.GetActiveProject()
    project.Activate() # type: ignore

    return project

def get_simulation_data(GEN_OUT: str, path: str, available_measurements: Optional[List[str]] = None, pfResultsName: str = "All calculations", dist_index: int = 104) -> tuple:
    """
    Extract simulation data and optionally filter to available measurements.
    
    Parameters:
    -----------
    GEN_OUT : str
        Name of the generator that experienced the outage
    path : str
        Path to the CSV file with simulation results
    available_measurements : list of str, optional
        List of generator NAMES that have available measurements.
        If None, all generators are included.
    pfResultsName : str, optional
        Name of the PowerFactory results set (default is "All calculations") .ElmRes
    dist_index : int, optional
        Index of the disturbance in the data (default is 104)
    
    Returns:
    --------
    tuple: (rdP, generator_name_order, data)
        rdP: Power redistribution percentages
        generator_name_order: List of generator names in order
        data: Full DataFrame
    """
    file_path = path
    data = pd.read_csv(file_path, delimiter=';', decimal=',', skiprows=0) # Read the CSV file
    # Remove the second 1st row from data (parameter description)
    data = data.drop(index=0)

    # Convert Time column to numeric 
    data[f'{pfResultsName}'] = data[f'{pfResultsName}'].str.replace(',', '.').astype(float)

    # Convert all other columns to numeric
    for col in data.columns[1:]:
        data[col] = data[col].str.replace(',', '.').astype(float)

    # Find the index of the first row where GEN_OUT hits its minimum value
    min_index = dist_index - 2  # -2 because first and second row define DataObject and parameter description
    
    # Get values from all columns except the first column at the minimum index
    values_at_min_index = data.iloc[min_index - 1, 1:].values

    # Get values from all columns except the first column at the index before the minimum index
    values_before_min_index = data.iloc[min_index - 2, 1:].values

    substraction = np.array(values_at_min_index) - np.array(values_before_min_index)
    
    # Get the index of generator that was disturbed
    gen_out_index = data.columns.get_loc(GEN_OUT)
    substraction[gen_out_index - 1] = 0 # type: ignore

    # Create generator name order list (all generators from CSV)
    generator_name_order = data.columns[1:].tolist()
    
    # Filter to only available measurements if specified
    if available_measurements is not None:
        # Create boolean mask for available generators
        mask = [gen in available_measurements for gen in generator_name_order]
        
        # Filter substraction values and generator names
        substraction = substraction[mask]
        generator_name_order = [gen for gen in generator_name_order if gen in available_measurements]
    
    # print(sum(substraction)) #! Dev
    rdP = substraction / sum(substraction)
    # print("Sum of substraction:", sum(substraction)) #! Dev

    return rdP, generator_name_order, data

def obtain_rms_results(app: pf.Application, filesPath: str, pfResultsName: str = "All calculations") -> None:
    # Get the ElmRes object
    elmRes = app.GetCalcRelevantObjects(f"*{pfResultsName}.ElmRes")[0]
    # elmRes = app.GetCalcRelevantObjects("*I.ElmRes")[0]
    logger.debug(f"Results object: {elmRes.GetAttribute('loc_name')}")
    
    # Get all generators
    all_generators = app.GetCalcRelevantObjects("*.ElmSym", 1, 1, 1)
    logger.debug(f"All generators found: {len(all_generators)}")
    generators: List[DataObject] = []
    for gen in all_generators:
        if gen.GetAttribute("outserv") == 1:
            continue
        if gen.IsEnergized() != 1:
            continue
        generators.append(gen)
    logger.debug(f"Active generators: {[gen.GetAttribute('loc_name') for gen in generators]}")

    # Get all voltage sources (ElmVac) - these will be monitored but not tripped
    all_voltage_sources = app.GetCalcRelevantObjects("*.ElmVac", 1, 1, 1)
    logger.debug(f"All voltage sources found: {len(all_voltage_sources)}")
    voltage_sources: List[DataObject] = []
    for vac in all_voltage_sources:
        if vac.GetAttribute("outserv") == 1:
            continue
        if vac.IsEnergized() != 1:
            continue
        voltage_sources.append(vac)
    logger.debug(f"Active voltage sources: {[vac.GetAttribute('loc_name') for vac in voltage_sources]}")

    # Get all external grids (ElmXnet) - these will be monitored but not tripped
    all_external_grids = app.GetCalcRelevantObjects("*.ElmXnet", 1, 1, 1)
    logger.debug(f"All external grids found: {len(all_external_grids)}")
    external_grids: List[DataObject] = []
    for xnet in all_external_grids:
        if xnet.GetAttribute("outserv") == 1:
            continue
        if xnet.IsEnergized() != 1:
            continue
        external_grids.append(xnet)
    logger.debug(f"Active external grids: {[xnet.GetAttribute('loc_name') for xnet in external_grids]}")

    # Setup monitoring for generators
    for gen in generators:
        obj = elmRes.CreateObject("IntMon")
        obj.SetAttribute("loc_name", gen.GetAttribute("loc_name"))
        obj.SetAttribute("obj_id", gen)
        selected_variables = ["s:P1"]
        obj.SetAttribute("vars", selected_variables)

    # Setup monitoring for voltage sources
    for vac in voltage_sources:
        obj = elmRes.CreateObject("IntMon")
        obj.SetAttribute("loc_name", vac.GetAttribute("loc_name"))
        obj.SetAttribute("obj_id", vac)
        selected_variables = ["m:P:bus1"]  # Active power output
        obj.SetAttribute("vars", selected_variables)

    # Setup monitoring for external grids
    for xnet in external_grids:
        obj = elmRes.CreateObject("IntMon")
        obj.SetAttribute("loc_name", xnet.GetAttribute("loc_name"))
        obj.SetAttribute("obj_id", xnet)
        selected_variables = ["m:P:bus1"]  # Active power output
        obj.SetAttribute("vars", selected_variables)

    # DISABLE ALL CURRENT SIMULATION EVENTS (set as out of service)
    active = app.GetActiveStudyCase()
    eventFolder: DataObject = next((con for con in active.GetContents() if con.loc_name == "Simulation Events/Fault"))
    simulation_events: List[DataObject] = eventFolder.GetContents()
    for event in simulation_events:
        event.SetAttribute("outserv", 1)

    iteration = 0
    created_events = []
    previous_event = None
    files_names = []

    # Only create switch events for generators (not voltage sources)
    for gen in generators:
        # if (gen.GetAttribute("loc_name") != "Mar. Otok Gen1"):
        #     continue
        iteration += 1

        gen_name = gen.GetAttribute("loc_name")
        logger.info(f"Processing generator outage {iteration}: {gen_name}")

        # ====== 1. We define the SwitchEvent (only for generators)
        new_event = eventFolder.CreateObject("EvtSwitch")
        new_event.SetAttribute("loc_name", gen_name+"_izpad")
        new_event.SetAttribute("p_target", gen)
        new_event.SetAttribute("time", 0.1)
        created_events.append(new_event)

        # ====== 2. We run the simulation
        # Calculate initial conditions
        oInit = app.GetFromStudyCase('ComInc')  # Get initial condition calculation object
        timeUnit = oInit.GetAttributeUnit("dtgrd") # Set to calculate initial conditions
        if timeUnit == "s": # If not in ms
            oInit.SetAttribute("dtgrd", 0.001) # Set to 10 ms
        if timeUnit == "ms": # If not in ms
            oInit.SetAttribute("dtgrd", 1) # Set sim step to 10 ms
        # oInit.SetAttribute("dtgrd", 1) # Set sim step to 1 ms
        oInit.SetAttribute("tstart", 0) # Set sim start time to 0 ms
        oInit.Execute() # type: ignore

        # Run RMS-simulation
        oRms = app.GetFromStudyCase('ComSim')   # Get RMS-simulation object
        oRms.SetAttribute("tstop", 0.15)  # Set simulation time to 0.5 seconds
        oRms.Execute() # type: ignore

        # ====== 3. We delete the current event if it exists
        if previous_event is not None:
            deleted = previous_event.Delete()
            if deleted == 0:
                pass
            else:
                logger.warning(f"Failed to delete event: {previous_event.GetAttribute('loc_name')}")
        new_event.SetAttribute("outserv", 1)

        # ====== 4. We get the results
        comRes = app.GetFromStudyCase("ComRes")
        comRes.pResult = elmRes         # Set ElmRes object to export from # type: ignore
        comRes.iopt_exp = 6             # Set export to csv - 6 # type: ignore
        comRes.iopt_sep = 0             # Set use the system seperator # type: ignore
        comRes.iopt_honly = 0           # To export data and not only the head er # type:ignore
        comRes.iopt_csel = 1            # Set export to only selected variables # type: ignore
        comRes.numberPrecisionFixed = 8 # Set the number precision to 6 decimals # type: ignore
        comRes.col_Sep = ";"            # Set the column separator to ; # type: ignore
        comRes.dec_Sep = ","            # Set the decimal separator to , # type: ignore

        # Set File path and name 
        file_name = "results_izpad_" + gen_name + ".csv"
        files_names.append(file_name)
        results_folder = filesPath
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        file_path = os.path.join(results_folder, file_name)
        comRes.f_name = file_path # type: ignore

        resultObject = [None] # type: ignore
        elements = [elmRes] # type: ignore
        variable = ["b:tnow"] # type: ignore

        # Add generators to export
        for g in generators:
            resultObject.append(None)
            elements.append(g)
            variable.append("s:P1")
        
        # Add voltage sources to export
        for vac in voltage_sources:
            resultObject.append(None)
            elements.append(vac)
            variable.append("m:P:bus1")
        
        # Add external grids to export
        for xnet in external_grids:
            resultObject.append(None)
            elements.append(xnet)
            variable.append("m:P:bus1")
        
        # Set the selected variables
        comRes.resultobj = resultObject # Export selected # type: ignore
        comRes.element = elements # type: ignore
        comRes.variable = variable # type: ignore

        # Export the results
        comRes.Execute() # type: ignore 

        # Await for the file to be accessible
        while not os.path.exists(file_path):
            time.sleep(0.1)

        # ====== 5. We disable the SwitchEvent (set to out of service here as cannot delete, will be deleted in the next iteration)
        new_event.SetAttribute("outserv", 1)
        previous_event = new_event

        app.ClearOutputWindow()