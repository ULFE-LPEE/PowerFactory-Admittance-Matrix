"""Generator-outage RMS simulations and their monitored results."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import powerfactory as pf


class MonitorSpec(NamedTuple):
    """Elements and PowerFactory variables to record in one monitor group."""

    elements: list[pf.DataObject]
    variables: list[str]


def setup_monitoring(results: pf.DataObject, specs: list[MonitorSpec]) -> None:
    """Replace the result object's monitors with the requested variables."""
    for monitor in results.GetContents("*.IntMon", 0):
        monitor.Delete()
    for spec in specs:
        for element in spec.elements:
            for variable in spec.variables:
                results.AddVariable(element, variable)  # type: ignore


def extract_to_dataframe(results: pf.DataObject, app: pf.Application) -> pd.DataFrame:
    """Read an ElmRes as time rows and (object, variable, unit) columns."""
    vector = app.GetFromStudyCase("IntVec")
    if vector is None:
        raise RuntimeError("PowerFactory study case has no IntVec result buffer.")

    results.Load()  # type: ignore
    try:
        results.GetColumnValues(vector, -1)  # type: ignore
        times = list(vector.GetAttribute("V"))
        labels = []
        values = {}
        for column in range(results.GetNumberOfColumns()):  # type: ignore
            element = results.GetObject(column)  # type: ignore
            name = element.GetAttribute("loc_name") if element else f"column_{column}"
            labels.append((name, results.GetVariable(column), results.GetUnit(column)))  # type: ignore
            results.GetColumnValues(vector, column)  # type: ignore
            values[column] = list(vector.GetAttribute("V"))

        frame = pd.DataFrame(values, index=pd.Index(times, name="time_s"))
        frame.columns = pd.MultiIndex.from_tuples(labels, names=["object", "variable", "unit"])
        return frame
    finally:
        results.Release()  # type: ignore
        results.Clear()  # type: ignore


def simulate_generator_outage(
    app: pf.Application,
    generator_name: str,
    *,
    specs: list[MonitorSpec],
    event_time: float = 0.5,
    stop_time: float = 1.0,
    step_ms: float = 1.0,
    results_name: str = "Izvoz RoCof Martin",
) -> pd.DataFrame:
    """Trip one generator, run ComInc/ComSim, and return the full RMS table.

    The event is removed and the calculation reset even if the simulation
    fails. The PowerFactory study case supplies the dynamic models.
    """
    if not 0 < event_time < stop_time or step_ms <= 0:
        raise ValueError("Require 0 < event_time < stop_time and step_ms > 0.")

    generators = app.GetCalcRelevantObjects(f"*{generator_name}.ElmSym", 1, 1, 1)
    generator = next(
        (item for item in generators if item.GetAttribute("loc_name") == generator_name),
        None,
    )
    if generator is None:
        raise ValueError(f"PowerFactory generator not found: {generator_name}")

    results = next(
        (
            item
            for item in app.GetCalcRelevantObjects(f"*{results_name}.ElmRes")
            if item.GetAttribute("loc_name") == results_name
        ),
        None,
    )
    if results is None:
        raise RuntimeError(f"PowerFactory result object not found: {results_name}")

    study_case = app.GetActiveStudyCase()
    event_folder = next(
        (item for item in study_case.GetContents("*", 0) if item.GetAttribute("loc_name") == "Simulation Events/Fault"),
        None,
    )
    if event_folder is None:
        raise RuntimeError("Simulation Events/Fault folder is missing.")

    results.Clear()
    setup_monitoring(results, specs)

    event = event_folder.CreateObject("EvtSwitch")
    if event is None:
        raise RuntimeError("PowerFactory could not create the switching event.")
    try:
        event.SetAttribute("loc_name", f"rms_{generator_name}_{uuid4().hex[:8]}")
        event.SetAttribute("p_target", generator)
        event.SetAttribute("time", event_time)

        initial = app.GetFromStudyCase("ComInc")
        simulation = app.GetFromStudyCase("ComSim")
        if initial is None or simulation is None:
            raise RuntimeError("PowerFactory study case requires ComInc and ComSim.")

        unit = initial.GetAttributeUnit("dtgrd")
        if unit not in {"s", "ms"}:
            raise RuntimeError(f"Unsupported ComInc dtgrd unit: {unit}")
        initial.SetAttribute("dtgrd", step_ms / 1000 if unit == "s" else step_ms)
        initial.SetAttribute("tstart", 0)
        if initial.Execute() != 0:  # type: ignore
            raise RuntimeError("PowerFactory ComInc failed; inspect its output window.")

        simulation.SetAttribute("tstop", stop_time)
        if simulation.Execute() != 0:  # type: ignore
            raise RuntimeError("PowerFactory ComSim failed; inspect its output window.")
        return extract_to_dataframe(results, app)
    finally:
        app.ResetCalculation()
        event.Delete()


def event_power_changes(
    results: pd.DataFrame,
    *,
    event_time: float,
    variable: str,
    outaged_generator: str,
    element_column: str = "responding_generator",
    object_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Compare the event-time sample with the first later RMS sample.

    PowerFactory may record the event instant more than once. In that case,
    the last event-time row is the baseline. If no exact event row exists,
    use the samples immediately bracketing the event.
    """
    times = results.index.to_numpy(dtype=float)
    if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("RMS result times must be finite and nondecreasing.")
    if event_time < times[0] or event_time >= times[-1]:
        raise ValueError(f"Event at {event_time:g} s lies outside RMS range [{times[0]:g}, {times[-1]:g}) s.")

    event_rows = np.flatnonzero(np.isclose(times, event_time, rtol=0, atol=1e-8))
    before = int(event_rows[-1]) if len(event_rows) else int(np.searchsorted(times, event_time) - 1)
    after = before + 1
    if after >= len(times):
        raise ValueError(f"No sample after the event at {event_time:g} s.")

    columns = results.columns.get_level_values("variable") == variable
    if object_names is not None:
        columns &= results.columns.get_level_values("object").isin(object_names)
    selected = results.loc[:, columns]
    if selected.empty:
        raise ValueError(f"No monitored columns for variable {variable!r}.")

    power_before = selected.iloc[before].to_numpy(dtype=float)
    power_after = selected.iloc[after].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "outaged_generator": outaged_generator,
            element_column: selected.columns.get_level_values("object"),
            "sample_before_s": times[before],
            "sample_after_s": times[after],
            "power_before_mw": power_before,
            "power_after_mw": power_after,
            "change_mw": power_after - power_before,
        }
    )


def save_rms_results(
    results: pd.DataFrame,
    directory: Path,
    *,
    generator_name: str,
    run_number: int,
) -> Path:
    """Save one complete outage time series with its three-level column header."""
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.-]+", "_", generator_name, flags=re.UNICODE).strip("._")
    filename = f"{run_number:03d}_results_izpad_{safe_name or 'generator'}.csv"
    path = directory / filename
    results.to_csv(path)
    return path
