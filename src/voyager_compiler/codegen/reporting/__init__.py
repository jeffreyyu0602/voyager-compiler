"""Latency / DRAM-traffic estimation and reporting for bufferized FX graphs.

Two stages:

  * ``estimate_schedule``  walk the graph -> per-node timing + DRAM traffic,
    folding each loop's steady state as it goes
  * ``write_excel_report`` / ``write_perfetto`` / ``write_calibration_form``
    write the workbook, the trace, and the RTL calibration form

``report`` runs the common case (call it after ``plan_memory``).
"""

import os

from voyager_compiler.codegen.reporting.calibration import (
    Calibration,
    KernelSignature,
    Measurement,
    kernel_signatures,
    load_calibration,
    write_calibration_form,
)
from voyager_compiler.codegen.reporting.excel import write_excel_report
from voyager_compiler.codegen.reporting.interpret import estimate_schedule
from voyager_compiler.codegen.reporting.model import (
    LoopSkip,
    LoopStats,
    OpInfo,
    ScheduleResult,
    TimingRecord,
)
from voyager_compiler.codegen.reporting.perfetto import write_perfetto
from voyager_compiler.codegen.reporting.summary import (
    KernelRow,
    coverage,
    kernel_rows,
)

__all__ = [
    "Calibration",
    "KernelRow",
    "KernelSignature",
    "LoopSkip",
    "LoopStats",
    "Measurement",
    "OpInfo",
    "ScheduleResult",
    "TimingRecord",
    "coverage",
    "estimate_schedule",
    "kernel_rows",
    "kernel_signatures",
    "load_calibration",
    "report",
    "write_calibration_form",
    "write_excel_report",
    "write_perfetto",
]


def report(
    model,
    config,
    *,
    output_dir: str = ".",
    basename: str = "schedule",
    perfetto: bool = True,
    full_walk: bool = False,
    calibration: Calibration = None,
) -> ScheduleResult:
    """Estimate and write the reports for a bufferized + memory-planned
    ``model``.

    Args:
        model: The graph, after ``plan_memory``.
        config: The ``AcceleratorConfig`` (physical units; ``cost.py``
            converts to cycles).
        output_dir: Where ``<basename>.xlsx`` (and, when ``perfetto``,
            ``<basename>.perfetto.json``) are written.
        basename: The report files' stem.
        perfetto: Also write the trace.
        full_walk: Walk every loop iteration instead of folding the steady
            state.
        calibration: RTL-measured kernel cycles to price compute ops by.

    Returns:
        The ``ScheduleResult``.
    """
    result = estimate_schedule(
        model, config, full_walk=full_walk, calibration=calibration
    )
    os.makedirs(output_dir, exist_ok=True)
    write_excel_report(result, os.path.join(output_dir, f"{basename}.xlsx"))
    if perfetto:
        write_perfetto(
            result, os.path.join(output_dir, f"{basename}.perfetto.json")
        )
    return result
