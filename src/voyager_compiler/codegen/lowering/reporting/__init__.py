"""Latency / DRAM-traffic estimation and reporting for bufferized FX graphs.

Three separate stages:

  * ``estimate_schedule``  walk the graph -> per-node timing + DRAM traffic
  * ``compress_schedule``  collapse steady-state loop iterations
  * ``write_excel_report`` / ``write_perfetto``  editable workbook + trace

``report`` runs all three for the common case (call it after ``plan_memory``).
"""

import os

from .compress import compress_schedule
from .excel import write_excel_report
from .interpret import estimate_schedule
from .model import (
    CostParams,
    LoopSummary,
    OpInfo,
    ScheduleResult,
    TimingRecord,
)
from .perfetto import write_perfetto

__all__ = [
    "CostParams",
    "LoopSummary",
    "OpInfo",
    "ScheduleResult",
    "TimingRecord",
    "estimate_schedule",
    "compress_schedule",
    "write_excel_report",
    "write_perfetto",
    "report",
]


def report(
    model,
    dram_bandwidth: float,
    dram_access_latency: float,
    frequency: float,
    unroll: tuple[int, int],
    *,
    output_dir: str = ".",
    basename: str = "schedule",
    perfetto: bool = True,
    compress_events: bool = False,
) -> ScheduleResult:
    """Estimate, compress, and write the reports for a bufferized + memory-
    planned ``model``.

    ``dram_bandwidth`` (GB/s), ``dram_access_latency`` (ns) and ``frequency``
    (GHz) are physical units (cost.py converts to cycles).  Writes
    ``<basename>.xlsx`` (and, when ``perfetto``, ``<basename>.perfetto.json``)
    under ``output_dir`` and returns the (compressed) ``ScheduleResult``.
    ``compress_events`` writes only the compressed schedule to the Events sheet,
    so it stays small (and fast to write) for large trip counts.
    """
    result = estimate_schedule(
        model,
        dram_bandwidth=dram_bandwidth,
        dram_access_latency=dram_access_latency,
        frequency=frequency,
        unroll=unroll,
    )
    compress_schedule(result)

    os.makedirs(output_dir, exist_ok=True)
    write_excel_report(
        result,
        os.path.join(output_dir, f"{basename}.xlsx"),
        compress_events=compress_events,
    )
    if perfetto:
        write_perfetto(
            result, os.path.join(output_dir, f"{basename}.perfetto.json")
        )
    return result
