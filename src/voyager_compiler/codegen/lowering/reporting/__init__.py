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
    DEFAULT_SETUP_CYCLES,
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
    bytes_per_cycle: float,
    unroll: tuple[int, int],
    *,
    setup_cycles: int = None,
    output_dir: str = ".",
    basename: str = "schedule",
    perfetto: bool = True,
) -> ScheduleResult:
    """Estimate, compress, and write the reports for a bufferized + memory-
    planned ``model``.

    ``bytes_per_cycle`` is ``dram_bandwidth / frequency``; ``setup_cycles``
    falls back to ``DEFAULT_SETUP_CYCLES`` when ``None``.  Writes
    ``<basename>.xlsx`` (and, when ``perfetto``, ``<basename>.perfetto.json``)
    under ``output_dir`` and returns the (compressed) ``ScheduleResult``.
    """
    if setup_cycles is None:
        setup_cycles = DEFAULT_SETUP_CYCLES

    result = estimate_schedule(
        model,
        bytes_per_cycle=bytes_per_cycle,
        setup_cycles=setup_cycles,
        unroll=unroll,
    )
    compress_schedule(result)

    os.makedirs(output_dir, exist_ok=True)
    write_excel_report(result, os.path.join(output_dir, f"{basename}.xlsx"))
    if perfetto:
        write_perfetto(
            result, os.path.join(output_dir, f"{basename}.perfetto.json")
        )
    return result
