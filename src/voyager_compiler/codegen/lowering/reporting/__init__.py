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
    *,
    tiler=None,
    bytes_per_cycle: float = None,
    setup_cycles: int = 1000,
    unroll=(16, 16),
    output_dir: str = ".",
    basename: str = "schedule",
    perfetto: bool = True,
) -> ScheduleResult:
    """Estimate, compress, and write the reports for a bufferized + memory-
    planned ``model``.

    ``bytes_per_cycle`` defaults to ``tiler.dram_bandwidth`` (already
    bytes/cycle) when a ``tiler`` is given.  Writes ``<basename>.xlsx`` (and,
    when ``perfetto``, ``<basename>.perfetto.json``) under ``output_dir`` and
    returns the (compressed) ``ScheduleResult``.
    """
    if bytes_per_cycle is None:
        if tiler is None:
            raise ValueError("pass tiler= or bytes_per_cycle=")
        bytes_per_cycle = tiler.dram_bandwidth
        unroll = getattr(tiler, "unroll", unroll)

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
