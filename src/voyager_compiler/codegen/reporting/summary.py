"""Per-kernel view of a schedule.

A kernel is one bufferized nest (see ``calibration``); ``kernel_rows``
folds a ``ScheduleResult``'s walked records, skips and loop stats into one
row per kernel -- its span, DRAM traffic, how much of its main loop was
walked versus folded, what one steady-state period runs and costs, and
whether the kernel is bound by compute or by DMA.  Both the workbook's
Kernels sheet and the calibration form are written from these rows.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from voyager_compiler.codegen.reporting.model import (
    LoopSkip,
    OpInfo,
    ScheduleResult,
)


@dataclass
class KernelRow:
    kernel: str
    anchor: str = ""
    group: str = ""  # calibration identity: meta['build_key']
    group_id: int = -1  # meta['build_group'], readable, per-compile
    num_shared_kernels: int = 0  # kernels sharing this build key
    start: int = 0
    end: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    trip_count: int = 0
    walked: int = 0
    skipped: int = 0
    period: int = 0
    repeats: int = 0
    period_cycles: int = 0
    compute_per_period: int = 0
    dram_per_period: int = 0
    ops_per_period: Dict[str, int] = field(default_factory=dict)
    analytic_per_iteration: float = 0.0
    calibration: str = ""
    measured_per_iteration: Optional[float] = None

    @property
    def span(self) -> int:
        return self.end - self.start

    @property
    def bound(self) -> str:
        if not self.period_cycles:
            return ""
        if self.compute_per_period >= self.dram_per_period:
            return "compute"
        return "dma"

    @property
    def ops_text(self) -> str:
        return " ".join(
            f"{k}x{v}" if v != 1 else k
            for k, v in sorted(self.ops_per_period.items())
        )


def _main_skip(skips: List[LoopSkip]) -> Optional[LoopSkip]:
    """The fold covering the most iterations: the kernel's steady state."""
    if not skips:
        return None
    return max(skips, key=lambda s: s.iterations)


def kernel_rows(result: ScheduleResult) -> List[KernelRow]:
    rows: Dict[str, KernelRow] = {}
    ops: Dict[str, OpInfo] = {op.key: op for op in result.ops}
    by_kernel: Dict[str, list] = {}
    for rec in result.records:
        by_kernel.setdefault(rec.kernel, []).append(rec)
    skips_by_kernel: Dict[str, list] = {}
    for skip in result.skips:
        skips_by_kernel.setdefault(skip.kernel, []).append(skip)
    loops_by_kernel: Dict[str, list] = {}
    for stats in result.loops:
        loops_by_kernel.setdefault(stats.kernel, []).append(stats)

    for kernel, recs in by_kernel.items():
        grp = result.kernel_groups.get(kernel)
        row = KernelRow(
            kernel=kernel,
            anchor=grp.anchor if grp else "",
            group=grp.key if grp else "",
            group_id=grp.group if grp else -1,
            num_shared_kernels=grp.num_shared_kernels if grp else 0,
            start=min(r.start for r in recs),
            end=max(r.end for r in recs),
        )
        skips = skips_by_kernel.get(kernel, [])
        for skip in skips:
            row.end = max(row.end, skip.end)
            row.read_bytes += skip.bytes.get("read", 0) * skip.repeats
            row.write_bytes += skip.bytes.get("write", 0) * skip.repeats
        for r in recs:
            if r.is_read:
                row.read_bytes += r.bytes
            elif r.bytes:
                row.write_bytes += r.bytes
        loops = loops_by_kernel.get(kernel, [])
        if loops:
            main = max(loops, key=lambda s: s.trip_count)
            row.trip_count = main.trip_count
            row.walked = main.walked
            row.skipped = main.skipped
            row.period = main.period
            row.period_cycles = main.shift
        skip = _main_skip(skips)
        if skip is not None:
            template = [result.records[e] for e in skip.template]
            row.repeats = sum(s.repeats for s in skips)
            row.compute_per_period = sum(
                r.end - r.start for r in template if r.kind == "compute"
            )
            row.dram_per_period = sum(
                r.end - r.start for r in template if r.resource == ("dram",)
            )
            counts = Counter(r.op_key for r in template if r.kind == "compute")
            iterations = skip.period
        else:
            loop_recs = [r for r in recs if r.loop_uid != -1]
            counts = Counter(r.op_key for r in loop_recs if r.kind == "compute")
            iterations = max(1, row.walked)
        row.ops_per_period = {
            k: v / iterations if v % iterations else v // iterations
            for k, v in counts.items()
        }
        analytic = sum(ops[k].analytic_cycles * v for k, v in counts.items())
        row.analytic_per_iteration = analytic / iterations
        measured = [
            ops[k] for k in counts if ops[k].measured_cycles is not None
        ]
        if measured:
            row.calibration = measured[0].calibration
            row.measured_per_iteration = (
                sum(ops[k].effective_cycles * v for k, v in counts.items())
                / iterations
            )
        rows[kernel] = row
    return sorted(rows.values(), key=lambda r: r.start)


def coverage(rows: List[KernelRow], total_latency: int) -> float:
    """The share of the makespan spent in calibrated kernels."""
    if not total_latency:
        return 0.0
    covered = sum(r.span for r in rows if r.calibration)
    return min(1.0, covered / total_latency)
