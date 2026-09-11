"""The schedule workbook (xlsxwriter), values only.

Sheets: ``Summary`` (totals, the busy split, calibration coverage),
``Kernels`` (one row per bufferized nest: span, traffic, how much of its
loop was walked or folded, what a steady-state period runs and costs, its
calibration), ``Operations`` (one row per node that costs
time -- every static compute node with its shapes, dtypes and ideal /
analytic / effective cycles, plus the DRAM materializations, tile transfers,
waits and launches -- each with how many times it ran and what it cost in
total), ``Events`` (every walked event, with
a ``fold`` row where a steady state was skipped), and ``Architecture``
(the config the estimate used).  Nothing recomputes in Excel: changing a
knob or entering a measurement means re-running the estimator.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List

from voyager_compiler.codegen.reporting.calibration import (
    FORM_SHEET,
    power_column,
    sheet_rows,
    calibration_sheet,
)
from voyager_compiler.codegen.reporting.model import ScheduleResult
from voyager_compiler.codegen.reporting.summary import (
    KernelRow,
    coverage,
    kernel_rows,
)

_XLS_ROW_MAX = 1_048_576  # Excel's hard row limit (row 0 holds the header)


def _table(wb, name: str, headers: List[str], rows, widths=None):
    ws = wb.add_worksheet(name)
    bold = wb.add_format({"bold": True})
    for c, h in enumerate(headers):
        ws.write(0, c, h, bold)
    for c, w in (widths or {}).items():
        ws.set_column(c, c, w)
    for r, row in enumerate(rows, start=1):
        for c, v in enumerate(row):
            if isinstance(v, bool):
                ws.write(r, c, str(v))
            elif isinstance(v, (int, float)):
                ws.write_number(r, c, v)
            elif v is None:
                pass
            else:
                ws.write(r, c, str(v))
    return ws


def _shape(op, key) -> str:
    dims = op.detail.get(key)
    if dims is None:
        return ""
    return "x".join(str(int(d)) for d in dims)


def _lane_cycles(result: ScheduleResult, lane: str) -> int:
    """Total cycles the Operations sheet's rows carry for one lane.

    ``compute`` and ``dram`` reproduce ``busy_compute`` / ``busy_dram``
    exactly, so the sheet is checkable against the Summary; ``control`` is
    the ``async_wait`` time, which overlaps the transfers it waits on and so
    belongs to neither.
    """
    total = 0
    for e in _rollup(result).values():
        res = e["resource"]
        if lane == "dram":
            hit = "dram" in res
        elif lane == "compute":
            hit = "mma" in res or "vector" in res
        else:
            hit = not ({"dram", "mma", "vector"} & set(res))
        if hit:
            total += e["cycles"]
    return total


def _summary(wb, result: ScheduleResult, rows: List[KernelRow]):
    overlap = result.busy_compute + result.busy_dram - result.busy_any
    walked = sum(s.walked for s in result.loops)
    skipped = sum(s.skipped for s in result.loops)
    calibrated = [r for r in rows if r.calibration]
    lines = [
        ("total_latency", result.total_latency),
        ("compute_only_cycles", result.busy_compute - overlap),
        ("dram_only_cycles", result.busy_dram - overlap),
        ("overlap_cycles", overlap),
        ("stall_cycles", result.total_latency - result.busy_any),
        ("dram_read_bytes", result.dram_read_bytes),
        ("dram_write_bytes", result.dram_write_bytes),
        ("dram_weight_bytes", result.dram_weight_bytes),
        ("dram_activation_bytes", result.dram_activation_bytes),
        ("dram_kv_bytes", result.dram_kv_bytes),
        ("iterations_walked", walked),
        ("iterations_folded", skipped),
        ("events_walked", len(result.records)),
        # The Operations sheet, summed by lane.  The compute and DRAM rows
        # each close exactly on the busy figures above; the wait rows are the
        # program clock blocked on a semaphore, which runs concurrently with
        # the transfer it waits on and must not be added to either.
        ("operations_compute_cycles", _lane_cycles(result, "compute")),
        ("operations_dram_cycles", _lane_cycles(result, "dram")),
        ("operations_wait_cycles", _lane_cycles(result, "control")),
        ("kernels", len(rows)),
        ("kernels_calibrated", len(calibrated)),
        ("calibration_coverage", coverage(rows, result.total_latency)),
    ]
    _table(wb, "Summary", ["Metric", "Value"], lines, {0: 24, 1: 16})


def _kernels(wb, rows: List[KernelRow]):
    headers = [
        "Kernel",
        "Anchor",
        "Key",
        "Group",
        "Start",
        "End",
        "Span",
        "Read B",
        "Write B",
        "Trip",
        "Walked",
        "Folded",
        "Period iters",
        "Periods folded",
        "Cycles/period",
        "Compute/period",
        "DMA/period",
        "Bound",
        "Ops/iteration",
        "Analytic cyc/iter",
        "Calibration",
        "Measured cyc/iter",
    ]
    lines = [
        (
            r.kernel,
            r.anchor,
            r.group,
            f"g{r.group_id}" if r.group_id >= 0 else "",
            r.start,
            r.end,
            r.span,
            r.read_bytes,
            r.write_bytes,
            r.trip_count,
            r.walked,
            r.skipped,
            r.period,
            r.repeats,
            r.period_cycles,
            r.compute_per_period,
            r.dram_per_period,
            r.bound,
            r.ops_text,
            r.analytic_per_iteration,
            r.calibration,
            r.measured_per_iteration,
        )
        for r in rows
    ]
    _table(wb, "Kernels", headers, lines, {0: 40, 1: 28, 2: 14, 18: 40})


def _rollup(result: ScheduleResult) -> Dict[str, dict]:
    """Per-node totals over the whole schedule: how many times each node ran
    and what it cost.

    A folded steady state contributes its template period once as a walked
    record and ``repeats`` more times as the fold, which is the same
    accounting ``summary.kernel_rows`` uses for bytes.  Compute events are
    keyed by ``op_key`` so they join the ``OpInfo`` rows; everything else by
    node name -- qualified by kind, because a kernel's launch record carries
    ``node_name = kernel`` and would otherwise collide with the anchor op of
    the same name, inflating that op's invocation count by one.
    """
    agg: Dict[str, dict] = {}

    def add(rec, times: int) -> None:
        key = (
            rec.op_key
            if rec.kind == "compute" and rec.op_key
            else f"{rec.kind}:{rec.node_name}"
        )
        e = agg.get(key)
        if e is None:
            e = agg[key] = {
                "node": rec.node_name,
                "kernel": rec.kernel,
                "kind": rec.kind,
                "resource": rec.resource,
                "category": rec.category,
                "sync": rec.sync,
                "first": rec.start,
                "count": 0,
                "cycles": 0,
                "read": 0,
                "write": 0,
            }
        e["first"] = min(e["first"], rec.start)
        e["count"] += times
        e["cycles"] += (rec.end - rec.start) * times
        if rec.bytes:
            if rec.is_read:
                e["read"] += rec.bytes * times
            else:
                e["write"] += rec.bytes * times

    for rec in result.records:
        add(rec, 1)
    for skip in result.skips:
        for eid in skip.template:
            add(result.records[eid], skip.repeats)
    return agg


# How a non-compute event is labelled in the Operations sheet's "Op type".
_KIND_LABEL = {
    "load": "materialize",  # sync: a pad / permute / cat / expand round trip
    "store": "materialize",
    "async_wait": "wait",
    "launch": "launch",
}


def _operations(wb, result: ScheduleResult, rows: List[KernelRow], agg):
    """One row per node that costs time.

    Compute nodes carry the cost model's shapes, utilization and cycles; the
    DRAM rows carry bytes, and their "utilization" is the fraction of peak
    bandwidth the transfer sustained (the fixed access latency is what pulls
    it below 1).  "Kernel read/write B" attributes traffic at the kernel
    level, which is the honest granularity: one transfer fills a buffer that
    several ops read and that is reused across tiles, so splitting its bytes
    between them would be a choice, not a fact.
    """
    headers = [
        "Node",
        "Kernel",
        "Op type",
        "Units",
        "Input",
        "Weight",
        "Output",
        "Dtypes",
        "Category",
        "Macs / Ops",
        "Ideal cycles",
        "Utilization",
        "Analytic cycles",
        "Effective cycles",
        "Calibration",
        "Invocations",
        "Total cycles",
        "Read B",
        "Write B",
        "Kernel read B",
        "Kernel write B",
    ]
    agg = dict(agg)  # popped below; the caller's copy is reused elsewhere
    per_kernel = {r.kernel: (r.read_bytes, r.write_bytes) for r in rows}
    bpc = result.cost.dram_bandwidth / result.cost.frequency
    lines = []  # (first start, row) -- sorted into execution order below

    for op in result.ops:
        e = agg.pop(op.key, None)
        k_read, k_write = per_kernel.get(op.kernel, (0, 0))
        lines.append(
            (
                e["first"] if e else 0,
                (
                    op.key,
                    op.kernel,
                    op.op_type,
                    "/".join(op.units),
                    _shape(op, "input"),
                    _shape(op, "weight"),
                    _shape(op, "output"),
                    op.detail.get("dtypes", ""),
                    "",
                    op.detail.get("macs", op.detail.get("ops", 0)),
                    op.ideal_cycles,
                    op.utilization,
                    op.analytic_cycles,
                    op.effective_cycles,
                    op.calibration,
                    e["count"] if e else 0,
                    e["cycles"] if e else 0,
                    0,
                    0,
                    k_read,
                    k_write,
                ),
            )
        )

    # Everything else the schedule spends time on: the whole-tensor DRAM
    # materializations the accelerator does not compute (pad / permute / cat /
    # expand, synchronous), the tile transfers, the waits, the launches.
    for key, e in agg.items():
        n = max(1, e["count"])
        n_bytes = e["read"] + e["write"]
        label = _KIND_LABEL.get(e["kind"], e["kind"])
        if label == "materialize" and not e["sync"]:
            label = "dma"
        ideal = math.ceil(n_bytes / bpc) if n_bytes and bpc else 0
        k_read, k_write = per_kernel.get(e["kernel"], (0, 0))
        lines.append(
            (
                e["first"],
                (
                    e["node"],
                    e["kernel"],
                    label,
                    "/".join(e["resource"]),
                    "",
                    "",
                    "",
                    "",
                    e["category"],
                    None,  # Macs / Ops is compute-only; bytes are in Read/Write B
                    ideal,
                    (ideal / e["cycles"]) if e["cycles"] and ideal else None,
                    round(e["cycles"] / n),
                    round(e["cycles"] / n),
                    "",
                    e["count"],
                    e["cycles"],
                    e["read"],
                    e["write"],
                    k_read,
                    k_write,
                ),
            )
        )

    lines = [row for _, row in sorted(lines, key=lambda t: t[0])]
    _table(wb, "Operations", headers, lines, {0: 32, 1: 40, 4: 16, 5: 16})


# --------------------------------------------------------------------------
# The per-operation table
#
# One line per distinct operation, each standing for every instance of it in
# the model.  Kernels collapse by build group -- bufferization built them
# from one cached nest, so they are the same operation -- and the resulting
# rows then merge by display name, which is presentation: the three RMSNorm
# positions read better as one line than as three.  Count says how many
# instances a row stands for, so latency and DRAM traffic still sum to the
# model's totals.
#
# Power comes from the Calibration sheet and corrects nothing -- the compiler
# has no power model.  It exists so a latency can become an energy.
# --------------------------------------------------------------------------

_LAYER_RE = re.compile(r"layers_\d+_")


def base_name(kernel: str) -> str:
    """A kernel name with its layer index, ``model_`` prefixes and FX
    instance counter stripped -- used to tell apart the several operations a
    single build group can serve, not to rename anything."""
    n = _LAYER_RE.sub("", kernel)
    n = re.sub(r"^(model_)+", "", n)
    n = re.sub(r"_\d+(?=_fused$)", "", n)
    n = re.sub(r"_\d+$", "", n)
    return n


@dataclass
class OperationRow:
    name: str
    units: str
    count: int = 0
    cycles: int = 0
    bytes: int = 0
    ideal: float = 0.0
    power_cycles: float = 0.0  # sum of power * span, for the weighted mean
    first: int = 0
    groups: List[str] = field(default_factory=list)

    @property
    def power(self) -> float:
        return self.power_cycles / self.cycles if self.cycles else 0.0

    @property
    def utilization(self) -> float:
        return self.ideal / self.cycles if self.cycles else 0.0

    def seconds(self, frequency_ghz: float) -> float:
        return self.cycles / (frequency_ghz * 1e9)

    def compute_energy(self, frequency_ghz: float) -> float:
        return self.power * self.seconds(frequency_ghz)

    def dram_energy(self, joules_per_byte: float) -> float:
        return self.bytes * joules_per_byte

    def energy(self, frequency_ghz: float, joules_per_byte: float) -> float:
        return self.compute_energy(frequency_ghz) + self.dram_energy(
            joules_per_byte
        )


def _per_kernel(result, agg) -> Dict[str, dict]:
    """Each kernel's ideal cycles and which engines it occupied, from the
    Operations roll-up."""
    out: Dict[str, dict] = {}
    by_key = {op.key: op for op in result.ops}
    for key, e in agg.items():
        k = e["kernel"]
        entry = out.setdefault(k, {"ideal": 0.0, "mma": False, "vec": False})
        op = by_key.get(key)
        if op is not None and op.op_type in ("gemm", "conv", "vector"):
            entry["ideal"] += op.ideal_cycles * e["count"]
            entry["mma"] |= "mma" in op.units
            entry["vec"] |= "vector" in op.units
    return out


def operation_rows(result, rows, agg, calibration=None) -> List[OperationRow]:
    """Collapse ``rows`` (one per kernel) into the per-operation table.

    Args:
        result: The ``ScheduleResult``.
        rows: ``kernel_rows(result)``.
        agg: ``excel._rollup(result)`` -- per-node invocations and cycles.
        calibration: Supplies each group's measured power, if any.
    """
    stats = _per_kernel(result, agg)
    bpc = result.cost.dram_bandwidth / result.cost.frequency
    groups: Dict[str, OperationRow] = {}
    order: List[str] = []

    for row in sorted(rows, key=lambda r: r.start):
        st = stats.get(row.kernel)
        if st is None:
            continue
        # Only nodes ``_bufferize_key`` could not sign reach here.  Bytes
        # over a span is a weak key -- unrelated kernels moving the same
        # bytes in the same time merge -- and stands in for nothing better.
        key = row.group or ("u", row.read_bytes, row.write_bytes, row.span)
        entry = groups.get(key)
        if entry is None:
            units = []
            if st["mma"]:
                units.append("MU")
            if st["vec"]:
                units.append("VU")
            entry = groups[key] = OperationRow(
                name="",  # set below, once every member is known
                units=",".join(units) or "DMA",
                first=row.start,
                groups=[key],
            )
            entry._names = []  # noqa: SLF001 -- distinct ops in this group
            order.append(key)
        if row.group:
            label = row.kernel
            if label not in entry._names:
                entry._names.append(label)
        elif not entry._names:
            # An ungrouped row stands for many identical kernels; Count says
            # how many, and the Kernels sheet names them all.
            entry._names.append(row.kernel)
        n_bytes = row.read_bytes + row.write_bytes
        entry.count += 1
        entry.cycles += row.span
        entry.bytes += n_bytes
        entry.ideal += (
            st["ideal"] if (st["mma"] or st["vec"]) else n_bytes / bpc
        )
        if calibration is not None:
            entry.power_cycles += calibration.power(row.group) * row.span

    table = [groups[k] for k in order]
    # One nest can serve several operations -- q, k and v are one build group
    # in some models -- and the row stands for all of them, so it names all of
    # them, as the graph names them.  Renaming to conventional layer names and
    # merging rows that share one are presentation, and belong to whatever
    # reads this sheet, not to the compiler.
    for entry in table:
        names = entry._names
        entry.name = ", ".join(names[:4]) + (
            f" (+{len(names) - 4} more)" if len(names) > 4 else ""
        )
    return table


def totals(table: List[OperationRow], total_latency: int) -> OperationRow:
    """The table's TOTAL line.  Utilization is against the model's makespan,
    not the row sum, because rows overlap."""
    out = OperationRow(
        name=f"TOTAL ({len(table)} distinct operations)", units=""
    )
    for row in table:
        out.count += row.count
        out.cycles += row.cycles
        out.bytes += row.bytes
        out.ideal += row.ideal
        out.power_cycles += row.power_cycles
    if total_latency:
        out.ideal = out.ideal / total_latency * out.cycles if out.cycles else 0
    return out


def _calibration(wb, rows, calibration):
    """The Calibration sheet.  Written by ``calibration.py``, which owns the
    column layout the loader reads back."""
    return calibration_sheet(wb, rows, calibration)


def _operation_table(wb, result: ScheduleResult, rows, agg, calibration):
    """The Operation Summary sheet: one line per distinct operation, with
    latency, DRAM traffic, utilization and energy.

    Power and the energy columns are formulas: power reads the Calibration
    sheet's cell for the row's build group, so editing a measured watt there
    updates the energy here without recompiling.  Latency, traffic and
    utilization are values -- they come from the scheduler, not arithmetic a
    spreadsheet could redo.
    """
    table = operation_rows(result, rows, agg, calibration)
    freq = result.cost.frequency
    jpb = result.cost.dram_energy_per_byte
    ws = wb.add_worksheet("Operation Summary")
    bold = wb.add_format({"bold": True, "align": "center", "border": 1})
    cell = wb.add_format({"border": 1})
    ctr = wb.add_format({"border": 1, "align": "center"})
    f2 = wb.add_format({"num_format": "0.00", "border": 1})
    f3 = wb.add_format({"num_format": "0.000", "border": 1})
    tb = wb.add_format({"bold": True, "border": 1, "top": 2})
    t2 = wb.add_format(
        {"bold": True, "num_format": "0.00", "border": 1, "top": 2}
    )
    headers = [
        "Kernel",
        "Active Units",
        "Count",
        "Latency (us)",
        "DRAM Traffic (MB)",
        "Util.",
        "Power (W)",
        "Compute Energy (mJ)",
        "DRAM Energy (mJ)",
        "Energy (mJ)",
    ]
    for c, h in enumerate(headers):
        ws.write(0, c, h, bold)
    ws.set_column(0, 0, 30)
    ws.set_column(1, 1, 14)
    ws.set_column(2, 9, 17)
    ws.freeze_panes(1, 0)

    # Where each build group's power lives on the Calibration sheet.
    col = power_column()
    power_cell = {
        row.group: f"'{FORM_SHEET}'!${col}${r}"
        for r, row in enumerate(sheet_rows(rows), start=2)
        if row.group
    }
    for r, row in enumerate(table, start=1):
        one = r + 1
        refs = [power_cell[g] for g in row.groups if g in power_cell]
        ws.write(r, 0, row.name, cell)
        ws.write(r, 1, row.units, ctr)
        ws.write_number(r, 2, row.count, ctr)
        ws.write_number(r, 3, row.cycles / freq / 1000.0, f2)
        ws.write_number(r, 4, row.bytes / 1e6, f3)
        ws.write_number(r, 5, row.utilization, f2)
        if refs:
            # Several groups on one display line: their span-weighted mean is
            # what the row draws, and with equal spans that is the average.
            expr = (
                refs[0]
                if len(refs) == 1
                else ("AVERAGE(" + ",".join(refs) + ")")
            )
            ws.write_formula(r, 6, f"={expr}", f3, row.power)
        else:
            # Blank, not a dash: the energy columns multiply this cell, and
            # Excel returns #VALUE! for text.  A DMA-only operation has no
            # build key and so no measured power.
            ws.write_blank(r, 6, None, f3)
        ws.write_formula(
            r,
            7,
            f"=D{one}/1000000*G{one}*1000",
            f2,
            row.compute_energy(freq) * 1e3,
        )
        ws.write_formula(
            r,
            8,
            f"=E{one}*1000000*{jpb}*1000",
            f2,
            row.dram_energy(jpb) * 1e3,
        )
        ws.write_formula(
            r, 9, f"=H{one}+I{one}", f2, row.energy(freq, jpb) * 1e3
        )
    r = len(table) + 1
    one = r + 1
    tot = totals(table, result.total_latency)
    ws.write(r, 0, tot.name, tb)
    ws.write(r, 1, "", tb)
    ws.write_number(r, 2, tot.count, tb)
    ws.write_number(r, 3, tot.cycles / freq / 1000.0, t2)
    ws.write_number(r, 4, tot.bytes / 1e6, t2)
    ws.write_number(r, 5, tot.utilization, t2)
    ws.write(r, 6, "", tb)
    for c in (7, 8, 9):
        col = chr(ord("A") + c)
        ws.write_formula(r, c, f"=SUM({col}2:{col}{r})", t2)


def _events(wb, result: ScheduleResult, max_rows: int):
    headers = [
        "EID",
        "Node",
        "Kernel",
        "Kind",
        "Resource",
        "Iter",
        "Bytes",
        "Category",
        "Start",
        "Latency",
        "End",
    ]
    folds = {}
    for s in result.skips:
        folds.setdefault(s.after_eid, []).append(s)

    def lines():
        n = 0
        for rec in result.records:
            if n >= max_rows:
                return
            yield (
                rec.eid,
                rec.node_name,
                rec.kernel,
                rec.kind,
                "/".join(rec.resource),
                str(rec.iteration_path),
                rec.bytes,
                rec.category,
                rec.start,
                rec.end - rec.start,
                rec.end,
            )
            n += 1
            for s in folds.get(rec.eid, ()):
                yield (
                    None,
                    s.kernel,
                    s.kernel,
                    "fold",
                    "",
                    f"{s.first_step}..{s.first_step + s.iterations - 1} "
                    f"({s.repeats} x {s.period} iters)",
                    (s.bytes.get("read", 0) + s.bytes.get("write", 0))
                    * s.repeats,
                    "",
                    s.start,
                    s.end - s.start,
                    s.end,
                )
                n += 1

    _table(wb, "Events", headers, lines(), {1: 28, 2: 40, 5: 24})


def _architecture(wb, result: ScheduleResult):
    """Every knob of the ``AcceleratorConfig`` this report was produced with,
    plus the derived quantities the cost model actually uses, so a workbook
    says on its own face which machine it describes."""
    cost = result.cost

    def maybe(value, fn=lambda v: v):
        return "" if value is None else fn(value)

    lines = [
        ("-- compute --", ""),
        ("pe_rows", int(cost.pe_array_size[0])),
        ("pe_cols", int(cost.pe_array_size[1])),
        (
            "pe_macs_per_cycle",
            int(cost.pe_array_size[0]) * int(cost.pe_array_size[1]),
        ),
        ("vector_unit_width", maybe(cost.vector_unit_width, int)),
        ("vector_lanes", int(cost.vector_lanes)),
        ("frequency_ghz", float(cost.frequency)),
        ("-- L1 systolic buffers (elements) --", ""),
        ("input_buffer_size", maybe(cost.input_buffer_size, int)),
        ("weight_buffer_size", maybe(cost.weight_buffer_size, int)),
        ("accum_buffer_size", maybe(cost.accum_buffer_size, int)),
        (
            "double_buffered_accum_buffer",
            bool(cost.double_buffered_accum_buffer),
        ),
        ("-- L2 scratchpad --", ""),
        ("scratchpad_size", maybe(cost.scratchpad_size, int)),
        ("scratchpad_offset", int(cost.scratchpad_offset)),
        ("usable_scratchpad_size", maybe(cost.usable_scratchpad_size, int)),
        ("num_banks", maybe(cost.num_banks, int)),
        ("usable_banks", maybe(cost.usable_banks, int)),
        ("bank_size", maybe(cost.bank_size, int)),
        ("bank_width", maybe(cost.bank_width, int)),
        ("double_buffered_l2", bool(cost.double_buffered_l2)),
        ("num_slots", int(cost.num_slots)),
        ("-- L3 DRAM --", ""),
        ("dram_size_gb", maybe(cost.dram_size, float)),
        ("dram_bandwidth_gbs", maybe(cost.dram_bandwidth, float)),
        ("dram_access_latency_ns", maybe(cost.dram_access_latency, float)),
        ("dram_energy_per_bit_pj", float(cost.dram_energy_per_bit)),
        ("-- derived --", ""),
        ("bytes_per_cycle", cost.bytes_per_cycle),
        ("access_latency_cycles", cost.access_latency_cycles),
        ("dram_energy_per_byte_j", cost.dram_energy_per_byte),
        ("ns_per_cycle", 1.0 / float(cost.frequency)),
    ]
    _table(wb, "Architecture", ["Knob", "Value"], lines, {0: 34, 1: 18})


def write_excel_report(
    result: ScheduleResult,
    path: str,
    *,
    max_events: int = _XLS_ROW_MAX - 1,
    calibration=None,
) -> str:
    """Write the workbook to ``path`` and return it.  The Events sheet is
    cut at ``max_events`` rows; every other sheet is complete.

    ``calibration`` supplies the measured power the Operation Table's energy
    columns use; without it those columns are zero and the latency, traffic
    and utilization columns are unaffected."""
    # ``xlsxwriter`` is imported here so that importing voyager_compiler never
    # hard-requires it; only writing a workbook does.
    import xlsxwriter

    rows = kernel_rows(result)
    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    agg = _rollup(result)
    _summary(wb, result, rows)
    _architecture(wb, result)
    _operation_table(wb, result, rows, agg, calibration)
    _kernels(wb, rows)
    _operations(wb, result, rows, agg)
    _events(wb, result, max_events)
    _calibration(wb, rows, calibration)
    wb.close()
    return path
