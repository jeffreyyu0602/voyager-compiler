"""The schedule workbook (xlsxwriter), values only.

Sheets: ``Summary`` (totals, the busy split, calibration coverage),
``Kernels`` (one row per bufferized nest: span, traffic, how much of its
loop was walked or folded, what a steady-state period runs and costs, its
calibration), ``Operations`` (per static compute node: shapes, dtypes,
ideal / analytic / effective cycles), ``Events`` (every walked event, with
a ``fold`` row where a steady state was skipped), and ``Architecture``
(the config the estimate used).  Nothing recomputes in Excel: changing a
knob or entering a measurement means re-running the estimator.
"""

from typing import List

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
        ("kernels", len(rows)),
        ("kernels_calibrated", len(calibrated)),
        ("calibration_coverage", coverage(rows, result.total_latency)),
    ]
    _table(wb, "Summary", ["Metric", "Value"], lines, {0: 24, 1: 16})


def _kernels(wb, rows: List[KernelRow]):
    headers = [
        "Kernel",
        "Anchor",
        "Signature",
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
            r.signature,
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
    _table(wb, "Kernels", headers, lines, {0: 40, 1: 28, 2: 14, 17: 40})


def _operations(wb, result: ScheduleResult):
    headers = [
        "Node",
        "Kernel",
        "Op type",
        "Units",
        "Input",
        "Weight",
        "Output",
        "Dtypes",
        "Macs / Ops",
        "Ideal cycles",
        "Utilization",
        "Analytic cycles",
        "Effective cycles",
        "Calibration",
    ]
    lines = [
        (
            op.key,
            op.kernel,
            op.op_type,
            "/".join(op.units),
            _shape(op, "input"),
            _shape(op, "weight"),
            _shape(op, "output"),
            op.detail.get("dtypes", ""),
            op.detail.get("macs", op.detail.get("ops", 0)),
            op.ideal_cycles,
            op.utilization,
            op.analytic_cycles,
            op.effective_cycles,
            op.calibration,
        )
        for op in result.ops
    ]
    _table(wb, "Operations", headers, lines, {0: 32, 1: 40, 4: 16, 5: 16})


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
    cost = result.cost
    lines = [
        ("frequency_ghz", float(cost.frequency)),
        ("dram_bandwidth_gbs", float(cost.dram_bandwidth)),
        ("dram_access_latency_ns", float(cost.dram_access_latency)),
        ("bytes_per_cycle", cost.dram_bandwidth / cost.frequency),
        ("pe_rows", int(cost.pe_array_size[0])),
        ("pe_cols", int(cost.pe_array_size[1])),
        ("vector_lanes", int(cost.vector_lanes)),
    ]
    _table(wb, "Architecture", ["Knob", "Value"], lines, {0: 24, 1: 14})


def write_excel_report(
    result: ScheduleResult, path: str, *, max_events: int = _XLS_ROW_MAX - 1
) -> str:
    """Write the workbook to ``path`` and return it.  The Events sheet is
    cut at ``max_events`` rows; every other sheet is complete."""
    # ``xlsxwriter`` is imported here so that importing voyager_compiler never
    # hard-requires it; only writing a workbook does.
    import xlsxwriter

    rows = kernel_rows(result)
    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    _summary(wb, result, rows)
    _kernels(wb, rows)
    _operations(wb, result)
    _events(wb, result, max_events)
    _architecture(wb, result)
    wb.close()
    return path
