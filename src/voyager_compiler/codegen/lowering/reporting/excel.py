"""Live Excel workbook (xlsxwriter) with a linked Gantt chart.

The schedule *structure* — event order, resource lanes, and each event's
``start_deps`` — is fixed by the scheduler and written once.  Only the
*durations* are formulas, so editing the ``Architecture`` inputs (frequency,
bandwidth, latency, unroll) or a node's ``Utilization`` recomputes every event's
``Start``/``End`` and the Gantt bars **inside Excel**, no Python re-run needed.
This is sound because the dependency wiring is structural and therefore
invariant to duration edits.

Sheets: ``Architecture`` (editable knobs), ``Operations`` (per-node cycles),
``Events`` (every scheduled event, fully live), ``Gantt`` (stacked-bar
timeline), and ``Summary`` (totals + per-loop view + a representative period).
"""

from typing import Dict, List

from .model import ScheduleResult, TimingRecord

# ``xlsxwriter`` is imported lazily in ``write_excel_report`` so that importing
# voyager_compiler never hard-requires it; only generating a workbook does.
xl_rowcol_to_cell = None  # bound in write_excel_report

# Events sheet column layout.
C_EID = 0
C_NODE = 1
C_KIND = 2
C_RES = 3
C_ITER = 4
C_BYTES = 5
C_START = 6
C_LAT = 7
C_END = 8
C_COMPUTE = 9
C_DRAM = 10
C_READ = 11
C_WRITE = 12

# Operations sheet column layout (shapes precede the derived Work scalar).
O_NODE = 0
O_TYPE = 1
O_IN = 2
O_WEIGHT = 3
O_OUT = 4
O_WORK = 5
O_IDEAL = 6
O_UTIL = 7
O_EFF = 8

_HEADERS = [
    "EID",
    "Node",
    "Kind",
    "Resource",
    "Iter",
    "Bytes",
    "Start",
    "Latency",
    "End",
    "Compute",
    "DRAM",
    "Read B",
    "Write B",
]
_MAX_GANTT_ROWS = 400  # keep the chart legible; large graphs are truncated


def _architecture(wb, result: ScheduleResult):
    ws = wb.add_worksheet("Architecture")
    bold = wb.add_format({"bold": True})
    edit = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    ws.write(0, 0, "Architecture (editable)", bold)
    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 14)
    rows = [
        ("frequency", float(result.cost.frequency)),  # GHz
        ("dram_bandwidth", float(result.cost.dram_bandwidth)),  # GB/s
        ("dram_access_latency", float(result.cost.dram_access_latency)),  # ns
        ("ic_unroll", int(result.cost.unroll[0])),
        ("oc_unroll", int(result.cost.unroll[1])),
    ]
    for i, (name, val) in enumerate(rows):
        r = i + 1
        ws.write(r, 0, name)
        ws.write_number(r, 1, val, edit)
        wb.define_name(name, f"=Architecture!${'B'}${r + 1}")
    ws.write(7, 0, "bytes/cycle", bold)
    ws.write_formula(
        7,
        1,
        "=dram_bandwidth/frequency",
        None,
        result.cost.dram_bandwidth / result.cost.frequency,
    )


def _operations(wb, result: ScheduleResult) -> Dict[str, int]:
    """Write the per-node compute table; return ``op.key -> Effective-cell
    row`` so ``Events`` can link each compute latency to it."""
    ws = wb.add_worksheet("Operations")
    bold = wb.add_format({"bold": True})
    edit = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    headers = [
        "Node",
        "Op type",
        "Input",
        "Weight",
        "Output",
        "Macs / Ops",
        "Ideal cycles",
        "Utilization",
        "Effective cycles",
    ]
    for c, h in enumerate(headers):
        ws.write(0, c, h, bold)
    ws.set_column(O_NODE, O_NODE, 18)
    ws.set_column(O_IN, O_OUT, 18)
    ws.set_column(O_IDEAL, O_EFF, 14)

    op_row: Dict[str, int] = {}
    for i, op in enumerate(result.ops):
        r = i + 1
        work = op.detail.get("macs", op.detail.get("ops", 0))
        ws.write(r, O_NODE, op.key)
        ws.write(r, O_TYPE, op.op_type)
        # Operand shapes (the ``Work`` scalar's provenance).
        for col, dim in (
            (O_IN, "input"),
            (O_WEIGHT, "weight"),
            (O_OUT, "output"),
        ):
            shape = op.detail.get(dim)
            if shape is not None:
                ws.write(r, col, "x".join(str(int(d)) for d in shape))
        ws.write_number(r, O_WORK, work)
        work_cell = xl_rowcol_to_cell(r, O_WORK)
        if op.op_type in ("gemm", "conv"):
            ideal = f"=CEILING({work_cell}/(ic_unroll*oc_unroll),1)"
        else:
            ideal = f"=CEILING({work_cell}/oc_unroll,1)"
        ws.write_formula(r, O_IDEAL, ideal, None, op.ideal_cycles)
        ws.write_number(r, O_UTIL, 1.0, edit)  # editable utilization
        ideal_cell = xl_rowcol_to_cell(r, O_IDEAL)
        util_cell = xl_rowcol_to_cell(r, O_UTIL)
        ws.write_formula(
            r,
            O_EFF,
            f"=CEILING({ideal_cell}/{util_cell},1)",
            None,
            op.ideal_cycles,
        )
        op_row[op.key] = r
    return op_row


def _latency_formula(
    rec: TimingRecord, row: int, op_row: Dict[str, int]
) -> str:
    if rec.latency_kind == "compute":
        cell = xl_rowcol_to_cell(op_row[rec.latency_ref], O_EFF)
        return f"='Operations'!{cell}"
    if rec.latency_kind == "dram":
        b = xl_rowcol_to_cell(row, C_BYTES)
        return (
            f"=CEILING((dram_access_latency+{b}/dram_bandwidth)"
            f"*frequency,1)"
        )
    return "=0"


def _events(wb, result: ScheduleResult, op_row: Dict[str, int]):
    ws = wb.add_worksheet("Events")
    bold = wb.add_format({"bold": True})
    for c, h in enumerate(_HEADERS):
        ws.write(0, c, h, bold)
    ws.set_column(C_NODE, C_NODE, 16)
    ws.set_column(C_START, C_END, 10)

    for rec in result.records:
        r = rec.eid + 1  # header occupies row 0
        ws.write_number(r, C_EID, rec.eid)
        ws.write(r, C_NODE, rec.node_name)
        ws.write(r, C_KIND, rec.kind)
        ws.write(r, C_RES, "/".join(rec.resource))
        ws.write(r, C_ITER, str(rec.iteration_path))
        ws.write_number(r, C_BYTES, rec.bytes)

        lat = xl_rowcol_to_cell(r, C_LAT)
        start = xl_rowcol_to_cell(r, C_START)
        res = xl_rowcol_to_cell(r, C_RES)
        if rec.start_deps:
            ends = ",".join(
                xl_rowcol_to_cell(d + 1, C_END) for d in rec.start_deps
            )
            start_f = f"=MAX({ends})"
        else:
            start_f = "=0"
        ws.write_formula(r, C_START, start_f, None, rec.start)
        ws.write_formula(
            r,
            C_LAT,
            _latency_formula(rec, r, op_row),
            None,
            rec.end - rec.start,
        )
        ws.write_formula(r, C_END, f"={start}+{lat}", None, rec.end)
        ws.write_formula(
            r,
            C_COMPUTE,
            f'=IF(AND({res}<>"dram",{res}<>"control"),{lat},0)',
            None,
            (
                rec.end - rec.start
                if rec.resource[0] not in ("dram", "control")
                else 0
            ),
        )
        ws.write_formula(
            r,
            C_DRAM,
            f'=IF({res}="dram",{lat},0)',
            None,
            rec.end - rec.start if rec.resource[0] == "dram" else 0,
        )
        ws.write_number(r, C_READ, rec.bytes if rec.is_read else 0)
        ws.write_number(r, C_WRITE, 0 if rec.is_read else rec.bytes)
    return ws


def _gantt(wb, result: ScheduleResult, lo: int, hi: int, title: str, name: str):
    """A stacked-bar Gantt over Events rows ``[lo, hi)`` (a transparent Start
    offset + visible Compute / DRAM bars), linked so it recalculates live."""
    ws = wb.add_worksheet(name)
    chart = wb.add_chart({"type": "bar", "subtype": "stacked"})
    n = hi - lo
    first, last = lo + 1, hi  # 1-based data rows on the Events sheet

    def col(c):
        return [
            "Events",
            first,
            c,
            last,
            c,
        ]

    cats = col(C_NODE)
    chart.add_series(
        {
            "name": "Start",
            "categories": cats,
            "values": col(C_START),
            "fill": {"none": True},
            "border": {"none": True},
        }
    )
    chart.add_series(
        {
            "name": "Compute",
            "categories": cats,
            "values": col(C_COMPUTE),
            "fill": {"color": "#4472C4"},
        }
    )
    chart.add_series(
        {
            "name": "DRAM",
            "categories": cats,
            "values": col(C_DRAM),
            "fill": {"color": "#ED7D31"},
        }
    )
    chart.set_title({"name": title})
    chart.set_x_axis({"name": "cycles", "reverse": False})
    chart.set_y_axis({"reverse": True})  # first event at the top
    chart.set_legend({"position": "bottom"})
    chart.set_size({"x_scale": 2.0, "y_scale": max(1.0, n / 18.0)})
    ws.insert_chart(1, 1, chart)


def _summary(wb, result: ScheduleResult):
    ws = wb.add_worksheet("Summary")
    bold = wb.add_format({"bold": True})
    ws.set_column(0, 0, 20)
    ws.set_column(1, 5, 14)
    ws.write(0, 0, "Totals", bold)
    n = len(result.records)
    last = n  # last data row (1-based)
    ws.write(1, 0, "total_latency")
    ws.write_formula(
        1,
        1,
        f"=MAX(Events!{xl_rowcol_to_cell(1, C_END)}:"
        f"{xl_rowcol_to_cell(last, C_END)})",
        None,
        result.total_latency,
    )
    ws.write(2, 0, "dram_read_bytes")
    ws.write_formula(
        2,
        1,
        f"=SUM(Events!{xl_rowcol_to_cell(1, C_READ)}:"
        f"{xl_rowcol_to_cell(last, C_READ)})",
        None,
        result.dram_read_bytes,
    )
    ws.write(3, 0, "dram_write_bytes")
    ws.write_formula(
        3,
        1,
        f"=SUM(Events!{xl_rowcol_to_cell(1, C_WRITE)}:"
        f"{xl_rowcol_to_cell(last, C_WRITE)})",
        None,
        result.dram_write_bytes,
    )

    ws.write(5, 0, "Loops (compressed)", bold)
    head = [
        "Loop",
        "Trip",
        "Period iters events",
        "Repeat",
        "Period cyc",
        "Compact",
    ]
    for c, h in enumerate(head):
        ws.write(6, c, h, bold)
    for i, L in enumerate(result.loops):
        r = 7 + i
        ws.write(r, 0, L.loop_name)
        ws.write_number(r, 1, L.trip_count)
        ws.write_number(r, 2, len(L.period_eids))
        ws.write_number(r, 3, L.repeat_count)
        ws.write_number(r, 4, L.period_duration)
        ws.write(r, 5, f"prefix + period x {L.repeat_count} + suffix")


def write_excel_report(
    result: ScheduleResult, path: str, *, max_gantt_rows: int = _MAX_GANTT_ROWS
) -> str:
    """Write the live workbook to ``path`` and return it.  Run
    ``compress_schedule`` first to populate the per-loop summary / period views.
    """
    global xl_rowcol_to_cell
    import xlsxwriter
    from xlsxwriter.utility import xl_rowcol_to_cell as _xrc

    xl_rowcol_to_cell = _xrc

    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    _architecture(wb, result)
    op_row = _operations(wb, result)
    _events(wb, result, op_row)

    n = len(result.records)
    hi = min(n, max_gantt_rows)
    label = "whole graph" if hi == n else f"first {hi} of {n} events"
    _gantt(wb, result, 0, hi, f"Schedule ({label})", "Gantt")

    # Representative single period of the first compressed loop (compute/DRAM
    # overlap inside one steady-state iteration-group).
    for L in result.loops:
        if L.period_eids:
            lo, hp = min(L.period_eids), max(L.period_eids) + 1
            _gantt(
                wb,
                result,
                lo,
                hp,
                f"Representative period: {L.loop_name}",
                "Period",
            )
            break

    _summary(wb, result)
    wb.close()
    return path
