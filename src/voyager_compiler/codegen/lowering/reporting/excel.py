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
        if "mma" in op.units:
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


def _start_term(dep, result, eid_to_row, ii_names):
    """The ``Start`` MAX-term for one dependency: its ``End`` cell when written,
    else the congruent last-written event's ``End`` shifted by ``skip`` live
    initiation intervals.  A truncated-away dep falls back to a cached literal.
    """
    if dep in eid_to_row:
        return xl_rowcol_to_cell(eid_to_row[dep], C_END)
    remap = result.dep_remap
    if dep in remap:
        written, skip, uid = remap[dep]
        if written in eid_to_row:
            end = xl_rowcol_to_cell(eid_to_row[written], C_END)
            return f"({end}+{skip}*{ii_names[uid]})"
    rec = result.record_by_eid(dep)
    return str(rec.end if rec else 0)


def _events(
    wb,
    result: ScheduleResult,
    op_row: Dict[str, int],
    written_recs: List[TimingRecord],
    eid_to_row: Dict[int, int],
    ii_names: Dict[int, str],
):
    ws = wb.add_worksheet("Events")
    bold = wb.add_format({"bold": True})
    for c, h in enumerate(_HEADERS):
        ws.write(0, c, h, bold)
    ws.set_column(C_NODE, C_NODE, 16)
    ws.set_column(C_START, C_END, 10)

    for rec in written_recs:
        r = eid_to_row[rec.eid]
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
            terms = ",".join(
                _start_term(d, result, eid_to_row, ii_names)
                for d in rec.start_deps
            )
            start_f = f"=MAX({terms})"
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


def _gantt(wb, first: int, last: int, title: str, name: str):
    """A stacked-bar Gantt over Events data rows ``[first, last]`` (1-based, a
    transparent Start offset + visible Compute / DRAM bars), linked so it
    recalculates live."""
    ws = wb.add_worksheet(name)
    chart = wb.add_chart({"type": "bar", "subtype": "stacked"})
    n = last - first + 1

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


def _summary(wb, result: ScheduleResult, last: int, compressed: bool):
    ws = wb.add_worksheet("Summary")
    bold = wb.add_format({"bold": True})
    ws.set_column(0, 0, 20)
    ws.set_column(1, 5, 14)
    ws.write(0, 0, "Totals", bold)
    ws.write(1, 0, "total_latency")
    ws.write_formula(
        1,
        1,
        f"=MAX(Events!{xl_rowcol_to_cell(1, C_END)}:"
        f"{xl_rowcol_to_cell(last, C_END)})",
        None,
        result.total_latency,
    )

    # Bytes depend on no editable knob; when compressed a live SUM would also
    # miss the elided rows, so write the exact totals as constants.
    def _bytes(row, name, col, total):
        ws.write(row, 0, name)
        if compressed:
            ws.write_number(row, 1, total)
        else:
            ws.write_formula(
                row,
                1,
                f"=SUM(Events!{xl_rowcol_to_cell(1, col)}:"
                f"{xl_rowcol_to_cell(last, col)})",
                None,
                total,
            )

    _bytes(2, "dram_read_bytes", C_READ, result.dram_read_bytes)
    _bytes(3, "dram_write_bytes", C_WRITE, result.dram_write_bytes)

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


_XLS_ROW_MAX = 1_048_576  # Excel's hard row limit (row 0 holds the header)


def _written_records(result: ScheduleResult) -> List[TimingRecord]:
    """The records the compressed Events sheet emits, in eid order: every
    top-level record, plus each loop's prefix, written periods and suffix (or
    every record of an uncompressed loop)."""
    unc = {L.loop_uid for L in result.loops if L.uncompressed}
    keep = set()
    for L in result.loops:
        if L.uncompressed:
            continue
        keep.update(L.prefix_eids)
        keep.update(L.suffix_eids)
        for period in L.written_periods:
            keep.update(period)
    return [
        r
        for r in result.records
        if r.loop_uid == -1 or r.loop_uid in unc or r.eid in keep
    ]


def write_excel_report(
    result: ScheduleResult,
    path: str,
    *,
    max_gantt_rows: int = _MAX_GANTT_ROWS,
    compress_events: bool = False,
) -> str:
    """Write the live workbook to ``path`` and return it.  Run
    ``compress_schedule`` first to populate the per-loop summary / period views.

    With ``compress_events`` the Events sheet emits only the compressed
    schedule (prefix + K representative periods + suffix per loop), staying live
    via per-loop initiation-interval cells -- so its size no longer grows with
    the loop trip counts.
    """
    global xl_rowcol_to_cell
    import xlsxwriter
    from xlsxwriter.utility import xl_rowcol_to_cell as _xrc

    xl_rowcol_to_cell = _xrc

    if compress_events:
        written_recs = _written_records(result)
    else:
        written_recs = list(result.records)
    if len(written_recs) > _XLS_ROW_MAX - 1:
        written_recs = written_recs[: _XLS_ROW_MAX - 1]
    eid_to_row = {rec.eid: i + 1 for i, rec in enumerate(written_recs)}

    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    _architecture(wb, result)
    op_row = _operations(wb, result)

    ii_names: Dict[int, str] = {}
    if compress_events:
        for i, L in enumerate(result.loops):
            if L.uncompressed:
                continue
            a1, a2 = L.ii_anchor_eids
            c1 = xl_rowcol_to_cell(eid_to_row[a1], C_START)
            c2 = xl_rowcol_to_cell(eid_to_row[a2], C_START)
            ii_names[L.loop_uid] = f"loop_ii_{i}"
            wb.define_name(ii_names[L.loop_uid], f"=Events!{c2}-Events!{c1}")

    _events(wb, result, op_row, written_recs, eid_to_row, ii_names)

    n = len(written_recs)
    hi = min(n, max_gantt_rows)
    label = "whole graph" if hi == n else f"first {hi} of {n} events"
    _gantt(wb, 1, hi, f"Schedule ({label})", "Gantt")

    for L in result.loops:
        rows = [eid_to_row[e] for e in L.period_eids if e in eid_to_row]
        if rows:
            _gantt(
                wb,
                min(rows),
                max(rows),
                f"Representative period: {L.loop_name}",
                "Period",
            )
            break

    _summary(wb, result, n, compress_events)
    wb.close()
    return path
