"""Kernel build groups and RTL calibration of the compute cost model.

A **kernel** is one bufferized nest: the top-level nodes sharing a
``meta['scope']``, with the loops and committed bodies under them.  Two
kernels are the *same operation* exactly when bufferization built them from
the same cached nest.  It records that twice:

* ``meta['build_key']`` -- a digest of the build cache's own key, which is
  the node's structural + shape/dtype signature.  Content addressed, so the
  same nest gets the same key in a two-layer compile and in the full model;
  this is what carries an RTL measurement between them.
* ``meta['build_group']`` -- a counter, readable and close to the ``group``
  column of ``layers.txt``, but meaningless outside its own compile.

The sheet is keyed by the first and shows the second.

Calibration maps a group to the cycles one steady-state iteration took in
RTL, measured with no DRAM model, so it replaces the *compute* term of the
estimate and nothing else: the analytic DMA cost and the scheduler's
overlap stay.  ``Calibration.apply`` prices every compute op of a measured
kernel by the ratio of measured to analytic cycles per iteration -- exact
for a one-op period, pro rata (``shared``) when a period runs several ops.

The same sheet carries the group's measured **power**, which corrects
nothing: the compiler has no power model, and the number exists only so the
reporting workbook can turn a latency into an energy.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch.fx import GraphModule

from voyager_compiler.codegen.reporting.model import OpInfo

_LAYER_RE = re.compile(r"layers_(\d+)_")


@dataclass
class KernelGroup:
    """One kernel's build group: ``key`` is what the calibration sheet is
    keyed by, ``members`` every kernel bufferization built the same way."""

    key: str  # the calibration identity: build_key, else "k:<kernel>"
    group: int  # build_group, readable but per-compile
    anchor: str = ""
    members: List[str] = field(default_factory=list)

    @property
    def num_shared_kernels(self) -> int:
        """Kernels built from this nest, counting repeats inside a layer as
        well as across layers -- q, k and v in every layer is 3 x n_layers."""
        return len(self.members)


def kernel_groups(model: GraphModule) -> Dict[str, KernelGroup]:
    """Every kernel of ``model``, keyed by kernel name, carrying the build
    group bufferization stamped on its nodes.

    A kernel whose nodes carry no ``build_group`` (bufferization never built
    it) is its own singleton group, keyed by name, so it still gets a row.
    """
    groups: Dict[str, KernelGroup] = {}
    out: Dict[str, KernelGroup] = {}
    for node in model.graph.nodes:
        scope = node.meta.get("scope")
        if scope is not None:
            kernel, anchor = scope[0], str(scope[1])
        elif node.op != "placeholder" and node.meta.get("build_key"):
            # Bufferization built no nest here, so there is no scope; the node
            # is its own kernel, named and anchored as the graph names it.
            kernel, anchor = node.name, str(node.target)
        else:
            continue
        if kernel in out:
            continue
        group = node.meta.get("build_group")
        # build_key is content addressed, so it identifies the same nest in
        # any compile; build_group is a per-run counter and only groups
        # within this one.
        key = node.meta.get("build_key") or (
            f"g{group}" if group is not None else f"k:{kernel}"
        )
        entry = groups.get(key)
        if entry is None:
            entry = groups[key] = KernelGroup(
                key=key,
                group=group if group is not None else -1,
                anchor=anchor,
            )
        entry.members.append(kernel)
        out[kernel] = entry
    return out


@dataclass
class Measurement:
    """One calibrated group: the cycles one steady-state iteration took in
    RTL against the analytic cycles the form reported for it, plus the
    silicon power the group draws (0 when unmeasured)."""

    measured_per_iteration: float
    analytic_per_iteration: float
    launch_cycles: int = 0
    exact: bool = False  # the period runs one op, so the ratio is exact
    power: float = 0.0  # W, measured on silicon; not a correction
    # The cells it was read from, so a calibrated compile can write the
    # measurement back onto its own sheet.
    n1: float = 0.0
    t1: float = 0.0
    n2: float = 0.0
    t2: float = 0.0

    @property
    def ratio(self) -> float:
        return self.measured_per_iteration / self.analytic_per_iteration


class Calibration:
    """Build group -> measurement, applied to each ``OpInfo`` as it is
    priced.  ``power`` is carried through untouched for the reporting
    workbook; only ``ratio`` and ``launch_cycles`` change a schedule."""

    def __init__(self, measurements: Dict[str, Measurement]):
        self.measurements = measurements
        self.applied: Dict[str, int] = {}

    def apply(self, op: OpInfo, group: Optional[str]) -> None:
        m = self.measurements.get(group) if group else None
        if m is None or not m.measured_per_iteration:
            return
        op.measured_cycles = max(1, round(op.analytic_cycles * m.ratio))
        op.calibration = "exact" if m.exact else "shared"
        self.applied[group] = self.applied.get(group, 0) + 1

    def launch_cycles(self, group: Optional[str]) -> int:
        """The measured fixed cost of entering a kernel (0 when unmeasured)."""
        m = self.measurements.get(group) if group else None
        return m.launch_cycles if m is not None else 0

    def power(self, group: Optional[str]) -> float:
        """The group's measured power in W (0 when unmeasured)."""
        m = self.measurements.get(group) if group else None
        return m.power if m is not None else 0.0


# --------------------------------------------------------------------------
# The calibration sheet: what to simulate, and reading the results back
# --------------------------------------------------------------------------

FORM_SHEET = "Calibration"
FORM_HEADERS = [
    "Key",
    "Group",
    "Kernel",
    "Anchor",
    "Instances",
    "Trip",
    "Period iters",
    "Ops per iteration",
    "Analytic cyc/iter",
    "N1",
    "T1",
    "N2",
    "T2",
    "Measured cyc/iter",
    "Launch cycles",
    "Power (W)",
]
# Suggested run lengths, in periods: both past the fill, far enough apart
# that the difference is a clean multiple of the period.
_N1_PERIODS = 8
_N2_PERIODS = 16


def sheet_rows(rows) -> List:
    """One row per distinct build key, in the order the kernels first run.

    A key is the unit of measurement -- every kernel sharing it runs the same
    nest -- so one row per key is exactly the list of things to simulate, and
    ``Instances`` says how many kernels each stands for.

    Every node carries a key now, DMA-only ones included, so computeless
    kernels are left off: there is nothing in them for the RTL to measure.
    """
    seen, out = set(), []
    for row in rows:
        if row.group and row.ops_per_period and row.group not in seen:
            seen.add(row.group)
            out.append(row)
    return out


def calibration_sheet(wb, rows, calibration=None, formats=None) -> int:
    """Write the calibration sheet into an open ``xlsxwriter`` workbook.

    One row per distinct build key, in first-run order (see ``sheet_rows``).

    ``calibration`` fills the measurement columns of the keys it matched, so
    a calibrated compile's workbook says what priced it -- and leaves the
    keys it did not match blank, which is where the coverage went.

    Returns the number of rows written.
    """
    ws = wb.add_worksheet(FORM_SHEET)
    fmt = formats or {}
    bold = fmt.get("bold") or wb.add_format({"bold": True})
    edit = fmt.get("edit") or wb.add_format(
        {"bg_color": "#FFF2CC", "border": 1}
    )
    for c, h in enumerate(FORM_HEADERS):
        ws.write(0, c, h, bold)
    ws.set_column(_col("Key"), _col("Key"), 14)
    ws.set_column(_col("Group"), _col("Group"), 8)
    ws.set_column(_col("Kernel"), _col("Kernel"), 46)
    ws.set_column(_col("Anchor"), _col("Anchor"), 30)
    ws.set_column(_col("Ops per iteration"), _col("Ops per iteration"), 40)
    ws.freeze_panes(1, 0)

    measured = calibration.measurements if calibration is not None else {}
    n1c, t1c = _letter("N1"), _letter("T1")
    n2c, t2c = _letter("N2"), _letter("T2")
    mc = _letter("Measured cyc/iter")

    written = 0
    for r, row in enumerate(sheet_rows(rows), start=1):
        written += 1
        ws.write(r, _col("Key"), row.group)
        ws.write(
            r,
            _col("Group"),
            f"g{row.group_id}" if row.group_id >= 0 else "",
        )
        ws.write(r, _col("Kernel"), row.kernel)
        ws.write(r, _col("Anchor"), row.anchor)
        ws.write_number(r, _col("Instances"), row.num_shared_kernels)
        ws.write_number(r, _col("Trip"), row.trip_count)
        period = max(1, row.period)
        ws.write_number(r, _col("Period iters"), period)
        ws.write(r, _col("Ops per iteration"), row.ops_text)
        ws.write_number(
            r, _col("Analytic cyc/iter"), row.analytic_per_iteration
        )
        m = measured.get(row.group)
        if m is not None and m.n1:
            ws.write_number(r, _col("N1"), m.n1, edit)
            ws.write_number(r, _col("T1"), m.t1, edit)
            if m.n2:
                ws.write_number(r, _col("N2"), m.n2, edit)
                ws.write_number(r, _col("T2"), m.t2, edit)
            else:
                ws.write_number(r, _col("N2"), _N2_PERIODS * period)
                ws.write_blank(r, _col("T2"), None, edit)
        else:
            ws.write_number(r, _col("N1"), _N1_PERIODS * period)
            ws.write_blank(r, _col("T1"), None, edit)
            ws.write_number(r, _col("N2"), _N2_PERIODS * period)
            ws.write_blank(r, _col("T2"), None, edit)
        if m is not None and m.power:
            ws.write_number(r, _col("Power (W)"), m.power, edit)
        else:
            ws.write_blank(r, _col("Power (W)"), None, edit)
        one = r + 1
        ws.write_formula(
            r,
            _col("Measured cyc/iter"),
            f'=IFERROR(IF(AND({t2c}{one}<>"",{n2c}{one}<>{n1c}{one}),'
            f"({t2c}{one}-{t1c}{one})/({n2c}{one}-{n1c}{one}),"
            f'{t1c}{one}/{n1c}{one}),"")',
        )
        ws.write_formula(
            r,
            _col("Launch cycles"),
            f'=IFERROR(MAX(0,{t1c}{one}-{n1c}{one}*{mc}{one}),"")',
        )
    return written


def power_column() -> str:
    """The Power column's A1 letter, for a formula on another sheet."""
    return _letter("Power (W)")


def _col(name: str) -> int:
    """A header's zero-based column index."""
    return FORM_HEADERS.index(name)


def _letter(name: str) -> str:
    """A header's A1 column letter, so the formulas follow the header order
    instead of hard-coding letters that shift when a column is added.  The
    sheet is under 26 columns wide, so one letter covers it."""
    return chr(ord("A") + _col(name))


def load_calibration(path: str) -> Calibration:
    """Read a filled-in sheet back, from a standalone form or from a
    reporting workbook (both name the sheet ``Calibration``).

    A row with ``T1`` and ``T2`` gives ``(T2 - T1) / (N2 - N1)`` cycles per
    iteration and ``T1 - N1 * that`` launch cycles; a row with only ``T1``
    gives ``T1 / N1`` and no launch term.  The arithmetic is redone here
    rather than read out of the ``Measured cyc/iter`` cell, because a
    workbook written but never opened in Excel has no cached formula
    results.  ``Power (W)`` is read straight through.  Rows with neither a
    measurement nor a power are ignored.  The sheet carries one row per key,
    so a repeated key means the file was edited by hand; the last row wins.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[FORM_SHEET]
    header = [c.value for c in ws[1]]
    col = {name: i for i, name in enumerate(header) if name in FORM_HEADERS}
    measurements: Dict[str, Measurement] = {}

    def cell(cells, name):
        i = col.get(name)
        return cells[i] if i is not None and i < len(cells) else None

    for cells in ws.iter_rows(min_row=2, values_only=True):
        group = cell(cells, "Key")
        if not group:
            continue
        t1 = cell(cells, "T1")
        power = cell(cells, "Power (W)")
        if t1 is None and power is None:
            continue
        n1, n2, t2 = (cell(cells, k) for k in ("N1", "N2", "T2"))
        analytic = float(cell(cells, "Analytic cyc/iter") or 0) or 1.0
        per_iteration = launch = 0.0
        if t1 is not None and n1:
            if t2 is not None and n2 is not None and n2 != n1:
                per_iteration = (float(t2) - float(t1)) / (
                    float(n2) - float(n1)
                )
                launch = float(t1) - float(n1) * per_iteration
            else:
                per_iteration = float(t1) / float(n1)
        ops = str(cell(cells, "Ops per iteration") or "")
        measurements[str(group)] = Measurement(
            measured_per_iteration=per_iteration,
            analytic_per_iteration=analytic,
            launch_cycles=max(0, round(launch)),
            exact=len(ops.split()) == 1,
            power=float(power) if power is not None else 0.0,
            n1=float(n1 or 0),
            t1=float(t1 or 0),
            n2=float(n2 or 0),
            t2=float(t2 or 0),
        )
    return Calibration(measurements)
