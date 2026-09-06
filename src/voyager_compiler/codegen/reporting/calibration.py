"""Kernel signatures and RTL calibration of the compute cost model.

A **kernel** is one bufferized nest: the top-level nodes sharing a
``meta['scope']``, with the loops and committed bodies under them.  Its
**signature** is a static digest of what the RTL would execute per tile:
the anchor op, the tile grid (minus the outermost extent, which only sets
how many times the steady state repeats), every compute op's shape and
dtype, and the DMA / wait structure.  Kernels with equal signatures --
the same projection in every decoder layer, the same GEMM at every
context length that tiles alike -- share one measurement.

Calibration maps a signature to the cycles one steady-state iteration
took in RTL, measured with no DRAM model, so it replaces the *compute*
term of the estimate and nothing else: the analytic DMA cost and the
scheduler's overlap stay.  ``Calibration.apply`` prices every compute op
of a measured kernel by the ratio of measured to analytic cycles per
iteration -- exact for a one-op period, pro rata (``shared``) when a
period runs several ops.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import is_compute_op
from voyager_compiler.codegen.reporting.cost import op_info
from voyager_compiler.codegen.reporting.model import OpInfo
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _produces_tensor,
)
from voyager_compiler.codegen.transform.bufferize.emit import (
    COMMIT,
    COND,
    WHILE_LOOP,
    _loop_extents,
)
from voyager_compiler.hardware_config import AcceleratorConfig

_ASYNC_COPY = "voyager.async_copy.default"
_ASYNC_WAIT = "voyager.async_wait.default"


@dataclass
class KernelSignature:
    """One kernel's static digest: ``key`` is the hash the calibration
    sheet is keyed by, ``text`` its readable expansion."""

    key: str
    text: str
    anchor: str
    extents: Tuple[tuple, ...]
    ops: List[str] = field(default_factory=list)


def _submodules(gm: GraphModule, node: Node) -> List[GraphModule]:
    """The graph modules a control-flow node runs: a loop's body, a cond's
    branches, a commit's region."""
    if node.op != "call_function":
        return []
    if node.target is WHILE_LOOP:
        handles = [node.args[1]]
    elif node.target is COND:
        handles = list(node.args[1:3])
    elif node.target is COMMIT:
        handles = [node.args[0]]
    else:
        return []
    subs = []
    for h in handles:
        sub = getattr(gm, str(h.target), None)
        if isinstance(sub, GraphModule):
            subs.append(sub)
    return subs


def _describe(node: Node, cost: AcceleratorConfig) -> Optional[str]:
    """A compute node's static shape line, or ``None`` when it is not one
    (or carries no shape the cost model can size)."""
    is_compute = node.op == "call_module" or (
        node.op == "call_function"
        and _produces_tensor(node)
        and is_compute_op(node)
    )
    if not is_compute:
        return None
    try:
        op = op_info(node, cost)
    except (ValueError, AttributeError, IndexError, TypeError):
        return None
    shapes = "/".join(
        "x".join(str(d) for d in op.detail[k])
        for k in ("input", "weight", "output")
        if k in op.detail
    )
    return f"{op.op_type}[{','.join(op.units)}] {shapes} {op.detail['dtypes']}"


def _scan(gm: GraphModule, nodes, cost, sig: dict, depth: int) -> None:
    for node in nodes:
        if node.op == "call_function" and node.target is WHILE_LOOP:
            extents = tuple(_loop_extents(node))
            if depth == 0:
                extents = extents[1:]
            sig["extents"].append(extents)
        if str(getattr(node, "target", "")) == _ASYNC_COPY:
            sig["copies"] += 1
        elif str(getattr(node, "target", "")) == _ASYNC_WAIT:
            sig["waits"] += 1
        line = _describe(node, cost)
        if line is not None:
            sig["ops"].append(line)
        for sub in _submodules(gm, node):
            _scan(sub, list(sub.graph.nodes), cost, sig, depth + 1)


def kernel_signatures(
    model: GraphModule, cost: AcceleratorConfig
) -> Dict[str, KernelSignature]:
    """Digest every bufferized nest of ``model``, keyed by kernel name."""
    by_kernel: Dict[str, list] = {}
    anchors: Dict[str, str] = {}
    for node in model.graph.nodes:
        scope = node.meta.get("scope")
        if scope is None:
            continue
        by_kernel.setdefault(scope[0], []).append(node)
        anchors[scope[0]] = str(scope[1])
    out = {}
    pe = f"pe={cost.pe_array_size[0]}x{cost.pe_array_size[1]}"
    for kernel, nodes in by_kernel.items():
        sig = {"extents": [], "ops": [], "copies": 0, "waits": 0}
        _scan(model, nodes, cost, sig, 0)
        text = "; ".join(
            [
                anchors[kernel],
                pe,
                "grid=" + " ".join(str(e) for e in sig["extents"]),
                f"copies={sig['copies']} waits={sig['waits']}",
            ]
            + sorted(sig["ops"])
        )
        key = hashlib.sha1(text.encode()).hexdigest()[:12]
        out[kernel] = KernelSignature(
            key=key,
            text=text,
            anchor=anchors[kernel],
            extents=tuple(sig["extents"]),
            ops=sorted(sig["ops"]),
        )
    return out


@dataclass
class Measurement:
    """One calibrated kernel: the cycles one steady-state iteration took in
    RTL against the analytic cycles the form reported for it."""

    measured_per_iteration: float
    analytic_per_iteration: float
    launch_cycles: int = 0
    exact: bool = False  # the period runs one op, so the ratio is exact

    @property
    def ratio(self) -> float:
        return self.measured_per_iteration / self.analytic_per_iteration


class Calibration:
    """Signature -> measurement, applied to each ``OpInfo`` as it is priced."""

    def __init__(self, measurements: Dict[str, Measurement]):
        self.measurements = measurements
        self.applied: Dict[str, int] = {}

    def apply(self, op: OpInfo, signature: Optional[str]) -> None:
        m = self.measurements.get(signature) if signature else None
        if m is None:
            return
        op.measured_cycles = max(1, round(op.analytic_cycles * m.ratio))
        op.calibration = "exact" if m.exact else "shared"
        self.applied[signature] = self.applied.get(signature, 0) + 1

    def launch_cycles(self, signature: Optional[str]) -> int:
        """The measured fixed cost of entering a kernel (0 when unmeasured)."""
        m = self.measurements.get(signature) if signature else None
        return m.launch_cycles if m is not None else 0


# --------------------------------------------------------------------------
# The calibration form: what to simulate, and reading the results back
# --------------------------------------------------------------------------

FORM_SHEET = "Calibration"
FORM_HEADERS = [
    "Signature",
    "Kernel",
    "Anchor",
    "Kernels sharing",
    "Design points",
    "Period iters",
    "Ops per iteration",
    "Analytic cyc/iter",
    "N1",
    "T1",
    "N2",
    "T2",
    "Measured cyc/iter",
    "Launch cycles",
    "Signature text",
]
# Suggested run lengths, in periods: both past the fill, far enough apart
# that the difference is a clean multiple of the period.
_N1_PERIODS = 8
_N2_PERIODS = 16


def write_calibration_form(rows_by_point, path: str) -> str:
    """Write the RTL recipe: one row per distinct kernel signature across
    the design points, with the two run lengths to simulate and empty
    cells for their total cycles.

    Args:
        rows_by_point: ``{design point label: kernel_rows(result)}``.
        path: The ``.xlsx`` to write.

    Returns:
        ``path``.
    """
    import xlsxwriter

    by_sig: Dict[str, dict] = {}
    for label, rows in rows_by_point.items():
        for row in rows:
            if not row.signature or not row.trip_count:
                continue
            entry = by_sig.setdefault(
                row.signature, {"row": row, "points": [], "kernels": []}
            )
            if label not in entry["points"]:
                entry["points"].append(label)
            name = re.sub(r"layers_\d+_", "layers_N_", row.kernel)
            if name not in entry["kernels"]:
                entry["kernels"].append(name)
    wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    ws = wb.add_worksheet(FORM_SHEET)
    bold = wb.add_format({"bold": True})
    edit = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    for c, h in enumerate(FORM_HEADERS):
        ws.write(0, c, h, bold)
    ws.set_column(0, 0, 14)
    ws.set_column(1, 1, 40)
    ws.set_column(3, 4, 30)
    ws.set_column(6, 6, 40)
    ws.set_column(14, 14, 80)
    for r, (sig, entry) in enumerate(sorted(by_sig.items()), start=1):
        row = entry["row"]
        ws.write(r, 0, sig)
        ws.write(r, 1, row.kernel)
        ws.write(r, 2, row.anchor)
        ws.write(r, 3, ", ".join(entry["kernels"]))
        ws.write(r, 4, ", ".join(entry["points"]))
        period = max(1, row.period)
        ws.write_number(r, 5, period)
        ws.write(r, 6, row.ops_text)
        ws.write_number(r, 7, row.analytic_per_iteration)
        ws.write_number(r, 8, _N1_PERIODS * period)
        ws.write_blank(r, 9, None, edit)
        ws.write_number(r, 10, _N2_PERIODS * period)
        ws.write_blank(r, 11, None, edit)
        ws.write_blank(r, 12, None)
        ws.write_blank(r, 13, None)
        ws.write(r, 14, _signature_text(rows_by_point, row.kernel, sig))
    wb.close()
    return path


def _signature_text(rows_by_point, kernel: str, sig: str) -> str:
    for rows in rows_by_point.values():
        for row in rows:
            if row.signature == sig and row.kernel == kernel:
                return row.signature_text
    return ""


def load_calibration(path: str) -> Calibration:
    """Read a filled-in form back.  A row with ``T1`` and ``T2`` gives
    ``(T2 - T1) / (N2 - N1)`` cycles per iteration and ``T1 - N1 * that``
    launch cycles; a row with only ``T1`` gives ``T1 / N1`` and no launch
    term.  Rows with no measurement are ignored."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[FORM_SHEET]
    header = [c.value for c in ws[1]]
    col = {name: i for i, name in enumerate(header)}
    measurements: Dict[str, Measurement] = {}
    for cells in ws.iter_rows(min_row=2, values_only=True):
        sig = cells[col["Signature"]]
        t1 = cells[col["T1"]]
        if not sig or t1 is None:
            continue
        n1, n2, t2 = (cells[col[k]] for k in ("N1", "N2", "T2"))
        analytic = float(cells[col["Analytic cyc/iter"]])
        if t2 is not None and n2 is not None and n2 != n1:
            per_iteration = (float(t2) - float(t1)) / (float(n2) - float(n1))
            launch = float(t1) - float(n1) * per_iteration
        else:
            per_iteration = float(t1) / float(n1)
            launch = 0.0
        ops = str(cells[col["Ops per iteration"]] or "")
        measurements[str(sig)] = Measurement(
            measured_per_iteration=per_iteration,
            analytic_per_iteration=analytic,
            launch_cycles=max(0, round(launch)),
            exact=len(ops.split()) == 1,
        )
    return Calibration(measurements)
