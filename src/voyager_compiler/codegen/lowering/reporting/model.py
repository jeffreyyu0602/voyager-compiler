"""Data structures shared across the reporting stages.

The estimator runs three separate stages over a bufferized FX graph:
scheduling (``interpret`` + ``scheduler``), repeated-pattern compression
(``compress``), and reporting (``excel`` / ``perfetto``).  These dataclasses are
the contract between them.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CostParams:
    """Editable hardware knobs of the latency / traffic model.

    ``bytes_per_cycle`` is already ``dram_bandwidth / frequency`` (the tiler
    stores it that way), so no further conversion is needed.  ``unroll`` is the
    ``(rows, cols)`` of the systolic array: GEMM/conv run at
    ``unroll[0]*unroll[1]`` MACs/cycle, vector ops at ``unroll[1]`` lanes.
    """

    bytes_per_cycle: float
    setup_cycles: int = 1000
    unroll: Tuple[int, int] = (16, 16)


@dataclass
class TimingRecord:
    """One scheduled execution of one FX node (a loop body node runs many
    times, so one node yields many records, tagged by ``iteration_path``).

    ``resource`` is the lane the event occupies: ``"compute"`` (systolic
    array), ``"dram"`` (the shared DRAM interface), or ``"control"`` — a
    synchronization that uses no bandwidth (an ``async_wait``: it can stall the
    program clock but draws no Gantt bar).  ``start_deps`` lists the ``eid``s
    whose ``end`` fed this event's ``start = max(...)`` — recorded so the
    workbook can wire ``Start = MAX(End_of_dep...)`` and recalculate live (the
    dependency wiring is structural, hence invariant to edits).
    """

    eid: int
    node_name: str
    kind: str  # compute | async_copy | async_wait
    resource: str  # compute | dram | control
    start: int
    end: int
    iteration_path: Tuple[int, ...] = ()
    loop_uid: int = -1  # id() of the enclosing while_loop (-1 = top level)
    bytes: int = 0  # DRAM ops only
    is_read: bool = False  # DRAM loads (vs stores)
    start_deps: Tuple[int, ...] = ()
    # How the workbook recomputes this event's latency.  ("compute", op_key)
    # links to the Operations sheet; ("dram", n_bytes) is the bandwidth
    # formula; ("const", cycles) is a fixed latency (e.g. a wait = 0).
    latency_kind: str = "const"
    latency_ref: object = None
    detail: dict = field(default_factory=dict)


@dataclass
class OpInfo:
    """One static compute node (a row of the Operations sheet).  Many
    ``TimingRecord``s with the same ``key`` reference one ``OpInfo``."""

    key: str
    op_type: str  # gemm | conv | vector
    ideal_cycles: int
    detail: dict = field(default_factory=dict)


@dataclass
class LoopSummary:
    """A compressed ``while_loop`` region: a steady-state period that repeats
    ``repeat_count`` times, framed by a non-repeating prefix / suffix."""

    loop_name: str
    trip_count: int
    prefix_eids: List[int]
    period_eids: List[int]  # one representative period (template)
    suffix_eids: List[int]
    repeat_count: int
    period_duration: int


@dataclass
class ScheduleResult:
    """Everything the reporting stage needs.

    ``records`` is the full, uncompressed event stream (one per dynamic node
    execution) — feeds Perfetto.  ``ops`` is the static compute table.
    ``loops`` (filled by ``compress``) collapses steady-state iterations for
    the compact Excel views.
    """

    records: List[TimingRecord]
    ops: List[OpInfo]
    total_latency: int
    dram_read_bytes: int
    dram_write_bytes: int
    cost: CostParams
    loops: List[LoopSummary] = field(default_factory=list)
    loop_names: dict = field(default_factory=dict)  # id(loop) -> node name

    def record_by_eid(self, eid: int) -> Optional[TimingRecord]:
        # records are appended in eid order, so index directly when aligned.
        if 0 <= eid < len(self.records) and self.records[eid].eid == eid:
            return self.records[eid]
        return next((r for r in self.records if r.eid == eid), None)
