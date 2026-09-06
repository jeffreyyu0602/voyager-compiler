"""Data structures shared across the reporting stages.

The estimator walks a bufferized FX graph once (``interpret`` +
``scheduler``), folding each loop's steady state as it goes, and the
reporting stage (``excel`` / ``perfetto`` / ``calibration``) reads the
result.  These dataclasses are the contract between them.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from voyager_compiler.hardware_config import AcceleratorConfig


@dataclass
class TimingRecord:
    """One scheduled execution of one FX node (a loop body node runs many
    times, so one node yields many records, tagged by ``iteration_path``).

    ``resource`` is the tuple of lanes the event occupies.  A compute pass
    holds ``("mma",)`` (matrix unit), ``("vector",)`` (vector unit), or
    both ``("mma", "vector")`` (a fused GEMM / conv, whose tail runs on the
    VU); a DMA holds ``("dram",)``; an ``async_wait`` holds
    ``("control",)`` -- a synchronization that uses no bandwidth.
    ``kernel`` names the bufferized nest the event belongs to, ``op_key``
    the ``OpInfo`` a compute event was priced by.
    """

    eid: int
    node_name: str
    kind: str  # compute | load | store | async_wait | launch
    resource: Tuple[str, ...]  # subset of {mma, vector, dram, control}
    start: int
    end: int
    iteration_path: Tuple[int, ...] = ()
    loop_uid: int = -1  # id() of the enclosing while_loop (-1 = top level)
    kernel: str = ""
    bytes: int = 0  # DRAM ops only
    is_read: bool = False  # DRAM loads (vs stores)
    category: str = ""
    op_key: str = ""


@dataclass
class OpInfo:
    """One static compute node (a row of the Operations sheet).  Many
    ``TimingRecord``s with the same ``key`` reference one ``OpInfo``.

    ``utilization`` is the fraction of peak the op sustains, so it costs
    ``ceil(ideal_cycles / utilization)`` cycles.  It is compute-only (DRAM
    is modeled separately, as ``async_copy`` events) and pre-computed by
    ``cost.py``.  A calibrated op carries the RTL-derived ``measured_cycles``
    instead, with ``calibration`` saying how it was derived (``exact`` for
    a one-op period, ``shared`` when the period's ops split one
    measurement pro rata).
    """

    key: str
    op_type: str  # gemm | conv | vector
    ideal_cycles: int
    detail: dict = field(default_factory=dict)
    units: Tuple[str, ...] = ("vector",)
    utilization: float = 1.0
    kernel: str = ""
    measured_cycles: Optional[int] = None
    calibration: str = ""

    @property
    def analytic_cycles(self) -> int:
        """The cost model's price: ideal cycles stretched by utilization."""
        return math.ceil(self.ideal_cycles / self.utilization)

    @property
    def effective_cycles(self) -> int:
        """The op's charged cost: the measurement when calibrated, else the
        analytic price."""
        if self.measured_cycles is not None:
            return self.measured_cycles
        return self.analytic_cycles


@dataclass
class LoopSkip:
    """A run of loop iterations the walk did not execute: ``repeats``
    periods of ``period`` iterations, each a time-shift by ``shift`` cycles
    of the ``template`` records (the last walked period), starting at
    iteration ``first_step`` and cycle ``start``.  ``bytes`` is one
    period's DRAM traffic by counter (``read`` / ``write`` / a category).
    """

    loop_uid: int
    kernel: str
    after_eid: int  # last walked record before the skip (-1 = none)
    first_step: int
    period: int
    repeats: int
    template: Tuple[int, ...]  # eids of the template period's records
    shift: int
    start: int
    bytes: Dict[str, int] = field(default_factory=dict)

    @property
    def iterations(self) -> int:
        return self.period * self.repeats

    @property
    def end(self) -> int:
        return self.start + self.repeats * self.shift


@dataclass
class LoopStats:
    """What the walk did with one ``while_loop``: how many iterations it
    walked and skipped, the period it found and how long a period takes.
    A nested loop is entered once per walked outer iteration; the counts
    accumulate over every entry."""

    loop_uid: int
    name: str
    kernel: str
    trip_count: int
    entries: int = 0
    walked: int = 0
    skipped: int = 0
    period: int = 0  # 0 = no period found
    shift: int = 0  # cycles per period
    start: int = 0  # clock at the first entry
    end: int = 0  # clock at the last exit


@dataclass
class ScheduleResult:
    """Everything the reporting stage needs.

    ``records`` holds the walked events; ``skips`` the folded steady-state
    runs between them, so the two together describe the whole schedule.
    ``busy_*`` are exact union lengths of the compute, DRAM and either
    lanes' busy intervals over the whole makespan.  ``kernel_signatures``
    maps each kernel to its ``calibration.KernelSignature``.
    """

    records: List[TimingRecord]
    ops: List[OpInfo]
    total_latency: int
    dram_read_bytes: int
    dram_write_bytes: int
    cost: AcceleratorConfig
    dram_weight_bytes: int = 0
    dram_activation_bytes: int = 0
    dram_kv_bytes: int = 0
    skips: List[LoopSkip] = field(default_factory=list)
    loops: List[LoopStats] = field(default_factory=list)
    busy_compute: int = 0
    busy_dram: int = 0
    busy_any: int = 0
    kernel_signatures: Dict[str, object] = field(default_factory=dict)
