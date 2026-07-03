"""The resource/timing state machine.

Compute is **synchronous**: a pass occupies the unit(s) named by its
``OpInfo.units`` — ``mma`` (systolic matrix array), ``vector`` (vector
unit), or both (a fused GEMM / conv, whose tail runs on the VU) — and
**advances the program clock**, so compute passes serialize.  The units
are labelled and queued separately (the report attributes each pass to its
unit), but they do **not** overlap — asynchronous matrix ∥ vector execution
is not modelled here yet.

``async_copy`` *is* asynchronous: it occupies DRAM for its transfer without
advancing the program clock, so a prefetch overlaps compute; ``async_wait``
blocks the program clock until the matching transfer (oldest into that
semaphore slot — a per-slot FIFO) completes.  Every event records the
``eid``s whose ``end`` determined its ``start = max(...)`` so the workbook
can recompute the schedule live.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

from .cost import dram_cycles, op_info
from .model import CostParams, OpInfo, TimingRecord


class ResourceState:
    def __init__(self, cost: CostParams):
        self.cost = cost
        self.now = 0
        self.mma_free = 0
        self.vector_free = 0
        self.dram_free = 0
        self.cur_loop = -1  # id() of the while_loop currently being walked
        self.loop_names: Dict[int, str] = {}  # id(loop) -> node name
        self.read_bytes = 0
        self.write_bytes = 0
        self.records: List[TimingRecord] = []
        self.ops: Dict[int, OpInfo] = {}  # id(node) -> OpInfo
        self._op_names: set = set()
        # semaphore slot -> FIFO of (completion_cycle, copy_eid)
        self.sem_fifos: Dict[object, deque] = {}
        # last events that advanced each clock (for start_deps wiring)
        self.last_now_eid: Optional[int] = None
        self.last_mma_eid: Optional[int] = None
        self.last_vector_eid: Optional[int] = None
        self.last_dram_eid: Optional[int] = None

    def _eid(self) -> int:
        return len(self.records)

    def _deps(self, *eids) -> Tuple[int, ...]:
        return tuple(e for e in eids if e is not None)

    def _unit_free(self, unit: str) -> int:
        return self.mma_free if unit == "mma" else self.vector_free

    def _unit_last(self, unit: str) -> Optional[int]:
        return self.last_mma_eid if unit == "mma" else self.last_vector_eid

    def _occupy(self, unit: str, end: int, eid: int) -> None:
        if unit == "mma":
            self.mma_free, self.last_mma_eid = end, eid
        else:
            self.vector_free, self.last_vector_eid = end, eid

    def get_op(self, node) -> OpInfo:
        """Memoize one ``OpInfo`` per static compute node (keyed by identity, so
        same-named nodes in different loop bodies stay distinct)."""
        key = id(node)
        op = self.ops.get(key)
        if op is None:
            op = op_info(node, self.cost)
            name = op.key
            i = 2
            while op.key in self._op_names:
                op.key = f"{name}#{i}"
                i += 1
            self._op_names.add(op.key)
            self.ops[key] = op
        return op

    # -- event kinds --------------------------------------------------------

    def compute(self, node, path) -> TimingRecord:
        eid = self._eid()
        op = self.get_op(node)
        start = max(self.now, *(self._unit_free(u) for u in op.units))
        end = start + op.ideal_cycles
        deps = self._deps(
            self.last_now_eid, *(self._unit_last(u) for u in op.units)
        )
        rec = TimingRecord(
            eid=eid,
            node_name=node.name,
            kind="compute",
            resource=op.units,
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            start_deps=deps,
            latency_kind="compute",
            latency_ref=op.key,
            detail=dict(op.detail),
        )
        self.records.append(rec)
        for u in op.units:
            self._occupy(u, end, eid)
        self.now = end
        self.last_now_eid = eid
        return rec

    def async_copy(
        self, node, n_bytes: int, is_load: bool, sem_key, path
    ) -> TimingRecord:
        eid = self._eid()
        start = max(self.now, self.dram_free)
        deps = self._deps(self.last_now_eid, self.last_dram_eid)
        end = start + dram_cycles(n_bytes, self.cost)
        rec = TimingRecord(
            eid=eid,
            node_name=node.name,
            kind="load" if is_load else "store",
            resource=("dram",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            bytes=n_bytes,
            is_read=is_load,
            start_deps=deps,
            latency_kind="dram",
            latency_ref=n_bytes,
        )
        self.records.append(rec)
        # DMA occupies the DRAM resource but does NOT advance the program clock.
        self.dram_free = end
        self.last_dram_eid = eid
        if is_load:
            self.read_bytes += n_bytes
        else:
            self.write_bytes += n_bytes
        self.sem_fifos.setdefault(sem_key, deque()).append((end, eid))
        return rec

    def async_wait(self, node, sem_key, path) -> TimingRecord:
        eid = self._eid()
        fifo = self.sem_fifos.get(sem_key)
        if not fifo:
            raise ValueError(
                f"async_wait {node.name} has no outstanding async_copy on "
                f"its semaphore slot -- a copy / wait mismatch"
            )
        completion, copy_eid = fifo.popleft()
        start = self.now
        end = max(self.now, completion)
        deps = self._deps(self.last_now_eid, copy_eid)
        rec = TimingRecord(
            eid=eid,
            node_name=node.name,
            kind="async_wait",
            resource=("control",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            start_deps=deps,
            latency_kind="const",
            latency_ref=0,
        )
        self.records.append(rec)
        self.now = end
        self.last_now_eid = eid
        return rec
