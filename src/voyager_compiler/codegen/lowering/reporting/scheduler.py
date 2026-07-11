"""The resource/timing state machine.

A compute occupies the unit(s) named by its ``OpInfo.units`` — ``mma``
(systolic matrix array), ``vector`` (vector unit), or both (a fused GEMM /
conv, whose tail runs on the VU).  Whether it advances the **program clock**
depends on how its result is written:

* **synchronous** (its ``insert`` carries no semaphore): advances the program
  clock, so it serializes and forms the clock the waits reconcile against.
* **asynchronous** (its result is published by a ``semaphore``-carrying
  ``insert``): occupies its unit(s) but does **not** advance the program clock,
  so it runs while the clock moves on other units; the ``insert`` posts its
  completion onto the semaphore FIFO, exactly as ``async_copy`` does.  Two ops
  on the same unit still serialize (shared ``*_free`` counter); overlap appears
  only across different units.

``async_copy`` *is* asynchronous: it occupies DRAM for its transfer without
advancing the program clock, so a prefetch overlaps compute; ``async_wait``
blocks the program clock until the matching post (oldest into that semaphore
slot — a per-slot FIFO), whether that post came from a DMA or an async compute.
Every event records the ``eid``s whose ``end`` determined its
``start = max(...)`` so the workbook can recompute the schedule live.
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
        # DRAM traffic (read + write) bucketed by tensor role.
        self.cat_bytes = {"weight": 0, "activation": 0, "kv": 0}
        self.records: List[TimingRecord] = []
        self.ops: Dict[int, OpInfo] = {}  # id(node) -> OpInfo
        self._op_names: set = set()
        # semaphore slot -> FIFO of (completion_cycle, producer_eid)
        self.sem_fifos: Dict[object, deque] = {}
        # id(async compute node) -> its (end, eid), handed to the following
        # insert so it can post the semaphore
        self.pending_post: Dict[int, Tuple[int, int]] = {}
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

    def compute(self, node, path, async_post: bool = False) -> TimingRecord:
        eid = self._eid()
        op = self.get_op(node)
        start = max(self.now, *(self._unit_free(u) for u in op.units))
        end = start + op.effective_cycles
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
        # An async compute occupies its unit(s) but leaves the program clock;
        # the following insert posts its completion.  A sync one advances it.
        if async_post:
            self.pending_post[id(node)] = (end, eid)
        else:
            self.now = end
            self.last_now_eid = eid
        return rec

    def post_semaphore(self, sem_key, src_node) -> None:
        """Publish the async compute ``src_node``'s completion onto ``sem_key``'s
        FIFO -- the compute-side twin of the post ``async_copy`` makes."""
        end_eid = self.pending_post.pop(id(src_node), None)
        if end_eid is None:
            end_eid = (self.now, self.last_now_eid)
        self.sem_fifos.setdefault(sem_key, deque()).append(end_eid)

    def async_copy(
        self, node, n_bytes: int, is_load: bool, sem_key, path, category=""
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
            category=category,
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
        if category in self.cat_bytes:
            self.cat_bytes[category] += n_bytes
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
