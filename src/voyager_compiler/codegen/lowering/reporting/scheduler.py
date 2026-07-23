"""The resource/timing state machine.

A compute occupies the unit(s) named by its ``OpInfo.units`` — ``mma``
(systolic matrix array), ``vector`` (vector unit), or both (a fused GEMM /
conv, whose tail runs on the VU).  Whether it advances the **program clock**
depends on how it is dispatched:

* **synchronous**: advances the program clock, so it serializes and forms the
  clock the waits reconcile against.
* **asynchronous** (dispatched inside a ``voyager.commit``): occupies its
  unit(s) but does **not** advance the program clock, so it runs while the clock
  moves on other units; the commit posts its completion onto the done-semaphore
  FIFO, exactly as ``async_copy`` does.  Two ops on the same unit still
  serialize (shared ``*_free`` counter); overlap appears only across different
  units.

``async_copy`` *is* asynchronous: it occupies DRAM for its transfer without
advancing the program clock, so a prefetch overlaps compute; ``async_wait``
blocks the program clock until the matching post (oldest into that semaphore
slot — a per-slot FIFO), whether that post came from a DMA or an async compute.
Every event records the ``eid``s whose ``end`` determined its
``start = max(...)`` so the workbook can recompute the schedule live.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cost import dram_cycles, op_info
from .model import OpInfo, TimingRecord
from ....hardware import AcceleratorConfig


@dataclass
class _PendingCommit:
    """A committed op parked until its dependency semaphores are all posted.
    ``run`` walks its body (and signals its done-sem) once that happens;
    ``floor`` is the running max of the resolved deps' completion cycles and
    ``dep_eids`` the producers that set it (for ``start_deps`` wiring)."""

    deps: list
    floor: int
    dep_eids: list
    run: object


class ResourceState:
    def __init__(self, cost: AcceleratorConfig):
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
        # committed (async) ops parked until their dependency semaphores post;
        # each runs (walks its body) only then.  See register_commit.
        self.pending_commits: List[_PendingCommit] = []
        self._resolving = False
        # set while a committed body is being walked: its ops floor their start
        # on commit_now (seeded to the dep-ready cycle, advanced by each body op)
        # and occupy their unit(s) without advancing the program clock.
        self.in_commit = False
        self.commit_now = 0
        self.last_commit_eid: Optional[int] = None
        self.last_commit_node = None
        self.commit_deps: Tuple[int, ...] = ()
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
        units_free = [self._unit_free(u) for u in op.units]
        if self.in_commit:
            # A committed op runs in program order on the datapath but off the
            # program clock: its start floors on the dep-ready cycle (commit_now,
            # advanced by each body op) and its unit(s), never on self.now.  The
            # first body op also carries the dep producers in its start_deps.
            start = max(self.commit_now, *units_free)
            deps = self._deps(
                self.last_commit_eid,
                *(self._unit_last(u) for u in op.units),
                *self.commit_deps,
            )
        else:
            start = max(self.now, *units_free)
            deps = self._deps(
                self.last_now_eid, *(self._unit_last(u) for u in op.units)
            )
        end = start + op.effective_cycles
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
        if self.in_commit:
            # A committed op occupies its unit(s) but leaves the program clock;
            # its done-sem post carries the completion.  A sync one advances it.
            self.commit_now = end
            self.last_commit_eid = eid
            self.last_commit_node = node
            self.commit_deps = ()
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

    def _dram_event(
        self, node, n_bytes: int, is_read: bool, category, path, sync: bool
    ) -> TimingRecord:
        eid = self._eid()
        start = max(self.now, self.dram_free)
        deps = self._deps(self.last_now_eid, self.last_dram_eid)
        end = start + dram_cycles(n_bytes, self.cost)
        rec = TimingRecord(
            eid=eid,
            node_name=node.name,
            kind="load" if is_read else "store",
            resource=("dram",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            bytes=n_bytes,
            is_read=is_read,
            category=category,
            start_deps=deps,
            latency_kind="dram",
            latency_ref=n_bytes,
        )
        self.records.append(rec)
        self.dram_free = end
        self.last_dram_eid = eid
        if sync:
            self.now = end
            self.last_now_eid = eid
        if is_read:
            self.read_bytes += n_bytes
        else:
            self.write_bytes += n_bytes
        if category in self.cat_bytes:
            self.cat_bytes[category] += n_bytes
        return rec

    def async_copy(
        self,
        node,
        n_bytes: int,
        is_load: bool,
        sem_key,
        path,
        category="",
        post_count: int = 1,
    ) -> TimingRecord:
        # A DMA occupies DRAM but does NOT advance the program clock: its
        # matching async_wait is what reconciles it.  One load signals its
        # semaphore ``post_count`` times -- a reused block feeds several per-step
        # consumers off a single load -- so it clears that many waits.
        rec = self._dram_event(
            node, n_bytes, is_load, category, path, sync=False
        )
        fifo = self.sem_fifos.setdefault(sem_key, deque())
        for _ in range(int(post_count)):
            fifo.append((rec.end, rec.eid))
        self._resolve_commits()
        return rec

    def seed_semaphore(self, sem_key, count: int) -> None:
        """Seed ``count`` initial credits onto a semaphore FIFO -- a
        ``voyager.fill`` (an output store-sem starts with a credit so a slot's
        first use waits nothing).  A credit is available from cycle 0 and has no
        producer, so consuming it stalls nothing."""
        fifo = self.sem_fifos.setdefault(sem_key, deque())
        for _ in range(int(count)):
            fifo.append((0, None))
        self._resolve_commits()

    def register_commit(self, deps, run) -> None:
        """Park a committed op: record its dependency semaphores and the ``run``
        thunk that walks its body, then try to resolve it at once.  In the
        common case every dep is already posted (the copy_in that feeds it ran
        just before), so it runs immediately; otherwise it waits for the posts.
        """
        self.pending_commits.append(
            _PendingCommit(deps=list(deps), floor=0, dep_eids=[], run=run)
        )
        self._resolve_commits()

    def _resolve_commits(self) -> None:
        """Flush every committed op whose dependency semaphores are now all
        posted: pop each dep's FIFO entry (its completion floors the body's
        start), and once a commit's deps are empty, run its body off the program
        clock.  A run signals its done-sem, which may in turn resolve a commit
        that waited on it, so the scan repeats until it makes no progress."""
        if self._resolving:
            return
        self._resolving = True
        try:
            progress = True
            while progress:
                progress = False
                for pc in list(self.pending_commits):
                    unresolved = []
                    for sk in pc.deps:
                        fifo = self.sem_fifos.get(sk)
                        if fifo:
                            completion, eid = fifo.popleft()
                            pc.floor = max(pc.floor, completion)
                            if eid is not None:
                                pc.dep_eids.append(eid)
                        else:
                            unresolved.append(sk)
                    pc.deps = unresolved
                    if unresolved:
                        continue
                    self.pending_commits.remove(pc)
                    self.in_commit = True
                    self.commit_now = pc.floor
                    self.last_commit_eid = None
                    self.last_commit_node = None
                    self.commit_deps = tuple(pc.dep_eids)
                    pc.run()
                    self.in_commit = False
                    progress = True
        finally:
            self._resolving = False

    def assert_commits_drained(self, where: str) -> None:
        """At graph end every committed op must have resolved; one still parked
        means its dependency semaphore is never posted -- a deadlock in the
        emitted schedule.  This is *not* checked at each async_wait: a committed
        tile legitimately stays in flight across the retire of another (an
        async_wait blocks on its own semaphore, not on every commit)."""
        self._resolve_commits()
        if self.pending_commits:
            raise ValueError(
                f"unresolved committed op at {where}: a dependency semaphore "
                f"is never posted -- the emitted schedule would deadlock"
            )

    def dram_copy(self, node, reads, write, path) -> TimingRecord:
        """A DRAM->DRAM materialization (``pad`` / ``expand`` / ``cat``): read
        every input, then write the output.  Each ``(bytes, category)`` side is
        sized from its own tensor, so a copy that reads a weight and writes an
        activation lands in both buckets.

        It carries no semaphore, so nothing can wait on it and it cannot be
        overlapped -- it is synchronous and stalls the program clock.
        """
        for n_bytes, category in reads:
            self._dram_event(node, n_bytes, True, category, path, sync=True)
        n_bytes, category = write
        return self._dram_event(node, n_bytes, False, category, path, sync=True)

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
