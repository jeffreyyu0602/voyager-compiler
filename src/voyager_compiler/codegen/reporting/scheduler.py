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

Every timestamp the machine holds -- the clocks, the unit-free cycles, the
posted completions and the parked commits' floors -- is exposed by
``snapshot``, so the loop walker can compare the state after two iterations
(``snapshot_recipe``) and, when one is the other advanced by a period,
advance the whole state by any number of periods at once
(``advance_snapshot`` + ``load_snapshot``) without walking the iterations in
between.  While ``trace`` is a list, every event also appends a
time-invariant tuple to it (its times relative to ``trace_base``), which is
how the walker recognises a repeating iteration.
"""

import copy
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from voyager_compiler.codegen.reporting.cost import dram_cycles, op_info
from voyager_compiler.codegen.reporting.model import (
    LoopSkip,
    LoopStats,
    OpInfo,
    TimingRecord,
)
from voyager_compiler.hardware_config import AcceleratorConfig

# Trace entry tags.
T_COMPUTE = 1
T_DMA = 2
T_POST = 3
T_SEED = 4
T_COMMIT = 5
T_WAIT = 6
T_LOOP = 7
T_SKIP = 8


@dataclass
class _PendingCommit:
    """A committed op parked until its dependency semaphores are all posted.
    ``run`` walks its body (and signals its done-sem) once that happens;
    ``floor`` is the running max of the resolved deps' completion cycles."""

    node: object
    deps: list
    floor: int
    run: object


@dataclass
class Checkpoint:
    """A copy of everything ``ResourceState.restore`` puts back."""

    clocks: tuple
    fifos: dict
    posts: dict
    commits: list
    flags: tuple
    counts: tuple
    bytes: tuple
    loop_stats: dict


class ResourceState:
    def __init__(self, cost: AcceleratorConfig, calibration=None):
        self.cost = cost
        self.calibration = calibration
        self.now = 0
        self.mma_free = 0
        self.vector_free = 0
        self.dram_free = 0
        self.cur_loop = -1  # id() of the while_loop currently being walked
        self.cur_kernel = ""  # the bufferized nest being walked
        self.kernel_groups: Dict[str, str] = {}
        self.launched: set = set()  # kernels charged their launch overhead
        self.loop_stats: Dict[int, LoopStats] = {}  # id(loop) -> stats
        self.read_bytes = 0
        self.write_bytes = 0
        # DRAM traffic (read + write) bucketed by tensor role.
        self.cat_bytes = {"weight": 0, "activation": 0, "kv": 0}
        self.records: List[TimingRecord] = []
        self.skips: List[LoopSkip] = []
        self.ops: Dict[int, OpInfo] = {}  # id(node) -> OpInfo
        self._op_names: set = set()
        # semaphore slot -> FIFO of completion cycles
        self.sem_fifos: Dict[object, Deque[int]] = {}
        # id(async compute node) -> its end, handed to the following insert
        # so it can post the semaphore
        self.pending_post: Dict[int, int] = {}
        # committed (async) ops parked until their dependency semaphores post;
        # each runs (walks its body) only then.  See register_commit.
        self.pending_commits: List[_PendingCommit] = []
        self._resolving = False
        # set while a committed body is being walked: its ops floor their
        # start on commit_now (seeded to the dep-ready cycle, advanced by each
        # body op) and occupy their unit(s) without advancing the program
        # clock.
        self.in_commit = False
        self.commit_now = 0
        self.last_commit_node = None
        self.trace: Optional[list] = None
        self.trace_base = 0

    def _eid(self) -> int:
        return len(self.records)

    def _unit_free(self, unit: str) -> int:
        return self.mma_free if unit == "mma" else self.vector_free

    def _occupy(self, unit: str, end: int) -> None:
        if unit == "mma":
            self.mma_free = end
        else:
            self.vector_free = end

    def clock(self) -> int:
        """The clock a loop iteration is measured against: the program clock,
        or the commit clock while a committed body is being walked."""
        return self.commit_now if self.in_commit else self.now

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
            op.kernel = self.cur_kernel
            if self.calibration is not None:
                self.calibration.apply(
                    op, self.kernel_groups.get(self.cur_kernel)
                )
            self.ops[key] = op
        return op

    # -- event kinds --------------------------------------------------------

    def compute(self, node, path) -> TimingRecord:
        op = self.get_op(node)
        units_free = [self._unit_free(u) for u in op.units]
        if self.in_commit:
            # A committed op runs in program order on the datapath but off
            # the program clock: its start floors on the dep-ready cycle
            # (commit_now, advanced by each body op) and its unit(s), never on
            # self.now.
            start = max(self.commit_now, *units_free)
        else:
            start = max(self.now, *units_free)
        end = start + op.effective_cycles
        rec = TimingRecord(
            eid=self._eid(),
            node_name=node.name,
            kind="compute",
            resource=op.units,
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            kernel=self.cur_kernel,
            op_key=op.key,
        )
        self.records.append(rec)
        for u in op.units:
            self._occupy(u, end)
        if self.in_commit:
            # A committed op occupies its unit(s) but leaves the program clock;
            # its done-sem post carries the completion.  A sync one advances it.
            self.commit_now = end
            self.last_commit_node = node
            self.pending_post[id(node)] = end
        else:
            self.now = end
        if self.trace is not None:
            base = self.trace_base
            self.trace.append(
                (T_COMPUTE, id(node), self.in_commit, start - base, end - base)
            )
        return rec

    def post_semaphore(self, sem_key, src_node) -> None:
        """Publish the async compute ``src_node``'s completion onto
        ``sem_key``'s FIFO -- the compute-side twin of ``async_copy``'s post.
        """
        end = self.pending_post.pop(id(src_node), self.now)
        self.sem_fifos.setdefault(sem_key, deque()).append(end)

    def _dram_event(
        self, node, n_bytes: int, is_read: bool, category, path, sync: bool
    ) -> TimingRecord:
        start = max(self.now, self.dram_free)
        end = start + dram_cycles(n_bytes, self.cost)
        rec = TimingRecord(
            eid=self._eid(),
            node_name=node.name,
            kind="load" if is_read else "store",
            resource=("dram",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            kernel=self.cur_kernel,
            bytes=n_bytes,
            is_read=is_read,
            category=category,
            sync=sync,
        )
        self.records.append(rec)
        self.dram_free = end
        if sync:
            self.now = end
        if is_read:
            self.read_bytes += n_bytes
        else:
            self.write_bytes += n_bytes
        if category in self.cat_bytes:
            self.cat_bytes[category] += n_bytes
        if self.trace is not None:
            base = self.trace_base
            self.trace.append(
                (T_DMA, id(node), n_bytes, is_read, start - base, end - base)
            )
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
        # semaphore ``post_count`` times -- a reused block feeds several
        # per-step consumers off a single load -- so it clears that many waits.
        rec = self._dram_event(
            node, n_bytes, is_load, category, path, sync=False
        )
        fifo = self.sem_fifos.setdefault(sem_key, deque())
        for _ in range(int(post_count)):
            fifo.append(rec.end)
        if self.trace is not None:
            self.trace.append((T_POST, sem_key, int(post_count)))
        self._resolve_commits()
        return rec

    def seed_semaphore(self, sem_key, count: int) -> None:
        """Seed ``count`` initial credits onto a semaphore FIFO -- a
        ``voyager.fill`` (an output store-sem starts with a credit so a slot's
        first use waits nothing).  A credit is available from cycle 0 and has no
        producer, so consuming it stalls nothing."""
        fifo = self.sem_fifos.setdefault(sem_key, deque())
        for _ in range(int(count)):
            fifo.append(0)
        if self.trace is not None:
            self.trace.append((T_SEED, sem_key, int(count)))
        self._resolve_commits()

    def register_commit(self, node, deps, run) -> None:
        """Park a committed op: record its dependency semaphores and the ``run``
        thunk that walks its body, then try to resolve it at once.  In the
        common case every dep is already posted (the copy_in that feeds it ran
        just before), so it runs immediately; otherwise it waits for the posts.
        """
        self.pending_commits.append(_PendingCommit(node, list(deps), 0, run))
        if self.trace is not None:
            self.trace.append((T_COMMIT, id(node), tuple(deps)))
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
                            pc.floor = max(pc.floor, fifo.popleft())
                        else:
                            unresolved.append(sk)
                    pc.deps = unresolved
                    if unresolved:
                        continue
                    self.pending_commits.remove(pc)
                    self.in_commit = True
                    self.commit_now = pc.floor
                    self.last_commit_node = None
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
        fifo = self.sem_fifos.get(sem_key)
        if not fifo:
            raise ValueError(
                f"async_wait {node.name} has no outstanding async_copy on "
                f"its semaphore slot -- a copy / wait mismatch"
            )
        completion = fifo.popleft()
        start = self.now
        end = max(self.now, completion)
        rec = TimingRecord(
            eid=self._eid(),
            node_name=node.name,
            kind="async_wait",
            resource=("control",),
            start=start,
            end=end,
            iteration_path=tuple(path),
            loop_uid=self.cur_loop,
            kernel=self.cur_kernel,
        )
        self.records.append(rec)
        self.now = end
        if self.trace is not None:
            base = self.trace_base
            self.trace.append(
                (T_WAIT, id(node), sem_key, start - base, end - base)
            )
        return rec

    def launch(self, kernel: str) -> None:
        """Charge a calibrated kernel's launch overhead once, as a synchronous
        control event at its entry."""
        if self.calibration is None or kernel in self.launched:
            return
        self.launched.add(kernel)
        cycles = self.calibration.launch_cycles(self.kernel_groups.get(kernel))
        if not cycles:
            return
        rec = TimingRecord(
            eid=self._eid(),
            node_name=kernel,
            kind="launch",
            resource=("control",),
            start=self.now,
            end=self.now + cycles,
            kernel=kernel,
        )
        self.records.append(rec)
        self.now = rec.end

    # -- loops ----------------------------------------------------------------

    def enter_loop(self, node, trip_count: int) -> LoopStats:
        stats = self.loop_stats.get(id(node))
        if stats is None:
            stats = LoopStats(
                loop_uid=id(node),
                name=node.name,
                kernel=self.cur_kernel,
                trip_count=trip_count,
                start=self.clock(),
            )
            self.loop_stats[id(node)] = stats
        stats.entries += 1
        return stats

    def exit_loop(self, stats: LoopStats) -> None:
        stats.end = self.clock()

    # -- steady-state folding -------------------------------------------------

    def counters(self) -> Tuple[int, ...]:
        """The additive totals a skipped period multiplies: DRAM bytes by
        direction and category."""
        return (
            self.read_bytes,
            self.write_bytes,
            self.cat_bytes["weight"],
            self.cat_bytes["activation"],
            self.cat_bytes["kv"],
        )

    def add_counters(self, delta: Tuple[int, ...], times: int) -> None:
        self.read_bytes += delta[0] * times
        self.write_bytes += delta[1] * times
        self.cat_bytes["weight"] += delta[2] * times
        self.cat_bytes["activation"] += delta[3] * times
        self.cat_bytes["kv"] += delta[4] * times

    def snapshot(self) -> tuple:
        """Every timestamp the state holds, as nested tuples that compare by
        value: ``(in_commit, clocks, fifos, posts, commits)`` with the FIFOs
        and posts sorted by key."""
        return (
            self.in_commit,
            (
                self.now,
                self.commit_now,
                self.mma_free,
                self.vector_free,
                self.dram_free,
            ),
            tuple(
                (k, tuple(self.sem_fifos[k])) for k in sorted(self.sem_fifos)
            ),
            tuple((k, self.pending_post[k]) for k in sorted(self.pending_post)),
            tuple(
                (id(pc.node), tuple(pc.deps), pc.floor)
                for pc in self.pending_commits
            ),
        )

    def load_snapshot(self, snap: tuple) -> None:
        """Write a snapshot of the same shape as the current state back."""
        in_commit, clocks, fifos, posts, commits = snap
        self.in_commit = in_commit
        (
            self.now,
            self.commit_now,
            self.mma_free,
            self.vector_free,
            self.dram_free,
        ) = clocks
        for k, values in fifos:
            fifo = self.sem_fifos[k]
            fifo.clear()
            fifo.extend(values)
        for k, value in posts:
            self.pending_post[k] = value
        for pc, (_, _, floor) in zip(self.pending_commits, commits):
            pc.floor = floor

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            clocks=(
                self.now,
                self.commit_now,
                self.mma_free,
                self.vector_free,
                self.dram_free,
            ),
            fifos={k: deque(v) for k, v in self.sem_fifos.items()},
            posts=dict(self.pending_post),
            commits=[
                _PendingCommit(pc.node, list(pc.deps), pc.floor, pc.run)
                for pc in self.pending_commits
            ],
            flags=(
                self.in_commit,
                self.last_commit_node,
                self.cur_loop,
                self.cur_kernel,
            ),
            counts=(len(self.records), len(self.skips)),
            bytes=(self.read_bytes, self.write_bytes, dict(self.cat_bytes)),
            loop_stats={k: copy.copy(v) for k, v in self.loop_stats.items()},
        )

    def restore(self, cp: Checkpoint) -> None:
        """Put a checkpoint back.  ``loop_stats`` entries are restored in
        place, so a walker holding one keeps a live reference."""
        (
            self.now,
            self.commit_now,
            self.mma_free,
            self.vector_free,
            self.dram_free,
        ) = cp.clocks
        self.sem_fifos = {k: deque(v) for k, v in cp.fifos.items()}
        self.pending_post = dict(cp.posts)
        self.pending_commits = [
            _PendingCommit(pc.node, list(pc.deps), pc.floor, pc.run)
            for pc in cp.commits
        ]
        (
            self.in_commit,
            self.last_commit_node,
            self.cur_loop,
            self.cur_kernel,
        ) = cp.flags
        del self.records[cp.counts[0] :]
        del self.skips[cp.counts[1] :]
        self.read_bytes, self.write_bytes = cp.bytes[0], cp.bytes[1]
        self.cat_bytes = dict(cp.bytes[2])
        for k in list(self.loop_stats):
            if k not in cp.loop_stats:
                del self.loop_stats[k]
        for k, saved in cp.loop_stats.items():
            self.loop_stats[k].__dict__.update(saved.__dict__)


@dataclass(frozen=True)
class Recipe:
    """How one period advances the state: which timestamps move by
    ``shift`` (the rest stay: an idle unit, an unconsumed credit), and how
    many entries each semaphore FIFO consumes off its head -- a load that
    posted one credit per iteration up front is drained one per iteration.
    """

    shift: int
    clocks: Tuple[bool, ...]
    fifos: Tuple[Tuple[int, Tuple[bool, ...]], ...]  # (consumed, moving)
    posts: Tuple[bool, ...]
    commits: Tuple[bool, ...]


def _moving(a: int, b: int, shift: int) -> Optional[bool]:
    if b - a == shift:
        return True
    if b == a:
        return False
    return None


def _moving_all(a, b, shift) -> Optional[Tuple[bool, ...]]:
    flags = []
    for va, vb in zip(a, b):
        m = _moving(va, vb, shift)
        if m is None:
            return None
        flags.append(m)
    return tuple(flags)


def snapshot_recipe(a: tuple, b: tuple, shift: int) -> Optional[Recipe]:
    """The recipe taking snapshot ``a`` to ``b`` one period of ``shift``
    cycles later, or ``None`` when ``b`` is not ``a`` advanced by a period.

    A FIFO may be shorter in ``b`` than in ``a``: the period consumed the
    difference off its head, and what remains must match ``a``'s tail.
    """
    if a[0] != b[0]:
        return None
    clocks = _moving_all(a[1], b[1], shift)
    if clocks is None:
        return None
    if len(a[2]) != len(b[2]):
        return None
    fifos = []
    for (ka, va), (kb, vb) in zip(a[2], b[2]):
        if ka != kb or len(vb) > len(va):
            return None
        consumed = len(va) - len(vb)
        moving = _moving_all(va[consumed:], vb, shift)
        if moving is None:
            return None
        fifos.append((consumed, moving))
    if [k for k, _ in a[3]] != [k for k, _ in b[3]]:
        return None
    posts = _moving_all([v for _, v in a[3]], [v for _, v in b[3]], shift)
    if posts is None:
        return None
    if [c[:2] for c in a[4]] != [c[:2] for c in b[4]]:
        return None
    commits = _moving_all([c[2] for c in a[4]], [c[2] for c in b[4]], shift)
    if commits is None:
        return None
    return Recipe(shift, clocks, tuple(fifos), posts, commits)


def advance_snapshot(snap: tuple, recipe: Recipe, periods: int):
    """``snap`` after ``periods`` more periods under ``recipe``, or ``None``
    when a FIFO would run out of entries to consume."""
    total = periods * recipe.shift

    def move(values, flags):
        return tuple(v + total if m else v for v, m in zip(values, flags))

    fifos = []
    for (k, values), (consumed, moving) in zip(snap[2], recipe.fifos):
        drop = consumed * periods
        if drop > len(values):
            return None
        fifos.append((k, move(values[drop:], moving)))
    return (
        snap[0],
        move(snap[1], recipe.clocks),
        tuple(fifos),
        tuple(
            (k, v + total if m else v)
            for (k, v), m in zip(snap[3], recipe.posts)
        ),
        tuple(
            (n, deps, floor + total if m else floor)
            for (n, deps, floor), m in zip(snap[4], recipe.commits)
        ),
    )
