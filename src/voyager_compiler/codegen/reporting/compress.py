"""Collapse a ``while_loop``'s steady-state iterations.

After scheduling, a software-pipelined loop's middle iterations become
structurally identical: the same events with the same durations and the same
intra-iteration offsets, differing only in absolute time.  Only the fill
(prologue) and drain (epilogue) iterations differ.  This stage finds the
longest run of identical iterations and reports the loop as
``prefix + period x repeat_count + suffix`` — which keeps the Excel ``Events``
sheet compact for large trip counts while the full record stream (for Perfetto)
is untouched.

Iterations are matched on a *normalized signature* (event order + per-event
kind/resource/duration, with absolute time removed), so a pattern is detected
even though async copies issued in one iteration complete in another.
"""

from dataclasses import dataclass
from typing import Dict, List

from .model import LoopSummary, ScheduleResult, TimingRecord


def _signature(recs: List[TimingRecord]):
    """Time-invariant shape of one iteration: the ordered per-event
    ``(node, kind, resource, duration)`` plus each event's offset from the
    iteration's start.  Two steady-state iterations hash equal."""
    base = min(r.start for r in recs)
    return tuple(
        (r.node_name, r.kind, r.resource, r.end - r.start, r.start - base)
        for r in recs
    )


_MAX_PERIOD = 256  # cap the period search (reduction / grid depths are small)


def _best_period(sigs: List) -> tuple:
    """Find the dominant repeating cycle: ``(period, start, repeats)`` such that
    ``period`` consecutive iterations from ``start`` repeat ``repeats`` times.

    A pipelined reduction loop cycles every ``k``-depth iterations
    (no-accumulate head, accumulate body, store tail), so the period is a
    *segment*, not a single iteration.  For each candidate period ``P`` we take
    the longest run where ``sig[i] == sig[i+P]``; the covered span ``repeats*P``
    (with ``repeats >= 2``) picks the winner, smallest ``P`` breaking ties.
    """
    n = len(sigs)
    best = (1, 0, 1)
    best_cov = 0
    for p in range(1, min(n // 2, _MAX_PERIOD) + 1):
        i = 0
        while i < n - p:
            j = i
            while j < n - p and sigs[j] == sigs[j + p]:
                j += 1
            reps = (j - i) // p + 1  # full periods covered by this run
            cov = reps * p
            if reps >= 2 and cov > best_cov:
                best_cov, best = cov, (p, i, reps)
            i = j + 1 if j > i else i + 1
    return best


def _carry_distance(recs, key_index, start, period, hi) -> int:
    """How many periods back a steady event's ``start_deps`` reach; the steady
    recurrence settles after this many periods."""
    eid_key = {r.eid: r.iteration_path for r in recs}
    d = 0
    for r in recs:
        ki = key_index[r.iteration_path]
        if not (start <= ki < hi):
            continue
        si = (ki - start) // period
        for dep in r.start_deps:
            pk = eid_key.get(dep)
            if pk is None:
                continue
            kj = key_index[pk]
            if start <= kj < hi:
                d = max(d, si - (kj - start) // period)
    return d


@dataclass
class _LoopCompact:
    """Transient per-loop data for the dep remap: each steady segment's eids in
    position order, so an elided event resolves to its last-written twin."""

    uid: int
    uncompressed: bool
    k: int  # number of written periods
    repeats: int
    seg_flat: List[List[int]]


def _summarize_loop(uid, name, recs):
    """Fold one loop's events into ``prefix + k periods + suffix``.

    ``recs`` is every event of a single ``while_loop``.  Group them by
    iteration, find the repeating steady block (``_best_period``), then keep
    only the first ``k = carry + 2`` periods: enough that a kept event's deps
    land on a kept period, and the last two periods give a settled initiation
    interval (the ``+2`` skips the still-ramping first periods).

    Example -- a 512-iteration reduction whose 64-iteration block repeats 6x
    starting at iteration 65 (``carry=1`` -> ``k=3``)::

        keys = [ 0..64 |     65..448      | 449..511 ]
                 fill      6 steady periods   drain
                           keep p0,p1,p2
                           drop p3,p4,p5

    Returns ``(LoopSummary, _LoopCompact)``: the summary carries the kept eids
    the sheet writes; the compact keeps *all* periods (incl. dropped) so
    ``_build_dep_remap`` can resolve a dropped event to its twin in period p2.
    """
    recs = sorted(recs, key=lambda r: r.eid)
    steps: Dict[tuple, List[TimingRecord]] = {}
    for r in recs:
        steps.setdefault(r.iteration_path, []).append(r)
    keys = sorted(steps, key=lambda p: min(x.eid for x in steps[p]))
    key_index = {k: i for i, k in enumerate(keys)}

    sigs = [_signature(steps[k]) for k in keys]
    period, start, repeats = _best_period(sigs)
    hi = start + repeats * period

    def eids(step_keys):
        return [r.eid for k in step_keys for r in steps[k]]

    def step_start(k):
        return min(r.start for r in steps[k])

    if repeats >= 2:  # one full period = start of next period - start of this
        period_duration = step_start(keys[start + period]) - step_start(
            keys[start]
        )
    elif keys[start : start + period]:
        span = keys[start]
        period_duration = max(r.end for r in steps[span]) - step_start(span)
    else:
        period_duration = 0

    carry = _carry_distance(recs, key_index, start, period, hi)
    k = carry + 2  # last two written periods are past the fill ramp
    base = dict(
        loop_name=name,
        trip_count=len(keys),
        repeat_count=repeats,
        period_duration=period_duration,
        loop_uid=uid,
        carry_distance=carry,
    )

    if repeats < k:  # too few steady periods to fold
        summary = LoopSummary(
            prefix_eids=[],
            period_eids=[],
            suffix_eids=[],
            uncompressed=True,
            **base,
        )
        return summary, _LoopCompact(uid, True, k, repeats, [])

    def seg(i):
        return eids(keys[start + i * period : start + (i + 1) * period])

    # Each pN is one steady period: seg(N), the eid list of one 64-iteration
    # block (e.g. p2 = [2321, 2322, ...]).  seg_flat holds all `repeats`
    # periods; written is the first k that reach the sheet (k=3 => p3,p4,p5
    # exist only in seg_flat, so the dep remap can find twins in them):
    #   seg_flat = [ p0, p1, p2, p3, p4, p5 ]   <- all 6, incl. dropped
    #   written  = [ p0, p1, p2 ]               <- what the sheet emits
    seg_flat = [seg(i) for i in range(repeats)]
    written = [seg_flat[i] for i in range(k)]
    summary = LoopSummary(
        prefix_eids=eids(keys[:start]),
        period_eids=written[0],  # first representative period
        suffix_eids=eids(keys[hi:]),
        written_periods=written,
        ii_anchor_eids=(written[k - 2][0], written[k - 1][0]),
        **base,
    )
    return summary, _LoopCompact(uid, False, k, repeats, seg_flat)


def compress_schedule(result: ScheduleResult) -> ScheduleResult:
    """Fill ``result.loops`` with one ``LoopSummary`` per ``while_loop`` (in
    first-appearance order) and ``result.dep_remap`` with the elided-event
    resolution the compressed live-formula sheet needs. Returns the same
    ``result`` (mutated)."""
    by_loop: Dict[int, List[TimingRecord]] = {}
    order: List[int] = []
    for r in result.records:
        if r.loop_uid == -1:
            continue
        if r.loop_uid not in by_loop:
            by_loop[r.loop_uid] = []
            order.append(r.loop_uid)
        by_loop[r.loop_uid].append(r)

    result.loops = []
    compacts: List[_LoopCompact] = []
    for uid in order:
        name = result.loop_names.get(uid, f"loop@{uid}")
        summary, compact = _summarize_loop(uid, name, by_loop[uid])
        result.loops.append(summary)
        compacts.append(compact)

    _build_dep_remap(result, order, by_loop, compacts)
    return result


def _build_dep_remap(result, order, by_loop, compacts) -> None:
    """Resolve every start-dep that lands on an elided (unwritten) event to its
    congruent last-written-period event.  Only drain / post-loop events reach
    there, so the resolved set is small: collect the needed eids, then map."""
    written = {r.eid for r in result.records if r.loop_uid == -1}
    for uid, summary, compact in zip(order, result.loops, compacts):
        if compact.uncompressed:
            written.update(r.eid for r in by_loop[uid])
        else:
            written.update(summary.prefix_eids)
            written.update(summary.suffix_eids)
            for period in summary.written_periods:
                written.update(period)

    needed = set()
    for r in result.records:
        if r.eid not in written:
            continue
        for dep in r.start_deps:
            if dep not in written:
                needed.add(dep)

    if not needed:
        return

    # Stack each dropped period on the last KEPT period (p_{k-1}) and match by
    # position: a dropped event is the twin directly above it, `skip` periods
    # later, so a survivor depending on it gets End(twin) + skip*II live.
    # (eids below are from the k=3 example; p2 kept, p3..p5 dropped)
    #
    #        pos:  0      1      2    ...
    #   p2 (kept): 2321   2322   2323 ...   <- twins, on the sheet
    #   p5 (drop): 4064   4065   4066 ...   -> dep_remap[4064] = (2321, skip=3)
    for compact in compacts:
        if compact.uncompressed:
            continue
        last = compact.seg_flat[compact.k - 1]
        for i in range(compact.k, compact.repeats):
            skip = i - (compact.k - 1)
            for pos, eid in enumerate(compact.seg_flat[i]):
                if eid in needed:
                    result.dep_remap[eid] = (last[pos], skip, compact.uid)
    # An unresolved dep (unexpected cross-loop edge) is left out; the writer
    # falls back to its cached End literal -- correct, just not live.
