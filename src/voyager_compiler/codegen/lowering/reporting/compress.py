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


def _summarize_loop(
    uid: int, name: str, recs: List[TimingRecord]
) -> LoopSummary:
    recs = sorted(recs, key=lambda r: r.eid)
    steps: Dict[tuple, List[TimingRecord]] = {}
    for r in recs:
        steps.setdefault(r.iteration_path, []).append(r)
    keys = sorted(steps, key=lambda p: min(x.eid for x in steps[p]))

    sigs = [_signature(steps[k]) for k in keys]
    period, start, repeats = _best_period(sigs)
    template = keys[start : start + period]
    prefix = keys[:start]
    suffix = keys[start + repeats * period :]

    def eids(step_keys):
        return [r.eid for k in step_keys for r in steps[k]]

    def step_start(k):
        return min(r.start for r in steps[k])

    if repeats >= 2:  # one full period = start of next period - start of this
        period_duration = step_start(keys[start + period]) - step_start(
            keys[start]
        )
    elif template:
        span = template[0]
        period_duration = max(r.end for r in steps[span]) - step_start(span)
    else:
        period_duration = 0

    return LoopSummary(
        loop_name=name,
        trip_count=len(keys),
        prefix_eids=eids(prefix),
        period_eids=eids(template),  # one representative period (P iterations)
        suffix_eids=eids(suffix),
        repeat_count=repeats,
        period_duration=period_duration,
    )


def compress_schedule(result: ScheduleResult) -> ScheduleResult:
    """Fill ``result.loops`` with one ``LoopSummary`` per ``while_loop`` (in
    first-appearance order).  Returns the same ``result`` (mutated)."""
    by_loop: Dict[int, List[TimingRecord]] = {}
    order: List[int] = []
    for r in result.records:
        if r.loop_uid == -1:
            continue
        if r.loop_uid not in by_loop:
            by_loop[r.loop_uid] = []
            order.append(r.loop_uid)
        by_loop[r.loop_uid].append(r)

    result.loops = [
        _summarize_loop(
            uid, result.loop_names.get(uid, f"loop@{uid}"), by_loop[uid]
        )
        for uid in order
    ]
    return result
