"""Exact busy-lane unions over a folded schedule.

The walk records only the iterations it executed; each ``LoopSkip`` stands
for ``repeats`` copies of a template period, every copy the template's
intervals shifted by one more ``shift``.  The union length of a lane's
intervals over the whole makespan is still computed exactly, without
expanding the copies: inside a family of copies, every period window that
only copies within reach can touch is covered by the template folded
modulo ``shift`` (an interval at least a period long covers the window
outright), so those windows contribute a constant each; the copies at
either end, which meet the walked events around the fold and spill past
the interior, are expanded and swept with them.  A walked interval that
does reach into the interior -- a long transfer issued before the fold --
is merged into the windows it touches one by one.
"""

import math
from typing import Dict, List, Sequence, Tuple

from voyager_compiler.codegen.reporting.model import LoopSkip, TimingRecord

Interval = Tuple[int, int]

_LANES = {
    "compute": lambda r: r.kind == "compute",
    "dram": lambda r: r.resource == ("dram",),
    "any": lambda r: r.kind == "compute" or r.resource == ("dram",),
}


def _union(intervals: Sequence[Interval]) -> List[Interval]:
    """Merge ``[start, end)`` intervals into sorted, disjoint ones."""
    out: List[Interval] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _length(merged: Sequence[Interval]) -> int:
    return sum(e - s for s, e in merged)


def _fold(template: Sequence[Interval], shift: int) -> List[Interval]:
    """The template's coverage of one period window, modulo ``shift``."""
    pieces = []
    for s, e in template:
        if e - s >= shift:
            return [(0, shift)]
        a = s % shift
        b = a + (e - s)
        if b <= shift:
            pieces.append((a, b))
        else:
            pieces.append((a, shift))
            pieces.append((0, b - shift))
    return _union(pieces)


class _Family:
    """One skip's copies for one lane: ``template`` relative to the base of
    copy 0, ``count`` copies ``shift`` apart from ``base``."""

    def __init__(self, base, shift, count, template):
        self.base = base
        self.shift = shift
        self.count = count
        self.template = _union(template)
        lo = min(s for s, _ in self.template)
        hi = max(e for _, e in self.template)
        # Copies further apart than this cannot touch the same window.
        self.reach = math.ceil((hi - lo) / shift) + 1 if shift > 0 else 0

    def copy(self, m: int) -> List[Interval]:
        d = self.base + m * self.shift
        return [(s + d, e + d) for s, e in self.template]


def _families(records, skips, lane) -> Dict[int, _Family]:
    """Each skip's family for ``lane``, keyed by its position in ``skips``.
    A skip nested inside another's template period is expanded into that
    template (its copies are part of the period)."""
    picks = _LANES[lane]

    def expand(skip: LoopSkip) -> List[Interval]:
        """Every interval of the skip's copies, absolute, nested skips
        included."""
        out = []
        for m in range(skip.repeats):
            d = (m + 1) * skip.shift
            out.extend((s + d, e + d) for s, e in template(skip))
        return out

    def template(skip: LoopSkip) -> List[Interval]:
        """The template period's intervals, absolute: its walked records
        plus the copies of any skip folded inside it."""
        first, last = skip.template[0], skip.template[-1]
        out = [
            (records[e].start, records[e].end)
            for e in skip.template
            if picks(records[e]) and records[e].end > records[e].start
        ]
        for inner in skips:
            if inner is not skip and first <= inner.after_eid <= last:
                out.extend(expand(inner))
        return out

    out = {}
    for i, skip in enumerate(skips):
        tmpl = template(skip)
        if not tmpl:
            continue
        base = skip.start
        out[i] = _Family(
            base,
            skip.shift,
            skip.repeats,
            [(s - base + skip.shift, e - base + skip.shift) for s, e in tmpl],
        )
    return out


def _nested(skips) -> set:
    """Indices of skips folded inside another skip's template period."""
    out = set()
    for i, inner in enumerate(skips):
        for outer in skips:
            if outer is not inner and (
                outer.template[0] <= inner.after_eid <= outer.template[-1]
            ):
                out.add(i)
    return out


def _lane_union(walked: List[Interval], families: List[_Family]) -> int:
    intervals = list(walked)
    regions = []  # (start, end, family) interior spans, in time order
    for fam in families:
        q = fam.reach
        if fam.shift <= 0 or fam.count <= 4 * q + 1:
            for m in range(fam.count):
                intervals.extend(fam.copy(m))
            continue
        # The first and last interior copies spill past the interior; they
        # are expanded too, and what falls inside it is already in the fold.
        for m in range(2 * q):
            intervals.extend(fam.copy(m))
        for m in range(fam.count - 2 * q, fam.count):
            intervals.extend(fam.copy(m))
        lo, hi = q, fam.count - q
        regions.append(
            (fam.base + lo * fam.shift, fam.base + hi * fam.shift, fam)
        )
    regions.sort()
    for (_, e0, _), (s1, _, _) in zip(regions, regions[1:]):
        if s1 < e0:
            raise ValueError("folded steady states overlap in time")

    total = 0
    pieces: Dict[int, List[Interval]] = {}  # region index -> clipped pieces
    for s, e in _union(intervals):
        total += e - s
        for i, (r0, r1, _) in enumerate(regions):
            lo, hi = max(s, r0), min(e, r1)
            if lo < hi:
                total -= hi - lo
                pieces.setdefault(i, []).append((lo, hi))

    for i, (r0, r1, fam) in enumerate(regions):
        shift = fam.shift
        folded = _fold(fam.template, shift)
        per_window = _length(folded)
        windows = (r1 - r0) // shift
        total += windows * per_window
        # Windows a walked interval reaches into: fully covered ones count
        # a whole period, partial ones the union of the fold and the piece.
        partial: Dict[int, List[Interval]] = {}
        for s, e in pieces.get(i, []):
            w0, w1 = (s - r0) // shift, (e - r0 - 1) // shift
            for w in range(w0, w1 + 1):
                ws = r0 + w * shift
                a, b = max(s, ws) - ws, min(e, ws + shift) - ws
                if a == 0 and b == shift:
                    total += shift - per_window
                else:
                    partial.setdefault(w, []).append((a, b))
        for w, ps in partial.items():
            total += _length(_union(list(folded) + ps)) - per_window
    return total


def busy_unions(
    records: List[TimingRecord], skips: List[LoopSkip]
) -> Tuple[int, int, int]:
    """``(compute, dram, any)``: cycles during which the compute units, the
    DRAM interface, or either were busy, over the whole schedule."""
    nested = _nested(skips)
    out = []
    for lane, picks in _LANES.items():
        walked = [
            (r.start, r.end) for r in records if picks(r) and r.end > r.start
        ]
        families = [
            fam
            for i, fam in _families(records, skips, lane).items()
            if i not in nested
        ]
        out.append(_lane_union(walked, families))
    return tuple(out)
