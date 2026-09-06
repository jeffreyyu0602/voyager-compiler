"""Find the period of a loop's iteration stream while it is being walked.

Each walked iteration is reduced to a hashable *key*: its event trace
relative to the iteration's own clock, the cycles the clock advanced, and
how the loop-carried scalars changed.  Two iterations with equal keys did
the same things at the same relative times.  ``PeriodFinder.push`` reports
the smallest ``P`` for which the last ``2P`` keys form two identical
periods -- the point at which the walker can check the scheduler state and
fold the remaining periods.

Keys are compared by hash while tracking candidate periods (a run counter
per candidate, seeded from earlier occurrences of the same hash); the
walker confirms a candidate against the full keys it keeps for the last
``2P`` iterations, so a hash collision can only cost a rejected candidate,
never a wrong fold.  When the loop's structure has a known period, only
its multiples are candidates: a timing period can be nothing else.
"""

from collections import defaultdict
from typing import Dict, List, Optional

MAX_PERIOD = 1 << 16


class PeriodFinder:
    def __init__(self, max_period: int = MAX_PERIOD, multiple_of: int = 1):
        self.max_period = max_period
        self.multiple_of = multiple_of  # only such periods are candidates
        self.hashes: List[int] = []
        self.positions: Dict[int, List[int]] = defaultdict(list)
        self.runs: Dict[int, int] = {}  # candidate period -> matched run
        self.rejected: Dict[int, int] = {}  # period -> retry after index

    def __len__(self) -> int:
        return len(self.hashes)

    def push(self, key) -> Optional[int]:
        """Append the next iteration's key; return the smallest period whose
        last two periods match (by hash), or ``None``."""
        h = hash(key)
        i = len(self.hashes)
        self.hashes.append(h)
        for period in list(self.runs):
            if self.hashes[i - period] == h:
                self.runs[period] += 1
            else:
                del self.runs[period]
        for j in self.positions[h]:
            period = i - j
            if period > self.max_period or period % self.multiple_of:
                continue
            if period not in self.runs:
                run = 0
                k = i
                while (
                    k - period >= 0
                    and self.hashes[k] == self.hashes[k - period]
                ):
                    run += 1
                    k -= 1
                self.runs[period] = run
        self.positions[h].append(i)
        best = None
        for period, run in self.runs.items():
            if run < period or self.rejected.get(period, -1) > i:
                continue
            if best is None or period < best:
                best = period
        return best

    def reject(self, period: int) -> None:
        """A candidate the walker could not confirm: do not offer it again
        until a further period of iterations has been walked."""
        self.rejected[period] = len(self.hashes) + period

    def truncate(self, n: int) -> None:
        """Forget every key past the first ``n`` (a trial walk undone)."""
        del self.hashes[n:]
        for h in list(self.positions):
            kept = [j for j in self.positions[h] if j < n]
            if kept:
                self.positions[h] = kept
            else:
                del self.positions[h]
        self.runs.clear()
