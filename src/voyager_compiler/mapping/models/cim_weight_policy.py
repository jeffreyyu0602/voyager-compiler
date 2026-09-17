# Model resident-weight loading, reuse, and controller descriptors
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Iterator
from ..target import CIMTarget
from ..schedule import Schedule, TemporalLevel, WEIGHTS, SPATIAL


# Specify valid logical B rows/columns and the controller's packed source fetch
@dataclass(frozen=True)
class WeightFetch:
    source_rows: int
    valid_columns: int
    burst_bytes: int
    pack_factor: int = 1
    transpose: bool = False


# Summarize exact descriptor policy without materializing large mapping traces
@dataclass(frozen=True)
class WeightPolicy:
    reader_l1: TemporalLevel
    reader_l2: TemporalLevel
    fits: bool
    sequence_sets: int
    compute_replays: int
    fetch_replays: int
    sequence_count: int
    macs_per_set_use: int

    # Count descriptor handshakes including oversized singleton streaming
    @property
    def descriptors(self) -> int:
        return self.sequence_count if self.fits else self.full_set_loads

    # Count full-set loads rather than individual B-port writes
    @property
    def full_set_loads(self) -> int:
        return self.sequence_count * self.sequence_sets * self.fetch_replays

    # Count all selected-set runs, including replayed uses
    @property
    def set_uses(self) -> int:
        return self.sequence_count * self.sequence_sets * self.compute_replays


# Apply WeightController::reader collapse, replay, and oversized-fetch policy
def weight_policy(target: CIMTarget, schedule: Schedule) -> WeightPolicy:
    l1, l2 = schedule.l1, schedule.l2
    active_l1 = [loop for loop in WEIGHTS if l1.bound(loop) > 1]
    reuse = [loop for loop in SPATIAL if all(l1.inside(loop, weight) for weight in active_l1)]
    replay = [loop for loop in SPATIAL if loop not in reuse and l1.bound(loop) > 1
              and all(l1.inside(weight, loop) for weight in active_l1)]
    reader_l1 = l1.replace_bounds({loop: 1 for loop in reuse + replay})
    if prod(reader_l1.bounds) > target.b_sets:
        reader_l1 = reader_l1.replace_bounds({loop: l1.bound(loop) for loop in replay})
        replay = []
    active_l2 = [loop for loop in ("IC", "OC", "FY") if l2.bound(loop) > 1]
    outer_reuse = [loop for loop in SPATIAL if all(l2.inside(loop, weight) for weight in active_l2)]
    reader_l2 = l2.replace_bounds({loop: 1 for loop in outer_reuse})
    sequence_sets = prod(reader_l1.bounds)
    replays = prod(l1.bound(loop) for loop in replay) * prod(l2.bound(loop) for loop in outer_reuse)
    fits = sequence_sets <= target.b_sets
    return WeightPolicy(reader_l1, reader_l2, fits, sequence_sets, replays,
                        1 if fits else replays, prod(reader_l2.bounds),
                        prod(l1.bound(loop) for loop in reuse))


# Traverse a semantic level in the same order as lowered hardware counters
def coordinates(level: TemporalLevel) -> Iterator[dict]:
    outer_order = tuple(reversed(level.order))
    for values in product(*(range(level.bound(loop)) for loop in outer_order)):
        yield dict(zip(outer_order, values))


# Identify the logical weight selected by a pair of temporal coordinates
def weight_key(outer: dict, inner: dict) -> tuple:
    return tuple(outer[loop] for loop in ("OC", "IC", "FY")) + tuple(inner[loop] for loop in WEIGHTS)


# Expand controller descriptors lazily for directed policy and lowering checks
def descriptor_trace(target: CIMTarget, schedule: Schedule) -> Iterator[tuple]:
    policy = weight_policy(target, schedule)
    for outer in coordinates(policy.reader_l2):
        if policy.fits:
            yield tuple(weight_key(outer, inner) for inner in coordinates(policy.reader_l1)), policy.compute_replays
        else:
            for _ in range(policy.fetch_replays):
                for inner in coordinates(policy.reader_l1):
                    yield (weight_key(outer, inner),), 1
