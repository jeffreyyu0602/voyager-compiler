# Describe loop bounds, order, and spatial factors without search-engine types
from dataclasses import dataclass, field
from typing import Tuple

LOOPS = ("OX", "OY", "IC", "OC", "FX", "FY")
SPATIAL = ("OX", "OY")
WEIGHTS = ("IC", "OC", "FX", "FY")
REDUCTIONS = ("IC", "FX", "FY")

# Represent a complete semantic temporal level in inner-to-outer order
@dataclass(frozen=True)
class TemporalLevel:
    bounds: Tuple[int, ...] = (1,) * 6
    order: Tuple[str, ...] = LOOPS

    # Validate complete immutable factors and semantic order
    def __post_init__(self):
        object.__setattr__(self, "bounds", tuple(self.bounds))
        object.__setattr__(self, "order", tuple(self.order))
        if len(self.bounds) != 6 or any(type(n) is not int or n <= 0 for n in self.bounds):
            raise ValueError("temporal factors must be six positive integers")
        if len(self.order) != 6 or set(self.order) != set(LOOPS):
            raise ValueError("temporal order must permute OX/OY/IC/OC/FX/FY")

    # Construct readable schedules while normalizing omitted unit dimensions
    @classmethod
    def make(cls, order=LOOPS, **bounds) -> "TemporalLevel":
        if set(bounds) - set(LOOPS):
            raise ValueError("unknown temporal dimension")
        order = tuple(order)
        return cls(tuple(bounds.get(loop, 1) for loop in LOOPS), order + tuple(loop for loop in LOOPS if loop not in order))

    # Return a named factor without exposing enum numbering to mappers
    def bound(self, loop: str) -> int:
        return self.bounds[LOOPS.index(loop)]

    # Compare semantic loop nesting in inner-to-outer order
    def inside(self, loop: str, other: str) -> bool:
        return self.order.index(loop) < self.order.index(other)

    # Replace selected factors without changing traversal order
    def replace_bounds(self, replacements: dict) -> "TemporalLevel":
        return TemporalLevel(tuple(replacements.get(loop, self.bound(loop)) for loop in LOOPS), self.order)


# Carry temporal scheduling and the command's final output path
@dataclass(frozen=True)
class Schedule:
    l1: TemporalLevel = field(default_factory=TemporalLevel)
    l2: TemporalLevel = field(default_factory=TemporalLevel)
    l0_temporal: Tuple[int, ...] = (1,) * 6
    write_output_to_accum_buffer: bool = False
    spatial_factors: Tuple[Tuple[int, ...], ...] = ((1,) * 6,) * 3

    # Freeze caller-owned factor arrays before hashing or evaluation
    def __post_init__(self):
        object.__setattr__(self, "l0_temporal", tuple(self.l0_temporal))
        if type(self.write_output_to_accum_buffer) is not bool:
            raise ValueError("write_output_to_accum_buffer must be boolean")


# Read the spatial factor for one loop at one storage level
def spatial_factor(schedule, loop, level=0):
    return schedule.spatial_factors[level][LOOPS.index(loop)]
