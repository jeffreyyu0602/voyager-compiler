# Return one complete result from either hardware model
from dataclasses import dataclass, field
from typing import Optional, Tuple


# Retain the complete backend score and its assumptions with a legal schedule
@dataclass(frozen=True)
class Evaluation:
    legal: bool
    reasons: Tuple[str, ...]
    policy: object = None
    traffic: object = None
    accumulation_footprint: Optional[int] = None
    local_accum_footprint: Optional[int] = None
    input_footprint: Optional[int] = None
    timing: object = None
    details: dict = field(default_factory=dict)
    schedule: object = None
    coverage: str = "dense MatrixUnit traffic and aggregate analytical timing; energy uncharacterized"

    # Read the single retained runtime estimate
    @property
    def runtime_cycles(self):
        return self.timing.runtime_cycles if self.legal and self.timing is not None else None


# Retain common runtime metrics for the search adapter
@dataclass(frozen=True)
class TimingEstimate:
    runtime_cycles: int
    ideal_cycles: float
    spatial_utilization: float
    useful_work_fraction: float

    @property
    def effective_utilization(self):
        return self.ideal_cycles / self.runtime_cycles
