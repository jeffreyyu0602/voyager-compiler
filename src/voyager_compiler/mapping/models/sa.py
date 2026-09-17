# Estimate SA tile overlap from a workload and normalized schedule
from dataclasses import dataclass
from math import prod
from ..schedule import LOOPS, spatial_factor
from ..timing.transfer import transfer_cycles
from .evaluation import Evaluation, TimingEstimate


@dataclass(frozen=True)
class TimingOptions:
    pass


class Evaluator:
    def __init__(self, target, workload, vector_timing, *, options=TimingOptions()):
        self.target, self.workload = target, workload
        self.output_cycles_per_vector = vector_timing

    def __call__(self, schedule):
        target, workload = self.target, self.workload
        l1, l2 = schedule.l1, schedule.l2
        weight_row_cycles = spatial_factor(schedule, "IC") + 2
        first_weight_loop = min(l1.order.index(loop) for loop in ("IC", "OC", "FX", "FY"))
        reuse_vectors = prod(l1.bound(loop) for loop in l1.order[:first_weight_loop])
        remaining_tiles = prod(l1.bound(loop) for loop in l1.order[first_weight_loop:]) * l2.bound("IC")
        compute_cycles = max(weight_row_cycles, reuse_vectors) * remaining_tiles
        input_fill = prod(l1.bound(loop) for loop in ("IC", "OY", "OX")) * transfer_cycles(target.k * target.input_bits, target.ic_port_bits)
        weight_fill = prod(l1.bound(loop) for loop in ("IC", "OC", "FY", "FX")) * spatial_factor(schedule, "IC") * transfer_cycles(target.n * target.weight_bits, target.oc_port_bits)
        output_vectors = prod(l1.bound(loop) for loop in ("OC", "OY", "OX"))
        vector_cycles = output_vectors * self.output_cycles_per_vector
        banked = target.double_buffered_accum and self.output_cycles_per_vector > 1
        tile_cycles = max(compute_cycles, input_fill, weight_fill, vector_cycles if banked else 0)
        outer_tiles = prod(l2.bound(loop) for loop in LOOPS if loop != "IC")
        drain = vector_cycles if target.double_buffered_accum else output_vectors * (self.output_cycles_per_vector - 1)
        runtime = max(input_fill, weight_fill) + outer_tiles * tile_cycles + drain
        fraction = workload.useful_work_fraction
        ideal = workload.input_channels * workload.output_channels * workload.output_x * workload.output_y * workload.filter_x * workload.filter_y / (target.k * target.n) * fraction
        utilization = prod(n for level in schedule.spatial_factors for n in level) / (target.k * target.n)
        return Evaluation(True, (), timing=TimingEstimate(runtime, ideal, utilization, fraction))
