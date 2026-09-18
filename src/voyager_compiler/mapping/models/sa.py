# Check SA input legality and estimate runtime for one normalized schedule
from dataclasses import asdict
from math import prod
from ..schedule import LOOPS, spatial_factor
from ..timing.transfer import transfer_cycles, packing_factor
from ..timing.buffers import buffer_completion
from .bias import bias_timing
from .input import input_tile_shape, input_bank_traffic
from .output import OutputOptions, output_timing
from .evaluation import Evaluation, TimingEstimate


class Evaluator:
    def __init__(self, target, workload, vector_timing, *, options=OutputOptions()):
        self.target, self.workload = target, workload
        self.vector_timing = vector_timing
        self.options = options.for_vector(target, vector_timing)
        self.input_cache = {}
        self.useful_fraction = workload.useful_work_fraction

    def input_loading(self, schedule):
        key = schedule.l1.bounds, schedule.l2.bounds
        if key not in self.input_cache:
            l1, l2 = (dict(zip(LOOPS, level.bounds)) for level in (schedule.l1, schedule.l2))
            width, height = input_tile_shape(l1, self.workload)
            bits = self.workload.input_bits
            pack = packing_factor(self.target.k * bits, self.target.ic_port_bits, l1["IC"])
            reasons = []
            if width * height * l1["IC"] > min(self.target.input_buffer_words, 65536):
                reasons.append("input tile including halo and padding exceeds one input-buffer bank")
            if width > 512 or height > 1023 or max(l2["OX"], l2["OY"], l1["IC"] // pack) > 512:
                reasons.append("input traversal exceeds the writer's signed 10-bit coordinates")
            payload = self.target.k * pack * bits
            if payload % 8 or payload // 8 > 1023 or pack > 16 or (pack - 1) * self.target.k * bits > 1023:
                reasons.append("input packing or burst size exceeds the command fields")
            if transfer_cycles(payload, self.target.ic_port_bits) > 15:
                reasons.append("input transfer exceeds the command beat count")
            self.input_cache[key] = ((None, reasons) if reasons else input_bank_traffic(
                l1, l2, self.workload, lanes=self.target.k, element_bits=bits,
                port_bits=self.target.ic_port_bits, pack=pack))
        return self.input_cache[key]

    def __call__(self, schedule):
        inputs, reasons = self.input_loading(schedule)
        if reasons:
            return Evaluation(False, tuple(reasons))
        target, workload = self.target, self.workload
        l1, l2 = schedule.l1, schedule.l2
        # Load SA rows while the preceding spatial tile computes.
        sa_weight_loading_cycles = spatial_factor(schedule, "IC") + 2
        first_weight_loop = min(l1.order.index(loop) for loop in ("IC", "OC", "FX", "FY"))
        reuse_vectors = prod(l1.bound(loop) for loop in l1.order[:first_weight_loop])
        remaining_tiles = prod(l1.bound(loop) for loop in l1.order[first_weight_loop:]) * l2.bound("IC")
        compute_cycles = max(sa_weight_loading_cycles, reuse_vectors) * remaining_tiles
        weight_rows = prod(l1.bound(loop) for loop in ("IC", "OC", "FY", "FX")) * spatial_factor(schedule, "IC")
        weight_fill_cycles = weight_rows * transfer_cycles(target.n * target.weight_bits, target.oc_port_bits)
        output_vectors = prod(l1.bound(loop) for loop in ("OC", "OY", "OX"))
        output_cycles = self.vector_timing.cycles_per_vector
        vector_cycles = output_vectors * output_cycles
        banked = target.double_buffered_accum and self.vector_timing.port_cycles_per_vector > 1
        tile_cycles = max(compute_cycles, weight_fill_cycles, vector_cycles if banked else 0)
        outer_tiles = prod(l2.bound(loop) for loop in LOOPS if loop != "IC")
        startup = max(inputs.first_fill_cycles, weight_fill_cycles)
        input_finish, steps, bounded = buffer_completion(
            1, 1, inputs.fills, 2, inputs.max_fill_cycles, 0,
            compute_cycles // l2.bound("IC"), startup)
        input_issue = max(0, input_finish - startup)
        input_readiness = dict(wait_cycles=max(0, input_issue - outer_tiles * compute_cycles),
                              sequence_steps=steps, serialized_bound=bounded,
                              max_fill_bound=inputs.max_fill_cycles != inputs.min_fill_cycles)
        matrix_cycles = max(outer_tiles * tile_cycles, input_issue)
        levels = tuple(tuple((loop, level.bound(loop)) for loop in level.order) for level in (l1, l2))
        prefix_cycles = max(0, sa_weight_loading_cycles - reuse_vectors)
        bias_readiness, output_readiness = {}, {}
        if workload.has_bias:
            bias, bias_readiness = bias_timing(target, *levels, prefix_at=first_weight_loop,
                                               prefix_cycles=prefix_cycles)
            matrix_cycles = max(matrix_cycles, bias.producer_cycles)
        if banked:
            runtime = startup + matrix_cycles + vector_cycles
        else:
            stream, output_readiness = output_timing(
                target, levels[0] + levels[1], output_cycles, direct=self.vector_timing.direct,
                options=self.options, prefix_at=first_weight_loop, prefix_cycles=prefix_cycles)
            runtime = startup + max(matrix_cycles + output_readiness['output_forward_cycles'], stream.consumer_cycles)
        pack = packing_factor(target.k * workload.input_bits, target.ic_port_bits, l1.bound("IC"))
        input_loading = dict(asdict(inputs), element_bits=workload.input_bits, lane_elements=target.k,
            pack_factor=pack, port_bits=target.ic_port_bits,
            external_beats=inputs.requests * transfer_cycles(target.k * pack * workload.input_bits, target.ic_port_bits),
            **input_readiness)
        ideal = (workload.input_channels * workload.output_channels * workload.output_x * workload.output_y
                 * workload.filter_x * workload.filter_y / (target.k * target.n) * self.useful_fraction)
        utilization = prod(n for level in schedule.spatial_factors for n in level) / (target.k * target.n)
        return Evaluation(True, (), timing=TimingEstimate(runtime, ideal, utilization, self.useful_fraction),
            details=dict(output_timing=output_readiness, input_loading=input_loading, bias_timing=bias_readiness))
