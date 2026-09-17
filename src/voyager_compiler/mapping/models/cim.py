# Check CIM legality, count hardware work, and assemble its evaluation
from dataclasses import dataclass
from math import gcd, prod
from ..timing.transfer import packing_factor, ceil_div
from .input import input_tile_shape, input_bank_traffic
from ..target import CIMTarget
from ..schedule import LOOPS, REDUCTIONS, SPATIAL, WEIGHTS, Schedule
from ..workload import Workload
from .evaluation import Evaluation
from .cim_timing import TimingOptions, estimate_cycles
from .cim_weight_policy import WeightFetch, weight_policy


# Count requested payloads, whole-port transfers, and on-chip accesses separately
@dataclass(frozen=True)
class Traffic:
    descriptors: int
    full_set_loads: int
    set_uses: int
    resident_set_switches: int
    weight_unique_useful_bytes: int
    weight_fetched_useful_bytes: int
    weight_requested_bytes: int
    weight_external_beats: int
    weight_external_transfer_bytes: int
    b_writes: int
    b_write_bytes: int
    a_beats: int
    a_bytes: int
    c_beats: int
    c_bytes: int
    compute_scalar_macs: int
    useful_scalar_macs: float
    local_accum_updates: int
    local_accum_intermediate_reads: int
    local_accum_intermediate_writes: int
    buffer_accum_reads: int
    buffer_accum_intermediate_writes: int
    buffer_accum_final_writes: int
    direct_output_vectors: int
    input_buffer_reads: int
    input_buffer_writes: int
    input_requests: int
    input_requested_bytes: int
    input_external_beats: int
    input_external_transfer_bytes: int
    output_external_beats: int
    output_external_transfer_bytes: int
    bias_requests: int
    bias_external_beats: int
    bias_external_transfer_bytes: int
    buffer_output_reads: int
    vector_output_vectors: int

# Identify outer spatial contexts using the processor's exact nesting predicate
def outer_context(schedule: Schedule, loop: str) -> bool:
    return any(schedule.l2.bound(reduction) > 1 and schedule.l2.inside(loop, reduction)
               for reduction in ("IC", "FY"))

# Reject shapes and schedules that overflow or disagree across the current ABI
def _legality(target, schedule, workload, fetch, pack, width, height):
    l1, l2 = schedule.l1, schedule.l2
    reasons = []
    if schedule.l0_temporal != (1,) * 6:
        reasons.append("L0 temporal factors must be one; compiler tilings omit L0")
    if l2.bound("FX") != 1:
        reasons.append("outer FX is unsupported by the current controller ABI")
    if any(n > 1023 for level in (l1, l2) for n in level.bounds):
        reasons.append("temporal factor exceeds the 10-bit controller bound")
    if l1.bound("FX") > 15 or l1.bound("FY") > 15:
        reasons.append("inner filter extent exceeds InputController's 4-bit field")
    if prod(l1.bounds) * prod(l2.bounds) > 0xFFFFFFFF:
        reasons.append("operation count exceeds the 32-bit controller counter")
    if l2.bound("OC") > 1 and outer_context(schedule, "OC"):
        reasons.append("outer output-column partial contexts are unsupported")
    if target.double_buffered_accum and schedule.write_output_to_accum_buffer and any(
            outer_context(schedule, loop) for loop in SPATIAL):
        reasons.append("outer partial contexts cannot use banked accumulation output")
    extents = {loop: l1.bound(loop) * l2.bound(loop) for loop in LOOPS}
    expected = dict(OX=workload.output_x, OY=workload.output_y,
                    IC=workload.input_channels, OC=workload.output_channels,
                    FX=workload.filter_x, FY=workload.filter_y)
    extents["IC"] *= target.k
    extents["OC"] *= target.n
    if extents != expected:
        reasons.append("mapping extents must match the padded workload and fixed IC/OC lanes")
    if workload.stride > 255 or workload.padding > 255:
        reasons.append("stride and padding exceed the 8-bit command fields")
    if max(workload.input_x, workload.input_y, workload.input_channels,
           workload.output_x, workload.output_y, workload.output_channels) > 65535:
        reasons.append("tensor extent exceeds a 16-bit controller stride")
    tensor_sizes = (workload.input_x * workload.input_y * workload.input_channels * target.input_bits // 8,
                    workload.filter_x * workload.filter_y * workload.input_channels * workload.output_channels * target.weight_bits // 8,
                    workload.output_x * workload.output_y * workload.output_channels * target.accum_bits // 8)
    if max(tensor_sizes) > 0xFFFFFFFF:
        reasons.append("tensor byte extent exceeds relative 32-bit addressing")
    if l2.bound("FY") > 1 and l1.bound("FY") != 1:
        reasons.append("outer FY requires unit inner FY in InputController")
    if workload.stride > 1 and (l1.bound("FX") == 1) != (l1.bound("FY") == 1):
        reasons.append("strided rectangular filters disagree with the input reader's row stride")
    if workload.input_transpose:
        reasons.append("input transpose must be materialized before CIM evaluation")
    if width > 512 or height > 1023 or max(l2.bound("OX"), l2.bound("OY"), l1.bound("IC") // pack) > 512:
        reasons.append("input traversal exceeds the writer's signed 10-bit coordinates")
    if (height - 1) * (workload.stride if l1.bound("FY") == 1 else 1) + l2.bound("FY") - 1 > 1023:
        reasons.append("projected input row exceeds the 10-bit coordinate field")
    if (width - 1) * (workload.stride if l1.bound("FX") == 1 else 1) > 1023:
        reasons.append("projected input column exceeds the 10-bit coordinate field")
    if width * height * l1.bound("IC") > min(target.input_buffer_words, 65536):
        reasons.append("input tile including halo and padding exceeds one input-buffer bank")
    if (pack > 16 or target.k * pack > 1023 or (pack - 1) * target.k * target.input_bits > 1023
            or ceil_div(target.k * pack * target.input_bits, target.ic_port_bits) > 15):
        reasons.append("input packing or burst size exceeds the command fields")
    if any(type(value) is not int or value <= 0 for value in
           (fetch.source_rows, fetch.valid_columns, fetch.burst_bytes, fetch.pack_factor)):
        reasons.append("weight fetch fields must be positive integers")
    else:
        vector_bytes = (fetch.source_rows if fetch.transpose else fetch.valid_columns) * target.weight_bits // 8
        if (fetch.source_rows > target.k or fetch.valid_columns > target.n
                or fetch.burst_bytes != vector_bytes * fetch.pack_factor
                or fetch.pack_factor not in (1, 2, 4, 8, 16)
                or l1.bound("OC") % fetch.pack_factor
                or fetch.burst_bytes > 1023
                or ceil_div(fetch.burst_bytes * 8, target.oc_port_bits) > 15):
            reasons.append("weight fetch must fit packed rows and command fields")
        if fetch.transpose and (target.k >= 64 or target.n >= 64 or fetch.pack_factor != 1
                                or (fetch.source_rows, fetch.valid_columns) != (target.k, target.n)):
            reasons.append("weight transpose requires a complete unpacked set with K and N below 64")
        if fetch.transpose and (workload.filter_x != 1 or workload.filter_y != 1):
            reasons.append("transposed weight addressing supports matrix products without filter axes")
        fx_stride = l2.bound("IC") * l1.bound("IC") * fetch.source_rows * workload.output_channels * target.weight_bits // 8
        if not fetch.transpose and ((workload.filter_x > 1 and fx_stride > 0xFFFFFF)
                                    or (workload.filter_y > 1 and workload.filter_x * fx_stride > 0xFFFFFF)):
            reasons.append("weight filter stride exceeds the 24-bit address-generation fields")
        max_fetch_bits = max(lanes * target.weight_bits * target.oc_port_bits // gcd(lanes * target.weight_bits, target.oc_port_bits)
                             for lanes in (target.k, target.n))
        if (fetch.burst_bytes * 8 > max_fetch_bits
                or (fetch.pack_factor - 1) * target.n * target.weight_bits > 1023
                or (fetch.pack_factor > 1 and fetch.valid_columns != target.n)):
            reasons.append("packed weights must fit the unpacker's fixed row slices and 10-bit offset")
    if type(fetch.transpose) is not bool or fetch.transpose != workload.weight_transpose:
        reasons.append("weight fetch transpose must match the workload")
    return reasons




# Bind immutable workload and timing assumptions once per search
class Evaluator:
    # Reuse geometry summaries across candidates with the same temporal factors
    def __init__(self, target, workload, *, options=TimingOptions()):
        self.target, self.workload = target, workload
        self.options = options
        self.useful_positions = workload.useful_positions
        self.input_cache = {}

    # Evaluate a complete schedule using the bound hardware and timing assumptions
    def __call__(self, schedule, fetch=None):
        target, workload = self.target, self.workload
        options = self.options
        l1, l2 = schedule.l1, schedule.l2
        pack = packing_factor(target.k * target.input_bits, target.ic_port_bits, l1.bound("IC"))
        weight_pack = 1 if workload.weight_transpose else packing_factor(target.n * target.weight_bits, target.oc_port_bits, l1.bound("OC"))
        fetch = fetch or WeightFetch(target.k, target.n,
                                    (target.k if workload.weight_transpose else target.n) * target.weight_bits // 8 * weight_pack,
                                    weight_pack, workload.weight_transpose)
        width, height = input_tile_shape(dict(zip(LOOPS, l1.bounds)), workload)
        reasons = _legality(target, schedule, workload, fetch, pack, width, height)
        x_extent = l1.bound("OX") * (l2.bound("OX") if outer_context(schedule, "OX") else 1)
        y_extent = l1.bound("OY") * (l2.bound("OY") if outer_context(schedule, "OY") else 1)
        footprint = l1.bound("OC") * x_extent * y_extent
        if footprint > target.accum_buffer_words:
            reasons.append("live partial outputs exceed accumulation capacity")
        policy = weight_policy(target, schedule)
        if policy.compute_replays > 65535:
            reasons.append("replay count exceeds the 16-bit descriptor field")
        if reasons:
            return Evaluation(False, tuple(reasons))
        key = (l1.bounds, l2.bounds)
        if key not in self.input_cache:
            self.input_cache[key] = input_bank_traffic(
                dict(zip(LOOPS, l1.bounds)), dict(zip(LOOPS, l2.bounds)), workload,
                lanes=target.k, element_bits=target.input_bits, port_bits=target.ic_port_bits, pack=pack,
                request_latency=options.memory_request_latency)
        inputs, reasons = self.input_cache[key]
        if reasons:
            return Evaluation(False, tuple(reasons))

        traffic = count_traffic(target, schedule, workload, fetch, policy, inputs, pack, self.useful_positions, footprint)
        timing = estimate_cycles(target, schedule, workload, fetch, options, policy, traffic, inputs)
        return Evaluation(True, (), policy=policy, traffic=traffic,
                          accumulation_footprint=footprint,
                          input_footprint=width * height * l1.bound("IC"),
                          timing=timing)


# Evaluate a standalone candidate with explicit target and timing assumptions
def evaluate(target: CIMTarget, schedule: Schedule, workload: Workload,
             fetch: WeightFetch = None, *, options: TimingOptions = TimingOptions()) -> Evaluation:
    return Evaluator(target, workload, options=options)(schedule, fetch)



# Count physical work for a legal schedule before timing it
def count_traffic(target, schedule, workload, fetch, policy, inputs, pack, useful_positions, footprint):
    l1, l2 = schedule.l1, schedule.l2
    operations = prod(l1.bounds) * prod(l2.bounds)
    outputs = prod(level.bound(loop) for level in (l1, l2) for loop in ("OX", "OY", "OC"))
    reductions = prod(level.bound(loop) for level in (l1, l2) for loop in REDUCTIONS)
    local_outputs = outputs // footprint * min(footprint, target.local_accum_contexts)
    buffer_outputs = outputs - local_outputs
    banked = target.double_buffered_accum and schedule.write_output_to_accum_buffer
    loads = policy.full_set_loads
    weight_requests = loads * (fetch.valid_columns if fetch.transpose else fetch.source_rows)
    weight_beats = weight_requests * ceil_div(fetch.burst_bytes * 8, target.oc_port_bits)
    unique_weights = prod(level.bound(loop) for level in (l1, l2) for loop in WEIGHTS)
    input_requests = inputs.requests
    input_beats = input_requests * ceil_div(target.k * pack * target.input_bits, target.ic_port_bits)
    output_beats = outputs * ceil_div(target.n * target.accum_bits, target.oc_port_bits) if workload.output_to_memory else 0
    bias_requests = prod(level.bound("OC") for level in (l1, l2)) * l2.bound("OX") * l2.bound("OY")
    bias_requests *= prod(l1.bound(loop) for loop in SPATIAL if l1.inside("OC", loop))
    bias_requests = bias_requests if workload.has_bias else 0
    bias_beats = bias_requests * ceil_div(target.n * target.accum_bits, target.oc_port_bits)
    useful_macs = useful_positions * l1.bound("IC") * l2.bound("IC") * l1.bound("OC") * l2.bound("OC") * fetch.source_rows * fetch.valid_columns
    useful_macs *= workload.channel_useful_fraction
    # Replays of a singleton retain the selection; all other runs advance the ring
    selected_runs = policy.set_uses if policy.fits and policy.sequence_sets > 1 else loads
    switches = max(0, selected_runs - 1) if target.b_sets > 1 else 0
    return Traffic(
        descriptors=policy.descriptors, full_set_loads=loads, set_uses=policy.set_uses,
        resident_set_switches=switches,
        weight_unique_useful_bytes=unique_weights * fetch.source_rows * fetch.valid_columns * target.weight_bits // 8,
        weight_fetched_useful_bytes=loads * fetch.source_rows * fetch.valid_columns * target.weight_bits // 8,
        weight_requested_bytes=weight_requests * fetch.burst_bytes,
        weight_external_beats=weight_beats, weight_external_transfer_bytes=weight_beats * target.oc_port_bits // 8,
        b_writes=loads * target.b_writes_per_set, b_write_bytes=loads * target.k * target.n * target.weight_bits // 8,
        a_beats=operations, a_bytes=operations * target.k * target.input_bits // 8,
        c_beats=operations, c_bytes=operations * target.n * target.accum_bits // 8,
        compute_scalar_macs=operations * target.k * target.n, useful_scalar_macs=useful_macs,
        local_accum_updates=local_outputs * reductions,
        local_accum_intermediate_reads=local_outputs * (reductions - 1),
        local_accum_intermediate_writes=local_outputs * (reductions - 1),
        buffer_accum_reads=buffer_outputs * (reductions - 1),
        buffer_accum_intermediate_writes=buffer_outputs * (reductions - 1),
        buffer_accum_final_writes=outputs if banked else 0,
        direct_output_vectors=0 if banked else outputs, input_buffer_reads=operations,
        input_buffer_writes=inputs.writes,
        input_requests=input_requests, input_requested_bytes=input_requests * target.k * pack * target.input_bits // 8,
        input_external_beats=input_beats, input_external_transfer_bytes=input_beats * target.ic_port_bits // 8,
        output_external_beats=output_beats, output_external_transfer_bytes=output_beats * target.oc_port_bits // 8,
        bias_requests=bias_requests, bias_external_beats=bias_beats,
        bias_external_transfer_bytes=bias_beats * target.oc_port_bits // 8,
        buffer_output_reads=outputs if banked else 0, vector_output_vectors=0 if workload.output_to_memory else outputs,
    )
