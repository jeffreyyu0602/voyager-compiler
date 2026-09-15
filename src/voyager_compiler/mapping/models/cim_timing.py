# Estimate CIM runtime from work counts and resident-weight boundaries
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import Tuple
from ..schedule import REDUCTIONS
from ..timing.transfer import ceil_div, transfer_cycles, stream_fill
from ..timing.buffers import buffer_completion
from .output import OutputOptions, output_timing

# Configure manually assumed pipeline delays and external service rates
@dataclass(frozen=True)
class TimingOptions(OutputOptions):
    memory_request_latency: int = 0
    output_cycles_per_vector: int = 1
    # TODO: Calibrate these assumed stage delays against the selected HLS build
    input_handoff_cycles: int = 1
    result_return_cycles: int = 5
    output_pipeline_cycles: int = 4
    # Zero leaves SRAM feedback safety unvalidated until an HLS latency is supplied
    accumulation_feedback_cycles: int = 0

    # Require a positive consumer service rate
    def __post_init__(self):
        super().__post_init__()
        if type(self.output_cycles_per_vector) is not int or self.output_cycles_per_vector <= 0:
            raise ValueError("output_cycles_per_vector must be a positive integer")

# Report aggregate stage demand separately from modeled elapsed runtime
@dataclass(frozen=True)
class TimingEstimate:
    runtime_cycles: int
    ideal_cycles: float
    compute_cycles: int
    spatial_utilization: float
    useful_work_fraction: float
    effective_utilization: float
    resource_cycles: dict
    startup_cycles: int
    drain_cycles: int
    options: TimingOptions = TimingOptions()
    readiness: dict = field(default_factory=dict)
    assumptions: Tuple[str, ...] = (
        "resident-set readiness and release use bounded repeated-state timing; other interfaces use overlapping resource work",
        "one beat per cycle plus specified request latency; double-buffered input prefetch",
        "result-slot capacity bounds average issue rate; final-reduction bursts use a bandwidth and effective-capacity envelope",
        "output bursts preserve backlog with a bandwidth and effective-capacity equation; loop repetitions compose algebraically",
        "without an explicit output capacity only the exported FIFO contributes elasticity",
        "SRAM reads and writes overlap accumulation at one vector per cycle per port; no dependency waits",
        "schedule and local contexts must cover SRAM feedback latency; RAW safety is not validated",
        "non-transposed weight streams sustain pipelined loads with explicit first-pass readiness and final-use release",
        "transposed weight sets gather then emit; single-set loads serialize with compute",
        "input bank reuse uses repeated two-bank timing; variable fills use the reported maximum-fill bound",
        "SRAM feedback spacing is reported; a supplied feedback latency identifies unsafe schedules",
        "banked output drain uses repeated two-bank timing",
        "independent readiness envelopes overlap by maximum; cross-interface stall phasing is approximate",
        "useful work excludes convolution and channel padding; repeated L2 slices use mean useful work",
        "excludes command serialization and unprofiled HLS stages; fused epilogue timing is derived from scheduled vector passes",
    )

# Count intervening output updates before the innermost active reduction advances
def feedback_spacing(schedule):
    spacing = 1
    for level in (schedule.l1, schedule.l2):
        for loop in level.order:
            bound = level.bound(loop)
            if bound > 1 and loop in REDUCTIONS:
                return spacing
            spacing *= bound
    return 0

# Overlap independent stage totals and bound issue rate by result-slot capacity
def estimate_cycles(target, schedule, workload, fetch, options, policy, traffic, inputs):
    interval = target.issue_interval
    latency = target.macro_result_latency + options.result_return_cycles
    compute = traffic.a_beats * interval
    # A result occupies a slot until the modeled capture/return latency elapses
    issue_interval = max(interval, ceil_div(latency, target.result_slots_per_output_lane))
    total_issue_cycles = traffic.a_beats * issue_interval
    weight_row_cycles = transfer_cycles(fetch.burst_bytes * 8, target.oc_port_bits, options.memory_request_latency)
    spans = target.output_axis_tiles // target.b_port_tiles
    first_weight, steady_fill_cycles = stream_fill(
        target.n if fetch.transpose else fetch.source_rows, weight_row_cycles,
        target.k, spans,
        blocking=fetch.transpose)
    weight_fill_cycles = steady_fill_cycles if not fetch.transpose and target.b_sets > 1 else first_weight
    total_weight_cycles = policy.full_set_loads * weight_fill_cycles
    output_cycles_per_vector = options.output_cycles_per_vector
    if workload.output_to_memory:
        output_cycles_per_vector = max(output_cycles_per_vector, ceil_div(target.n * target.accum_bits, target.oc_port_bits))
    outputs = traffic.direct_output_vectors + traffic.buffer_output_reads
    total_output_cycles = outputs * output_cycles_per_vector
    total_bias_cycles = traffic.bias_requests * (
        ceil_div(target.n * target.accum_bits, target.oc_port_bits) + options.memory_request_latency)
    sram_reads = traffic.buffer_accum_reads + traffic.buffer_output_reads
    sram_writes = traffic.buffer_accum_intermediate_writes + traffic.buffer_accum_final_writes
    # II=1 accumulation overlaps independent DualPortBuffer reads and writes
    # Feedback spacing is a schedule assumption, not a hardware dependency stall
    total_accumulation_cycles = max(traffic.a_beats, sram_reads, sram_writes)
    startup = max(inputs.first_fill_cycles, first_weight) + options.input_handoff_cycles
    sequence_sets = policy.sequence_sets if policy.fits else 1
    replays = policy.compute_replays if policy.fits else 1
    sequence_count = policy.sequence_count if policy.fits else policy.full_set_loads
    weight_finish, weight_steps, weight_bound = buffer_completion(
        sequence_sets, replays, sequence_count, target.b_sets, weight_fill_cycles,
        max(0, first_weight - weight_fill_cycles), policy.macs_per_set_use * issue_interval,
        inputs.first_fill_cycles)
    weight_issue = weight_finish - max(inputs.first_fill_cycles, first_weight)
    # Uniform bank fills are exact here; irregular boundaries use a visible maximum-fill bound
    input_finish, input_steps, input_bound = buffer_completion(
        1, 1, inputs.fills, 2, inputs.max_fill_cycles, 0,
        traffic.a_beats // inputs.fills * issue_interval, max(first_weight, inputs.first_fill_cycles))
    input_issue = input_finish - max(inputs.first_fill_cycles, first_weight)
    spacing = feedback_spacing(schedule) * issue_interval if traffic.buffer_accum_reads else 0
    feedback_safe = (True if not traffic.buffer_accum_reads else
                     spacing >= options.accumulation_feedback_cycles
                     if options.accumulation_feedback_cycles else None)
    resource_cycles = dict(
        issue=max(total_issue_cycles, weight_issue),
        input=max(0, input_issue),
        accumulation=total_accumulation_cycles,
        bias=total_bias_cycles,
    )
    # Keep the last output vector or completed bank in the non-overlapped drain
    banked = target.double_buffered_accum and schedule.write_output_to_accum_buffer
    final_vectors = prod(schedule.l1.bound(loop) for loop in ("OX", "OY", "OC")) if banked else 1
    final_output = min(outputs, final_vectors) * output_cycles_per_vector
    resource_cycles['output'] = max(0, total_output_cycles - final_output)
    output_bound = False
    output_readiness = {}
    if not banked:
        loops = tuple((loop, level.bound(loop)) for level in (schedule.l1, schedule.l2) for loop in level.order)
        stream, output_readiness = output_timing(target, options, loops, output_cycles_per_vector,
                                               interval=issue_interval, direct=workload.output_to_memory)
        resource_cycles['issue'] = max(resource_cycles['issue'], stream.producer_cycles)
    if banked:
        blocks = outputs // final_vectors
        bank_finish, _, output_bound = buffer_completion(
            1, 1, blocks, 2, traffic.a_beats // blocks * issue_interval,
            0, final_vectors * output_cycles_per_vector, 0)
        resource_cycles['output'] = max(resource_cycles['output'], bank_finish - final_output)
    drain = latency + options.output_pipeline_cycles + final_output
    runtime = startup + max(resource_cycles.values()) + drain
    ideal = traffic.useful_scalar_macs * interval / (target.k * target.n)
    return TimingEstimate(
        runtime_cycles=runtime, ideal_cycles=ideal, compute_cycles=compute,
        spatial_utilization=1.0,
        useful_work_fraction=traffic.useful_scalar_macs / traffic.compute_scalar_macs,
        effective_utilization=ideal / runtime,
        resource_cycles=dict(compute=compute, result_slots=total_issue_cycles,
                            input=inputs.total_fill_cycles, weight=total_weight_cycles,
                            accumulation=total_accumulation_cycles, output=total_output_cycles, bias=total_bias_cycles,
                            accumulation_sram_reads=sram_reads, accumulation_sram_writes=sram_writes),
        startup_cycles=startup, drain_cycles=drain, options=options,
        readiness=dict(**output_readiness, weight_wait_cycles=max(0, weight_issue - total_issue_cycles),
                       weight_sequence_steps=weight_steps, weight_serialized_bound=weight_bound,
                       input_wait_cycles=max(0, input_issue - total_issue_cycles),
                       input_sequence_steps=input_steps,
                       input_max_fill_bound=inputs.max_fill_cycles != inputs.min_fill_cycles,
                       input_serialized_bound=input_bound,
                       accumulation_feedback_spacing_cycles=spacing,
                       accumulation_feedback_safe=feedback_safe,
                       output_bank_serialized_bound=output_bound),
    )
