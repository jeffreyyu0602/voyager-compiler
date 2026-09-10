# Estimate CIM runtime from work counts and resident-weight boundaries
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import Tuple
from ..schedule import REDUCTIONS
from ..timing.transfer import ceil_div, transfer_cycles, stream_fill

# Configure manually assumed pipeline delays and external service rates
@dataclass(frozen=True)
class TimingOptions:
    memory_request_latency: int = 0
    output_cycles_per_vector: int = 1
    # TODO: Calibrate these assumed stage delays against the selected HLS build
    input_handoff_cycles: int = 1
    result_return_cycles: int = 5
    output_pipeline_cycles: int = 4
    sram_dependency_cycles: int = 1

    # Require a positive consumer service rate
    def __post_init__(self):
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
    assumptions: Tuple[str, ...] = (
        "aggregate service totals overlap across independent input, weight, bias, and output interfaces",
        "one beat per cycle plus specified request latency; double-buffered input prefetch",
        "result-slot capacity bounds average issue rate; no per-operation FIFO simulation",
        "configured SRAM dependency service approximates ordered write/read waits",
        "writes into a single resident CIM set serialize with compute",
        "input packing and bank boundaries use aggregate service estimates",
        "useful work excludes convolution and channel padding; repeated L2 slices use mean useful work",
        "excludes command serialization, HLS-added stages, and downstream vector work",
    )

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
    total_weight_cycles = policy.full_set_loads * first_weight
    output_cycles_per_vector = options.output_cycles_per_vector
    if workload.output_to_memory:
        output_cycles_per_vector = max(output_cycles_per_vector, ceil_div(target.n * target.accum_bits, target.oc_port_bits))
    outputs = traffic.direct_output_vectors + traffic.buffer_output_reads
    total_output_cycles = outputs * output_cycles_per_vector
    total_bias_cycles = traffic.bias_requests * (
        ceil_div(target.n * target.accum_bits, target.oc_port_bits) + options.memory_request_latency)
    # Approximate ordered SRAM dependencies; local feedback needs no extra transfer
    total_accumulation_cycles = traffic.a_beats + traffic.buffer_accum_reads * options.sram_dependency_cycles
    sram_reads = traffic.buffer_accum_reads + traffic.buffer_output_reads
    sram_writes = traffic.buffer_accum_intermediate_writes + traffic.buffer_accum_final_writes
    startup = max(inputs.first_fill_cycles, first_weight) + options.input_handoff_cycles
    remaining_weights = total_weight_cycles - first_weight
    # Writing weights and computing cannot overlap when both use the only resident set
    serialized_weights = remaining_weights if target.b_sets == 1 else 0
    resource_cycles = dict(
        issue=total_issue_cycles + serialized_weights,
        input=max(0, inputs.total_fill_cycles - inputs.first_fill_cycles),
        weight=0 if target.b_sets == 1 else remaining_weights,
        accumulation=total_accumulation_cycles,
        bias=total_bias_cycles,
    )
    # Keep the last output vector or completed bank in the non-overlapped drain
    banked = target.double_buffered_accum and schedule.write_output_to_accum_buffer
    final_vectors = prod(schedule.l1.bound(loop) for loop in ("OX", "OY", "OC")) if banked else 1
    final_output = min(outputs, final_vectors) * output_cycles_per_vector
    resource_cycles['output'] = max(0, total_output_cycles - final_output)
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
    )
