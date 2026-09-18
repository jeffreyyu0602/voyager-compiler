# Estimate CIM runtime from work counts and resident-weight boundaries
from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from typing import Tuple
from ..schedule import REDUCTIONS
from ..timing.transfer import ceil_div, transfer_cycles, stream_fill
from ..timing.buffers import buffer_completion
from ..timing.overlap import BufferSlots, OverlapTiming, TimingBudgetExceeded
from .bias import bias_timing
from .output import OutputOptions, output_timing

# Configure external service and optional SRAM feedback constraints
@dataclass(frozen=True)
class TimingOptions(OutputOptions):
    memory_request_latency: int = 0
    output_cycles_per_vector: int = 1
    # Zero leaves SRAM feedback safety unvalidated until an HLS latency is supplied
    accumulation_feedback_cycles: int = 0
    # Current II=1 controller: three cycles from release to reuse and fill to first issue
    # These are scheduled control delays, independently overridable through timing_options
    weight_release_cycles: int = 3
    weight_ready_cycles: int = 3

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
    readiness: dict = field(default_factory=dict)
    assumptions: Tuple[str, ...] = (
        "resident-set readiness and release use bounded repeated-state timing; other interfaces use overlapping resource work",
        "one beat per cycle plus specified request latency; double-buffered input prefetch",
        "result-slot capacity bounds average issue rate; final-reduction bursts use a bandwidth and effective-capacity envelope",
        "output bursts preserve backlog with a bandwidth and effective-capacity equation; loop repetitions compose algebraically",
        "output capacity counts exported design-defined storage; HLS-inserted registers and unspecified stage delays are omitted",
        "SRAM reads and writes overlap accumulation at one vector per cycle per port; no dependency waits",
        "schedule and local contexts must cover SRAM feedback latency; RAW safety is not validated",
        "weight fills keep steady transfer throughput; explicit release and ready delays model resident-slot turnaround",
        "default weight control delays describe the current synthesized controller and can be overridden",
        "bias bandwidth follows first-reduction demand and one complete prefetched vector",
        "transposed weight sets gather then emit; single-set loads serialize with compute",
        "input bank reuse uses repeated two-bank timing; variable fills use the reported maximum-fill bound",
        "SRAM feedback spacing is reported; a supplied feedback latency identifies unsafe schedules",
        "banked output drain uses repeated two-bank timing",
        "interacting operand waits and output backlog compose at burst and loop boundaries",
        "bounded schedule evaluation falls back to independent envelopes for unrecognized long transients",
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
    latency = target.macro_result_latency
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
    first_weight_ready = first_weight + options.weight_ready_cycles
    startup = max(inputs.first_fill_cycles, first_weight_ready)
    sequence_sets = policy.sequence_sets if policy.fits else 1
    replays = policy.compute_replays if policy.fits else 1
    sequence_count = policy.sequence_count if policy.fits else policy.full_set_loads
    weight_finish, weight_steps, weight_bound = buffer_completion(
        sequence_sets, replays, sequence_count, target.b_sets, weight_fill_cycles,
        options.weight_ready_cycles, policy.macs_per_set_use * issue_interval,
        inputs.first_fill_cycles, release_delay=options.weight_release_cycles,
        load_start=max(0, first_weight - weight_fill_cycles))
    weight_issue = weight_finish - startup
    # Uniform bank fills are exact here; irregular boundaries use a visible maximum-fill bound
    input_finish, input_steps, input_bound = buffer_completion(
        1, 1, inputs.fills, 2, inputs.max_fill_cycles, 0,
        traffic.a_beats // inputs.fills * issue_interval, startup)
    input_issue = input_finish - startup
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
    bias_readiness = {}
    if workload.has_bias:
        levels = tuple(tuple((loop, level.bound(loop)) for loop in level.order)
                       for level in (schedule.l1, schedule.l2))
        bias, bias_readiness = bias_timing(target, *levels, interval=issue_interval,
                                          request_latency=options.memory_request_latency)
        resource_cycles['bias'] = max(total_bias_cycles, bias.producer_cycles)
    # Keep the last output vector or completed bank in the non-overlapped drain
    banked = target.double_buffered_accum and schedule.write_output_to_accum_buffer
    final_vectors = prod(schedule.l1.bound(loop) for loop in ("OX", "OY", "OC")) if banked else 1
    final_output = min(outputs, final_vectors) * output_cycles_per_vector
    resource_cycles['output'] = max(0, total_output_cycles - final_output)
    output_bound = False
    output_readiness = {}
    if not banked:
        loops = tuple((loop, level.bound(loop)) for level in (schedule.l1, schedule.l2) for loop in level.order)
        stream, output_readiness = output_timing(target, loops, output_cycles_per_vector,
                                               interval=issue_interval, direct=workload.output_to_memory)
        resource_cycles['issue'] = max(resource_cycles['issue'], stream.producer_cycles)
    if banked:
        blocks = outputs // final_vectors
        bank_finish, _, output_bound = buffer_completion(
            1, 1, blocks, 2, traffic.a_beats // blocks * issue_interval,
            0, final_vectors * output_cycles_per_vector, 0)
        resource_cycles['output'] = max(resource_cycles['output'], bank_finish - final_output)
    drain = latency + final_output
    runtime = startup + max(resource_cycles.values()) + drain
    coupled_readiness = {}
    interacting = sum(wait > 0 for wait in (
        weight_issue - total_issue_cycles, input_issue - total_issue_cycles,
        bias_readiness.get('bias_wait_cycles', 0), output_readiness.get('output_stall_cycles', 0))) > 1
    if not banked and interacting:
        coupled = coupled_timing(target, schedule, policy, inputs, interval=issue_interval,
                                 fill=weight_fill_cycles, load_start=max(0, first_weight - weight_fill_cycles),
                                 output_cycles_per_vector=output_cycles_per_vector, output_capacity_vectors=output_readiness['output_capacity_vectors'],
                                 options=options, has_bias=workload.has_bias)
        coupled_readiness['coupled_timing_limit'] = coupled is None
        if coupled is not None:
            finish, output_finish, waits, steps = coupled
            runtime = max(runtime, finish + drain, output_finish + latency)
            coupled_readiness.update(coupled_burst_steps=steps,
                                     coupled_weight_wait_cycles=waits[0], coupled_input_wait_cycles=waits[1],
                                     coupled_bias_wait_cycles=waits[2], coupled_output_stall_cycles=waits[3])
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
        readiness=dict(**output_readiness, **bias_readiness, **coupled_readiness, weight_wait_cycles=max(0, weight_issue - total_issue_cycles),
                       weight_fill_cycles=weight_fill_cycles,
                       weight_ready_cycles=options.weight_ready_cycles,
                       weight_release_cycles=options.weight_release_cycles,
                       weight_sequence_steps=weight_steps, weight_serialized_bound=weight_bound,
                       input_wait_cycles=max(0, input_issue - total_issue_cycles),
                       input_sequence_steps=input_steps,
                       input_max_fill_bound=inputs.max_fill_cycles != inputs.min_fill_cycles,
                       input_serialized_bound=input_bound,
                       accumulation_feedback_spacing_cycles=spacing,
                       accumulation_feedback_safe=feedback_safe,
                       output_bank_serialized_bound=output_bound),
    )


# Cache active loops as (bound, reduction, resident replay, bias reuse)
@lru_cache(maxsize=4096)
def _completion(l1, l2, reuse, bias_requests, interval, output_cycles_per_vector, output_capacity_vectors,
                capacity, fill, ready, release, load_start, input_fill, input_first, bias_cycles_per_vector):
    try:
        weights = BufferSlots(capacity, fill, ready_delay=ready, release_delay=release, load_start=load_start)
        inputs = BufferSlots(2, input_fill, first_fill_cycles=input_first)
        pipeline = OverlapTiming((weights, inputs), output_cycles_per_vector, output_capacity_vectors)
        loops = l1 + ((0, False, False, False),) + l2

        # Split only first/final reduction and resident-replay phases; repeat the middle algebraically
        def visit(depth, first=True, final=True, load=True, release=True, bias_first=True):
            if depth < 0:
                slot = pipeline.acquire(0, load=load)
                pipeline.produce(reuse * interval, reuse if final else 0,
                                 requests=bias_requests if first and bias_first else 0,
                                 request_cycles=bias_cycles_per_vector)
                if release:
                    pipeline.release(0, slot)
                return
            bound, reduction, replay, bias_reuse = loops[depth]
            if bound == 0:
                slot = pipeline.acquire(1)
                visit(depth - 1, first, final, load, release, bias_first)
                pipeline.release(1, slot)
                return
            position = weights.position

            # A replay traverses the same resident slots while time and other streams advance
            def body(at_first=True, at_last=True):
                if replay:
                    weights.position = position
                visit(depth - 1, first and (not reduction or at_first), final and (not reduction or at_last),
                      load and (not replay or at_first), release and (not replay or at_last),
                      bias_first and (not bias_reuse or at_first))

            if reduction or replay or bias_reuse:
                body(True, False)
                pipeline.repeat(bound - 2, lambda: body(False, False))
                body(False, True)
            else:
                pipeline.repeat(bound, body)

        visit(len(loops) - 1)
        return pipeline.producer_at, pipeline.consumer_at, tuple(pipeline.waits), pipeline.steps
    except TimingBudgetExceeded:
        return None


# Collapse consecutive spatial reuse into one burst and mark input-bank and residency boundaries
def coupled_timing(target, schedule, policy, inputs, *, interval, fill, load_start,
                   output_cycles_per_vector, output_capacity_vectors, options, has_bias):
    l1, l2 = schedule.l1, schedule.l2
    active_weights = [loop for loop in ('IC', 'OC', 'FX', 'FY') if l1.bound(loop) > 1]
    reuse = {loop for loop in ('OX', 'OY') if all(l1.inside(loop, weight) for weight in active_weights)}
    bias_reuse = {loop for loop in ('OX', 'OY') if l1.inside(loop, 'OC')}
    levels = []
    for index, (level, reader) in enumerate(((l1, policy.reader_l1), (l2, policy.reader_l2))):
        levels.append(tuple((level.bound(loop), loop in ('IC', 'FX', 'FY'),
                             policy.fits and reader.bound(loop) != level.bound(loop),
                             index == 0 and loop in bias_reuse)
                            for loop in level.order if level.bound(loop) > 1 and not (index == 0 and loop in reuse)))
    output_vectors_per_burst = prod(l1.bound(loop) for loop in reuse)
    bias_requests = prod(l1.bound(loop) for loop in reuse - bias_reuse) if has_bias else 0
    bias_cycles_per_vector = (target.n * target.accum_bits + target.oc_port_bits - 1) // target.oc_port_bits
    return _completion(*levels, output_vectors_per_burst, bias_requests, interval, output_cycles_per_vector, output_capacity_vectors,
                       min(target.b_sets, policy.full_set_loads), fill,
                       options.weight_ready_cycles, options.weight_release_cycles, load_start,
                       inputs.max_fill_cycles, inputs.first_fill_cycles,
                       bias_cycles_per_vector + options.memory_request_latency)
