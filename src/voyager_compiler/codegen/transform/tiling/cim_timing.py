"""CIM command timing from controller traffic and loop order.

Operand buffers, resident weight sets, and output storage overlap at loop
boundaries. Long repeated schedules use bounded state tracking; an unresolved
transient falls back to a conservative bound. DRAM timing remains in the
shared runtime calculator.
"""

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import prod

import interstellar
from voyager_compiler.codegen.transform.tiling.timing.bias import bias_timing
from voyager_compiler.codegen.transform.tiling.timing.buffers import (
    buffer_completion,
)
from voyager_compiler.codegen.transform.tiling.timing.input import (
    input_bank_traffic,
)
from voyager_compiler.codegen.transform.tiling.timing.output import (
    OutputOptions,
    output_timing,
)
from voyager_compiler.codegen.transform.tiling.timing.overlap import (
    BufferSlots,
    OverlapTiming,
    TimingBudgetExceeded,
)
from voyager_compiler.codegen.transform.tiling.timing.transfer import (
    ceil_div,
    packing_factor,
    stream_fill,
    transfer_cycles,
)

le = interstellar.le
REDUCTIONS = ("IC", "FX", "FY")


@dataclass(frozen=True)
class TimingOptions(OutputOptions):
    """Interface timing assumptions for standalone CIM estimates."""

    memory_request_latency: int = 0
    output_cycles_per_vector: int = 1
    # Zero leaves SRAM feedback safety unvalidated until an HLS latency is supplied.
    accumulation_feedback_cycles: int = 0
    # Scheduled II=1 controller delays from release to reuse and fill to first issue.
    weight_release_cycles: int = 3
    weight_ready_cycles: int = 3

    def __post_init__(self):
        super().__post_init__()
        if (
            type(self.output_cycles_per_vector) is not int
            or self.output_cycles_per_vector <= 0
        ):
            raise ValueError(
                "output_cycles_per_vector must be a positive integer"
            )
        for name in (
            "memory_request_latency",
            "accumulation_feedback_cycles",
            "weight_release_cycles",
            "weight_ready_cycles",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class Traffic:
    """Physical transfers and accumulation accesses for one matrix command."""

    input_requests: int
    input_external_beats: int
    input_buffer_writes: int
    weight_requests: int
    weight_external_beats: int
    weight_write_beats: int
    bias_requests: int
    bias_external_beats: int
    output_vectors: int
    local_accum_updates: int
    buffer_accum_reads: int
    buffer_accum_intermediate_writes: int
    buffer_accum_final_writes: int
    buffer_output_reads: int


@dataclass(frozen=True)
class TimingEstimate:
    """Estimated command duration, resource demand, and remaining stalls."""

    runtime_cycles: int
    compute_cycles: int
    startup_cycles: int
    drain_cycles: int
    traffic: Traffic | None
    resource_cycles: dict
    readiness: dict = field(default_factory=dict)


def _levels(mapping):
    """L1 and L2 bounds in inner-to-outer order, excluding the unit batch loop."""
    return tuple(
        tuple(
            (le.table[d], mapping.loop_blockings[d][level])
            for d in sorted(
                (d for d in range(le.NUM) if d != le.ON),
                key=lambda d: mapping.loop_orders[d][level],
            )
        )
        for level in (1, 2)
    )


# Count intervening output updates before the innermost active reduction advances.
def feedback_spacing(levels):
    spacing = 1
    for level in levels:
        for loop, bound in level:
            if bound > 1 and loop in REDUCTIONS:
                return spacing
            spacing *= bound
    return 0


def estimate_cycles(rc, mapping, work, bank_groups, interval):
    """Estimate command timing for the selected operand widths and fused tail."""
    config, options = rc.config, rc.timing_options
    levels = _levels(mapping)
    l1, l2 = (dict(level) for level in levels)
    ic, oc = config.pe_array_size
    port = rc.sram_bandwidth
    latency = interval + config.cim_mac_latency - 1
    # CIMTile registers A before issue and C after the element's retirement.
    latency += 2
    compute = work.mac_requests * interval
    # Each reduced MAC reserves one result slot per output tile. Bound
    # the sustained issue rate by the slots occupied until its return.
    issue_interval = max(
        interval, ceil_div(latency, config.cim_array_result_slots)
    )
    total_issue_cycles = work.mac_requests * issue_interval

    input_pack = packing_factor(ic * rc.input_dtype_width, port, l1["IC"])
    input_key = (
        rc.input_dtype_width,
        port,
        rc.stride,
        options.memory_request_latency,
        tuple(
            mapping.loop_blockings[d][level]
            for level in (1, 2)
            for d in range(le.NUM)
        ),
    )
    if input_key not in rc._input_cache:
        inputs, reasons = input_bank_traffic(
            l1,
            l2,
            input_x=(l1["OX"] * l2["OX"] - 1) * rc.stride[1] + l1["FX"],
            input_y=(l1["OY"] * l2["OY"] - 1) * rc.stride[0] + l1["FY"],
            stride=rc.stride[0],
            lanes=ic,
            element_bits=rc.input_dtype_width,
            port_bits=port,
            pack=input_pack,
            request_latency=options.memory_request_latency,
        )
        if reasons:
            return TimingEstimate(
                math.inf,
                compute,
                0,
                0,
                None,
                {},
                {"input_reasons": tuple(reasons)},
            )
        rc._input_cache[input_key] = inputs
    inputs = rc._input_cache[input_key]

    policy = work.policy
    weight_pack = packing_factor(oc * rc.weight_dtype_width, port, l1["OC"])
    weight_bits = oc * rc.weight_dtype_width * weight_pack
    weight_row_cycles = transfer_cycles(
        weight_bits, port, options.memory_request_latency
    )
    spans = config.cim_output_axis_tiles // config.cim_b_port_tiles
    first_weight, steady_fill_cycles = stream_fill(
        ic, weight_row_cycles, ic, spans
    )
    weight_fill_cycles = (
        steady_fill_cycles if config.cim_weight_sets > 1 else first_weight
    )
    total_weight_cycles = policy.full_set_loads * weight_fill_cycles

    output_width = (
        rc.accum_dtype_width
        if mapping.loop_blockings[le.IC][3] > 1
        else rc.output_dtype_width
    )
    banked = rc.uses_banked_output(mapping)
    bias_requests = (
        l1["OC"] * l2["OC"] * l2["OX"] * l2["OY"] if rc.bias_width else 0
    )
    inner_order = tuple(loop for loop, _ in levels[0])
    bias_requests *= prod(
        l1[loop]
        for loop in ("OX", "OY")
        if inner_order.index("OC") < inner_order.index(loop)
    )
    local_outputs = (
        work.output_vectors
        - work.output_vectors
        // work.local_contexts
        * max(0, work.local_contexts - config.cim_local_accum_contexts)
    )
    traffic = Traffic(
        input_requests=inputs.requests,
        input_external_beats=inputs.requests
        * ceil_div(ic * input_pack * rc.input_dtype_width, port),
        input_buffer_writes=inputs.writes,
        weight_requests=policy.full_set_loads * ic,
        weight_external_beats=policy.full_set_loads
        * ic
        * ceil_div(weight_bits, port),
        weight_write_beats=work.weight_write_beats,
        bias_requests=bias_requests,
        bias_external_beats=bias_requests * ceil_div(oc * rc.bias_width, port),
        output_vectors=work.output_vectors,
        local_accum_updates=local_outputs
        * work.mac_requests
        // work.output_vectors,
        buffer_accum_reads=work.buffer_partial_updates,
        buffer_accum_intermediate_writes=work.buffer_partial_updates,
        buffer_accum_final_writes=work.output_vectors if banked else 0,
        buffer_output_reads=work.output_vectors if banked else 0,
    )

    # Use the shared runtime contract for split reductions and separate tails.
    output_elems = work.output_vectors * oc
    if mapping.loop_blockings[le.IC][3] > 1:
        output_words = {
            "scratch": 2 * rc._bus_words(output_elems, rc.accum_dtype_width)
        }
    elif rc.single_k_tail_extra_pass and not rc.tail_keeps_shape:
        output_words = {
            "scratch": rc._bus_words(output_elems, rc.output_dtype_width)
        }
    else:
        output_words = rc._tail_words(mapping, 2)
        if rc.single_k_tail_extra_pass:
            output_words["output"] += 2 * rc._bus_words(
                output_elems, rc.output_dtype_width
            )
    output_cycles_per_vector = max(
        options.output_cycles_per_vector,
        ceil_div(output_width * oc, port),
        math.ceil(
            rc._bank_cycles(output_words, bank_groups) / work.output_vectors
        ),
    )
    streaming_tail = (
        rc.has_tail
        and mapping.loop_blockings[le.IC][3] == 1
        and not rc.single_k_tail_extra_pass
    )
    if streaming_tail:
        output_cycles_per_vector = max(
            output_cycles_per_vector, ceil_div(oc, config.vector_lanes)
        )
        for dims, bits in rc.tail_specs:
            output_cycles_per_vector = max(
                output_cycles_per_vector,
                ceil_div(bits * (oc if le.OC in dims else 1), port),
            )
    if options.output_pipeline is not None:
        options.output_pipeline.validate_geometry(
            oc, config.vector_lanes, rc.accum_dtype_width, port
        )
        if (
            not streaming_tail
            or options.output_pipeline.output_bits != rc.output_dtype_width
        ):
            options = replace(options, output_pipeline=None)

    outputs = work.output_vectors
    total_output_cycles = outputs * output_cycles_per_vector
    total_bias_cycles = (
        traffic.bias_external_beats
        + bias_requests * options.memory_request_latency
    )
    sram_reads = traffic.buffer_accum_reads + traffic.buffer_output_reads
    sram_writes = (
        traffic.buffer_accum_intermediate_writes
        + traffic.buffer_accum_final_writes
    )
    # II=1 accumulation overlaps independent DualPortBuffer reads and writes.
    # Feedback spacing is a schedule assumption, not a hardware dependency stall.
    total_accumulation_cycles = max(work.mac_requests, sram_reads, sram_writes)
    first_weight_ready = first_weight + options.weight_ready_cycles
    startup = max(inputs.first_fill_cycles, first_weight_ready)
    sequence_sets = policy.sequence_sets if policy.fits else 1
    replays = policy.compute_replays if policy.fits else 1
    sequence_count = (
        policy.sequence_count if policy.fits else policy.full_set_loads
    )
    weight_finish, weight_steps, weight_bound = buffer_completion(
        sequence_sets,
        replays,
        sequence_count,
        config.cim_weight_sets,
        weight_fill_cycles,
        options.weight_ready_cycles,
        policy.macs_per_set_use * issue_interval,
        inputs.first_fill_cycles,
        release_delay=options.weight_release_cycles,
        load_start=max(0, first_weight - weight_fill_cycles),
    )
    weight_issue = weight_finish - startup
    # Uniform bank fills are exact here; irregular boundaries use a visible maximum-fill bound.
    input_finish, input_steps, input_bound = buffer_completion(
        1,
        1,
        inputs.fills,
        2,
        inputs.max_fill_cycles,
        0,
        work.mac_requests // inputs.fills * issue_interval,
        startup,
    )
    input_issue = input_finish - startup
    spacing = (
        feedback_spacing(levels) * issue_interval
        if traffic.buffer_accum_reads
        else 0
    )
    feedback_safe = (
        True
        if not traffic.buffer_accum_reads
        else (
            spacing >= options.accumulation_feedback_cycles
            if options.accumulation_feedback_cycles
            else None
        )
    )
    resource_cycles = dict(
        issue=max(total_issue_cycles, weight_issue),
        input=max(0, input_issue),
        accumulation=total_accumulation_cycles,
        bias=total_bias_cycles,
    )
    bias_readiness = {}
    if rc.bias_width:
        bias, bias_readiness = bias_timing(
            *levels,
            vector_bits=oc * rc.bias_width,
            port_bits=port,
            interval=issue_interval,
            request_latency=options.memory_request_latency,
        )
        resource_cycles["bias"] = max(total_bias_cycles, bias.producer_cycles)
    # Keep the last output vector or completed bank in the non-overlapped drain.
    final_vectors = (
        prod(l1[loop] for loop in ("OX", "OY", "OC")) if banked else 1
    )
    final_output = min(outputs, final_vectors) * output_cycles_per_vector
    resource_cycles["output"] = max(0, total_output_cycles - final_output)
    output_bound = False
    output_readiness = {}
    if not banked:
        # Only final-output storage absorbs a finished reduction's burst.
        # These eight vectors are CIMProcessor::OUTPUT_FIFO_DEPTH; metadata
        # and intermediate accumulation queues are not output capacity.
        stream, output_readiness = output_timing(
            levels[0] + levels[1],
            output_cycles_per_vector,
            elements_per_vector=oc,
            output_storage={"matrix_output": oc * 8, "vector_pipeline": 0},
            interval=issue_interval,
            direct=not streaming_tail,
            options=options,
        )
        resource_cycles["issue"] = max(
            resource_cycles["issue"], stream.producer_cycles
        )
    if banked:
        blocks = outputs // final_vectors
        bank_finish, _, output_bound = buffer_completion(
            1,
            1,
            blocks,
            2,
            work.mac_requests // blocks * issue_interval,
            0,
            final_vectors * output_cycles_per_vector,
            0,
        )
        resource_cycles["output"] = max(
            resource_cycles["output"], bank_finish - final_output
        )
    output_forward = output_readiness.get("output_forward_cycles", 0)
    drain = latency + final_output + output_forward
    runtime = startup + max(resource_cycles.values()) + drain
    if not banked:
        # Include the same queued output drain on the independent and coupled paths.
        runtime = max(runtime, startup + stream.consumer_cycles + latency)
    coupled_readiness = {}
    output_wait = max(
        output_readiness.get("output_stall_cycles", 0),
        output_readiness.get("output_backlog_cycles", 0),
    )
    operand_waits = (
        max(0, weight_issue - total_issue_cycles),
        max(0, input_issue - total_issue_cycles),
        bias_readiness.get("bias_wait_cycles", 0),
    )
    interacting = sum(wait > 0 for wait in (*operand_waits, output_wait)) > 1
    if not banked and interacting:
        coupled = coupled_timing(
            config,
            levels,
            policy,
            inputs,
            interval=issue_interval,
            fill=weight_fill_cycles,
            load_start=max(0, first_weight - weight_fill_cycles),
            output_cycles_per_vector=output_cycles_per_vector,
            output_capacity_vectors=output_readiness["output_capacity_vectors"],
            options=options,
            bias_bits=oc * rc.bias_width,
            port_bits=port,
            output_credit_delay=output_readiness["output_credit_delay_cycles"],
        )
        coupled_readiness["coupled_timing_limit"] = coupled is None
        if coupled is None:
            # Unknown overlap must not give an unfinished candidate an optimistic ranking.
            waits = sum(operand_waits) + output_readiness["output_stall_cycles"]
            runtime = max(
                runtime,
                startup
                + total_issue_cycles
                + waits
                + output_readiness["output_backlog_cycles"]
                + latency,
            )
        else:
            finish, output_finish, waits, steps = coupled
            output_finish += output_forward
            runtime = max(runtime, finish + drain, output_finish + latency)
            coupled_readiness.update(
                coupled_burst_steps=steps,
                coupled_weight_wait_cycles=waits[0],
                coupled_input_wait_cycles=waits[1],
                coupled_bias_wait_cycles=waits[2],
                coupled_output_stall_cycles=waits[3],
                coupled_output_backlog_cycles=output_finish - finish,
            )

    # Shared scratchpad roles queue on a bank's one port, even when their
    # controllers otherwise overlap. Retain that total-traffic lower bound.
    bank_words = dict(
        input=traffic.input_external_beats,
        weight=traffic.weight_external_beats,
        bias=traffic.bias_external_beats,
    )
    bank_words.update(output_words)
    bank_cycles = rc._bank_cycles(bank_words, bank_groups)
    runtime = max(runtime, bank_cycles)
    return TimingEstimate(
        runtime_cycles=runtime,
        compute_cycles=compute,
        startup_cycles=startup,
        drain_cycles=drain,
        traffic=traffic,
        resource_cycles=dict(
            compute=compute,
            result_slots=total_issue_cycles,
            input=inputs.total_fill_cycles,
            weight=total_weight_cycles,
            accumulation=total_accumulation_cycles,
            output=total_output_cycles,
            bias=total_bias_cycles,
            scratchpad=bank_cycles,
            accumulation_sram_reads=sram_reads,
            accumulation_sram_writes=sram_writes,
        ),
        readiness=dict(
            **output_readiness,
            **bias_readiness,
            **coupled_readiness,
            weight_wait_cycles=max(0, weight_issue - total_issue_cycles),
            weight_fill_cycles=weight_fill_cycles,
            weight_ready_cycles=options.weight_ready_cycles,
            weight_release_cycles=options.weight_release_cycles,
            weight_sequence_steps=weight_steps,
            weight_serialized_bound=weight_bound,
            input_wait_cycles=max(0, input_issue - total_issue_cycles),
            input_sequence_steps=input_steps,
            input_max_fill_bound=inputs.max_fill_cycles
            != inputs.min_fill_cycles,
            input_serialized_bound=input_bound,
            accumulation_feedback_spacing_cycles=spacing,
            accumulation_feedback_safe=feedback_safe,
            output_bank_serialized_bound=output_bound,
            result_latency_cycles=latency,
        ),
    )


# Cache active loops as (bound, reduction, resident replay, bias reuse)
@lru_cache(maxsize=4096)
def _completion(
    l1,
    l2,
    reuse,
    bias_requests,
    interval,
    output_cycles_per_vector,
    output_capacity_vectors,
    capacity,
    fill,
    ready,
    release,
    load_start,
    input_fill,
    input_first,
    bias_cycles_per_vector,
    output_credit_delay=0,
):
    try:
        weights = BufferSlots(
            capacity,
            fill,
            ready_delay=ready,
            release_delay=release,
            load_start=load_start,
        )
        inputs = BufferSlots(2, input_fill, first_fill_cycles=input_first)
        pipeline = OverlapTiming(
            (weights, inputs),
            output_cycles_per_vector,
            output_capacity_vectors,
            output_credit_delay=output_credit_delay,
        )
        loops = l1 + ((0, False, False, False),) + l2

        # Split only first/final reduction and resident-replay phases; repeat the middle algebraically
        def visit(
            depth,
            first=True,
            final=True,
            load=True,
            release=True,
            bias_first=True,
        ):
            if depth < 0:
                slot = pipeline.acquire(0, load=load)
                pipeline.produce(
                    reuse * interval,
                    reuse if final else 0,
                    requests=bias_requests if first and bias_first else 0,
                    request_cycles=bias_cycles_per_vector,
                )
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
                visit(
                    depth - 1,
                    first and (not reduction or at_first),
                    final and (not reduction or at_last),
                    load and (not replay or at_first),
                    release and (not replay or at_last),
                    bias_first and (not bias_reuse or at_first),
                )

            if reduction or replay or bias_reuse:
                body(True, False)
                pipeline.repeat(bound - 2, lambda: body(False, False))
                body(False, True)
            else:
                pipeline.repeat(bound, body)

        visit(len(loops) - 1)
        return (
            pipeline.producer_at,
            pipeline.consumer_at,
            tuple(pipeline.waits),
            pipeline.steps,
        )
    except TimingBudgetExceeded:
        return None


# Collapse consecutive spatial reuse into one burst and mark input-bank and residency boundaries.
def coupled_timing(
    config,
    levels,
    policy,
    inputs,
    *,
    interval,
    fill,
    load_start,
    output_cycles_per_vector,
    output_capacity_vectors,
    options,
    bias_bits,
    port_bits,
    output_credit_delay=0,
):
    l1 = dict(levels[0])
    order = tuple(loop for loop, _ in levels[0])
    active_weights = [loop for loop in ("IC", "OC", "FX", "FY") if l1[loop] > 1]
    reuse = {
        loop
        for loop in ("OX", "OY")
        if all(
            order.index(loop) < order.index(weight) for weight in active_weights
        )
    }
    bias_reuse = {
        loop for loop in ("OX", "OY") if order.index(loop) < order.index("OC")
    }
    phases = []
    for index, (level, reader) in enumerate(
        zip(levels, (policy.reader_l1, policy.reader_l2))
    ):
        phases.append(
            tuple(
                (
                    bound,
                    loop in REDUCTIONS,
                    policy.fits and reader[getattr(le, loop)] != bound,
                    index == 0 and loop in bias_reuse,
                )
                for loop, bound in level
                if bound > 1 and not (index == 0 and loop in reuse)
            )
        )
    output_vectors_per_burst = prod(l1[loop] for loop in reuse)
    bias_requests = (
        prod(l1[loop] for loop in reuse - bias_reuse) if bias_bits else 0
    )
    bias_cycles_per_vector = ceil_div(bias_bits, port_bits)
    return _completion(
        *phases,
        output_vectors_per_burst,
        bias_requests,
        interval,
        output_cycles_per_vector,
        output_capacity_vectors,
        min(config.cim_weight_sets, policy.full_set_loads),
        fill,
        options.weight_ready_cycles,
        options.weight_release_cycles,
        load_start,
        inputs.max_fill_cycles,
        inputs.first_fill_cycles,
        bias_cycles_per_vector + options.memory_request_latency,
        output_credit_delay,
    )
