"""CIM command timing from controller traffic and loop order.

Operand buffers, resident weight sets, and output storage are priced with
closed formulas. When waits interact, add their costs and the output backlog
conservatively rather than walking the loop nest. DRAM timing remains in the
shared runtime calculator.
"""

import math
from dataclasses import dataclass, field, replace
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
            input_y=(l1["OY"] * l2["OY"] - 1) * rc.stride[0]
            + l1["FY"] * l2["FY"],
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
        work.buffer_feedback_spacing * issue_interval
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
        # Include queued output work even when another resource dominates.
        runtime = max(runtime, startup + stream.consumer_cycles + latency)
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
    additive_cycles = 0
    if not banked and interacting:
        # Price interacting waits serially. This is the additive estimate
        # used by the previous loop walk's conservative fallback.
        waits = sum(operand_waits) + output_readiness["output_stall_cycles"]
        additive_cycles = (
            startup
            + total_issue_cycles
            + waits
            + output_readiness["output_backlog_cycles"]
            + latency
        )
        runtime = max(runtime, additive_cycles)

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
            additive_wait_bound_cycles=additive_cycles,
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
