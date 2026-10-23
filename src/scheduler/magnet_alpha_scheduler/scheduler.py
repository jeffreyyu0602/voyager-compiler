# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deterministic, single-call GEMM scheduling for Magnet Alpha.

The scheduler consumes only frozen scheduler IR and the declared machine.  It
does not inspect FX, allocate storage, or serialize collateral.  Its result is
a canonical three-level ``MappingPoint`` wrapped in ``LocalScheduleResult``.
"""

import math
from dataclasses import replace

from scheduler.magnet_alpha_scheduler.factors import (
    mapping_point_to_alpha_factors,
)
from scheduler.types import (
    NUM_LOOPS,
    GemmWorkload,
    LocalScheduleResult,
    Loop,
    MappingPoint,
)

MAGNET_ALPHA_RULE_BASED_SCHEDULER = "magnet_alpha_rule_based"


class MagnetAlphaSchedulingError(ValueError):
    """The requested call cannot be represented by Alpha's GEMM ABI."""


def _encode_loop_orders(
    loop_blockings,
    loop_partitionings,
    preferred_orders,
) -> tuple[tuple[int, ...], ...]:
    blockings = tuple(tuple(row) for row in loop_blockings)
    partitionings = tuple(tuple(row) for row in loop_partitionings)
    num_levels = len(blockings[0])
    if len(blockings) != NUM_LOOPS or len(partitionings) != NUM_LOOPS:
        raise ValueError("loop factors need one row per canonical loop")
    if any(len(row) != num_levels for row in blockings + partitionings):
        raise ValueError("loop factor rows must have equal length")
    if len(preferred_orders) != num_levels:
        raise ValueError("preferred loop orders must match mapping levels")

    level_orders = []
    for level, preferred in enumerate(preferred_orders):
        preferred = tuple(Loop(loop) for loop in preferred)
        if len(set(preferred)) != len(preferred):
            raise ValueError(f"loop order at level {level} has duplicates")
        active = {
            Loop(loop)
            for loop in range(NUM_LOOPS)
            if blockings[loop][level] != 1
            or partitionings[loop][level] != 1
        }
        missing = active - set(preferred)
        if missing:
            names = ", ".join(sorted(loop.name for loop in missing))
            raise ValueError(f"loop order at level {level} omits {names}")
        encoded = [NUM_LOOPS - 1] * NUM_LOOPS
        for order, loop in enumerate(loop for loop in preferred if loop in active):
            encoded[loop] = order
        level_orders.append(encoded)
    return tuple(
        tuple(level[loop] for level in level_orders)
        for loop in range(NUM_LOOPS)
    )


def project_gemm_layer(
    layer: GemmWorkload, hardware_config, input_dtype_width: int
) -> GemmWorkload:
    """Project resident channels onto the physical matrix lanes issued."""
    input_lanes = hardware_config.input_lanes(input_dtype_width)
    output_lanes = hardware_config.output_lanes(input_dtype_width)

    physical_c = layer.nifm
    if physical_c % input_lanes:
        if not hardware_config.supports_partial_matrix_input(physical_c):
            raise MagnetAlphaSchedulingError(
                f"resident IC={physical_c} is not divisible by "
                f"IC0={input_lanes}"
            )
        physical_c = math.ceil(physical_c / input_lanes) * input_lanes

    physical_k = layer.nofm
    if physical_k % output_lanes:
        if not hardware_config.supports_partial_matrix_output(
            physical_k, input_dtype_width
        ):
            raise MagnetAlphaSchedulingError(
                f"resident OC={physical_k} is neither divisible by "
                f"OC0={output_lanes} nor a supported partial output"
            )
        physical_k = math.ceil(physical_k / output_lanes) * output_lanes

    if physical_c == layer.nifm and physical_k == layer.nofm:
        return layer
    return replace(layer, nifm=physical_c, nofm=physical_k)


def _divisors_desc(value: int) -> list[int]:
    divisors = []
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate:
            continue
        divisors.append(candidate)
        paired = value // candidate
        if paired != candidate:
            divisors.append(paired)
    return sorted(divisors, reverse=True)


def _local_input_depth(hardware_config) -> int:
    level = hardware_config.local_memory
    if level is None or level.input_memory is None:
        raise MagnetAlphaSchedulingError(
            "Alpha scheduling requires a bound local input memory"
        )
    memory = level.instance(level.input_memory)
    if memory.depth_per_bank is None:
        raise MagnetAlphaSchedulingError(
            "Alpha scheduling requires local input-buffer depth"
        )
    return memory.depth_per_bank


def _choose_local_factors(
    layer,
    hardware_config,
    input_lanes,
    output_lanes,
    q_alignment=1,
):
    collector = hardware_config.output_collector
    if collector is None:
        raise MagnetAlphaSchedulingError(
            "Alpha scheduling requires an output collector"
        )
    collector_depth = collector.depth_words
    channel_groups = layer.nifm // input_lanes
    output_groups = layer.nofm // output_lanes
    input_depth = _local_input_depth(hardware_config)

    for q1 in _divisors_desc(layer.wofm):
        if q1 % q_alignment or q1 > collector_depth:
            continue
        q2 = layer.wofm // q1
        can_flatten_p = (
            layer.wfil == 1
            and layer.hfil == 1
            and layer.wstd == 1
            and layer.hstd == 1
            and q2 == 1
        )
        p_candidates = _divisors_desc(layer.hofm) if can_flatten_p else [1]
        for p1 in p_candidates:
            if p1 * q1 > collector_depth:
                continue
            input_width = (q1 - 1) * layer.wstd + layer.wfil
            input_height = (p1 - 1) * layer.hstd + layer.hfil
            if input_width * input_height * channel_groups <= input_depth:
                return p1, q1, channel_groups, output_groups
    raise MagnetAlphaSchedulingError(
        "no Alpha streamed call fits the input buffer and collector"
    )


_MATRIX_PREFERRED_ORDERS = (
    (Loop.OC, Loop.IC, Loop.FX, Loop.FY, Loop.OX, Loop.OY, Loop.ON),
    (Loop.OX, Loop.OY, Loop.IC, Loop.FX, Loop.FY, Loop.OC, Loop.ON),
    (Loop.IC, Loop.OX, Loop.OY, Loop.OC, Loop.ON, Loop.FX, Loop.FY),
)


def _validate_coverage(layer, blockings, partitionings) -> None:
    for loop, extent in enumerate(layer.sizes):
        covered = math.prod(blockings[loop]) * math.prod(partitionings[loop])
        if covered != extent:
            raise MagnetAlphaSchedulingError(
                f"{Loop(loop).name} covers {covered}, expected {extent}"
            )


def schedule(layer, hardware_config, input_dtype_width, *, q_alignment=1):
    """Build Alpha's deterministic three-level single-call mapping."""
    if isinstance(q_alignment, bool) or int(q_alignment) != q_alignment:
        raise MagnetAlphaSchedulingError("Q alignment must be positive")
    q_alignment = int(q_alignment)
    if q_alignment <= 0:
        raise MagnetAlphaSchedulingError("Q alignment must be positive")

    input_lanes = hardware_config.input_lanes(input_dtype_width)
    output_lanes = hardware_config.output_lanes(input_dtype_width)
    matrix = hardware_config.unit("matrix")
    if matrix is None:
        raise MagnetAlphaSchedulingError(
            "Alpha scheduling requires a matrix compute-unit binding"
        )
    if (
        matrix.input_word_bits is not None
        and input_lanes * input_dtype_width != matrix.input_word_bits
    ):
        raise MagnetAlphaSchedulingError(
            "input lanes and dtype do not fill one matrix input word"
        )
    if layer.nifm % input_lanes:
        raise MagnetAlphaSchedulingError(
            f"IC={layer.nifm} is not divisible by IC0={input_lanes}"
        )
    if layer.nofm % output_lanes:
        raise MagnetAlphaSchedulingError(
            f"OC={layer.nofm} is not divisible by OC0={output_lanes}"
        )

    p1, q1, c1, k1 = _choose_local_factors(
        layer,
        hardware_config,
        input_lanes,
        output_lanes,
        q_alignment,
    )
    blockings = [[1, 1, 1] for _ in range(NUM_LOOPS)]
    partitionings = [[1, 1, 1] for _ in range(NUM_LOOPS)]
    partitionings[Loop.IC][0] = input_lanes
    partitionings[Loop.OC][0] = output_lanes
    blockings[Loop.FX][1] = layer.wfil
    blockings[Loop.FY][1] = layer.hfil
    blockings[Loop.OX][1] = q1
    blockings[Loop.OY][1] = p1
    blockings[Loop.OC][1] = k1
    blockings[Loop.IC][1] = c1
    blockings[Loop.OX][2] = layer.wofm // q1
    blockings[Loop.OY][2] = layer.hofm // p1
    blockings[Loop.OC][2] = (layer.nofm // output_lanes) // k1
    blockings[Loop.IC][2] = (layer.nifm // input_lanes) // c1
    blockings[Loop.ON][2] = layer.nimg

    blockings = tuple(tuple(row) for row in blockings)
    partitionings = tuple(tuple(row) for row in partitionings)
    _validate_coverage(layer, blockings, partitionings)
    return MappingPoint(
        _encode_loop_orders(
            blockings, partitionings, _MATRIX_PREFERRED_ORDERS
        ),
        blockings,
        partitionings,
    )


_ALPHA_UINT16_MAX = (1 << 16) - 1
_ALPHA_UINT32_MAX = (1 << 32) - 1
_ALPHA_INT16_MAX = (1 << 15) - 1
_ALPHA_WEIGHT_DIMENSION_MAX_EXTENT = 1 << 8
_ALPHA_BIAS_SCALE_GROUPS = 8


def _require_field_fit(name: str, value: int, maximum: int) -> None:
    if value <= 0 or value > maximum:
        raise MagnetAlphaSchedulingError(
            f"Alpha GEMM {name}={value} exceeds ABI range 1..{maximum}"
        )


def validate_single_call_mapping(
    mapping: MappingPoint, hardware_config, input_dtype_width: int
) -> None:
    """Reject mappings the Alpha ABI cannot represent as one GEMM call."""
    factors = mapping_point_to_alpha_factors(mapping)
    illegal = {
        name: factors[name]
        for name in ("c2", "c3", "k2", "k3")
        if factors[name] != 1
    }
    if illegal:
        raise MagnetAlphaSchedulingError(
            "Alpha ABI requires C2=C3=K2=K3=1; " f"found {illegal}"
        )
    p0, p1, p2 = factors["p0"], factors["p1"], factors["p2"]
    q0, q1, q2 = factors["q0"], factors["q1"], factors["q2"]
    c1, k1 = factors["c1"], factors["k1"]
    if factors["batch_size"] != 1:
        raise MagnetAlphaSchedulingError(
            "Alpha one-call GEMM supports batch one only"
        )
    kernel_words = factors["r"] * factors["s"] * c1
    _require_field_fit(
        "R*S*C1 weight dimension",
        kernel_words,
        _ALPHA_WEIGHT_DIMENSION_MAX_EXTENT,
    )
    _require_field_fit("K1 bias/scale groups", k1, _ALPHA_BIAS_SCALE_GROUPS)
    if p1 % p0 or q1 % q0:
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM requires integral P1/P0 and Q1/Q0"
        )

    temporal = p2 * q2
    weights_per_tile = kernel_words * k1
    for name, value in (
        (
            "num_wgt_tiles",
            temporal * (p1 // p0) * (q1 // q0) * weights_per_tile,
        ),
        ("num_to_read_0", weights_per_tile),
        ("num_to_read_1", p1 * q1 * weights_per_tile),
        ("temporal_tiling_factor", temporal),
    ):
        _require_field_fit(name, value, _ALPHA_UINT16_MAX)
    _require_field_fit(
        "MAC iterations",
        p1 * q1 * weights_per_tile * temporal,
        _ALPHA_UINT32_MAX + 1,
    )

    input_lanes = hardware_config.input_lanes(input_dtype_width)
    output_lanes = hardware_config.output_lanes(input_dtype_width)
    local_word_bits = input_lanes * input_dtype_width
    global_level = hardware_config.global_memory
    if global_level is None or global_level.weight_memory is None:
        raise MagnetAlphaSchedulingError(
            "Alpha scheduling requires a global weight memory"
        )
    global_weight = global_level.instance(global_level.weight_memory)
    if global_weight.bits_per_bank_entry % local_word_bits:
        raise MagnetAlphaSchedulingError(
            "global weight lines must contain integral local vectors"
        )
    vectors_per_line = global_weight.bits_per_bank_entry // local_word_bits
    if vectors_per_line <= 0 or output_lanes % vectors_per_line:
        raise MagnetAlphaSchedulingError(
            "output lanes need an integral number of global weight lines"
        )
    sweep_lines = weights_per_tile * (output_lanes // vectors_per_line)
    entries = global_weight.bank_count * global_weight.depth_per_bank
    _require_field_fit(
        "global weight sweep lines",
        sweep_lines,
        min(2 * entries, _ALPHA_INT16_MAX),
    )
    payload_bits = weights_per_tile * output_lanes * local_word_bits
    if payload_bits > global_weight.size_bits:
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM weight payload exceeds global weight capacity: "
            f"required={payload_bits} bits, available={global_weight.size_bits} bits"
        )


def _resident_input_channels(channels: int, input_dtype_width: int) -> int:
    """Alpha's compact resident form for a partial image input."""
    if input_dtype_width == 8 and channels == 3:
        return 8
    return channels


def _partial_input_q_alignment(
    layer: GemmWorkload, physical_layer: GemmWorkload, input_dtype_width: int
) -> int:
    resident = _resident_input_channels(layer.nifm, input_dtype_width)
    issued = physical_layer.nifm
    if issued == resident:
        return 1
    if resident <= 0 or issued % resident:
        raise MagnetAlphaSchedulingError(
            "partial input does not divide the issued Alpha vector"
        )
    return issued // resident


def single_call_gemm_schedule_result(
    layer: GemmWorkload, hardware_config, input_dtype_width: int
) -> LocalScheduleResult:
    """Return the complete physically legal Alpha GEMM schedule."""
    physical = project_gemm_layer(layer, hardware_config, input_dtype_width)
    mapping = schedule(
        physical,
        hardware_config,
        input_dtype_width,
        q_alignment=_partial_input_q_alignment(
            layer, physical, input_dtype_width
        ),
    )
    validate_single_call_mapping(mapping, hardware_config, input_dtype_width)
    return LocalScheduleResult(
        mapping=mapping,
        backend=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
        named_factors=mapping_point_to_alpha_factors(mapping),
    )


def single_call_gemm_schedule(
    layer: GemmWorkload, hardware_config, input_dtype_width: int
) -> MappingPoint:
    return single_call_gemm_schedule_result(
        layer, hardware_config, input_dtype_width
    ).mapping


__all__ = [
    "MAGNET_ALPHA_RULE_BASED_SCHEDULER",
    "MagnetAlphaSchedulingError",
    "project_gemm_layer",
    "schedule",
    "single_call_gemm_schedule",
    "single_call_gemm_schedule_result",
    "validate_single_call_mapping",
]
