# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deterministic scheduling for Magnet Alpha's near-memory processor.

The NMP has no mapping search. It streams one final operation directly from
global SRAM, so its canonical mapping has an implicit execution level and a
global traversal level carrying the endpoint's ``P1/Q1/C1/K1`` factors.
"""

import math
from typing import Optional

from scheduler.types import (
    LevelAccessCounts,
    LocalScheduleResult,
    Loop,
    MappingPoint,
    NUM_LOOPS,
    OperationWorkload,
)
from scheduler.magnet_alpha_scheduler.scheduler import (
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
)

# Kept as a compatibility name for callers that distinguish the physical NMP
# unit.  ``LocalScheduleResult.backend`` names the selected scheduling engine,
# which is the same rule-based provider for both Alpha units.
MAGNET_ALPHA_NMP_SCHEDULER = MAGNET_ALPHA_RULE_BASED_SCHEDULER

NMP_OPERATIONS = frozenset(
    {
        "channel_shuffle",
        "channel_shuffle_copy",
        "depthwise_conv2d",
        "group_norm",
        "pool2d",
        "softmax",
        "upsample_nearest2d",
    }
)

# Retained as a measured compatibility helper. The integrated performance
# model below derives pooling work from input/output shapes instead.
POOL_CYCLES_PER_ELEMENT = 0.5


def _channel_groups(channels: int, lanes: int) -> int:
    return max(1, math.ceil(int(channels) / int(lanes)))


def nmp_cycles(
    rows: int,
    columns: int,
    channel_groups: int,
    kernel_area: int = 1,
    *,
    pooling: bool = False,
) -> float:
    """Compatibility model for a known P/Q/K traversal."""
    cycles = float(rows * columns * channel_groups * kernel_area)
    return cycles * POOL_CYCLES_PER_ELEMENT if pooling else cycles


def _mapping(rows: int, columns: int, c_groups: int, k_groups: int):
    blockings = [[1, 1] for _ in range(NUM_LOOPS)]
    partitionings = [[1, 1] for _ in range(NUM_LOOPS)]
    blockings[Loop.OY][1] = int(rows)
    blockings[Loop.OX][1] = int(columns)
    blockings[Loop.IC][1] = int(c_groups)
    blockings[Loop.OC][1] = int(k_groups)
    # Unit factors still have a stable order; the endpoint record contains all
    # four even when one equals one.
    order_at_global = {
        Loop.OY: 0,
        Loop.OX: 1,
        Loop.IC: 2,
        Loop.OC: 3,
    }
    orders = [
        (NUM_LOOPS - 1, order_at_global.get(Loop(loop), NUM_LOOPS - 1))
        for loop in range(NUM_LOOPS)
    ]
    return MappingPoint(orders, blockings, partitionings)


def _activation_dims(shape, transposed=False):
    if shape is None or len(shape) != 4:
        return None
    if transposed:
        batch, height, width, channels = shape
    else:
        batch, channels, height, width = shape
    return tuple(int(value) for value in (batch, channels, height, width))


def _input_shape(workload, *ports):
    for port in ports:
        shapes = tuple(
            operand.shape
            for operand in workload.operands_for_port(port)
            if operand.shape is not None
        )
        if len(shapes) > 1:
            raise ValueError(f"multiple {port!r} shapes in NMP workload")
        if shapes:
            return shapes[0]
    return None


def _spatial_pair(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return tuple(int(item) for item in value)
    return None


def _shape_numel(shape) -> int:
    return 0 if shape is None else math.prod(int(value) for value in shape)


def _pool_work_per_output(classification, input_shape, output_shape) -> int:
    kernel = classification.attribute("kernel_size")
    if isinstance(kernel, int):
        return kernel * kernel
    if isinstance(kernel, (tuple, list)) and kernel:
        return math.prod(int(value) for value in kernel)
    output_elements = _shape_numel(output_shape)
    if not output_elements:
        return 1
    return max(1, math.ceil(_shape_numel(input_shape) / output_elements))


def _depthwise_input_accesses(
    classification, input_shape, output_shape, weight_shape
):
    transposed = classification.attribute("transposed", False)
    source = _activation_dims(input_shape, transposed)
    result = _activation_dims(output_shape, transposed)
    if source is None or result is None or not weight_shape:
        return None
    stride = _spatial_pair(classification.attribute("stride", 1))
    padding = _spatial_pair(classification.attribute("padding", 0))
    dilation = _spatial_pair(classification.attribute("dilation", 1))
    if stride is None or padding is None or dilation is None:
        return None
    batch, _, input_height, input_width = source
    _, output_channels, output_height, output_width = result
    kernel_height, kernel_width = int(weight_shape[-2]), int(weight_shape[-1])

    def valid_taps(input_extent, output_extent, kernel_extent, step, pad, gap):
        return sum(
            0 <= output_index * step - pad + kernel_index * gap < input_extent
            for output_index in range(output_extent)
            for kernel_index in range(kernel_extent)
        )

    return (
        batch
        * output_channels
        * valid_taps(
            input_height,
            output_height,
            kernel_height,
            stride[0],
            padding[0],
            dilation[0],
        )
        * valid_taps(
            input_width,
            output_width,
            kernel_width,
            stride[1],
            padding[1],
            dilation[1],
        )
    )


def _validate_operand_bits(workload, config, *ports) -> None:
    if config is None:
        return
    unit = config.unit("nmp")
    accepted = unit.accepted_operand_bits if unit is not None else frozenset()
    for port in ports:
        operands = workload.operands_for_port(port)
        if not operands:
            raise ValueError(f"Alpha NMP scheduling requires {port!r}")
        for operand in operands:
            if operand.bits is None:
                raise ValueError(f"Alpha NMP {port!r} width is unresolved")
            if accepted and operand.bits not in accepted:
                raise ValueError(
                    f"Alpha NMP {port!r} uses {operand.bits} bits; "
                    f"accepted: {sorted(accepted)}"
                )


def _traversal(workload, config, lanes):
    classification = workload.operation
    kind = classification.kind
    transposed = classification.attribute("transposed", False)
    output_dims = _activation_dims(workload.output.shape, transposed)

    if kind == "channel_shuffle_copy":
        _validate_operand_bits(workload, config, "input")
        if output_dims is None:
            raise ValueError("Alpha shuffle copy requires a rank-four output")
        batch, _, height, width = output_dims
        if batch != 1:
            raise ValueError("Alpha shuffle copy supports batch one only")
        block = classification.attribute("block_channels", lanes)
        if int(block) != lanes:
            raise ValueError(
                f"Alpha shuffle-copy block {block} does not match {lanes} lanes"
            )
        return height, width, 1, 1

    if kind == "depthwise_conv2d":
        _validate_operand_bits(workload, config, "input", "weight")
        input_dims = _activation_dims(
            _input_shape(workload, "input"), transposed
        )
        if input_dims is None or output_dims is None:
            raise ValueError("Alpha depthwise scheduling needs rank-four tensors")
        groups = int(classification.attribute("groups", 1))
        _, input_channels, _, _ = input_dims
        _, output_channels, height, width = output_dims
        if groups != input_channels or groups == 1:
            return None
        if output_channels != input_channels:
            raise ValueError("Alpha depthwise supports channel multiplier one")
        channel_groups = _channel_groups(output_channels, lanes)
        return height, width, channel_groups, channel_groups

    if kind == "pool2d":
        _validate_operand_bits(workload, config, "input")
        input_dims = _activation_dims(
            _input_shape(workload, "input"), transposed
        )
        if input_dims is None or output_dims is None:
            raise ValueError("Alpha pooling needs rank-four tensors")
        _, input_channels, input_height, input_width = input_dims
        _, output_channels, _, _ = output_dims
        return (
            input_height,
            input_width,
            _channel_groups(input_channels, lanes),
            _channel_groups(output_channels, lanes),
        )

    if kind == "channel_shuffle":
        _validate_operand_bits(workload, config, "input1", "input2")
        if output_dims is None:
            raise ValueError("Alpha shuffle requires a rank-four output")
        groups = int(classification.attribute("groups", 0))
        _, channels, height, width = output_dims
        if groups <= 0 or channels % groups:
            raise ValueError("Alpha shuffle groups must divide output channels")
        return (
            height,
            width,
            _channel_groups(channels // groups, lanes),
            _channel_groups(channels, lanes),
        )

    if kind in {"upsample_nearest2d", "group_norm"}:
        _validate_operand_bits(workload, config, "input")
        if output_dims is None:
            raise ValueError(f"Alpha {kind} requires a rank-four output")
        _, channels, height, width = output_dims
        groups = _channel_groups(channels, lanes)
        return height, width, groups, groups

    if kind == "softmax":
        _validate_operand_bits(workload, config, "input")
        shape = _input_shape(workload, "input")
        if shape is None or len(shape) < 2:
            raise ValueError("Alpha softmax requires a resolved input")
        dimension = int(classification.attribute("dim", -1))
        if dimension < 0:
            dimension += len(shape)
        if dimension != len(shape) - 1:
            raise ValueError("Alpha softmax supports the last dimension only")
        groups = _channel_groups(shape[-1], lanes)
        return int(shape[-2]), 1, groups, groups
    return None


def _route_cycles(config, transfers) -> float:
    if config is None or not config.hierarchy_bandwidths:
        return 0.0
    grouped = {}
    for source, destination, role, bits in transfers:
        role_name = getattr(role, "value", role)
        matches = [
            route
            for route in config.hierarchy_bandwidths
            if route.source == source
            and route.destination == destination
            and any(
                getattr(candidate, "value", candidate) == role_name
                for candidate in route.roles
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {role_name} NMP route from {source} to "
                f"{destination}, found {len(matches)}"
            )
        route = matches[0]
        group = route.shared_group or f"route:{route.name}"
        grouped[group] = grouped.get(group, 0.0) + (
            float(bits) / route.bandwidth_bits_per_cycle
        )
    return max(grouped.values(), default=0.0)


def _performance(workload, config, lanes):
    classification = workload.operation
    kind = classification.kind
    input_shape = _input_shape(workload, "input") or workload.output.shape
    output_shape = workload.output.shape
    input_elements = _shape_numel(input_shape)
    output_elements = _shape_numel(output_shape)
    parameter_elements = sum(
        _shape_numel(operand.shape)
        for port in ("weight", "bias")
        for operand in workload.operands_for_port(port)
    )
    work_per_output = 1
    input_passes = 1

    if kind == "channel_shuffle_copy":
        dims = _activation_dims(
            output_shape, classification.attribute("transposed", False)
        )
        batch, _, height, width = dims
        block = int(classification.attribute("block_channels", lanes))
        output_elements = batch * height * width * block
        input_accesses, weight_accesses = output_elements, 0
    elif kind == "depthwise_conv2d":
        weight_shape = _input_shape(workload, "weight")
        if weight_shape is not None:
            work_per_output = math.prod(weight_shape[-2:])
        input_accesses = _depthwise_input_accesses(
            classification, input_shape, output_shape, weight_shape
        )
        if input_accesses is None:
            input_accesses = output_elements * work_per_output
        weight_accesses = output_elements * work_per_output
    elif kind == "pool2d":
        work_per_output = _pool_work_per_output(
            classification, input_shape, output_shape
        )
        input_accesses = output_elements * work_per_output
        weight_accesses = 0
    elif kind in {"group_norm", "softmax"}:
        input_passes = 3
        input_accesses = input_elements * input_passes
        weight_accesses = parameter_elements
    else:
        input_accesses = output_elements
        weight_accesses = parameter_elements

    global_counts = LevelAccessCounts(
        input=input_accesses,
        output=output_elements,
        weight=weight_accesses,
    )
    compute_cycles = math.ceil(
        max(
            input_accesses,
            output_elements * work_per_output * input_passes,
        )
        / lanes
    )
    if config is None:
        return (LevelAccessCounts(), global_counts), float(compute_cycles)

    unit = config.unit("nmp")
    input_bits = max(
        (
            operand.bits
            for port in ("input", "input1", "input2")
            for operand in workload.operands_for_port(port)
            if operand.bits is not None
        ),
        default=(
            min(unit.accepted_operand_bits)
            if unit.accepted_operand_bits
            else 8
        ),
    )
    weight_bits = max(
        (
            operand.bits
            for operand in workload.operands_for_port("weight")
            if operand.bits is not None
        ),
        default=input_bits,
    )
    output_bits = workload.output.bits
    if output_bits is None:
        raise ValueError("Alpha NMP output width is unresolved")
    transfers = [
        (
            config.global_memory.name,
            unit.name,
            "activation",
            input_accesses * input_bits,
        ),
        (
            unit.name,
            config.global_memory.name,
            "output",
            output_elements * output_bits,
        ),
    ]
    if weight_accesses:
        transfers.append(
            (
                config.global_memory.name,
                unit.name,
                "weight",
                weight_accesses * weight_bits,
            )
        )
    runtime = max(compute_cycles, math.ceil(_route_cycles(config, transfers)))
    return (LevelAccessCounts(), global_counts), float(runtime)


def schedule_nmp_operation(
    workload: OperationWorkload,
    config,
    *,
    lanes: int,
) -> Optional[LocalScheduleResult]:
    """Claim a supported NMP operation, or decline it without guessing."""
    if not isinstance(workload, OperationWorkload):
        raise TypeError("Alpha NMP scheduling requires OperationWorkload")
    classification = workload.operation
    if classification is None or classification.kind not in NMP_OPERATIONS:
        return None
    decision = _traversal(workload, config, int(lanes))
    if decision is None:
        return None
    rows, columns, c_groups, k_groups = decision
    mapping = _mapping(rows, columns, c_groups, k_groups)
    accesses, runtime = _performance(workload, config, int(lanes))
    return LocalScheduleResult(
        mapping=mapping,
        backend=MAGNET_ALPHA_NMP_SCHEDULER,
        access_counts=accesses,
        runtime=runtime,
        named_factors={
            "p1": rows,
            "q1": columns,
            "c1": c_groups,
            "k1": k_groups,
        },
    )


__all__ = [
    "MAGNET_ALPHA_NMP_SCHEDULER",
    "NMP_OPERATIONS",
    "nmp_cycles",
    "schedule_nmp_operation",
]
