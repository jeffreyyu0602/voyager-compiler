# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Freeze final Alpha FX records and bind target-local schedules to them.

Recognition lives here because it is about Alpha's final operator sequences.
The scheduler receives only immutable shapes, widths, and scalar attributes;
it never reaches back into FX.  Results bind to the top-level node that owns
the serialized record, including when that record is a fused submodule.
"""

import logging
from typing import Optional

import torch
from torch.fx import GraphModule, Node

from scheduler.dispatch import schedule_operation
from scheduler.types import (
    GemmWorkload,
    OperationClassification,
    OperationOperand,
    OperationWorkload,
)
from scheduler.voyager import bind_local_schedule
from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.targets.magnet_alpha.operators import (
    execution_kind,
)

LOGGER = logging.getLogger(__name__)


# Positional geometry follows each target operator's public schema.  VSQ adds
# ``weight_scale`` ahead of the convolution attributes, so treating every
# accumulator like the int8 schema shifts all four fields by one and mistakes
# the scale FX node for ``stride``.  Keep this translation beside workload
# extraction: the scheduler IR itself intentionally knows nothing about FX or
# target operator signatures.
_CONVOLUTION_GEOMETRY_ARGUMENTS = {
    "conv2d_int8_accumulator": {
        "stride": 2,
        "padding": 3,
        "dilation": 4,
        "groups": 5,
    },
    "conv2d_vsq": {
        "stride": 3,
        "padding": 4,
        "dilation": 5,
        "groups": 6,
    },
}


def _convolution_geometry_arg(node: Node, name: str, default):
    record = _record_name(node)
    try:
        position = _CONVOLUTION_GEOMETRY_ARGUMENTS[record][name]
    except KeyError as error:
        raise ValueError(
            f"{node.name}: no {name} schema position for Alpha record "
            f"{record!r}"
        ) from error
    return get_arg_value(node, position, name, default)


def _submodule(model: GraphModule, node: Node):
    if node.op != "call_module":
        return None
    candidate = getattr(model, str(node.target), None)
    return candidate if isinstance(candidate, GraphModule) else None


def _record_nodes(model: GraphModule, owner: Node):
    # Deferred to avoid a module cycle: serialization calls this module after
    # its own record vocabulary has finished initializing.
    from voyager_compiler.codegen.targets.magnet_alpha.serialization import (
        record_target,
    )

    submodule = _submodule(model, owner)
    candidates = list(submodule.graph.nodes) if submodule is not None else [owner]
    return tuple(node for node in candidates if record_target(node) is not None)


def _value(node):
    value = getattr(node, "value", None)
    if value is None and isinstance(node, Node):
        value = node.meta.get("val")
    return value


def _shape(node) -> Optional[tuple[int, ...]]:
    value = _value(node)
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = getattr(node, "shape", None)
    return None if shape is None else tuple(int(dimension) for dimension in shape)


def _output_shape(owner: Node) -> tuple[int, ...]:
    value = _value(owner)
    if isinstance(value, (tuple, list)):
        # Alpha's PVS replacement preserves quantize_vsq's public tuple order:
        # coarse scale, fine scale, payload.  The payload is what the record
        # writes and hence what the schedule covers.
        value = value[-1] if value else None
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{owner.name}: scheduled record has no tensor output")
    return tuple(int(dimension) for dimension in shape)


def _record_name(node: Node) -> str:
    from voyager_compiler.codegen.targets.magnet_alpha.serialization import (
        record_target,
    )

    name = record_target(node)
    if name is None:
        raise ValueError(f"{node.name}: no Alpha record target")
    return name


def _bound_operand(model: GraphModule, owner: Node, candidate):
    """Resolve a fused-body placeholder to its current top-level operand."""
    if not isinstance(candidate, Node) or candidate.op != "placeholder":
        return candidate
    submodule = _submodule(model, owner)
    if submodule is None or candidate.graph is not submodule.graph:
        return candidate
    placeholders = [
        node for node in submodule.graph.nodes if node.op == "placeholder"
    ]
    try:
        index = placeholders.index(candidate)
    except ValueError:
        return candidate
    return owner.args[index] if index < len(owner.args) else candidate


def _output_bits(records, default=8) -> int:
    for node in reversed(records):
        name = _record_name(node)
        if name == "pvs_quantize":
            return int(get_arg_value(node, 2, "output_precision", 4))
        if name == "ppu_fixed_relu":
            return int(get_arg_value(node, 7, "output_precision", default))
        if name == "nmp_fixed_requantize":
            return int(get_arg_value(node, 5, "output_precision", default))
    return int(default)


def _matrix_workload(
    model: GraphModule, owner: Node, records, *, depthwise: bool
):
    head = records[0]
    input_node = _bound_operand(
        model, owner, get_arg_value(head, 0, "input")
    )
    weight_node = _bound_operand(
        model, owner, get_arg_value(head, 1, "weight")
    )
    input_shape, weight_shape = _shape(input_node), _shape(weight_node)
    output_shape = _output_shape(owner)
    if input_shape is None or weight_shape is None:
        raise ValueError(f"{owner.name}: convolution operands lack shapes")
    if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
        raise ValueError(
            f"{owner.name}: Alpha convolution scheduling requires rank four"
        )

    input_bits = 4 if _record_name(head) == "conv2d_vsq" else 8
    output_bits = _output_bits(records)
    stride = _convolution_geometry_arg(head, "stride", (1, 1))
    if isinstance(stride, int):
        stride = (stride, stride)
    stride = tuple(int(value) for value in stride)
    if len(stride) != 2:
        raise ValueError(f"{owner.name}: convolution stride must be a pair")

    attributes = {}
    kind = "gemm"
    gemm = None
    if depthwise:
        kind = "depthwise_conv2d"
        attributes = {
            "groups": int(_convolution_geometry_arg(head, "groups", 1)),
            "stride": stride,
            "padding": tuple(
                _convolution_geometry_arg(head, "padding", (0, 0))
            ),
            "dilation": tuple(
                _convolution_geometry_arg(head, "dilation", (1, 1))
            ),
            "kernel_area": int(weight_shape[-2] * weight_shape[-1]),
            "transposed": False,
        }
    else:
        gemm = GemmWorkload(
            nifm=weight_shape[1],
            nofm=weight_shape[0],
            wofm=output_shape[-1],
            hofm=output_shape[-2],
            wfil=weight_shape[-1],
            hfil=weight_shape[-2],
            nimg=output_shape[0],
            wstd=stride[-1],
            hstd=stride[-2],
        )

    return OperationWorkload(
        operands=(
            OperationOperand("input", input_shape, input_bits),
            OperationOperand("weight", weight_shape, input_bits),
        ),
        output=OperationOperand("output", output_shape, output_bits),
        operation=OperationClassification(kind, attributes),
        gemm=gemm,
        # Every matrix record contains an accumulator-to-PPU boundary wider
        # than its encoded input.  This is the fork's high-precision vector
        # input fact, made structural here instead of rediscovered by width
        # heuristics over unrelated placeholders.
        has_high_precision_vector_input=not depthwise,
    )


def _standalone_workload(owner: Node, records, execution: str):
    node = records[0]
    output_shape = _output_shape(owner)
    attributes = {"transposed": False}
    operands = []

    if execution == "nmp_channel_shuffle":
        kind = "channel_shuffle"
        attributes["groups"] = int(get_arg_value(node, 2, "groups"))
        ports = (("input1", 0), ("input2", 1))
    elif execution == "nmp_copy":
        kind = "channel_shuffle_copy"
        metadata = node.meta.get("magnet_alpha")
        block_channels = (
            metadata.get("block_channels")
            if isinstance(metadata, dict)
            else None
        )
        if (
            isinstance(block_channels, bool)
            or not isinstance(block_channels, int)
            or block_channels <= 0
        ):
            raise ValueError(
                f"{node.name}: an Alpha shuffle copy requires positive "
                "block_channels rewrite metadata"
            )
        attributes["block_channels"] = block_channels
        ports = (("input", 0),)
    elif execution == "nmp_max_pool":
        kind = "pool2d"
        attributes.update(
            {
                "kernel_size": get_arg_value(node, 1, "kernel_size"),
                "stride": get_arg_value(node, 2, "stride", ()),
                "padding": get_arg_value(node, 3, "padding", 0),
                "dilation": get_arg_value(node, 4, "dilation", 1),
                "ceil_mode": bool(get_arg_value(node, 5, "ceil_mode", False)),
            }
        )
        ports = (("input", 0),)
    elif execution == "nmp_nearest_upsample":
        kind = "upsample_nearest2d"
        ports = (("input", 0),)
    else:
        raise ValueError(f"{owner.name}: unsupported Alpha execution {execution}")

    for port, index in ports:
        operand_node = get_arg_value(node, index, port)
        shape = _shape(operand_node)
        if shape is None:
            raise ValueError(f"{owner.name}: {port} has no resolved shape")
        operands.append(OperationOperand(port, shape, 8))
    return OperationWorkload(
        operands=tuple(operands),
        output=OperationOperand("output", output_shape, _output_bits(records)),
        operation=OperationClassification(kind, attributes),
    )


def build_alpha_workload(model: GraphModule, owner: Node):
    """Freeze the final record owned by ``owner``, or return ``None``."""
    records = _record_nodes(model, owner)
    if not records:
        return None
    sequence = tuple(_record_name(node) for node in records)
    execution = execution_kind(sequence)
    if execution is None:
        return None
    if execution == "gemm_convolution":
        return _matrix_workload(model, owner, records, depthwise=False)
    if execution == "nmp_depthwise_convolution":
        return _matrix_workload(model, owner, records, depthwise=True)
    return _standalone_workload(owner, records, execution)


def bind_alpha_schedules(
    model: GraphModule,
    config,
    providers,
    backend=None,
) -> int:
    """Schedule and bind every final executable Alpha record."""
    result_cache = {}
    bound = 0
    for owner in model.graph.nodes:
        workload = build_alpha_workload(model, owner)
        if workload is None:
            continue
        result = schedule_operation(
            workload,
            config,
            result_cache=result_cache,
            providers=providers,
            backend=backend,
        )
        if result is None:
            raise ValueError(f"{owner.name}: no Alpha scheduler claimed record")
        bind_local_schedule(owner, result, workload)
        bound += 1
        LOGGER.debug(
            "[%s] %s: runtime=%s", result.backend, owner.name, result.runtime
        )
    return bound


__all__ = ["bind_alpha_schedules", "build_alpha_workload"]
