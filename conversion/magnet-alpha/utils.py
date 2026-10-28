#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared constants and pure parsers for Voyager-to-CMOD collateral."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

CONCAT_FIELDS = (
    "is_to_concat",
    "concat_channel_index_0",
    "concat_idx_0",
    "concat_size_0",
    "concat_total_size_0",
)
SCHEDULE_FACTOR_PREFIX = "schedule_factor_"
GEMM_SCHEDULE_FACTORS = (
    "batch_size",
    "c0",
    "c1",
    "c2",
    "c3",
    "k0",
    "k1",
    "k2",
    "k3",
    "p0",
    "p1",
    "p2",
    "p2s",
    "p3",
    "q0",
    "q1",
    "q2",
    "q2s",
    "q3",
    "r",
    "s",
)
NMP_SCHEDULE_FACTORS = ("c1", "k1", "p1", "q1")
CMOD_FUSIBLE_TARGETS = frozenset(
    {
        "dequantize",
        "quantize",
    }
)
CMOD_SKIP_REASON_HOST_OUTPUT_LAYOUT = "host_output_layout"
CMOD_HOST_OUTPUT_LAYOUT_TARGETS = frozenset(
    {
        "contiguous",
        "permute",
    }
)
CMOD_CONVOLUTION_TARGETS = frozenset(
    {
        "conv2d_int8_accumulator",
        "conv2d_vsq",
    }
)
CMOD_COPY_TARGET = "channel_shuffle_copy"
CMOD_LEGACY_COPY_TARGET = "add"
NMP_FLAGS = (
    "is_dwconv",
    "is_shuffling",
    "is_transpose",
    "is_copy",
    "is_copy_query",
    "is_upsampling",
    "is_maxpool",
    "is_avgpool",
    "is_groupnorm_full",
    "is_groupnorm_backend",
    "is_softmax_nmp",
    "is_softmax_backend",
    "is_warp_perspective",
)
GROUP_FIELDS = (
    "bias_scale_buffer_base",
    "is_next_dwconv",
    "is_next_pool",
    "is_next_shuffle",
    "is_next_upsample",
    "num_ppu_outputs",
    "workload_sequence",
)
ADDRESS_FIELD = re.compile(r"^(inputs|outputs)([0-9]+)_baseaddr$")
O4_PRECISION = 4
O4_SCALE_PRECISION = 8
MAX_PPU_OUTPUTS = (1 << 16) - 1
# The Alpha C-model ABI applies this name-based exception instead of the
# generic concat offset. Keep the ABI quirk local to conversion until the
# format exposes an explicit schedule field.
CMOD_CONCAT_OUTPUT_OFFSET_OVERRIDES = {"model_4_6_shuffle": 2}


def cmod_concat_output_offset(
    layer: str,
    *,
    is_to_concat: int,
    concat_index: int,
    k1: int | None,
) -> int:
    """Return the output-base adjustment applied by Alpha's CMOD ABI."""
    if is_to_concat != 1 or concat_index != 1:
        return 0
    override = CMOD_CONCAT_OUTPUT_OFFSET_OVERRIDES.get(layer)
    if override is not None:
        return override
    if k1 is None:
        raise ValueError(f"{layer}: a concat output adjustment requires k1")
    return int(k1)


def argument_value(argument: Any) -> Any:
    """Decode one protobuf ``Argument`` without importing its generated type."""
    kind = argument.WhichOneof("arg_type")
    if kind == "tensor":
        return argument.tensor
    if kind == "tensor_list":
        return list(argument.tensor_list.tensors)
    if kind in {"int_list", "bool_list", "scalar_list"}:
        return list(getattr(argument, kind).values)
    if kind is None:
        return None
    return getattr(argument, kind)


def operation_kwarg(op: Any, name: str, default: Any = None) -> Any:
    """Decode one serialized operation kwarg."""
    if name not in op.kwargs:
        return default
    return argument_value(op.kwargs[name])


def inner_operations(operation: Any) -> list[Any]:
    """Return the ordinary operations contained by one serialized record."""
    kind = operation.WhichOneof("op_type")
    if kind == "op":
        return [operation.op]
    if kind == "fused_op":
        return list(operation.fused_op.op_list)
    return []


def operation_name(operation: Any) -> str:
    """Return the stable name of one serialized operation record."""
    kind = operation.WhichOneof("op_type")
    if kind == "op":
        return operation.op.name
    if kind == "fused_op":
        return operation.fused_op.name
    if kind == "loop":
        return operation.loop.node
    return "unknown"


def output_tensors(operation: Any) -> list[Any]:
    """Return every tensor emitted by one serialized operation record."""
    kind = operation.WhichOneof("return_type")
    if kind == "output":
        return [operation.output]
    if kind == "outputs":
        return list(operation.outputs.tensors)
    return []


def input_tensors(operation: Any) -> Iterable[Any]:
    """Yield tensor arguments consumed by one serialized operation record."""
    for op in inner_operations(operation):
        for argument in list(op.args) + list(op.kwargs.values()):
            value = argument_value(argument)
            if hasattr(value, "node"):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, "node"):
                        yield item


def operation_target(operation: Any) -> str:
    """Render the target or fused target sequence for diagnostics."""
    operations = inner_operations(operation)
    if len(operations) == 1:
        return operations[0].target
    return "+".join(op.target for op in operations)


def dtype_precision(dtype: str) -> int:
    """Return the CMOD payload width encoded by a dump dtype."""
    if "int4" in dtype:
        return 4
    if "int8" in dtype or "uint8" in dtype:
        return 8
    return 0


def tensor_address(tensor: Any) -> int | None:
    """Return a finalized scratchpad address when one is serialized."""
    if not tensor.HasField("scratchpad"):
        return None
    if not tensor.scratchpad.HasField("address"):
        return None
    return int(tensor.scratchpad.address)


def format_address(address: int) -> str:
    """Render an address using CMOD's hexadecimal text convention."""
    return f"0x{address:x}"


def parse_schedule(path: Path) -> dict[str, str]:
    """Read the key/value subset used by Alpha text schedules."""
    fields = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def render_schedule(schedule: Mapping[str, Any]) -> str:
    """Render schedule fields in their existing insertion order."""
    return "".join(f"{key} = {value}\n" for key, value in schedule.items())
