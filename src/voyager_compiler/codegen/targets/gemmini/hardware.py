# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The Gemmini machine: what the silicon is, independent of any compile.

Split out of the target module so a fact about the datapath has one home.
"""

from voyager_compiler.hardware_config import (
    ComputeUnit,
    DTypeMode,
    UnrollDimension,
)

#: Systolic array edge; the array is DIM x DIM.
DIM = 16
INPUT_BITS = 8

#: Present in the helper API, so this is accelerator work, not a boundary.
HAS_MAX_POOL = True
HAS_DEPTHWISE_CONVS = True

DTYPE_MODES = (
    DTypeMode(operand_bits=INPUT_BITS, input_lanes=DIM, channel_alignment=DIM),
)

SYSTOLIC_UNIT = ComputeUnit(
    name="systolic_array",
    lanes=DIM,
    accepted_operations=frozenset(
        {"gemm"}
        | ({"pool2d"} if HAS_MAX_POOL else set())
        | ({"depthwise_conv2d"} if HAS_DEPTHWISE_CONVS else set())
    ),
    accepted_operand_bits=frozenset({INPUT_BITS}),
    # The square array unrolls input against output channels. The copy here
    # omitted these for as long as nothing read it; the live definition in the
    # test profile has always carried them.
    unroll_dimensions=(
        UnrollDimension("IC", DIM),
        UnrollDimension("OC", DIM),
    ),
)

COMPUTE_UNITS = (SYSTOLIC_UNIT,)


__all__ = [
    "COMPUTE_UNITS",
    "SYSTOLIC_UNIT",
    "DIM",
    "DTYPE_MODES",
    "HAS_DEPTHWISE_CONVS",
    "HAS_MAX_POOL",
    "INPUT_BITS",
]
