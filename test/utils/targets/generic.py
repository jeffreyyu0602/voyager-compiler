# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The default Voyager target.

This is the machine the compiler has always assumed when no target is named: a
systolic MXU with a vector unit behind it, an L2 scratchpad, and DRAM. Its
policy is the behaviour that predates the target axis, moved here unchanged so
that ``--target generic`` reproduces earlier output exactly.

On the hardware config
----------------------
``accelerator_config`` returns the compiler's ``AcceleratorConfig`` and nothing
else. That class is the *compiler's* hardware contract -- every pass and the
tiler read it and only it -- so a target's job is to **construct** one, not to
extend it.

A target that knows more about its machine than ``AcceleratorConfig`` can
express keeps that knowledge in its own modules, where its own scheduler and
lowering read it. Widening the shared config with fields only one target sets
would make every other target carry them, and make the compiler's contract a
union of every machine ever adopted.
"""

import torch

from voyager_compiler import OpMatcher, get_default_quantizer
from voyager_compiler.codegen.node_info import is_fully_connected
from voyager_compiler.hardware_config import AcceleratorConfig

TARGET = "generic"


# --- fusion predicates -------------------------------------------------------
# What may share a pipelined stage is a property of the datapath, so these live
# with the target rather than with any workload.


def _is_bf16_fc(node):
    # BF16 FC are ran on vector unit and thus cannot be fused
    if hasattr(node, "value") and is_fully_connected(node):
        input_node = node.args[0]
        return input_node.meta.get("dtype") is None
    return False


def _is_spmm(node):
    return node.kwargs.get("A_data") is not None


def _can_fuse(node):
    return not _is_spmm(node) and not _is_bf16_fc(node)


def _is_constant_div(node):
    if node.target != torch.ops.aten.div.Tensor:
        return True

    divisor = node.args[1]
    if isinstance(divisor, torch.fx.Node):
        return divisor.value.numel() == 1

    return True


MXU_OPS = ["conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"]
QUANT_OPS = ["quantize", "quantize_mx", "quantize_mx_outlier"]

VECTOR_PIPELINE = [
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("add", "sub", "mul", "div", predicate=_is_constant_div),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div", predicate=_is_constant_div),
        OpMatcher(*QUANT_OPS, "mul", "div"),
    ],
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("gelu", "sigmoid", "silu", "tanh", "hardtanh"),
        OpMatcher(*QUANT_OPS, "mul", "div"),
    ],
    # Fused SpMM operation will use the first stage in the pipeline
    [
        OpMatcher(*MXU_OPS, predicate=_is_spmm),
        OpMatcher("dequantize"),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div"),
        OpMatcher(*QUANT_OPS),
    ],
    [
        OpMatcher("layer_norm", "softmax"),
        OpMatcher(*QUANT_OPS),
    ],
]


# --- profile hooks -----------------------------------------------------------


def apply_defaults(args):
    """No machine defaults to fill; the CLI governs alone."""
    return args


def vector_pipeline(args):
    """Fusion pattern stages for this target.

    Takes ``args`` because a target may pipeline differently depending on how
    it was configured; this one does not.
    """
    del args
    return VECTOR_PIPELINE


def quantizer(args):
    """The quantizer to prepare PT2E with.

    Operand dtypes come from the CLI here. A target whose units accept only
    certain operand widths, or that assigns different specs per operation
    class, builds that policy in its own profile instead.
    """
    return get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )


def accelerator_config(args) -> AcceleratorConfig:
    """The accelerator resources the compiler plans against."""
    return AcceleratorConfig.from_args(args)


__all__ = [
    "apply_defaults",
    "MXU_OPS",
    "QUANT_OPS",
    "TARGET",
    "VECTOR_PIPELINE",
    "accelerator_config",
    "quantizer",
    "vector_pipeline",
]
