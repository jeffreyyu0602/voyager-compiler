"""Auto-generated hardware-layout twins of aten operators.

The data-layout passes re-store certain operands in hardware-friendly
layouts: NHWC activations + HWIO conv weights, and CK-ordered
(transposed) GEMM weights.  Each affected aten operator gets a
``quantized_ops`` twin with the SAME name and the SAME schema (extracted
verbatim from the aten ``OpOverload``, so the backend keeps seeing the
familiar targets); the generated eager impl permutes the
layout-transformed inputs back to aten's native layout, calls the
original operator, and permutes the result into the hardware layout.
The op identity therefore carries the layout contract: eager execution
of a transformed graph is correct with no process-global monkeypatching.

This module is a leaf (it imports only torch): the twins are registered
the moment ``decomposed`` imports it, before any module that references
``torch.ops.quantized_ops.conv2d`` etc. at import time can load.
"""

from typing import Dict, List

import torch

NCHW_TO_NHWC = (0, 2, 3, 1)
NHWC_TO_NCHW = (0, 3, 1, 2)
WEIGHT_NCHW_TO_HWIO = (2, 3, 1, 0)
WEIGHT_HWIO_TO_NCHW = (3, 2, 0, 1)

# Operators consuming NHWC activations (their twin takes and returns
# hardware-layout tensors; a conv weight is HWIO).
NHWC_LAYOUT_OPS: List[torch._ops.OpOverload] = [
    torch.ops.aten.conv2d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten._adaptive_avg_pool2d.default,
]

# GEMM-family operators.  Each gets a twin whose ONLY difference from
# the aten op is that the second operand (the weight / right matrix)
# arrives with its last two dims swapped.  The layout pass unifies all
# GEMM right-operands to one layout by flipping the storage of exactly
# the family that is not natively in it — and retargeting precisely
# those nodes to the twin: KC target => flip matmul (CK-native), CK
# target => flip linear (KC-native).  Untouched nodes keep their aten
# target, so eager execution is correct either way with no global
# patching.
GEMM_OPS: List[torch._ops.OpOverload] = [
    torch.ops.aten.linear.default,
    torch.ops.aten.matmul.default,
]

# The twins join the ``quantized_ops`` namespace ``decomposed.py`` owns
# (a FRAGMENT extends it; the handle must stay alive for the
# registrations to persist).
_QUANTIZED_OPS_FRAGMENT = torch.library.Library("quantized_ops", "FRAGMENT")


def _define_layout_variant(aten_op, arg_transforms, out_transform):
    """Register the ``quantized_ops`` twin of ``aten_op`` — SAME name,
    SAME schema — and return its ``OpOverload``.

    ``arg_transforms`` maps an argument name to ``fn(value, bound)`` that
    converts that input from the hardware layout back to aten's native
    layout (``bound`` is the full name->value argument mapping, so a
    transform may depend on another argument, e.g. conv ``groups``);
    ``out_transform`` converts the aten result into the hardware layout
    (``None`` = unchanged).  The impl is ``CompositeExplicitAutograd``,
    so fake tensors / export trace through it and no per-op fake impl is
    needed.
    """
    schema = aten_op._schema
    assert not schema.is_mutable and all(
        a.alias_info is None for a in schema.arguments
    ), f"layout variant of aliasing/mutating op {schema.name} unsupported"
    assert len(schema.returns) == 1, f"{schema.name}: single return required"

    name = schema.name.split("::")[1]
    _QUANTIZED_OPS_FRAGMENT.define(str(schema).split("::", 1)[1])

    arg_names = [a.name for a in schema.arguments]

    def wrapper(*args, **kwargs):
        bound = dict(zip(arg_names, args)) | kwargs
        args = list(args)
        for key, fn in arg_transforms.items():
            i = arg_names.index(key)
            if i < len(args) and args[i] is not None:
                args[i] = fn(args[i], bound)
            elif kwargs.get(key) is not None:
                kwargs[key] = fn(kwargs[key], bound)
        out = aten_op(*args, **kwargs)
        return out_transform(out) if out_transform is not None else out

    torch.library.impl(
        _QUANTIZED_OPS_FRAGMENT, name, "CompositeExplicitAutograd"
    )(wrapper)
    return getattr(torch.ops.quantized_ops, name).default


def register_nhwc_operators() -> Dict:
    """Generate the NHWC twin of every ``NHWC_LAYOUT_OPS`` entry: the
    first (activation) argument arrives NHWC, a ``weight`` argument
    arrives HWIO (kept as-is for a depthwise conv, whose weight layout
    is not transformed), and the result is returned NHWC.  Returns
    ``{aten OpOverload: generated OpOverload}``.
    """

    def to_nchw(t, bound):
        return t.permute(NHWC_TO_NCHW)

    def weight_to_nchw(t, bound):
        return (
            t.permute(WEIGHT_HWIO_TO_NCHW) if bound.get("groups", 1) == 1 else t
        )

    def to_nhwc(t):
        return t.permute(NCHW_TO_NHWC)

    variants = {}
    for op in NHWC_LAYOUT_OPS:
        arg_names = [a.name for a in op._schema.arguments]
        transforms = {arg_names[0]: to_nchw}
        if "weight" in arg_names:
            transforms["weight"] = weight_to_nchw
        variants[op] = _define_layout_variant(op, transforms, to_nhwc)
    return variants


def register_gemm_operators() -> Dict:
    """Generate the twin of every ``GEMM_OPS`` entry: identical to the
    aten op except that the second (weight / right-matrix) argument
    arrives with its last two dims swapped, un-swapped around the aten
    call.  Returns ``{aten OpOverload: generated OpOverload}``.
    """

    def swap_last_two(t, bound):
        return t.transpose(-2, -1)

    variants = {}
    for op in GEMM_OPS:
        second = op._schema.arguments[1].name
        variants[op] = _define_layout_variant(op, {second: swap_last_two}, None)
    return variants


NHWC_OP_VARIANTS = register_nhwc_operators()
GEMM_OP_VARIANTS = register_gemm_operators()
