# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Alpha's operator vocabulary, and which fusions the endpoint can execute.

These names are an **ABI**, not an internal choice. ``model.txt`` preserves the
*sequence* carried by a fused record, so a lowering that emits the right
arithmetic under an unregistered name would produce an incomplete Alpha
program instead of failing where the mistake was made.

The vocabulary is therefore stated here once. For the current YOLOv5n-0.5
workload it accounts for all 144 emitted Alpha calls:

===========================================  =====  ==========================
sequence                                     count  execution kind
===========================================  =====  ==========================
``conv2d_vsq -> ppu_fixed_relu``                24   gemm_convolution
``conv2d_int8_accumulator -> ppu_fixed_relu``   22   gemm_convolution
``conv2d_vsq -> ppu_fixed_relu -> pvs``         12   gemm_convolution
``conv2d_int8_accumulator -> ppu -> pvs``        8   gemm_convolution
``conv2d_int8_accumulator -> nmp_req -> pvs``   18   nmp_depthwise_convolution
``conv2d_int8_accumulator -> nmp_req``           1   nmp_depthwise_convolution
``channel_shuffle_copy``                        40   nmp_copy
``channel_shuffle``                             16   nmp_channel_shuffle
``upsample_nearest2d``                           2   nmp_nearest_upsample
``max_pool2d``                                   1   nmp_max_pool
===========================================  =====  ==========================

66 + 19 + 40 + 16 + 2 + 1 = **144**.

Two additional sequences remain registered for models that need them: a
``saturating_add`` epilogue (residual fused into the convolution) and the
legacy ``add``-shaped copy.
"""

from typing import Dict, FrozenSet, Optional, Sequence, Tuple

__all__ = [
    "CONVOLUTION_HEADS",
    "EXECUTABLE_SEQUENCES",
    "SINGLE_TARGET_KINDS",
    "execution_kind",
    "is_executable_sequence",
]

#: The matrix operations. Both take an int8 activation; ``conv2d_vsq`` reads a
#: vector-scaled weight, whose fine scale rides in the same allocation.
CONV2D_INT8_ACCUMULATOR = "conv2d_int8_accumulator"
CONV2D_VSQ = "conv2d_vsq"

#: Epilogues. ``ppu_fixed_relu`` is the matrix unit's requantize-and-clamp;
#: ``nmp_fixed_requantize`` is the near-memory unit's. Which one appears is
#: what makes an otherwise identical convolution a GEMM or a depthwise.
PPU_FIXED_RELU = "ppu_fixed_relu"
NMP_FIXED_REQUANTIZE = "nmp_fixed_requantize"
#: A trailing re-quantization to the next operation's domain.
PVS_QUANTIZE = "pvs_quantize"
#: A residual folded into the convolution, saturating rather than wrapping.
SATURATING_ADD = "saturating_add"

#: Near-memory operations that stand alone.
CHANNEL_SHUFFLE = "channel_shuffle"
CHANNEL_SHUFFLE_COPY = "channel_shuffle_copy"
MAX_POOL2D = "max_pool2d"
UPSAMPLE_NEAREST2D = "upsample_nearest2d"

#: A convolution can head a fused record; nothing else can.
CONVOLUTION_HEADS: FrozenSet[str] = frozenset(
    {CONV2D_INT8_ACCUMULATOR, CONV2D_VSQ}
)

#: Operations whose record is one target long.
SINGLE_TARGET_KINDS: Dict[str, str] = {
    CHANNEL_SHUFFLE: "nmp_channel_shuffle",
    CHANNEL_SHUFFLE_COPY: "nmp_copy",
    MAX_POOL2D: "nmp_max_pool",
    UPSAMPLE_NEAREST2D: "nmp_nearest_upsample",
}

#: Which epilogue suffixes each head may carry, and what the endpoint then
#: executes. A suffix absent from here is not "unsupported arithmetic" -- it
#: is a record the converter will not emit at all.
_MATRIX_SUFFIXES: Dict[Tuple[str, ...], str] = {
    (PPU_FIXED_RELU,): "gemm_convolution",
    (PPU_FIXED_RELU, PVS_QUANTIZE): "gemm_convolution",
}
#: The residual epilogue is int8-accumulator only: the VSQ head has no
#: saturating add path.
_INT8_ONLY_SUFFIXES: Dict[Tuple[str, ...], str] = {
    (PPU_FIXED_RELU, SATURATING_ADD): "gemm_convolution",
    (NMP_FIXED_REQUANTIZE,): "nmp_depthwise_convolution",
    (NMP_FIXED_REQUANTIZE, PVS_QUANTIZE): "nmp_depthwise_convolution",
}

EXECUTABLE_SEQUENCES: Dict[Tuple[str, ...], str] = {
    **{(target,): kind for target, kind in SINGLE_TARGET_KINDS.items()},
    **{
        (head,) + suffix: kind
        for head in CONVOLUTION_HEADS
        for suffix, kind in _MATRIX_SUFFIXES.items()
    },
    **{
        (CONV2D_INT8_ACCUMULATOR,) + suffix: kind
        for suffix, kind in _INT8_ONLY_SUFFIXES.items()
    },
}


def execution_kind(targets: Sequence[str]) -> Optional[str]:
    """What the endpoint executes for this target sequence, or ``None``.

    ``None`` means the converter will not emit a layer for it. That is a
    useful answer at lowering time and a useless one at conversion time, which
    is the whole reason this table lives on the compiler side too.
    """
    return EXECUTABLE_SEQUENCES.get(tuple(targets))


def is_executable_sequence(targets: Sequence[str]) -> bool:
    return execution_kind(targets) is not None
