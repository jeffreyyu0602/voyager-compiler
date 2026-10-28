# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The ``magnet_alpha`` torch.library namespace: what the endpoint executes.

Alpha's datapath is not expressible as a composition of ATen operators. Its
convolution returns a raw **int32 accumulator**, and its epilogue is a
two-stage fixed-point requantization with two int16 wraps in the middle. An
``aten.conv2d`` followed by arithmetic in float agrees with it on typical data
and disagrees exactly where the wraps matter, which is the only place worth
being right.

So the operators are declared, and their implementations are the endpoint's
integer arithmetic -- `fixed_point.execute_alpha_requantization`, already
checked against every requantization record of a reference compile.

**What is on an operator and what is not.** A record in `model.txt` carries two
kinds of field, and only one of them belongs here:

*semantic* -- stride, padding, dilation, groups, the encoded multipliers and
shifts, the bias, the shuffle's group count. These decide the result, so they
are operator arguments.

*placement* -- ``output_payload``, ``output_addresses``,
``*_memory_instance``, ``copy_source_entry``, ``padding_c`` / ``padding_k``,
``concat_*``, ``nmp_c2`` / ``nmp_k2``, ``l2_tiling``. These say where the
result lives, and they come from allocation at serialization time. Putting
them on the operator would make two passes able to disagree about an address
with nothing to arbitrate.

The names are an ABI; see ``operators.py`` for why, and for which fused
sequences the converter will actually emit.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torch.library import Library, impl

from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    alpha_int8_conv2d_accumulator,
    alpha_nmp_epilogue,
    alpha_vsq_conv2d_accumulator,
)
from voyager_compiler.ops.quantized import expand

# "DEF" creates and owns the namespace; keep the handle alive at module level.
magnet_alpha_lib = Library("magnet_alpha", "DEF")

__all__ = ["magnet_alpha_lib"]

#: The accumulator's type. Alpha accumulates 8-bit products in 32 bits, and
#: the epilogue is what narrows them.
ACCUMULATOR_DTYPE = torch.int32

#: What a direct O8 epilogue and every PVS payload return.  An O4 epilogue
#: instead feeds PVS as a wide int64 intermediate; PVS is the stage that
#: narrows that value into this container.
ACTIVATION_DTYPE = torch.int8


def _conv_accumulate(
    activation: torch.Tensor,
    weight: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """The convolution's integer accumulator.

    Computed in float64 and cast, not because the hardware does: an int64
    convolution has no ATen kernel, and every product here fits a double's
    53-bit mantissa many times over (8-bit operands, at most a few thousand
    terms). The cast back is exact.
    """
    accumulated = torch.nn.functional.conv2d(
        activation.to(torch.float64),
        weight.to(torch.float64),
        None,
        list(stride),
        list(padding),
        list(dilation),
        groups,
    )
    return accumulated.round().to(ACCUMULATOR_DTYPE)


magnet_alpha_lib.define(
    "conv2d_int8_accumulator(Tensor input, Tensor weight, int[] stride, "
    "int[] padding, int[] dilation, int groups) -> Tensor"
)


@impl(magnet_alpha_lib, "conv2d_int8_accumulator", "CompositeExplicitAutograd")
def conv2d_int8_accumulator(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
) -> torch.Tensor:
    return alpha_int8_conv2d_accumulator(
        input, weight, stride, padding, dilation, groups
    ).to(ACCUMULATOR_DTYPE)


#: The endpoint's own names for a vector-scaled call's operands.
#:
#: ``weight_scale`` and ``input_scale`` rather than ``fine_scale`` and
#: ``input_fine_scale``: these are what a record carries and what the
#: converter binds by, and naming the operator's parameters the same way is
#: what keeps the serializer from needing a translation table that could drift
#: from the schema it translates.
magnet_alpha_lib.define(
    "conv2d_vsq(Tensor input, Tensor weight, Tensor weight_scale, "
    "int[] stride, int[] padding, int[] dilation, int groups, *, "
    "Tensor? input_scale=None, int? block_size=None, "
    "Tensor? folded_coarse_scale=None, "
    "Tensor? semantic_input_coarse_scale=None, "
    "Tensor? semantic_weight_coarse_scale=None, "
    "float input_fine_quant_max=7.0, "
    "float input_coarse_scale_quant_max=127.0, "
    "float weight_fine_quant_max=7.0, "
    "float weight_coarse_scale_quant_max=127.0) -> Tensor"
)


@impl(magnet_alpha_lib, "conv2d_vsq", "CompositeExplicitAutograd")
def conv2d_vsq(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    *,
    input_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    folded_coarse_scale: Optional[torch.Tensor] = None,
    semantic_input_coarse_scale: Optional[torch.Tensor] = None,
    semantic_weight_coarse_scale: Optional[torch.Tensor] = None,
    input_fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    input_coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    weight_fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    weight_coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
) -> torch.Tensor:
    """A vector-scaled convolution: narrow payloads and uint8 fine scales.

    The fine scales are applied to the *operands*, before the reduction, which
    is what makes them scales on the operand rather than another epilogue
    stage -- and it is why the accumulator this returns is 65536x larger than
    the same computation on plain 8-bit operands would leave it, which the
    requantization encoder undoes with ``vsq_operands=True``.

    Both operands carry one. A reference compile's records name a weight scale
    shaped ``(K, C/block, kH, kW)`` and an input scale shaped
    ``(1, C/block, H, W)`` -- one value per block of ``block_size`` reduction
    channels, not one per output channel. Treating the weight's as per-channel
    is right only when the reduction is a single block, and wrong *silently*
    for every layer wider than that, so the extent is read from the tensor:

    * a scale with one entry per output channel is broadcast along the rest,
      which is the shape a caller writes by hand and what the tests use;
    * anything else is expanded against ``block_size``, which is the shape the
      quantizer produces.

    ``input_scale`` is optional because a caller that only has the weight's --
    an early lowering, a test -- should not have to invent one; absent means
    the activation is not vector scaled.

    Both scales are **uint8 mantissas**, not the quantizer's floats, and that
    is not a detail this operator can be relaxed about: the epilogue is
    encoded with ``vsq_operands=True``, which divides out the 65536 those two
    codes multiply in. Handed floats instead, the two halves disagree by that
    factor. ``lowering.finalize_alpha_vsq_operands`` is what guarantees it.
    """
    # Carried, not applied. ``folded_coarse_scale`` is the *epilogue's*
    # factor -- it is folded into the primary multiplier, and applying it here
    # as well would scale the accumulator twice. The semantic coarse scales
    # and the four endpoints are recorded so a reader can reconstruct the
    # split; nothing in the accumulator reads them. They ride on the operator
    # rather than in metadata because a record must carry them and metadata
    # does not survive fusion.
    del folded_coarse_scale
    del semantic_input_coarse_scale, semantic_weight_coarse_scale
    del input_fine_quant_max, input_coarse_scale_quant_max
    del weight_fine_quant_max, weight_coarse_scale_quant_max

    if input_scale is not None:
        if block_size is None:
            raise ValueError("a vector-scaled input requires block_size")
        return alpha_vsq_conv2d_accumulator(
            input,
            weight,
            input_scale,
            weight_scale,
            stride,
            padding,
            dilation,
            groups,
            block_size=block_size,
        ).to(ACCUMULATOR_DTYPE)

    # Compatibility for a manually constructed weight-only call. Compiled
    # Alpha VSQ records always carry both mantissas and take the exact path.
    scale = weight_scale.to(torch.float64)
    if scale.numel() == weight.shape[0]:
        scale = scale.reshape([-1] + [1] * (weight.dim() - 1))
    else:
        scale = expand(scale, weight.shape, block_size)
    return _conv_accumulate(
        input, weight.to(torch.float64) * scale,
        stride, padding, dilation, groups,
    )


magnet_alpha_lib.define(
    "ppu_fixed_relu(Tensor accumulator, Tensor primary_multiplier, "
    "int primary_shift, Tensor? bias, int secondary_multiplier, "
    "int secondary_shift, bool relu, int output_precision, *, "
    "bool integer_accumulator) -> Tensor"
)


@impl(magnet_alpha_lib, "ppu_fixed_relu", "CompositeExplicitAutograd")
def ppu_fixed_relu(
    accumulator: torch.Tensor,
    primary_multiplier: torch.Tensor,
    primary_shift: int,
    bias: Optional[torch.Tensor],
    secondary_multiplier: int,
    secondary_shift: int,
    relu: bool,
    output_precision: int,
    *,
    integer_accumulator: bool,
) -> torch.Tensor:
    """The matrix unit's epilogue, in the endpoint's own order.

    Delegated rather than restated: `execute_alpha_requantization` is the
    stage order and the two int16 wraps, and it reproduces every
    requantization record of a reference compile.  This target operator has
    no floating-accumulator mode, so the explicit instruction fact is
    required and may only be true.
    """
    from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
        AlphaRequantizationPipeline,
        execute_alpha_requantization,
    )

    if integer_accumulator is not True:
        raise ValueError("Alpha PPU requires an integer accumulator")

    return execute_alpha_requantization(
        accumulator,
        AlphaRequantizationPipeline.from_encoded(
            primary_multiplier=primary_multiplier,
            primary_shift=primary_shift,
            bias=bias,
            secondary_multiplier=secondary_multiplier,
            secondary_shift=secondary_shift,
            relu=relu,
            output_precision=output_precision,
        ),
    )


magnet_alpha_lib.define(
    "nmp_fixed_requantize(Tensor accumulator, int scale_factor, "
    "int binary_point, int bias_factor, bool relu, int output_precision) "
    "-> Tensor"
)


@impl(magnet_alpha_lib, "nmp_fixed_requantize", "CompositeExplicitAutograd")
def nmp_fixed_requantize(
    accumulator: torch.Tensor,
    scale_factor: int,
    binary_point: int,
    bias_factor: int,
    relu: bool,
    output_precision: int,
) -> torch.Tensor:
    """The near-memory unit's epilogue: one scalar multiplier and one shift.

    Simpler than the matrix unit's on purpose -- there is no per-channel
    vector here, and the bias is a single saturating int8 term.
    """
    if output_precision not in (4, 8):
        raise ValueError("Alpha NMP output precision must be 4 or 8")
    codes = alpha_nmp_epilogue(
        accumulator,
        scale_factor,
        binary_point,
        bias_factor,
        relu=relu,
    )
    if output_precision == 4:
        # As on the PE path, O4 describes the payload written by the
        # following PVS stage.  Its input remains wide so the block mantissa
        # is selected from the actual postprocessed magnitude.
        return codes
    limit = (1 << (output_precision - 1)) - 1
    # int8, not int32, and not because 4-bit values need eight of them: this
    # is the width the *matrix* unit's epilogue returns, and two epilogues
    # producing the same kind of value in different types is how a lowering
    # ends up inserting a cast nobody asked for.
    return codes.clamp(-limit - 1, limit).to(ACTIVATION_DTYPE)


magnet_alpha_lib.define(
    "pvs_quantize(Tensor input, Tensor coarse_scale, *, "
    "int output_precision, int block_size) -> (Tensor, Tensor, Tensor)"
)


@impl(magnet_alpha_lib, "pvs_quantize", "CompositeExplicitAutograd")
def pvs_quantize(
    input: torch.Tensor,
    coarse_scale: torch.Tensor,
    *,
    output_precision: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-vector stage: a coarse scale, a mantissa, and a payload.

    **Three results, in the order ``quantized_ops.quantize_vsq`` returns
    them** -- ``(coarse_scale, fine_scale, payload)``. That is not imitation:
    this stage *replaces* that quantizer in the graph, and returning the same
    tuple in the same order means every consumer's ``getitem`` index keeps
    its meaning, so the replacement is a substitution rather than a rewrite
    of everything downstream.

    Returning one tensor -- the payload -- is what this used to do, and it
    could not work. A per-vector output is a payload *and* the companion
    scale the next convolution reads as its ``input_scale``; a stage that
    emits only the first leaves the original quantizer in the graph to supply
    the second, and that quantizer holds the whole pre-lowering float chain
    alive behind it. *Measured* on the yolo lane: 30 convolutions survived a
    pass that reported all 85 rewritten, and operator fusion then met an int8
    activation and a float bias.

    The coarse scale is passed through rather than recomputed. It is an
    operand of the record -- the endpoint is programmed with it -- and
    returning it is what lets index 0 mean what it meant before.

    Args:
        input: The epilogue's integer output, NCHW.
        coarse_scale: The output domain's coarse scale, already rebalanced
            into Alpha's split. One value.
        output_precision: 4 or 8.
        block_size: How many channels share one mantissa. The machine's, and
            checked rather than trusted.

    Returns:
        ``(coarse_scale, mantissa, payload)``.

    Raises:
        ValueError: If the input is not NCHW, the block size is not the
            machine's, or the coarse scale is not one positive finite value.
    """
    from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
        alpha_pvs_quantize,
    )

    _check_pvs_arguments(input, coarse_scale, output_precision, block_size)
    encoded = alpha_pvs_quantize(
        input, output_precision=output_precision, vector_dim=1
    )
    return coarse_scale, encoded.mantissa, encoded.payload


def _check_pvs_arguments(input, coarse_scale, output_precision, block_size):
    """What the per-vector stage requires of its arguments.

    Shared by the implementation and the fake rule so the two cannot come to
    disagree about what is legal -- a divergence that yields a graph which
    traces and then raises when it runs.
    """
    from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
        VSQ_BLOCK_VALUES,
    )

    if input.ndim != 4:
        raise ValueError("Alpha's per-vector stage reads one NCHW tensor")
    if block_size != VSQ_BLOCK_VALUES:
        raise ValueError(
            f"Alpha blocks {VSQ_BLOCK_VALUES} channels to a scale, not "
            f"{block_size}"
        )
    if output_precision not in (4, 8):
        raise ValueError("Alpha writes 4- or 8-bit payloads")
    if coarse_scale.numel() != 1:
        raise ValueError(
            "a per-vector output domain has one coarse scale; a per-channel "
            "one would be a different factor per output channel, which the "
            "block structure has no way to express"
        )


magnet_alpha_lib.define(
    "vsq_scale_encode(Tensor logical_scale, float fine_quant_max=7.0, "
    "float coarse_scale_quant_max=127.0) -> Tensor"
)


@impl(magnet_alpha_lib, "vsq_scale_encode", "CompositeExplicitAutograd")
def vsq_scale_encode(
    logical_scale: torch.Tensor,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
) -> torch.Tensor:
    """The uint8 mantissa Alpha stores in place of a float fine scale.

    An operator rather than a helper because the scale is not always a
    constant: a *weight*'s is known at conversion and becomes a ``get_attr``,
    while an activation's is produced at runtime, and only a graph node can
    encode that one. The same operator serves both, so there is one
    definition of the encoding rather than two that could disagree.

    Not a rounding. The quantizer already chose the code -- this reads it back
    out, and a scale that is off the codebook is refused rather than rounded
    to the nearest legal one. See ``fixed_point.encode_alpha_vsq_mantissa``.
    """
    from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
        ALPHA_VSQ_MANTISSA_MAX,
        alpha_vsq_rebalance,
        encode_alpha_vsq_mantissa,
    )

    return encode_alpha_vsq_mantissa(
        logical_scale,
        rebalance=alpha_vsq_rebalance(
            fine_quant_max, coarse_scale_quant_max, ALPHA_VSQ_MANTISSA_MAX
        ),
        mantissa_max=ALPHA_VSQ_MANTISSA_MAX,
    )


@torch.library.register_fake("magnet_alpha::vsq_scale_encode")
def _vsq_scale_encode_fake(
    logical_scale,
    fine_quant_max=ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max=ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
):
    del fine_quant_max, coarse_scale_quant_max
    return torch.empty_like(logical_scale, dtype=torch.uint8)


magnet_alpha_lib.define(
    "channel_shuffle(Tensor input1, Tensor input2, int groups) -> Tensor"
)


@impl(magnet_alpha_lib, "channel_shuffle", "CompositeExplicitAutograd")
def channel_shuffle(
    input1: torch.Tensor, input2: torch.Tensor, groups: int
) -> torch.Tensor:
    """Concatenate two branches and interleave the channels.

    **Two inputs, one output.** The two inputs are real -- the operator
    absorbs the ``cat`` that would otherwise precede it, because on Alpha the
    join is where the branches are written rather than a movement of its own.
    The single output is equally deliberate: a reference record is
    ``channel_shuffle`` followed by a *separate* ``chunk``, so a split here
    would produce two graph values where the endpoint has one.

    The record's ``output0_memory_instance`` / ``output1_memory_instance`` are
    what makes that tempting, and they are not two values: they are two memory
    instances one output spans, which is placement and arrives from
    allocation.
    """
    joined = torch.cat((input1, input2), dim=1)
    batch, channels, height, width = joined.shape
    return (
        joined.reshape(batch, groups, channels // groups, height, width)
        .transpose(1, 2)
        .reshape(batch, channels, height, width)
    )


magnet_alpha_lib.define("channel_shuffle_copy(Tensor input) -> Tensor")


@impl(magnet_alpha_lib, "channel_shuffle_copy", "CompositeExplicitAutograd")
def channel_shuffle_copy(input: torch.Tensor) -> torch.Tensor:
    """One block's worth of movement.

    The value is unchanged: a copy is where bytes go, not what they are. It
    exists as an operator because the endpoint issues it as one, and because
    its schedule -- one channel group, whatever the tensor's width -- is a
    record the converter emits.
    """
    return input.clone()


magnet_alpha_lib.define(
    "upsample_nearest2d(Tensor input, SymInt[]? output_size, "
    "float[]? scale_factors) -> Tensor"
)


@impl(magnet_alpha_lib, "upsample_nearest2d", "CompositeExplicitAutograd")
def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> torch.Tensor:
    """Replicate Alpha's physical signed-INT8 payload byte for byte.

    PyTorch's CPU nearest-neighbour kernel does not accept ``torch.int8``.
    Alpha's NMP operation only replicates stored codes, so converting those
    bytes modulo 256 to ``uint8``, resizing, and converting them back is an
    exact executable reference.  Other dtypes keep ATen's ordinary semantics.
    """
    if input.dtype != torch.int8:
        return torch.ops.aten.upsample_nearest2d.vec(
            input, output_size, scale_factors
        )
    unsigned = input.to(torch.uint8)
    output = torch.ops.aten.upsample_nearest2d.vec(
        unsigned, output_size, scale_factors
    )
    return output.to(torch.int8)


# ---------------------------------------------------------------------------
# Fake-tensor rules. Shapes only; a fake tensor has no data to requantize.
# ---------------------------------------------------------------------------


def _conv_shape(activation, weight, stride, padding, dilation, groups):
    del groups
    batch, _, height, width = activation.shape
    out_channels = weight.shape[0]
    extents = []
    for index, size in enumerate((height, width)):
        effective = dilation[index] * (weight.shape[2 + index] - 1) + 1
        extents.append((size + 2 * padding[index] - effective)
                       // stride[index] + 1)
    return (batch, out_channels, *extents)


@torch.library.register_fake("magnet_alpha::conv2d_int8_accumulator")
def _conv2d_int8_accumulator_fake(
    input, weight, stride, padding, dilation, groups
):
    return torch.empty(
        _conv_shape(input, weight, stride, padding, dilation, groups),
        dtype=ACCUMULATOR_DTYPE,
        device=input.device,
    )


@torch.library.register_fake("magnet_alpha::conv2d_vsq")
def _conv2d_vsq_fake(
    input, weight, weight_scale, stride, padding, dilation, groups,
    *, input_scale=None, block_size=None, folded_coarse_scale=None,
    semantic_input_coarse_scale=None, semantic_weight_coarse_scale=None,
    input_fine_quant_max=ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    input_coarse_scale_quant_max=ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    weight_fine_quant_max=ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    weight_coarse_scale_quant_max=ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
):
    del weight_scale, input_scale, block_size, folded_coarse_scale
    del semantic_input_coarse_scale, semantic_weight_coarse_scale
    del input_fine_quant_max, input_coarse_scale_quant_max
    del weight_fine_quant_max, weight_coarse_scale_quant_max
    return torch.empty(
        _conv_shape(input, weight, stride, padding, dilation, groups),
        dtype=ACCUMULATOR_DTYPE,
        device=input.device,
    )


@torch.library.register_fake("magnet_alpha::ppu_fixed_relu")
def _ppu_fixed_relu_fake(
    accumulator, primary_multiplier, primary_shift, bias,
    secondary_multiplier, secondary_shift, relu, output_precision, *,
    integer_accumulator,
):
    if integer_accumulator is not True:
        raise ValueError("Alpha PPU requires an integer accumulator")
    if output_precision not in (4, 8):
        raise ValueError("Alpha PPU output precision must be 4 or 8")
    dtype = torch.int64 if output_precision == 4 else ACTIVATION_DTYPE
    del (primary_multiplier, primary_shift, bias, secondary_multiplier,
         secondary_shift, relu, output_precision, integer_accumulator)
    return torch.empty_like(accumulator, dtype=dtype)


@torch.library.register_fake("magnet_alpha::nmp_fixed_requantize")
def _nmp_fixed_requantize_fake(
    accumulator, scale_factor, binary_point, bias_factor, relu,
    output_precision,
):
    if output_precision not in (4, 8):
        raise ValueError("Alpha NMP output precision must be 4 or 8")
    dtype = torch.int64 if output_precision == 4 else ACTIVATION_DTYPE
    del scale_factor, binary_point, bias_factor, relu, output_precision
    return torch.empty_like(accumulator, dtype=dtype)


@torch.library.register_fake("magnet_alpha::pvs_quantize")
def _pvs_quantize_fake(input, coarse_scale, *, output_precision, block_size):
    """The three shapes, without reading a FakeTensor's data.

    The mantissa's channel extent is the *block count*, not the channel
    count: one scale per ``block_size`` channels, rounded up so a partial
    trailing block still has one.
    """
    _check_pvs_arguments(input, coarse_scale, output_precision, block_size)
    batch, channels, height, width = input.shape
    mantissa = torch.empty(
        (batch, (channels + block_size - 1) // block_size, height, width),
        dtype=torch.uint8,
        device=input.device,
    )
    return (
        coarse_scale,
        mantissa,
        torch.empty_like(input, dtype=ACTIVATION_DTYPE),
    )


@torch.library.register_fake("magnet_alpha::channel_shuffle")
def _channel_shuffle_fake(input1, input2, groups):
    del groups
    batch, _, height, width = input1.shape
    channels = input1.shape[1] + input2.shape[1]
    return torch.empty(
        (batch, channels, height, width),
        dtype=input1.dtype,
        device=input1.device,
    )


@torch.library.register_fake("magnet_alpha::channel_shuffle_copy")
def _channel_shuffle_copy_fake(input):
    return torch.empty_like(input)


@torch.library.register_fake("magnet_alpha::upsample_nearest2d")
def _upsample_nearest2d_fake(input, output_size, scale_factors):
    return upsample_nearest2d(input, output_size, scale_factors)
