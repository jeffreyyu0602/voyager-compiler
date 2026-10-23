# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Alpha's requantization: what the PPU and the NMP actually compute.

A matrix call ends in an integer accumulator, and something has to turn that
into the narrow codes the next operation reads. On Alpha that something is a
two-stage rescale with a bias between the stages, and the exact shape of it is
the reason this module exists rather than a single multiplier:

    normalized = (accumulator * primary)   >> primary_shift      # int16
    biased     = normalized + bias                               # int16
    activated  = max(biased, 0)   if relu
    output     = (activated * secondary) >> secondary_shift      # int4/int8

The **primary** stage lands the accumulator in a normalized intermediate. The
**secondary** stage moves that intermediate to the output's own scale. Two
stages rather than one, because the bias is added in between: it is a
per-channel constant in the *normalized* domain, and folding the two
multipliers together would leave nowhere to add it.

That intermediate domain is Q8.8 -- eight fractional bits, so a raw payload of
256 means one normalization unit. It has one degree of freedom, the
``normalization_scale``, and choosing it is :func:`select_alpha_pe_normalization`.

The bit widths and the back-off loop live in
:mod:`voyager_compiler.codegen.targets.magnet_fixed_point`, shared by Magnet
targets. Alpha adds what is Alpha's: which domain to normalize into, how the
vector scales change the storage factor, and where the bias goes.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from voyager_compiler.codegen.targets.magnet_fixed_point import (
    MagnetMultiplier,
    encode_magnet_multiplier,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
    VSQ_BLOCK_VALUES,
)
from voyager_compiler.fixed_point import FixedPointFormat, FixedPointRounding

__all__ = [
    "ALPHA_ACCUMULATOR_FORMAT",
    "ALPHA_BIAS_MAX",
    "ALPHA_BIAS_MIN",
    "ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT",
    "ALPHA_DEFAULT_VSQ_FINE_ENDPOINT",
    "ALPHA_INT16_INTERMEDIATE",
    "alpha_output_domain",
    "alpha_int8_conv2d_accumulator",
    "alpha_nmp_epilogue",
    "alpha_pe_postprocess",
    "ALPHA_Q8_8_FRACTIONAL_BITS",
    "ALPHA_Q8_8_NORMALIZATION",
    "ALPHA_Q8_8_STEP",
    "alpha_vsq_folded_coarse_scale",
    "alpha_vsq_conv2d_accumulator",
    "alpha_w8_accumulator_real_bound",
    "ALPHA_VSQ_MANTISSA_EMPTY",
    "ALPHA_VSQ_MANTISSA_MAX",
    "alpha_vsq_rebalance",
    "AlphaOutputDomain",
    "AlphaPEDirectHeadroomError",
    "AlphaPENormalization",
    "AlphaRequantizationPipeline",
    "AlphaVSQScaleEncoding",
    "encode_alpha_nmp_pipeline",
    "encode_alpha_pe_pipeline",
    "encode_alpha_vsq_mantissa",
    "encode_alpha_vsq_scales",
    "encode_bias",
    "execute_alpha_requantization",
    "primary_factors",
    "rebalance_alpha_vsq_coarse_scale",
    "select_alpha_pe_normalization",
]

#: The intermediate's binary point. A raw payload of ``2 ** 8`` is one
#: normalization unit.
ALPHA_Q8_8_FRACTIONAL_BITS = 8
ALPHA_Q8_8_STORAGE_FACTOR = 1 << ALPHA_Q8_8_FRACTIONAL_BITS

#: The normalization the datapath uses when it cannot use the output's own.
#: A payload of 256 stands for 16.0, so one code is 1/16.
ALPHA_Q8_8_NORMALIZATION = 16.0
ALPHA_Q8_8_STEP = ALPHA_Q8_8_NORMALIZATION / ALPHA_Q8_8_STORAGE_FACTOR

#: Vector-scale quantization carries a uint8 fine scale on *each* of the two
#: operands, and those two scales are already folded into the accumulator's
#: value. Relative to an 8-bit operand pair the accumulator is therefore
#: 256 x 256 larger, and the primary factor has to undo exactly that.
ALPHA_VSQ_STORAGE_FACTOR = ALPHA_Q8_8_STORAGE_FACTOR * ALPHA_Q8_8_STORAGE_FACTOR

#: The bias is assigned into a signed 8-bit datapath type. The configuration
#: word that carries it is wider; the type it lands in is not.
ALPHA_BIAS_MIN, ALPHA_BIAS_MAX = -128, 127

#: The intermediate between the two stages. It *wraps* rather than saturating
#: -- an accumulator that overflows its 16 bits wraps on this machine, and
#: clamping it here would hide an overflow the endpoint reproduces.
ALPHA_INT16_INTERMEDIATE = FixedPointFormat(
    storage_dtype=torch.int32,
    bit_width=16,
    saturate=False,
)

#: The MAC's own accumulator.
ALPHA_ACCUMULATOR_FORMAT = FixedPointFormat(
    storage_dtype=torch.int64,
    bit_width=32,
    saturate=False,
)


class AlphaPEDirectHeadroomError(ValueError):
    """Direct-output normalization would overflow the int16 intermediate.

    A recoverable condition, and the *only* recoverable one: it means the
    chosen normalization is too fine for this layer's range, so the fixed
    Q8.8 domain is used instead. A malformed qparam is not this, and must not
    be turned into it.
    """


@dataclass(frozen=True)
class AlphaOutputDomain:
    """Where the second stage lands, and what that implies.

    Alpha's policy, and it is a policy rather than a consequence: a 4-bit
    output is always paired with per-vector scaling, and an 8-bit output
    never is. Stating it here means a caller cannot assemble the other two
    combinations by accident.
    """

    scale: float
    bits: int
    uses_pvs: bool

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError("Alpha writes 4- or 8-bit outputs")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("an output scale must be finite and positive")
        if self.bits == 4 and not self.uses_pvs:
            raise ValueError("a 4-bit Alpha output is per-vector scaled")
        if self.bits == 8 and self.uses_pvs:
            raise ValueError("an 8-bit Alpha output is not per-vector scaled")

    @property
    def format(self) -> FixedPointFormat:
        return FixedPointFormat(
            storage_dtype=torch.int8,
            bit_width=self.bits,
            value_scale=self.scale,
            saturate=True,
        )


def alpha_output_domain(
    logical_scale,
    bits: int,
    uses_pvs: Optional[bool] = None,
    *,
    fine_quant_max: Optional[float] = None,
    coarse_scale_quant_max: Optional[float] = None,
) -> AlphaOutputDomain:
    """Build the physical output domain shared by Alpha PE and NMP paths.

    An O8 scale is already physical.  An O4/PVS observer instead records the
    logical coarse scale reconstructed from its payload and uint8 mantissa;
    the endpoint stores that scale with the VSQ rebalance divided out.  Do
    that conversion here so calibration and lowering cannot choose different
    interpretations of the same qparam.
    """
    if uses_pvs is None:
        uses_pvs = bits == 4
    if fine_quant_max is None:
        fine_quant_max = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
    if coarse_scale_quant_max is None:
        coarse_scale_quant_max = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
    scale = torch.as_tensor(logical_scale).detach().reshape(-1)
    if scale.numel() != 1:
        raise ValueError("an Alpha output scale must be scalar")
    if uses_pvs:
        scale = rebalance_alpha_vsq_coarse_scale(
            scale,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
    return AlphaOutputDomain(
        scale=float(scale[0]),
        bits=bits,
        uses_pvs=bool(uses_pvs),
    )


@dataclass(frozen=True)
class AlphaPENormalization:
    """The intermediate domain the first stage lands in.

    ``direct_output`` means the intermediate *is* the output's scale, which
    makes the second stage an exact identity and costs nothing. Otherwise the
    intermediate is the fixed Q8.8 domain and the second stage does real
    work.

    ``fallback_reason`` records why a direct normalization was not taken, so
    a lane that expected one can find out rather than infer it from the
    numbers.
    """

    normalization_scale: float
    direct_output: bool
    fallback_reason: Optional[str] = None

    @property
    def semantic_step(self) -> float:
        """The real value one unit of the intermediate represents."""
        return self.normalization_scale / ALPHA_Q8_8_STORAGE_FACTOR


def select_alpha_pe_normalization(
    output: AlphaOutputDomain,
    *,
    accumulator_scale,
    accumulator_bound=None,
    relu: bool,
) -> AlphaPENormalization:
    """Choose the first stage's domain.

    Direct-output normalization is taken only for a plain 8-bit boundary with
    no ReLU and no per-vector scaling. Everything else keeps the fixed Q8.8
    domain, for reasons that are each about a different stage:

    * **a 4-bit output** is per-vector scaled, so its scale is not one number
      the first stage could normalize into;
    * **a ReLU** clamps in the intermediate, between the stages, so the
      intermediate has to be a domain the clamp is meaningful in rather than
      the output's;
    * **too large a range** would wrap the int16 intermediate, which is the
      one recoverable case and is reported as such.

    ``accumulator_bound`` is the largest accumulator magnitude this layer can
    produce. Without it the headroom cannot be checked and the fixed domain
    is used, which is the conservative answer rather than an optimistic one.
    """
    if output.uses_pvs:
        return AlphaPENormalization(
            ALPHA_Q8_8_NORMALIZATION, False, "per-vector scaled output"
        )
    if relu:
        return AlphaPENormalization(
            ALPHA_Q8_8_NORMALIZATION, False, "relu clamps in the intermediate"
        )
    if accumulator_bound is None:
        return AlphaPENormalization(
            ALPHA_Q8_8_NORMALIZATION, False, "no accumulator bound was proved"
        )

    normalization = output.scale * ALPHA_Q8_8_STORAGE_FACTOR
    try:
        _require_int16_headroom(accumulator_scale, accumulator_bound, normalization)
    except AlphaPEDirectHeadroomError as error:
        return AlphaPENormalization(
            ALPHA_Q8_8_NORMALIZATION, False, str(error)
        )
    return AlphaPENormalization(normalization, True)


def _require_int16_headroom(accumulator_scale, bound, normalization):
    """Reject a real-domain bound the intermediate cannot hold."""
    scales = torch.as_tensor(accumulator_scale, dtype=torch.float64).reshape(-1)
    bounds = torch.as_tensor(bound, dtype=torch.float64).reshape(-1)
    if (
        scales.numel() == 0
        or not torch.isfinite(scales).all()
        or torch.any(scales <= 0)
    ):
        raise AlphaPEDirectHeadroomError("no channel is computed")
    if (
        bounds.numel() == 0
        or not torch.isfinite(bounds).all()
        or torch.any(bounds < 0)
    ):
        raise AlphaPEDirectHeadroomError(
            "the accumulator bound is not finite and nonnegative"
        )
    if bounds.numel() == 1:
        bounds = bounds.expand(scales.numel())
    elif scales.numel() == 1:
        scales = scales.expand(bounds.numel())
    elif bounds.numel() != scales.numel():
        raise AlphaPEDirectHeadroomError(
            "accumulator bounds do not match output channels"
        )
    # ``bound`` is already in real units. Multiplying by the accumulator
    # scale again would apply that scale twice.
    normalized = bounds * ALPHA_Q8_8_STORAGE_FACTOR / normalization
    limit = (1 << 15) - 1
    required = float(torch.amax(normalized))
    if required > limit:
        raise AlphaPEDirectHeadroomError(
            f"direct normalization needs {required:.0f} of the int16 "
            f"intermediate's {limit}"
        )


def primary_factors(
    accumulator_scale,
    normalization_scale: float,
    *,
    vsq_operands: bool,
) -> torch.Tensor:
    """The real ratio the first stage applies, per output channel.

    ``accumulator_scale`` is what one accumulator unit is worth -- the
    product of the two operands' scales. The stage has to turn that into the
    intermediate's units, which is a factor of ``storage / normalization``.

    A channel whose scale is zero is one the array computes only because the
    output-channel alignment required it -- the three detection heads declare
    48 channels and the datapath rounds them to 64. Its factor encodes as
    zero and the endpoint reads that as "nothing here", which is why zero is
    permitted per channel and refused for a lone multiplier.

    The storage factor is where vector-scale quantization shows up. Its two
    operands each carry a uint8 fine scale that is already multiplied into
    the accumulator, so the accumulator is 65536x larger than the same
    computation on plain 8-bit operands would leave it, and the factor undoes
    exactly that.
    """
    scales = torch.as_tensor(accumulator_scale, dtype=torch.float64).reshape(-1)
    if not torch.all(torch.isfinite(scales)) or bool((scales < 0).any()):
        raise ValueError("accumulator scales must be finite and non-negative")
    if bool((scales <= 0).all()):
        raise ValueError("at least one output channel must be computed")
    if not math.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError("a normalization scale must be finite and positive")
    storage = (
        ALPHA_VSQ_STORAGE_FACTOR if vsq_operands else ALPHA_Q8_8_STORAGE_FACTOR
    )
    return scales * (storage / normalization_scale)


def encode_bias(bias, normalization_scale: float) -> torch.Tensor:
    """Move a real bias into the intermediate's 8-bit field.

    The datapath adds its bias *between* the two stages, so the value it
    wants is the bias expressed in normalized units -- not the convolution's
    bias, and not the bias in accumulator units either. Getting this wrong is
    invisible at a unit normalization and wrong everywhere else.
    """
    values = torch.as_tensor(bias, dtype=torch.float64).reshape(-1)
    if not math.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError("a normalization scale must be finite and positive")
    scaled = torch.round(
        values * (ALPHA_Q8_8_STORAGE_FACTOR / normalization_scale)
    )
    return torch.clamp(scaled, ALPHA_BIAS_MIN, ALPHA_BIAS_MAX).to(torch.int64)


@dataclass(frozen=True)
class AlphaRequantizationPipeline:
    """Everything the epilogue needs, encoded."""

    primary: MagnetMultiplier
    bias: Tuple[int, ...]
    relu: bool
    secondary: MagnetMultiplier
    output: AlphaOutputDomain
    normalization: AlphaPENormalization

    @property
    def semantic_step(self) -> float:
        return self.normalization.semantic_step

    @classmethod
    def from_encoded(
        cls,
        *,
        primary_multiplier,
        primary_shift: int,
        secondary_multiplier: int,
        secondary_shift: int,
        bias=None,
        relu: bool = False,
        output_precision: int = 8,
    ) -> "AlphaRequantizationPipeline":
        """Rebuild a pipeline from the fields a *record* carries.

        ``encode_alpha_pe_pipeline`` goes the other way, from calibrated
        scales. This direction is what an operator needs: by the time
        ``ppu_fixed_relu`` runs, the choosing is over and only the encoded
        integers remain. Both must end up executing the same stages, which is
        why there is one pipeline type and not two.

        The domain fields are reconstructed as far as the encoded form allows
        and no further: an encoded record does not say which normalization was
        chosen, only what the multipliers came out as, so ``normalization_scale``
        is recorded as the fixed one and ``direct_output`` as unknown-false. No
        stage of the execution reads either -- they exist for diagnostics -- so
        this cannot silently change a result.
        """
        values = (
            tuple(int(v) for v in primary_multiplier)
            if hasattr(primary_multiplier, "__iter__")
            else int(primary_multiplier)
        )
        if hasattr(primary_multiplier, "tolist"):
            listed = primary_multiplier.reshape(-1).tolist()
            values = tuple(int(v) for v in listed)
            if len(values) == 1:
                values = values[0]
        if bias is None:
            encoded_bias: Tuple[int, ...] = ()
        elif hasattr(bias, "tolist"):
            encoded_bias = tuple(int(v) for v in bias.reshape(-1).tolist())
        else:
            encoded_bias = tuple(int(v) for v in bias)
        return cls(
            primary=MagnetMultiplier(values, int(primary_shift)),
            bias=encoded_bias,
            relu=bool(relu),
            secondary=MagnetMultiplier(
                int(secondary_multiplier), int(secondary_shift)
            ),
            output=AlphaOutputDomain(
                scale=1.0,
                bits=int(output_precision),
                uses_pvs=int(output_precision) == 4,
            ),
            normalization=AlphaPENormalization(
                normalization_scale=ALPHA_Q8_8_NORMALIZATION,
                direct_output=False,
                fallback_reason="reconstructed from an encoded record",
            ),
        )


def _expand_pe_channels(values, channels: int, description: str):
    values = torch.as_tensor(values).detach().reshape(-1)
    if values.numel() == 1:
        return values.expand(channels)
    if values.numel() != channels:
        raise AlphaPEDirectHeadroomError(
            f"{description} do not match output channels"
        )
    return values


def _require_encoded_direct_headroom(
    accumulator_scale,
    accumulator_real_bound,
    primary: MagnetMultiplier,
    encoded_bias,
) -> None:
    """Prove both int16 narrowing points for a direct-output pipeline."""
    bounds = torch.as_tensor(
        accumulator_real_bound, dtype=torch.float64
    ).detach().reshape(-1)
    scales = torch.as_tensor(
        accumulator_scale, dtype=torch.float64
    ).detach().reshape(-1)
    channels = max(bounds.numel(), scales.numel(), len(primary.values))
    bounds = _expand_pe_channels(bounds, channels, "accumulator bounds")
    scales = _expand_pe_channels(scales, channels, "accumulator scales")
    multipliers = _expand_pe_channels(
        primary.values, channels, "primary multipliers"
    ).to(torch.int64)
    biases = _expand_pe_channels(
        encoded_bias if encoded_bias else (0,),
        channels,
        "encoded biases",
    ).to(torch.int64)
    if (
        not torch.isfinite(bounds).all()
        or torch.any(bounds < 0)
        or not torch.isfinite(scales).all()
        or torch.any(scales <= 0)
    ):
        raise AlphaPEDirectHeadroomError(
            "direct normalization needs finite per-channel bounds and scales"
        )

    # Recover a conservative raw accumulator magnitude per channel. The
    # nextafter/ceil pair makes a prior floating multiplication incapable of
    # rounding an unsafe bound down by one code.
    raw_bounds = torch.ceil(
        torch.nextafter(
            bounds / scales,
            torch.full_like(bounds, torch.inf),
        )
    ).to(torch.int64)
    positive = torch.bitwise_right_shift(
        raw_bounds * multipliers, primary.shift
    )
    negative = torch.bitwise_right_shift(
        -raw_bounds * multipliers, primary.shift
    )
    lower, upper = -(1 << 15), (1 << 15) - 1
    for stage, low_values, high_values in (
        ("primary", negative, positive),
        ("post-bias", negative + biases, positive + biases),
    ):
        if torch.any(low_values < lower) or torch.any(high_values > upper):
            required = int(
                torch.maximum(
                    torch.amax(torch.abs(low_values)),
                    torch.amax(torch.abs(high_values)),
                ).item()
            )
            raise AlphaPEDirectHeadroomError(
                f"direct normalization {stage} needs {required} of the "
                f"signed-int16 range [{lower}, {upper}]"
            )


def encode_alpha_pe_pipeline(
    *,
    accumulator_scale,
    output: AlphaOutputDomain,
    bias=None,
    relu: bool,
    vsq_operands: bool,
    accumulator_bound=None,
) -> AlphaRequantizationPipeline:
    """Encode one matrix call's requantization.

    Returns the two multiplier/shift pairs, the 8-bit bias, and the
    intermediate domain that was chosen -- which the serializer emits as
    ``semantic_step`` so a reader can tell a direct normalization from the
    fixed one without recomputing the choice.
    """
    normalization = select_alpha_pe_normalization(
        output,
        accumulator_scale=accumulator_scale,
        accumulator_bound=accumulator_bound,
        relu=relu,
    )
    def encode_stages(selected):
        factors = primary_factors(
            accumulator_scale,
            selected.normalization_scale,
            vsq_operands=vsq_operands,
        )
        primary_stage = encode_magnet_multiplier(
            [float(value) for value in factors]
        )
        # At direct normalization this is exact unity; the multiplier
        # encoder's back-off makes that value fit its unsigned field.
        secondary_stage = encode_magnet_multiplier(
            selected.semantic_step / output.scale
        )
        bias_stage: Tuple[int, ...] = ()
        if bias is not None:
            bias_stage = tuple(
                int(value)
                for value in encode_bias(
                    bias, selected.normalization_scale
                )
            )
        return primary_stage, secondary_stage, bias_stage

    primary, secondary, encoded_bias = encode_stages(normalization)
    if normalization.direct_output:
        try:
            _require_encoded_direct_headroom(
                accumulator_scale,
                accumulator_bound,
                primary,
                encoded_bias,
            )
        except AlphaPEDirectHeadroomError as error:
            normalization = AlphaPENormalization(
                ALPHA_Q8_8_NORMALIZATION,
                False,
                str(error),
            )
            primary, secondary, encoded_bias = encode_stages(normalization)
    return AlphaRequantizationPipeline(
        primary=primary,
        bias=encoded_bias,
        relu=relu,
        secondary=secondary,
        output=output,
        normalization=normalization,
    )


def encode_alpha_nmp_pipeline(
    *,
    accumulator_scale: float,
    output_scale: float,
    output_bits: int,
    bias=None,
    relu: bool = False,
) -> Tuple[MagnetMultiplier, int]:
    """Encode one near-memory call's requantization.

    One stage, not two: the NMP applies ``(accumulator * scale) >> shift``
    directly. Its bias is a single scalar broadcast across the lanes, which
    is why a depthwise convolution's per-channel bias has to be projected
    onto one value before it ever reaches here.

    Returns ``(multiplier, encoded_bias)``.
    """
    if not math.isfinite(accumulator_scale) or accumulator_scale <= 0:
        raise ValueError("an accumulator scale must be finite and positive")
    if not math.isfinite(output_scale) or output_scale <= 0:
        raise ValueError("an output scale must be finite and positive")
    multiplier = encode_magnet_multiplier(accumulator_scale / output_scale)
    if any(value == 0 for value in multiplier.values):
        raise ValueError(
            "an Alpha NMP multiplier must be nonzero: the endpoint reads a "
            "zero as an implicit factor of one"
        )
    encoded_bias = 0
    if bias is not None:
        # Bias is quantized into accumulator codes before NMP applies its
        # accumulator-to-output ratio. Rounding the real bias directly here
        # would be dimensionally wrong and turns most YOLO biases into zero.
        real_bias = torch.as_tensor(bias, dtype=torch.float64).reshape(-1)
        if real_bias.numel() == 0 or not torch.isfinite(real_bias).all():
            raise ValueError("an Alpha NMP bias must be finite and nonempty")
        accumulator_bias = torch.round(real_bias / accumulator_scale)
        projected = torch.round(
            accumulator_bias * (accumulator_scale / output_scale)
        ).to(torch.int64)
        if not torch.all(projected == projected[0]):
            raise ValueError(
                "an Alpha NMP depthwise convolution requires one scalar bias"
            )
        encoded_bias = int(
            torch.clamp(
                projected[0], ALPHA_BIAS_MIN, ALPHA_BIAS_MAX
            ).item()
        )
    del output_bits, relu  # carried on the instruction, not in the encoding
    return multiplier, encoded_bias


# -----------------------------------------------------------------------------
# The arithmetic itself
# -----------------------------------------------------------------------------


def _shift_right(values: torch.Tensor, shift: int) -> torch.Tensor:
    """An arithmetic right shift, which is a floor and not a truncation.

    The distinction is only visible on negative values, where truncation
    rounds toward zero and a shift rounds down -- and half of a signed
    accumulator's values are negative.
    """
    return torch.floor_divide(values, 1 << shift)


def execute_alpha_requantization(
    accumulator: torch.Tensor,
    pipeline: AlphaRequantizationPipeline,
) -> torch.Tensor:
    """Reproduce the epilogue exactly, in integers.

    The stage order is the endpoint's and is not negotiable:

        accumulator -> * primary >> shift -> wrap int16
                    -> + bias             -> wrap int16
                    -> relu
                    -> * secondary >> shift
                    -> PVS, or saturate to the direct output width

    The two wraps are what make this bit-accurate rather than approximately
    right. An intermediate that exceeds int16 wraps on the silicon; computing
    in a wider type and clamping at the end agrees with it on typical data
    and disagrees exactly where it matters.
    """
    from voyager_compiler.fixed_point import quantize_fixed_point

    codes = accumulator.to(torch.int64)
    primary = torch.as_tensor(pipeline.primary.values, dtype=torch.int64)
    if primary.numel() > 1:
        shape = [1] * codes.dim()
        # Per output channel, which is dimension 1 of an NCHW activation.
        shape[1 if codes.dim() > 1 else 0] = primary.numel()
        primary = primary.reshape(shape)
    scaled = _shift_right(codes * primary, pipeline.primary.shift)
    scaled = quantize_fixed_point(
        scaled, ALPHA_INT16_INTERMEDIATE, already_scaled=True
    ).to(torch.int64)

    if pipeline.bias:
        bias = torch.as_tensor(pipeline.bias, dtype=torch.int64)
        if bias.numel() > 1:
            shape = [1] * scaled.dim()
            shape[1 if scaled.dim() > 1 else 0] = bias.numel()
            bias = bias.reshape(shape)
        scaled = quantize_fixed_point(
            scaled + bias, ALPHA_INT16_INTERMEDIATE, already_scaled=True
        ).to(torch.int64)

    if pipeline.relu:
        scaled = torch.clamp(scaled, min=0)

    secondary = torch.as_tensor(pipeline.secondary.values, dtype=torch.int64)
    if secondary.numel() > 1:
        shape = [1] * scaled.dim()
        shape[1 if scaled.dim() > 1 else 0] = secondary.numel()
        secondary = secondary.reshape(shape)
    output = _shift_right(scaled * secondary, pipeline.secondary.shift)
    if pipeline.output.uses_pvs:
        # O4 is not the width of this intermediate.  The following PVS stage
        # chooses a block mantissa from the wide value and only then emits an
        # O4 payload.  Truncating here makes every mantissa one and destroys
        # the dynamic per-vector scale.
        return output
    return quantize_fixed_point(
        output, pipeline.output.format, already_scaled=True
    )


# ---------------------------------------------------------------------------
# The per-vector stage: a payload and the companion mantissa beside it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaPVS:
    """What the per-vector stage produces: a narrow payload and its mantissa.

    Two tensors, because that is what the stage emits and what the next
    convolution reads -- the payload as its operand and the mantissa as its
    ``input_scale``. A stage modelled as returning one of them cannot supply
    the other, which is how the quantizer it replaces ends up surviving in
    the graph.
    """

    payload: torch.Tensor
    mantissa: torch.Tensor


def alpha_pvs_quantize(
    values: torch.Tensor,
    output_precision: int,
    vector_dim: int = -1,
) -> AlphaPVS:
    """Alpha's per-vector quantization, in integers, one block at a time.

    This is not a rescaling by a float. The datapath picks each block's
    mantissa from the block's own magnitude and divides by a *reciprocal it
    computes in integers*, so the encoding is a sequence of shifts and a
    floor -- and a float model of it agrees on most values and disagrees on
    the boundaries, which is exactly where an integer endpoint comparison
    looks.

    Per block of ``VSQ_BLOCK_VALUES`` along ``vector_dim``:

    * the mantissa is ``max(|v|) >> 3``, or 1 when the block fits in four
      bits -- a block whose values are all small needs no scaling, and a
      mantissa of zero would claim the block holds nothing;
    * the payload is ``(clamp(v) * (8192 // mantissa)) >> 13``, saturated to
      the output width.

    The trailing partial block is zero padded and the padding dropped again,
    so a channel count that is not a multiple of the block still encodes --
    the zeros cannot raise the block maximum, so they cost nothing.

    Args:
        values: The epilogue's integer output.
        output_precision: 4 or 8.
        vector_dim: Which axis the blocks run along. Channels, for an
            activation.

    Returns:
        The payload and its uint8 mantissa.

    Raises:
        ValueError: If the width is not one Alpha writes, or the tensor has
            no axis to block along.
    """
    if output_precision not in (4, 8):
        raise ValueError("Alpha writes 4- or 8-bit payloads")
    if values.ndim == 0:
        raise ValueError("a per-vector stage needs an axis to block along")
    vector_dim %= values.ndim
    moved = torch.movedim(values.to(torch.int64), vector_dim, -1)
    lanes = moved.shape[-1]
    padding = (-lanes) % VSQ_BLOCK_VALUES
    if padding:
        moved = torch.nn.functional.pad(moved, (0, padding))
    packets = moved.reshape(*moved.shape[:-1], -1, VSQ_BLOCK_VALUES)

    maximum = torch.amax(torch.abs(packets), dim=-1)
    mantissa = torch.where(
        maximum <= 7,
        torch.ones_like(maximum),
        torch.bitwise_right_shift(maximum, 3),
    )
    mantissa = torch.bitwise_and(mantissa, 0xFF)
    # A wrapped mantissa of zero would divide by zero below and would mean
    # "no scale" where the block does have one.
    mantissa = torch.where(mantissa == 0, 1, mantissa)
    reciprocal = torch.div(8192, mantissa, rounding_mode="floor")

    clipped = torch.clamp(packets, -8191, 8191)
    payload = _shift_right(clipped * reciprocal.unsqueeze(-1), 13)
    payload = alpha_truncate_payload(payload, output_precision).reshape(
        moved.shape
    )
    return AlphaPVS(
        payload=torch.movedim(payload[..., :lanes], -1, vector_dim),
        mantissa=torch.movedim(mantissa, -1, vector_dim).to(torch.uint8),
    )


def alpha_truncate_payload(
    values: torch.Tensor, output_precision: int
) -> torch.Tensor:
    """Saturate integer codes to the width Alpha writes.

    Always into ``int8`` storage: a 4-bit value rides in the same container
    and ``output_precision`` is what says which, exactly as
    ``ops.ACTIVATION_DTYPE`` records.
    """
    if output_precision not in (4, 8):
        raise ValueError("Alpha writes 4- or 8-bit payloads")
    limit = (1 << (output_precision - 1)) - 1
    return torch.clamp(values, -limit - 1, limit).to(torch.int8)


# ---------------------------------------------------------------------------
# Vector-scale quantization: the two scale levels, and the codes Alpha stores
# ---------------------------------------------------------------------------
#
# A VSQ operand is a narrow payload under *two* scales: a per-block ``fine``
# scale and a per-channel (or per-tensor) ``coarse`` one. The compiler's
# quantizer produces both as floats -- ``FakeQuantize.quantize_vsq_components``
# returns ``(coarse_scale, fine_scale, payload)``.
#
# Alpha does not store the fine scale as a float. It stores an **unsigned
# 8-bit mantissa**, and it recovers the value by folding the discarded range
# back into the coarse scale. That is the whole of the encoding:
#
#     mantissa     = fine_scale * rebalance
#     coarse_out   = coarse_scale / rebalance
#
# so ``mantissa * coarse_out == fine_scale * coarse_scale`` -- the represented
# value is unchanged, and only the *split* between the two levels moves.
#
# ``rebalance`` is fixed by the quantizer, not chosen here. Upstream's
# fine-scale codebook (``quantization.fake_quantize.get_vsq_scale_quantization
# _map``) quantizes the fine scale onto a grid of
#
#     step = coarse_scale_quant_max / (fine_quant_max * mantissa_max)
#
# and ``rebalance`` is exactly ``1 / step``. The mantissa is therefore not a
# rounding of the fine scale -- it is the integer the quantizer *already
# chose*, read back out. Recovering it with a ``round`` that then has to
# tolerate error would hide the case this is arranged to catch: a fine scale
# that never went through that codebook is off the grid, and dividing it
# yields a non-integer rather than a wrong integer.
#
# The emitted ``folded_coarse_scale`` is
# ``(input_coarse / rebalance_input) * (weight_coarse / rebalance_weight)``.
# The endpoints come from the pairing's ``sqmax`` and ``csqmax`` declarations,
# so this identity connects the quantizer declaration directly to the emitted
# collateral.

#: The mantissa is unsigned 8-bit. ``mantissa_max`` is a parameter rather than
#: this constant everywhere because it is the *codebook's* top code, and the
#: codebook is the quantizer's; a machine that stored a narrower mantissa
#: would rebalance by a different factor.
ALPHA_VSQ_MANTISSA_MAX = 255

#: A block whose values are all zero has no scale to store. The quantizer
#: emits ``0`` for it -- outside the codebook's ``[1, mantissa_max]`` range,
#: deliberately, because a code of one would claim the block holds a value
#: one step wide. It is admitted here for the same reason: refusing it would
#: reject a legal tensor, and clamping it up would resurrect a dead block.
ALPHA_VSQ_MANTISSA_EMPTY = 0

#: How far a code may sit from an integer after a float32 round trip. Four
#: ULPs of float32's 24-bit significand: ``code / rebalance`` is rounded once
#: on the way in and the product is rounded again on the way out, and the
#: divisor is not a power of two. Wide enough for both roundings, and far
#: narrower than the half-step that would be needed to hide a wrong code.
_FLOAT32_CODE_SLACK = 4 * 2.0 ** -24

#: What the pairing declares for a weight, and the default the endpoint's own
#: records carry. The activation side differs (8.0 / 126.0) and is passed
#: explicitly; there is no default that is right for both.
ALPHA_DEFAULT_VSQ_FINE_ENDPOINT = 7.0
ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT = 127.0


def alpha_vsq_rebalance(
    fine_quant_max: float,
    coarse_scale_quant_max: float,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> float:
    """The factor moved from the fine scale into the coarse one.

    The reciprocal of the fine-scale codebook's step, stated as a
    multiplication because that is the direction the encoder reads it.

    Args:
        fine_quant_max: The fine level's positive endpoint (``sqmax``).
        coarse_scale_quant_max: The coarse level's (``csqmax``).
        mantissa_max: The codebook's top code.

    Returns:
        ``mantissa_max * fine_quant_max / coarse_scale_quant_max``.

    Raises:
        ValueError: If an endpoint is not finite and positive, or the
            mantissa maximum does not fit unsigned 8 bits.
    """
    for value, name in (
        (fine_quant_max, "fine_quant_max"),
        (coarse_scale_quant_max, "coarse_scale_quant_max"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 1 <= mantissa_max <= ALPHA_VSQ_MANTISSA_MAX:
        raise ValueError(
            f"mantissa_max {mantissa_max} must fit unsigned 8 bits and be "
            f"positive"
        )
    return mantissa_max * fine_quant_max / coarse_scale_quant_max


def encode_alpha_vsq_mantissa(
    fine_scale: torch.Tensor,
    *,
    rebalance: float,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> torch.Tensor:
    """Read the codebook index back out of one quantized fine scale.

    Args:
        fine_scale: The quantizer's fine scale, already on its codebook.
        rebalance: From :func:`alpha_vsq_rebalance`, with the same endpoints
            the fine scale was quantized under.
        mantissa_max: The codebook's top code.

    Returns:
        The unsigned 8-bit mantissa, one code per block.

    Raises:
        ValueError: If a scale is negative or not finite, or if one does not
            land on the codebook -- which means it was quantized under
            different endpoints, or not quantized at all. Either is a real
            defect and neither is recoverable by rounding: the coarse scale
            has already been divided by *this* ``rebalance``, so a mantissa
            that is off by one is a value that is off by one step.
    """
    values = fine_scale.detach()
    if not values.is_floating_point():
        values = values.to(torch.float64)
    if values.numel() == 0:
        raise ValueError("a VSQ fine scale must be nonempty")
    if not torch.isfinite(values).all():
        raise ValueError("a VSQ fine scale must be finite")
    if bool(torch.any(values < 0)):
        raise ValueError("a VSQ fine scale must be nonnegative")

    exact = values.to(torch.float64) * float(rebalance)
    codes = torch.round(exact)
    # The codebook makes this an integer *in exact arithmetic*. It is not one
    # here, because the scale was stored as float32 -- ``code / rebalance``
    # rounded to 24 bits and multiplied back does not land on the code. So the
    # admissible residual is what that storage can introduce and no more:
    # a few ULPs of the code's own magnitude. An absolute tolerance would be
    # simultaneously too tight for code 255 and too loose for code 1.
    residual = torch.abs(exact - codes)
    allowed = torch.clamp(torch.abs(exact), min=1.0) * _FLOAT32_CODE_SLACK
    if bool(torch.any(residual > allowed)):
        worst = int(torch.argmax(residual - allowed))
        raise ValueError(
            f"VSQ fine scale {float(values.reshape(-1)[worst])!r} is not on "
            f"the codebook for rebalance {rebalance!r}: it decodes to "
            f"{float(exact.reshape(-1)[worst])!r}, which is not an integer "
            f"code. The scale was quantized under different endpoints, or "
            f"not quantized at all"
        )
    if bool(torch.any(codes > mantissa_max)):
        raise ValueError(
            f"VSQ mantissa {int(torch.max(codes))} exceeds the codebook's "
            f"top code {mantissa_max}"
        )
    return codes.to(torch.uint8)


@dataclass(frozen=True)
class AlphaVSQScaleEncoding:
    """One operand's two scale levels, as Alpha stores them."""

    #: Unsigned 8-bit, one code per block. ``0`` marks an empty block.
    mantissa: torch.Tensor
    #: The coarse scale with ``rebalance`` divided out of it.
    coarse_scale: torch.Tensor
    #: The factor that moved between the two.
    rebalance: float


def encode_alpha_vsq_scales(
    fine_scale: torch.Tensor,
    coarse_scale: torch.Tensor,
    *,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> AlphaVSQScaleEncoding:
    """Split one operand's scales the way Alpha's datapath reads them.

    Args:
        fine_scale: The quantizer's per-block fine scale.
        coarse_scale: Its per-channel or per-tensor coarse scale.
        fine_quant_max: The fine endpoint the pairing declared (``sqmax``).
        coarse_scale_quant_max: The coarse endpoint (``csqmax``).
        mantissa_max: The fine codebook's top code.

    Returns:
        The mantissa, the rebalanced coarse scale, and the factor between.

    Raises:
        ValueError: If the coarse scale is empty, not finite, or negative, or
            if the fine scale is not on its codebook.
    """
    rebalance = alpha_vsq_rebalance(
        fine_quant_max, coarse_scale_quant_max, mantissa_max
    )
    return AlphaVSQScaleEncoding(
        mantissa=encode_alpha_vsq_mantissa(
            fine_scale, rebalance=rebalance, mantissa_max=mantissa_max
        ),
        coarse_scale=rebalance_alpha_vsq_coarse_scale(
            coarse_scale,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
            mantissa_max=mantissa_max,
        ),
        rebalance=rebalance,
    )


def rebalance_alpha_vsq_coarse_scale(
    coarse_scale: torch.Tensor,
    *,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> torch.Tensor:
    """The coarse half of the split, on its own.

    Separate from :func:`encode_alpha_vsq_scales` because the two halves are
    independent, and a caller that has only the coarse scale is common: a
    record's ``folded_coarse_scale`` is a property of the two coarse scales
    and their endpoints, and does not involve a mantissa at all. Folding this
    into the pair encoder would make such a caller invent a fine scale, which
    the encoder would then -- correctly -- refuse for not being on a codebook.

    Args:
        coarse_scale: The quantizer's coarse scale.
        fine_quant_max: The fine endpoint the pairing declared (``sqmax``).
        coarse_scale_quant_max: The coarse endpoint (``csqmax``).
        mantissa_max: The fine codebook's top code.

    Returns:
        ``coarse_scale`` divided by the rebalance factor.

    Raises:
        ValueError: If the scale is empty, not finite, or negative.
    """
    coarse = coarse_scale.detach()
    if not coarse.is_floating_point():
        coarse = coarse.to(torch.float64)
    if coarse.numel() == 0:
        raise ValueError("a VSQ coarse scale must be nonempty")
    if not torch.isfinite(coarse).all():
        raise ValueError("a VSQ coarse scale must be finite")
    if bool(torch.any(coarse < 0)):
        raise ValueError("a VSQ coarse scale must be nonnegative")
    return coarse / alpha_vsq_rebalance(
        fine_quant_max, coarse_scale_quant_max, mantissa_max
    )


def alpha_vsq_folded_coarse_scale(
    activation_coarse_scale: torch.Tensor,
    weight_coarse_scale: torch.Tensor,
) -> torch.Tensor:
    """The one coarse scale a matrix record carries.

    Both operands' coarse scales multiply the same accumulator and the
    datapath applies one factor rather than two, so the record carries their
    product, per output channel.

    Args:
        activation_coarse_scale: The activation's *rebalanced* coarse scale,
            from :func:`rebalance_alpha_vsq_coarse_scale` under the
            activation's own endpoints.
        weight_coarse_scale: The weight's, under the weight's -- which are
            different numbers, and that is the point of taking two arguments
            rather than one pair of endpoints.

    Returns:
        ``folded_coarse_scale``, one value per output channel.

    Raises:
        ValueError: If the activation's coarse scale is not a single value.
            Per-channel there would mean a different factor per *input*
            channel, which no single multiply can apply after a reduction.
    """
    scale = activation_coarse_scale.reshape(-1)
    if scale.numel() != 1:
        raise ValueError(
            f"a VSQ activation's coarse scale is applied once after the "
            f"reduction, so it must be one value; this one has "
            f"{scale.numel()}"
        )
    return scale[0] * weight_coarse_scale.reshape(-1)


def _signed_wrap(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply one signed two's-complement width in an int64 container."""
    modulus = 1 << bits
    half = modulus >> 1
    values = torch.remainder(values.to(torch.int64), modulus)
    return torch.where(values >= half, values - modulus, values)


def _pair_argument(value, name: str, *, allow_zero: bool) -> Tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be one integer or a pair")
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in pair):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {qualifier}")
    return pair


def _integer_codes(
    values: torch.Tensor,
    name: str,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        if not torch.equal(values, torch.round(values)):
            raise ValueError(f"{name} must contain integer values")
    result = values.to(torch.int64)
    if torch.any((result < minimum) | (result > maximum)):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def alpha_w8_accumulator_real_bound(
    weight_payload: torch.Tensor,
    accumulator_scale: torch.Tensor,
    *,
    input_code_amax: int = 128,
) -> torch.Tensor:
    """Conservative per-output-channel real bound for a W8 PE dot."""
    if weight_payload.ndim < 2:
        raise ValueError("an Alpha W8 bound requires matrix weights")
    weight = _integer_codes(
        weight_payload, "Alpha W8 weight", -128, 127
    ).to(torch.float64)
    raw = torch.sum(
        torch.abs(weight) * float(input_code_amax),
        dim=tuple(range(1, weight.ndim)),
    )
    raw = torch.clamp_max(raw, float(1 << 31))
    scale = accumulator_scale.detach().to(torch.float64).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand_as(raw)
    if (
        scale.shape != raw.shape
        or not torch.isfinite(scale).all()
        or torch.any(scale <= 0)
    ):
        raise ValueError("W8 accumulator scales must match output channels")
    return raw * scale


def alpha_int8_conv2d_accumulator(
    input_payload: torch.Tensor,
    weight_payload: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    """Bias-free signed-W8 convolution with wrapping int32 accumulation."""
    if input_payload.ndim != 4 or weight_payload.ndim != 4:
        raise ValueError("Alpha W8 Conv2d expects NCHW and KCRS operands")
    inputs = _integer_codes(input_payload, "input payload", -128, 127)
    weights = _integer_codes(weight_payload, "weight payload", -128, 127)
    accumulated = torch.nn.functional.conv2d(
        inputs.to(torch.float64),
        weights.to(torch.float64),
        None,
        _pair_argument(stride, "stride", allow_zero=False),
        _pair_argument(padding, "padding", allow_zero=True),
        _pair_argument(dilation, "dilation", allow_zero=False),
        int(groups),
    )
    return _signed_wrap(torch.round(accumulated).to(torch.int64), 32)


def alpha_vsq_conv2d_accumulator(
    input_payload: torch.Tensor,
    weight_payload: torch.Tensor,
    input_mantissa: torch.Tensor,
    weight_mantissa: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    block_size: int = VSQ_BLOCK_VALUES,
) -> torch.Tensor:
    """Run block dots, truncated scale products, then an int32 MAC."""
    if input_payload.ndim != 4 or weight_payload.ndim != 4:
        raise ValueError("Alpha VSQ Conv2d expects NCHW and KCRS operands")
    if groups != 1 or block_size != VSQ_BLOCK_VALUES:
        raise ValueError("Alpha VSQ Conv2d requires groups=1 and 32 lanes")
    inputs = _integer_codes(input_payload, "input payload", -8, 7)
    weights = _integer_codes(weight_payload, "weight payload", -7, 7)
    input_scales = _integer_codes(
        input_mantissa, "input mantissa", 0, 255
    )
    weight_scales = _integer_codes(
        weight_mantissa, "weight mantissa", 0, 255
    )
    batch, channels, input_h, input_w = inputs.shape
    outputs, weight_channels, kernel_h, kernel_w = weights.shape
    if channels != weight_channels or channels % block_size:
        raise ValueError("Alpha VSQ channels must match and be block aligned")
    blocks = channels // block_size
    if tuple(input_scales.shape) != (batch, blocks, input_h, input_w):
        raise ValueError("Alpha input mantissa has the wrong shape")
    expected_weight_scales = (
        outputs,
        blocks,
        kernel_h,
        kernel_w,
    )
    if tuple(weight_scales.shape) != expected_weight_scales:
        raise ValueError("Alpha weight mantissa has the wrong shape")

    stride_h, stride_w = _pair_argument(
        stride, "stride", allow_zero=False
    )
    pad_h, pad_w = _pair_argument(padding, "padding", allow_zero=True)
    dilation_h, dilation_w = _pair_argument(
        dilation, "dilation", allow_zero=False
    )
    output_h = (
        input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if output_h <= 0 or output_w <= 0:
        raise ValueError("Alpha VSQ Conv2d has an empty output")

    padded = torch.nn.functional.pad(inputs, (pad_w, pad_w, pad_h, pad_h))
    padded_scales = torch.nn.functional.pad(
        input_scales, (pad_w, pad_w, pad_h, pad_h)
    )
    accumulator = torch.zeros(
        (batch, outputs, output_h, output_w),
        dtype=torch.int64,
        device=inputs.device,
    )
    elements = max(1, batch * blocks * output_h * output_w)
    output_chunk = max(1, (8 * 1024 * 1024) // elements)
    blocked_weight = weights.reshape(
        outputs, blocks, block_size, kernel_h, kernel_w
    )
    for kernel_row in range(kernel_h):
        row_start = kernel_row * dilation_h
        row_stop = row_start + output_h * stride_h
        for kernel_column in range(kernel_w):
            column_start = kernel_column * dilation_w
            column_stop = column_start + output_w * stride_w
            patch = padded[
                :,
                :,
                row_start:row_stop:stride_h,
                column_start:column_stop:stride_w,
            ].reshape(batch, blocks, block_size, output_h, output_w)
            scale_patch = padded_scales[
                :,
                :,
                row_start:row_stop:stride_h,
                column_start:column_stop:stride_w,
            ]
            for start in range(0, outputs, output_chunk):
                stop = min(outputs, start + output_chunk)
                kernel = blocked_weight[
                    start:stop, :, :, kernel_row, kernel_column
                ]
                dot = torch.einsum("ncbhw,ocb->ncohw", patch, kernel)
                kernel_scales = weight_scales[
                    start:stop, :, kernel_row, kernel_column
                ].transpose(0, 1)
                product = torch.bitwise_right_shift(
                    scale_patch[:, :, None]
                    * kernel_scales[None, :, :, None, None],
                    8,
                )
                accumulator[:, start:stop] += torch.sum(
                    dot * product, dim=1
                )
    return _signed_wrap(accumulator, 32)


def alpha_pe_postprocess(
    accumulator: torch.Tensor,
    primary_multiplier: torch.Tensor,
    primary_shift: int,
    bias: torch.Tensor,
    secondary_multiplier: int,
    secondary_shift: int,
    *,
    relu: bool,
) -> torch.Tensor:
    """Execute PE stages up to the integer consumed by O8 or PVS."""
    values = _shift_right(
        accumulator.to(torch.int64) * primary_multiplier.to(torch.int64),
        primary_shift,
    )
    values = _signed_wrap(values, 16)
    values = _signed_wrap(values + bias.to(torch.int64), 16)
    if relu:
        values = torch.clamp_min(values, 0)
    return _shift_right(values * int(secondary_multiplier), secondary_shift)


def alpha_nmp_epilogue(
    accumulator: torch.Tensor,
    scale_factor: int,
    binary_point: int,
    bias_factor: int,
    *,
    relu: bool,
) -> torch.Tensor:
    """Execute the signed-int24 NMP accumulator and scalar postprocess."""
    values = _signed_wrap(accumulator, 24)
    values = _shift_right(values * int(scale_factor), int(binary_point))
    values = _signed_wrap(values, 16)
    values = _signed_wrap(values + int(bias_factor), 16)
    return torch.clamp_min(values, 0) if relu else values
