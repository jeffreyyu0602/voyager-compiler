# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed-point formats, and the arithmetic that lands a value in one.

An accelerator's datapath does not hold real numbers. It holds integers with
an agreed binary point, and every quantized operation is ultimately a
statement about where that point is and what happens at the edges. This
module states those two things once:

* :class:`FixedPointFormat` -- storage width, fractional bits, and the
  logical scale one unit represents;
* :func:`quantize_fixed_point` / :func:`reconstruct_fixed_point` -- the round
  trip through that format, with the rounding and saturation the hardware
  actually performs.

Two details in here are not conveniences, and getting either wrong produces
numbers that look plausible and are not what the endpoint computes.

**Saturation is about the logical width, not the container's.** A 24-bit
accumulator held in an ``int32`` wraps at 24 bits. Letting the container
decide would silently give it eight bits of headroom the silicon does not
have.

**Rounding is a property of the format.** A datapath that truncates toward
zero and one that rounds half to even disagree on half of all boundary
values, and the disagreement is systematic rather than noise: it is a bias.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

__all__ = [
    "FixedPointFormat",
    "FixedPointRounding",
    "quantize_fixed_point",
    "reconstruct_fixed_point",
]


class FixedPointRounding(str, Enum):
    """How a value that falls between two representable ones is resolved."""

    NEAREST_EVEN = "nearest_even"
    NEAREST_AWAY_FROM_ZERO = "nearest_away_from_zero"
    TOWARD_POSITIVE_INFINITY = "nearest_toward_positive_infinity"
    TOWARD_ZERO = "toward_zero"
    FLOOR = "floor"


def _round(values: torch.Tensor, rounding: FixedPointRounding) -> torch.Tensor:
    if rounding is FixedPointRounding.NEAREST_EVEN:
        return torch.round(values)
    if rounding is FixedPointRounding.NEAREST_AWAY_FROM_ZERO:
        return torch.sign(values) * torch.floor(torch.abs(values) + 0.5)
    if rounding is FixedPointRounding.TOWARD_POSITIVE_INFINITY:
        return torch.floor(values + 0.5)
    if rounding is FixedPointRounding.TOWARD_ZERO:
        return torch.trunc(values)
    return torch.floor(values)


@dataclass(frozen=True)
class FixedPointFormat:
    """One integer domain, and what a unit of it represents.

    The logical step -- the real value one code differs by -- is
    ``value_scale * 2 ** -fractional_bits``. Splitting it into those two
    factors rather than carrying the product is what lets a Q8.8 payload keep
    its binary point while its normalization moves: the endpoint's shift is
    ``fractional_bits`` and is fixed by the datapath, while ``value_scale`` is
    calibrated.

    ``saturate=False`` wraps modulo ``2 ** bit_width``, which is a real mode:
    an accumulator that overflows wraps, it does not clamp, and modelling it
    as clamping hides the overflow instead of reproducing it.
    """

    storage_dtype: torch.dtype
    bit_width: int
    fractional_bits: int = 0
    value_scale: float = 1.0
    signed: bool = True
    rounding: FixedPointRounding = FixedPointRounding.NEAREST_EVEN
    saturate: bool = True

    def __post_init__(self):
        object.__setattr__(self, "rounding", FixedPointRounding(self.rounding))
        if self.bit_width <= 0:
            raise ValueError("bit_width must be positive")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must not be negative")
        if not self.signed and self.fractional_bits >= self.bit_width:
            raise ValueError("an unsigned format needs an integer bit")
        if self.value_scale <= 0:
            raise ValueError("value_scale must be positive")
        info = torch.iinfo(self.storage_dtype)
        if self.bit_width > info.bits:
            raise ValueError(
                f"a {self.bit_width}-bit format does not fit "
                f"{self.storage_dtype}"
            )

    @property
    def step(self) -> float:
        """The real value one code represents."""
        return self.value_scale * (2.0 ** -self.fractional_bits)

    @property
    def quant_min(self) -> int:
        return -(1 << (self.bit_width - 1)) if self.signed else 0

    @property
    def quant_max(self) -> int:
        width = self.bit_width - 1 if self.signed else self.bit_width
        return (1 << width) - 1


def _wrap(values: torch.Tensor, format: FixedPointFormat) -> torch.Tensor:
    """Reduce modulo the *logical* width, keeping the sign convention."""
    modulus = 1 << format.bit_width
    reduced = torch.remainder(values, modulus)
    if not format.signed:
        return reduced
    half = modulus >> 1
    return torch.where(reduced >= half, reduced - modulus, reduced)


def quantize_fixed_point(
    values: torch.Tensor,
    format: FixedPointFormat,
    *,
    already_scaled: bool = False,
) -> torch.Tensor:
    """Round ``values`` into ``format``'s integer codes.

    Args:
        values: Real values, or raw codes when ``already_scaled``.
        format: The domain to land in.
        already_scaled: Skip the division by ``step``. Used when the caller
            has performed the scaling itself in the integer domain -- a
            multiplier and a shift -- and only wants the rounding and the
            range applied.
    """
    scaled = values if already_scaled else values / format.step
    codes = _round(scaled, format.rounding)
    if format.saturate:
        codes = torch.clamp(codes, format.quant_min, format.quant_max)
    else:
        codes = _wrap(codes, format)
    return codes.to(format.storage_dtype)


def reconstruct_fixed_point(
    codes: torch.Tensor,
    format: FixedPointFormat,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """The real values ``codes`` stand for."""
    return codes.to(dtype or torch.float32) * format.step
