# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet's shared fixed-point requantization encoding.

Every Magnet datapath rescales an integer accumulator the same way::

    output = (accumulator * multiplier) >> shift

with an unsigned 16-bit ``multiplier`` and an unsigned 8-bit ``shift``. The
logical graph states the same rescaling as a ratio of two float scales -- the
accumulator's dequantize scale over the consumer's quantize scale -- so
somebody has to turn that ratio into the pair the hardware consumes.

That is target arithmetic, and it belongs to the compiler even though today
only collateral conversion calls it: deriving a target's fixed-point data at
serialization or conversion time is how two components end up disagreeing
about what the hardware will compute. A downstream converter reads the encoding
from here; Alpha's NMP will read the same one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MAGNET_BIAS_MAX",
    "MAGNET_BIAS_MIN",
    "MAGNET_MULTIPLIER_BITS",
    "MAGNET_MULTIPLIER_MAX",
    "MAGNET_SHIFT_BITS",
    "MAGNET_SHIFT_MAX",
    "MagnetMultiplier",
    "encode_magnet_bias",
    "encode_magnet_multiplier",
]

#: The multiplier field is unsigned 16-bit and the shift field unsigned 8-bit.
#: Both Magnet datapaths in the tree use these widths; they are parameters on
#: the functions below so that a third one does not need a second encoder.
MAGNET_MULTIPLIER_BITS = 16
MAGNET_SHIFT_BITS = 8
MAGNET_MULTIPLIER_MAX = (1 << MAGNET_MULTIPLIER_BITS) - 1
MAGNET_SHIFT_MAX = (1 << MAGNET_SHIFT_BITS) - 1

#: The bias field is assigned into a signed 8-bit datapath type, whatever the
#: width of the configuration word that carries it.
MAGNET_BIAS_MIN, MAGNET_BIAS_MAX = -128, 127


@dataclass(frozen=True)
class MagnetMultiplier:
    """Multipliers and the single right shift they are paired with.

    ``multiplier`` is one integer, or a tuple of them when the rescaling is
    per output channel. The shift is shared in either case, because the
    datapath has one shifter: a per-channel shift is not something it can
    apply, so the encoding has to find one exponent that serves every
    channel. That is why :func:`encode_magnet_multiplier` sizes the shift
    from the *largest* factor.

    A zero in a vector is meaningful and legal: it marks an output channel
    the array computes because the datapath's lane count required it, and
    that nothing reads. A *lone* zero is not, because there is no other
    channel it could be about -- see ``__post_init__``.
    """

    multiplier: int
    shift: int

    def __post_init__(self):
        values = (
            self.multiplier
            if isinstance(self.multiplier, tuple)
            else (self.multiplier,)
        )
        if not values:
            raise ValueError("a multiplier needs at least one value")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("multiplier must be an integer")
            if not 0 <= value <= MAGNET_MULTIPLIER_MAX:
                raise ValueError(
                    f"multiplier {value} does not fit the native field "
                    f"[0, {MAGNET_MULTIPLIER_MAX}]"
                )
        if isinstance(self.shift, bool) or not isinstance(self.shift, int):
            raise ValueError("shift must be an integer")
        if not 0 <= self.shift <= MAGNET_SHIFT_MAX:
            raise ValueError(
                f"shift {self.shift} does not fit the native field "
                f"[0, {MAGNET_SHIFT_MAX}]"
            )
        if len(values) == 1 and values[0] == 0:
            # With one multiplier there is no other channel for a zero to be
            # about. The endpoint reads a zero as an implicit factor of one,
            # so this would silently apply a rescaling nobody asked for.
            raise ValueError("a lone Magnet multiplier must be nonzero")
        if all(value == 0 for value in values):
            raise ValueError(
                "a Magnet multiplier vector must rescale some channel"
            )

    @property
    def values(self) -> tuple:
        """Every multiplier, whether one was given or many."""
        if isinstance(self.multiplier, tuple):
            return self.multiplier
        return (self.multiplier,)

    @property
    def represented_ratio(self) -> float:
        """The ratio the encoded pair actually applies.

        Defined for a scalar encoding only: a per-channel one applies a
        different ratio per channel, and reducing that to a number would
        invite it to be compared with something.
        """
        if isinstance(self.multiplier, tuple):
            raise ValueError(
                "a per-channel multiplier represents one ratio per channel"
            )
        return math.ldexp(float(self.multiplier), -self.shift)


def encode_magnet_multiplier(ratio) -> MagnetMultiplier:
    """Encode one positive ratio, or a vector of them, on the binary point.

    The shift places the *largest* ratio's most significant bit just below
    the multiplier field's width. Sizing from the largest is what makes one
    shared shift legal: every smaller factor then lands inside the field, and
    a per-channel shift is not something the datapath has.

    Rounding at an exact binary boundary can carry into the seventeenth bit,
    which one fewer shift bit still represents, so the shift backs off until
    every multiplier fits. This is not a rare case -- a ratio of exactly 1.0
    hits it, and a ratio of exactly 1.0 is what an identity requantization
    is.

    Args:
        ratio: A positive real, or a sequence of them for a per-channel
            rescaling.

    Returns:
        The encoded pair. A scalar in gives a scalar ``multiplier``; a
        sequence gives a tuple, in order.
    """
    scalar = isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
    ratios = [float(ratio)] if scalar else [float(value) for value in ratio]
    if not ratios:
        raise ValueError("a multiplier encoding needs at least one ratio")
    for value in ratios:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Magnet multiplier ratio must be finite and non-negative: "
                f"{value!r}"
            )
    if scalar and ratios[0] == 0.0:
        raise ValueError("a lone Magnet multiplier ratio must be positive")

    # Sized from the largest, so the shift a padded channel's zero shares
    # with the rest is whatever the real channels needed.
    largest = max(ratios)
    if largest <= 0.0:
        raise ValueError(
            "a Magnet multiplier vector must rescale some channel"
        )
    shift = MAGNET_MULTIPLIER_BITS - math.ceil(math.log2(largest))
    def encode(at_shift):
        return [round(value * math.ldexp(1.0, at_shift)) for value in ratios]

    multipliers = encode(shift)
    while max(multipliers) > MAGNET_MULTIPLIER_MAX:
        shift -= 1
        multipliers = encode(shift)
    return MagnetMultiplier(
        multiplier=(
            int(multipliers[0]) if scalar else tuple(int(m) for m in multipliers)
        ),
        shift=int(shift),
    )


def encode_magnet_bias(
    bias: int,
    ratio: float,
    *,
    minimum: int = MAGNET_BIAS_MIN,
    maximum: int = MAGNET_BIAS_MAX,
) -> int:
    """Move one bias term from the accumulator's units into the output's.

    The datapath adds its bias *after* the multiplier and shift, so the value
    it wants is not the integer convolution's bias but that bias already
    rescaled::

        output = ((dot * multiplier) >> shift) + encoded_bias

    A graph that adds its bias inside the convolution therefore has to hand
    over ``round(bias * ratio)``, or the endpoint applies it at the wrong
    scale. The two are indistinguishable at a unit ratio, which is why a
    profile with one proves nothing about this.
    """
    if isinstance(bias, bool) or not isinstance(bias, int):
        raise ValueError("Magnet bias must be an integer code")
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            f"Magnet bias ratio must be finite and positive: {ratio!r}"
        )
    encoded = round(bias * ratio)
    if not minimum <= encoded <= maximum:
        raise ValueError(
            f"encoded bias {encoded} does not fit the native field "
            f"[{minimum}, {maximum}]"
        )
    return encoded
