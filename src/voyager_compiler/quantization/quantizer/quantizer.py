# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
from torchao.quantization.pt2e import ObserverOrFakeQuantize
from torchao.quantization.pt2e.quantizer.quantizer import QuantizationSpecBase

from voyager_compiler.quantization.fake_quantize import FusedAmaxObsFakeQuantize
from voyager_compiler.quantization.qspec import (
    DEFAULT_QUANTIZATION_ROUNDING,
    SUPPORTED_QUANTIZATION_ROUNDING,
    QScheme,
    parse_spec_fields,
)

__all__ = [
    "QuantizationSpec",
]


@dataclass(eq=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, qscheme, quant_max etc.
    """

    dtype: str
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        FusedAmaxObsFakeQuantize
    )
    quant_min: Optional[float] = None
    quant_max: Optional[float] = None
    scale_quant_max: Optional[float] = None
    rounding: str = DEFAULT_QUANTIZATION_ROUNDING
    qscheme: Optional[QScheme] = None
    amax_history_len: Optional[int] = None
    ch_axis: Optional[Union[int, List[int]]] = None
    block_size: Optional[Union[int, List[int]]] = None
    scale_dtype: Optional[str] = None
    outlier_threshold: Optional[float] = None
    outlier_pct: Optional[float] = None
    #: VSQ only: the outer quantization this one nests inside.
    coarse: Optional["QuantizationSpec"] = None
    is_dynamic: bool = False  # required by sharing nodes

    @staticmethod
    def from_str(s):
        """Build a spec from a comma-separated spec string."""
        params = parse_spec_fields(s)
        coarse = params.pop("coarse", None)
        # A coarse level that names no dtype is not a spec; leave it unset so
        # __post_init__ reports the missing level rather than a TypeError.
        if coarse and "dtype" in coarse:
            params["coarse"] = QuantizationSpec(**coarse)
        return QuantizationSpec(**params)

    def __post_init__(self):
        if self.qscheme is not None and self.quant_max is None:
            raise ValueError("quant_max is required for quantization.")

        if self.scale_quant_max is not None and (
            not math.isfinite(self.scale_quant_max) or self.scale_quant_max <= 0
        ):
            raise ValueError("scale_quant_max must be positive.")
        if self.rounding not in SUPPORTED_QUANTIZATION_ROUNDING:
            raise ValueError(
                "rounding must be one of "
                f"{sorted(SUPPORTED_QUANTIZATION_ROUNDING)}."
            )

        if (
            self.qscheme
            in [QScheme.MICROSCALING, QScheme.GROUP_WISE_AFFINE, QScheme.VSQ]
            and self.block_size is None
        ):
            raise ValueError("block_size is required for block quantization.")

        # The coarse level validated itself when it was constructed; only what
        # nesting adds is checked here.
        if self.qscheme == QScheme.VSQ:
            if self.coarse is None:
                raise ValueError("VSQ requires a coarse quantization.")
            if self.coarse.qscheme not in {
                QScheme.PER_TENSOR_SYMMETRIC,
                QScheme.PER_CHANNEL_SYMMETRIC,
            }:
                raise ValueError(
                    "VSQ coarse qscheme must be per-tensor or per-channel."
                )
            per_channel = self.coarse.qscheme == QScheme.PER_CHANNEL_SYMMETRIC
            if per_channel and self.coarse.ch_axis is None:
                raise ValueError("per-channel VSQ coarse needs a ch_axis.")
            if not per_channel and self.coarse.ch_axis is not None:
                raise ValueError("ch_axis is only valid for per-channel VSQ.")
            if (
                self.outlier_threshold is not None
                or self.outlier_pct is not None
            ):
                raise ValueError("Outlier filtering is not supported for VSQ.")
        elif self.coarse is not None:
            raise ValueError("coarse quantization is only meaningful for VSQ.")


EdgeOrNode = Union[Tuple[Node, Node], Node]


@dataclass(eq=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """quantization spec for the Tensors whose quantization parameters are
    derived from other Tensors
    """

    derived_from: List[EdgeOrNode]
    derive_qparams_fn: Callable[
        [List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]
    ]
    dtype: str
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[QScheme] = None
