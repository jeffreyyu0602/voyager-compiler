# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import math
import re
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchao.quantization.pt2e import FakeQuantizeBase, ObserverOrFakeQuantize

from voyager_compiler.ops import expand, quantize_mx, vmap
from voyager_compiler.quantization.dtypes import (
    _quantize_elemwise_core,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.quantization.mx_utils import _reshape_to_blocks
from voyager_compiler.quantization.qspec import (
    DEFAULT_QUANTIZATION_ROUNDING,
    SUPPORTED_QUANTIZATION_ROUNDING,
    QScheme,
)

__all__ = [
    "FusedAmaxObsFakeQuantize",
    "get_vsq_scale_quantization_map",
    "_DerivedObserverOrFakeQuantize",
]


logger = logging.getLogger(__name__)


def _validate_integer_bounds(
    dtype,
    quant_min,
    quant_max,
    storage_min,
    storage_max,
):
    """Reject bounds the storage dtype could not hold."""
    if any(
        not math.isfinite(bound) or bound != round(bound)
        for bound in (quant_min, quant_max)
    ):
        raise ValueError(f"{dtype} quantization bounds must be finite integers")
    if quant_min > quant_max:
        raise ValueError(f"{dtype} quant_min must not exceed quant_max")
    if quant_min < storage_min or quant_max > storage_max:
        raise ValueError(
            f"{dtype} quantization bounds [{quant_min}, {quant_max}] "
            f"exceed storage range [{storage_min}, {storage_max}]"
        )


def get_quantization_map(
    dtype,
    device=None,
    quant_min=None,
    quant_max=None,
    rounding=DEFAULT_QUANTIZATION_ROUNDING,
):
    """Return a quantization map for the given dtype.

    ``quant_min`` / ``quant_max`` narrow an integer dtype below its storage
    range, and ``rounding`` selects truncation for a datapath that floors
    rather than rounds to nearest.
    """
    if rounding not in SUPPORTED_QUANTIZATION_ROUNDING:
        raise ValueError(
            f"rounding must be one of "
            f"{sorted(SUPPORTED_QUANTIZATION_ROUNDING)}"
        )
    round_values = torch.floor if rounding == "floor" else torch.round
    values = torch.arange(2**16, dtype=torch.int16, device=device)
    values = values.view(torch.bfloat16)
    if dtype is None:
        return values

    # PyTorch native dtypes
    if dtype in ["float32", "float16", "bfloat16"]:
        torch_dtype = getattr(torch, dtype)
        return values.to(torch_dtype).to(torch.bfloat16)

    # Integer data types
    if match := re.fullmatch(r"int(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        storage_min, storage_max = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
        quant_min = storage_min if quant_min is None else quant_min
        quant_max = storage_max if quant_max is None else quant_max
        _validate_integer_bounds(
            dtype, quant_min, quant_max, storage_min, storage_max
        )
        return torch.clamp(round_values(values), quant_min, quant_max)

    # Unsigned integer
    if match := re.fullmatch(r"uint(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        storage_min, storage_max = 0, 2**nbits - 1
        quant_min = storage_min if quant_min is None else quant_min
        quant_max = storage_max if quant_max is None else quant_max
        _validate_integer_bounds(
            dtype, quant_min, quant_max, storage_min, storage_max
        )
        return torch.clamp(round_values(values), quant_min, quant_max)

    # Custom implementation for NVIDIA FP8 format
    if match := re.fullmatch(r"(?:fp8\.)?(e4m3|e5m2)", dtype, re.IGNORECASE):
        fp8_format = match.group(1).lower()
        if fp8_format == "e4m3":
            return quantize_to_fp8_e4m3(values)
        else:
            return quantize_to_fp8_e5m2(values)

    # Floating point implementation adopted from microscaling code
    if match := re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype):
        nbits, ebits, mbits = map(int, match.groups())
        assert nbits == ebits + mbits + 1 or nbits == ebits + mbits

        # Unsigned float for scaling factors
        if nbits == ebits + mbits:
            values = torch.abs(values)

        mbits = mbits + 2
        emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)
        if dtype != "fp8_e4m3":
            max_norm = 2**emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
        else:
            max_norm = 2**emax * 1.75  # FP8 has custom max_norm

        return _quantize_elemwise_core(
            values, mbits, ebits, max_norm, round="even", saturate_normals=True
        )

    # Posit data types invented by John Gustafson
    # https://www.johndcook.com/blog/2018/04/11/anatomy-of-a-posit-number/
    if match := re.fullmatch(r"posit(\d+)_(\d+)", dtype):
        nbits, es = match.groups()
        return quantize_to_posit(
            values, int(nbits), int(es), round_to_even=True
        )

    # NormalFloat by Tim Dettmers and adopted from the bitsandbytes library
    # This data type returns two tensors: a mapping to the NF index, and
    # a value map
    if match := re.fullmatch(r"nf(\d+)(?:_(\d+))?", dtype):
        nbits = int(match.group(1))
        int_bits = int(match.group(2)) if match.group(2) else None
        return quantize_to_nf(values, nbits, int_bits=int_bits)

    raise ValueError(f"Unsupported dtype: {dtype}")


def get_vsq_scale_quantization_map(
    dtype,
    device,
    fine_quant_max,
    coarse_scale_quant_max,
):
    """Return a normalized VSQ fine-scale quantization map.

    Integer fine-scale storage represents a *fraction* of the full logical
    fine-scale range rather than an integer-valued scale, so an unsigned
    storage dtype is remapped onto that fraction. Any other dtype has no such
    reinterpretation and uses the ordinary map.
    """
    if (
        match := re.fullmatch(r"uint(\d+)", dtype or "", re.IGNORECASE)
    ) is None:
        return get_quantization_map(dtype, device)

    nbits = int(match.group(1))
    # A zero code would erase the block, so the smallest scale is one step.
    quant_min, quant_max = 1, 2**nbits - 1

    for bound, name in (
        (fine_quant_max, "fine_quant_max"),
        (coarse_scale_quant_max, "coarse_scale_quant_max"),
    ):
        if bound is None or not math.isfinite(bound) or bound <= 0:
            raise ValueError(f"{name} must be finite and positive")

    step = coarse_scale_quant_max / (fine_quant_max * quant_max)
    values = torch.arange(2**16, dtype=torch.int16, device=device)
    values = values.view(torch.bfloat16)
    codes = torch.floor(values.to(torch.float32) / step)
    return torch.clamp(codes, quant_min, quant_max) * step


class FakeQuantizeFunction(torch.autograd.Function):
    """This function quantizes the inputs against an observed scale."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        qmap: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if fake_quant_enabled[0] == 0:
            return input
        scale = scale.to(input.dtype)
        return vmap(input / scale, qmap) * scale

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: only ``input`` takes a gradient."""
        return (grad_output,) + (None,) * 3


class MXFakeQuantizeFunction(torch.autograd.Function):
    """This function performs MX quantization by calculating the scaling
    factor using absolute maximum values in the input.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        scale: torch.Tensor,
        qmap: torch.Tensor,
        axes: Union[int, List[int]],
        block_size: Union[int, List[int]],
        quant_max: float,
        force_scale_power_of_two=False,
        scale_qmap=None,
    ):
        if fake_quant_enabled[0] == 0:
            return input
        sf, input = quantize_mx(
            input,
            qmap,
            axes,
            block_size,
            quant_max,
            force_scale_power_of_two,
            scale_qmap=scale_qmap,
        )
        scale.resize_(sf.shape).copy_(sf)
        input = input * expand(sf, input.shape, block_size)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: only ``input`` takes a gradient."""
        return (grad_output,) + (None,) * 8


class GroupWiseAffineFakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axes: Union[int, List[int]],
        block_size: Union[int, List[int]],
        quant_min: float,
        quant_max: float,
        scale_qmap=None,
    ):
        if fake_quant_enabled[0] == 0:
            return input

        assert block_size > 0

        # Make sure axes is a list of non-negative numbers
        axes = [axes] if type(axes) == int else axes
        axes = [x + input.ndim if x < 0 else x for x in axes]

        # Perform tiling to the hardware vector size
        reshaped_input, axes, orig_shape, padded_shape = _reshape_to_blocks(
            input, axes, block_size
        )

        shared_exp_axes = [x + 1 for x in axes]

        min = torch.amin(reshaped_input, dim=shared_exp_axes)
        max = torch.amax(reshaped_input, dim=shared_exp_axes)

        sf = (max - min) / (quant_max - quant_min)
        sf = torch.where(sf > 0.0, sf, 1.0)
        zp = -min / sf + quant_min

        # Quantize the scale using the codebook
        if scale_qmap is not None:
            sf = vmap(sf, scale_qmap)
            zp = vmap(zp, scale_qmap)

        scale.resize_(sf.shape).copy_(sf)
        zero_point.resize_(zp.shape).copy_(zp)

        sf = expand(sf, input.shape, block_size)
        zp = expand(zp, input.shape, block_size)

        q = torch.clamp(torch.round(input / sf + zp), quant_min, quant_max)
        return (q - zp) * sf

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: only ``input`` takes a gradient."""
        return (grad_output,) + (None,) * 8


class FusedAmaxObsFakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    Observer module for computing the quantization parameters based on the
    historical amax values.

    This observer uses the tensor amax statistics to compute the quantization
    parameters. The module records the maximums of absolute values of output
    tensors, and uses this statistic to compute the quantization parameters.
    """

    qmap: torch.Tensor
    scale_qmap: Optional[torch.Tensor]
    amax_history: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        dtype: str,
        qscheme: Optional[str] = None,
        quant_min: Optional[float] = None,
        quant_max: Optional[float] = None,
        rounding: str = DEFAULT_QUANTIZATION_ROUNDING,
        amax_history_len: int = None,
        ch_axis: Optional[int] = None,
        block_size: Optional[int] = None,
        scale_dtype: Optional[str] = None,
        scale_quant_max: Optional[float] = None,
        coarse: Optional[dict] = None,
        force_scale_power_of_two: bool = False,
        record_histogram: bool = False,
        outlier_threshold: Optional[float] = None,
        outlier_pct: Optional[float] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        if rounding not in SUPPORTED_QUANTIZATION_ROUNDING:
            raise ValueError(
                "rounding must be one of "
                f"{sorted(SUPPORTED_QUANTIZATION_ROUNDING)}"
            )
        self.rounding = rounding
        self.amax_history_len = amax_history_len
        self.ch_axis = ch_axis
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        # A scale's own range defaults to the range of the data it scales.
        self.scale_quant_max = (
            quant_max if scale_quant_max is None else scale_quant_max
        )
        self.force_scale_power_of_two = force_scale_power_of_two
        self.record_histogram = record_histogram
        self.outlier_pct = outlier_pct
        assert (
            outlier_pct is None or outlier_threshold is None
        ), "Only one of outlier_pct and outlier_threshold can be set."
        self.max_outlier_pct = 0.0

        factory_kwargs = {"device": device, "dtype": torch.float}

        # Generate quantization map buffers
        quant_map = get_quantization_map(
            dtype,
            device,
            quant_min,
            quant_max,
            rounding,
        )
        self.is_codebook_quantization = isinstance(quant_map, tuple)
        if self.is_codebook_quantization:
            indices, values = quant_map
            quant_map = values[indices]
        self.register_buffer("qmap", quant_map, persistent=False)

        # VSQ nests a coarse quantization around this one.  Its fields
        # arrive as a plain dict because the spec is splatted with asdict().
        self.is_vsq = self.qscheme == QScheme.VSQ
        coarse = coarse or {}
        self.coarse = dict(coarse)
        self.coarse_quant_min = coarse.get("quant_min")
        self.coarse_quant_max = coarse.get("quant_max")
        self.coarse_scale_quant_max = coarse.get("scale_quant_max")
        if self.coarse_scale_quant_max is None:
            self.coarse_scale_quant_max = self.coarse_quant_max
        self.coarse_ch_axis = coarse.get("ch_axis")
        self.coarse_is_per_channel = (
            coarse.get("qscheme") == QScheme.PER_CHANNEL_SYMMETRIC
        )
        coarse_map = (
            get_quantization_map(
                coarse["dtype"],
                device,
                coarse.get("quant_min"),
                coarse.get("quant_max"),
                coarse.get("rounding", DEFAULT_QUANTIZATION_ROUNDING),
            )
            if coarse.get("dtype") is not None
            else None
        )
        if isinstance(coarse_map, tuple):
            indices, values = coarse_map
            coarse_map = values[indices]
        self.register_buffer("coarse_qmap", coarse_map, persistent=False)

        if self.scale_dtype is None:
            scale_map = None
        elif self.is_vsq:
            # The fine scale is stored as a fraction of the coarse range.
            scale_map = get_vsq_scale_quantization_map(
                self.scale_dtype,
                device,
                self.scale_quant_max,
                self.coarse_scale_quant_max,
            )
        else:
            scale_map = get_quantization_map(self.scale_dtype, device)
        self.register_buffer("scale_qmap", scale_map, persistent=False)

        # Create amax history and scale buffers
        self.register_buffer("amax_history", torch.tensor([], **factory_kwargs))
        self.register_buffer("scale", torch.tensor([1.0], **factory_kwargs))
        self.register_buffer(
            "fine_scale",
            torch.tensor([1.0], **factory_kwargs) if self.is_vsq else None,
        )
        self.register_buffer(
            "zero_point", torch.tensor([1.0], **factory_kwargs)
        )

        # Create histogram buffer
        if self.record_histogram:
            self.register_buffer(
                "histogram",
                torch.zeros(254, **factory_kwargs),
                persistent=False,
            )

        self.observer_enabled[0] = self.qscheme is not None
        self.is_per_channel = self.qscheme == QScheme.PER_CHANNEL_SYMMETRIC

        if outlier_pct is not None:
            self.outlier_ema_decay = kwargs.get("outlier_ema_decay", 0.9)
            self.register_buffer(
                "outlier_threshold", torch.tensor([], **factory_kwargs)
            )
        else:
            self.outlier_threshold = outlier_threshold

    @torch.no_grad()
    def _update_amax_scale(
        self,
        x: torch.Tensor,
        *,
        is_per_channel: Optional[bool] = None,
        ch_axis: Optional[int] = None,
        quant_max: Optional[float] = None,
    ) -> None:
        """Fold ``x``'s amax into the history and recompute ``scale``.

        The absolute maximum is taken over every axis, or over every axis
        but ``ch_axis`` when the observer is per-channel. The history and
        the scale are lazily sized on the first call, since the observed
        shape is only known once a tensor arrives.

        Args:
            x: The tensor being observed.
        """
        # VSQ observes on its *coarse* level, so the geometry is a
        # parameter rather than always this observer's own.
        if is_per_channel is None:
            is_per_channel = self.is_per_channel
        if ch_axis is None:
            ch_axis = self.ch_axis
        if quant_max is None:
            quant_max = self.scale_quant_max

        if is_per_channel:
            axis = ch_axis % x.ndim
            dim = tuple(i for i in range(x.ndim) if i != axis)
            amax_cur = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        else:
            amax_cur = torch.amax(torch.abs(x))

        if self.amax_history.numel() == 0:
            size = (self.amax_history_len,) + amax_cur.shape
            self.amax_history.resize_(size).fill_(0.0)
            self.scale.resize_(amax_cur.shape).fill_(1.0)

        amax = torch.amax(self.amax_history, dim=0)
        self.amax_history.copy_(torch.roll(self.amax_history, -1, 0))
        self.amax_history[0] = amax_cur

        # Deliberately the pre-roll amax: the scale reflects the history as
        # it stood before this sample, which is the behaviour every existing
        # lane's collateral was produced with.
        self._apply_amax(amax, quant_max)

    @torch.no_grad()
    def _apply_amax(self, amax, quant_max=None) -> None:
        """Set ``scale`` from one observed absolute maximum.

        Extracted so that the calibrated range and the scale it implies have
        one relation, wherever the range came from: a forward pass, or a
        quantization region joining several observers' ranges into one. Two
        copies of this arithmetic is how a region ends up nominally joined
        and numerically not.

        A zero or non-finite amax leaves the scale alone, which is what keeps
        an observer that has seen nothing from collapsing its scale to zero.
        """
        if quant_max is None:
            quant_max = (
                self.coarse_scale_quant_max
                if self.is_vsq
                else self.scale_quant_max
            )
        sf = amax / quant_max
        sf = torch.where(amax > 0.0, sf, self.scale)
        sf = torch.where(torch.isfinite(amax), sf, self.scale)
        if self.force_scale_power_of_two:
            sf = torch.pow(2, torch.ceil(torch.log2(sf)))
        self.scale.copy_(sf)

    @torch.no_grad()
    def join_amax(self, amax) -> None:
        """Adopt ``amax`` as this observer's whole calibrated range.

        Used by quantization-region formation, where several observers have
        to end up with one scale. The history is replaced rather than folded
        into, because the joined range *is* the region's range: leaving the
        observer's own samples in it would let a later read of the history
        disagree with the scale.
        """
        if self.amax_history.numel() == 0:
            self.amax_history.resize_(
                (self.amax_history_len,) + tuple(amax.shape)
            ).fill_(0.0)
            self.scale.resize_(tuple(amax.shape)).fill_(1.0)
        self.amax_history.copy_(amax.expand_as(self.amax_history))
        self._apply_amax(amax)

    def quantize_vsq_components(
        self,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``values`` into ``(coarse_scale, fine_scale, payload)``.

        This is what the converter stamps onto the graph: the calibrated
        coarse scale, the per-block fine scale, and the narrow payload.
        """
        if not self.is_vsq:
            raise ValueError("VSQ components require a VSQ observer")
        axes = self.ch_axis
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        return torch.ops.quantized_ops.quantize_vsq.default(
            values,
            self.calculate_qparams().to(
                device=values.device, dtype=values.dtype
            ),
            self.coarse_qmap.to(values.device),
            self.qmap.to(values.device),
            axes,
            self.block_size,
            self.scale_quant_max,
            self.force_scale_power_of_two,
            (
                self.scale_qmap.to(values.device)
                if self.scale_qmap is not None
                else None
            ),
            None,
            self.coarse_scale_quant_max,
        )

    @torch.jit.export
    def calculate_qparams(self):
        # Forward intentionally derives its live scale from the history as it
        # stood before the current sample was rolled in.  At a calibration
        # boundary the last sample is already in ``amax_history`` and must be
        # incorporated before the qparam is frozen.
        if self.amax_history.numel() > 0:
            quant_max = (
                self.coarse_scale_quant_max
                if self.is_vsq
                else self.scale_quant_max
            )
            self._apply_amax(torch.amax(self.amax_history, dim=0), quant_max)
        if self.qscheme == QScheme.GROUP_WISE_AFFINE:
            return self.scale, self.zero_point
        return self.scale

    def reset_observer_statistics(self) -> None:
        """Clear collected ranges while preserving observer enablement."""
        self.amax_history.resize_(0)
        self.scale.resize_(1).fill_(1.0)

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, dtype={}, amax_history_len={}, "
            "quant_max={}, qscheme={}, ch_axis={}, block_size={}, force_scale_power_of_two={}, "
            "scale={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.dtype,
                self.amax_history_len,
                self.quant_max,
                self.qscheme,
                self.ch_axis,
                self.block_size,
                self.force_scale_power_of_two,
                self.scale,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_histogram:
            exp = torch.floor(torch.log2(torch.abs(x.detach().float())))
            self.histogram += torch.histc(exp, 254, min=-126, max=127)

        if self.outlier_pct is not None and self.observer_enabled[0] == 1:
            flat = x.abs().flatten()
            k = max(1, math.ceil(self.outlier_pct * flat.numel()))

            vals = torch.topk(flat, k, largest=True, sorted=False).values
            threshold = vals.min()

            if self.outlier_threshold.numel() == 0:
                self.outlier_threshold.resize_as_(threshold)
                self.outlier_threshold.copy_(threshold)
            else:
                self.outlier_threshold.mul_(self.outlier_ema_decay).add_(
                    threshold * (1.0 - self.outlier_ema_decay)
                )

        # Remove outliers from x before quantization
        skip = None
        if self.outlier_threshold is not None:
            skip = x.abs() >= self.outlier_threshold
            orig_x = x.clone()
            x = x.masked_fill(skip, 0.0)
            outlier_pct = skip.sum().item() / x.numel()
            self.max_outlier_pct = max(outlier_pct, self.max_outlier_pct)

        if self.is_vsq:
            # Two nested levels: a coarse scale observed over the whole
            # tensor, then a fine per-block scale on what remains once the
            # coarse scale is divided out.  They stay separate because the
            # datapath applies them at different granularities.
            if self.observer_enabled[0] == 1:
                self._update_amax_scale(
                    x,
                    is_per_channel=self.coarse_is_per_channel,
                    ch_axis=self.coarse_ch_axis,
                    quant_max=self.coarse_quant_max,
                )
            coarse_dequantized = FakeQuantizeFunction.apply(
                x, self.fake_quant_enabled, self.coarse_qmap, self.scale
            )
            coarse_data = coarse_dequantized / self.scale.to(x.dtype)
            fine_dequantized = MXFakeQuantizeFunction.apply(
                coarse_data,
                self.fake_quant_enabled,
                self.fine_scale,
                self.qmap,
                self.ch_axis,
                self.block_size,
                self.scale_quant_max,
                self.force_scale_power_of_two,
                self.scale_qmap,
            )
            x = fine_dequantized * self.scale.to(x.dtype)
        elif self.qscheme == QScheme.MICROSCALING:
            x = MXFakeQuantizeFunction.apply(
                x,
                self.fake_quant_enabled,
                self.scale,
                self.qmap,
                self.ch_axis,
                self.block_size,
                self.quant_max,
                self.force_scale_power_of_two,
                self.scale_qmap,
            )
        elif self.qscheme == QScheme.GROUP_WISE_AFFINE:
            x = GroupWiseAffineFakeQuantFunction.apply(
                x,
                self.fake_quant_enabled,
                self.scale,
                self.zero_point,
                self.ch_axis,
                self.block_size,
                self.quant_min,
                self.quant_max,
                self.scale_qmap,
            )
        else:
            if self.observer_enabled[0] == 1:
                self._update_amax_scale(x)
            x = FakeQuantizeFunction.apply(
                x, self.fake_quant_enabled, self.qmap, self.scale
            )

        # Restore all outlier positions.
        if skip is not None:
            x = torch.where(skip, orig_x, x)

        return x

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "amax_history"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and amax_history
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "scale":
                    self.scale.resize_(val.shape)
                else:
                    assert name == "amax_history"
                    self.amax_history.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        assert name == "amax_history"
                        self.amax_history.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _DerivedObserverOrFakeQuantize(FakeQuantizeBase):
    r"""This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: List[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[
            [List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]
        ],
    ):
        super().__init__()
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.register_buffer(
            "qmap", get_quantization_map(dtype), persistent=False
        )
        self.observer_enabled[0] = 0
        self.dtype = dtype
        self.qscheme = obs_or_fqs[1].qscheme

    def forward(self, x: Tensor) -> Tensor:
        devices = {p.device for p in self.buffers()}
        if len(devices) != 1 or next(iter(devices)) != x.device:
            self.to(x.device)

        return FakeQuantizeFunction.apply(
            x, self.fake_quant_enabled, self.qmap, self.calculate_qparams()
        )

    def calculate_qparams(self):
        return self.derive_qparams_fn(self.obs_or_fqs)
