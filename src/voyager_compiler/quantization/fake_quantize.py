import functools
import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchao.quantization.pt2e import FakeQuantizeBase, ObserverOrFakeQuantize

from voyager_compiler.ops import compiled_on_gpu, expand, quantize_mx, vmap
from voyager_compiler.quantization.dtypes import (
    quantize_to_minifloat,
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.quantization.mx_utils import _reshape_to_blocks
from voyager_compiler.quantization.qspec import (
    QScheme,
    float_format,
    int_range,
    parse_codebook_dtype,
    posit_format,
)

__all__ = [
    "DirectCastFakeQuantize",
    "FusedAmaxObsFakeQuantize",
    "GroupWiseAffineFakeQuantize",
    "MXFakeQuantize",
    "_DerivedObserverOrFakeQuantize",
    "entry_levels",
    "fake_quantize_class",
]


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def get_quantization_map(dtype, device=None):
    """Return the lookup table that quantizes a bfloat16 value to ``dtype``.

    Entry ``i`` is the value the bfloat16 whose bits read ``i`` rounds to,
    so quantizing is one gather.  Floats saturate: a value past the largest
    finite one, infinity included, takes it.  A lookup table's entries are
    NormalFloat's levels, snapped to its entry dtype's grid.  Each table is
    built once per dtype and device and shared by every caller, so it must
    not be modified in place.

    Args:
        dtype: ``int<N>``, ``uint<N>``, ``fp<B>_e<X>m<Y>``, ``posit<N>_<es>``
            or ``lut<I>[_to_<E>]``; None returns the identity.
        device: Where the table lives.

    Returns:
        A ``2**16``-entry bfloat16 tensor.

    Raises:
        ValueError: ``dtype`` names none of these.
    """
    values = torch.arange(2**16, dtype=torch.int16, device=device)
    values = values.view(torch.bfloat16)
    if dtype is None:
        return values

    if (bounds := int_range(dtype)) is not None:
        return torch.clamp(torch.round(values), *bounds)

    if (fmt := float_format(dtype)) is not None:
        if not fmt.signed:
            values = torch.abs(values)
        values = torch.clamp(values, -fmt.max_norm, fmt.max_norm)
        # The rounding runs in float32: bfloat16 cannot hold the halfway
        # sums it compares against.
        return quantize_to_minifloat(
            values.float(),
            fmt.mbits + 2,
            fmt.ebits,
            fmt.max_norm,
            round="even",
            saturate_normals=True,
        ).to(torch.bfloat16)

    if (posit := posit_format(dtype)) is not None:
        return quantize_to_posit(values, *posit, round_to_even=True)

    if (codebook := parse_codebook_dtype(dtype)) is not None:
        index_bits, entry = codebook
        grid = None if entry is None else entry_levels(entry, device)
        indices, levels = quantize_to_nf(values, index_bits, grid=grid)
        return levels[indices]

    raise ValueError(f"Unsupported dtype: {dtype}")


def entry_levels(dtype, device=None):
    """Return every value a lookup-table entry of ``dtype`` can hold.

    The grid is symmetric, so an integer's extra negative value is left out:
    ``int6`` spans [-31, 31], the range its ``quant_max`` gives.

    Args:
        dtype: The entry dtype, ``int<N>`` or ``fp<B>_e<X>m<Y>``.
        device: Where to build the grid.

    Returns:
        The distinct values in ascending order, as float32, with one zero.
    """
    table = get_quantization_map(dtype, device).float()
    levels = torch.unique(table[torch.isfinite(table)])
    # ``torch.unique`` can keep -0 as the zero; adding 0 turns it into +0.
    return levels[levels >= -levels.max()] + 0.0


# A fake-quant graph sees the same shapes window after window.
_compiled_per_shape = compiled_on_gpu(dynamic=False)


@_compiled_per_shape
def _fake_quant(input, qmap, scale):
    """Fake-quantize ``input`` against a per-tensor or per-channel scale.

    Args:
        input: Tensor to quantize.
        qmap: Lookup table, one entry per bfloat16 bit pattern.
        scale: Scale, broadcastable to ``input``.

    Returns:
        The fake-quantized tensor.
    """
    scale = scale.to(input.dtype)
    return vmap(input / scale, qmap) * scale


@_compiled_per_shape
def _mx_fake_quant(
    input, qmap, axes, block_size, quant_max, power_of_two, scale_qmap
):
    """Fake-quantize ``input`` in microscaling blocks.

    Args:
        input: Tensor to quantize.
        qmap: Lookup table or codebook the elements are quantized into.
        axes: Axis or axes the blocks run along.
        block_size: Elements per block.
        quant_max: Largest magnitude ``qmap`` represents.
        power_of_two: Round each block scale to a power of two.
        scale_qmap: Lookup table the scales are quantized into, or None.

    Returns:
        ``(scale, output)``: the block scales and the fake-quantized tensor.
    """
    scale, output = quantize_mx(
        input,
        qmap,
        axes,
        block_size,
        quant_max,
        power_of_two,
        scale_qmap=scale_qmap,
    )
    return scale, output * expand(scale, output.shape, block_size)


@_compiled_per_shape
def _group_wise_affine_fake_quant(
    input, axes, block_size, quant_min, quant_max, scale_qmap
):
    """Fake-quantize ``input`` with a scale and zero point per block.

    Each block maps its own ``[min, max]`` onto ``[quant_min, quant_max]``.

    Args:
        input: Tensor to quantize.
        axes: Axis or axes the blocks run along.
        block_size: Elements per block.
        quant_min: Smallest code.
        quant_max: Largest code.
        scale_qmap: Lookup table the scale and zero point are quantized
            into, or None.

    Returns:
        ``(scale, zero_point, output)``: the per-block parameters and the
        fake-quantized tensor.
    """
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
    # Quantize the scale using the codebook; a scale below its range
    # rounds to zero and the block is one level, as with no range.
    if scale_qmap is not None:
        sf = vmap(sf, scale_qmap)
        sf = torch.where(sf > 0.0, sf, 1.0)
    zp = -min / sf + quant_min
    if scale_qmap is not None:
        zp = vmap(zp, scale_qmap)

    expanded_sf = expand(sf, input.shape, block_size)
    expanded_zp = expand(zp, input.shape, block_size)

    q = torch.round(input / expanded_sf + expanded_zp)
    q = torch.clamp(q, quant_min, quant_max)
    return sf, zp, (q - expanded_zp) * expanded_sf


class FakeQuantizeFunction(torch.autograd.Function):
    """This function quantizes the inputs against an observed scale."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        enabled: bool,
        qmap: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if not enabled:
            return input
        return _fake_quant(input, qmap, scale)

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
        enabled: bool,
        scale: torch.Tensor,
        qmap: torch.Tensor,
        axes: Union[int, List[int]],
        block_size: Union[int, List[int]],
        quant_max: float,
        force_scale_power_of_two=False,
        scale_qmap=None,
    ):
        if not enabled:
            return input
        sf, output = _mx_fake_quant(
            input,
            qmap,
            axes,
            block_size,
            quant_max,
            force_scale_power_of_two,
            scale_qmap,
        )
        scale.resize_(sf.shape).copy_(sf)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: only ``input`` takes a gradient."""
        return (grad_output,) + (None,) * 8


class GroupWiseAffineFakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        enabled: bool,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axes: Union[int, List[int]],
        block_size: Union[int, List[int]],
        quant_min: float,
        quant_max: float,
        scale_qmap=None,
    ):
        if not enabled:
            return input
        sf, zp, output = _group_wise_affine_fake_quant(
            input, axes, block_size, quant_min, quant_max, scale_qmap
        )
        scale.resize_(sf.shape).copy_(sf)
        zero_point.resize_(zp.shape).copy_(zp)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator: only ``input`` takes a gradient."""
        return (grad_output,) + (None,) * 8


class _FreezableFlags(FakeQuantizeBase):
    """``FakeQuantizeBase`` whose enable flags freezing can fix.

    The flags are device tensors, so reading one on every call waits for
    the device, and a CUDA graph cannot capture the read.  Once
    ``freeze_flags`` has read them, the module goes by those values.
    """

    _frozen_flags: Optional[Tuple[bool, bool]] = None

    def freeze_flags(self) -> None:
        """Fix the enable flags at their current values."""
        self._frozen_flags = (
            bool(self.fake_quant_enabled[0]),
            bool(self.observer_enabled[0]),
        )

    def fake_quant_on(self) -> bool:
        """Whether ``forward`` fake-quantizes."""
        if self._frozen_flags is not None:
            return self._frozen_flags[0]
        return bool(self.fake_quant_enabled[0])

    def observer_on(self) -> bool:
        """Whether ``forward`` updates the statistics."""
        if self._frozen_flags is not None:
            return self._frozen_flags[1]
        return bool(self.observer_enabled[0])


class _FakeQuantize(_FreezableFlags):
    """The parts every fake-quant scheme shares.

    The dtype's lookup table and the ``scale`` every scheme reports.  A
    spec is passed whole to the class that implements its scheme, so each
    class takes the fields it uses and the rest fall through to
    ``**unused``.  A subclass fake-quantizes in ``forward`` and reports
    its parameters from ``calculate_qparams``.

    Args:
        dtype: Element dtype, as a spec string names it.
        device: Where the buffers live.
        **unused: The spec's fields this scheme does not use.
    """

    qmap: torch.Tensor
    scale: torch.Tensor

    #: Buffers that take their shape from the first input; a loaded one is
    #: resized to the saved shape before its values are copied in.
    _lazily_sized = ("scale",)

    def __init__(
        self,
        dtype: str,
        device: Optional[torch.device] = None,
        **unused,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": torch.float}

        codebook = parse_codebook_dtype(dtype)
        self.is_codebook_quantization = codebook is not None
        if self.is_codebook_quantization:
            self.index_bits, self.code_dtype = codebook
        self.register_buffer(
            "qmap", get_quantization_map(dtype, device), persistent=False
        )
        self.register_buffer("scale", torch.tensor([1.0], **factory_kwargs))

    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, "
            f"observer_enabled={self.observer_enabled}, dtype={self.dtype}, "
            f"scale={self.scale}"
        )

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
        for name in self._lazily_sized:
            key = prefix + name
            if key in state_dict:
                getattr(self, name).resize_(state_dict[key].shape)
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


class DirectCastFakeQuantize(_FakeQuantize):
    """Fake-quantize by rounding each value to the dtype, with no scale.

    For a spec with no scheme, such as ``fp8_e4m3`` or ``posit8_1``: every
    value rounds to the nearest one the dtype represents, ``scale`` stays 1
    and nothing is observed.

    Args:
        *args: Forwarded to ``_FakeQuantize``.
        **kwargs: Forwarded to ``_FakeQuantize``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.disable_observer()

    def calculate_qparams(self):
        return self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FakeQuantizeFunction.apply(
            x, self.fake_quant_on(), self.qmap, self.scale
        )


class FusedAmaxObsFakeQuantize(_FakeQuantize):
    """Fake-quantize against a scale observed from recent amaxes.

    The per-tensor and per-channel schemes keep the absolute maxima of the
    last ``amax_history_len`` inputs, one per tensor or one per channel
    along ``ch_axis``, and scale by the largest.  An input is quantized
    before its own amax joins the history, so the first call uses scale
    1.0, as Transformer Engine's delayed scaling does (whose default
    history is 1024 long).  The history and the scale are kept in the
    dtype of the tensor quantized.

    Args:
        dtype: Element dtype, as a spec string names it.
        qscheme: ``PER_TENSOR_SYMMETRIC`` or ``PER_CHANNEL_SYMMETRIC``.
        quant_max: Largest magnitude the dtype represents.
        amax_history_len: How many past amaxes the scale is taken over.
        ch_axis: The channel axis of a per-channel scale.
        force_scale_power_of_two: Round each scale up to a power of two.
        **kwargs: Forwarded to ``_FakeQuantize``.

    Raises:
        ValueError: ``qscheme`` is not per-tensor or per-channel.
    """

    amax_history: torch.Tensor

    _lazily_sized = ("scale", "amax_history")

    def __init__(
        self,
        dtype: str,
        qscheme: QScheme,
        quant_max: float,
        amax_history_len: int,
        ch_axis: Optional[int] = None,
        force_scale_power_of_two: bool = False,
        **kwargs,
    ) -> None:
        if qscheme not in (
            QScheme.PER_TENSOR_SYMMETRIC,
            QScheme.PER_CHANNEL_SYMMETRIC,
        ):
            raise ValueError(
                f"{qscheme} is fake-quantized by "
                f"{fake_quantize_class(qscheme).__name__}"
            )
        super().__init__(dtype, **kwargs)
        self.qscheme = qscheme
        self.quant_max = quant_max
        self.amax_history_len = amax_history_len
        self.ch_axis = ch_axis
        self.force_scale_power_of_two = force_scale_power_of_two
        self.is_per_channel = qscheme == QScheme.PER_CHANNEL_SYMMETRIC
        self.register_buffer(
            "amax_history",
            torch.tensor([], device=self.scale.device, dtype=torch.float),
        )

    def start_history(self, shape: Tuple[int, ...], dtype) -> None:
        """Size the history and the scale as before any input.

        Args:
            shape: Shape of one amax: ``()`` per tensor, or the channel
                shape per channel.
            dtype: Dtype of the tensor quantized.
        """
        device = self.scale.device
        self.amax_history = torch.zeros(
            (self.amax_history_len, *shape), dtype=dtype, device=device
        )
        self.scale = torch.ones(shape, dtype=dtype, device=device)

    @torch.no_grad()
    def _update_amax_scale(self, x: torch.Tensor) -> None:
        """Fold ``x``'s amax into the history and recompute ``scale``.

        The absolute maximum is taken over every axis, or over every axis
        but ``ch_axis`` when the observer is per-channel. The history and
        the scale are started on the first call, since the observed shape
        and dtype are only known once a tensor arrives.

        Args:
            x: The tensor being observed.
        """
        if self.is_per_channel:
            ch_axis = self.ch_axis % x.ndim
            dim = tuple(i for i in range(x.ndim) if i != ch_axis)
            amax_cur = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        else:
            amax_cur = torch.amax(torch.abs(x))

        if self.amax_history.numel() == 0:
            self.start_history(amax_cur.shape, x.dtype)

        amax = torch.amax(self.amax_history, dim=0)
        self.amax_history.copy_(torch.roll(self.amax_history, -1, 0))
        self.amax_history[0] = amax_cur

        sf = amax / self.quant_max
        sf = torch.where(amax > 0.0, sf, self.scale)
        sf = torch.where(torch.isfinite(amax), sf, self.scale)
        if self.force_scale_power_of_two:
            sf = torch.pow(2, torch.ceil(torch.log2(sf)))
        self.scale.copy_(sf)

    def calculate_qparams(self):
        return self.scale

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, qscheme={self.qscheme}, "
            f"quant_max={self.quant_max}, ch_axis={self.ch_axis}, "
            f"amax_history_len={self.amax_history_len}, "
            f"force_scale_power_of_two={self.force_scale_power_of_two}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_on():
            self._update_amax_scale(x)
        return FakeQuantizeFunction.apply(
            x, self.fake_quant_on(), self.qmap, self.scale
        )


class _BlockFakeQuantize(_FakeQuantize):
    """The parts the block schemes share.

    Blocks of ``block_size`` elements along ``ch_axis`` each take their
    parameters from their own values on every call, so nothing is observed;
    each block scale may itself be quantized to ``scale_dtype``.

    Args:
        dtype: Element dtype, as a spec string names it.
        quant_max: Largest code the blocks map onto.
        ch_axis: Axis or axes the blocks run along.
        block_size: Elements per block.
        scale_dtype: Dtype the block parameters are quantized to, or None.
        **kwargs: Forwarded to ``_FakeQuantize``.
    """

    scale_qmap: Optional[torch.Tensor]

    def __init__(
        self,
        dtype: str,
        quant_max: float,
        ch_axis: Union[int, List[int]],
        block_size: int,
        scale_dtype: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dtype, **kwargs)
        self.quant_max = quant_max
        self.ch_axis = ch_axis
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        scale_qmap = (
            get_quantization_map(scale_dtype, self.scale.device)
            if scale_dtype is not None
            else None
        )
        self.register_buffer("scale_qmap", scale_qmap, persistent=False)

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, quant_max={self.quant_max}, "
            f"ch_axis={self.ch_axis}, block_size={self.block_size}, "
            f"scale_dtype={self.scale_dtype}"
        )


class MXFakeQuantize(_BlockFakeQuantize):
    """Fake-quantize in microscaling blocks.

    Each block's scale is its absolute maximum over ``quant_max``; ``scale``
    holds the last call's block scales.  Outliers, set by a threshold or by
    the fraction of elements to keep out, skip quantization and are restored
    afterwards.

    Args:
        force_scale_power_of_two: Round each block scale to a power of two.
        outlier_threshold: Magnitude from which an element is an outlier.
        outlier_pct: Fraction of elements to treat as outliers instead; the
            threshold is calibrated while the observer is on.
        *args: Forwarded to ``_BlockFakeQuantize``.
        **kwargs: Forwarded to ``_BlockFakeQuantize``.
    """

    def __init__(
        self,
        *args,
        force_scale_power_of_two: bool = False,
        outlier_threshold: Optional[float] = None,
        outlier_pct: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.force_scale_power_of_two = force_scale_power_of_two
        assert (
            outlier_pct is None or outlier_threshold is None
        ), "Only one of outlier_pct and outlier_threshold can be set."
        self.outlier_pct = outlier_pct
        self.max_outlier_pct = 0.0
        if outlier_pct is not None:
            self.register_buffer(
                "outlier_threshold",
                torch.tensor([], device=self.scale.device, dtype=torch.float),
            )
        else:
            self.outlier_threshold = outlier_threshold

    def calculate_qparams(self):
        return self.scale

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, "
            f"force_scale_power_of_two={self.force_scale_power_of_two}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outlier_pct is not None and self.observer_on():
            flat = x.abs().flatten()
            k = max(1, math.ceil(self.outlier_pct * flat.numel()))

            vals = torch.topk(flat, k, largest=True, sorted=False).values
            threshold = vals.min()

            # The largest threshold any calibration batch asked for, so
            # the bound holds on every batch seen.
            if self.outlier_threshold.numel() == 0:
                self.outlier_threshold.resize_as_(threshold)
                self.outlier_threshold.copy_(threshold)
            else:
                self.outlier_threshold.copy_(
                    torch.maximum(self.outlier_threshold, threshold)
                )

        # Remove outliers from x before quantization
        skip = None
        if self.outlier_threshold is not None:
            skip = x.abs() >= self.outlier_threshold
            orig_x = x.clone()
            x = x.masked_fill(skip, 0.0)
            outlier_pct = skip.sum().item() / x.numel()
            self.max_outlier_pct = max(outlier_pct, self.max_outlier_pct)

        x = MXFakeQuantizeFunction.apply(
            x,
            self.fake_quant_on(),
            self.scale,
            self.qmap,
            self.ch_axis,
            self.block_size,
            self.quant_max,
            self.force_scale_power_of_two,
            self.scale_qmap,
        )

        # Restore all outlier positions.
        if skip is not None:
            x = torch.where(skip, orig_x, x)
        return x


class GroupWiseAffineFakeQuantize(_BlockFakeQuantize):
    """Fake-quantize with a scale and a zero point per block.

    Each block maps its own ``[min, max]`` onto ``[quant_min, quant_max]``;
    ``scale`` and ``zero_point`` hold the last call's.

    Args:
        quant_min: Smallest code the blocks map onto.
        *args: Forwarded to ``_BlockFakeQuantize``.
        **kwargs: Forwarded to ``_BlockFakeQuantize``.
    """

    zero_point: torch.Tensor

    _lazily_sized = ("scale", "zero_point")

    def __init__(self, *args, quant_min: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min = quant_min
        self.register_buffer(
            "zero_point",
            torch.tensor([1.0], device=self.scale.device, dtype=torch.float),
        )

    def calculate_qparams(self):
        return self.scale, self.zero_point

    def extra_repr(self):
        return f"{super().extra_repr()}, quant_min={self.quant_min}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GroupWiseAffineFakeQuantFunction.apply(
            x,
            self.fake_quant_on(),
            self.scale,
            self.zero_point,
            self.ch_axis,
            self.block_size,
            self.quant_min,
            self.quant_max,
            self.scale_qmap,
        )


def fake_quantize_class(qscheme):
    """Return the fake-quant class that implements ``qscheme``.

    Args:
        qscheme: A ``QScheme``, or None for a cast with no scale.

    Returns:
        The class a spec with this scheme is built with.
    """
    if qscheme is None:
        return DirectCastFakeQuantize
    if qscheme == QScheme.MICROSCALING:
        return MXFakeQuantize
    if qscheme == QScheme.GROUP_WISE_AFFINE:
        return GroupWiseAffineFakeQuantize
    return FusedAmaxObsFakeQuantize


class _DerivedObserverOrFakeQuantize(_FreezableFlags):
    r"""This observer is used to describe an observer whose quantization
    parameters are derived from other observers
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

    def forward(self, x: Tensor) -> Tensor:
        devices = {p.device for p in self.buffers()}
        if len(devices) != 1 or next(iter(devices)) != x.device:
            self.to(x.device)

        return FakeQuantizeFunction.apply(
            x, self.fake_quant_on(), self.qmap, self.calculate_qparams()
        )

    def calculate_qparams(self):
        return self.derive_qparams_fn(self.obs_or_fqs)
