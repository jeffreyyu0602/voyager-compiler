import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.library import Library, impl

logger = logging.getLogger(__name__)


quantized_ops_lib = Library("quantized_ops", "DEF")

# Registers the auto-generated hardware-layout twins (conv2d /
# max_pool2d / adaptive_avg_pool2d / linear / matmul) into the namespace
# just created, before any importer references them.
from voyager_compiler.ops import layout as _register_layout  # noqa: E402,F401

# ``mean`` / ``variance`` / ``normalized`` are the on-chip regions the
# accelerator keeps between its passes over the data.  The graph never reads
# them, so eager ignores them; naming them on the op is what tells the backend
# which bytes are its to write (see ``reduction_scratch``).
quantized_ops_lib.define(
    "layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, "
    "Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True, "
    "Tensor(a!)? mean=None, Tensor(b!)? variance=None, "
    "Tensor(c!)? normalized=None) "
    "-> Tensor"
)


@impl(quantized_ops_lib, "layer_norm", "CompositeExplicitAutograd")
def layer_norm(
    input: torch.Tensor,
    normalized_shape: Union[int, Tuple[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    cudnn_enable: bool = True,
    mean: Optional[torch.Tensor] = None,
    variance: Optional[torch.Tensor] = None,
    normalized: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k = normalized_shape[-1]
    output = torch.ops.aten.layer_norm.default(
        input[..., :k],
        normalized_shape,
        None if weight is None else weight[..., :k],
        None if bias is None else bias[..., :k],
        eps,
        cudnn_enable,
    )
    # Pad the output back to the original input shape
    output = torch.nn.functional.pad(
        output, (0, input.shape[-1] - normalized_shape[-1])
    )
    return output


quantized_ops_lib.define(
    "softmax(Tensor input, int dim, ScalarType? dtype=None, "
    "Tensor(a!)? max=None, Tensor(b!)? sum=None) -> Tensor"
)


@impl(quantized_ops_lib, "softmax", "CompositeExplicitAutograd")
def softmax(
    input: torch.Tensor,
    dim: int,
    dtype: Optional[torch.dtype] = None,
    max: Optional[torch.Tensor] = None,
    sum: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``aten.softmax`` that also names its on-chip scratch.

    ``max`` and ``sum`` are the row statistics the accelerator holds between
    its passes; eager computes the softmax in one go and ignores them.
    """
    return torch.ops.aten.softmax.int(input, dim, dtype)


def expand(input, shape, block_size):
    while input.ndim < len(shape):
        input = input.unsqueeze(0)

    # Repeat the input along each dimension to match the target shape
    for dim in range(len(shape)):
        if input.shape[dim] != shape[dim]:
            input = torch.repeat_interleave(input, block_size, dim)

    # If the shape is not a multiple of block_size, we may need to slice
    if list(input.shape) != list(shape):
        slices = [slice(0, x) for x in shape]
        input = input[slices]
    return input


quantized_ops_lib.define("vmap(Tensor self, Tensor other) -> Tensor")


@impl(quantized_ops_lib, "vmap", "CompositeExplicitAutograd")
def vmap(
    input: torch.Tensor, qmap: torch.Tensor, chunk_size=1024 * 1024
) -> torch.Tensor:
    input_dtype = input.dtype

    if input.dtype != torch.bfloat16:
        input = input.to(torch.bfloat16)

    indices = input.view(torch.int16)

    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    indices_flat = indices.reshape(-1)
    output_flat = output.view(-1)

    for start in range(0, indices_flat.numel(), chunk_size):
        end = min(start + chunk_size, indices_flat.numel())
        indices_chunk = indices_flat[start:end].to(torch.int32) & 0xFFFF
        output_flat[start:end] = qmap[indices_chunk]

    return output.to(input_dtype)


#: Entries in a bfloat16-indexed lookup table, one per bit pattern.  A
#: quantization map that size is such a table; a smaller one is the
#: codebook itself, and is searched rather than indexed.
QMAP_SIZE = 2**16

#: Axis an attention operand carries its heads on: Q, K, P and V all enter
#: their matmul as ``[batch, head, ...]``.  A microscaling block never
#: straddles two heads, so a head owns whole blocks and can carry its own
#: codebook.
HEAD_AXIS = 1


def encode(
    input: torch.Tensor,
    codebook: torch.Tensor,
    bounds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantize each element against a codebook.

    The entries are ascending, so which one an element rounds to is the
    interval it falls in between their midpoints -- the same rule a
    lookup table bakes in, without materialising an entry per bfloat16
    bit pattern.  Searching those midpoints yields the entry's index, and
    reading the entry back is one more gather.

    Args:
        input: Tensor to quantize, already divided by its block scale.
        codebook: Entries in ascending order.  A 2-D codebook holds one
            row per attention head and is indexed by ``input``'s
            ``HEAD_AXIS``, which lets one tensor carry a table per head.
        bounds: The midpoints between entries, passed when the caller
            wants the index each element falls to rather than the entry
            itself -- what a compiled graph stores, to be decoded where
            it is read.  None derives them and returns the entry.

    Returns:
        Tensor shaped like ``input``, holding either the entry each
        element rounded to or that entry's index.

    Raises:
        ValueError: A per-head codebook has a different number of rows
            than the tensor has heads.
    """
    edges = bounds
    if edges is None:
        edges = (codebook[..., :-1] + codebook[..., 1:]) / 2

    # Rounding to bfloat16 first is what the lookup table does by
    # construction, so the two agree on everything but the entries.
    if codebook.dim() == 1:
        index = torch.searchsorted(edges, input.bfloat16().float())
        picked = index if bounds is not None else codebook[index]
        return picked.to(input.dtype)

    # One row per head, which ``searchsorted`` reaches only when the head
    # leads both operands: the rest of the tensor flattens behind it.
    heads = codebook.shape[0]
    if input.shape[HEAD_AXIS] != heads:
        raise ValueError(
            f"the codebook holds {heads} rows, but the tensor reaching it "
            f"has {input.shape[HEAD_AXIS]} heads"
        )
    moved = input.movedim(HEAD_AXIS, 0)
    flat = moved.reshape(heads, -1).bfloat16().float()
    index = torch.searchsorted(edges.contiguous(), flat.contiguous())
    picked = index if bounds is not None else codebook.gather(1, index)
    return picked.view(moved.shape).movedim(0, HEAD_AXIS).to(input.dtype)


def decode(input: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Read each stored index back as the entry it names.

    Args:
        input: Tensor of indices, as ``encode`` emitted them.
        codebook: The entries.  A 2-D codebook holds one row per attention
            head and is indexed by ``input``'s ``HEAD_AXIS``.

    Returns:
        Tensor shaped like ``input``, holding the entry each index names.

    Raises:
        ValueError: A per-head codebook has a different number of rows
            than the tensor has heads.
    """
    index = input.to(torch.long)
    if codebook.dim() == 1:
        return codebook[index].to(input.dtype)

    heads = codebook.shape[0]
    if input.shape[HEAD_AXIS] != heads:
        raise ValueError(
            f"the codebook holds {heads} rows, but the tensor reaching it "
            f"has {input.shape[HEAD_AXIS]} heads"
        )
    moved = index.movedim(HEAD_AXIS, 0)
    picked = codebook.gather(1, moved.reshape(heads, -1)).view(moved.shape)
    return picked.movedim(0, HEAD_AXIS).to(input.dtype)


quantized_ops_lib.define(
    "quantize(Tensor input, Tensor scale, Tensor? zero_point=None, "
    "SymInt[]? axes=None, int? block_size=None, Tensor? qmap=None, "
    "Tensor? output_code=None) -> Tensor"
)


@impl(quantized_ops_lib, "quantize", "CompositeExplicitAutograd")
def quantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    qmap: torch.Tensor = None,
    output_code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantization for the Tensor using scales and zero points to map
    from floating point to quantized values

    Args:
        input (torch.Tensor): original float32 or bfloat16 Tensor
        scale (torch.Tensor): scale factors for quantization
        zero_point (torch.Tensor): zero point for quantization, default is None
        axes (Tuple[int]): axes for group-wise quantization, default is None
        block_size (int): block size for group-wise quantization,
            default is None
        qmap (torch.Tensor): quantization map for mapping from float to
            quantized values. A map of ``QMAP_SIZE`` entries is a lookup
            table indexed by the bfloat16 bit pattern; a smaller one is a
            codebook, searched for the nearest entry, and a 2-D codebook
            carries one row per attention head.
        output_code (torch.Tensor): codebook for quantizing the output

    Returns:
        Tensor with requested dtype (e.g. int8), note the quantization
        parameters are not stored in the Tensor, we are storing them in
        function arguments instead
    """
    assert qmap is not None, "qmap must be provided for quantization"

    if block_size is not None:
        scale = expand(scale, input.shape, block_size)
        if zero_point is not None:
            zero_point = expand(zero_point, input.shape, block_size)

    if zero_point is None:
        input = input / scale
    else:
        input = input / scale + zero_point

    # Both a value table and the index table ``convert_pt2e`` swaps in have
    # one entry per bit pattern, so size is what tells a table from a
    # codebook -- their dtypes differ.
    if qmap.numel() == QMAP_SIZE and qmap.dim() == 1:
        return vmap(input, qmap)
    return encode(input, qmap, output_code)


quantized_ops_lib.define(
    "dequantize(Tensor input, Tensor scale, Tensor? zero_point=None, "
    "SymInt[]? axes=None, int? block_size=None, Tensor? input_qmap=None, "
    "Tensor? output_qmap=None) -> Tensor"
)


@impl(quantized_ops_lib, "dequantize", "CompositeExplicitAutograd")
def dequantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    input_qmap: Optional[torch.Tensor] = None,
    output_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantization for the Tensor using the same quantization parameters
    to map from floating point to quantized values

    Args:
        input (torch.Tensor): original float32 or bfloat16 Tensor
        scale (torch.Tensor): scale factors for dequantization
        zero_point (torch.Tensor): zero point for quantization, default is None
        axes (Tuple[int]): axes for group-wise quantization, default is None
        block_size (int): block size for group-wise quantization,
            default is None
        input_qmap (torch.Tensor): quantization map used to quantize the input
        output_qmap (torch.Tensor): quantization map used to quantize the output

    Returns:
        Tensor with floating point types, note the quantization parameters
        are not stored in the Tensor, we are storing them in function
        arguments instead
    """

    if input_qmap is not None:
        input = vmap(input, input_qmap)

    if block_size is not None:
        scale = expand(scale, input.shape, block_size)
        if zero_point is not None:
            zero_point = expand(zero_point, input.shape, block_size)

    if zero_point is None:
        dequantized = input * scale
    else:
        dequantized = (input - zero_point) * scale

    if output_qmap is not None:
        dequantized = vmap(dequantized, output_qmap)

    return dequantized


quantized_ops_lib.define(
    "conv2d_mx(Tensor input, Tensor weight, Tensor? bias=None, "
    "SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, "
    "SymInt groups=1, *, Tensor? input_scale=None, Tensor? weight_scale=None, "
    "int? block_size=None, Tensor? input_code=None, "
    'Tensor? weight_code=None, str layout="nchw") -> Tensor'
)


@impl(quantized_ops_lib, "conv2d_mx", "CompositeExplicitAutograd")
def conv2d_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    *,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
    layout: str = "nchw",
) -> torch.Tensor:
    from voyager_compiler.codegen.node_info import _pair

    assert layout in ("nchw", "nhwc"), layout

    # For codebook quantization, decode input and weight into float values first
    if input_code is not None:
        input = decode(input, input_code)
    if weight_code is not None:
        weight = decode(weight, weight_code)

    # Replicate scales to match input and weight shapes
    if input_scale is not None:
        input = input * expand(input_scale, input.shape, block_size)
    if weight_scale is not None:
        weight = weight * expand(weight_scale, weight.shape, block_size)

    # The dispatcher fills omitted / default-valued args from this Python
    # signature's scalar defaults; the strict op calls below need pairs.
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)

    if layout == "nhwc":
        # The generated NHWC twin (NHWC activations, HWIO weight).
        return torch.ops.quantized_ops.conv2d(
            input, weight, bias, stride, padding, dilation, groups
        )
    return torch.ops.aten.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )


quantized_ops_lib.define(
    "linear_mx(Tensor input, Tensor weight, Tensor? bias=None, *, "
    "Tensor? input_scale=None, Tensor? weight_scale=None, "
    "int? block_size=None, Tensor? input_code=None, Tensor? weight_code=None, "
    "Tensor? A_data=None, Tensor? A_indices=None, Tensor? A_indptr=None, "
    'str weight_layout="kc") -> Tensor'
)


@impl(quantized_ops_lib, "linear_mx", "CompositeExplicitAutograd")
def linear_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    *,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
    A_data: Optional[torch.Tensor] = None,
    A_indices: Optional[torch.Tensor] = None,
    A_indptr: Optional[torch.Tensor] = None,
    weight_layout: str = "kc",
) -> torch.Tensor:
    assert weight_layout in ("kc", "ck"), weight_layout

    if input_code is not None:
        input = decode(input, input_code)

    if input_scale is not None:
        input = input * expand(input_scale, input.shape, block_size)

    decoded_weight = weight
    if weight_code is not None:
        decoded_weight = decode(weight, weight_code)

    if weight_scale is not None:
        decoded_weight = decoded_weight * expand(
            weight_scale, weight.shape, block_size
        )

    # Call the operator matching the weight's storage layout: aten for
    # the KC-native layout, the layout twin for a CK-stored weight.
    if weight_layout == "kc":
        dense_out = torch.ops.aten.linear(input, decoded_weight, bias)
    else:
        dense_out = torch.ops.quantized_ops.linear(input, decoded_weight, bias)

    if A_data is not None:
        spmm_out = torch.ops.quantized_ops.spmm_csr(
            A_data,
            A_indices,
            A_indptr,
            weight,
            weight_scale,
            weight_code,
            block_size,
            weight_layout,
        )
        return dense_out + spmm_out

    return dense_out


@torch.library.register_fake("quantized_ops::linear_mx")
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    **kwargs,
):
    if kwargs.get("weight_layout", "kc") == "ck":
        return torch.ops.quantized_ops.linear(input, weight, bias)
    return torch.ops.aten.linear(input, weight, bias)


quantized_ops_lib.define(
    "matmul_mx(Tensor self, Tensor other, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, int? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None, Tensor? A_data=None, Tensor? A_indices=None, "
    'Tensor? A_indptr=None, str weight_layout="ck") -> Tensor'
)


@impl(quantized_ops_lib, "matmul_mx", "CompositeExplicitAutograd")
def matmul_mx(
    self: torch.Tensor,
    other: torch.Tensor,
    *,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
    A_data: Optional[torch.Tensor] = None,
    A_indices: Optional[torch.Tensor] = None,
    A_indptr: Optional[torch.Tensor] = None,
    weight_layout: str = "ck",
) -> torch.Tensor:
    assert weight_layout in ("kc", "ck"), weight_layout

    if input_code is not None:
        self = decode(self, input_code)
    if input_scale is not None:
        self = self * expand(input_scale, self.shape, block_size)

    decoded_other = other
    if weight_code is not None:
        decoded_other = decode(other, weight_code)
    if weight_scale is not None:
        decoded_other = decoded_other * expand(
            weight_scale, other.shape, block_size
        )

    # Call the operator matching the right operand's storage layout:
    # aten for the CK-native layout, the layout twin for KC storage.
    if weight_layout == "ck":
        dense_out = torch.ops.aten.matmul(self, decoded_other)
    else:
        dense_out = torch.ops.quantized_ops.matmul(self, decoded_other)

    if A_data is not None:
        spmm_out = torch.ops.quantized_ops.spmm_csr(
            A_data,
            A_indices,
            A_indptr,
            other,
            weight_scale,
            weight_code,
            block_size,
            weight_layout,
        )
        return dense_out + spmm_out

    return dense_out


@torch.library.register_fake("quantized_ops::matmul_mx")
def _(
    self: torch.Tensor,
    other: torch.Tensor,
    **kwargs,
):
    if kwargs.get("weight_layout", "ck") == "kc":
        return torch.ops.quantized_ops.matmul(self, other)
    return torch.ops.aten.matmul(self, other)


quantized_ops_lib.define(
    "calculate_mx_qparam(Tensor self, SymInt[] axes, int block_size, "
    "float quant_max, bool force_scale_power_of_two=False, "
    "Tensor scale_qmap=None) -> Tensor"
)


@impl(quantized_ops_lib, "calculate_mx_qparam", "CompositeExplicitAutograd")
def calculate_mx_qparam(
    input: torch.Tensor,
    axes: Union[int, List[int]],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Deferred to break a real cycle: ``quantization`` imports this module for
    # the ops, so it cannot be imported here at module scope.
    from voyager_compiler.quantization.mx_utils import (
        _reshape_to_blocks,
        _shared_exponents,
    )

    assert block_size > 0

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + input.ndim if x < 0 else x for x in axes]

    # Perform tiling to the hardware vector size
    input, axes, orig_shape, padded_shape = _reshape_to_blocks(
        input, axes, block_size
    )

    shared_exp_axes = [x + 1 for x in axes]

    if force_scale_power_of_two:
        # Get shared exponents
        shared_exp = _shared_exponents(
            input,
            method="max",
            axes=shared_exp_axes,
            ebits=0,
        )

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - math.floor(math.log2(quant_max))

        for axis in reversed(axes):
            # Remove extra dimension
            shared_exp = torch.squeeze(shared_exp, dim=axis + 1)

        scale = 2**shared_exp
    else:
        # Use absolute maximum value to compute scaling factors
        amax = torch.amax(torch.abs(input), dim=shared_exp_axes)
        scale = amax / quant_max

        # Quantize the scale using the codebook
        if scale_qmap is not None:
            scale = vmap(scale, scale_qmap)

    scale = torch.where(scale > 0.0, scale, 1.0)
    return scale


quantized_ops_lib.define(
    "quantize_mx(Tensor self, Tensor qmap, SymInt[] axes, int block_size, "
    "float quant_max, bool force_scale_power_of_two=False, "
    "Tensor scale_qmap=None, Tensor output_code=None) -> (Tensor, Tensor)"
)


@impl(quantized_ops_lib, "quantize_mx", "CompositeExplicitAutograd")
def quantize_mx(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
    output_code: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    scale = calculate_mx_qparam(
        input,
        axes=axes,
        block_size=block_size,
        quant_max=quant_max,
        force_scale_power_of_two=force_scale_power_of_two,
        scale_qmap=scale_qmap,
    )
    input = quantize(input, scale, None, axes, block_size, qmap, output_code)
    return scale, input


quantized_ops_lib.define(
    "quantize_affine(Tensor self, Tensor qmap, SymInt[] axes, int block_size, "
    "float quant_min, float quant_max, Tensor scale_qmap=None) "
    "-> (Tensor, Tensor, Tensor)"
)


@impl(quantized_ops_lib, "quantize_affine", "CompositeExplicitAutograd")
def quantize_affine(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    quant_min: float,
    quant_max: float,
    scale_qmap: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    """Group-wise affine quantization with the qparams computed on the fly.

    Each block of ``block_size`` elements along the one axis in ``axes`` maps
    its own ``[min, max]`` onto ``[quant_min, quant_max]``, so the result
    carries a scale *and* a zero point per block.  The runtime twin of the
    calibrated ``quantize``, the way ``quantize_mx`` is for microscaling.

    Args:
        input: The tensor to quantize.
        qmap: Quantization map turning the scaled values into codes.
        axes: The single axis the blocks lie along, as a one-element list
            (the same argument shape as ``quantize_mx``).
        block_size: Elements per block along the axis.
        quant_min: Smallest code of the target dtype.
        quant_max: Largest code of the target dtype.
        scale_qmap: Quantization map for the scale and zero point, or
            ``None`` to keep them in the input's dtype.

    Returns:
        ``(scale, zero_point, value)``.
    """
    # Deferred to break a real cycle: ``quantization`` imports this module for
    # the ops, so it cannot be imported here at module scope.
    from voyager_compiler.quantization.mx_utils import _reshape_to_blocks

    assert block_size > 0

    (axis,) = axes
    axes = [axis % input.ndim]
    blocked, block_axes, _, _ = _reshape_to_blocks(input, axes, block_size)
    reduce_axes = [x + 1 for x in block_axes]

    low = torch.amin(blocked, dim=reduce_axes)
    high = torch.amax(blocked, dim=reduce_axes)
    scale = (high - low) / (quant_max - quant_min)
    scale = torch.where(scale > 0.0, scale, 1.0)
    zero_point = -low / scale + quant_min
    if scale_qmap is not None:
        scale = vmap(scale, scale_qmap)
        zero_point = vmap(zero_point, scale_qmap)

    value = quantize(input, scale, zero_point, axes, block_size, qmap)
    return scale, zero_point, value


quantized_ops_lib.define(
    "filter_outlier(Tensor input, float threshold, float max_pct=0.01) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)


def _pad_csr(
    values: torch.Tensor,
    col_indices: torch.Tensor,
    crow_indices: torch.Tensor,
    max_nnz: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nse = crow_indices[-1].item()

    pad_len = max_nnz - nse

    if pad_len >= 0:
        data = F.pad(values, (0, pad_len), mode="constant", value=0)
        indices = F.pad(col_indices, (0, pad_len), mode="constant", value=-1)
        return data, indices

    # Overflow cannot raise here: ``ShapeProp(recurse=True)`` stamps a nest by
    # running both ``cond`` branches and one ``while_loop`` iteration from the
    # carried *initial* values, so a tiled quantize is propagated over an
    # accumulator no step has written yet and reads uninitialized memory as
    # nearly all-outlier.  The capacity a tile really needs is checked against
    # the live activation where the builder derives its geometry.
    logger.warning(
        f"Number of outliers {nse} exceeds capacity {max_nnz}; "
        f"{nse - max_nnz} dropped."
    )
    crow_indices.clamp_(max=max_nnz)

    return values[:max_nnz], col_indices[:max_nnz]


@impl(quantized_ops_lib, "filter_outlier", "CompositeExplicitAutograd")
def filter_outlier(
    input: torch.Tensor, threshold: float, max_pct: float = 0.01
) -> Tuple[torch.Tensor]:
    """Filter out outliers in the input tensor based on a threshold.

    Args:
        input (torch.Tensor): Input tensor.
        threshold (float): Threshold for filtering out outliers.

    Returns:
        torch.Tensor: Filtered tensor.
    """
    is_outlier = torch.abs(input) > threshold
    inlier = torch.where(is_outlier, 0, input)
    outliers = torch.where(is_outlier, input, 0)

    sparsity = (1 - torch.sum(is_outlier) / input.numel()) * 100
    logger.info(f"Outlier sparsity level: {sparsity:.2f}%")

    batch_shape = input.shape[:-2]
    mat_shape = input.shape[-2:]
    max_nnz = int(math.prod(mat_shape) * max_pct)

    num_batches = int(math.prod(batch_shape))

    outliers_flat = outliers.reshape(num_batches, *mat_shape)

    all_crow_indices = []
    all_col_indices = []
    all_values = []

    for i in range(num_batches):
        csr = outliers_flat[i].to_sparse_csr()

        crow_indices = csr.crow_indices().to(torch.int32)
        col_indices = csr.col_indices().to(torch.int32)
        values = csr.values()

        data, indices = _pad_csr(values, col_indices, crow_indices, max_nnz)

        all_crow_indices.append(crow_indices)
        all_col_indices.append(indices)
        all_values.append(data)

    crow_indices = torch.stack(all_crow_indices, dim=0).reshape(
        *batch_shape, -1
    )
    indices = torch.stack(all_col_indices, dim=0).reshape(*batch_shape, -1)
    data = torch.stack(all_values, dim=0).reshape(*batch_shape, -1)

    return data, indices, crow_indices, inlier


@torch.library.register_fake("quantized_ops::filter_outlier")
def _(
    input: torch.Tensor,
    threshold: float,
    max_pct: float = 0.01,
):
    batch_shape = input.shape[:-2]
    mat_shape = input.shape[-2:]
    max_nnz = int(math.prod(mat_shape) * max_pct)

    indptr = input.new_empty(
        (*batch_shape, mat_shape[0] + 1), dtype=torch.int32
    )
    indices = input.new_empty((*batch_shape, max_nnz), dtype=torch.int32)
    data = input.new_empty((*batch_shape, max_nnz))

    inliers = torch.empty_like(input)
    return data, indices, indptr, inliers


quantized_ops_lib.define(
    "quantize_mx_outlier(Tensor self, Tensor qmap, SymInt[] axes, "
    "int block_size, float quant_max, bool force_scale_power_of_two=False, "
    "Tensor scale_qmap=None, Tensor output_code=None, float? threshold=None, "
    "float max_pct=0.01, SymInt indptr_offset=0) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


@impl(quantized_ops_lib, "quantize_mx_outlier", "CompositeExplicitAutograd")
def quantize_mx_outlier(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
    output_code: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    max_pct: float = 0.01,
    indptr_offset: int = 0,
) -> Tuple[torch.Tensor]:
    """Split an input into quantized inliers and a CSR of outliers.

    Args:
        input: Tensor to quantize.
        qmap: Codebook the inliers are quantized into.
        axes: Axes the microscaling blocks run along.
        block_size: Elements per microscaling block.
        quant_max: Largest representable magnitude of ``qmap``.
        force_scale_power_of_two: Round each block scale down to a power
            of two.
        scale_qmap: Codebook the scales are quantized into.
        output_code: What a codebook emits per entry, in place of the
            entry itself; ignored when ``qmap`` is a lookup table.
        threshold: Magnitude above which an element is an outlier.
        max_pct: Fraction of the matrix the CSR is sized to hold.
        indptr_offset: Value the returned row pointers start from, so a
            tile's CSR can extend one that is already partly built. The
            consumer subtracts ``indptr[0]`` to recover data positions.

    Returns:
        ``(data, indices, indptr, scale, inliers)``.
    """
    data, indices, indptr, inliers = filter_outlier(input, threshold, max_pct)

    if indptr_offset:
        indptr = indptr + indptr_offset

    scale = calculate_mx_qparam(
        inliers,
        axes=axes,
        block_size=block_size,
        quant_max=quant_max,
        force_scale_power_of_two=force_scale_power_of_two,
        scale_qmap=scale_qmap,
    )
    inliers = quantize(
        inliers, scale, None, axes, block_size, qmap, output_code
    )

    return data, indices, indptr, scale, inliers


@torch.library.register_fake("quantized_ops::quantize_mx_outlier")
def _(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
    output_code: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    max_pct: float = 0.01,
    indptr_offset: int = 0,
):
    batch_shape = input.shape[:-2]
    mat_shape = input.shape[-2:]
    max_nnz = int(math.prod(mat_shape) * max_pct)

    indptr = input.new_empty(
        (*batch_shape, mat_shape[0] + 1), dtype=torch.int32
    )
    indices = input.new_empty((*batch_shape, max_nnz), dtype=torch.int32)
    data = input.new_empty((*batch_shape, max_nnz))

    scale_shape = list(input.shape)
    for axis in axes:
        scale_shape[axis] = math.ceil(scale_shape[axis] / block_size)
    scale = input.new_empty(scale_shape)

    inliers = torch.empty_like(input, memory_format=torch.contiguous_format)
    return data, indices, indptr, scale, inliers


quantized_ops_lib.define(
    "slice_csr_tensor(Tensor data, Tensor indices, Tensor indptr, int dim=0, "
    "SymInt? start=None, SymInt? end=None, float size_factor=1.0) "
    "-> (Tensor, Tensor, Tensor)"
)


@impl(quantized_ops_lib, "slice_csr_tensor", "CompositeExplicitAutograd")
def slice_csr_tensor(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    dim: int = 0,
    start: int = None,
    end: int = None,
    size_factor: float = 1,
) -> Tuple[torch.Tensor]:
    dim = dim + 2 if dim < 0 else dim

    if dim not in [0, 1]:
        raise ValueError(f"Cannot slice sparse CSR matrix along dim {dim}")

    if dim == 0:
        start_idx = indptr[start].item()
        end_idx = indptr[end].item()

        new_indptr = indptr[start : end + 1] - start_idx
        values = data[start_idx:end_idx]
        col_indices = indices[start_idx:end_idx]
    else:
        mask = (indices >= start) & (indices < end)

        row_lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
        nse = indptr[-1].item()

        counts = torch.segment_reduce(
            mask[:nse].to(torch.float32),
            "sum",
            lengths=row_lengths,
        ).to(indptr.dtype)

        new_indptr = torch.empty_like(indptr)
        new_indptr[0] = 0
        new_indptr[1:] = counts.cumsum(0).to(indptr.dtype)

        values = data[mask]
        col_indices = indices[mask] - start

    max_nnz = int(data.shape[0] * size_factor)
    new_data, new_indices = _pad_csr(values, col_indices, new_indptr, max_nnz)

    return new_data, new_indices, new_indptr


@torch.library.register_fake("quantized_ops::slice_csr_tensor")
def _(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    dim: int = 0,
    start: int = None,
    end: int = None,
    size_factor: float = 1,
):
    fake_data = torch.empty_like(data)
    fake_indices = torch.empty_like(indices)
    fake_indptr = torch.empty_like(indptr)
    if dim == 0 or dim == -2:
        return fake_data, fake_indices, fake_indptr[start : end + 1]
    return fake_data, fake_indices, fake_indptr


quantized_ops_lib.define(
    "spmm_csr(Tensor data, Tensor indices, Tensor indptr, Tensor B, "
    "Tensor? B_scale=None, Tensor? B_code=None, int? block_size=None, "
    'str weight_layout="kc") -> Tensor'
)


@impl(quantized_ops_lib, "spmm_csr", "CompositeExplicitAutograd")
def spmm_csr(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor] = None,
    B_code: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    weight_layout: str = "kc",
) -> torch.Tensor:
    assert weight_layout in ("kc", "ck"), weight_layout

    if B_code is not None:
        B = decode(B, B_code)
    if B_scale is not None:
        B = B * expand(B_scale, B.shape, block_size)

    # A KC-stored B is [out, contraction]; the sparse mm needs
    # [contraction, out], so transpose it.  A CK-stored B is already
    # in that layout.
    if weight_layout == "kc":
        B = B.mT

    batch_shape = indptr.shape[:-1]
    num_batches = int(math.prod(batch_shape))

    indptr = indptr.reshape(-1, indptr.shape[-1])
    indices = indices.reshape(-1, indices.shape[-1])
    data = data.reshape(-1, data.shape[-1])

    if B.ndim > 2:
        B = B.reshape(num_batches, B.shape[-2], B.shape[-1])

    outputs = []

    for i in range(num_batches):
        B_batch = B[i] if B.ndim == 3 else B

        input_size = (indptr[i].numel() - 1, B_batch.shape[0])

        # The row pointers may name a range lifted out of a longer array,
        # in which case they do not start at 0 and data/indices hold only
        # the span they cover.  Rebase both onto that span's start.
        base = indptr[i][0].item()
        end_index = indptr[i][-1].item() - base

        csr = torch.sparse_csr_tensor(
            indptr[i] - base,
            indices[i, :end_index],
            data[i, :end_index],
            dtype=torch.float32,
            size=input_size,
        )

        # Sparse mm only supports float32 for now
        output = torch.sparse.mm(csr, B_batch.to(torch.float32)).to(B.dtype)
        outputs.append(output)

    output = torch.stack(outputs, dim=0)
    output = output.reshape(*batch_shape, -1, B.shape[-1])

    return output


@torch.library.register_fake("quantized_ops::spmm_csr")
def _(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor] = None,
    B_code: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    weight_layout: str = "kc",
):
    batch_shape = indptr.shape[:-1]
    X = indptr.shape[-1] - 1
    K = B.shape[-1] if weight_layout == "ck" else B.shape[-2]
    return data.new_empty((*batch_shape, X, K))
