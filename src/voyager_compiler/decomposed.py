import logging
import math
from typing import Tuple, Union, Optional, List

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from .mx_utils import _reshape_to_blocks, _shared_exponents

logger = logging.getLogger(__name__)


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_ops", "DEF")


quantized_decomposed_lib.define(
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, "
    "SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor"
)


@impl(quantized_decomposed_lib, "conv2d", "CompositeExplicitAutograd")
def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    return F.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )


quantized_decomposed_lib.define(
    "max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, "
    "int[2] dilation=1, bool ceil_mode=False) -> Tensor"
)


@impl(quantized_decomposed_lib, "max_pool2d", "CompositeExplicitAutograd")
def max_pool2d(
    self: torch.Tensor,
    kernel_size: Union[int, Tuple[int]] = 1,
    stride: Union[int, Tuple[int]] = None,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False
) -> torch.Tensor:
    return F.max_pool2d(
        self.permute(0, 3, 1, 2),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    ).permute(0, 2, 3, 1)


quantized_decomposed_lib.define(
    "adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor"
)


@impl(quantized_decomposed_lib, "adaptive_avg_pool2d", "CompositeExplicitAutograd")
def adaptive_avg_pool2d(self: torch.Tensor, output_size: Union[int, Tuple[int]] = 1) -> torch.Tensor:
    return F.adaptive_avg_pool2d(self.permute(0, 3, 1, 2), output_size).permute(0, 2, 3, 1)


quantized_decomposed_lib.define(
    "linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "linear", "CompositeExplicitAutograd")
def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return F.linear(input, weight, bias)


quantized_decomposed_lib.define(
    "matmul(Tensor self, Tensor other) -> Tensor"
)


@impl(quantized_decomposed_lib, "matmul", "CompositeExplicitAutograd")
def matmul(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.matmul(self, other)


quantized_decomposed_lib.define(
    "layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, "
    "Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"
)


@impl(quantized_decomposed_lib, "layer_norm", "CompositeExplicitAutograd")
def layer_norm(
    input: torch.Tensor,
    normalized_shape: Union[int, Tuple[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    cudnn_enable: bool = True
) -> torch.Tensor:
    output = torch.ops.aten.layer_norm.default(
        input[..., :normalized_shape[-1]],
        normalized_shape,
        weight[..., :normalized_shape[-1]],
        bias[..., :normalized_shape[-1]],
        eps,
        cudnn_enable
    )
    # Pad the output back to the original input shape
    output = torch.nn.functional.pad(
        output, (0, input.shape[-1] - normalized_shape[-1])
    )
    return output


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


quantized_decomposed_lib.define("vmap(Tensor self, Tensor other) -> Tensor")


@impl(quantized_decomposed_lib, "vmap", "CompositeExplicitAutograd")
def vmap(input: torch.Tensor, qmap: torch.Tensor, chunk_size=1024*1024) -> torch.Tensor:
    input_dtype = input.dtype

    if input.dtype != torch.bfloat16:
        input = input.to(torch.bfloat16)

    indices = input.view(torch.int16)

    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    indices_flat = indices.reshape(-1)
    output_flat = output.view(-1)

    for start in range(0, indices_flat.numel(), chunk_size):
        end = min(start + chunk_size, indices_flat.numel())
        indices_chunk = indices_flat[start:end].to(torch.int32) & 0xffff
        output_flat[start:end] = qmap[indices_chunk]

    return output.to(input_dtype)


quantized_decomposed_lib.define(
    "quantize(Tensor input, Tensor scale, Tensor? zero_point=None, SymInt[]? axes=None, "
    "int? block_size=None, Tensor? qmap=None, Tensor? output_code=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize", "CompositeExplicitAutograd")
def quantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    qmap: torch.Tensor = None,
    output_code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ Quantization for the Tensor using scales and zero points to map
    from floating point to quantized values

    Args:
        input (torch.Tensor): original float32 or bfloat16 Tensor
        scale (torch.Tensor): scale factors for quantization
        zero_point (torch.Tensor): zero point for quantization, default is None
        axes (Tuple[int]): axes for group-wise quantization, default is None
        block_size (int): block size for group-wise quantization, default is None
        qmap (torch.Tensor): quantization map for mapping from float to quantized values
        output_code (torch.Tensor): codebook for quantizing the output

    Returns:
        Tensor with requested dtype (e.g. int8), note the quantization parameters
        are not stored in the Tensor, we are storing them in function arguments instead
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

    return vmap(input, qmap)


@torch.library.register_fake("quantized_ops::quantize")
def _(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    qmap: torch.Tensor = None,
    output_code: Optional[torch.Tensor] = None,
):
    return torch.empty_like(input)


quantized_decomposed_lib.define(
    "dequantize(Tensor input, Tensor scale, Tensor? zero_point=None, SymInt[]? axes=None, "
    "int? block_size=None, Tensor? input_qmap=None, Tensor? output_qmap=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize", "CompositeExplicitAutograd")
def dequantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    input_qmap: Optional[torch.Tensor] = None,
    output_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ Dequantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
        input (torch.Tensor): original float32 or bfloat16 Tensor
        scale (torch.Tensor): scale factors for dequantization
        zero_point (torch.Tensor): zero point for quantization, default is None
        axes (Tuple[int]): axes for group-wise quantization, default is None
        block_size (int): block size for group-wise quantization, default is None
        input_qmap (torch.Tensor): quantization map used to quantize the input
        output_qmap (torch.Tensor): quantization map used to quantize the output

    Returns:
        Tensor with floating point types, note the quantization parameters
        are not stored in the Tensor, we are storing them in function arguments instead
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


@torch.library.register_fake("quantized_ops::dequantize")
def _(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    input_qmap: Optional[torch.Tensor] = None,
    output_qmap: Optional[torch.Tensor] = None,
):
    return torch.empty_like(input)


quantized_decomposed_lib.define(
    "conv2d_mx(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, "
    "SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, int? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "conv2d_mx", "CompositeExplicitAutograd")
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
) -> torch.Tensor:
    # For codebook quantization, decode input and weight into float values first
    if input_code is not None:
        input = input_code[input.to(torch.long)].to(input.dtype)
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)].to(weight.dtype)

    # Replicate scales to match input and weight shapes
    if input_scale is not None:
        input = input * expand(input_scale, input.shape, block_size)
    if weight_scale is not None:
        weight = weight * expand(weight_scale, weight.shape, block_size)

    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@torch.library.register_fake("quantized_ops::conv2d_mx")
def _(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    **kwargs,
):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


quantized_decomposed_lib.define(
    "linear_mx(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor? input_scale=None, "
    "Tensor? weight_scale=None, int? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None, Tensor? A_data=None, Tensor? A_indices=None, Tensor? A_indptr=None, "
    "bool weight_transposed=False) -> Tensor"
)


@impl(quantized_decomposed_lib, "linear_mx", "CompositeExplicitAutograd")
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
    weight_transposed=False
) -> torch.Tensor:
    if input_code is not None:
        input = input_code[input.to(torch.long)].to(input.dtype)

    if input_scale is not None:
        input = input * expand(input_scale, input.shape, block_size)

    decoded_weight = weight
    if weight_code is not None:
        decoded_weight = weight_code[weight.to(torch.long)].to(weight.dtype)

    if weight_scale is not None:
        decoded_weight = decoded_weight * expand(weight_scale, weight.shape, block_size)

    dense_out = F.linear(input, decoded_weight, bias)

    if A_data is not None:
        spmm_out = torch.ops.quantized_ops.spmm_csr(
            A_data,
            A_indices,
            A_indptr,
            weight,
            weight_scale,
            weight_code,
            block_size,
            weight_transposed,
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
    return F.linear(input, weight, bias)


quantized_decomposed_lib.define(
    "matmul_mx(Tensor self, Tensor other, *, Tensor? input_scale=None, Tensor? "
    "weight_scale=None, int? block_size=None, Tensor? input_code=None, Tensor? "
    "weight_code=None, Tensor? A_data=None, Tensor? A_indices=None, Tensor? A_indptr=None, "
    "bool weight_transposed=True) -> Tensor"
)


@impl(quantized_decomposed_lib, "matmul_mx", "CompositeExplicitAutograd")
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
    weight_transposed=True
) -> torch.Tensor:
    if input_code is not None:
        self = input_code[self.to(torch.long)].to(self.dtype)
    if input_scale is not None:
        self = self * expand(input_scale, self.shape, block_size)

    decoded_other = other
    if weight_code is not None:
        decoded_other = weight_code[other.to(torch.long)].to(other.dtype)
    if weight_scale is not None:
        decoded_other = decoded_other * expand(weight_scale, other.shape, block_size)

    dense_out = torch.matmul(self, decoded_other)

    if A_data is not None:
        spmm_out = torch.ops.quantized_ops.spmm_csr(
            A_data,
            A_indices,
            A_indptr,
            other,
            weight_scale,
            weight_code,
            block_size,
            weight_transposed,
        )
        return dense_out + spmm_out

    return dense_out


@torch.library.register_fake("quantized_ops::matmul_mx")
def _(
    self: torch.Tensor,
    other: torch.Tensor,
    **kwargs,
):
    return torch.matmul(self, other)


quantized_decomposed_lib.define(
    "calculate_mx_qparam(Tensor self, SymInt[] axes, int block_size, float quant_max, "
    "bool force_scale_power_of_two=False, Tensor scale_qmap=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "calculate_mx_qparam", "CompositeExplicitAutograd")
def calculate_mx_qparam(
    input: torch.Tensor,
    axes: Union[int, List[int]],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
            input, method="max", axes=shared_exp_axes, ebits=0,
        )

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - math.floor(math.log2(quant_max))

        for axis in reversed(axes):
            # Remove extra dimension
            shared_exp = torch.squeeze(shared_exp, dim=axis + 1)

        scale = 2 ** shared_exp
    else:
        # Use absolute maximum value to compute scaling factors
        amax = torch.amax(torch.abs(input), dim=shared_exp_axes)
        scale = amax / quant_max

        # Quantize the scale using the codebook
        if scale_qmap is not None:
            scale = vmap(scale, scale_qmap)

    scale = torch.where(scale > 0.0, scale, 1.0)
    return scale


@torch.library.register_fake("quantized_ops::calculate_mx_qparam")
def _(
    input: torch.Tensor,
    axes: Union[int, List[int]],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
):
    scale_shape = list(input.shape)
    for axis in axes:
        scale_shape[axis] = math.ceil(scale_shape[axis] / block_size)
    return input.new_empty(scale_shape)


quantized_decomposed_lib.define(
    "quantize_mx(Tensor self, Tensor qmap, SymInt[] axes, int block_size, float quant_max, "
    "bool force_scale_power_of_two=False, Tensor scale_qmap=None, Tensor output_code=None) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "quantize_mx", "CompositeExplicitAutograd")
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
    input = quantize(input, scale, None, axes, block_size, qmap)
    return scale, input


@torch.library.register_fake("quantized_ops::quantize_mx")
def _(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
    output_code: Optional[torch.Tensor] = None,
):
    scale = torch.ops.quantized_ops.calculate_mx_qparam(
        input,
        axes,
        block_size,
        quant_max,
        force_scale_power_of_two,
        scale_qmap,
    )
    return scale, torch.empty_like(input)


quantized_decomposed_lib.define(
    "filter_outlier(Tensor input, float threshold, float max_pct=0.01) -> (Tensor, Tensor, Tensor, Tensor)"
)


def _pad_csr(
    values: torch.Tensor,
    col_indices: torch.Tensor,
    crow_indices: torch.Tensor,
    max_nnz: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nse = crow_indices[-1].item()

    pad_len = max_nnz - nse

    if pad_len > 0:
        data = F.pad(values, (0, pad_len), mode='constant', value=0)
        indices = F.pad(col_indices, (0, pad_len), mode='constant', value=-1)
    else:
        logger.warning(f"Number of outliers {nse} exceeds capacity {max_nnz}.")
        data = values[:max_nnz]
        indices = col_indices[:max_nnz]
        crow_indices.clamp_(max=max_nnz)

    return data, indices


@impl(quantized_decomposed_lib, "filter_outlier", "CompositeExplicitAutograd")
def filter_outlier(input: torch.Tensor, threshold: float, max_pct: float = 0.01) -> Tuple[torch.Tensor]:
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

    crow_indices = torch.stack(all_crow_indices, dim=0).reshape(*batch_shape, -1)
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

    indptr = input.new_empty((*batch_shape, mat_shape[0] + 1), dtype=torch.int32)
    indices = input.new_empty((*batch_shape, max_nnz), dtype=torch.int32)
    data = input.new_empty((*batch_shape, max_nnz))

    inliers = torch.empty_like(input)
    return data, indices, indptr, inliers


quantized_decomposed_lib.define(
    "quantize_mx_outlier(Tensor self, Tensor qmap, SymInt[] axes, int block_size, "
    "float quant_max, bool force_scale_power_of_two=False, Tensor scale_qmap=None, "
    "Tensor output_code=None, float? threshold=None, float max_pct=0.01) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "quantize_mx_outlier", "CompositeExplicitAutograd")
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
    max_pct: float = 0.01
) -> Tuple[torch.Tensor]:
    data, indices, indptr, inliers = filter_outlier(input, threshold, max_pct)

    scale = calculate_mx_qparam(
        inliers,
        axes=axes,
        block_size=block_size,
        quant_max=quant_max,
        force_scale_power_of_two=force_scale_power_of_two,
        scale_qmap=scale_qmap,
    )
    inliers = quantize(inliers, scale, None, axes, block_size, qmap)

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
    max_pct: float = 0.01
):
    batch_shape = input.shape[:-2]
    mat_shape = input.shape[-2:]
    max_nnz = int(math.prod(mat_shape) * max_pct)

    indptr = input.new_empty((*batch_shape, mat_shape[0] + 1), dtype=torch.int32)
    indices = input.new_empty((*batch_shape, max_nnz), dtype=torch.int32)
    data = input.new_empty((*batch_shape, max_nnz))

    scale_shape = list(input.shape)
    for axis in axes:
        scale_shape[axis] = math.ceil(scale_shape[axis] / block_size)
    scale = input.new_empty(scale_shape)

    inliers = torch.empty_like(input)
    return data, indices, indptr, scale, inliers


quantized_decomposed_lib.define(
    "slice_csr_tensor(Tensor data, Tensor indices, Tensor indptr, int dim=0, "
    "SymInt? start=None, SymInt? end=None, float size_factor=1.0) -> (Tensor, Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "slice_csr_tensor", "CompositeExplicitAutograd")
def slice_csr_tensor(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    dim: int = 0,
    start: int = None,
    end: int = None,
    size_factor: float = 1
) -> Tuple[torch.Tensor]:
    dim = dim + 2 if dim < 0 else dim

    if dim not in [0, 1]:
        raise ValueError(f"Cannot slice sparse CSR matrix along dim {dim}")

    if dim == 0:
        start_idx = indptr[start].item()
        end_idx = indptr[end].item()

        new_indptr = indptr[start:end + 1] - start_idx
        values = data[start_idx:end_idx]
        col_indices = indices[start_idx:end_idx]
    else:
        mask = (indices >= start) & (indices < end)

        row_lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
        nse = indptr[-1].item()

        counts = torch.segment_reduce(
            mask[:nse].to(torch.float32), "sum", lengths=row_lengths,
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
    size_factor: float = 1
):
    fake_data = torch.empty_like(data)
    fake_indices = torch.empty_like(indices)
    fake_indptr = torch.empty_like(indptr)
    if dim == 0 or dim == -2:
        return fake_data, fake_indices, fake_indptr[start:end + 1]
    return fake_data, fake_indices, fake_indptr


quantized_decomposed_lib.define(
    "spmm_csr(Tensor data, Tensor indices, Tensor indptr, Tensor B, Tensor? B_scale=None, "
    "Tensor? B_code=None, int? block_size=None, bool weight_transposed=False) -> Tensor"
)


@impl(quantized_decomposed_lib, "spmm_csr", "CompositeExplicitAutograd")
def spmm_csr(
    data: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor] = None,
    B_code: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    weight_transposed=False
) -> torch.Tensor:
    if B_code is not None:
        B = B_code[B.to(torch.long)]
    if B_scale is not None:
        B = B * expand(B_scale, B.shape, block_size)

    if not weight_transposed:
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
        end_index = indptr[i][-1].item()

        csr = torch.sparse_csr_tensor(
            indptr[i],
            indices[i,:end_index],
            data[i,:end_index],
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
    weight_transposed=False
):
    batch_shape = indptr.shape[:-1]
    X = indptr.shape[-1] - 1
    K = B.shape[-1] if weight_transposed else B.shape[-2]
    return data.new_empty((*batch_shape, X, K))
