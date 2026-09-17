# Interpret compiler operation shapes and logical channel metadata for both backends
from dataclasses import replace
from math import prod
from .workload import Workload
from .timing.transfer import ceil_div, transfer_cycles
import re
from .models.cim_timing import TimingOptions
import os


# Select the matrix operation without silently accepting nested compiler loops
def matrix_operation(operation):
    if operation.HasField("op"):
        return operation.op
    if operation.HasField("fused_op") and operation.fused_op.op_list:
        return operation.fused_op.op_list[0]
    raise ValueError("mapping requires a flat model with nonempty operations")


# Match the compiler consumer's logical reshape and SoC tile selection
def tensor_shape(tensor, *, reshape=True):
    if reshape and tensor.HasField("reshape"):
        shape = tensor.reshape.kwargs["output_shape"].int_list.values
    elif os.environ.get("SOC_SIM", "0") == "1" and tensor.tiled_shape:
        shape = tensor.tiled_shape
    else:
        shape = tensor.shape
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("mapping requires positive tensor shapes")
    return tuple(shape)


# Normalize the command ABI's equal X/Y stride, padding, or dilation
def symmetric_parameter(operation, name, default):
    if name not in operation.kwargs:
        return default
    value = operation.kwargs[name]
    values = tuple(value.int_list.values) if value.HasField("int_list") else ()
    if len(values) not in (1, 2) or any(item != values[0] for item in values):
        raise ValueError(f"mapping requires symmetric {name}")
    return values[0]


# Keep both original channels and their padded domain for repeated L2 slices
def channel_metadata(matrix):
    if not matrix.HasField("mapping_channels"):
        return (), ()
    channels = matrix.mapping_channels
    logical = (channels.input_channels, channels.output_channels)
    padded = (channels.padded_input_channels, channels.padded_output_channels)
    if any(b <= 0 or a < 0 or a > b for a, b in zip(logical, padded)):
        raise ValueError("invalid logical/padded mapping channels")
    return logical, padded




# Keep separate execution paths outside ordinary matrix temporal search
def skip_reason(matrix):
    if matrix.target not in ("conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"):
        return "non-matrix operation"
    shape = tensor_shape(matrix.kwargs["input"].tensor)
    if matrix.target.startswith("conv2d"):
        if matrix.kwargs.get("groups") is not None and matrix.kwargs["groups"].int_value != 1:
            return "grouped-convolution path"
        if shape[-1] == 3:
            return "three-channel convolution path"
    elif prod(shape[:-1]) == 1:
        return "matrix-vector unit"
    return None

# Return a compiler dtype's scalar width
def dtype_bits(dtype: str):
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    return int(bit_search.groups()[0])


# Match MatrixOps.h output banking and model the external consumer's vector_timing rate
def cim_output_timing(target, operations, output):
    direct = len(operations) == 1 and not output.HasField("reshape")
    supported = {f"int{target.accum_bits}": target.accum_bits} if direct else {"int8": 8, "bfloat16": 16}
    if output.dtype not in supported:
        raise ValueError(f"unsupported CIM {'matrix' if direct else 'vector'} output dtype: {output.dtype}")
    widths = [supported[output.dtype]]
    for operation in operations[1:]:
        if operation.target in ("quantize_mx", "quantize_mx_outlier"):
            raise ValueError("CIM search does not model microscaled vector output")
        key = "scale" if operation.target == "quantize" else "other"
        if key not in operation.kwargs:
            continue
        other = operation.kwargs[key]
        if not other.HasField("tensor") or prod(tensor_shape(other.tensor)) == 1:
            continue
        tensor = other.tensor if other.tensor.HasField("memory") else operation.kwargs["input"].tensor
        if tensor.dtype not in ("int8", "bfloat16"):
            raise ValueError(f"unsupported CIM vector operand dtype: {tensor.dtype}")
        widths.append(8 if tensor.dtype == "int8" else 16)
    bits = max(widths) * target.n
    return direct, target.double_buffered_accum and bits > target.oc_port_bits, TimingOptions(
        output_cycles_per_vector=ceil_div(bits, target.oc_port_bits))


# Describe dense matrix work without inventing padding or materializing transposes
def parse_cim_operation(target, operation):
    operations = [operation.op] if operation.HasField("op") else list(operation.fused_op.op_list)
    if not operations:
        raise ValueError("CIM mapping requires a matrix operation")
    matrix = operations[0]
    if matrix.target not in ("conv2d", "linear", "matmul"):
        raise ValueError(f"unsupported CIM operation: {matrix.target}")
    if any(key in matrix.kwargs for key in ("input_code", "weight_code", "A_indptr", "A_indices", "A_data")):
        raise ValueError("CIM search requires dense operands without codebooks or sparse fusion")
    input_tensor = matrix.kwargs["input"].tensor
    weight = matrix.kwargs["other" if matrix.target == "matmul" else "weight"].tensor
    if input_tensor.dtype != "int8" or weight.dtype != "int8":
        raise ValueError("CIM search requires signed int8 input and weight tensors")
    if input_tensor.HasField("reshape") and input_tensor.reshape.target != "reshape":
        raise ValueError("input transpose or head permutation must be materialized before CIM search")
    if weight.HasField("reshape") and weight.reshape.target not in ("reshape", "transpose"):
        raise ValueError("unsupported CIM weight layout transformation")
    inputs, weights = tensor_shape(input_tensor), tensor_shape(weight)
    outputs = [operation.output] if operation.HasField("output") else list(operation.outputs.tensors)
    if len(outputs) != 1:
        raise ValueError("CIM search requires one dense output tensor")
    output = outputs[0]
    direct, banked, options = cim_output_timing(target, operations, output)
    bias = matrix.kwargs.get("bias")
    if bias is not None and (not bias.HasField("tensor") or bias.tensor.dtype != f"int{target.accum_bits}"):
        raise ValueError("CIM bias must use the configured accumulation datatype")
    if matrix.target == "conv2d":
        if len(inputs) != 4 or len(weights) != 4 or inputs[0] != 1:
            raise ValueError("CIM convolution requires batch-one NHWC input and HWIO weights")
        if matrix.kwargs.get("groups") is not None and matrix.kwargs["groups"].int_value != 1:
            raise ValueError("grouped convolution uses a separate compiler path")
        if symmetric_parameter(matrix, "dilation", 1) != 1:
            raise ValueError("CIM convolution requires unit dilation")
        fy, fx, ic, oc = weights
        workload = Workload(inputs[2], inputs[1], ic, oc, fx, fy,
                            symmetric_parameter(matrix, "stride", 1),
                            symmetric_parameter(matrix, "padding", 0), bias is not None, direct,
                            weight_transpose=weight.reshape.target == "transpose")
    else:
        if len(weights) != 2:
            raise ValueError("CIM search requires a two-dimensional weight matrix")
        ic, oc = weights
        workload = Workload(prod(inputs[:-1]), 1, ic, oc, has_bias=bias is not None,
                            output_to_memory=direct, weight_transpose=weight.reshape.target == "transpose")
    if inputs[-1] != ic:
        raise ValueError("CIM input channels do not match the logical weight shape")
    if prod(tensor_shape(output)) != workload.output_x * workload.output_y * oc:
        raise ValueError("CIM output shape does not match the dense matrix or convolution workload")
    if bias is not None and prod(tensor_shape(bias.tensor)) != oc:
        raise ValueError("CIM bias must contain one value per output channel")
    logical, padded = channel_metadata(matrix)
    workload = replace(workload, logical_channels=logical, padded_channels=padded)
    return workload, options.output_cycles_per_vector


# Convert SA shapes and output-port demand without compiler types in the model
def parse_sa_operation(target, operation, epilogue=None):
    matrix = matrix_operation(operation)
    inputs = tensor_shape(matrix.kwargs["input"].tensor)
    weight = matrix.kwargs["other" if matrix.target.startswith("matmul") else "weight"].tensor
    weights = tensor_shape(weight)
    if matrix.target.startswith("conv2d"):
        if len(weights) != 4:
            raise ValueError("SA convolution requires HWIO weights")
        fy, fx, ic, oc = weights
        workload = Workload(inputs[2], inputs[1], ic, oc, fx, fy,
                            symmetric_parameter(matrix, "stride", 1),
                            symmetric_parameter(matrix, "padding", 0))
    else:
        if len(weights) != 2:
            raise ValueError("SA mapping requires two-dimensional weights")
        ic, oc = weights
        workload = Workload(prod(inputs[:-1]), 1, ic, oc)
    output = operation.output if operation.HasField("output") else operation.outputs.tensors[-1]
    widths = [dtype_bits(output.dtype)]
    for op in operation.fused_op.op_list[1:]:
        widths.extend(dtype_bits(arg.tensor.dtype) for arg in op.kwargs.values()
                      if arg.HasField("tensor") and arg.tensor.HasField("memory"))
    output_cycles = transfer_cycles(target.n * max(widths), target.oc_port_bits)
    logical, padded = channel_metadata(matrix)
    workload = replace(workload, logical_channels=logical, padded_channels=padded,
                       input_bits=dtype_bits(matrix.kwargs["input"].tensor.dtype), weight_bits=dtype_bits(weight.dtype))
    return workload, output_cycles


def parse_operation(target, operation, epilogue=None):
    if target.backend == "cim":
        return parse_cim_operation(target, operation)
    return parse_sa_operation(target, operation, epilogue)
