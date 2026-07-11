import copy
import logging
import math
import numpy as np
import operator
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

import interstellar

# Re-exported: both predicates are generated from the Core ATen IR by
# ``tools/gen_aten_classifier.py`` and are imported from here across codegen.
from .aten_classifier import is_compute_op, is_elementwise_op  # noqa: F401
from .param_pb2 import (
    Argument,
    OpOverload,
    Tensor,
    Tiling,
    LevelTiling,
    LoopBound,
    LevelAccessCount,
)
from .passes.utils import get_arg_value

logger = logging.getLogger(__name__)


# Global variable to store the custom save function
_custom_save_function = None


def register_save_function(custom_function):
    """
    Decorator to register a custom function for saving tensors.
    The custom function should accept two arguments: tensor and filename.
    """
    global _custom_save_function
    if not callable(custom_function):
        raise ValueError("The custom function must be callable.")
    _custom_save_function = custom_function
    print(f"Custom save function '{custom_function.__name__}' registered.")
    return custom_function


_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def _write_numpy(np_array: np.ndarray, filename: str, shape: tuple):
    """Worker function: write the tensor and log when done."""
    try:
        np_array.tofile(filename)
        logger.info(f"✅ Saved tensor {shape} -> {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save tensor {shape} -> {filename}: {e}")


def _save_tensor(tensor: torch.Tensor, filename: str):
    """Asynchronously save tensor to a binary file."""
    t = tensor.detach().cpu().contiguous().to(torch.float32)
    np_array = t.numpy()
    _executor.submit(_write_numpy, np_array, filename, tuple(t.shape))


def save_tensor(tensor, filename):
    """
    Save the tensor to a file using the custom save function if defined,
    otherwise use the default _save_tensor function.
    """
    if _custom_save_function is not None:
        _custom_save_function(tensor, filename)
    else:
        _save_tensor(tensor, filename)


def _apply_transform(node, key, field, is_output=False):
    if (fused_op := node.meta.get(key)) is not None:
        fused_op.meta["tiled_shapes"] = node.meta.get("_tiled_shapes")
        fused_op.meta["scratchpad_map"] = node.meta.get("_scratchpad_map")
        field.CopyFrom(map_node(fused_op))
        if key == "reshape":
            field.kwargs["output_shape"].int_list.values.extend(node.shape)
        return fused_op.args[0] if not is_output else node
    return node


def _set_meminfo(field, segment):
    field.partition = int(segment.memory_space)
    field.address = int(segment.start)


def set_tensor_field(field, node, output_dir=None, is_output=False):
    if not isinstance(node, Node) or not hasattr(node, "value"):
        raise TypeError(f"Expected node with value attribute, got {node!r}")

    tiled_shapes = node.meta.get("_tiled_shapes")
    tile_strides = node.meta.get("_tile_strides")
    scratchpad_map = node.meta.get("_scratchpad_map")

    # Apply transformations
    if node.op != "call_module" or is_output:
        node = _apply_transform(node, "reshape", field.reshape, is_output)
    node = _apply_transform(node, "dequantize", field.dequant)

    if (source_node := node.meta.get("source_node")) is not None:
        node = source_node

    if tiled_shapes is not None and node in tiled_shapes:
        field.tiled_shape.extend(tiled_shapes[node])

    if tile_strides is not None and node in tile_strides:
        field.tile_strides.extend(tile_strides[node])

    if scratchpad_map is not None and node in scratchpad_map:
        _set_meminfo(field.scratchpad, scratchpad_map[node])

    # Bufferized path: each tile node carries its own scratchpad ``Segment``
    # directly (symmetric with ``memory`` below), rather than a per-consumer
    # ``scratchpad_map`` dict.
    elif (scratchpad := node.meta.get("scratchpad")) is not None:
        _set_meminfo(field.scratchpad, scratchpad)

    # Tensor properties
    field.node = node.name
    field.shape.extend(node.shape or [1])

    if (dtype := node.meta.get("dtype")) is not None:
        field.dtype = dtype
    else:
        field.dtype = str(node.value.dtype).split(".")[1]

    if (memory := node.meta.get("memory")) is not None:
        _set_meminfo(field.memory, memory)

    if output_dir is not None:
        save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))


def set_output_field(param, node, output_dir):
    if isinstance(node.value, torch.Tensor):
        node.meta["_tiled_shapes"] = node.meta.get("tiled_shapes")
        node.meta["_scratchpad_map"] = node.meta.get("scratchpad_map")

        set_tensor_field(param.output, node, output_dir, True)

        node.meta.pop("_tiled_shapes", None)
        node.meta.pop("_scratchpad_map", None)
    elif (
        isinstance(node.value, (tuple, list))
        and node.value
        and all(isinstance(v, int) for v in node.value)
    ):
        # An integer index vector (``increment_indices`` / ``delinearize_index``): a
        # named handle that the per-dimension component getitems index into — a tile
        # block index — not a tensor list.
        param.output.node = node.name
    elif isinstance(node.value, (tuple, list)):
        if (memory := node.meta.get("memory")) is not None:
            memory_copy = copy.copy(memory)
            output_sizes = node.meta["output_sizes"]

        scratchpad_map = node.meta.get("scratchpad_map")
        if scratchpad_map is not None and node in scratchpad_map:
            scratchpad_copy = copy.copy(scratchpad_map[node])
            tiled_output_sizes = node.meta["tiled_output_sizes"]
        else:
            scratchpad_map = None

        # The bufferized path may leave a stale ``tiled_shapes`` (keyed by the
        # pre-bufferization nodes) on a copied op; only use it when it keys ``node``.
        tiled_shape = node.meta.get("tiled_shapes")
        if tiled_shape is not None and node in tiled_shape:
            shapes = tiled_shape[node]
        else:
            tiled_shape = None

        if (dtypes := node.meta.get("dtype")) is None:
            dtypes = [None] * len(node.value)

        for i, t in enumerate(node.value):
            tensor = Tensor(
                node=f"{node.name}_{i}",
                shape=list(t.shape),
                dtype=dtypes[i] or str(t.dtype).split(".")[1],
            )

            if memory is not None:
                _set_meminfo(tensor.memory, memory_copy)
                memory_copy.start += output_sizes[i]

            if scratchpad_map is not None:
                _set_meminfo(tensor.scratchpad, scratchpad_copy)
                scratchpad_copy.start += tiled_output_sizes[i]

            if tiled_shape is not None:
                tensor.tiled_shape.extend(shapes[i])

            if output_dir is not None:
                save_tensor(t, os.path.join(output_dir, f"{node.name}_{i}.bin"))

            param.outputs.tensors.append(tensor)
    elif isinstance(node.value, (int, bool)):
        # A scalar-valued op is an integer tile-index computation (a pipelined
        # prefetch's ``j + 1``); its result is a named scalar a load/store references,
        # not a tensor — record the name so the reference resolves.
        param.output.node = node.name
    elif node.value is None and node.target in (
        torch.ops.voyager.insert.default,
        torch.ops.voyager.async_copy.default,
    ):
        # A side-effecting tile write (``insert`` / ``async_copy``; returns ``None`` — its
        # dest buffer is a closed-over additional input mutated in place).  Its result is
        # the tile it writes, described from the dest buffer (``args[1]``), which carries
        # the planned address; the source tile and block index travel in the op's args.
        set_tensor_field(param.output, node.args[1], output_dir, is_output=True)
    elif node.value is None and node.target is (
        torch.ops.voyager.async_wait.default
    ):
        # ``async_wait`` synchronizes a semaphore and produces no tensor — leave
        # the Operation's return field unset.
        pass
    else:
        logger.warning(f"Unsupported output type: {type(node.value)}")


def build_tiling_proto(node):
    """Convert node.meta['interstellar_tiling'] to a Tiling proto."""
    mapping, access_list = node.meta["interstellar_tiling"]
    architecture = node.meta["interstellar_architecture"]
    tiling = Tiling(name=node.name)

    for level in range(1, architecture.num_levels):  # skip level 0 (PE)
        lt = LevelTiling()
        loop_index = 0
        while loop_index < interstellar.le.NUM:
            matched = False
            for loop in range(interstellar.le.NUM):
                if mapping.loop_orders[loop][level] == loop_index:
                    lt.loop_bounds.append(
                        LoopBound(
                            loop=loop,
                            bound=mapping.loop_blockings[loop][level],
                        )
                    )
                    loop_index += 1
                    matched = True
                    break
            if not matched:
                break
        tiling.level_tilings.append(lt)
        tiling.level_access_counts.append(
            LevelAccessCount(
                input_access_count=int(access_list[level][0]),
                output_access_count=int(access_list[level][1]),
                weight_access_count=int(access_list[level][2]),
            )
        )

    return tiling


def convert_arg(value, output_dir=None) -> Argument:
    """
    Converts an argument (which could be a Tensor, list, int, float, etc.)
    into an Argument protobuf.
    """
    arg = Argument()

    if isinstance(value, torch.fx.Node):
        val = getattr(value, "value", None)

        if isinstance(val, torch.Tensor):
            set_tensor_field(arg.tensor, value, output_dir)
        elif isinstance(val, (tuple, list)):
            arg.tensor_list.tensors.extend(
                [Tensor(node=f"{value.name}_{i}") for i in range(len(val))]
            )
        else:
            # Scalar-valued node: a tile block index — the loop induction variable, or
            # a computed index such as a pipelined prefetch's ``j + 1`` (emitted as its
            # own Operation by ``_emit_node``).  Reference it by name so the address is
            # explicit (the loop's ``node`` names the induction variable), rather than
            # an empty positional placeholder that can only mean the bare counter.
            arg.tensor.node = value.name
    elif isinstance(value, bool):
        arg.bool_value = value
    elif isinstance(value, int):
        arg.int_value = value
    elif isinstance(value, float):
        arg.float_value = value
    elif isinstance(value, str):
        arg.str_value = value
    elif isinstance(
        value, (torch.dtype, torch.layout, torch.device, torch.memory_format)
    ):
        arg.str_value = str(value).split(".")[-1]
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, torch.fx.Node) or x is None for x in value):
            arg.tensor_list.tensors.extend(
                [
                    (
                        convert_arg(x).tensor
                        if x is not None
                        else Tensor(is_none=True)
                    )
                    for x in value
                ]
            )
        elif all(isinstance(x, bool) for x in value):
            arg.bool_list.values.extend(value)
        elif all(isinstance(x, int) for x in value):
            arg.int_list.values.extend(value)
        elif all(isinstance(x, (int, float, bool)) for x in value):
            arg.scalar_list.values.extend(value)
        else:
            raise TypeError(f"Unsupported list value: {value}")
    else:
        raise TypeError(f"Unsupported arg type: {type(value)}")

    return arg


def map_node(node: torch.fx.Node, output_dir=None) -> OpOverload:
    """
    Converts a torch.fx.Node into an OpOverload protobuf message.
    """
    if hasattr(node.target, "_schema"):
        target = node.target._schema.name.split("::")[1]
    else:
        target = str(node.target)

    op_overload = OpOverload(
        name=node.name,
        op=node.op,
        target=target,
    )

    if (
        is_nop(node)
        or is_addressing_op(node)
        or node.target == operator.getitem
    ):
        op_overload.op = "nop"

    if node.target == torch.ops.aten.pad.default:
        op_overload.op = "cpu"

    new_args_and_kwargs = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )

    if new_args_and_kwargs is not None:
        args, kwargs = new_args_and_kwargs.args, new_args_and_kwargs.kwargs
    else:
        args, kwargs = node.args, node.kwargs

    # Pass L2 tiling metadata to input nodes
    for n in node.all_input_nodes:
        n.meta["_tiled_shapes"] = node.meta.get("tiled_shapes")
        n.meta["_tile_strides"] = node.meta.get("tile_strides")
        n.meta["_scratchpad_map"] = node.meta.get("scratchpad_map")

    # Convert positional arguments
    for arg in args:
        op_overload.args.append(convert_arg(arg))

    # Convert keyword arguments
    for key, value in kwargs.items():
        if not "qmap" in key and value is not None:
            op_overload.kwargs[key].CopyFrom(convert_arg(value, output_dir))

    for n in node.all_input_nodes:
        n.meta.pop("_tiled_shapes", None)
        n.meta.pop("_tile_strides", None)
        n.meta.pop("_scratchpad_map", None)

    if "l2_tiling" in node.meta:
        op_overload.kwargs["l2_tiling"].int_list.values.extend(
            node.meta["l2_tiling"]
        )

    return op_overload


aten = torch.ops.aten

# Sentinel ``aten.slice`` uses for ``end`` to mean "to the end of the dim".
INT64_MAX = torch.iinfo(torch.int64).max


# Quantization lookup-table args (qmaps and codebooks) across the
# quantized_ops quantize/dequantize/quantize_mx family. They are indexed by
# value, not by iteration position, so they are passed whole (never tiled /
# position-mapped).
QUANT_TABLE_PARAMS = {
    "qmap",
    "scale_qmap",
    "input_qmap",
    "output_qmap",
    "code",
    "input_code",
    "weight_code",
    "output_code",
}


def ancestors(node: Node) -> set:
    """Every transitive input node of ``node`` (its operand prelude): for a
    matmul ``Q @ Kᵀ`` this is the operand placeholders *and* the ``transpose``
    that builds ``Kᵀ``, so a pre-anchor relayout is not mistaken for a fused
    post-op."""
    if node is None:
        return set()
    result = set()
    stack = list(node.all_input_nodes)
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(current.all_input_nodes)
    return result


def quant_table_arg_nodes(node: Node) -> set:
    """Tensor args of ``node`` that are quantization lookup tables (qmaps /
    codebooks), identified by schema arg name so positions need not be
    hardcoded."""
    result = set()
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return result
    for i, arg in enumerate(schema.arguments):
        if arg.name not in QUANT_TABLE_PARAMS:
            continue
        val = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
        if isinstance(val, Node):
            result.add(val)
    return result


def is_gemm_op(node: Node) -> bool:
    return is_conv2d(node) or is_linear(node) or is_matmul(node)


def is_conv2d(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx.default,
    ]


def is_depthwise_conv(node: Node) -> bool:
    return is_conv2d(node) and get_arg_value(node, 6, "groups", 1) != 1


def is_linear(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.linear.default,
        torch.ops.quantized_ops.linear.default,
        torch.ops.quantized_ops.linear_mx.default,
    ]


def is_matmul(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.matmul.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def is_bmm(node: Node) -> bool:
    if is_matmul(node):
        input_shape = node.args[0].shape
        other_shape = node.args[1].shape
        return len(input_shape) > 2 or len(other_shape) > 2
    return False


def is_fully_connected(node: Node) -> bool:
    if is_linear(node):
        input_shape = node.args[0].shape
        return all(s == 1 for s in input_shape[:-1])

    if is_matmul(node):
        input_shape = node.args[0].shape
        other_shape = node.args[1].shape

        if is_bmm(node):
            return input_shape[-2] == 1
        else:
            return all(s == 1 for s in input_shape[:-1])

    return False


def is_pooling(node: Node) -> bool:
    return node.target in [
        # Core Aten IR
        aten._adaptive_avg_pool2d,
        aten._adaptive_avg_pool3d,
        aten.adaptive_avg_pool1d,
        aten.avg_pool1d,
        aten.avg_pool2d,
        aten.avg_pool3d,
        aten.max_pool2d_with_indices,
        aten.max_pool3d_with_indices,
        # export_for_training IR
        aten.adaptive_avg_pool2d.default,
        aten.avg_pool2d.default,
        aten.max_pool2d.default,
        # NHWC variants (after the data-layout transform)
        torch.ops.quantized_ops.max_pool2d.default,
        torch.ops.quantized_ops.avg_pool2d.default,
        torch.ops.quantized_ops.adaptive_avg_pool2d.default,
        torch.ops.quantized_ops._adaptive_avg_pool2d.default,
    ]


def is_reshape_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ]


def is_prunable_op(node: Node) -> bool:
    """Operations that can be safely deleted from fx.Graph."""
    if node.target == torch.ops.aten.alias.default:
        return True

    # A slice from 0 to the end of the input tensor
    if node.target == torch.ops.aten.slice.Tensor:
        dim = get_arg_value(node, 1, "dim", 0)
        start = get_arg_value(node, 2, "start")
        end = get_arg_value(node, 3, "end")
        step = get_arg_value(node, 4, "step", 1)
        if start is not None and start != 0 or step != 1:
            return False
        if end is not None and hasattr(node.args[0], "shape"):
            return end >= node.args[0].shape[dim]
        return (start is None and end is None) or end == INT64_MAX

    if node.target == torch.ops.aten.expand.default:
        return all(x == 1 or x == -1 for x in node.args[1])

    # Dropout with zero probability is the identity.
    if node.target == torch.ops.aten.dropout.default:
        return get_arg_value(node, 1, "p") == 0.0

    # A same-dtype ``to.dtype`` is a pure pass-through.
    if node.target == torch.ops.aten.to.dtype:
        dtype = get_arg_value(node, 1, "dtype")
        inp = node.args[0]
        val = getattr(inp, "value", inp.meta["val"])
        return isinstance(val, torch.Tensor) and dtype == val.dtype

    return False


def is_nop(node: Node) -> bool:
    """
    The following operations do not require any computation nor handling
    on the memory placement side. Generate a NOP instruction for these ops
    to keep the compute graph intact.
    """
    if is_prunable_op(node):
        return True

    # A select operation that selects the entire tensor
    if node.target == torch.ops.aten.select.int:
        shape = getattr(node.args[0], "shape", None)
        return shape is not None and shape[node.args[1]] == 1

    return node.target in [
        torch.ops.aten.as_strided.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    ]


def is_shape_changing_nop(node: Node) -> bool:
    """A ``nop`` (no compute) whose output shape differs from its input shape
    — a ``view`` / ``reshape`` / ``squeeze`` / ``unsqueeze`` / size-1 ``select``
    that regroups or drops dims.  Such a node sitting *between* two fused
    compute stages breaks the single-iteration-space assumption and is relocated
    to the fused module's boundary by the iteration-space normalizer (see
    ``normalize.py``).  A shape-*preserving* nop (same in/out shape) can stay
    inside the fused chain.
    """
    if not is_nop(node):
        return False
    inp = node.args[0]
    return (
        hasattr(node, "shape")
        and hasattr(inp, "shape")
        and tuple(node.shape) != tuple(inp.shape)
    )


def is_addressing_op(node: Node) -> bool:
    """
    The following operations are handled by the memory placement and
    thus require no additional handling:
    """
    if node.target == torch.ops.aten.select.int:
        return all(d == 1 for d in node.args[0].value.shape[: node.args[1]])

    if node.target in [
        torch.ops.aten.stack.default,
        torch.ops.aten.cat.default,
    ]:
        return len(node.args) == 1 or node.args[1] == 0

    return False


def is_memory_op(node: Node) -> bool:
    """
    The following operators requires explicit data movement and thus require
    additional handling. Note that some operators are storage-preserving
    in PyTorch. However, the current compiler hasn't implemented handling for
    storage-preserving operators that change the number of elements in the
    tensor. For example, ``torch.ops.aten.expand.default`` may increase the
    number of elements in the tensor. We will add support in the future.
    """
    if node.op != "call_function" or is_nop(node):
        return False

    return node.target in [
        torch.ops.aten.clone.default,
        torch.ops.aten.copy_.default,
        torch.ops.aten.embedding.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.index_copy_.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.to.dtype,
    ]
