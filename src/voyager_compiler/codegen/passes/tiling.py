import copy
import logging
import math
import operator
import re
from typing import List, Tuple, Generator, Optional, Union

import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.node import map_arg
from torch.fx.operator_schemas import normalize_function

from .utils import get_arg_value, _pair
from ..mapping import (
    get_node_bytes,
    get_node_to_key_map,
    normalize_shape,
    replace_node_with_graph_module,
    _nodes_sequential,
    _create_and_insert_subgraph,
)
from ..mapping_utils import (
    is_conv2d,
    is_bmm,
    is_depthwise_conv,
    is_elementwise_op,
    is_linear,
    is_matmul,
    is_pooling,
    is_prunable_op,
)
from ..banking import (
    get_banking_strategies_for_op,
    require_allocation,
    _get_scope,
)
from ...pt2e_utils import WrapperModule, fetch_attr, propagate_shape
from ...quantize_pt2e import create_getattr_from_value, export_model

logger = logging.getLogger(__name__)

__all__ = [
    "run_matrix_op_l2_tiling",
    "run_pool_op_l2_tiling",
    "run_vector_op_l2_tiling",
    "run_vector_op_node_l2_tiling",
]

DEFAULT_CACHE_SIZE = 8 * 1024 * 1024  # 8 MiB


def _get_tiling_context(node_to_fuse: torch.fx.Node) -> List[Tuple[int, int, int]]:
    """
    Backtracks from node_to_fuse to find slice/pad ops and calculates
    the equivalent output slice parameters.
    """
    conv_node = node_to_fuse
    if not is_conv2d(conv_node) and len(conv_node.args) > 1:
         # Fallback for patterns like relu(conv) or add(conv, bias)
         # Assuming conv is the second arg based on original logic, but safer to check
         potential_conv = conv_node.args[1]
         if isinstance(potential_conv, torch.fx.Node) and is_conv2d(potential_conv):
             conv_node = potential_conv

    if "tiled_shapes" not in conv_node.meta:
        return []

    input_shape = conv_node.meta["tiled_shapes"]["input"]
    output_shape = conv_node.meta["tiled_shapes"]["output"]

    slice_args = []

    # Traverse backwards to find slicing details
    current_node = conv_node.args[0]
    while isinstance(current_node, torch.fx.Node) and current_node.target in [
        torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default
    ]:
        if current_node.target == torch.ops.aten.slice.Tensor:
            dim, start, end = current_node.args[1:]
            # Ensure we only calculate slices for spatial dims (H=2, W=3)
            if dim in (2, 3):
                # Calculate tile index relative to the *tile shape* or *full shape*
                tile_idx = round(float(start) / input_shape[dim])
                tile_start = tile_idx * output_shape[dim]
                tile_end = tile_start + output_shape[dim]
                slice_args.insert(0, (dim, tile_start, tile_end))

        current_node = current_node.args[0]

    return slice_args


def _clone_parameter(
    model: torch.fx.GraphModule, arg: torch.fx.Node, anchor: torch.fx.Node
) -> torch.fx.Node:
    """Creates a new get_attr node for a parameter."""
    param = fetch_attr(model, arg.target)
    prefix = arg.name

    if match := re.fullmatch(r'(code|qmap)(_\d+)?', arg.name):
        prefix = match.group(1)

    with model.graph.inserting_before(anchor):
        new_attr = create_getattr_from_value(model, model.graph, prefix, param)

    propagate_shape(new_attr, model)
    new_attr.meta["dtype"] = arg.meta.get("dtype")
    return new_attr


def _slice_side_input(
    model: torch.fx.GraphModule,
    arg: torch.fx.Node,
    slice_args: List[Tuple[int, int, int]],
    anchor: torch.fx.Node,
) -> torch.fx.Node:
    """Slices a side input (e.g. residual connection) to match the current tile."""
    current_slice_node = arg

    # Optimization: If the side input is not spatial (e.g. 1D bias), do not slice.
    if len(arg.shape) < 4:
        return arg

    for dim, start, end in slice_args:
        with model.graph.inserting_before(anchor):
            current_slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor,
                (current_slice_node, dim, start, end),
            )
        propagate_shape(current_slice_node, model)
        current_slice_node.meta["dtype"] = arg.meta.get("dtype")

    return current_slice_node


def create_new_chain(model, node_to_fuse, cat_node, fusable):
    """
    Replicates the 'fusable' chain of nodes (originally after 'cat_node')
    to be placed after 'node_to_fuse', applying appropriate spatial tiling.
    """
    slice_params = _get_tiling_context(node_to_fuse)

    anchor = node_to_fuse.next
    value_remap = {cat_node: node_to_fuse}

    for n in fusable:
        for arg in n.all_input_nodes:
            if arg in value_remap:
                continue

            if arg.op == "get_attr":
                value_remap[arg] = _clone_parameter(model, arg, anchor)
            else:
                value_remap[arg] = _slice_side_input(
                    model, arg, slice_params, node_to_fuse
                )

        with model.graph.inserting_before(anchor):
            new_node = model.graph.node_copy(
                n, lambda n: value_remap.get(n, n)
            )
        propagate_shape(new_node, model)
        new_node.meta["dtype"] = n.meta.get("dtype")

        value_remap[n] = new_node

    # Reconnect users of the original node to the end of the new chain
    last_node = value_remap[fusable[-1]]
    first_node = value_remap[fusable[0]]

    for user in list(node_to_fuse.users):
        if user != first_node:
            user.replace_input_with(node_to_fuse, last_node)


def move_fusable_ops_after_conv2d(model, node):
    order = {n: i for i, n in enumerate(model.graph.nodes)}
    fusable_ops = []
    next_node = next(iter(node.users))
    while is_elementwise_op(next_node):
        chain = [node] + fusable_ops + [next_node]
        if _nodes_sequential(chain, order):
            fusable_ops.append(next_node)
        else:
            break
        # Stop fusing if last node is a quantize op
        if (
            len(next_node.users) != 1
            or next_node.target == torch.ops.quantized_ops.quantize.default
        ):
            break
        next_node = next(iter(next_node.users))

    if not fusable_ops:
        return

    # Find all the conv2d nodes to fuse with
    conv2d_nodes = []
    cat_and_slice_nodes = []
    stack = node.all_input_nodes[:]
    while stack:
        curr = stack.pop()
        if curr.target in [
            torch.ops.aten.cat.default, torch.ops.aten.slice.Tensor,
        ]:
            cat_and_slice_nodes.append(curr)
            stack.extend(curr.all_input_nodes)
        else:
            conv2d_nodes.append(curr)

    for conv_node in conv2d_nodes:
        create_new_chain(model, conv_node, node, fusable_ops)

    for n in cat_and_slice_nodes:
        n.meta["dtype"] = fusable_ops[-1].meta.get("dtype")

    fusable_ops[-1].replace_all_uses_with(node)
    for n in reversed(fusable_ops):
        model.graph.erase_node(n)


def _make_conv2d_tiled_module(
    X, C, tile_x, tile_c, pad_value, is_dwc, is_mx_conv, configs
):
    """
    Factory function to create a Conv2dTiled module class with captured parameters.
    """
    class Conv2dTiled(torch.nn.Module):
        def __init__(self, stride=1, padding=0, dilation=1, groups=1, block_size=16):
            super().__init__()
            self.stride = _pair(stride)
            self.padding = _pair(padding)
            self.dilation = _pair(dilation)
            self.groups = groups
            self.block_size = block_size

        def get_input_tile(self, input, cfg):
            c0, c1 = cfg["c_tile"]
            y0, y1 = cfg["y_tile"]
            x0, x1 = cfg["x_tile"]

            if is_dwc:
                tile = input[:, :, y0:y1, x0:x1]
            else:
                tile = input[:, c0:c1, y0:y1, x0:x1]

            padding = cfg["input_padding"]
            if any(p > 0 for p in padding):
                tile = F.pad(tile, padding, "constant", pad_value)

            return tile

        def get_weight_tile(self, weight, cfg):
            c0, c1 = cfg["c_tile"]
            return weight[:, c0:c1, :, :]

        def get_scales(self, input_scale, weight_scale, cfg):
            if not is_mx_conv:
                return None, None

            c0, c1 = cfg["c_tile"]
            y0, y1 = cfg["y_tile"]
            x0, x1 = cfg["x_tile"]

            bs = self.block_size

            if is_dwc:
                in_s = input_scale[:, :, y0:y1, x0:x1]
                wt_s = weight_scale[:, 0:1, :, :]
            else:
                s0, s1 = c0 // bs, c1 // bs
                in_s = input_scale[:, s0:s1, y0:y1, x0:x1]
                wt_s = weight_scale[:, s0:s1, :, :]

            padding = cfg["input_padding"]
            if any(p > 0 for p in padding):
                tile = F.pad(tile, padding, "constant", 1.0)

            return in_s, wt_s

        def run_conv(self, input_tile, weight_tile, bias, cfg, scales, codes):
            args = (
                input_tile,
                weight_tile,
                bias,
                self.stride,
                cfg["conv_padding"],
                self.dilation,
                self.groups,
            )

            if not is_mx_conv:
                return torch.ops.aten.conv2d.default(*args)

            return torch.ops.quantized_ops.conv2d_mx(
                *args,
                input_scale=scales[0],
                weight_scale=scales[1],
                block_size=self.block_size,
                input_code=codes[0],
                weight_code=codes[1],
            )

        def trim_output(self, out, cfg):
            if not cfg["keep_dims_and_padding"]:
                return out

            oy, ox = cfg["output_offset"]
            oh, ow = cfg["output_sizes"]
            return out[:, :, oy:oy+oh, ox:ox+ow]

        def forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            input_scale: Optional[torch.Tensor] = None,
            weight_scale: Optional[torch.Tensor] = None,
            input_code: Optional[torch.Tensor] = None,
            weight_code: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            row_tiles = []
            col_tiles = []

            # Track when to break rows
            tile_c_count = C // tile_c
            tile_x_count = X // tile_x

            for tile_idx, cfg in enumerate(configs):
                input_tile = self.get_input_tile(input, cfg)
                weight_tile = self.get_weight_tile(weight, cfg)
                scale_tiles = self.get_scales(input_scale, weight_scale, cfg)
                out = self.run_conv(
                    input_tile,
                    weight_tile,
                    bias if cfg["c_tile"][1] == C else None,
                    cfg,
                    scale_tiles,
                    (input_code, weight_code),
                )
                out = self.trim_output(out, cfg)

                # accumulate partial channels
                acc = out if (tile_idx % tile_c_count == 0) else (acc + out)

                # when finishing tile_c accumulations
                if (tile_idx + 1) % tile_c_count == 0:
                    col_tiles.append(acc)

                # when finishing tile_x tiles (a full row)
                if len(col_tiles) == tile_x_count:
                    row_tiles.append(torch.cat(col_tiles, dim=-1))
                    col_tiles = []

            return torch.cat(row_tiles, dim=2)

    return Conv2dTiled


def _decompose_conv2d_node(model, node, tile_sizes, tiled_shapes, configs):
    stride = get_arg_value(node, 3, "stride", 1)
    padding = get_arg_value(node, 4, "padding", 0)
    dilation = get_arg_value(node, 5, "dilation", 1)
    groups = get_arg_value(node, 6, "groups", 1)
    bs = node.kwargs.get("block_size", 1)

    N, K, Y, X = node.shape
    _, C, kH, kW = node.args[1].shape

    tile_y, tile_x, tile_c, tile_k = tile_sizes

    # Compute pad_value once
    pad_value = 0
    input_code = node.kwargs.get("input_code")
    if input_code is not None:
        code = model.get_buffer(input_code.target)
        pad_value = (code == 0).nonzero()[0].item()

    is_dwc = is_depthwise_conv(node)
    is_mx_conv = node.target == torch.ops.quantized_ops.conv2d_mx.default

    # Create the tiled module using factory function
    Conv2dTiled = _make_conv2d_tiled_module(
        X, C, tile_x, tile_c, pad_value, is_dwc, is_mx_conv, configs
    )

    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

    mod = Conv2dTiled(stride, padding, dilation, groups, bs)
    kwargs = {k: v for k, v in node.kwargs.items() if v is not None}
    kwargs.pop("block_size", None)
    gm = export_model(mod, load_arg(node.args[:3]), load_arg(kwargs))

    for n in list(gm.graph.nodes):
        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            gm.graph.erase_node(n)
    gm.graph.lint()

    value_remap = {}
    output = replace_node_with_graph_module(model, node, gm, value_remap)

    # Update metadata on new nodes in the graph
    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.slice.Tensor and n.args[0].op == "get_attr":
            c_start, c_end = n.args[2], n.args[3]
            with model.graph.inserting_before(n):
                sliced_param = _slice_tensor(n.args[0], 1, c_start, c_end, model)
            n.replace_all_uses_with(sliced_param)
            model.graph.erase_node(n)
            continue

        if n.target in [
            torch.ops.aten.slice.Tensor, torch.ops.aten.pad.default,
        ]:
            n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target == node.target:
            n.meta.update({
                "tiled_shapes": tiled_shapes.pop(0),
                "l2_tiling": (1, K // tile_k, 1, 1),
                "dtype": node.meta.get("dtype"),
            })

    if output[0].target == torch.ops.aten.cat.default:
        move_fusable_ops_after_conv2d(model, output[0])


def _pad_input(model, node, arg, padding, pad_value):
    with model.graph.inserting_before(node):
        new_arg = model.graph.call_function(
            torch.ops.aten.pad.default,
            (arg, padding, "constant", pad_value),
        )
    propagate_shape(new_arg, model)
    new_arg.meta["dtype"] = arg.meta.get("dtype")
    node.replace_input_with(arg, new_arg)
    return new_arg


def split_conv2d_node(model, node, tile_sizes):
    """
    Replace a conv2d node with a tiled conv2d subgraph.

    Args:
        model: GraphModule
        node: node (must be aten.conv2d or quantized conv2d)
        tile_sizes: (Y, X, C, K)
    """
    stride  = _pair(get_arg_value(node, 3, "stride", 1))
    padding = _pair(get_arg_value(node, 4, "padding", 0))
    dilation = _pair(get_arg_value(node, 5, "dilation", 1))
    bs = node.kwargs.get("block_size", 1)

    is_conv1 = node.args[0].shape[1] == 3
    is_dwc = is_depthwise_conv(node)

    N, K, Y, X = node.shape
    _, C, kH, kW = node.args[1].shape
    _, _, IX, IY = node.args[0].shape
    tile_y, tile_x, tile_c, tile_k = tile_sizes

    if tile_y == Y and tile_x == X and tile_c == C and tile_k == K:
        return  # No tiling needed

    pad_value = 0
    if (input_code := node.kwargs.get("input_code")) is not None:
        code = model.get_buffer(input_code.target)
        pad_value = (code == 0).nonzero()[0].item()

    if (tile_x != X or tile_y != Y) and any(p for p in padding) and not is_conv1:
        pad_hw = (padding[1], padding[1], padding[0], padding[0])
        _pad_input(model, node, node.args[0], pad_hw, pad_value)

        if input_scale := node.kwargs.get("input_scale"):
            _pad_input(model, node, input_scale, pad_hw, 1.0)

        padding = _pair(0)
        node.update_arg(4, padding)
        propagate_shape(node, model)
        _, _, IY, IX = node.args[0].shape

    def _compute_spatial_tile_region(y, x, oh, ow):
        if tile_x == X and tile_y == Y:
            return {
                "y_tile": (0, IY),
                "x_tile": (0, IX),
                "input_padding": (0, 0, 0, 0),
                "output_offset": (0, 0),
            }

        y_in_start = y * stride[0] - padding[0]
        y_in_end = y_in_start + (oh - 1) * stride[0] + (kH - 1) * dilation[0] + 1
        x_in_start = x * stride[1] - padding[1]
        x_in_end = x_in_start + (ow - 1) * stride[1] + (kW - 1) * dilation[1] + 1

        # Grab more input pixels so that after applying padding of three, the
        # new convolution still aligns with the original one, just with some
        # garbage outputs on the right and bottom side
        if is_conv1:
            y_in_start = y_in_start - (padding[0] % stride[0])
            x_in_start = x_in_start - (padding[1] % stride[1])

        # Adjust receptive field to multiple of stride for depthwise conv
        if is_dwc:
            if rem := (y_in_end - y_in_start) % stride[0]:
                y_in_end += stride[0] - rem
            if rem := (x_in_end - x_in_start) % stride[1]:
                x_in_end += stride[1] - rem

        y_in_start_clamped = max(y_in_start, 0)
        x_in_start_clamped = max(x_in_start, 0)

        if not is_conv1:
            x_in_end_clamped = min(x_in_end, IX)
            y_in_end_clamped = min(y_in_end, IY)

            assert (
                y_in_start_clamped == y_in_start
                or x_in_start_clamped == x_in_start
                or y_in_end_clamped == y_in_end
                or x_in_end_clamped == x_in_end
            ), f"{node}: Unexpected input tile size"

            # downsample layers input size must be multiples of stride
            if kH == 1 and kW == 1:
                y_in_end_clamped = y_in_start_clamped + oh * stride[0]
                x_in_end_clamped = x_in_start_clamped + ow * stride[1]

            pad_top, pad_left, pad_bottom, pad_right = 0, 0, 0, 0
            x_offset, y_offset = 0, 0
        else:
            # conv1 hardware replication constraints
            y_in_end += (16 - ((y_in_end - y_in_start_clamped) % 16))
            x_in_end += (16 - ((x_in_end - x_in_start_clamped) % 16))
            y_in_end_clamped = min(y_in_end, IY)
            x_in_end_clamped = min(x_in_end, IX)

            pad_top = 0
            pad_left = 0
            pad_bottom = y_in_end - y_in_end_clamped
            pad_right = x_in_end - x_in_end_clamped

            y_offset = (y * stride[0] - y_in_start_clamped) // stride[0]
            x_offset = (x * stride[1] - x_in_start_clamped) // stride[1]

        return {
            "y_tile": (y_in_start_clamped, y_in_end_clamped),
            "x_tile": (x_in_start_clamped, x_in_end_clamped),
            "input_padding": (pad_left, pad_right, pad_top, pad_bottom),
            "output_offset": (y_offset, x_offset),
        }

    def _compute_per_tile_shapes(cfg):
        (y0, y1) = cfg["y_tile"]
        (x0, x1) = cfg["x_tile"]
        (pad_left, pad_right, pad_top, pad_bottom) = cfg["input_padding"]
        conv_padding = cfg["conv_padding"]
        (oh, ow) = cfg["output_sizes"]
        (c_start, c_end) = cfg["c_tile"]

        h_in = (y1 - y0) + pad_top + pad_bottom
        w_in = (x1 - x0) + pad_left + pad_right
        h_out = (h_in + 2 * conv_padding[0] - kH) // stride[0] + 1
        w_out = (w_in + 2 * conv_padding[1] - kW) // stride[1] + 1

        assert h_out >= oh and w_out >= ow, (
            f"Output sizes mismatch: ({h_out}, {w_out}) vs ({oh}, {ow})"
        )

        c_tile = c_end - c_start
        tiled_shape = {}

        if is_dwc:
            tiled_shape["input"] = (N, tile_k, h_in, w_in)
            tiled_shape["input_scale"] = (N, tile_k // bs, h_in, w_in)
            tiled_shape["weight_scale"] = (tile_k, 1, kH, kW)
        else:
            tiled_shape["input"] = (N, c_tile, h_in, w_in)
            tiled_shape["input_scale"] = (N, c_tile // bs, h_in, w_in)
            tiled_shape["weight_scale"] = (tile_k, c_tile // bs, kH, kW)

        tiled_shape["weight"] = (tile_k, c_tile, kH, kW)
        tiled_shape["bias"] = (tile_k,)
        tiled_shape["output"] = (N, tile_k, h_out, w_out)
        return tiled_shape

    tiled_shapes = []
    tile_configs = []
    for y in range(0, Y, tile_y):
        for x in range(0, X, tile_x):
            oh = min(tile_y, Y - y)
            ow = min(tile_x, X - x)

            spatial = _compute_spatial_tile_region(y, x, oh, ow)

            for c_start in range(0, C, tile_c):
                c_end = min(c_start + tile_c, C)
                cfg = {
                    **spatial,
                    "c_tile": (c_start, c_end),
                    "conv_padding": padding,
                    "output_sizes": (oh, ow),
                    "keep_dims_and_padding": is_conv1,
                }

                tile_configs.append(cfg)
                tiled_shapes.append(_compute_per_tile_shapes(cfg))

    if tile_c != C or is_conv1:
        _decompose_conv2d_node(
            model, node, tile_sizes, tiled_shapes, tile_configs
        )
    else:
        h_in = tile_y * stride[0]
        w_in = tile_x * stride[1]
        node.meta["tiled_shapes"] = tiled_shapes[0]
        node.meta["l2_tiling"] = (1, K // tile_k, Y // tile_y, X // tile_x)
        node.meta["tile_strides"] = {
            "input": (1, tile_c, h_in, w_in),
            "input_scale": (1, tile_c // bs, h_in, w_in),
        }


def _prime_factors(n: int):
    f, p = [], 2
    while p * p <= n:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 1 if p == 2 else 2  # 2,3,5,7,...
    if n > 1:
        f.append(n)
    return f


def construct_tiled_shape(full_shape, tiled_dim: int, dims):
    """
    Reconstruct full-rank tiled shape.

    Args:
      full_shape: tuple/list[int] original shape (len N)
      tiled_dim: int, flattened size of the compressed (tiled) dims
      dims: iterable[int], indices of dims that were flattened into tiled_dim

    Returns:
      Tuple[int] of length N
    """
    full_shape = tuple(full_shape)
    N = len(full_shape)
    if N == 0:
        raise ValueError("full_shape must have at least one dimension.")

    # Normalize & validate compressed dims
    comp = sorted(set(int(i) for i in dims))
    if not comp:
        raise ValueError("dims cannot be empty.")
    if any(i < 0 or i >= N for i in comp):
        raise IndexError(f"dims must be in [0, {N-1}]. Got {dims}.")

    # Distribute prime factors of R across compressed dims (greedy balance)
    tiled = {i: 1 for i in comp}
    for p in _prime_factors(tiled_dim):
        for i in reversed(comp):
            if full_shape[i] % p == 0:
                tiled[i] *= p
                break

    # Build final shape
    out = [tiled[i] if i in comp else full_shape[i] for i in range(N)]
    return tuple(out)


def _slice_tensor(node, dim, start, end, model):
    """
    Slice a tensor along a specific dimension using the given start and end indices.

    Args:
        node (Node): The node representing the tensor to be sliced.
        dim (int): The dimension along which to slice.
        start (int): The starting index for the slice.
        end (int): The ending index for the slice.
        graph (Graph): The computational graph to insert the slice operation.

    Returns:
        Node: A new node representing the sliced tensor.
    """
    graph = model.graph

    if node.op == "get_attr":
        param = fetch_attr(model, node.target)
        sliced_data = param.data.narrow(dim, start, end - start)

        tiled_node = create_getattr_from_value(
            model, graph, node.target + "_tiled", sliced_data
        )
    else:
        tiled_node = graph.call_function(
            torch.ops.aten.slice.Tensor, (node, dim, start, end),
        )

    propagate_shape(tiled_node, model)
    tiled_node.meta["dtype"] = node.meta.get("dtype")
    return tiled_node


def _get_node_attribute(node):
    args_and_kwargs = normalize_function(
        node.target,
        node.args,
        node.kwargs,
        normalize_to_only_use_kwargs=True
    )

    def sanitize_value(v):
        # Convert internal torch immutable lists/tuples to standard ones
        if isinstance(v, (list, tuple)) or "immutable" in str(type(v)):
            return list(v)
        return v

    return {
        k: sanitize_value(v)
        for k, v in args_and_kwargs.kwargs.items()
        if v is not None and not isinstance(v, Node)
    }


def _make_tiled_gemm_module(node, C, tile_c):
    """
    Factory function to create a TiledGemm module class with
    captured parameters.
    """
    bs = node.kwargs.get("block_size", None)
    is_mat = is_matmul(node)

    def slice_wt(weight, c0, c1):
        return weight[c0:c1] if is_mat else weight[:, c0:c1]

    def forward(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        psums = None
        for c in range(0, C, tile_c):
            c0, c1 = c, min(c + tile_c, C)

            args = (input[..., c0:c1], slice_wt(weight, c0, c1))

            op_kwargs = {}
            if bs is not None:
                bc0, bc1 = c0 // bs, c1 // bs
                op_kwargs.update({
                    "input_scale": input_scale[..., bc0:bc1],
                    "weight_scale": slice_wt(weight_scale, bc0, bc1),
                    "block_size": bs,
                    "input_code": input_code,
                    "weight_code": weight_code,
                })

            if not is_mat:
                args += (bias if c1 == C else None,)

            output = node.target(*args, **op_kwargs)
            psums = output if psums is None else psums + output

        return psums

    return WrapperModule(forward)


def _make_tiled_linear_with_outlier_filter_module(
    quantize_node, node, C, tile_c,
):
    """
    Factory function to create a TiledGemm module class with
    captured parameters.
    """
    quantized_ops = torch.ops.quantized_ops

    quantize_mx_outlier_kwargs = _get_node_attribute(quantize_node)
    gemm_kwargs = _get_node_attribute(node)

    bs = quantize_mx_outlier_kwargs["block_size"]
    is_mat = is_matmul(node)

    def slice_wt(weight, c0, c1):
        return weight[c0:c1] if is_mat else weight[:, c0:c1]

    def forward(
        input: torch.Tensor,
        input_qmap: Optional[torch.Tensor],
        scale_qmap: Optional[torch.Tensor],
        output_code: Optional[torch.Tensor],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        psums = None
        for c in range(0, C, tile_c):
            c0, c1 = c, min(c + tile_c, C)

            (
                data, indices, indptr, input_scale, inliers
            ) = quantized_ops.quantize_mx_outlier(
                input[..., c0:c1],
                qmap=input_qmap,
                scale_qmap=scale_qmap,
                output_code=output_code,
                **quantize_mx_outlier_kwargs,
            )

            args = (inliers, slice_wt(weight, c0, c1))

            kwargs = {
                "input_scale": input_scale,
                "weight_scale": slice_wt(weight_scale, c0 // bs, c1 // bs),
                "input_code": input_code,
                "weight_code": weight_code,
                "A_data": data,
                "A_indices": indices,
                "A_indptr": indptr,
            }

            if not is_mat:
                args += (bias if c1 == C else None,)

            output = node.target(*args, **kwargs, **gemm_kwargs)
            psums = output if psums is None else psums + output

        return psums

    return WrapperModule(forward)


def split_gemm_node(model, node, tile_sizes, tiled_shapes):
    """
    Transform a GEMM node (matmul/linear) into a tiled version along the
    reduction (C) dimension. Emits tiled sub-ops and replaces the original
    node in the FX graph.

    Args:
        model: FX GraphModule
        node: GEMM node to tile
        tile_sizes: tuple of (x_tiled, c_tiled, k_tiled)
        tiled_shapes: dict of tiled shapes
    """
    x_tiled, c_tiled, k_tiled = tile_sizes

    input_shape = node.args[0].shape
    X = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])
    C = input_shape[-1]

    is_mat = is_matmul(node)
    weight_shape = node.args[1].shape
    K = weight_shape[-1] if is_mat else weight_shape[0]

    if x_tiled == X and c_tiled == C and k_tiled == K:
        return

    tiling_meta = {
        "dtype": node.meta.get("dtype"),
        "l2_tiling": (X // x_tiled, K // k_tiled),
        "tiled_shapes": tiled_shapes,
        "tile_strides": {"A_indptr": tiled_shapes["input"][:-1]},
    }

    if C == c_tiled:
        node.meta.update(tiling_meta)
        return

    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

    A_data = node.kwargs.get("A_data")
    if A_data is not None:
        quantize_mx_node = A_data.args[0]

        example_inputs = (
            quantize_mx_node.args[0],
            get_arg_value(quantize_mx_node, 1, "qmap"),
            get_arg_value(quantize_mx_node, 6, "scale_qmap"),
            get_arg_value(quantize_mx_node, 7, "output_code"),
            node.args[1],
            node.args[2] if len(node.args) > 2 else None,
            node.kwargs.get("weight_scale"),
            node.kwargs.get("input_code"),
            node.kwargs.get("weight_code"),
        )

        module = _make_tiled_linear_with_outlier_filter_module(
            quantize_mx_node, node, C, c_tiled
        )

        # We first create a subgraph for the quantize_mx + gemm then replace
        # the whole subgraph with a tiled version of linear_mx with outlier filter
        named_modules = dict(model.named_modules())
        fused_nodes = [quantize_mx_node, node] + list(quantize_mx_node.users)
        node_to_replace = _create_and_insert_subgraph(
            fused_nodes, model, named_modules
        )

        gm = export_model(module, load_arg(example_inputs))
    else:
        node_to_replace = node

        args = node.args[:2] + (None,) if is_mat else node.args[:3]
        kwargs = {k: v for k, v in node.kwargs.items() if v is not None}

        module = _make_tiled_gemm_module(node, C, c_tiled)
        gm = export_model(module, load_arg(args), load_arg(kwargs))

    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)

        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            gm.graph.erase_node(n)

    gm.graph.lint()

    value_remap = {}
    replace_node_with_graph_module(model, node_to_replace, gm, value_remap)

    for n in list(value_remap.values()):
        if n.target == torch.ops.aten.slice.Tensor:
            if n.args[0].op == "get_attr":
                c_start = get_arg_value(n, 2, "start", None)
                c_end = get_arg_value(n, 3, "end", None)
                with model.graph.inserting_before(n):
                    sliced_param = _slice_tensor(n.args[0], 1, c_start, c_end, model)
                n.replace_all_uses_with(sliced_param)
                model.graph.erase_node(n)
            else:
                n.meta["dtype"] = n.args[0].meta.get("dtype")

        if n.target == operator.getitem:
            if (dtypes := n.args[0].meta.get("dtype")) is not None:
                idx = n.args[1]
                n.meta["dtype"] = dtypes[idx]

        if n.target == torch.ops.quantized_ops.quantize_mx_outlier.default:
            n.meta["dtype"] = quantize_mx_node.meta.get("dtype")

        if n.target == node.target:
            n.meta.update(copy.deepcopy(tiling_meta))


def get_valid_tiling(
    input_shape: Tuple[int, ...],
    min_sizes: Optional[Union[List[int], Tuple[int, ...]]] = None,
    multiple_of: Optional[Union[List[int], Tuple[int, ...]]] = None,
    order: Optional[Union[List[int], Tuple[int, ...]]] = None,
    fixed_dims: Optional[Union[List[int], Tuple[int, ...]]] = None,
    last_dim: Optional[int] = None,
    reverse: bool = False,
    round_robin: bool = False,
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    """
    Yields tile shapes by progressively reducing dimensions in a specified order.

    Args:
        input_shape: The original shape (e.g., (1024, 1024)).
        min_sizes: Minimum size for each dimension. If the list is shorter than
                   input_shape, it is padded with 1s on the left.
        multiple_of: Required multiple for each dimension's tile size. If the list
                     is shorter than input_shape, it is padded with 1s on the left
                     (1 means no constraint). E.g., multiple_of=(1, 16) requires the
                     last dimension's tile to be a multiple of 16.
        order: Explicit order of dimension indices to reduce.
        fixed_dims: Indices of dims that should remain at full size.
        last_dim: Convenience arg to fix dimensions starting from this index.
        reverse: If True, reverses the traversal order (ignored if `order` is provided).
        round_robin: If True, cycles through dimensions reducing them one step at a time.
                     If False, fully reduces one dimension before moving to the next.

    Yields:
        (current_shape, tiling_factors)
    """
    ndim = len(input_shape)

    # --- 1. Normalize Inputs ---

    # helper: resolving negative indices to positive
    def resolve_idx(i): return i + ndim if i < 0 else i

    # Set up fixed dimensions set
    fixed_indices = set()
    if fixed_dims:
        fixed_indices.update(resolve_idx(d) for d in fixed_dims)
    if last_dim is not None:
        start = resolve_idx(last_dim)
        fixed_indices.update(range(start, ndim))

    # Set up traversal order
    if order:
        traversal_order = [resolve_idx(i) for i in order]
    else:
        traversal_order = list(range(ndim))
        if reverse:
            traversal_order.reverse()

    # Align min_sizes to input_shape length (pad left with 1s)
    targets = list(min_sizes) if min_sizes else []
    if len(targets) < ndim:
        targets = [1] * (ndim - len(targets)) + targets

    # Align multiple_of to input_shape length (pad left with 1s — 1 means no constraint)
    multiples = list(multiple_of) if multiple_of else []
    if len(multiples) < ndim:
        multiples = [1] * (ndim - len(multiples)) + multiples

    # --- 2. Pre-calculate Valid Factors ---

    # We calculate all valid tiling sizes for every dimension upfront.
    # A factor is valid if it divides the dimension, >= min_size, and is a
    # multiple of the required multiple (if specified).
    # Example: input 128, multiple_of=16 -> [128, 64, 32, 16]
    dim_factors = {}
    for i in range(ndim):
        limit = max(1, targets[i]) # Ensure min_size is at least 1
        size = input_shape[i]
        limit = min(limit, size)  # Cap limit to size

        if i in fixed_indices:
            factors = [size]
        else:
            mult = multiples[i]
            # Generate factors in descending order, respecting divisibility and multiple_of
            factors = [f for f in range(size, limit - 1, -1) if size % f == 0 and f % mult == 0]
            if not factors:
                logger.warning(
                    f"No valid tiling found for dim {i} (size={size}, min={limit}, "
                    f"multiple_of={mult}); keeping full size."
                )
                factors = [size]

        dim_factors[i] = factors

    # --- 3. Traversal Logic ---

    # Current state: indices pointing to the current factor used for each dimension
    # initialized to 0 (which corresponds to the full input size)
    current_factor_indices = {i: 0 for i in range(ndim)}

    def get_current_state():
        """Constructs the shape and tiling tuple based on current indices."""
        shape = tuple(dim_factors[i][current_factor_indices[i]] for i in range(ndim))
        tiling = tuple(input_shape[i] // shape[i] for i in range(ndim))
        return shape, tiling

    # Yield the initial full shape
    yield get_current_state()

    if not round_robin:
        # --- Sequential Mode ---
        # Reduce Dim A fully, then move to Dim B, etc.
        for dim_idx in traversal_order:
            if dim_idx in fixed_indices:
                continue

            factors = dim_factors[dim_idx]
            # Iterate through the remaining factors for this dimension
            for i in range(1, len(factors)):
                current_factor_indices[dim_idx] = i
                yield get_current_state()

    else:
        # --- Round Robin Mode ---
        # Reduce Dim A (step 1), yield. Reduce Dim B (step 1), yield. Repeat.
        active_dims = [d for d in traversal_order if d not in fixed_indices]

        while True:
            progress_made = False

            for dim_idx in active_dims:
                current_idx = current_factor_indices[dim_idx]
                max_idx = len(dim_factors[dim_idx]) - 1

                # If this dimension can be reduced further
                if current_idx < max_idx:
                    # Move one step down in size
                    current_factor_indices[dim_idx] += 1
                    progress_made = True
                    yield get_current_state()

            # If we went through all dims and none could change, we are done
            if not progress_made:
                break


def _conv2d_layout(shape, is_weight=False, do_transpose=False):
    assert len(shape) == 4, "Conv2d shape must be 4D"
    if not do_transpose:
        return shape
    if is_weight:
        return (shape[2], shape[3], shape[1], shape[0])
    else:
        return (shape[0], shape[2], shape[3], shape[1])


def _build_conv2d_shape_map(node, tile_sizes, divisor=None):
    bs = node.kwargs.get("block_size", 1)
    transposed = node.meta.get("transposed", False)

    y_tile, x_tile, c_tile, k_tile = tile_sizes
    c_scaled = c_tile // bs

    stride = _pair(get_arg_value(node, 3, "stride", 1))
    padding = _pair(get_arg_value(node, 4, "padding", 0))
    dilation = _pair(get_arg_value(node, 5, "dilation", 1))

    weight_shape = node.args[1].shape
    kH, kW, _, _ = _conv2d_layout(weight_shape, True, not transposed)

    iy_tile = (
        (y_tile - 1) * stride[0] - 2 * padding[0]
        + dilation[0] * (kH - 1) + 1
    )
    ix_tile = (
        (x_tile - 1) * stride[1] - 2 * padding[1]
        + dilation[1] * (kW - 1) + 1
    )

    def apply_layout(shape, is_weight):
        return _conv2d_layout(shape, is_weight, transposed)

    return {
        "input": apply_layout((1, c_tile, iy_tile, ix_tile), False),
        "weight": apply_layout((k_tile, c_tile, kH, kW), True),
        "bias": (k_tile,),
        "input_scale": apply_layout((1, c_scaled, iy_tile, ix_tile), False),
        "weight_scale": apply_layout((k_tile, c_scaled, kH, kW), True),
        "output": apply_layout((1, k_tile, y_tile, x_tile), False),
    }


def _build_gemm_shape_map(node, tile_sizes, divisor=None):
    bs = node.kwargs.get("block_size", 1)

    x_tiled, c_tiled, k_tiled = tile_sizes
    c_scaled = c_tiled // bs

    input_shape = node.args[0].shape
    tiled_input_shape = construct_tiled_shape(
        input_shape, x_tiled, list(range(len(input_shape) - 1))
    )

    input_dims = tiled_input_shape[:-1]
    batch_dims = tiled_input_shape[:-2]

    is_mat = is_matmul(node)
    weight_transposed = is_mat ^ node.meta.get("transposed", False)

    if weight_transposed:
        weight_shape = (c_tiled, k_tiled)
        weight_scale_shape = (c_scaled, k_tiled)
    else:
        weight_shape = (k_tiled, c_tiled)
        weight_scale_shape = (k_tiled, c_scaled)

    if is_bmm(node):
        weight_shape = batch_dims + weight_shape
        weight_scale_shape = batch_dims + weight_scale_shape

    A_indptr = node.kwargs.get("A_indptr")
    if A_indptr is not None:
        value = A_indptr.value.reshape(-1)
        diffs = value[x_tiled::x_tiled] - value[:-x_tiled:x_tiled]

        # Round up to avoid underestimating nnz per tile
        if divisor is not None:
            ratio = divisor[0] * divisor[1]
        else:
            X, C = math.prod(input_shape[:-1]), input_shape[-1]
            ratio = (X / x_tiled) * (C / c_tiled)
        A_data = node.kwargs.get("A_data")
        nnz = max(int(A_data.value.numel() / ratio), diffs.max())

    return {
        "input": input_dims + (c_tiled,),
        "other" if is_mat else "weight": weight_shape,
        "bias": (k_tiled,),
        "input_scale": input_dims + (c_scaled,),
        "weight_scale": weight_scale_shape,
        "A_data": batch_dims + (nnz,) if A_indptr else None,
        "A_indices": batch_dims + (nnz,) if A_indptr else None,
        "A_indptr": batch_dims + (x_tiled + 1,),
        "output": input_dims + (k_tiled,),
    }


def _log_tiling_details(node, tiled_shapes, strategy):
    def fmt(s):
        if s is None: return "?"
        return str(tuple(s)).replace(" ", "")

    logger.info(f"Selected tiling for {node} (Strategy: {strategy}):")

    for n in node.all_input_nodes:
        if n in tiled_shapes and require_allocation(n):
            orig_shape = fmt(n.shape)
            tile_shape = fmt(tiled_shapes[n])
            logger.info(f"  In[{n}]: {orig_shape} -> {tile_shape}")

    orig_shape = fmt(node.shape)
    tile_shape = fmt(tiled_shapes[node])
    logger.info(f"  Out[{node}]: {orig_shape} -> {tile_shape}")


def _merge_tiling(a, b):
    if b is None:
        return a

    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + a
    b = (1,) * (n - len(b)) + b

    return tuple(ai * bi for ai, bi in zip(a, b))


def _search_tiling(
    node,
    full_shape,
    min_sizes,
    shape_func,
    cache_size,
    bank_width,
    bank_size,
    order=None,
    last_dim=None,
    fixed_dims=None,
    base_tiling=None,
    multiple_of=None,
    extra_size_fn=None,
):
    """
    Generic driver that iterates over banking strategies and valid tilings.
    """
    op_scope = _get_scope(node.target)
    node_to_key = get_node_to_key_map(node)
    key_to_node = {f"{op_scope}::{v}": k for k, v in node_to_key.items()}

    strategies = get_banking_strategies_for_op(node.target)

    for strategy in strategies:
        for tile_sizes, tiling in get_valid_tiling(
            full_shape,
            min_sizes=min_sizes,
            multiple_of=multiple_of,
            order=order,
            last_dim=last_dim,
            fixed_dims=fixed_dims,
        ):
            global_tiling = _merge_tiling(tiling, base_tiling)

            logger.debug(
                f"Trying tiling {global_tiling} with tile sizes {tile_sizes}"
            )

            tiled_shapes = shape_func(node, tile_sizes, global_tiling)
            node_to_shape = normalize_shape(node, tiled_shapes)

            total_size, _ = strategy.evaluate(
                key_to_node, node, node_to_shape, bank_width, bank_size
            )

            if extra_size_fn is not None:
                total_size += extra_size_fn(node, tile_sizes, tiled_shapes)

            if total_size <= cache_size:
                _log_tiling_details(node, node_to_shape, strategy)
                return tile_sizes, tiled_shapes

    logger.warning(f"Failed to tile {node} with cache size {cache_size}.")
    return None, None


def search_conv2d_tiling(node, unroll_dims, cache_size, bank_width, bank_size):
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    N, K, Y, X = node.shape
    C = node.args[0].shape[1]

    full_shape = (Y, X, C, K)

    min_xy = int(math.sqrt(unroll_dims[0]))

    # conv1 has special hardware constraints
    if C == 3:
        min_xy = 56

    min_sizes = (min_xy, min_xy, unroll_dims[0], unroll_dims[1])

    order = (3, 0, 1, 2)

    logger.info(f"Running L2 tiling for matrix op: {node}")

    return _search_tiling(
        node=node,
        full_shape=full_shape,
        min_sizes=min_sizes,
        order=order,
        shape_func=_build_conv2d_shape_map,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=bank_size
    )


def search_gemm_tiling(node, unroll_dims, cache_size, bank_width, bank_size):
    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    input_shape = node.args[0].shape
    X = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])
    C = input_shape[-1]

    is_mat = is_matmul(node)
    weight_shape = node.args[1].shape
    K = weight_shape[-1] if is_mat else weight_shape[0]

    x_min_size = min(sum(unroll_dims), X)

    num_c_tile = 1
    if bank_size is not None:
        input_bytes = get_node_bytes(node.args[0])
        c_max_size = bank_size / input_bytes / x_min_size
        for (c,), (num_c_tile,) in get_valid_tiling((C,), min_sizes=(unroll_dims[0],)):
            if c <= c_max_size:
                break
        else:
            logger.warning(
                f"Cannot find valid C tiling for {node} that fits bank size {bank_size}."
            )

    full_shape = (X, C // num_c_tile, K)
    min_sizes = (x_min_size, unroll_dims[0], unroll_dims[1])
    order = (2, 0, 1)

    logger.info(f"Running L2 tiling for matrix op: {node}")

    common_args = dict(
        node=node,
        full_shape=full_shape,
        min_sizes=min_sizes,
        multiple_of=unroll_dims,
        order=order,
        shape_func=_build_gemm_shape_map,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=bank_size,
        base_tiling=(1, num_c_tile, 1),
    )

    def _gemm_residual_size(node, tile_sizes, tiled_shapes):
        """Extra L2 cost of the accumulator buffer when C is tiled."""
        _, c_tiled, _ = tile_sizes
        if c_tiled < C:
            return math.prod(tiled_shapes["output"]) * get_node_bytes(node)
        return 0

    # Tiling for non-first sub-GEMMs (budget for the accumulator)
    tile_sizes, tiled_shapes = _search_tiling(
        **common_args, extra_size_fn=_gemm_residual_size
    )

    c_tiled = tile_sizes[1]

    if c_tiled < C:
        # Tiling for the first sub-GEMM (no accumulator buffer)
        search_args = {
            **common_args,
            "full_shape": (full_shape[0], c_tiled, full_shape[2]),
            "base_tiling": (1, C // c_tiled, 1),
            "fixed_dims": (1,),  # Fix C dim to ensure same C tile size
        }
        tile_sizes, tiled_shapes = _search_tiling(**search_args)

    return tile_sizes, tiled_shapes


def run_matrix_op_l2_tiling(
    model, unroll, cache_size=None, num_banks=None, bank_width=None,
):
    """
    Perform tiling on GEMM operations to fit intermediate data into cache.

    Args:
        model: A model object with a FX Graph containing GEMM nodes.
        unroll (int): Systolic array input and output channel unrolling dimension.
        cache_size (int): Total cache size in bytes.
        num_banks (int, optional): Number of cache banks for bank-aligned tiling.
        bank_width (int, optional): Width of memory for bank-aligned tiling.
    """
    graph = model.graph

    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE

    bank_size = None if num_banks is None else cache_size // num_banks

    for node in list(graph.nodes):
        if is_conv2d(node):
            tile_sizes, tiled_shape = search_conv2d_tiling(
                node, unroll, cache_size, None, bank_size
            )

            if tile_sizes is not None:
                split_conv2d_node(model, node, tile_sizes)

        elif is_linear(node) or is_matmul(node):
            tile_sizes, tiled_shapes = search_gemm_tiling(
                node, unroll, cache_size, None, bank_size
            )

            if tile_sizes is not None:
                split_gemm_node(model, node, tile_sizes, tiled_shapes)
            else:
                raise RuntimeError(f"Failed to tile GEMM node: {node}")

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def compute_tiled_shape(shape, divisor):
    ndim = len(shape)
    m = len(divisor)

    # Align divisor to shape dimensions
    if m < ndim:
        divisor = (1,) * (ndim - m) + divisor
    elif m > ndim:
        divisor = divisor[-ndim:]

    return tuple(
        s // d if s > 1 else s
        for s, d in zip(shape, divisor)
    )


def compute_output_tiled_shapes(node, tiling, override_shapes=None):
    """
    Computes tiled shape for an output node

    Args:
        node: The output node containing value and shape.
        tiling: The tiling divisor/size configuration.
        override_shapes: Optional shapes to use instead of node's value shapes.
    """
    if isinstance(node.value, torch.Tensor):
        return compute_tiled_shape(override_shapes or node.shape, tiling)
    elif isinstance(node.value, (tuple, list)):
        shapes = []
        has_sparse_outputs = len(node.value) > 2

        for i, tensor in enumerate(node.value):
            old_shape = override_shapes[i] if override_shapes else tensor.shape
            if has_sparse_outputs and i < 3:
                if i == 2:
                    old_shape = old_shape[:-1] + (old_shape[-1] - 1,)
                output_shape = old_shape + (1,)
                s = compute_tiled_shape(output_shape, tiling)[-2]
                if i == 2:
                    s = s + 1
                shapes.append(old_shape[:-1] + (s,))
            else:
                shapes.append(compute_tiled_shape(old_shape, tiling))
        return tuple(shapes)

    return None


def _build_vector_op_shape_map(node, tile_sizes, divisor):
    node_to_key = get_node_to_key_map(node)
    shapes_map = {}
    for n, k in node_to_key.items():
        if k == "output":
            shapes_map[k] = compute_output_tiled_shapes(node, divisor)
        elif require_allocation(n):
            shapes_map[k] = compute_tiled_shape(tuple(n.shape), divisor)
    return shapes_map


def run_vector_op_node_l2_tiling(
    node,
    unroll,
    cache_size=None,
    num_banks=None,
    bank_width=None,
):
    if not is_elementwise_op(node) and node.target not in [
        torch.ops.aten.softmax.int,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.quantized_ops.layer_norm.default,
        torch.ops.quantized_ops.calculate_mx_qparam.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        return

    # Certain dimensions cannot be tiled, e.g., transpose and reduction dims
    last_dim = -1
    min_sizes = (unroll,)
    if node.target == torch.ops.aten.softmax.int:
        last_dim = get_arg_value(node, 1, "dim", -1)
    elif node.target == torch.ops.aten.layer_norm.default:
        normalized_shape = get_arg_value(node, 1, "normalized_shape", None)
        last_dim = -len(normalized_shape) if normalized_shape is not None else -1
    elif node.target == torch.ops.quantized_ops.calculate_mx_qparam.default:
        axes = get_arg_value(node, 1, "axes", None)
        block_size = get_arg_value(node, 2, "block_size", None)
        ndim = len(node.args[0].shape)

        last_dim = None
        axes = set(axes or ())
        min_sizes = tuple(
            max(block_size, unroll) if i == ndim - 1
            else block_size if i in axes
            else 1
            for i in range(ndim)
        )
    elif node.target in [
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        last_dim = min(node.args[2])
    elif node.target == torch.ops.aten.transpose.int:
        last_dim = min(*node.args[1:])
    elif node.target == torch.ops.aten.permute.default:
        last_dim = next((i for i, d in enumerate(node.args[1]) if i != d), None)

    output_shape = (
        node.value.shape if isinstance(node.value, torch.Tensor)
        else node.value[-1].shape
    )

    logger.info(f"Running L2 tiling for vector op: {node}")

    tile_sizes, tiled_shapes = _search_tiling(
        node=node,
        full_shape=output_shape,
        min_sizes=min_sizes,
        last_dim=last_dim,
        shape_func=_build_vector_op_shape_map,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=None if num_banks is None else cache_size // num_banks,
    )

    if tile_sizes is not None:
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = tuple(
            s // ts for s, ts in zip(output_shape, tile_sizes)
        )


def run_vector_op_l2_tiling(
    model, unroll, cache_size=None, num_banks=None, bank_width=None,
):
    """
    Perform tiling on vector operations in a model to fit intermediate data into cache.

    Args:
        model: A model object with a FX Graph containing vector operation nodes.
        unroll (int): Minimum unrolling dimension for vector operations.
        cache_size (int): Total cache size in bytes.
        bank_width (int, optional): Width of memory for bank-aligned tiling.
        num_banks (int, optional): Number of memory banks for bank-aligned tiling.
    """
    graph = model.graph

    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE

    for node in list(graph.nodes):
        run_vector_op_node_l2_tiling(node, unroll, cache_size, num_banks)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


# ---------------------------------------------------------------------------
# Op sets used throughout the pooling tiling section
# ---------------------------------------------------------------------------

# Ops that use NHWC layout (quantized_ops, produced after the data_layout pass)
_NHWC_POOL_OPS = {
    torch.ops.quantized_ops.max_pool2d.default,
    torch.ops.quantized_ops.adaptive_avg_pool2d.default,
}

# Max-pooling ops (pad with -inf at boundaries); everything else pads with 0
_MAX_POOL_OPS = {
    torch.ops.quantized_ops.max_pool2d.default,
    torch.ops.aten.max_pool2d.default,
}

# Non-adaptive: fixed kernel/stride/padding → full 4-D (N, H, W, C) tiling
_NON_ADAPTIVE_POOL_OPS = {
    torch.ops.quantized_ops.max_pool2d.default,  # NHWC (after data_layout)
    torch.ops.aten.max_pool2d.default,            # NCHW (before data_layout)
}

# Adaptive: input window is proportional to H_in/H_out → only (N, C) tiling
_ADAPTIVE_POOL_OPS = {
    torch.ops.quantized_ops.adaptive_avg_pool2d.default,  # NHWC
    torch.ops.aten.adaptive_avg_pool2d.default,            # NCHW
    torch.ops.aten._adaptive_avg_pool2d,                   # NCHW (core aten)
}


# ---------------------------------------------------------------------------
# Non-adaptive pooling L2 tiling (max_pool2d, avg_pool2d, ...)
#
# These ops have explicit kernel_size/stride/padding/dilation, so each output
# tile maps to a deterministic input window.  Full 4-D (N, H, W, C) tiling is
# possible; boundary tiles are handled with F.pad before the pool call.
# ---------------------------------------------------------------------------

def _build_non_adaptive_pool_shape_map(node, tile_sizes, divisor=None):
    """
    Compute tiled input/output shapes for non-adaptive pooling ops.

    tile_sizes = (tile_N, tile_H, tile_W, tile_C), where H/W refer to the
    output spatial dimensions.  The corresponding input tile is derived from
    stride and dilation (padding does not change the input tile footprint).

    Handles both NHWC (quantized_ops, transposed) and NCHW (aten) layouts.
    The shape tuple ordering mirrors the node's actual tensor layout so that
    banking / scratchpad-size estimates are correct.

    Returns a dict with keys matching normalized op argument names:
        "input"   -> shape of the input tile
        "output" -> shape of the output tile
    """
    tile_N, tile_H, tile_W, tile_C = tile_sizes

    stride = _pair(get_arg_value(node, 2, "stride", 1))
    dilation = _pair(get_arg_value(node, 4, "dilation", 1))
    kernel_size = _pair(get_arg_value(node, 1, "kernel_size"))

    tile_H_in = (tile_H - 1) * stride[0] + dilation[0] * (kernel_size[0] - 1) + 1
    tile_W_in = (tile_W - 1) * stride[1] + dilation[1] * (kernel_size[1] - 1) + 1

    if node.target in _NHWC_POOL_OPS:  # NHWC: (N, H, W, C)
        return {
            "input": (tile_N, tile_H_in, tile_W_in, tile_C),
            "output": (tile_N, tile_H, tile_W, tile_C),
        }
    else:  # NCHW: (N, C, H, W)
        return {
            "input": (tile_N, tile_C, tile_H_in, tile_W_in),
            "output": (tile_N, tile_C, tile_H, tile_W),
        }


def _slice_dim(x, dim, start, end, size):
    """Slice x along dim only when the range is not the full extent."""
    if start > 0 or end < size:
        return x[(slice(None),) * dim + (slice(start, end),)]
    return x


def _cat_or_single(parts, dim):
    """Return parts[0] directly when there is only one part, skipping torch.cat."""
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


def _make_tiled_non_adaptive_pool_module(node, tile_sizes):
    """
    Factory function that creates a tiled WrapperModule for non-adaptive pooling.

    Tiles over (N, H_out, W_out, C).  Each spatial output tile [h0:h1, w0:w1]
    maps to a contiguous input window computed from stride/dilation/kernel_size.
    Boundary tiles are pre-padded in-layout via F.pad, then node.target is
    called with padding=0.  No explicit NHWC↔NCHW permutes are needed because
    node.target handles layout conversion internally.

    NHWC (quantized_ops): slice [n,h,w,c], pad H/W with (0,0, l,r, t,b).
    NCHW (aten):          slice [n,c,h,w], pad H/W with (l,r, t,b).
    """
    kernel_size = _pair(get_arg_value(node, 1, "kernel_size"))
    stride = _pair(get_arg_value(node, 2, "stride", kernel_size))
    padding = _pair(get_arg_value(node, 3, "padding", 0))
    dilation = _pair(get_arg_value(node, 4, "dilation", 1))
    ceil_mode = get_arg_value(node, 5, "ceil_mode", False)

    tile_N, tile_H, tile_W, tile_C = tile_sizes
    kH, kW = kernel_size
    stride_h, stride_w = stride
    dil_h, dil_w = dilation
    pad_h, pad_w = padding

    is_nhwc = node.target in _NHWC_POOL_OPS
    pad_value = float("-inf") if node.target in _MAX_POOL_OPS else 0.0
    target = node.target

    def forward(input: torch.Tensor) -> torch.Tensor:
        if is_nhwc:
            N, H_in, W_in, C = input.shape
        else:
            N, C, H_in, W_in = input.shape

        if ceil_mode:
            H_out = math.ceil((H_in + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1)
            W_out = math.ceil((W_in + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1)
        else:
            H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
            W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

        n_parts = []
        for n0 in range(0, N, tile_N):
            n1 = min(n0 + tile_N, N)
            h_parts = []
            for h0 in range(0, H_out, tile_H):
                h1 = min(h0 + tile_H, H_out)
                h_in_start = h0 * stride_h - pad_h
                h_in_end = (h1 - 1) * stride_h + dil_h * (kH - 1) + 1 - pad_h
                ptop = max(0, -h_in_start)
                pbot = max(0, h_in_end - H_in)
                h_in_s = max(0, h_in_start)
                h_in_e = min(H_in, h_in_end)

                w_parts = []
                for w0 in range(0, W_out, tile_W):
                    w1 = min(w0 + tile_W, W_out)
                    w_in_start = w0 * stride_w - pad_w
                    w_in_end = (w1 - 1) * stride_w + dil_w * (kW - 1) + 1 - pad_w
                    pleft = max(0, -w_in_start)
                    pright = max(0, w_in_end - W_in)
                    w_in_s = max(0, w_in_start)
                    w_in_e = min(W_in, w_in_end)

                    c_parts = []
                    for c0 in range(0, C, tile_C):
                        c1 = min(c0 + tile_C, C)
                        if is_nhwc:
                            # dim order: N=0, H=1, W=2, C=3
                            x = _slice_dim(input, 0, n0, n1, N)
                            x = _slice_dim(x, 1, h_in_s, h_in_e, H_in)
                            x = _slice_dim(x, 2, w_in_s, w_in_e, W_in)
                            x = _slice_dim(x, 3, c0, c1, C)
                            if ptop or pbot or pleft or pright:
                                x = F.pad(x, (0, 0, pleft, pright, ptop, pbot), value=pad_value)
                        else:
                            # dim order: N=0, C=1, H=2, W=3
                            x = _slice_dim(input, 0, n0, n1, N)
                            x = _slice_dim(x, 1, c0, c1, C)
                            x = _slice_dim(x, 2, h_in_s, h_in_e, H_in)
                            x = _slice_dim(x, 3, w_in_s, w_in_e, W_in)
                            if ptop or pbot or pleft or pright:
                                x = F.pad(x, (pleft, pright, ptop, pbot), value=pad_value)
                        c_parts.append(target(x, kernel_size, stride, 0, dilation, ceil_mode))
                    cat_dim = -1 if is_nhwc else 1
                    w_parts.append(_cat_or_single(c_parts, cat_dim))
                h_parts.append(_cat_or_single(w_parts, 2 if is_nhwc else 3))
            n_parts.append(_cat_or_single(h_parts, 1 if is_nhwc else 2))
        return _cat_or_single(n_parts, 0)

    return WrapperModule(forward)


def split_non_adaptive_pool_node(model, node, tile_sizes, tiled_shapes):
    """
    Replace a non-adaptive pooling node with a tiled subgraph, or annotate
    it with tiling metadata when no graph splitting is needed.

    Fast path (metadata-only): when every output dimension is exactly divisible
    by its tile size, all tiles are uniform and the input window advances by a
    constant stride per tile step.  In this case only tiled_shapes, l2_tiling,
    and tile_strides are set on the node — the hardware drives the tiled loads.

    Slow path: at least one dimension has a partial boundary tile.  Exports the
    tiled forward module, prunes no-op nodes, and splices it into the parent
    graph via replace_node_with_graph_module.  Validates at DEBUG log level.
    """
    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

    is_nhwc = node.target in _NHWC_POOL_OPS
    if is_nhwc:
        N, H_out, W_out, C = node.shape
    else:
        N, C, H_out, W_out = node.shape
    tile_N, tile_H, tile_W, tile_C = tile_sizes
    tiling = (N // tile_N, H_out // tile_H, W_out // tile_W, C // tile_C)

    # Uniform tiles: every dimension divides evenly → input window advances by a
    # constant number of pixels per tile step, so the hardware can drive tiled
    # loads without graph splitting.
    if N % tile_N == 0 and H_out % tile_H == 0 and W_out % tile_W == 0 and C % tile_C == 0:
        kernel_size = _pair(get_arg_value(node, 1, "kernel_size"))
        stride = _pair(get_arg_value(node, 2, "stride", kernel_size))
        h_in_stride = tile_H * stride[0]
        w_in_stride = tile_W * stride[1]
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = tiling
        if is_nhwc:
            node.meta["tile_strides"] = {"input": (tile_N, h_in_stride, w_in_stride, tile_C)}
        else:
            node.meta["tile_strides"] = {"input": (tile_N, tile_C, h_in_stride, w_in_stride)}
        return

    module = _make_tiled_non_adaptive_pool_module(node, tile_sizes)
    example_inputs = load_arg(tuple(node.all_input_nodes))

    if logger.isEnabledFor(logging.DEBUG):
        ref_out = node.target(*example_inputs, **load_arg(node.kwargs))
        tiled_out = module(*example_inputs)
        torch.testing.assert_close(tiled_out, ref_out, rtol=1e-4, atol=1e-4)
        logger.debug(f"Validation passed for tiled non-adaptive pool node: {node}")

    gm = export_model(module, example_inputs)

    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)
        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            gm.graph.erase_node(n)

    gm.graph.lint()

    value_remap = {}
    replace_node_with_graph_module(model, node, gm, value_remap)

    for n in value_remap.values():
        if n.target == node.target:
            n.meta.update({
                "tiled_shapes": copy.deepcopy(tiled_shapes),
                "l2_tiling": tiling,
                "dtype": node.meta.get("dtype"),
            })


# ---------------------------------------------------------------------------
# Adaptive pooling L2 tiling (adaptive_avg_pool2d, ...)
#
# These ops compute each output element from an adaptively-sized input window
# (proportional to H_in/H_out).  Spatial tiling requires remapping those
# windows, which is non-trivial.  In practice adaptive pooling almost always
# reduces spatial dims to (1, 1), so tiling N and C is sufficient.
# ---------------------------------------------------------------------------

def _build_adaptive_pool_shape_map(node, tile_sizes, divisor=None):
    """
    Compute tiled input/output shapes for adaptive pooling ops.

    tile_sizes = (tile_N, tile_C).  The full spatial extent of the input is
    always needed per tile because the adaptive window spans the whole input.

    Handles both NHWC (quantized_ops) and NCHW (aten) layouts.

    Returns a dict with keys matching normalized op argument names:
        "input"   -> shape of the input tile
        "output" -> shape of the output tile
    """
    tile_N, tile_C = tile_sizes
    if node.target in _NHWC_POOL_OPS:  # NHWC: (N, H, W, C)
        H_in, W_in = node.args[0].shape[1], node.args[0].shape[2]
        H_out, W_out = node.shape[1], node.shape[2]
        return {
            "input": (tile_N, H_in, W_in, tile_C),
            "output": (tile_N, H_out, W_out, tile_C),
        }
    else:  # NCHW: (N, C, H, W)
        H_in, W_in = node.args[0].shape[2], node.args[0].shape[3]
        H_out, W_out = node.shape[2], node.shape[3]
        return {
            "input": (tile_N, tile_C, H_in, W_in),
            "output": (tile_N, tile_C, H_out, W_out),
        }


def _make_tiled_adaptive_pool_module(node, tile_sizes):
    """
    Factory function that creates a tiled WrapperModule for adaptive pooling.

    Tiles over (N, C) only; the full spatial extent of the input is required
    for each tile because the adaptive window covers the entire input spatially.

    Handles both NHWC (quantized_ops, transposed) and NCHW (aten) layouts:
      - NHWC: input sliced as [n, :, :, c], permuted to NCHW, pooled, permuted back.
      - NCHW: input sliced as [n, c, :, :], pooled directly without permutation.
    """
    output_size = get_arg_value(node, 1, "output_size")
    tile_N, tile_C = tile_sizes
    is_nhwc = node.target in _NHWC_POOL_OPS

    def forward(input: torch.Tensor) -> torch.Tensor:
        N = input.shape[0]
        C = input.shape[-1] if is_nhwc else input.shape[1]
        n_parts = []
        for n0 in range(0, N, tile_N):
            n1 = min(n0 + tile_N, N)
            c_parts = []
            for c0 in range(0, C, tile_C):
                c1 = min(c0 + tile_C, C)
                if is_nhwc:
                    x = _slice_dim(input, 0, n0, n1, N)
                    x = _slice_dim(x, 3, c0, c1, C)
                    out_tile = F.adaptive_avg_pool2d(
                        x.permute(0, 3, 1, 2), output_size
                    ).permute(0, 2, 3, 1)
                else:
                    x = _slice_dim(input, 0, n0, n1, N)
                    x = _slice_dim(x, 1, c0, c1, C)
                    out_tile = F.adaptive_avg_pool2d(x, output_size)
                c_parts.append(out_tile)
            cat_dim = -1 if is_nhwc else 1
            n_parts.append(_cat_or_single(c_parts, cat_dim))
        return _cat_or_single(n_parts, 0)

    return WrapperModule(forward)


def split_adaptive_pool_node(model, node, tile_sizes, tiled_shapes):
    """
    Replace an adaptive pooling node with a tiled subgraph.

    If the tile covers the full output (no actual split), only sets node
    metadata.  Otherwise exports the tiled module, cleans up prunable ops,
    and splices it into the parent graph via replace_node_with_graph_module.
    Validates output correctness at DEBUG log level.
    """
    def load_arg(a):
        return map_arg(a, lambda n: n.value if isinstance(n, Node) else n)

    N = node.shape[0]
    C = node.shape[-1] if node.target in _NHWC_POOL_OPS else node.shape[1]
    tile_N, tile_C = tile_sizes
    tiling = (N // tile_N, C // tile_C)

    if tile_N == N and tile_C == C:
        node.meta["tiled_shapes"] = tiled_shapes
        node.meta["l2_tiling"] = tiling
        return

    module = _make_tiled_adaptive_pool_module(node, tile_sizes)
    example_inputs = load_arg(tuple(node.all_input_nodes))

    if logger.isEnabledFor(logging.DEBUG):
        ref_out = node.target(*example_inputs, **load_arg(node.kwargs))
        tiled_out = module(*example_inputs)
        torch.testing.assert_close(tiled_out, ref_out, rtol=1e-4, atol=1e-4)
        logger.debug(f"Validation passed for tiled adaptive pool node: {node}")

    gm = export_model(module, example_inputs)

    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)
        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            gm.graph.erase_node(n)

    gm.graph.lint()

    value_remap = {}
    replace_node_with_graph_module(model, node, gm, value_remap)

    for n in value_remap.values():
        if n.target == node.target:
            n.meta.update({
                "tiled_shapes": copy.deepcopy(tiled_shapes),
                "l2_tiling": tiling,
                "dtype": node.meta.get("dtype"),
            })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pool_op_l2_tiling(
    model, unroll, cache_size=None, num_banks=None, bank_width=None,
):
    """
    Perform tiling on pooling operations to fit intermediate data into Scratchpad.

    Dispatches to the appropriate tiling strategy based on whether the op is
    adaptive (tiles N and C) or non-adaptive (tiles N, H, W, and C).

    Args:
        model: A model object with a FX Graph containing pooling nodes.
        unroll (int): Minimum channel tile size (aligned to hardware unroll factor).
        cache_size (int): Total Scratchpad size in bytes.
        num_banks (int, optional): Number of cache banks for bank-aligned tiling.
        bank_width (int, optional): Width of memory for bank-aligned tiling.
    """
    graph = model.graph

    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE

    bank_size = None if num_banks is None else cache_size // num_banks

    for node in list(graph.nodes):
        if node.target in _NON_ADAPTIVE_POOL_OPS:
            if node.target in _NHWC_POOL_OPS:
                N, H_out, W_out, C = node.shape
            else:
                N, C, H_out, W_out = node.shape
            logger.info(f"Running L2 tiling for non-adaptive pool op: {node}")
            tile_sizes, tiled_shapes = _search_tiling(
                node=node,
                full_shape=(N, H_out, W_out, C),
                min_sizes=(1, 1, 1, unroll),
                order=(3, 0, 1, 2),
                shape_func=_build_non_adaptive_pool_shape_map,
                cache_size=cache_size,
                bank_width=bank_width,
                bank_size=bank_size,
            )
            if tile_sizes is not None:
                split_non_adaptive_pool_node(model, node, tile_sizes, tiled_shapes)

        elif node.target in _ADAPTIVE_POOL_OPS:
            N = node.shape[0]
            C = node.shape[-1] if node.target in _NHWC_POOL_OPS else node.shape[1]
            logger.info(f"Running L2 tiling for adaptive pool op: {node}")
            tile_sizes, tiled_shapes = _search_tiling(
                node=node,
                full_shape=(N, C),
                min_sizes=(1, unroll),
                order=(1, 0),
                shape_func=_build_adaptive_pool_shape_map,
                cache_size=cache_size,
                bank_width=bank_width,
                bank_size=bank_size,
            )
            if tile_sizes is not None:
                split_adaptive_pool_node(model, node, tile_sizes, tiled_shapes)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
