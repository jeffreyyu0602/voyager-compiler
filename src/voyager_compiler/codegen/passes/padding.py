import logging
import math
import operator
from typing import List

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

from .utils import get_arg_value
from ..mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_gemm_op,
    is_matmul,
)
from ...pt2e_utils import get_aten_graph_module, fetch_attr, propagate_shape
from ...quantize_pt2e import create_getattr_from_value

logger = logging.getLogger(__name__)

__all__ = [
    "pad_matrix_op_dimensions",
    "pad_vector_op_dimensions",
    "pad_vit_embeddings_output",
]


def slicing_and_padding_cancel_out(shape, slice_dim, start, end, pad):
    ndim = len(shape)
    k = len(pad) // 2

    pad_pairs = list(reversed(list(zip(pad[0::2], pad[1::2]))))
    full_pad_pairs = [(0, 0)] * (ndim - k) + pad_pairs

    for dim, (left, right) in enumerate(full_pad_pairs):
        if dim != slice_dim:
            if left != 0 or right != 0:
                return False
        else:
            if left != start or right != shape[slice_dim] - end:
                return False

    return True


# Relayout ops a pad is immune to: each leaves the padded dim's extent -- and
# its position from the end -- alone, so a slice above them cancels a pad below
# them exactly as an adjacent one would.  ``repeat_kv`` is why this earns its
# keep: its unsqueeze / expand / reshape sit between the two.
_PAD_IMMUNE_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
)

# The immune ops that *name* their output extents, so the padded dim's size has
# to be restated on them once the slice below is gone.
_SIZED_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
)


def _padded_dim(pad):
    """The one dim (counted from the end) ``pad`` grows, or None if it grows
    several -- no single slice cancels that."""
    dims = [
        -(k + 1) for k in range(len(pad) // 2) if pad[2 * k] or pad[2 * k + 1]
    ]
    return dims[0] if len(dims) == 1 else None


def _find_cancelling_slice(node, pad):
    """``(slice, chain)`` for the slice above ``node`` that ``pad`` exactly
    undoes, or ``(None, [])``.  ``chain`` is the immune relayout ops in
    between, nearest ``node`` first.  Each is required to have a single user:
    restating its extents must not disturb anyone else."""
    dim = _padded_dim(pad)
    if dim is None:
        return None, []

    chain = []
    curr = node
    while curr.target in _PAD_IMMUNE_OPS and len(curr.users) == 1:
        src = curr.args[0]
        if not isinstance(src, Node):
            return None, []
        if src.value.shape[dim] != curr.value.shape[dim]:
            return None, []
        chain.append(curr)
        curr = src

    if curr.target is not torch.ops.aten.slice.Tensor:
        return None, []
    if not slicing_and_padding_cancel_out(
        curr.args[0].value.shape, *curr.args[1:], pad
    ):
        return None, []
    return curr, chain


def _drop_slice_through_chain(cancel, chain, dim):
    """Re-point ``chain`` at what ``cancel`` sliced and restate the extents the
    relayout ops name for ``dim``, so the full-width tensor flows through them
    and the pad below is unnecessary."""
    full = cancel.args[0]
    size = full.value.shape[dim]
    chain[-1].replace_input_with(cancel, full)
    for n in reversed(chain):
        if n.target in _SIZED_OPS:
            sizes = list(n.args[1])
            i = dim + len(sizes)
            if sizes[i] != -1:
                sizes[i] = size
                n.update_arg(1, sizes)
        propagate_shape(n)


def pad_input_node(model, node, input, pad, scale, scale_pad, code):
    pad_quantize_mx_input = (
        scale is not None
        and input.target == operator.getitem
        and scale.target == operator.getitem
        and input.args[0] == scale.args[0]
        and input.args[0].target == torch.ops.quantized_ops.quantize_mx.default
    )

    node_to_pad = input.args[0].args[0] if pad_quantize_mx_input else input

    # Padding the pre-quantize float contributes 0 with a plain 0 fill; padding
    # the quantized value pads *codebook indices*, so it must fill with the
    # index that decodes to 0 (NF4 keeps 0 at index 7, not 0 -- index 0 is
    # -1.0), or the padded contraction adds a nonzero term.
    pad_value = 0
    if not pad_quantize_mx_input and code is not None:
        zeros = (fetch_attr(model, code.target) == 0).nonzero().flatten()
        pad_value = int(zeros[0]) if len(zeros) else 0

    skip_padding = False
    if node_to_pad.target == torch.ops.aten.slice.Tensor:
        arg = node_to_pad.args[0]
        skip_padding = slicing_and_padding_cancel_out(
            arg.value.shape, *node_to_pad.args[1:], pad
        )

    if skip_padding:
        new_input = node_to_pad.args[0]
    else:
        with model.graph.inserting_after(node_to_pad):
            new_input = model.graph.call_function(
                torch.ops.aten.pad.default,
                (node_to_pad, pad, "constant", pad_value),
            )

        propagate_shape(new_input)
        new_input.meta["dtype"] = node_to_pad.meta.get("dtype")

    if pad_quantize_mx_input:
        input.args[0].replace_input_with(node_to_pad, new_input)
        propagate_shape(input.args[0])
        propagate_shape(input)
        propagate_shape(scale)
    else:
        node.replace_input_with(node_to_pad, new_input)

        if scale is not None and any(x for x in scale_pad):
            with model.graph.inserting_before(node):
                padded_scale = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (scale, scale_pad),
                )

            node.replace_input_with(scale, padded_scale)

            propagate_shape(padded_scale)
            padded_scale.meta["dtype"] = scale.meta.get("dtype")


def slice_output(model, output_node, slice_args):
    sliced_output_users = []

    for user in list(output_node.users.keys()):
        # Cannot slice microscaling quantization ops.
        if user.target in [
            torch.ops.quantized_ops.dequantize.default,
            torch.ops.quantized_ops.quantize.default,
        ]:
            bs = get_arg_value(user, 4, "block_size")
            if bs is None:
                slice_output(model, user, slice_args)
                continue

        if is_elementwise_op(user):
            if len(user.all_input_nodes) == 1:
                propagate_shape(user)
                slice_output(model, user, slice_args)
                continue

            # If all inputs have the same slice args, strip the redundant slice.
            if all(
                n == output_node
                or (
                    n.target == torch.ops.aten.slice.Tensor
                    and n.args[1:] == slice_args
                )
                for n in user.all_input_nodes
            ):
                for n in user.all_input_nodes:
                    if n.target == torch.ops.aten.slice.Tensor:
                        user.replace_input_with(n, n.args[0])

                propagate_shape(user)
                slice_output(model, user, slice_args)
                continue

        sliced_output_users.append(user)

    if sliced_output_users:
        with model.graph.inserting_after(output_node):
            slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor,
                (output_node, *slice_args),
            )

        propagate_shape(slice_node)
        slice_node.meta["dtype"] = output_node.meta.get("dtype")

        for n in sliced_output_users:
            n.replace_input_with(output_node, slice_node)


def pad_matrix_op_dimensions(
    model: GraphModule,
    C_unroll,
    K_unroll,
) -> GraphModule:
    """
    Pad inputs and weights to conv2d nodes in a torch.fx.GraphModule so that
    the input channels (C) and output channels (K) are multiples of the
    provided unroll factors.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        C_unroll (int): Unroll factor for the input channels (C_in).
        K_unroll (int): Unroll factor for the output channels (C_out).

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    for node in list(model.graph.nodes):
        if not is_gemm_op(node):
            continue

        is_dw = is_depthwise_conv(node)
        is_conv = is_conv2d(node)
        is_mm = is_matmul(node)

        input = node.args[0]
        C_in = input.shape[1] if is_conv else input.shape[-1]

        # Skip CNN first layer with input channels equal to 3
        if is_conv and C_in == 3:
            continue

        pad_C = (C_unroll - (C_in % C_unroll)) % C_unroll

        bs = node.kwargs.get("block_size", 1)

        # Pad input along C dimension
        if pad_C:
            input_scale = node.kwargs.get("input_scale")

            if is_conv:
                input_pad = [0, 0, 0, 0, 0, pad_C]
                scale_pad = [0, 0, 0, 0, 0, pad_C // bs]
            else:
                input_pad = [0, pad_C]
                scale_pad = [0, pad_C // bs]

            pad_input_node(
                model,
                node,
                input,
                input_pad,
                input_scale,
                scale_pad,
                node.kwargs.get("input_code"),
            )

        weight = node.args[1]
        C_in = weight.shape[-2] if is_mm else weight.shape[1]
        C_out = weight.shape[-1] if is_mm else weight.shape[0]

        if is_dw:
            C_in *= node.args[6]
            node.args = node.args[:-1] + (node.args[-1] + pad_C,)

        pad_C = (C_unroll - (C_in % C_unroll)) % C_unroll
        pad_K = (K_unroll - (C_out % K_unroll)) % K_unroll

        if is_dw:
            pad_K = pad_C

        # Pad weight along K and C dimensions
        if pad_C or pad_K:
            weight_scale = node.kwargs.get("weight_scale")

            if is_dw:
                weight_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
            elif is_conv:
                weight_pad = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, pad_C // bs, 0, pad_K]
            elif is_mm:
                weight_pad = [0, pad_K, 0, pad_C]
                ws_pad = [0, pad_K, 0, pad_C // bs]
            else:
                weight_pad = [0, pad_C, 0, pad_K]
                ws_pad = [0, pad_C // bs, 0, pad_K]

            if weight.op == "get_attr":
                logger.debug(f"Pad {weight} with {weight_pad}")
                param = fetch_attr(model, weight.target)
                param.data = F.pad(param.data, weight_pad)
                propagate_shape(weight, model)

                if weight_scale is not None:
                    logger.debug(f"Pad {weight_scale} with {ws_pad}")
                    scale_param = fetch_attr(model, weight_scale.target)
                    scale_param.data = F.pad(scale_param.data, ws_pad)
                    propagate_shape(weight_scale, model)
            else:
                pad_input_node(
                    model,
                    node,
                    weight,
                    weight_pad,
                    weight_scale,
                    ws_pad,
                    node.kwargs.get("weight_code"),
                )

            if pad_K and len(node.args) > 2 and node.args[2] is not None:
                bias = node.args[2]
                bias_param = fetch_attr(model, bias.target)
                bias_param.data = F.pad(bias_param.data, [0, pad_K])
                propagate_shape(bias, model)

        propagate_shape(node)

        if pad_K:
            slice_dim = 1 if is_conv else -1
            slice_output(model, node, (slice_dim, 0, C_out))

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def _pad_layer_norm(
    model: GraphModule,
    node: Node,
    unroll: int,
) -> GraphModule:
    input = node.args[0]
    normalize_shape = node.args[1]
    weight = node.args[2]
    bias = node.args[3] if len(node.args) > 3 else None

    orig_k = input.shape[-1]
    pad_k = (-orig_k) % unroll
    if pad_k == 0:
        return model

    logger.info(f"Padding layer_norm {node} last dimension with {pad_k}")

    def pad_param(attr_node: Node):
        if attr_node.op == "get_attr":
            param = fetch_attr(model, attr_node.target)
            new_param = F.pad(param, [0, pad_k])
            with model.graph.inserting_after(attr_node):
                new_attr = create_getattr_from_value(
                    model, model.graph, f"{attr_node.target}_padded", new_param
                )
        else:
            with model.graph.inserting_after(attr_node):
                new_attr = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (attr_node, [0, pad_k]),
                )
        propagate_shape(new_attr, model)
        return new_attr

    new_weight = pad_param(weight)
    new_bias = pad_param(bias) if bias is not None else None

    with model.graph.inserting_before(node):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, [0, pad_k]),
        )
        layer_norm = model.graph.call_function(
            torch.ops.quantized_ops.layer_norm.default,
            (new_input, normalize_shape, new_weight, new_bias) + node.args[4:],
        )
        slice_node = model.graph.call_function(
            torch.ops.aten.slice.Tensor,
            (layer_norm, -1, 0, orig_k),
        )

    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")

    propagate_shape(layer_norm)

    propagate_shape(slice_node)
    node.replace_all_uses_with(slice_node)
    model.graph.erase_node(node)


def _pad_quantize_mx(model, node, unroll):
    input = node.args[0]
    axes = get_arg_value(node, 2, "axes")
    block_size = get_arg_value(node, 3, "block_size")
    ndim = len(input.shape)
    axes = {a % ndim for a in axes}

    # A tile boundary on the last dim must land on a hardware-unroll multiple,
    # and no quantization block may straddle two tiles, so each quant axis must
    # land on a block multiple.  A dim that is both must satisfy their lcm.  We
    # do not rely on the GEMM input padding to have aligned the last dim.
    pad_dims = {}
    for i in range(ndim):
        multiple = 1
        if i == ndim - 1:
            multiple = unroll
        if i in axes:
            multiple = math.lcm(multiple, block_size)
        if multiple > 1 and input.shape[i] % multiple:
            pad_dims[i] = (-input.shape[i]) % multiple

    if not pad_dims:
        return

    logger.info(f"Padding quantize_mx {node} with {pad_dims}")

    getitems = list(node.users)
    orig_shapes = {g: tuple(g.shape) for g in getitems}

    min_pad_dim = min(pad_dims)
    pad_tuple = []
    for dim in range(ndim - 1, min_pad_dim - 1, -1):
        pad_tuple.extend([0, pad_dims.get(dim, 0)])

    with model.graph.inserting_before(node):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, pad_tuple, "constant", 0),
        )

    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")
    node.replace_input_with(input, new_input)
    propagate_shape(node)

    # Slice each output back to its pre-pad shape, but only along dims that
    # actually grew: the quantized value carries every padded dim, whereas the
    # scale grows only where the pad added whole blocks or non-quant columns.
    for g in getitems:
        propagate_shape(g)
        orig = orig_shapes[g]
        users = list(g.users)
        sliced = g
        for d in range(len(orig)):
            if g.shape[d] != orig[d]:
                with model.graph.inserting_after(sliced):
                    sliced = model.graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        (sliced, d, 0, orig[d]),
                    )
                propagate_shape(sliced)
                sliced.meta["dtype"] = g.meta.get("dtype")
        if sliced is not g:
            for u in users:
                u.replace_input_with(g, sliced)


def _pad_softmax(model, node, unroll):
    input = node.args[0]

    # The hardware fetches in units of ``unroll`` elements along the last
    # dim, so pad it to a multiple of ``unroll`` with -inf and slice back.
    orig = input.shape[-1]
    pad_k = (-orig) % unroll
    if pad_k == 0:
        return

    logger.info(f"Padding softmax {node} with {pad_k}")

    with model.graph.inserting_after(input):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, [0, pad_k], "constant", float("-inf")),
        )
    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")
    node.replace_input_with(input, new_input)
    propagate_shape(node)

    with model.graph.inserting_after(node):
        slice_node = model.graph.call_function(
            torch.ops.aten.slice.Tensor,
            (node, -1, 0, orig),
        )
    node.replace_all_uses_with(slice_node)
    slice_node.replace_input_with(slice_node, node)
    propagate_shape(slice_node)
    slice_node.meta["dtype"] = node.meta.get("dtype")


def pad_vector_op_dimensions(
    model: GraphModule,
    K_unroll,
) -> GraphModule:
    """
    Pad inputs to vector operations to multiples of the hardware unroll size.
    Only support softmax operation for now.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        K_unroll (int): Unroll factor for the output channels.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.layer_norm.default:
            _pad_layer_norm(model, node, K_unroll)
        elif node.target == torch.ops.aten.softmax.int:
            _pad_softmax(model, node, K_unroll)
        elif node.target == torch.ops.quantized_ops.quantize_mx.default:
            _pad_quantize_mx(model, node, K_unroll)

    for node in list(model.graph.nodes):
        if node.target is not torch.ops.aten.pad.default:
            continue

        pad = get_arg_value(node, 1, "pad")
        cancel, chain = _find_cancelling_slice(node.args[0], pad)
        if cancel is None:
            continue

        logger.info(f"Eliminating slice / pad round-trip: {cancel} and {node}")
        if chain:
            _drop_slice_through_chain(cancel, chain, _padded_dim(pad))
            node.replace_all_uses_with(node.args[0])
        else:
            node.replace_all_uses_with(cancel.args[0])

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_vit_embeddings_output(
    model: GraphModule,
    embeddings,
    example_inputs,
    dynamic_shapes=None,
    unroll=32,
):
    original_graph = model.graph

    pattern = get_aten_graph_module(
        embeddings, example_inputs, dynamic_shapes=dynamic_shapes
    )
    pattern_graph = pattern.graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)
    logger.info(f"Found {len(_matches)} matches")

    if not _matches:
        return model

    vit_embed_out = _matches[0].returning_nodes[0]
    orig_dim = vit_embed_out.meta["val"].shape[-2]
    pad = (unroll - (orig_dim % unroll)) % unroll
    logger.info(f"Padding {vit_embed_out} with {pad}")

    with model.graph.inserting_after(vit_embed_out):
        pad_node = model.graph.call_function(
            torch.ops.aten.pad.default,
            (vit_embed_out, [0, 0, 0, pad]),
        )

    propagate_shape(pad_node)
    pad_node.meta["val"] = pad_node.value

    for user in list(vit_embed_out.users):
        if id(user) != id(pad_node):
            user.replace_input_with(vit_embed_out, pad_node)

    for node in model.graph.nodes:
        if node.target in [
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
        ]:
            new_size = [x if x != orig_dim else x + pad for x in node.args[1]]
            node.args = (node.args[0], new_size)

    model.graph.lint()
    model.recompile()
    return model
