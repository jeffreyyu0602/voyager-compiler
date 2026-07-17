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
    _BROADCAST_OPS,
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
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
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


def _pad_for_dim(pad, dim, new_dim):
    """``pad``'s one non-zero pair, restated to grow ``new_dim`` rather than
    ``dim`` -- both counted from the end, as ``F.pad`` orders its pairs."""
    j = -dim - 1
    out = [0] * (2 * -new_dim)
    out[-2 * new_dim - 2] = pad[2 * j]
    out[-2 * new_dim - 1] = pad[2 * j + 1]
    return out


def _dim_above(node, dim):
    """``dim`` -- counted from the end of ``node``'s output -- restated against
    its input, or ``None`` if ``node`` does not hand that dim through intact.

    A transpose or permute genuinely moves the dim, so it is remapped; the rest
    of ``_PAD_IMMUNE_OPS`` only touch dims to its left and leave it where it is.
    """
    src = node.args[0]
    if not isinstance(src, Node) or not hasattr(src, "value"):
        return None
    out, inp = tuple(node.value.shape), tuple(src.value.shape)
    rank = len(out)

    if node.target is torch.ops.aten.transpose.int:
        a, b = (int(d) % rank - rank for d in node.args[1:3])
        above = {a: b, b: a}.get(dim, dim)
    elif node.target is torch.ops.aten.permute.default:
        perm = [int(p) % rank for p in node.args[1]]
        above = perm[dim + rank] - rank
    else:
        above = dim

    if len(inp) < -above or inp[above] != out[dim]:
        return None
    return above


def _relayout_chain_above(node, dim):
    """``(chain, src, src_dim, dims)`` -- the immune relayout ops feeding
    ``node`` (nearest first), the tensor they re-address, ``dim`` restated
    against it, and ``dim`` as each chain op's own output sees it.  Walks while
    the op hands the dim through and nobody else reads it: restating an op's
    extents must disturb no one."""
    chain, dims = [], []
    curr = node
    while curr.target in _PAD_IMMUNE_OPS and len(curr.users) == 1:
        above = _dim_above(curr, dim)
        if above is None:
            break
        chain.append(curr)
        dims.append(dim)
        dim = above
        curr = curr.args[0]
    return chain, curr, dim, dims


def _find_cancelling_slice(node, pad):
    """``(slice, chain, dims)`` for the slice above ``node`` that ``pad``
    exactly undoes, or ``(None, [], [])``."""
    dim = _padded_dim(pad)
    if dim is None:
        return None, [], []

    chain, curr, src_dim, dims = _relayout_chain_above(node, dim)
    if src_dim != dim:
        # The pad and the slice name different dims; they cannot be compared
        # without rebuilding the pad, which the cancel check does not do.
        return None, [], []
    if curr.target is not torch.ops.aten.slice.Tensor:
        return None, [], []
    if not slicing_and_padding_cancel_out(
        curr.args[0].value.shape, *curr.args[1:], pad
    ):
        return None, [], []
    return curr, chain, dims


def _restate_chain(chain, dims, size):
    """Restate the extent each op in ``chain`` names for the padded dim -- at
    that op's own view of it -- so a tensor grown to ``size`` flows through."""
    for n, d in zip(reversed(chain), reversed(dims)):
        if n.target in _SIZED_OPS:
            sizes = list(n.args[1])
            i = d + len(sizes)
            if sizes[i] != -1:
                sizes[i] = size
                n.update_arg(1, sizes)
        propagate_shape(n)


def _drop_slice_through_chain(cancel, chain, dims):
    """Re-point ``chain`` at what ``cancel`` sliced and restate the extents the
    relayout ops name, so the full-width tensor flows through them and the pad
    below is unnecessary."""
    full = cancel.args[0]
    chain[-1].replace_input_with(cancel, full)
    _restate_chain(chain, dims, full.value.shape[dims[-1]])


_INDEX_COPY = torch.ops.aten.index_copy_.default

# Whether a pad on a KV-cache write is taken into the buffer rather than run
# over the whole cache every step (``_fold_pad_into_cache``).  It rewrites the
# buffer's contents and its dtype, so it is worth being able to turn off.
FOLD_PAD_INTO_CACHE = True


def _fold_pad_into_cache(model, idx, pad, pad_value, fold_cache):
    """Pad the KV cache buffer itself, and the token written into it, rather
    than the whole cache on every step.

    The write puts one token into a buffer the array wants wider.  Widening the
    buffer is a compile-time edit of its contents, and the token then arrives
    already wide -- a pad on one position, which cancels against the slice its
    projection left on it.  So the cache pays nothing.

    Not when the pad grows the axis the write indexes: that would move every
    position the cache holds.
    """
    dim = _padded_dim(pad)
    if dim is None:
        return False

    cache = idx.args[0]
    if not isinstance(cache, Node) or cache.op != "get_attr":
        return False
    if idx.args[1] % len(idx.value.shape) == dim % len(idx.value.shape):
        return False

    baked = F.pad(fetch_attr(model, cache.target), pad, "constant", pad_value)
    with model.graph.inserting_before(idx):
        wide = create_getattr_from_value(
            model, model.graph, cache.target + "_padded", baked
        )
        propagate_shape(wide, model)

    # The token has to arrive wide, one position's worth.
    token = _insert_pad(model, idx.args[3], pad, pad_value, fold_cache)

    wide.meta["dtype"] = cache.meta.get("dtype")
    idx.update_arg(0, wide)
    idx.update_arg(3, token)
    propagate_shape(idx)
    return True


def _insert_pad(model, node, pad, pad_value, fold_cache=False):
    """``node`` grown by ``pad``: a pad node, or -- where a slice above it left
    exactly what the pad puts back -- what that slice was hiding, with neither
    op left behind.

    Every pad goes in through here, so each way of not paying for one is tried
    in turn: a slice above that already left room, then a cache write that can
    take the pad into its buffer, then a buffer that is simply padded as it
    stands.  The round-trip sweep would reach the same graph on its own, but
    only after L2 tiling and the layout pass have read a pad that was never
    needed, and tiling is what it would mislead.
    """
    cancel, chain, dims = _find_cancelling_slice(node, pad)
    if cancel is not None:
        logger.info(f"Padding {node} cancels the slice {cancel}")
        if not chain:
            return cancel.args[0]
        _drop_slice_through_chain(cancel, chain, dims)
        return node

    if (
        fold_cache
        and node.target is _INDEX_COPY
        and _fold_pad_into_cache(model, node, pad, pad_value, fold_cache)
    ):
        return node

    if node.op == "get_attr":
        # Nothing writes it, so its padded form is known now: widen the buffer
        # once here rather than the tensor it holds on every step.
        baked = F.pad(
            fetch_attr(model, node.target), pad, "constant", pad_value
        )
        with model.graph.inserting_after(node):
            wide = create_getattr_from_value(
                model, model.graph, node.target + "_padded", baked
            )
        propagate_shape(wide, model)
        wide.meta["dtype"] = node.meta.get("dtype")
        return wide

    with model.graph.inserting_after(node):
        padded = model.graph.call_function(
            torch.ops.aten.pad.default, (node, pad, "constant", pad_value)
        )
    propagate_shape(padded)
    padded.meta["dtype"] = node.meta.get("dtype")
    return padded


def _hoist_pad_above_repeat(model, node, pad, pad_value, fold_cache):
    """Pad what the relayout ops feeding ``node`` re-address, rather than what
    they hand on, and let the pad flow back down through them.

    ``repeat_kv`` is the case that matters: padding its output fills the 32
    heads it broadcast, where padding its input fills the 8 the cache holds --
    the same fill on a quarter of the data, and the broadcast still folds into
    the consumer's addressing.  Only worth it when the chain actually repeats;
    over a plain relayout the pad is the same size either side of it.  A
    transpose on the way up moves the dim the pad grows, so the pad is rebuilt
    against the dim the source holds it in.

    Returns the tensor the op should read: ``node`` itself where the pad was
    hoisted (its chain now carries the wide tensor), and otherwise whatever
    padding ``node`` directly comes to.
    """
    dim = _padded_dim(pad)
    if dim is None:
        return _insert_pad(model, node, pad, pad_value, fold_cache)

    chain, src, src_dim, dims = _relayout_chain_above(node, dim)
    if not any(n.target in _BROADCAST_OPS for n in chain):
        return _insert_pad(model, node, pad, pad_value, fold_cache)

    padded = _insert_pad(
        model, src, _pad_for_dim(pad, dim, src_dim), pad_value, fold_cache
    )
    chain[-1].replace_input_with(src, padded)
    _restate_chain(chain, dims, padded.value.shape[src_dim])
    return node


def pad_input_node(model, node, is_weight, pad, scale_pad, fold_cache):
    if is_weight:
        input = node.args[1]
        scale = node.kwargs.get("weight_scale")
        code = node.kwargs.get("weight_code")
    else:
        input = node.args[0]
        scale = node.kwargs.get("input_scale")
        code = node.kwargs.get("input_code")

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

    new_input = _hoist_pad_above_repeat(
        model, node_to_pad, pad, pad_value, fold_cache
    )

    if pad_quantize_mx_input:
        input.args[0].replace_input_with(node_to_pad, new_input)
        propagate_shape(input.args[0])
        propagate_shape(input)
        propagate_shape(scale)
    else:
        node.replace_input_with(node_to_pad, new_input)

        if scale is not None and any(x for x in scale_pad):
            node.replace_input_with(
                scale,
                _hoist_pad_above_repeat(model, scale, scale_pad, 0, fold_cache),
            )


def _crossable(node, user, slice_args):
    """Whether ``user`` can read ``node`` unsliced -- and widen it to, if so.

    An op that keeps every lane to itself does not care that the pad left extra
    ones, so the slice can sink past it and be spent nearer a pad that cancels
    it.  A per-tensor quantize is such an op; a blocked one is not, a slice
    below it cutting the blocks it scaled.  A second operand agrees only if it
    is already sliced the same way -- and stripping those slices is what widens
    the op.
    """
    # Only support non-MX quantization
    if user.target in [
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
    ]:
        return get_arg_value(user, 4, "block_size") is None

    if not is_elementwise_op(user):
        return False

    if len(user.all_input_nodes) != 1:
        if not all(
            n == node
            or (
                n.target == torch.ops.aten.slice.Tensor
                and n.args[1:] == slice_args
            )
            for n in user.all_input_nodes
        ):
            return False

        for n in user.all_input_nodes:
            if n.target == torch.ops.aten.slice.Tensor:
                user.replace_input_with(n, n.args[0])

    propagate_shape(user)
    return True


def slice_output(model, node, slice_args):
    frontier = [node]

    while frontier:
        node = frontier.pop()
        needs_slice = []
        for user in list(node.users.keys()):
            if _crossable(node, user, slice_args):
                frontier.append(user)
            else:
                needs_slice.append(user)
        if not needs_slice:
            continue

        with model.graph.inserting_after(node):
            slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor, (node, *slice_args)
            )

        propagate_shape(slice_node)
        slice_node.meta["dtype"] = node.meta.get("dtype")

        for n in needs_slice:
            n.replace_input_with(node, slice_node)


def pad_matrix_op_dimensions(
    model: GraphModule,
    C_unroll,
    K_unroll,
    fold_cache: bool = FOLD_PAD_INTO_CACHE,
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

        # Pad input along input channel dimension
        if pad_C:
            lead = [0, 0, 0, 0] if is_conv else []
            in_pad = lead + [0, pad_C]
            in_scale_pad = lead + [0, pad_C // bs]
            pad_input_node(model, node, False, in_pad, in_scale_pad, fold_cache)

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

        # Pad weight along input and output channel dimensions
        if pad_C or pad_K:
            if is_dw:
                w_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
            elif is_conv:
                w_pad = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, pad_C // bs, 0, pad_K]
            elif is_mm:
                w_pad = [0, pad_K, 0, pad_C]
                ws_pad = [0, pad_K, 0, pad_C // bs]
            else:
                w_pad = [0, pad_C, 0, pad_K]
                ws_pad = [0, pad_C // bs, 0, pad_K]
            pad_input_node(model, node, True, w_pad, ws_pad, fold_cache)

        bias = get_arg_value(node, 2, "bias")
        if pad_K and bias is not None:
            new_bias = _insert_pad(model, bias, [0, pad_K], 0)
            node.replace_input_with(bias, new_bias)

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


def _pad_quantize_mx(model, node, unroll, fold_cache):
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

    new_input = _insert_pad(model, input, pad_tuple, 0, fold_cache)
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
    fold_cache: bool = FOLD_PAD_INTO_CACHE,
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
            _pad_quantize_mx(model, node, K_unroll, fold_cache)

    for node in list(model.graph.nodes):
        if node.target is not torch.ops.aten.pad.default:
            continue

        pad = get_arg_value(node, 1, "pad")
        cancel, chain, dims = _find_cancelling_slice(node.args[0], pad)
        if cancel is None:
            continue

        logger.info(f"Eliminating slice / pad round-trip: {cancel} and {node}")
        if chain:
            _drop_slice_through_chain(cancel, chain, dims)
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
