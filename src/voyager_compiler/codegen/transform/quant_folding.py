"""Fold quantize / dequantize nodes into the ops around them.

Two passes over an already-quantized graph, both of which move a quantize or
dequantize rather than compute it:

  * ``fuse_quantize_dequantize_with_previous_op`` runs inside ``transform()``.
    It hoists a quantize into its producer (so a value is stored already
    narrow), replays it above a relayout, folds one into a KV cache write, and
    splits a quantized cache feeding a GEMV.
  * ``fuse_dequantize_quantize`` collapses a ``get_attr -> dequantize -> layout
    ops -> quantize`` chain into a single ``dequantize`` with pre-multiplied
    scales, storing a grouped-query parameter once rather than once per head.

Both work purely on the ``quantized_ops`` schema and FX, not on the quantizer
that produced the graph.
"""

import copy
import logging
import math
import operator
from typing import Optional, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

from ..aten_classifier import is_compute_op
from ..node_info import (
    _BROADCAST_OPS,
    get_arg_value,
    is_gemm_op,
    is_mha_qkv_permute,
    is_nop,
    is_reshape_op,
    reshape_preserves_full_blocks,
)
from ..subgraph import (
    create_and_insert_subgraph,
    replace_node_with_graph_module,
)
from ...export_utils import create_getattr_from_value, get_aten_graph_module
from ...ops.quantized import expand
from ...shape_prop import fetch_attr, propagate_shape

logger = logging.getLogger(__name__)

__all__ = [
    "fuse_dequantize_quantize",
    "fuse_quantize_dequantize_with_previous_op",
]


_HOISTABLE_OPS = (
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.expand.default,
    torch.ops.aten.repeat.default,
)

# Ops that regroup dims without moving an element: they preserve row-major
# order, so a quantize can be lifted over one even when it cuts across the axis
# the quantize blocks along -- as long as the blocks come out the same
# (``_blocks_survive_regroup``).  These are also the only ops whose size
# argument ``_replay_relayout`` knows how to rebuild for the scale.
_REGROUP_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
)

# MHA head splitting rejoins the per-head results with one of these, so a
# quantize lifted over it has to be duplicated onto every branch.
_FORK_OPS = (
    torch.ops.aten.stack.default,
    torch.ops.aten.cat.default,
)

_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
_QUANTIZE_OPS = (
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.dequantize.default,
    _QUANTIZE_MX,
)

# ``quantize_mx`` returns ``(scale, value)``.  The value inherits the relayout
# ops the quantize was lifted over; the scale gets a replayed copy of them.
_MX_VALUE = 1


def _is_relayout(node) -> bool:
    return (
        isinstance(node, Node)
        and (
            is_nop(node) or is_reshape_op(node) or node.target in _HOISTABLE_OPS
        )
        and len(node.all_input_nodes) == 1
    )


def _axes_above(
    node: Node, axes: Tuple[int, ...], block_size: Optional[int]
) -> Optional[Tuple[int, ...]]:
    """``axes`` -- the axes a microscaling quantize blocks along, read against
    ``node``'s output -- restated against its input.  ``None`` if the blocks do
    not survive ``node``, which is then as far as the quantize can be lifted.

    Axes count from the end, so an op that only rearranges dims to the *left*
    of a block axis leaves it alone: that covers every op on the ``repeat_kv``
    path (``unsqueeze``, ``expand``, the head-flattening ``reshape``).  A
    transpose or a permute genuinely moves the axis, so it is remapped.  A
    reshape that regroups the block axis itself still passes if the blocks come
    out the same set of elements.  A per-tensor quantize passes ``()`` and is
    unaffected.
    """
    out_shape = tuple(node.value.shape)
    in_shape = tuple(node.args[0].value.shape)
    rank = len(out_shape)

    if node.target is torch.ops.aten.transpose.int:
        a, b = (int(d) % rank - rank for d in node.args[1:3])
        swap = {a: b, b: a}
        return tuple(swap.get(x, x) for x in axes)

    if node.target is torch.ops.aten.permute.default:
        perm = [int(p) % rank for p in node.args[1]]  # out dim i <- in perm[i]
        return tuple(perm[x + rank] - rank for x in axes)

    if any(in_shape[x:] != out_shape[x:] for x in axes):
        # A reshape regrouping the block axis is still crossable if the blocks
        # come out the same sets of elements.
        if node.target not in _REGROUP_OPS or len(axes) != 1:
            return None
        a = axes[0]
        if block_size is None or len(in_shape) < -a or len(out_shape) < -a:
            return None
        if not reshape_preserves_full_blocks(
            in_shape,
            a + len(in_shape),
            out_shape,
            a + len(out_shape),
            block_size,
        ):
            return None
    return axes


def _relayout_path(
    start,
    axes: Tuple[int, ...],
    block_size: Optional[int],
    keep_head_permute=False,
):
    """Walk up from ``start`` over relayout ops, restating ``axes`` at each.

    Returns ``(src, path, axes_at, src_axes)``: ``path`` is the ops crossed,
    nearest ``start`` first; ``src`` the node that actually computed the data;
    ``axes_at[k]`` the block axes as seen at ``path[k]``'s output; ``src_axes``
    those at ``src``.  The walk stops at the first node that computes, that
    someone else also reads, or that the blocks would not survive.

    ``keep_head_permute`` also stops it at an MHA head permute -- a *fusable*
    reshape, one the GEMM below can store straight through
    (``fuse_reshape_with_output``).  That is exactly where a multi-output
    quantize wants to sit: fused as the last op of that group, quantizing the
    tile on its way out.  Lifted past it, it would leave a ``getitem`` between
    the two and neither could fuse.  A single-output quantize has no ``getitem``
    and steps over freely.
    """
    path, axes_at = [], []
    src = start
    while (
        _is_relayout(src)
        and len(src.users) == 1
        and not (keep_head_permute and is_mha_qkv_permute(src))
    ):
        above = _axes_above(src, axes, block_size)
        if above is None:
            break
        path.append(src)
        axes_at.append(axes)
        axes = above
        src = src.args[0]
    return src, path, axes_at, axes


def _copy_quantize_above(model: GraphModule, node: Node, src: Node, axes):
    """A copy of quantize ``node`` reading ``src``, inserted right after it."""

    graph = model.graph
    remap = {node.args[0]: src}
    with graph.inserting_before(src.next):
        for n in node.all_input_nodes:
            if n not in remap:
                remap[n] = graph.node_copy(n)
        new = graph.node_copy(node, lambda n: remap[n])

    if node.target is _QUANTIZE_MX:
        args = list(new.args)
        args[2] = list(axes)
        new.args = tuple(args)

    for n in list(remap.values()) + [new]:
        propagate_shape(n, model)
    new.meta = {
        k: copy.deepcopy(v) if k != "val" else v.clone()
        for k, v in node.meta.items()
    }
    return new


def _replay_relayout(
    graph: Graph, node: Node, src: Node, axes, block_size: int
) -> Node:
    """Copy relayout ``node`` onto ``src``.  ``src`` is the scale of a hoisted
    ``quantize_mx``, so along ``axes`` it holds one element per *block* where
    the original input held one per element -- a shape argument keeps that dim's
    own extent and divides it by the block size, rounding up: an axis shorter
    than a block (e.g. a 128-wide head under a 256 block) is one block, matching
    how ``_reshape_to_blocks`` pads a partial axis up to a single tile.  Every
    other dim is untouched: ``_axes_above`` already proved the blocks survive.
    """
    new = graph.node_copy(node, lambda n: src if n is node.args[0] else n)
    out_shape = tuple(node.value.shape)

    if node.target in _REGROUP_OPS:
        shape = list(out_shape)
        for a in axes:
            shape[a] = -(-out_shape[a] // block_size)
        new.args = (src, shape)
    elif node.target is torch.ops.aten.expand.default:
        # ``-1`` keeps a dim, so naming only the dims this expand actually grows
        # makes the sizes independent of how long the block axis is.
        in_shape = tuple(node.args[0].value.shape)
        new.args = (
            src,
            [
                out_shape[d] if out_shape[d] != in_shape[d] else -1
                for d in range(len(out_shape))
            ],
        )
    return new


def _annotate(path, source: Node) -> None:
    """The relayout ops now sit *below* the quantize, so they carry the dtype of
    the tensor that flows through them -- or none at all, once a dequantize has
    put it back in the clear.  ``source`` is that tensor: the quantize itself,
    or, for a multi-output one, the single output the path was rewired onto (not
    the op, whose ``dtype`` is the pair it returns).
    """
    is_dequantize = source.target is torch.ops.quantized_ops.dequantize.default
    for n in path:
        if is_dequantize:
            n.meta.pop("dtype", None)
        else:
            n.meta["dtype"] = source.meta.get("dtype", None)


def _hoist_forked(model: GraphModule, node: Node) -> bool:
    """Lift a single-output quantize over the relayout ops feeding it.

    A ``stack`` / ``cat`` on the way up (MHA head splitting rejoining its heads)
    forks the walk: every branch is quantized on its own, and the concat then
    joins pieces that are already quantized.
    """
    graph = model.graph
    on_path, moved = [], False
    todo = [(node.args[0], node)]  # (tensor to lift over, the node reading it)

    while todo:
        start, reader = todo.pop()
        src, path, _, _ = _relayout_path(start, (), None)
        on_path.extend(path)
        if path:
            reader = path[-1]

        if src.target in _FORK_OPS and len(src.users) == 1:
            on_path.append(src)
            todo.extend((a, src) for a in src.all_input_nodes)
            continue

        if not path and reader is node:
            continue  # already sitting on its producer

        new = _copy_quantize_above(model, node, src, ())
        reader.replace_input_with(src, new)
        moved = True

    if not moved:
        return False
    _annotate(on_path, node)
    node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)
    return True


_INDEX_COPY = torch.ops.aten.index_copy_.default


def _cache_write_below(node: Node):
    """``(index_copy_, prelude)`` for the KV-cache write ``node`` quantizes, or
    ``(None, [])``.  ``prelude`` is the ops between them, nearest ``node``
    first: a ``pad`` widening the cache to the array.  They apply to the cache
    as a whole, so they can be applied to the buffer and to the token written
    into it instead of to every position, every step."""
    prelude = []
    curr = node.args[0]
    while isinstance(curr, Node) and curr.target is torch.ops.aten.pad.default:
        prelude.append(curr)
        curr = curr.args[0]

    if (
        not isinstance(curr, Node)
        or curr.target is not _INDEX_COPY
        or not isinstance(curr.args[0], Node)
        or curr.args[0].op != "get_attr"
    ):
        return None, []
    return curr, prelude


def _replay(target_fn, value, ops):
    """``ops`` (nearest-consumer first) applied to ``value``, oldest first."""
    for op in reversed(ops):
        value = target_fn(op, value)
    return value


_MATMUL_MX = torch.ops.quantized_ops.matmul_mx.default


def _cone_to_matmul(idx: Node, node: Node):
    """``(nodes, matmul)`` -- the cone from cache write ``idx`` through quantize
    ``node`` and the relayout ops under it (``repeat_kv``, once for the value
    and once for the scale), down to the one ``matmul_mx`` reading it.  Both
    ``(None, None)`` if it is not that shape.

    Dead code has to be gone first: the hoist leaves the quantize it lifted
    behind, still reading this cone, and the walk would stop on it.
    """
    cone, frontier, matmuls = {idx: None, node: None}, [node], set()
    while frontier:
        for user in frontier.pop().users:
            if user in cone:
                continue
            cone[user] = None
            if user.target is _MATMUL_MX:
                matmuls.add(user)
            elif _is_relayout(user) or user.target is operator.getitem:
                frontier.append(user)
            else:
                return None, None
    if len(matmuls) != 1:
        return None, None
    return list(cone), next(iter(matmuls))


def _split_quantized_cache(
    model: GraphModule, node: Node, idx: Node, context_len: int, max_gen: int
) -> bool:
    """Split a cache the quantize blocks along the written axis in two, and
    read each half with its own GEMV.

    ``_fold_quantize_into_cache`` cannot bake such a cache: the token lands
    mid-block, whose scale depends on tokens not written yet.  But every block
    below the write is final, so the prefix under ``context_len`` is baked once
    now, and only the residual -- the write's block and the generation slots
    after it -- is re-quantized each step, at a cost that no longer grows with
    the context.

    The halves are never rejoined -- that concat is the traffic being removed.
    The GEMV reads them as two partial sums instead: the split axis is the one
    it reduces over, so summing the two halves' results reconstructs the dot
    product.

    Args:
        model: The graph module being lowered; edited in place.
        node: The ``quantize_mx`` hoisted onto the cache write, blocking
            along the axis the write indexes.
        idx: The ``index_copy_`` cache write feeding ``node``.
        context_len: Positions already written in the exported cache -- the
            prefix that can be baked quantized.
        max_gen: Generation slots that follow, sizing the residual window.

    Returns:
        ``True`` if the cache was split; ``False`` if the cone below ``node``
        is not a single cache -> GEMV, or the prefix is under one block so
        there is nothing to bake.

    Raises:
        RuntimeError: The cache's exported length along the written axis does
            not match ``context_len + max_gen`` rounded up to ``block_size``
            -- the compiler was told a shape the graph was not exported with.
    """

    cone, matmul = _cone_to_matmul(idx, node)
    if cone is None:
        logger.debug(f"Skip splitting {node}: not a cache -> GEMV cone.")
        return False
    cache = idx.args[0]
    rank = len(node.args[0].value.shape)
    dim = idx.args[1] % rank
    block_size = node.args[3]

    contents = fetch_attr(model, cache.target)
    cache_len = contents.shape[dim]
    expect = -(-(context_len + max_gen) // block_size) * block_size
    if cache_len != expect:
        raise RuntimeError(
            f"KV split: {cache.target} holds {cache_len} positions along dim "
            f"{dim}, but context_len={context_len} + max_gen={max_gen} rounded "
            f"up to the {block_size} block is {expect} -- the compiler was "
            f"told a different shape than the graph was exported with"
        )

    # Round *down*: a position at or past ``context_len`` is unwritten, so a
    # block straddling it is not final and belongs to the residual.
    split = context_len // block_size * block_size
    if split == 0:
        return False

    q_args = node.args[1:]
    consts = [
        fetch_attr(model, a.target) if isinstance(a, Node) else a
        for a in q_args
    ]
    main_scale, main_value = torch.ops.quantized_ops.quantize_mx(
        contents.narrow(dim, 0, split), *consts
    )
    residual = contents.narrow(dim, split, cache_len - split).clone()

    # The quantize's tensor arguments arrive as placeholders, in the order
    # ``all_input_nodes`` lists them; the rest are constants, baked at export.
    template = [None if isinstance(a, Node) else a for a in q_args]
    slots = [i for i, a in enumerate(q_args) if isinstance(a, Node)]

    # One KV head feeds ``groups`` query heads; the scale carries one element
    # per block where the value carries one per position.
    groups = matmul.args[1].value.shape[1] // contents.shape[1]
    blocks = split // block_size

    # The GEMV's kwargs, rebuilt in the order it had them: each half brings its
    # own two scales, a code table is handed through untouched (both halves are
    # the same dtype, so they read the same table), the rest are constants.
    mm_order = list(matmul.kwargs)
    mm_consts = {
        k: v for k, v in matmul.kwargs.items() if not isinstance(v, Node)
    }
    code_names = [
        k
        for k, v in matmul.kwargs.items()
        if isinstance(v, Node) and k not in ("input_scale", "weight_scale")
    ]

    def mx_kwargs(input_scale, weight_scale, codes):
        named = {
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            **mm_consts,
            **dict(zip(code_names, codes)),
        }
        return {k: named[k] for k in mm_order}

    def repeat(x):
        b, h, s, d = x.shape
        return (
            x.unsqueeze(2)
            .expand(b, h, groups, s, d)
            .reshape(b, h * groups, s, d)
        )

    base = cache.target.replace(".", "_")
    dtypes = node.meta.get("dtype", (None, None))
    parts = {
        base + "_scale": (main_scale, dtypes[0]),
        base + "_full": (main_value, dtypes[_MX_VALUE]),
        base + "_residual": (residual, None),
    }
    scale_name, value_name, residual_name = parts

    class SplitCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, (tensor, _) in parts.items():
                self.register_buffer(name, tensor)

        def forward(self, cache, index, token, *rest):
            n = len(slots)
            tensors, probs, probs_scale = rest[:n], rest[n], rest[n + 1]
            codes = rest[n + 2 :]
            args = list(template)
            for slot, tensor in zip(slots, tensors):
                args[slot] = tensor

            written = getattr(self, residual_name).index_copy_(
                dim, index - split, token
            )
            res_scale, res_value = torch.ops.quantized_ops.quantize_mx(
                written, *args
            )
            main = torch.ops.quantized_ops.matmul_mx(
                probs[..., :split],
                repeat(getattr(self, value_name)),
                **mx_kwargs(
                    probs_scale[..., :blocks],
                    repeat(getattr(self, scale_name)),
                    codes,
                ),
            )
            residue = torch.ops.quantized_ops.matmul_mx(
                probs[..., split:],
                repeat(res_value),
                **mx_kwargs(
                    probs_scale[..., blocks:], repeat(res_scale), codes
                ),
            )
            return main + residue

    new_node = create_and_insert_subgraph(cone, model)
    example = tuple(n.value.clone() for n in new_node.all_input_nodes)
    gm = get_aten_graph_module(SplitCache(), example)
    value_remap = {}
    outs = replace_node_with_graph_module(model, new_node, gm, value_remap)

    for out in outs:
        out.meta["dtype"] = matmul.meta.get("dtype")
    for n in gm.graph.nodes:
        new = value_remap.get(n)
        if new is None or n.op == "placeholder":
            continue
        if n.op == "get_attr" and n.target in parts:
            new.meta["dtype"] = parts[n.target][1]
        elif n.target is _QUANTIZE_MX:
            new.meta["dtype"] = dtypes
        elif n.target is operator.getitem:
            new.meta["dtype"] = dtypes[n.args[1]]
        elif n.target is _MATMUL_MX:
            new.meta["dtype"] = matmul.meta.get("dtype")
        elif _is_relayout(new):
            new.meta["dtype"] = new.args[0].meta.get("dtype")

    model.graph.erase_node(new_node)
    delattr(model, new_node.target)
    logger.info(
        f"Split {cache.target} at {split}: {split} positions baked quantized, "
        f"{cache_len - split} re-quantized per step; one GEMV each"
    )
    return True


def _fold_quantize_into_cache(
    model: GraphModule, node: Node, context_len: int, max_gen: int
) -> bool:
    """Fold a ``quantize_mx`` over a KV cache into the cache itself.

    The write puts one token in; the quantize then sweeps all of it, every
    step.  Since each token's blocks are its own, quantizing at write time is
    the same arithmetic -- so the buffer is baked already quantized and the
    quantize moves onto the token the write carries.  ``quantize_mx`` returns a
    pair, so the one cache buffer becomes two (values and scales), each with
    its own write.  A ``pad`` above the write folds in the same way: the buffer
    is baked wide, and only the token still pays for it.

    Only when the blocked axis is not the one the write indexes.  Otherwise a
    token lands mid-block and its block's scale depends on tokens not yet
    written -- that needs a KIVI-style residual window, not this.
    """

    graph = model.graph
    idx, prelude = _cache_write_below(node)
    if idx is None:
        return False

    outs = {}
    for user in node.users:
        if user.target is not operator.getitem:
            return False
        outs[user.args[1]] = user

    rank = len(node.args[0].value.shape)
    dim = idx.args[1] % rank
    if any(a % rank == dim for a in node.args[2]):
        # A token lands mid-block, so the cache cannot be baked whole -- but
        # the blocks below the write can be.  Split it instead.
        if prelude or context_len is None or max_gen is None:
            logger.debug(f"Skip folding {node}: blocked along written axis.")
            return False
        return _split_quantized_cache(model, node, idx, context_len, max_gen)

    cache = idx.args[0]
    q_args = node.args[1:]

    # Bake the cache: the same pad + quantize, run once, on its contents.
    baked = _replay(
        lambda op, v: op.target(v, *op.args[1:]),
        fetch_attr(model, cache.target),
        prelude,
    )
    consts = [
        fetch_attr(model, a.target) if isinstance(a, Node) else a
        for a in q_args
    ]
    scale, value = torch.ops.quantized_ops.quantize_mx(baked, *consts)

    with graph.inserting_before(node):
        buffers = {
            0: create_getattr_from_value(
                model, graph, cache.target + "_scale", scale
            ),
            _MX_VALUE: create_getattr_from_value(
                model, graph, cache.target + "_full", value
            ),
        }
        for buffer in buffers.values():
            propagate_shape(buffer, model)

        # The token pays for the pad and the quantize now, in its own right.
        token = _replay(
            lambda op, v: graph.call_function(op.target, (v, *op.args[1:])),
            idx.args[3],
            prelude,
        )
        for n in [token] if token is not idx.args[3] else []:
            propagate_shape(n, model)

        new_q = graph.call_function(node.target, (token, *q_args))
        new_q.meta = {
            k: copy.deepcopy(v) if k != "val" else v
            for k, v in node.meta.items()
            if k != "val"
        }
        propagate_shape(new_q, model)

    # One write per output, into the buffer that now holds it.
    for i, old in outs.items():
        with graph.inserting_before(node):
            part = graph.call_function(operator.getitem, (new_q, i))
            part.meta["dtype"] = old.meta.get("dtype", None)
            propagate_shape(part, model)
            written = graph.call_function(
                _INDEX_COPY, (buffers[i], idx.args[1], idx.args[2], part)
            )
            written.meta["dtype"] = old.meta.get("dtype", None)
            propagate_shape(written, model)
        buffers[i].meta["dtype"] = old.meta.get("dtype", None)
        old.replace_all_uses_with(written)

    # The old write mutates a buffer nothing reads now, but ``index_copy_`` is
    # side-effecting, so dead-code elimination will not collect it -- nor the
    # cache it keeps alive.  Erase the cone by hand, users first.
    for n in [*outs.values(), node, *prelude, idx, cache]:
        if not n.users:
            graph.erase_node(n)
    return True


def _hoist_microscaling(model: GraphModule, node: Node) -> bool:
    """Lift a ``quantize_mx`` over the relayout ops feeding it, so it quantizes
    the tensor they re-address rather than the one they hand on.

    Those ops move no element, so quantizing above them is the same arithmetic
    on less data -- and what they were going to do (broadcast a KV head, lay a
    tile out for the MXU) the consumer folds into its addressing rather than
    materializing.  Two things halt the walk: an op the quantization blocks do
    not survive (``_axes_above``), and an MHA head permute, where the quantize
    wants to stop -- the GEMM below stores straight through that permute, and
    the quantize fuses onto the end of it (``fuse_reshape_with_output``).

    The op has two outputs, so the value keeps the relayout ops it was lifted
    over and the scale gets a replayed copy of them.  It is never forked: a
    concat can move the axis it blocks along, and head splitting -- the only
    thing that forks -- never runs where microscaling is used.
    """

    graph = model.graph
    outs = {}
    for user in node.users:
        if user.target is not operator.getitem:
            return False
        outs[user.args[1]] = user

    block_size = node.args[3]
    src, path, axes_at, src_axes = _relayout_path(
        node.args[0], node.args[2], block_size, keep_head_permute=True
    )
    if not path:
        return False

    # TODO: we only move quantize_mx when there is a fusable anchor or the
    # op/param gets repeated in the memory. However this check is not robust.
    # In the future we should move this into operator fusion.
    if (
        not is_compute_op(src)
        and not is_mha_qkv_permute(src)
        and not any(n.target in _BROADCAST_OPS for n in path)
    ):
        logger.debug(f"Skip moving {node} because there is no fusable anchor.")
        return False

    new = _copy_quantize_above(model, node, src, src_axes)

    def unpack(i: int) -> Node:
        """Output ``i`` of the hoisted quantize.  ``quantize_mx``'s ``dtype`` is
        the *pair* it returns, so each output takes its own element of it -- the
        one the ``getitem`` it replaces carried."""
        out = graph.call_function(operator.getitem, (new, i))
        out.meta["dtype"] = outs[i].meta.get("dtype", None)
        propagate_shape(out, model)
        return out

    # The value keeps the relayout ops it was lifted over: rewire them onto it.
    with graph.inserting_before(path[-1]):
        value = unpack(_MX_VALUE)
    path[-1].replace_input_with(src, value)
    _annotate(path, outs[_MX_VALUE])
    outs[_MX_VALUE].replace_all_uses_with(path[0])

    # The scale is one element per block, so it needs its own copy of them.
    for i, old in outs.items():
        if i == _MX_VALUE:
            continue
        with graph.inserting_before(node):
            cur = unpack(i)
            for k in reversed(range(len(path))):
                cur = _replay_relayout(
                    graph, path[k], cur, axes_at[k], block_size
                )
                cur.meta["dtype"] = old.meta.get("dtype", None)
                propagate_shape(cur, model)
        old.replace_all_uses_with(cur)

    return True


def fuse_quantize_dequantize_with_previous_op(
    model: GraphModule,
    context_len: Optional[int] = None,
    max_gen: Optional[int] = None,
):
    """Move each quantize / dequantize up the graph to sit directly after the
    op that computed its input, so the two can fuse into one kernel.

    Everything it is lifted over only relayouts data -- a reshape, a transpose,
    the ``stack`` MHA splitting leaves behind, the ``expand`` of GQA's
    ``repeat_kv`` -- so quantizing above them is the same arithmetic on less
    data.  A microscaling ``quantize_mx`` also blocks along an axis and returns
    a scale beside its value, so it takes the ``_hoist_microscaling`` route;
    the rest share the walk but fork over a concat.

    Args:
        model: The graph module to rewrite in place.
        context_len: Positions already written in the decode cache the graph
            was exported with.  ``None`` outside decode.
        max_gen: Generation slots that follow those positions.  ``None``
            outside decode.  With ``context_len`` it lets a quantize that
            blocks along the written axis split the cache rather than sweep it
            (``_split_quantized_cache``).

    Returns:
        ``model``, rewritten in place.
    """
    graph = model.graph

    for node in list(graph.nodes):
        if node.target not in _QUANTIZE_OPS:
            continue
        if node.target is _QUANTIZE_MX:
            _hoist_microscaling(model, node)
            continue
        # A blocked plain quantize would need the same axis bookkeeping as
        # quantize_mx, which it does not have; only per-tensor is lifted.
        block_size = get_arg_value(node, 4, "block_size")
        if block_size is not None and block_size > 1:
            continue
        _hoist_forked(model, node)

    graph.eliminate_dead_code()
    for node in list(graph.nodes):
        if node.target is _QUANTIZE_MX:
            _fold_quantize_into_cache(model, node, context_len, max_gen)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()

    return model


def run_through_ops(model, input, nodes):
    env = {nodes[0].args[0]: input}

    def map_node(n):
        if n.op == "get_attr":
            return fetch_attr(model, n.target)
        return env[n]

    def load_arg(a):
        return torch.fx.graph.map_arg(a, map_node)

    for n in nodes:
        env[n] = n.target(*load_arg(n.args), **load_arg(n.kwargs))
    return env[nodes[-1]]


def validate_and_map_group_axes_for_reshape(old_shape, new_shape, axes):
    """
    Check if a reshape preserves group membership for arbitrary group axes.
    Returns True if safe, else False.
    """
    axes = tuple(sorted(axes))
    groups = [
        old_shape[i:j] for i, j in zip((0,) + axes, axes + (len(old_shape),))
    ]
    block_size = [math.prod(g) for g in groups]

    numel = 1
    idx = 0
    new_dims = []
    for i, s in enumerate(new_shape):
        numel *= s
        if numel == block_size[idx]:
            numel = 1
            idx += 1
            new_dims.append(i + 1)
            if idx == len(block_size):
                if (
                    i < len(new_shape) - 1
                    and math.prod(new_shape[i + 1 :]) != 1
                ):
                    logger.warning("Extra trailing dimensions after last group")
                    return None
                break
        elif numel > block_size[idx]:
            logger.warning(f"Overshot group {idx} at new axis {i}")
            return None

    if idx != len(block_size):
        logger.warning("Not all groups matched")
        return None

    return new_dims[:-1]


def propagate_group_axes_through_op(node, input, axes, block_size):
    """
    Track which axes correspond to group-wise quantization through layout ops.

    Args:
        node (torch.fx.Node): layout op node
        input (torch.Tensor): tensor before layout op
        axes (tuple[int]): axes where grouping/quantization is performed
        block_size (int): size of quantization blocks along grouped axes

    Returns:
        tuple[int]: new axes for grouping after transformations
    Raises:
        RuntimeError: if reshape or any op makes grouping ambiguous
    """
    axes = list(axes)
    tgt = node.target

    if tgt == torch.ops.aten.unsqueeze.default:
        dim = int(node.args[1])
        axes = [a + 1 if a >= dim else a for a in axes]
        output = tgt(input, dim)
    elif tgt == torch.ops.aten.slice.Tensor:
        default = [0, 0, 9223372036854775807, 1]
        dim, start, end, step = (
            list(node.args[1:]) + default[len(node.args) - 1 :]
        )
        if dim in axes:
            start, end = int(start / block_size), int(end / block_size)
        args = (dim, start, end, step)
        output = tgt(input, *args)
    elif tgt == torch.ops.aten.expand.default:
        size = [
            math.ceil(s / block_size) if d in axes else s
            for d, s in enumerate(node.args[1])
        ]
        output = tgt(input, size)
    elif tgt == torch.ops.aten.transpose.int:
        a0, a1 = node.args[1:3]
        axes = [a1 if a == a0 else a0 if a == a1 else a for a in axes]
        output = tgt(input, a0, a1)
    elif tgt == torch.ops.aten.permute.default:
        perm = node.args[1]
        axes = [perm.index(a + input.ndim if a < 0 else a) for a in axes]
        output = tgt(input, perm)
    elif tgt in (torch.ops.aten.reshape.default, torch.ops.aten.view.default):
        orig_shape = [
            s * block_size if i in axes else s
            for i, s in enumerate(input.shape)
        ]
        axes = validate_and_map_group_axes_for_reshape(
            orig_shape, node.args[1], axes
        )
        if axes is None:
            raise RuntimeError("Invalid reshape")
        new_shape = [
            math.ceil(s / block_size) if d in axes else s
            for d, s in enumerate(node.args[1])
        ]
        output = tgt(input, new_shape)
    else:
        raise RuntimeError(f"Unsupported layout op: {tgt}")

    return output, tuple(axes)


LAYOUT_OPS = {
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.expand.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
}


def store_qparam_unrepeated(
    model, param, expand_node, name, insert_before, dtype
):
    """Store ``param`` once and repeat it in the graph, not in the buffer.

    Grouped-query attention shares one KV head between several query heads, so a
    quantization parameter run through that broadcast (``expand_node``) holds
    every value that many times over.  Rebuilt as ``unsqueeze -> expand ->
    reshape`` above the value it was quantized with, the copies never happen:
    the consumer folds the repeat into its addressing and reads head
    ``h // factor`` (``repeat_of``).

    Returns the node the ops end at, or ``None`` if ``param`` is not that repeat
    after all.
    """
    # The broadcast grows one dim, which the reshape under it then folds into
    # the dim above -- the head dim.
    grown = [
        d
        for d, s in enumerate(expand_node.args[1])
        if s != expand_node.args[0].shape[d]
    ]
    if len(grown) != 1:
        return None
    dim = grown[0] - 1
    factor = expand_node.args[1][grown[0]]

    graph = model.graph
    base = param.index_select(
        dim, torch.arange(0, param.shape[dim], factor, device=param.device)
    )
    if not torch.equal(param, base.repeat_interleave(factor, dim=dim)):
        return None

    sizes = list(base.shape)
    sizes.insert(dim + 1, factor)
    with graph.inserting_before(insert_before):
        attr = create_getattr_from_value(model, graph, name, base)
        unsqueezed = graph.call_function(
            torch.ops.aten.unsqueeze.default, (attr, dim + 1)
        )
        expanded = graph.call_function(
            torch.ops.aten.expand.default, (unsqueezed, sizes)
        )
        reshaped = graph.call_function(
            torch.ops.aten.reshape.default, (expanded, list(param.shape))
        )
    for n in (attr, unsqueezed, expanded, reshaped):
        n.meta["dtype"] = dtype
        propagate_shape(n, model)
    return reshaped


def run_qparam_through_nodes(model, input, nodes, axes, block_size):
    env = {nodes[0].args[0]: input}

    def map_node(n):
        if n.op == "get_attr":
            return fetch_attr(model, n.target)
        return env[n]

    def load_arg(a):
        return torch.fx.graph.map_arg(a, map_node)

    axes = tuple(a + input.ndim if a < 0 else a for a in axes)

    for n in nodes:
        if n.target in LAYOUT_OPS:
            env[n], axes = propagate_group_axes_through_op(
                n, env[n.args[0]], axes, block_size
            )
        else:
            env[n] = n.target(*load_arg(n.args), **load_arg(n.kwargs))
    return env[nodes[-1]], axes


def fuse_dequantize_quantize(model: torch.fx.GraphModule):
    """
    Fuses consecutive dequantize -> quantize operations in a quantized model
    for optimization.

    A broadcast qparam (GQA's ``repeat_kv``) is stored once per KV head rather
    than once per query head, with the repeat put back as graph ops: a smaller
    buffer for a longer graph, which pays off because the GEMM folds the repeat
    into its tile addressing instead of copying.

    Args:
        model (GraphModule): The FX-traced model to optimize.

    Returns:
        GraphModule: The optimized model with fused operations.
    """
    graph = model.graph
    for node in list(graph.nodes):
        if node.target not in (
            torch.ops.quantized_ops.quantize.default,
            torch.ops.quantized_ops.quantize_mx.default,
        ):
            continue

        # For quantize_mx, qparam is the first user node
        if node.target == torch.ops.quantized_ops.quantize_mx.default:
            scale_node = next(iter(node.users))
        else:
            scale_node = node.args[1]

        prev_node = node.args[0]
        nodes_on_path = [node]

        while len(prev_node.users) == 1:
            target = prev_node.target
            if not (
                is_nop(prev_node)
                or is_reshape_op(prev_node)
                or target
                in (
                    torch.ops.aten.expand.default,
                    torch.ops.aten.slice.Tensor,
                )
            ):
                break

            nodes_on_path.append(prev_node)
            prev_node = prev_node.args[0]

        # Only support fusing get_attr -> dq -> ops -> q pattern
        if (
            prev_node.target != torch.ops.quantized_ops.dequantize.default
            or prev_node.args[0].op != "get_attr"
        ):
            continue

        dq_node = prev_node
        nodes_on_path = [dq_node] + list(reversed(nodes_on_path))

        # Check block size compatibility
        if node.target == torch.ops.quantized_ops.quantize_mx.default:
            block_size = node.args[3]
        else:
            block_size = get_arg_value(node, 4, "block_size", 1)
        dq_block_size = get_arg_value(dq_node, 4, "block_size", 1)
        if block_size != dq_block_size:
            continue

        # Pre-compute the transformed scales and zero points
        dq_input = fetch_attr(model, dq_node.args[0].target)
        if node.target == torch.ops.quantized_ops.quantize_mx.default:
            q_scale = run_through_ops(model, dq_input, nodes_on_path)[0]
        else:
            q_scale = fetch_attr(model, scale_node.target)

        dq_axes = get_arg_value(dq_node, 3, "axes")
        dq_scale = fetch_attr(model, dq_node.args[1].target)
        dq_scale, new_dq_axes = run_qparam_through_nodes(
            model, dq_scale, nodes_on_path[1:-1], dq_axes, block_size
        )

        if len(dq_node.args) > 2:
            zero_point = fetch_attr(model, dq_node.args[2].target)
            zero_point, _ = run_qparam_through_nodes(
                model, zero_point, nodes_on_path[1:-1], dq_axes, block_size
            )

        output = run_through_ops(model, dq_input, nodes_on_path[:-1])
        rank = output.ndim
        dq_axes = tuple((a + rank) % rank for a in new_dq_axes)

        # quantize_mx puts axes at arg index 2 (index 3 is block_size);
        # plain quantize keeps axes at index 3.
        if node.target == torch.ops.quantized_ops.quantize_mx.default:
            q_axes = node.args[2]
        else:
            q_axes = get_arg_value(node, 3, "axes")
        q_axes = tuple((a + rank) % rank for a in q_axes)
        new_axes = tuple(set(q_axes) & set(dq_axes))

        # Broadcast scales to the same shape
        nd = max(dq_scale.ndim, q_scale.ndim)
        while dq_scale.ndim < nd:
            dq_scale = dq_scale.unsqueeze(0)
        while q_scale.ndim < nd:
            q_scale = q_scale.unsqueeze(0)
        shape = list(max(a, b) for a, b in zip(q_scale.shape, dq_scale.shape))

        q_scale_expanded = expand(q_scale, shape, block_size)
        dq_scale_expanded = expand(dq_scale, shape, block_size)
        fused_scale = dq_scale_expanded / q_scale_expanded

        # The qparams were run through the path, so a broadcast on it (GQA's
        # ``repeat_kv``) is baked into them: they hold each value once per query
        # head sharing a KV head.  Where the repeat is free in the graph -- it
        # folds into the tile's block index -- put it back and store the qparam
        # once.
        expand_node = next(
            (
                n
                for n in nodes_on_path[1:-1]
                if n.target is torch.ops.aten.expand.default
            ),
            None,
        )

        qparam_dtype = scale_node.meta.get("dtype")

        def create_qparam(value, name):
            if expand_node is not None:
                repeated = store_qparam_unrepeated(
                    model, value, expand_node, name, node, qparam_dtype
                )
                if repeated is not None:
                    return repeated
            with graph.inserting_before(node):
                attr = create_getattr_from_value(model, graph, name, value)
            attr.meta["dtype"] = qparam_dtype
            return attr

        input_node = dq_node.args[0]
        new_scale = create_qparam(fused_scale, input_node.name + "_scale")
        new_zero_point = (
            create_qparam(zero_point, input_node.name + "_zero_point")
            if len(dq_node.args) > 2
            else None
        )
        with graph.inserting_before(node):
            # qmap is at arg index 1 for quantize_mx, index 5 for plain
            # quantize.
            if node.target == torch.ops.quantized_ops.quantize_mx.default:
                output_qmap = graph.node_copy(node.args[1])
            else:
                output_qmap = graph.node_copy(node.args[5])
            new_dq = graph.call_function(
                torch.ops.quantized_ops.dequantize.default,
                (
                    node.args[0],
                    new_scale,
                    new_zero_point,
                    new_axes,
                    block_size,
                    None,
                    output_qmap,
                ),
            )

        if scale_node.op != "get_attr":
            if (
                any(is_gemm_op(n) for n in scale_node.users)
                and q_scale.shape[-1] != output.shape[-1]
            ):
                q_scale = torch.repeat_interleave(
                    q_scale,
                    repeats=output.shape[-1] // q_scale.shape[-1],
                    dim=-1,
                )

            mx_scale = create_qparam(q_scale, input_node.name + "_scale")
            scale_node.replace_all_uses_with(mx_scale)

        if node.target == torch.ops.quantized_ops.quantize_mx.default:
            value_getitem = next(
                u
                for u in node.users
                if u.target == operator.getitem and u.args[1] == 1
            )
            value_getitem.replace_all_uses_with(new_dq)
            new_dq.meta["dtype"] = node.meta["dtype"][1]
        else:
            node.replace_all_uses_with(new_dq)
            graph.erase_node(node)
            new_dq.meta["dtype"] = node.meta.get("dtype")

        dq_node.replace_all_uses_with(input_node)
        graph.erase_node(dq_node)

        for n in nodes_on_path[1:-1]:
            n.meta["dtype"] = input_node.meta.get("dtype")

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
