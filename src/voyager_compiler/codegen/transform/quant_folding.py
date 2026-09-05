"""Fold quantize / dequantize nodes into the ops around them.

Three passes.  ``split_kv_cache`` runs on the exported decode graph before
quantization: it splits each KV cache into the completed, block-aligned
chunks (which the quantizer then sees as one tensor) and a small
full-precision residual holding the chunk being filled.  The other two run
over an already-quantized graph and move a quantize or dequantize rather
than compute it:

  * ``fuse_quantize_dequantize_with_producer`` runs inside ``transform()``.
    It hoists a quantize into its producer (so a value is stored already
    narrow), replays it above a relayout, and folds one (a ``quantize_mx``
    or a group-wise ``quantize_affine``) into a KV cache write.
  * ``fuse_dequantize_quantize`` collapses a ``get_attr -> dequantize -> layout
    ops -> quantize`` chain into a single ``dequantize`` with pre-multiplied
    scales, storing a grouped-query parameter once rather than once per head.

All three work purely on the ``quantized_ops`` schema and FX, not on the
quantizer that produced the graph.
"""

import copy
import logging
import math
import operator
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from voyager_compiler.codegen.aten_classifier import is_compute_op
from voyager_compiler.codegen.node_info import (
    _BROADCAST_OPS,
    QUANTIZE_FAMILY_OPS,
    get_arg_value,
    is_gemm_op,
    is_mha_qkv_permute,
    is_nop,
    is_reshape_op,
    reshape_preserves_full_blocks,
)
from voyager_compiler.codegen.subgraph import get_new_node_name_with_prefix
from voyager_compiler.export_utils import create_getattr_from_value
from voyager_compiler.ops.quantized import expand
from voyager_compiler.shape_prop import (
    fetch_attr,
    propagate_shape,
    set_node_value,
)

logger = logging.getLogger(__name__)

__all__ = [
    "fuse_dequantize_quantize",
    "fuse_quantize_dequantize_with_producer",
    "sink_cache_folds",
    "split_kv_cache",
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

_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
_QUANTIZE_AFFINE = torch.ops.quantized_ops.quantize_affine.default
_QUANTIZE_MX_OUTLIER = torch.ops.quantized_ops.quantize_mx_outlier.default

# A dynamic quantize returns its qparams beside the value, the value last:
# ``quantize_mx`` gives ``(scale, value)``, ``quantize_affine`` ``(scale,
# zero_point, value)``.  The value inherits the relayout ops the quantize was
# lifted over; each qparam gets a replayed copy of them.  A cache baked through
# one becomes a buffer per output, named ``<cache>_<suffix>``.
_OUTPUT_NAMES = {
    _QUANTIZE_MX: ("scale", "full"),
    _QUANTIZE_AFFINE: ("scale", "zero_point", "full"),
}
_MX_VALUE = 1


def _value_index(op) -> int:
    return len(_OUTPUT_NAMES[op]) - 1


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
    axes = tuple(a - rank if a >= 0 else a for a in axes)

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
    order = {n: i for i, n in enumerate(graph.nodes)}
    for n in sorted(dict.fromkeys(on_path), key=order.__getitem__):
        propagate_shape(n, model)
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


_MATMUL = torch.ops.aten.matmul.default
_COND = torch.ops.higher_order.cond

# The KV-cache buffers of a Hugging Face static-cache decode export, and the
# residual buffers ``split_kv_cache`` puts beside them.
_KV_CACHE = re.compile(r"^(key|value)_cache_(\d+)$")


def _shape_of(node: Node):
    """``node``'s tensor shape, from ``ShapeProp``'s value or, on a graph
    fresh from export, the fake value export stamped."""
    value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        value = node.meta["val"]
    return tuple(value.shape)


def _axis_through(node: Node, axis: int):
    """Where input dim ``axis`` of relayout ``node`` lands in its output, or
    ``None`` when the op moves, merges or cuts it.  A quantize, dequantize,
    ``getitem`` or ``expand`` keeps every dim in place."""
    target = node.target
    if target is operator.getitem or target in QUANTIZE_FAMILY_OPS:
        return axis
    rank = len(_shape_of(node.args[0]))
    if target is torch.ops.aten.unsqueeze.default:
        dim = node.args[1] % (rank + 1)
        return axis + 1 if axis >= dim else axis
    if target is torch.ops.aten.transpose.int:
        a0, a1 = (d % rank for d in node.args[1:3])
        return a1 if axis == a0 else a0 if axis == a1 else axis
    if target is torch.ops.aten.permute.default:
        return [d % rank for d in node.args[1]].index(axis)
    if target in _REGROUP_OPS:
        mapped = validate_and_map_group_axes_for_reshape(
            _shape_of(node.args[0]), _shape_of(node), [axis]
        )
        return mapped[0] if mapped else None
    if target is torch.ops.aten.slice.Tensor and node.args[1] % rank == axis:
        return None
    return axis


def _attention_cone(idx: Node):
    """``(relayouts, matmul)``: the relayout ops from cache write ``idx`` down
    to the one ``aten.matmul`` that reads the cache as its right operand, in
    graph order; ``(None, None)`` if the write reaches anything else."""
    relayouts, frontier, matmuls, seen = [], [idx], [], {idx}
    while frontier:
        for user in frontier.pop().users:
            if user in seen:
                continue
            seen.add(user)
            if user.target is _MATMUL:
                matmuls.append(user)
            elif _is_relayout(user):
                relayouts.append(user)
                frontier.append(user)
            else:
                return None, None
    if len(matmuls) != 1 or matmuls[0].args[1] not in seen:
        return None, None
    order = {n: i for i, n in enumerate(idx.graph.nodes)}
    return sorted(relayouts, key=order.__getitem__), matmuls[0]


def _stamp_fake(node: Node, fake_mode) -> None:
    """Give a node built after export the ``meta["val"]`` export would have
    stamped: the op run on its operands' fake values, under the graph's own
    fake mode.  The quantizer's annotation and observer insertion read it."""
    if node.op == "get_attr":
        value = fetch_attr(node.graph.owning_module, node.target)
        node.meta["val"] = fake_mode.from_tensor(value, static_shapes=True)
        return

    def load(a):
        return map_arg(a, lambda n: n.meta["val"])

    with fake_mode:
        node.meta["val"] = node.target(*load(node.args), **load(node.kwargs))


def _copy_scope(node: Node, like: Node) -> None:
    """Stamp ``node`` with the module scope ``like`` was traced from, so the
    quantizer's scope- and order-based filters see it beside ``like``."""
    for key in ("nn_module_stack", "source_fn_stack"):
        if key in like.meta:
            node.meta[key] = like.meta[key]


def _split_one_cache(
    model: GraphModule,
    idx: Node,
    residual_length: int,
    context_len: int,
    fake_mode,
) -> None:
    """Split the cache written by ``idx`` (see ``split_kv_cache``)."""
    graph = model.graph
    cache, dim, index, token = idx.args
    length = residual_length
    relayouts, matmul = _attention_cone(idx)
    if matmul is None:
        raise ValueError(
            f"{cache.target}: the cache write does not reach a single "
            "attention matmul through relayout ops"
        )
    dim = dim % len(_shape_of(idx))

    # Follow the written axis down to the matmul's right operand: reaching
    # its last dim makes the residual's scores replace a window of the main
    # scores; reaching the reduction dim makes them add.
    axis_of = {idx: dim}
    for n in relayouts:
        src = next(a for a in n.all_input_nodes if a in axis_of)
        axis_of[n] = _axis_through(n, axis_of[src])
        if axis_of[n] is None:
            raise ValueError(
                f"{cache.target}: {n} loses the cache's position axis"
            )
    weight = matmul.args[1]
    weight_rank = len(_shape_of(weight))
    if axis_of[weight] == weight_rank - 1:
        join = "scores"
    elif axis_of[weight] == weight_rank - 2:
        join = "sum"
    else:
        raise ValueError(
            f"{cache.target}: position axis {axis_of[weight]} of the "
            "attention matmul's operand is neither its output nor its "
            "reduction axis"
        )

    # Split the contents: completed chunks stay, zeros above them; the tail
    # moves into the residual buffer.
    contents = fetch_attr(model, cache.target)
    cache_len = contents.shape[dim]
    if cache_len % length:
        raise ValueError(
            f"{cache.target}: {cache_len} positions is not a multiple of "
            f"the residual length {length}"
        )
    done = context_len // length * length
    residual = torch.zeros_like(contents.narrow(dim, 0, length))
    if context_len > done:
        residual.narrow(dim, 0, context_len - done).copy_(
            contents.narrow(dim, done, context_len - done)
        )
    contents.narrow(dim, done, cache_len - done).zero_()

    # The write lands in the residual at ``p mod R``; the attention's main
    # half reads the cache buffer directly.  The residual keeps the cache's
    # name as its prefix, which is what a traffic report files it under.
    with graph.inserting_before(idx):
        residual_attr = create_getattr_from_value(
            model, graph, f"{cache.target}_residual", residual
        )
        _stamp_fake(residual_attr, fake_mode)
        slot = graph.call_function(
            torch.ops.aten.remainder.Scalar, (index, length)
        )
        chunk = graph.call_function(
            torch.ops.aten.floor_divide.default, (index, length)
        )
        base = graph.call_function(torch.ops.aten.mul.Tensor, (chunk, length))
        arange = graph.call_function(
            torch.ops.aten.arange.default,
            (length,),
            {"dtype": torch.int64, "device": _shape_device(index)},
        )
        offsets = graph.call_function(torch.ops.aten.add.Tensor, (base, arange))
        pred = graph.call_function(torch.ops.aten.eq.Scalar, (slot, length - 1))
        for n in (slot, chunk, base, arange, offsets, pred):
            _copy_scope(n, idx)
            _stamp_fake(n, fake_mode)
    idx.replace_all_uses_with(cache)
    idx.update_arg(0, residual_attr)
    idx.update_arg(2, slot)
    _stamp_fake(idx, fake_mode)

    # The residual half: the same relayouts and matmul, over the residual's
    # ``R`` positions, right after the main matmul so the quantizer counts
    # them main, residual, main, residual within the layer.  The main half
    # now reads the cache where it read the write.  A matmul that read the
    # write directly (no relayout: multi-head attention's values) keeps it.
    remap = {cache: idx, idx: idx}
    with graph.inserting_before(matmul.next):
        for n in relayouts:
            new = graph.node_copy(n, lambda a: remap[a])
            if n.target in _REGROUP_OPS:
                shape = list(_shape_of(n))
                shape[axis_of[n]] = length
                new.args = (new.args[0], shape)
            elif n.target is torch.ops.aten.expand.default:
                in_shape = _shape_of(n.args[0])
                new.args = (
                    new.args[0],
                    [
                        o if o != i else -1
                        for o, i in zip(_shape_of(n), in_shape)
                    ],
                )
            _stamp_fake(new, fake_mode)
            remap[n] = new
        if join == "scores":
            lhs = matmul.args[0]
        else:
            lhs = graph.call_function(
                torch.ops.aten.index_select.default,
                (matmul.args[0], -1, offsets),
            )
            _copy_scope(lhs, matmul)
            _stamp_fake(lhs, fake_mode)
        residual_matmul = graph.node_copy(matmul, lambda a: remap.get(a, a))
        residual_matmul.args = (lhs, remap[weight])
        _stamp_fake(residual_matmul, fake_mode)
        if join == "scores":
            joined = graph.call_function(
                torch.ops.aten.index_copy.default,
                (matmul, -1, offsets, residual_matmul),
            )
        else:
            joined = graph.call_function(
                torch.ops.aten.add.Tensor, (matmul, residual_matmul)
            )
        _copy_scope(joined, matmul)
        _stamp_fake(joined, fake_mode)
    matmul.replace_all_uses_with(
        joined, delete_user_cb=lambda u: u is not joined
    )

    # The fold, after the attention has read the main cache: on the step
    # that completes a chunk, copy the residual into it.  Spelled without a
    # branch -- chunk ``c`` is rewritten every step, with the residual on
    # the completing step and with what it already holds otherwise -- so
    # the graph stays plain tensor ops (a CUDA graph can replay it) and the
    # lowering turns the pattern into a real conditional.  The cache is
    # reached through its own ``get_attr`` so that an observer on the read
    # never sits between the fold and the buffer.
    with graph.inserting_before(joined.next):
        handle = graph.get_attr(cache.target)
        _stamp_fake(handle, fake_mode)
        window = graph.call_function(
            torch.ops.aten.index_select.default, (handle, dim, offsets)
        )
        folded = graph.call_function(
            torch.ops.aten.where.self, (pred, idx, window)
        )
        fold = graph.call_function(_INDEX_COPY, (handle, dim, offsets, folded))
        for n in (window, folded, fold):
            _copy_scope(n, idx)
            _stamp_fake(n, fake_mode)
    logger.info(
        f"Split {cache.target}: {done} positions in completed chunks, "
        f"{context_len - done} in the {length}-slot residual; residual "
        f"{'replaces a window of' if join == 'scores' else 'adds to'} the "
        "main attention"
    )


def _shape_device(node: Node):
    value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        value = node.meta["val"]
    return value.device


def split_kv_cache(
    model: GraphModule, residual_length: int, context_len: int
) -> int:
    """Split each KV cache of an exported decode graph into its completed
    chunks and a full-precision residual.

    The main cache buffer keeps only whole chunks of ``residual_length``
    positions, zeros above them; a residual buffer of ``residual_length``
    slots, in the cache's dtype, holds the chunk being filled, the token at
    position ``p`` in slot ``p mod residual_length``.  The attention reads
    both: the residual's scores replace the main scores at the chunk's
    positions (``q @ K^T``) and its values add to the main ones (``P @ V``,
    over the same window of ``P``).  After the attention has read the
    cache, chunk ``p // residual_length`` of the main cache is rewritten:
    with the residual on the step that fills its last slot, with what it
    already holds otherwise.  A quantizer then annotates the main cache's
    read and leaves the residual alone; the lowering turns the rewrite
    into a conditional store of the quantized chunk.

    Args:
        model: The decode graph from ``convert_and_export_with_cache``, its
            cache buffers loaded with ``context_len`` positions; rewritten
            in place.
        residual_length: Positions per chunk, a multiple of the quantizer's
            block size along the sequence.
        context_len: Positions already written in each cache.

    Returns:
        The number of caches split.

    Raises:
        ValueError: A cache is not a multiple of ``residual_length`` long, or
            its write does not reach a single attention matmul.
    """
    graph = model.graph
    fake_mode = next(
        n.meta["val"].fake_mode
        for n in graph.nodes
        if isinstance(n.meta.get("val"), FakeTensor)
    )
    count = 0
    for idx in list(graph.nodes):
        if (
            idx.target is not _INDEX_COPY
            or not isinstance(idx.args[0], Node)
            or idx.args[0].op != "get_attr"
            or _KV_CACHE.match(str(idx.args[0].target)) is None
        ):
            continue
        _split_one_cache(model, idx, residual_length, context_len, fake_mode)
        count += 1
    graph.lint()
    model.recompile()
    return count


def _fold_quantize_into_cache(model: GraphModule, node: Node) -> bool:
    """Fold a dynamic quantize over a KV cache write into the cache itself.

    The write puts one token in; the quantize then sweeps all of it, every
    step.  Since each token's blocks are its own, quantizing at write time is
    the same arithmetic -- so the buffer is baked already quantized and the
    quantize moves onto the token the write carries.  The quantize returns its
    qparams beside the value, so the one cache buffer becomes one per output
    (``_OUTPUT_NAMES``), each with its own write.  A ``pad`` above the write
    folds in the same way: the buffer is baked wide, and only the token still
    pays for it.

    Only when the blocked axis is not the one the write indexes.  Otherwise a
    token lands mid-block and its block's qparams depend on tokens not yet
    written -- that is what ``split_kv_cache``'s residual is for.
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
        # Blocked along the written axis a token lands mid-block, so the
        # cache cannot be baked whole.
        logger.debug(f"Skip folding {node}: blocked along the written axis.")
        return False

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
    baked_outputs = node.target(baked, *consts)

    with graph.inserting_before(node):
        buffers = {
            i: create_getattr_from_value(
                model, graph, f"{cache.target}_{name}", baked_outputs[i]
            )
            for i, name in enumerate(_OUTPUT_NAMES[node.target])
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


_QUANTIZE = torch.ops.quantized_ops.quantize.default


class _FoldBranch:
    """The branches of a split cache's fold, lowered to a ``torch.cond``.

    On a chunk-completing step the ``true`` branch quantizes the residual
    and stores the outputs into the baked cache buffers at the chunk's own
    entries; the ``false`` branch is empty.  Both take the same operands,
    positionally: a top-level node the ``true`` branch reads gets a
    placeholder in each (``placeholder``).  Nodes are built in the ``true``
    graph ahead of its output and shape-propagated as they are made, on
    copies of the operands' values, so a store executed during propagation
    never reaches the module's buffers.
    """

    def __init__(self, model: GraphModule, cache: str, chunk_index: Node):
        self.model = model
        self.cache = cache
        self.chunk_index = chunk_index
        self.operands = []
        self.placeholders = {}
        self.offsets = {}
        self.cond = None
        self.true = GraphModule(torch.nn.Module(), Graph())
        self.false = GraphModule(torch.nn.Module(), Graph())
        self.outputs = [g.graph.output((0,)) for g in (self.true, self.false)]
        self.last_placeholder = [None, None]
        self.handles = []
        for prefix, branch in (
            ("true_graph", self.true),
            ("false_graph", self.false),
        ):
            name = get_new_node_name_with_prefix(prefix)(model)
            setattr(model, name, branch)
            self.handles.append(name)

    @property
    def graph(self) -> Graph:
        return self.true.graph

    def placeholder(self, node: Node) -> Node:
        """``node``, a top-level node, as the ``true`` branch reads it."""
        if node not in self.placeholders:
            self.operands.append(node)
            made = []
            for k, branch in enumerate((self.true, self.false)):
                anchor = self.last_placeholder[k]
                with (
                    branch.graph.inserting_after(anchor)
                    if anchor is not None
                    else branch.graph.inserting_before(self.outputs[k])
                ):
                    made.append(branch.graph.placeholder(node.name))
                made[k].meta["dtype"] = node.meta.get("dtype")
                set_node_value(made[k], node.value)
                self.last_placeholder[k] = made[k]
            self.placeholders[node] = made
        return self.placeholders[node][0]

    def arguments(self, args) -> tuple:
        """``args`` as the branch takes them: a top-level node becomes its
        placeholder, anything else passes as is."""
        return tuple(
            self.placeholder(a) if isinstance(a, Node) else a for a in args
        )

    def call(self, target, args, dtype=None, kwargs=None) -> Node:
        """A node computed in the ``true`` branch, stamped with its value."""
        with self.graph.inserting_before(self.outputs[0]):
            node = self.graph.call_function(target, args, kwargs)
        return self.stamp(node, dtype)

    def copy(self, template: Node, source: Node) -> Node:
        """``template``, a top-level relayout, copied into the ``true``
        branch onto ``source``.  The caller restates its arguments for
        ``source``'s extents and then stamps it."""
        with self.graph.inserting_before(self.outputs[0]):
            node = self.graph.node_copy(
                template, lambda a: source if a is template.args[0] else a
            )
        node.meta = {}
        return node

    def stamp(self, node: Node, dtype) -> Node:
        node.meta["dtype"] = dtype
        propagate_shape(node, self.true)
        return node

    def store(self, source: Node, buffer: Node, axis: int) -> Node:
        """Store ``source``, a chunk-sized branch value, into ``buffer``, a
        baked cache buffer, at the chunk's own entries along ``axis``:
        ``index_copy_`` at the ``span`` entries from ``chunk_index * span``,
        ``span`` being the entries a chunk covers there."""
        span = source.value.shape[axis]
        if span not in self.offsets:
            start = self.call(
                operator.mul, (self.placeholder(self.chunk_index), span)
            )
            end = self.call(operator.add, (start, span))
            self.offsets[span] = self.call(
                torch.ops.aten.arange.start_step,
                (start, end, 1),
                dtype=None,
                kwargs={
                    "dtype": torch.int64,
                    # The buffer itself: ShapeProp's value may be a copy
                    # elsewhere.
                    "device": fetch_attr(self.model, buffer.target).device,
                },
            )
        return self.call(
            _INDEX_COPY,
            (self.placeholder(buffer), axis, self.offsets[span], source),
            dtype=buffer.meta.get("dtype"),
        )

    def unstore(self, write: Node) -> None:
        """Remove a store, and its offsets once nothing else reads them."""
        self.graph.erase_node(write)
        for span, offsets in list(self.offsets.items()):
            if offsets.users:
                continue
            start, end = offsets.args[:2]
            for n in (offsets, end, start):
                self.graph.erase_node(n)
            del self.offsets[span]

    def place(self, predicate, before: Node) -> None:
        """Put the ``cond`` running the branches on ``predicate`` into the
        top-level graph, before ``before``."""
        graph = self.model.graph
        with graph.inserting_before(before):
            handles = [graph.get_attr(name) for name in self.handles]
            for handle in handles:
                propagate_shape(handle, self.model)
            self.cond = graph.call_function(_COND, (predicate, *handles, ()))
        # Scoped like a bufferized nest, so its branch nodes get model-wide
        # unique names and it lands in the layer table as the cache's fold.
        self.cond.meta["scope"] = (f"{self.cache}_fold", None)
        set_node_value(self.cond, (0,))
        self.bind()

    def bind(self) -> None:
        """Hand the ``cond`` the operands the branches take now, dropping
        a placeholder nothing reads any more."""
        for node, made in list(self.placeholders.items()):
            if made[0].users:
                continue
            for k, branch in enumerate((self.true, self.false)):
                if self.last_placeholder[k] is made[k]:
                    self.last_placeholder[k] = made[k].prev
                    if self.last_placeholder[k].op != "placeholder":
                        self.last_placeholder[k] = None
                branch.graph.erase_node(made[k])
            del self.placeholders[node]
            self.operands.remove(node)
        self.cond.args = (*self.cond.args[:3], tuple(self.operands))


@dataclass
class _Fold:
    """A split KV cache's fold, as the lowering rebuilds it.

    ``split_kv_cache`` spells the fold as a masked write-back of chunk
    ``p // R`` of the residual into the main cache.  The lowering keeps
    the main cache baked in its quantized form and turns the write-back
    into ``branch``, a ``torch.cond`` on the completing-step predicate
    whose ``true`` branch quantizes the residual and stores the outputs
    into the baked buffers.  ``parts`` are the branch-side outputs of that
    quantize, by output index; ``stores`` the branch's ``index_copy_`` of
    each into its baked buffer, by output index, until a re-encode replaces
    it; ``scale_qmap`` the table that quantize rounds its scale through (a
    top-level ``get_attr``), or ``None`` when the qparams keep the cache's
    dtype.
    """

    cache: str
    dim: int
    branch: _FoldBranch
    parts: dict
    stores: dict
    scale_qmap: Optional[Node]


def _fold_write(graph: Graph, target: str) -> Optional[Node]:
    """The masked write-back ``split_kv_cache`` left on cache ``target``:
    ``index_copy_(cache, dim, offsets, where(pred, residual, window))``."""
    for n in graph.nodes:
        if (
            n.target is _INDEX_COPY
            and isinstance(n.args[0], Node)
            and n.args[0].op == "get_attr"
            and n.args[0].target == target
            and isinstance(n.args[3], Node)
            and n.args[3].target is torch.ops.aten.where.self
        ):
            return n
    return None


def _is_split_cache(graph: Graph, node) -> bool:
    """Whether ``node`` reads a main KV cache ``split_kv_cache`` left a fold
    on -- the buffer a dynamic quantize on the read folds into."""
    return (
        isinstance(node, Node)
        and node.op == "get_attr"
        and _fold_write(graph, node.target) is not None
    )


def _fold_quantize_into_split_cache(
    model: GraphModule, node: Node, folds: dict
) -> bool:
    """Fold a dynamic quantize (``quantize_affine`` or ``quantize_mx``) over
    a split KV cache into the cache.

    The quantize sweeps the whole main cache every step, though only whole,
    block-aligned chunks are ever written into it.  So the buffer is baked
    quantized once, into one buffer per quantize output
    (``<cache>_scale`` [/ ``_zero_point``] / ``_full``), the read path takes
    those, and the fold's ``cond`` quantizes the residual instead -- ``R``
    positions, on the completing step only -- storing its outputs into the
    baked buffers at the chunk's entries.  The chunk's blocks are its own
    because ``R`` is a multiple of the block size, so this is the arithmetic
    the sweep did, on the chunk alone.

    Returns:
        ``True`` if ``node`` read a split cache and was folded.

    Raises:
        ValueError: The residual is not a whole number of blocks long.
        RuntimeError: The main cache already holds data where this step's
            fold lands, i.e. the graph was run past a fold before
            ``transform``.
    """
    graph = model.graph
    cache = node.args[0]
    if not isinstance(cache, Node) or cache.op != "get_attr":
        return False
    write = _fold_write(graph, cache.target)
    if write is None:
        return False
    outs = {}
    for user in node.users:
        if user.target is not operator.getitem:
            return False
        outs[user.args[1]] = user

    _, dim, offsets, folded = write.args
    residual = folded.args[1]
    dim %= cache.value.ndim
    length = residual.value.shape[dim]
    block_size = node.args[3]
    if length % block_size:
        raise ValueError(
            f"{cache.target}: a {length}-position residual is not a whole "
            f"number of {block_size}-wide blocks"
        )
    contents = fetch_attr(model, cache.target)
    chunk = offsets.value.to(contents.device)
    if contents.index_select(dim, chunk).abs().sum() != 0:
        raise RuntimeError(
            f"{cache.target} already holds data at the chunk this step folds:"
            " the graph was run past a fold before transform"
        )

    consts = [
        fetch_attr(model, a.target) if isinstance(a, Node) else a
        for a in node.args[1:]
    ]
    baked = node.target(contents, *consts)
    with graph.inserting_before(node):
        buffers = {
            i: create_getattr_from_value(
                model, graph, f"{cache.target}_{name}", baked[i]
            )
            for i, name in enumerate(_OUTPUT_NAMES[node.target])
        }
    for i, old in outs.items():
        buffers[i].meta["dtype"] = old.meta.get("dtype")
        propagate_shape(buffers[i], model)
        old.replace_all_uses_with(buffers[i])

    # The chunk index and the completing-step predicate as control-
    # processor scalars.  ``scalarize_index_arithmetic`` has usually
    # already made the offsets an ``arange`` from ``c * R`` and the
    # predicate a host-written ``full``; otherwise they are read out.
    with graph.inserting_before(write):
        if offsets.target is torch.ops.aten.arange.start_step:
            chunk_index = graph.call_function(
                operator.floordiv, (offsets.args[0], length)
            )
            new_nodes = (chunk_index,)
        else:
            first = graph.call_function(
                torch.ops.aten.slice.Tensor, (offsets, 0, 0, 1)
            )
            chunk_tensor = graph.call_function(
                torch.ops.aten.floor_divide.default, (first, length)
            )
            chunk_index = graph.call_function(
                torch.ops.aten._local_scalar_dense.default, (chunk_tensor,)
            )
            new_nodes = (first, chunk_tensor, chunk_index)
        flag = folded.args[0]
        if flag.target is torch.ops.aten.full.default:
            predicate = flag.args[1]
        else:
            predicate = graph.call_function(
                torch.ops.aten._local_scalar_dense.default, (flag,)
            )
            new_nodes = (*new_nodes, predicate)
        for n in new_nodes:
            propagate_shape(n, model)

    branch = _FoldBranch(model, cache.target, chunk_index)
    chunk = branch.call(
        node.target,
        (branch.placeholder(residual), *branch.arguments(node.args[1:])),
        dtype=node.meta.get("dtype"),
    )
    fold = _Fold(
        cache=cache.target,
        dim=dim,
        branch=branch,
        parts={
            i: branch.call(
                operator.getitem, (chunk, i), dtype=outs[i].meta.get("dtype")
            )
            for i in sorted(outs)
        },
        stores={},
        scale_qmap=(
            get_arg_value(node, 6, "scale_qmap")
            if node.target is _QUANTIZE_AFFINE
            else None
        ),
    )
    for i, part in fold.parts.items():
        fold.stores[i] = branch.store(part, buffers[i], dim)
    branch.place(predicate, before=write)

    # The masked write-back's own reads: the ``where``, the window it
    # selects from the cache, and the handle on the cache.
    reads = (folded, folded.args[2], write.args[0])
    graph.erase_node(write)
    for n in reads:
        if not n.users:
            graph.erase_node(n)
    folds[buffers[_value_index(node.target)].target] = fold

    for n in (*outs.values(), node):
        if not n.users:
            graph.erase_node(n)
    logger.info(
        f"Folded {cache.target}: baked quantized, {length}-position chunks "
        "quantized on their fold"
    )
    return True


def _repeat_in_branch(branch: _FoldBranch, node: Node, dim: int, factor: int):
    """``node`` with every entry along ``dim`` repeated ``factor`` times in
    place, spelled ``unsqueeze -> expand -> reshape`` -- the relayouts the
    lowering already folds into addressing."""
    shape = list(node.value.shape)
    grown = [-1] * (len(shape) + 1)
    grown[dim + 1] = factor
    merged = list(shape)
    merged[dim] *= factor
    dtype = node.meta.get("dtype")
    unsqueezed = branch.call(
        torch.ops.aten.unsqueeze.default, (node, dim + 1), dtype=dtype
    )
    expanded = branch.call(
        torch.ops.aten.expand.default, (unsqueezed, grown), dtype=dtype
    )
    return branch.call(
        torch.ops.aten.reshape.default, (expanded, merged), dtype=dtype
    )


def _fold_reencode(
    model: GraphModule,
    fold: _Fold,
    cache_axes,
    block_axes,
    block_size,
    quantize_mx: Node,
    bases: dict,
    column_axis: int,
    column_repeat: int,
) -> None:
    """Extend a split cache's fold with the qparams its GEMV reads through
    the fused dequantize: the chunk's int6 block scale, and its fused scale
    (affine scale over block scale) and zero point.  All are in cache
    layout, as ``fuse_dequantize_quantize`` baked them, so each is stored
    as the chunk's quantize returns it, in place of the chunk's affine
    scale and zero point.

    Args:
        model: The graph module being lowered.
        fold: The cache's fold.
        cache_axes: Axes the affine groups lie along.
        block_axes: Axes the re-encode blocks along.
        block_size: The block size the groups and blocks share.
        quantize_mx: The re-encode, whose arguments the chunk's copy takes.
        bases: ``"scale"`` (the fused scale), ``"zero_point"`` and ``"mx"``
            -> the baked buffer, as ``fuse_dequantize_quantize`` stored it.
        column_axis: The axis of the GEMV's output columns.
        column_repeat: Entries the block scale is repeated to per block
            along it, as the baked one was.
    """
    branch = fold.branch
    scale_c, zp_c, codes_c = (fold.parts[i] for i in range(3))
    qparam_dtype = bases["scale"].meta.get("dtype")
    for i in (0, 1):
        branch.unstore(fold.stores.pop(i))

    decoded = branch.call(
        _DEQUANTIZE, (codes_c, scale_c, zp_c, cache_axes, block_size)
    )
    mx_args = list(branch.arguments(quantize_mx.args[1:]))
    mx_args[1] = list(block_axes)
    mx = branch.call(
        quantize_mx.target,
        (decoded, *mx_args),
        dtype=quantize_mx.meta.get("dtype"),
    )
    mx_scale = branch.call(
        operator.getitem, (mx, 0), dtype=bases["mx"].meta.get("dtype")
    )

    # The fused scale divides the affine scale by the block scale on the
    # affine groups' grid: the block scale is repeated up to it.
    divisor = mx_scale
    for d, (have, want) in enumerate(
        zip(mx_scale.value.shape, scale_c.value.shape)
    ):
        if have != want:
            divisor = _repeat_in_branch(branch, divisor, d, want // have)
    fused = branch.call(
        torch.ops.aten.div.Tensor, (scale_c, divisor), dtype=qparam_dtype
    )
    if fold.scale_qmap is not None:
        # Rounded through the affine scale's table, as the baked one was: a
        # per-tensor quantize with a unit scale.
        graph = model.graph
        with graph.inserting_before(branch.cond):
            # On the cache's device: propagated values live on the CPU.
            unit = create_getattr_from_value(
                model,
                graph,
                f"{fold.cache}_unit_scale",
                fused.value.new_ones(
                    1, device=fetch_attr(model, fold.cache).device
                ),
            )
        propagate_shape(unit, model)
        fused = branch.call(
            _QUANTIZE,
            (
                fused,
                *branch.arguments((unit, None, None, None, fold.scale_qmap)),
            ),
            dtype=qparam_dtype,
        )

    columns = mx_scale
    if column_repeat > 1:
        columns = _repeat_in_branch(
            branch, mx_scale, column_axis, column_repeat
        )

    branch.store(fused, bases["scale"], fold.dim)
    branch.store(zp_c, bases["zero_point"], fold.dim)
    branch.store(columns, bases["mx"], fold.dim)
    branch.bind()


def sink_cache_folds(model: GraphModule) -> GraphModule:
    """Move each split cache's fold below the last reader of the buffers it
    writes.

    The fold's ``cond`` stands where the eager graph wrote the chunk back:
    after the attention read the cache.  Operator fusion can carry a read
    past it, since a fused group lands at its last op and the GEMV reading
    the cache may fuse with ops that follow the fold.  A read after the
    fold would see the completing chunk in both the main cache and the
    residual, so the ``cond`` and its branch handles go back after the
    last reader of any buffer the ``true`` branch writes in place.

    Runs at the end of ``fuse_operator``.

    Args:
        model: The graph module to reorder in place.

    Returns:
        ``model``, reordered in place.
    """
    graph = model.graph
    for cond in list(graph.nodes):
        if cond.target is not _COND:
            continue
        true_graph = getattr(model, cond.args[1].target).graph
        placeholders = [n for n in true_graph.nodes if n.op == "placeholder"]
        written = {
            cond.args[3][placeholders.index(n.args[0])].target
            for n in true_graph.nodes
            if n.target is _INDEX_COPY
        }
        order = {n: i for i, n in enumerate(graph.nodes)}
        last = cond
        for n in graph.nodes:
            if n.op != "get_attr" or n.target not in written:
                continue
            for user in n.users:
                if order[user] > order[last]:
                    last = user
        if last is cond:
            continue
        reader = last
        for moved in (*cond.args[1:3], cond):
            last.append(moved)
            last = moved
        logger.info(f"Sunk {cond} below {reader}")
    graph.lint()
    model.recompile()
    return model


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

    The op returns its qparams beside the value (``_OUTPUT_NAMES``): the
    value keeps the relayout ops it was lifted over and each qparam gets a
    replayed copy of them.  It is never forked: a
    concat can move the axis it blocks along, and head splitting -- the only
    thing that forks -- never runs where microscaling is used.
    """

    graph = model.graph
    outs = {}
    for user in node.users:
        if user.target is not operator.getitem:
            return False
        outs[user.args[1]] = user

    value_index = _value_index(node.target)
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
        and not _is_split_cache(graph, src)
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
        value = unpack(value_index)
    path[-1].replace_input_with(src, value)
    for n in reversed(path):
        propagate_shape(n, model)
    _annotate(path, outs[value_index])
    outs[value_index].replace_all_uses_with(path[0])

    # A qparam is one element per block, so it needs its own copy of them.
    for i, old in outs.items():
        if i == value_index:
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


def fuse_quantize_dequantize_with_producer(model: GraphModule):
    """Move each quantize / dequantize up the graph to sit directly after the
    op that computed its input, so the two can fuse into one kernel.

    Everything it is lifted over only relayouts data -- a reshape, a transpose,
    the ``stack`` MHA splitting leaves behind, the ``expand`` of GQA's
    ``repeat_kv`` -- so quantizing above them is the same arithmetic on less
    data.  A dynamic quantize (``quantize_mx``, ``quantize_affine``) also
    blocks along an axis and returns its qparams beside its value, so it takes
    the ``_hoist_microscaling`` route; the rest share the walk but fork over a
    concat.

    A ``quantize_affine`` on a KV cache is folded into the cache first: a
    split cache (``split_kv_cache``) is baked quantized and its chunks are
    quantized on their fold, a cache written token by token is baked and
    quantized on the write.  The baked cache then reads ``get_attr ->
    dequantize -> relayouts -> quantize_mx``, which ``fuse_dequantize_quantize``
    collapses into one dequantize -- extending a split cache's fold with the
    qparams that dequantize reads -- before the ``quantize_mx`` hoist would
    lift the re-encode above the relayouts.  The ``quantize_mx`` hoist and
    cache fold come last.

    Args:
        model: The graph module to rewrite in place.

    Returns:
        ``model``, rewritten in place.
    """
    graph = model.graph

    folds = {}
    for node in list(graph.nodes):
        if node.target is _QUANTIZE_AFFINE:
            if not _fold_quantize_into_split_cache(model, node, folds):
                _fold_quantize_into_cache(model, node)
    graph.eliminate_dead_code()
    fuse_dequantize_quantize(model, folds)

    for node in list(graph.nodes):
        if node.target not in QUANTIZE_FAMILY_OPS:
            continue
        if node.target is _QUANTIZE_MX_OUTLIER:
            # The sparse quantize's CSR outputs have no relayout replay; it
            # stays where the quantizer put it.
            continue
        if node.target in _OUTPUT_NAMES:
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
            if not _fold_quantize_into_split_cache(model, node, folds):
                _fold_quantize_into_cache(model, node)

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
        # ``-1`` keeps the dim, so it is not a size to divide.
        size = [
            math.ceil(s / block_size) if d in axes and s != -1 else s
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
        # The node's own output resolves a ``-1`` in the requested shape.
        target_shape = list(getattr(node, "value", node.args[1]).shape)
        axes = validate_and_map_group_axes_for_reshape(
            orig_shape, target_shape, axes
        )
        if axes is None:
            raise RuntimeError(
                f"{node}: reshape {orig_shape} -> {target_shape} cuts a "
                "quantization block"
            )
        new_shape = [
            math.ceil(s / block_size) if d in axes else s
            for d, s in enumerate(target_shape)
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
    # the dim above -- the head dim.  A ``-1`` size keeps its dim.
    grown = [
        d
        for d, s in enumerate(expand_node.args[1])
        if s != -1 and s != expand_node.args[0].shape[d]
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
    axes = tuple(a + input.ndim if a < 0 else a for a in axes)
    if not nodes:
        # Nothing between the dequantize and the quantize (multi-head
        # attention's values): the qparams reach it as stored.
        return input, axes
    env = {nodes[0].args[0]: input}

    def map_node(n):
        if n.op == "get_attr":
            return fetch_attr(model, n.target)
        return env[n]

    def load_arg(a):
        return torch.fx.graph.map_arg(a, map_node)

    for n in nodes:
        if n.target in LAYOUT_OPS:
            env[n], axes = propagate_group_axes_through_op(
                n, env[n.args[0]], axes, block_size
            )
        else:
            env[n] = n.target(*load_arg(n.args), **load_arg(n.kwargs))
    return env[nodes[-1]], axes


def fuse_dequantize_quantize(model: torch.fx.GraphModule, folds=None):
    """
    Fuses consecutive dequantize -> quantize operations in a quantized model
    for optimization.

    A broadcast qparam (GQA's ``repeat_kv``) is stored once per KV head rather
    than once per query head, with the repeat put back as graph ops: a smaller
    buffer for a longer graph, which pays off because the GEMM folds the repeat
    into its tile addressing instead of copying.

    A chain reading a split KV cache's baked codes (``folds``, keyed by the
    codes buffer's target) gets the same fused dequantize, with three
    differences.  Its qparams are kept in cache layout, like the codes --
    the block scale taken on the cache's own blocks, the group scale and
    zero point as the cache holds them -- with the relayouts to the GEMM
    put back into the graph on the qparams, so the fold can store a chunk's
    qparams as its quantize returns them.  Its fused scale and zero point
    take the affine qparams' dtype, the fused scale rounded through the
    affine scale's table when there is one, since the fold stores a run-time
    quotient into that buffer.  And the cache's fold is extended to store
    the chunk's block scale, fused scale and zero point beside its codes
    (``_fold_reencode``).

    Args:
        model (GraphModule): The FX-traced model to optimize.
        folds: ``codes target -> _Fold`` for the split caches
            ``_fold_quantize_into_split_cache`` baked, or ``None``.

    Returns:
        GraphModule: The optimized model with fused operations.
    """
    graph = model.graph
    folds = folds or {}
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
        fold = (
            folds.get(dq_node.args[0].target)
            if node.target == torch.ops.quantized_ops.quantize_mx.default
            else None
        )
        cache_axes = get_arg_value(dq_node, 3, "axes")
        dq_scale = fetch_attr(model, dq_node.args[1].target)
        zero_point = (
            fetch_attr(model, dq_node.args[2].target)
            if len(dq_node.args) > 2
            else None
        )
        relaid_scale, new_dq_axes = run_qparam_through_nodes(
            model, dq_scale, nodes_on_path[1:-1], cache_axes, block_size
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

        if fold is not None:
            # A split cache's qparams stay in cache layout, like its codes:
            # the block scale is taken on the cache's own blocks, the group
            # scale and zero point as the cache holds them, and the
            # relayouts to the GEMM go back into the graph on the qparams
            # (``relaid``), so the fold stores a chunk's qparams as its
            # quantize returns them.
            src, path, _, block_axes = _relayout_path(
                node.args[0], node.args[2], block_size
            )
            # The GEMM's output columns: the last axis below the relayouts.
            src_c, _, _, (column_axis,) = _relayout_path(
                node.args[0], (-1,), None
            )
            if src is not dq_node or src_c is not dq_node:
                raise RuntimeError(
                    f"{dq_node.args[0].target}: the read path's relayouts "
                    "cut the re-encode's blocks"
                )
            decoded = run_through_ops(model, dq_input, [dq_node])
            mx_args = [
                fetch_attr(model, a.target) if isinstance(a, Node) else a
                for a in node.args[1:]
            ]
            mx_args[1] = list(block_axes)
            q_scale = node.target(decoded, *mx_args)[0]
            column_axis %= decoded.ndim
            columns = decoded.shape[column_axis]
            block_axes = sorted(a % decoded.ndim for a in block_axes)
            mx_blocked = [a for a in block_axes if a != column_axis]
        else:
            if node.target == torch.ops.quantized_ops.quantize_mx.default:
                q_scale = run_through_ops(model, dq_input, nodes_on_path)[0]
            else:
                q_scale = fetch_attr(model, scale_node.target)
            dq_scale = relaid_scale
            if zero_point is not None:
                zero_point, _ = run_qparam_through_nodes(
                    model,
                    zero_point,
                    nodes_on_path[1:-1],
                    cache_axes,
                    block_size,
                )
            path, mx_blocked = (), ()
            column_axis, columns = -1, output.shape[-1]

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
        fold = (
            folds.get(dq_node.args[0].target)
            if node.target == torch.ops.quantized_ops.quantize_mx.default
            else None
        )
        if fold is not None and fold.scale_qmap is not None:
            fused_scale = torch.ops.quantized_ops.quantize(
                fused_scale,
                fused_scale.new_ones(1),
                qmap=fetch_attr(model, fold.scale_qmap.target),
            )

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

        # The fused scale and zero point keep the dtype of the qparams they
        # derive from; the block scale keeps its own.
        qparam_dtype = dq_node.args[1].meta.get("dtype")
        mx_dtype = scale_node.meta.get("dtype")
        # A split cache's baked qparam buffers, by role, for its fold.
        bases = {}

        def relaid(attr, blocked, dtype):
            """``attr``, a cache-layout qparam with one entry per block along
            ``blocked``, read through the path's relayouts."""
            cur, axes = attr, [a % attr.value.ndim for a in blocked]
            with graph.inserting_before(node):
                for op in reversed(path):
                    axes = [_axis_through(op, a) for a in axes]
                    cur = _replay_relayout(graph, op, cur, axes, block_size)
                    cur.meta["dtype"] = dtype
                    propagate_shape(cur, model)
            return cur

        def create_qparam(value, name, role, blocked, dtype):
            if fold is not None:
                with graph.inserting_before(node):
                    attr = create_getattr_from_value(model, graph, name, value)
                attr.meta["dtype"] = dtype
                propagate_shape(attr, model)
                bases[role] = attr
                return relaid(attr, blocked, dtype)
            if expand_node is not None:
                stored = store_qparam_unrepeated(
                    model, value, expand_node, name, node, dtype
                )
                if stored is not None:
                    return stored
            with graph.inserting_before(node):
                attr = create_getattr_from_value(model, graph, name, value)
            attr.meta["dtype"] = dtype
            propagate_shape(attr, model)
            return attr

        input_node = dq_node.args[0]
        new_scale = create_qparam(
            fused_scale,
            input_node.name + "_scale",
            "scale",
            cache_axes,
            qparam_dtype,
        )
        new_zero_point = (
            create_qparam(
                zero_point,
                input_node.name + "_zero_point",
                "zero_point",
                cache_axes,
                qparam_dtype,
            )
            if zero_point is not None
            else None
        )
        with graph.inserting_before(node):
            # qmap is at arg index 1 for quantize_mx, index 5 for plain
            # quantize.
            if node.target == torch.ops.quantized_ops.quantize_mx.default:
                output_qmap = graph.node_copy(node.args[1])
            else:
                output_qmap = graph.node_copy(node.args[5])
            propagate_shape(output_qmap, model)
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

        column_repeat = 1
        if scale_node.op != "get_attr":
            if (
                any(is_gemm_op(n) for n in scale_node.users)
                and q_scale.shape[column_axis] != columns
            ):
                column_repeat = columns // q_scale.shape[column_axis]
                q_scale = torch.repeat_interleave(
                    q_scale, repeats=column_repeat, dim=column_axis
                )

            mx_scale = create_qparam(
                q_scale, input_node.name + "_scale", "mx", mx_blocked, mx_dtype
            )
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
        propagate_shape(new_dq, model)

        for n in nodes_on_path[1:-1]:
            n.meta["dtype"] = input_node.meta.get("dtype")

        if fold is not None:
            _fold_reencode(
                model,
                fold,
                cache_axes,
                block_axes,
                block_size,
                node,
                bases,
                column_axis,
                column_repeat,
            )

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
