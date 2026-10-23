"""Fold view-like glue ops into ``voyager.subview`` windows.

Runs at the end of ``bufferize_graph``, on the bufferized graph: the nests
and their output allocs exist, and emit resolves every reference through
the graph itself (sub-graph placeholders bind positionally), so rewiring
here reaches everything the builders made.

``aten.select`` / ``aten.slice`` / ``aten.unbind`` (read side): when every
dim before the named one has extent 1, the named bytes are one contiguous
run of the source, so the op is pure addressing -- replace it with a
``voyager.subview`` and emit folds the window into each consumer's
``TensorBoxRef``.  ``unbind`` is a ``select`` at every index of its dim, so
each element becomes its own window.  A select on an extent-1 dim and a
whole-extent slice are already nops (``is_nop``); a non-contiguous one
stays an op.

``aten.cat`` / ``aten.stack`` (write side): when every dim before the join
dim has extent 1, each source occupies one contiguous window of the
result -- ``stack`` joining along a dim its sources do not carry, so its
windows are size 1 there and squeeze it away.  Allocate the result once,
then point each source's storage at its window: a source produced by a
nest has its output alloc *replaced* by the window, so the nest stores
straight into the join buffer -- zero copies.  A nest whose output reaches
the join through a reshape keeps storing at its own shape, so it takes the
window reshaped back to it; a source with no alloc to redirect (a
parameter) is host-copied into its window with ``insert(clone(src),
window)``, the sanctioned buffer-to-buffer move.
"""

import operator

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value, is_nop
from voyager_compiler.codegen.transform.bufferize.ops import MemoryLevel
from voyager_compiler.shape_prop import set_node_value

_ALLOC = torch.ops.voyager.alloc.default
_CAT = torch.ops.aten.cat.default
_CLONE = torch.ops.aten.clone.default
_INSERT = torch.ops.voyager.insert.default
_RESHAPE = torch.ops.aten.reshape.default
_SELECT = torch.ops.aten.select.int
_SLICE = torch.ops.aten.slice.Tensor
_STACK = torch.ops.aten.stack.default
_SUBVIEW = torch.ops.voyager.subview.default
_UNBIND = torch.ops.aten.unbind.int
_WHILE_LOOP = torch.ops.higher_order.while_loop


def lower_views(model: GraphModule) -> GraphModule:
    """Rewrite foldable ``select`` / ``slice`` / ``cat`` / ``stack`` nodes
    into subview windows."""
    for node in list(model.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is _SELECT:
            _fold_select(model, node)
        elif node.target is _SLICE:
            _fold_slice(model, node)
        elif node.target is _UNBIND:
            _fold_unbind(model, node)
        elif node.target in (_CAT, _STACK):
            _fold_join(model, node)
    model.graph.lint()
    model.recompile()
    return model


def _leading_unit_dims(shape, dim) -> bool:
    return all(s == 1 for s in shape[:dim])


def _fold_select(model: GraphModule, node: Node) -> None:
    source, dim, index = node.args
    value = getattr(source, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    index = index + shape[dim] if index < 0 else index
    # An extent-1 dim is a pure rename (``is_nop``); a non-unit dim ahead
    # of the selected one makes the bytes several disjoint runs, which one
    # window cannot express.
    if shape[dim] == 1 or not _leading_unit_dims(shape, dim):
        return
    offsets = [0] * len(shape)
    offsets[dim] = index
    sizes = list(shape)
    sizes[dim] = 1
    graph = model.graph
    with graph.inserting_before(node):
        subview = graph.call_function(
            _SUBVIEW,
            (source, offsets, sizes, [1] * len(shape)),
            {"squeeze_dim": [dim]},
        )
    set_node_value(subview, node.value)
    node.replace_all_uses_with(subview)
    graph.erase_node(node)


def _fold_unbind(model: GraphModule, node: Node) -> None:
    source = node.args[0]
    dim = get_arg_value(node, 1, "dim", 0)
    value = getattr(source, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    # ``unbind`` is a ``select`` at every index of ``dim``, and carries the
    # same rule: a non-unit dim ahead of it makes each element several
    # disjoint runs, which one window cannot express.
    if not _leading_unit_dims(shape, dim):
        return
    # The elements are reached one at a time; an unbind consumed whole (a
    # list handed to another op) names no single window.
    users = list(node.users)
    if any(
        u.op != "call_function" or u.target is not operator.getitem
        for u in users
    ):
        return

    graph = model.graph
    for user in users:
        index = user.args[1]
        index = index + shape[dim] if index < 0 else index
        offsets = [0] * len(shape)
        offsets[dim] = index
        sizes = list(shape)
        sizes[dim] = 1
        with graph.inserting_before(user):
            subview = graph.call_function(
                _SUBVIEW,
                (source, offsets, sizes, [1] * len(shape)),
                {"squeeze_dim": [dim]},
            )
        set_node_value(subview, user.value)
        user.replace_all_uses_with(subview)
        graph.erase_node(user)
    graph.erase_node(node)


def _fold_slice(model: GraphModule, node: Node) -> None:
    source = node.args[0]
    dim = get_arg_value(node, 1, "dim", 0)
    start = get_arg_value(node, 2, "start")
    end = get_arg_value(node, 3, "end")
    step = get_arg_value(node, 4, "step", 1)
    value = getattr(source, "value", None)
    # A runtime bound names a window the reference cannot carry statically,
    # and a step skips bytes one window cannot express.
    if not isinstance(value, torch.Tensor) or step != 1:
        return
    if isinstance(start, Node) or isinstance(end, Node):
        return
    shape = list(value.shape)
    dim = dim + len(shape) if dim < 0 else dim
    extent = shape[dim]
    start = 0 if start is None else (start + extent if start < 0 else start)
    end = extent if end is None else (end + extent if end < 0 else end)
    start = max(0, min(start, extent))
    end = max(start, min(end, extent))
    # A whole-extent slice is already a nop (``is_prunable_op``); a non-unit
    # dim ahead of the sliced one makes the bytes several disjoint runs,
    # which one window cannot express.
    if (start, end) == (0, extent) or not _leading_unit_dims(shape, dim):
        return
    offsets = [0] * len(shape)
    offsets[dim] = start
    sizes = list(shape)
    sizes[dim] = end - start
    graph = model.graph
    with graph.inserting_before(node):
        subview = graph.call_function(
            _SUBVIEW, (source, offsets, sizes, [1] * len(shape))
        )
    set_node_value(subview, node.value)
    node.replace_all_uses_with(subview)
    graph.erase_node(node)


def _storage_of(node):
    """The builder-created output ``alloc`` whose bytes ``node`` names, walked
    through result handles and views; ``None`` when there is none to redirect
    (a parameter, or a shape this walk does not cover)."""
    while isinstance(node, Node):
        if node.op != "call_function":
            return None
        if node.target is _ALLOC:
            return node
        if node.target is operator.getitem:
            src, index = node.args
            if isinstance(src, Node) and src.target is _WHILE_LOOP:
                carried = list(src.args[2])
                if index < len(carried):
                    node = carried[index]
                    continue
            return None
        if node.target is _SUBVIEW or is_nop(node):
            node = node.args[0]
            continue
        return None
    return None


def _fold_join(model: GraphModule, node: Node) -> None:
    stacked = node.target is _STACK
    sources = list(node.args[0])
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        return
    out_shape = list(value.shape)
    rank = len(out_shape)
    dim = dim + rank if dim < 0 else dim
    if not _leading_unit_dims(out_shape, dim):
        return
    values = [getattr(s, "value", None) for s in sources]
    # ``stack`` joins along a dim its sources do not carry.
    src_rank = rank - 1 if stacked else rank
    if any(
        not isinstance(v, torch.Tensor)
        or len(v.shape) != src_rank
        or v.dtype != value.dtype
        for v in values
    ):
        return
    # A shared producer cannot land in two windows at once.
    targets = [_storage_of(s) for s in sources]
    redirected = [t for t in targets if t is not None]
    if len(set(redirected)) != len(redirected):
        return

    graph = model.graph
    ordered = list(graph.nodes)
    order = {n: i for i, n in enumerate(ordered)}
    anchors = [t if t is not None else s for t, s in zip(targets, sources)]
    first = min(anchors, key=lambda n: order[n])

    # The cat's storage, alive from the first producer's stores onward.
    # Space rides positionally: ``annotate_tensor_spaces`` reads ``args[2]``.
    with graph.inserting_before(first):
        buf = graph.call_function(
            _ALLOC, (out_shape, value.dtype, int(MemoryLevel.DRAM), 0)
        )
    set_node_value(buf, value)
    if (scope := first.meta.get("scope")) is not None:
        buf.meta["scope"] = scope

    offset = 0
    last = buf
    for s, t, v in zip(sources, targets, values):
        offsets = [0] * rank
        offsets[dim] = offset
        if stacked:
            sizes = list(out_shape)
            sizes[dim] = 1
            squeeze = {"squeeze_dim": [dim]}
            offset += 1
        else:
            sizes = list(v.shape)
            squeeze = {}
            offset += v.shape[dim]
        with graph.inserting_after(last):
            window = graph.call_function(
                _SUBVIEW, (buf, offsets, sizes, [1] * rank), squeeze
            )
        set_node_value(window, v)
        last = window
        if t is not None:
            dest = window
            if tuple(t.value.shape) != tuple(v.shape):
                with graph.inserting_after(last):
                    dest = graph.call_function(
                        _RESHAPE, (window, list(t.value.shape))
                    )
                set_node_value(dest, t.value)
                last = dest
            t.replace_all_uses_with(dest)
            graph.erase_node(t)
        else:
            with graph.inserting_before(node):
                clone = graph.call_function(_CLONE, (s,))
                graph.call_function(_INSERT, (clone, window))
            set_node_value(clone, v)

    node.replace_all_uses_with(buf)
    graph.erase_node(node)
