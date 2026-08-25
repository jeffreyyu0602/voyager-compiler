"""Fold foldable view ops into ``voyager.subview`` windows.

An ``aten.select`` the model performs (a BERT-family CLS read, a ViT class
token) survives bufferization as an executed cpu op with a buffer of its
own, which the runtime must then implement.  But when the bytes it reads
are one contiguous run, the op is pure addressing: lower it to a
``voyager.subview`` here, before ``bufferize_graph`` runs, and emit folds
the window into the consumer's ``TensorBoxRef`` -- no instruction, no
allocation, and the consumer reads the parent buffer in place.

A select folds exactly when every dim before the selected one has extent
1 -- otherwise the selected bytes are several disjoint runs, which a
window cannot express (the backend's ``resolve_window`` rejects it).  A
select *on* an extent-1 dim is already a nop (``is_nop``) and stays on
that path.  Anything unfoldable is left in place.

Future view ops that reduce to windows belong here too -- ``stack`` and
friends fold on the destination side (each producer writes a window of
the stacked buffer), which is a separate rule, not a variant of this one.
"""

import torch
from torch.fx import GraphModule

from voyager_compiler.codegen.subgraph import update_submod_user_meta
from voyager_compiler.shape_prop import propagate_shape

_SELECT = torch.ops.aten.select.int
_SUBVIEW = torch.ops.voyager.subview.default


def lower_views(model: GraphModule) -> GraphModule:
    """Rewrite each foldable ``aten.select`` into a ``voyager.subview``."""
    graph = model.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not _SELECT:
            continue
        source, dim, index = node.args
        value = getattr(source, "value", None)
        if not isinstance(value, torch.Tensor):
            continue
        shape = list(value.shape)
        dim = dim + len(shape) if dim < 0 else dim
        index = index + shape[dim] if index < 0 else index
        if shape[dim] == 1 or any(s != 1 for s in shape[:dim]):
            continue
        offsets = [0] * len(shape)
        offsets[dim] = index
        sizes = list(shape)
        sizes[dim] = 1
        with graph.inserting_before(node):
            subview = graph.call_function(
                _SUBVIEW,
                (source, offsets, sizes, [1] * len(shape)),
                {"squeeze_dim": [dim]},
            )
        propagate_shape(subview, model)
        node.replace_all_uses_with(subview)
        update_submod_user_meta(model, subview)
        graph.erase_node(node)
    graph.lint()
    model.recompile()
    return model
