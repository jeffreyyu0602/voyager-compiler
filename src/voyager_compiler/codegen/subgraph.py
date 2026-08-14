"""Moving code between FX graphs.

Outlining -- lift a run of nodes into a submodule and call it -- and its
inverse, splicing an exported ``GraphModule`` back into a parent graph in
place of a node.  Plus the naming rules both directions use.
"""

import logging
import operator
from typing import Dict, List, Optional

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg
from transformers.utils.import_utils import is_torch_greater_or_equal

from voyager_compiler.codegen.iteration_space import (
    IterationSpaceNormalizer,
    NormalizationError,
)
from voyager_compiler.codegen.node_info import is_gemm_op, is_nop, is_reshape_op
from voyager_compiler.export_utils import create_getattr_from_value
from voyager_compiler.shape_prop import (
    ShapeProp,
    fetch_attr,
    propagate_shape,
    set_node_value,
)

logger = logging.getLogger(__name__)


def update_submod_user_meta(model, node, named_modules=None):
    """
    Update the metadata of all user nodes that consume the given node.
    """
    if named_modules is None:
        named_modules = dict(model.named_modules())

    for user in list(node.users):
        if user.op != "call_module":
            continue

        index = user.all_input_nodes.index(node)

        submod = named_modules[user.target]
        placeholders = [n for n in submod.graph.nodes if n.op == "placeholder"]

        assert index < len(placeholders)

        placeholder = placeholders[index]
        placeholder.name = node.name
        placeholder.meta["source_node"] = node


def copy_graph_module(
    gm: GraphModule, remap: Optional[Dict[Node, Node]] = None
) -> GraphModule:
    """An independent-graph copy of ``gm`` (recursively for child
    GraphModules) that preserves each node's ``meta`` and ``.value`` and shares
    the constant params / buffers — WITHOUT recompiling ``forward``.

    The body stays runnable: ``node_copy`` preserves node names, so the shared
    (already-compiled) ``forward`` works on the copied graph — we skip only the
    *recompile*, which ``deepcopy`` and the ``.graph`` setter both force.  But a
    shared body *object* breaks per-site meta (e.g. scratchpad Segments stamped
    on its nodes), so each splice needs its own node objects; this gives them
    cheaply.

    ``object.__new__`` (not ``gm.__class__.__new__``) is required: fx's
    ``GraphModule.__new__`` mints a fresh ``forward``-less subclass each call,
    whereas we want to keep ``gm``'s class (which carries the compiled
    ``forward``).
    """
    root = remap is None
    if remap is None:
        remap = {}

    new = object.__new__(gm.__class__)
    new.__dict__ = dict(gm.__dict__)  # share params / buffers / forward code
    new._modules = {
        k: copy_graph_module(v, remap) if isinstance(v, GraphModule) else v
        for k, v in gm._modules.items()
    }
    new_graph = Graph()
    for n in gm.graph.nodes:
        c = new_graph.node_copy(n, lambda x: remap[x])
        remap[n] = c
        if (val := getattr(n, "value", None)) is not None:
            c.value, c.shape = val, getattr(n, "shape", None)
    new._graph = new_graph

    if root:
        # Rebind ``source_node`` to this copy: it is how codegen resolves a
        # fused op's tile (name / address / value), and left alone it points
        # into the template the build cache shares between identical ops.
        for copied in remap.values():
            src = copied.meta.get("source_node")
            if src in remap:
                copied.meta["source_node"] = remap[src]
    return new


def replace_node_with_graph_module(
    model: GraphModule,
    source: Node,
    replacement: GraphModule,
    value_remap=None,
    propagate: bool = True,
) -> List[Node]:
    """Copy ``replacement`` (an exported subgraph) into ``model`` just before
    ``source``, mapping its placeholders to ``source.all_input_nodes``; return
    the output value node(s).

    ``propagate`` re-runs ``propagate_shape`` on each copied node to recover its
    shape / value.  Pass ``False`` when ``replacement`` is already
    shape-propagated and its placeholder shapes match this site (e.g. a cached
    bufferized nest): the precomputed value is carried over instead, which
    avoids re-executing the whole tiled ``while_loop`` per splice and the
    O(N^2) ``named_modules`` rescans ``propagate_shape`` does each call.
    """
    graph = model.graph
    if value_remap is None:
        value_remap = {}

    arg_nodes = iter(source.all_input_nodes)
    output = None

    for n in list(replacement.graph.nodes):
        if n.op == "placeholder":
            value_remap[n] = next(arg_nodes, None)
            continue
        if n.op == "output":
            output = n.args[0]
            if len(output) == 1:
                source.replace_all_uses_with(value_remap[output[0]])
            else:
                for user in list(source.users):
                    assert user.target == operator.getitem
                    idx = user.args[1]
                    user.replace_all_uses_with(value_remap[output[idx]])
            continue
        with graph.inserting_before(source):
            if n.op == "get_attr":
                attr = fetch_attr(replacement, n.target)
                if isinstance(attr, GraphModule):
                    # cond / body subgraph of a while_loop: register an
                    # independent copy (shared body objects would collide when
                    # later passes stamp per-site meta on their nodes).
                    name = get_new_node_name_with_prefix(n.target)(model)
                    setattr(model, name, copy_graph_module(attr))
                    value_remap[n] = graph.create_node("get_attr", name)
                else:
                    value_remap[n] = create_getattr_from_value(
                        model, graph, n.target, attr
                    )
            elif n.op == "call_module":
                # A fused compute submodule (the attention builders run
                # _fuse_passes, leaving top-level fused call_modules): register
                # an independent copy under a fresh name, like a body subgraph,
                # and retarget the copied call at it.
                sub = fetch_attr(replacement, n.target)
                name = get_new_node_name_with_prefix(n.target)(model)
                setattr(model, name, copy_graph_module(sub))
                new = graph.node_copy(n, lambda x: value_remap[x])
                new.target = name
                value_remap[n] = new
            else:
                value_remap[n] = graph.node_copy(n, lambda n: value_remap[n])
            if propagate:
                propagate_shape(value_remap[n], model)

    return [value_remap[n] for n in output]


def create_subgraph(nodes: List[Node]):
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    for node in nodes:
        for n in node.all_input_nodes:
            if n not in value_remap:
                value_remap[n] = new_graph.placeholder(n.name)
                new_args.append(n)
                value_remap[n].meta["source_node"] = n
        value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

    new_graph.output(value_remap[nodes[-1]])
    new_graph.lint()
    gm = torch.fx.GraphModule(torch.nn.Module(), new_graph)
    return gm, tuple(new_args)


OP_PARAM_ARG_INDEX = {
    torch.ops.aten.conv2d.default: 1,
    torch.ops.aten.layer_norm.default: 2,
    torch.ops.aten.linear.default: 1,
    torch.ops.quantized_ops.conv2d.default: 1,
    torch.ops.quantized_ops.conv2d_mx.default: 1,
    torch.ops.quantized_ops.layer_norm.default: 2,
    torch.ops.quantized_ops.linear.default: 1,
    torch.ops.quantized_ops.linear_mx.default: 1,
}


def get_unique_node_name(node: Node):
    """
    Generate a unique and meaningful name for the node based on its parameter.
    """
    if (pos := OP_PARAM_ARG_INDEX.get(node.target)) is not None:
        weight_node = node.args[pos]
        # There are cases where weights are sliced. Trace up to find the
        # get_attr node and use the parameter name
        while weight_node.target == torch.ops.aten.slice.Tensor:
            weight_node = weight_node.args[0]

        if weight_node.op == "get_attr":
            return weight_node.name.split("_weight")[0]

    return node.name


def get_new_node_name_with_prefix(prefix: str):
    """
    Generate a new attribute name with a given prefix that is not already used
    in the module's graph.
    """
    prefix = prefix.replace(".", "_")

    def get_new_node_name(module: torch.nn.Module):
        existing_names = {n.name for n in module.graph.nodes}
        existing_names.update(dict(module.named_modules()).keys())

        if prefix not in existing_names:
            return prefix

        i = 1
        while f"{prefix}_{i}" in existing_names:
            i += 1

        node_name = f"{prefix}_{i}"
        logger.debug(f"Generated new unique node name: {node_name}")
        return node_name

    return get_new_node_name


def get_submodule_name(module, nodes: List[Node]):
    prefix = "submodule"
    if is_torch_greater_or_equal("2.5"):
        anchor_node = None
        for n in nodes:
            if n.target in OP_PARAM_ARG_INDEX or is_gemm_op(n):
                anchor_node = n
                break
            if (
                n.op == "call_function"
                and not is_nop(n)
                and not is_reshape_op(n)
                and (
                    anchor_node is None
                    or anchor_node.target
                    == torch.ops.quantized_ops.dequantize.default
                )
            ):
                anchor_node = n
        if anchor_node is not None:
            prefix = get_unique_node_name(anchor_node)
            if len(nodes) > 1:
                prefix += "_fused"

    get_new_node_name = get_new_node_name_with_prefix(prefix)
    return get_new_node_name(module)


def rename_nodes_with_param_names(model: GraphModule):
    if not is_torch_greater_or_equal("2.5"):
        return
    graph = model.graph
    named_modules = dict(model.named_modules())
    for node in list(graph.nodes):
        if node.target in OP_PARAM_ARG_INDEX:
            node.name = get_submodule_name(model, [node])
            update_submod_user_meta(model, node, named_modules)
    graph.lint()
    model.recompile()


def create_and_insert_subgraph(
    nodes: List[Node],
    model: torch.nn.Module,
    node_order: Dict[Node, int] = None,
    normalize_iteration_space: bool = False,
) -> Optional[Node]:
    if node_order is None:
        node_order = {n: i for i, n in enumerate(model.graph.nodes)}
    nodes.sort(key=lambda n: node_order[n])
    submodule, new_args = create_subgraph(nodes)
    node_name = get_submodule_name(model, nodes)
    setattr(model, node_name, submodule)
    with model.graph.inserting_after(nodes[-1]):
        new_node = model.graph.create_node(
            "call_module", node_name, new_args, {}
        )
    new_node.meta["submodule"] = submodule
    if (dtype := nodes[-1].meta.get("dtype", None)) is not None:
        new_node.meta["dtype"] = dtype
    nodes[-1].replace_all_uses_with(new_node)

    # An operand is cloned so propagation cannot write through it, but a group
    # may also take a scalar -- an index or a running offset read out of a
    # buffer, which the datapath walk stops at -- and those are passed as is.
    args = map_arg(
        new_node.args,
        lambda n: (
            n.value.clone() if isinstance(n.value, torch.Tensor) else n.value
        ),
    )
    result = ShapeProp(submodule).propagate(*args)
    set_node_value(new_node, result)

    # Normalize the fused submodule to the anchor's iteration space; skip the
    # group (leaving it fused as-is) if it cannot share one iteration space.
    if normalize_iteration_space:
        try:
            IterationSpaceNormalizer().normalize(model, new_node)
        except NormalizationError as exc:
            logger.warning("normalization failed: %s: %s", new_node, exc)
            new_node.replace_all_uses_with(nodes[-1])
            delattr(model, node_name)
            model.graph.erase_node(new_node)
            return None

    for node in reversed(nodes):
        if not node.users:
            model.graph.erase_node(node)
    return new_node
