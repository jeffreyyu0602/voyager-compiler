"""Capture an FX graph, add to one, and read what capture recorded.

``export_model`` / ``get_aten_graph_module`` turn eager code into a
``GraphModule``; ``create_getattr_from_value`` materialises a tensor into
an existing graph as a buffer; ``get_module_stack``,
``get_node_name_to_scope`` and ``print_node_scope_tabular`` read the module
provenance capture stamps on every node.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch._export.utils import _disable_aten_to_metadata_assertions
from torch.fx import Graph, GraphModule, Node

logger = logging.getLogger(__name__)

__all__ = [
    "create_getattr_from_value",
    "export_model",
    "get_aten_graph_module",
    "get_conv_bn_layers",
    "get_module_stack",
    "get_node_name_to_scope",
    "print_node_scope_tabular",
]


def export_model(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    strict: bool = False,
):
    """Export ``model`` to a training-safe ``GraphModule``.

    Suppresses the ``_assert_tensor_metadata`` nodes that each ``.to(dtype)``
    would otherwise pin to the dtype seen at trace time.

    Args:
        model: Module to export.
        args: Positional example inputs.
        kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to ``torch.export``.
        strict: Whether export runs under torchdynamo.

    Returns:
        The exported program's ``GraphModule``.
    """
    with _disable_aten_to_metadata_assertions():
        gm = torch.export.export(
            model, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict
        )
    return gm.module(check_guards=False)


def get_conv_bn_layers(model: torch.nn.Module) -> List[List[str]]:
    """Name every ``Conv2d`` directly followed by a ``BatchNorm2d``.

    A pair is two consecutive children of one module, the conv registered
    right before the batch norm, named as ``torch.ao.quantization``'s
    ``fuse_modules`` takes them.

    Args:
        model: Module searched recursively.

    Returns:
        ``[conv, bn]`` pairs of qualified module names.
    """
    layers = []
    module_names = list(model._modules)
    for k, name in enumerate(module_names):
        if len(list(model._modules[name]._modules)) > 0:
            conv_bn_pairs = get_conv_bn_layers(model._modules[name])
            layers.extend(
                [
                    [f"{name}.{conv}", f"{name}.{bn}"]
                    for conv, bn in conv_bn_pairs
                ]
            )
        elif isinstance(model._modules[name], torch.nn.BatchNorm2d):
            previous = model._modules[module_names[k - 1]]
            if isinstance(previous, torch.nn.Conv2d):
                layers.append([module_names[k - 1], name])
    return layers


def get_aten_graph_module(
    pattern: Callable,
    example_inputs: Tuple[Any, ...],
    example_kwargs: Dict[str, Any] = None,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], None] = None,
    is_cuda: bool = False,
) -> GraphModule:
    """Convert ``pattern`` to an FX graph of decomposed aten ops.

    Args:
        pattern: Callable or module to trace.
        example_inputs: Positional example inputs.
        example_kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to export.
        is_cuda: Move tensor inputs to CUDA before tracing.

    Returns:
        The traced pattern, dead code eliminated.
    """
    if is_cuda:
        example_inputs = tuple(
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in example_inputs
        )
    aten_pattern = export_model(
        pattern,
        example_inputs,
        example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern


def create_getattr_from_value(
    module: torch.nn.Module,
    graph: Graph,
    prefix: str,
    value: Any,
    producer: Optional[GraphModule] = None,
) -> Node:
    """Register ``value`` as a buffer and return a ``get_attr`` node for it.

    Args:
        module: Module the buffer is registered on.
        graph: Graph the node is created in.
        prefix: Base attribute name; dots become underscores and a numeric
            suffix is appended until the name is unused (``s``, ``s_1``, …).
        value: Tensor or scalar to store.  A fake tensor stores a shape-only
            stand-in, which a real run replaces by running ``producer``.
        producer: A graph over the module's own buffers that computes the
            real value; kept on the node as ``meta['producer']``.

    Returns:
        The ``get_attr`` node referencing the new buffer.
    """
    prefix = prefix.replace(".", "_")
    attr_name, i = prefix, 0
    while hasattr(module, attr_name):
        i += 1
        attr_name = f"{prefix}_{i}"

    new_value = (
        value.clone().detach()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value)
    )
    module.register_buffer(attr_name, new_value)
    node = graph.create_node("get_attr", attr_name)
    if producer is not None:
        node.meta["producer"] = producer
    return node


def derived_producer(
    source: Node, fn, *args, **kwargs
) -> Optional[GraphModule]:
    """A producer for a constant that is ``fn(constant of source, *args)``,
    or ``None`` when ``source`` has no producer, so its value is real and
    needs none.  The source's producer is copied in, not referenced, so
    the derived constant outlives the source's buffer."""
    src = source.meta.get("producer")
    if src is None:
        return None
    graph = Graph()
    local = {}
    for n in src.graph.nodes:
        if n.op == "output":
            value = local[n.args[0]]
            break
        local[n] = graph.node_copy(n, lambda x: local[x])
    graph.output(graph.call_function(fn, (value, *args), kwargs))
    return GraphModule(src, graph)


def get_module_stack(node: Node) -> Dict[str, Tuple[str, Any]]:
    """The module stack capture recorded for ``node``.

    A node of an AOTAutograd backward pass has the stack of the forward op
    it differentiates.  A node traced outside any module has none.

    Args:
        node: Node of an exported or AOTAutograd graph.

    Returns:
        The stack, outermost module first; empty if there is none.
    """
    return (
        node.meta.get("nn_module_stack")
        or node.meta.get("fwd_nn_module_stack")
        or {}
    )


def get_node_name_to_scope(
    model: GraphModule,
) -> Dict[str, Tuple[str, type, int]]:
    """Map each node's name to the module scope capture recorded for it.

    Args:
        model: Exported or AOTAutograd graph module.

    Returns:
        ``node.name`` -> ``(module_path, module_type, call_index)`` taken from
        the innermost frame of the node's module stack.
    """
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = (
        defaultdict(lambda: defaultdict(int))
    )
    for n in model.graph.nodes:
        if not (nn_module_stack := get_module_stack(n)):
            node_name_to_scope[n.name] = [("", type(None))]
            continue

        current_scope = []
        for bt in nn_module_stack.values():
            module_path = bt[0]
            cur_object_type_idx = submodule_to_object_type_to_cur_idx[
                module_path
            ][n.target]
            submodule_to_object_type_to_cur_idx[module_path][n.target] += 1
            current_scope.append((module_path, bt[1], cur_object_type_idx))
        node_name_to_scope[n.name] = current_scope[-1]

    return node_name_to_scope


def print_node_scope_tabular(gm: GraphModule):
    """Print each node alongside the module scope it was traced from."""
    # Deferred: ``tabulate`` is only needed for this debugging printer.
    try:
        from tabulate import tabulate
    except ImportError:
        print(
            "`print_tabular` relies on the library `tabulate`, which could "
            "not be found on this machine. Run `pip install tabulate` to "
            "install the library."
        )
        raise

    node_name_to_scope = get_node_name_to_scope(gm)
    node_specs = [
        [n.op, n.name, n.target, node_name_to_scope[n.name]]
        for n in gm.graph.nodes
        if n.name in node_name_to_scope
    ]
    print(tabulate(node_specs, headers=["opcode", "name", "target", "scope"]))
