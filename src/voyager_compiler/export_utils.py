"""Capture an FX graph, add to one, and read what capture recorded.

``export_model`` / ``get_aten_graph_module`` turn eager code into a
``GraphModule``; ``create_getattr_from_value`` materialises a tensor into an
existing graph as a buffer; ``get_node_name_to_scope`` /
``print_node_scope_tabular`` read the ``nn_module_stack`` provenance that
export stamps on every node.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.fx import Graph, GraphModule, Node
from transformers.utils import is_torch_greater_or_equal

logger = logging.getLogger(__name__)

__all__ = [
    "create_getattr_from_value",
    "export_model",
    "get_aten_graph_module",
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

    Picks the newest export entry point the installed torch offers, and
    suppresses the ``_assert_tensor_metadata`` nodes that each ``.to(dtype)``
    would otherwise pin to the dtype seen at trace time.

    Args:
        model: Module to export.
        args: Positional example inputs.
        kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to ``torch.export``.
        strict: Whether export runs under torchdynamo.

    Returns:
        The exported program's ``GraphModule``.

    Raises:
        RuntimeError: If the installed torch predates 2.0.
    """
    export_args = (model, args, kwargs)
    export_kwargs = {"dynamic_shapes": dynamic_shapes, "strict": strict}

    if is_torch_greater_or_equal("2.10"):
        from torch._export.utils import (
            _disable_aten_to_metadata_assertions,
        )

        with _disable_aten_to_metadata_assertions():
            gm = torch.export.export(*export_args, **export_kwargs)
        return gm.module(check_guards=False)
    elif is_torch_greater_or_equal("2.8"):
        from torch._export.utils import (
            _disable_aten_to_metadata_assertions,
        )

        with _disable_aten_to_metadata_assertions():
            gm = torch.export.export_for_training(*export_args, **export_kwargs)
        return gm.module()
    elif is_torch_greater_or_equal("2.5"):
        return torch.export.export_for_training(
            *export_args, **export_kwargs
        ).module()
    elif is_torch_greater_or_equal("2.0"):
        return torch._export.capture_pre_autograd_graph(
            model, args, kwargs, dynamic_shapes=dynamic_shapes
        )
    else:
        raise RuntimeError(f"Require torch>=2.0, but found {torch.__version__}")


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
    module: torch.nn.Module, graph: Graph, prefix: str, value: Any
) -> Node:
    """Register ``value`` as a buffer and return a ``get_attr`` node for it.

    Args:
        module: Module the buffer is registered on.
        graph: Graph the node is created in.
        prefix: Base attribute name; dots become underscores and a numeric
            suffix is appended until the name is unused (``s``, ``s_1``, …).
        value: Tensor or scalar to store.

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
    return graph.create_node("get_attr", attr_name)


def get_node_name_to_scope(
    model: GraphModule,
) -> Dict[str, Tuple[str, type, int]]:
    """Map each node's name to the module scope export recorded for it.

    Args:
        model: Exported graph module.

    Returns:
        ``node.name`` -> ``(module_path, module_type, call_index)`` taken from
        the innermost frame of the node's ``nn_module_stack``.
    """
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = (
        defaultdict(lambda: defaultdict(int))
    )
    for n in model.graph.nodes:
        if (nn_module_stack := n.meta.get("nn_module_stack", None)) is None:
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
