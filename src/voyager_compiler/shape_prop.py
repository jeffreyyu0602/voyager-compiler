"""Execute an FX graph and record each node's shape / dtype.

``ShapeProp`` walks a whole ``GraphModule``; ``propagate_shape`` is its
single-node twin, used by the passes that build nodes one at a time.  Both
resolve ``get_attr`` through ``fetch_attr`` and stamp their result with
``set_node_value``, so all four live together.
"""

import logging
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.graph import map_arg
from torch.fx.node import Node

logger = logging.getLogger(__name__)

__all__ = [
    "ShapeProp",
    "fetch_attr",
    "propagate_shape",
    "set_node_value",
]

_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond


def fetch_attr(module, target):
    """Resolve a dotted ``get_attr`` target against ``module``."""
    target_atoms = target.split(".")
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                "Node referenced nonexistant target "
                f"{'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def set_node_value(node: Node, value):
    """Record ``value`` on ``node`` as ``.value`` (plus ``.shape``)."""
    if isinstance(value, torch.Tensor):
        node.shape = value.shape
        node.value = value.cpu().clone()
    elif isinstance(value, (tuple, list)):
        # A tuple may mix tensors with scalars (e.g. the integer loop counters
        # carried by a while_loop); keep non-tensor elements as-is.
        node.shape = tuple(
            x.shape if isinstance(x, torch.Tensor) else None for x in value
        )
        node.value = tuple(
            x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in value
        )
    else:
        node.value = value


def propagate_shape(node: Node, model: GraphModule = None):
    """Run a single ``node`` on its operands' recorded values and stamp it."""

    def load_arg(a):
        return map_arg(a, lambda n: getattr(n, "value", n.meta.get("val")))

    modules = dict(model.named_modules()) if model is not None else {}

    if node.op == "get_attr":
        result = fetch_attr(model, node.target)
    elif node.op == "call_function":
        result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
    elif node.op == "call_method":
        self_obj, *args = load_arg(node.args)
        kwargs = load_arg(node.kwargs)
        result = getattr(self_obj, node.target)(*args, **kwargs)
    elif node.op == "call_module":
        result = modules[node.target](
            *load_arg(node.args), **load_arg(node.kwargs)
        )
    elif node.op == "output":
        result = load_arg(node.args[0])

    set_node_value(node, result)


def _commit_op():
    """The ``voyager`` ``commit`` HOP, resolved lazily — it is registered by
    ``ops.py``, which imports after this low-level module."""
    return getattr(torch.ops.higher_order, "commit", None)


class ShapeProp:
    """Execute a ``GraphModule`` node-by-node with the given args, recording
    each node's output ``.value`` (shape / dtype) via ``set_node_value``.

    ``recurse=True`` walks *into* the ``while_loop`` / ``cond`` HOPs and
    ``call_module`` submodules — a loop body (and its condition) propagated a
    single iteration with the carried *initial* values, each ``cond`` branch
    once — so their inner nodes are stamped too, in one pass.  The default runs
    them as opaque callables (what other callers rely on).
    """

    def __init__(
        self,
        mod,
        mode: Optional[FakeTensorMode] = None,
        recurse: bool = False,
    ):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules(remove_duplicate=False))
        self._mode = mode
        self._recurse = recurse

    def propagate(self, *args):
        with self._mode or nullcontext():
            return self._propagate(*args)

    def _subprop(self, target, inputs):
        """Recursively propagate a HOP / ``call_module`` subgraph, returning its
        output value(s)."""
        return ShapeProp(self.modules[str(target)], recurse=True).propagate(
            *inputs
        )

    def _propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return map_arg(a, lambda n: env[n.name])

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values.  We snapshot ``.value`` at the last use, not the definition:
        # an in-place-filled buffer (``voyager.alloc``) is produced empty and
        # only holds its live contents once a later node has written it.
        node_to_last_use: Dict[Node, Node] = {}
        user_to_last_uses: Dict[Node, List[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(self.graph.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = fetch_attr(self.mod, node.target)
            elif node.op == "output":
                result = load_arg(node.args[0])
            elif self._recurse and node.target is _WHILE_LOOP:
                cond_g, body_g, carried, extra = node.args
                ins = load_arg(list(carried) + list(extra))
                # Stamp the loop-condition graph too (same carried + extra).
                self._subprop(cond_g.target, ins)
                result = self._subprop(body_g.target, ins)
            elif self._recurse and node.target is _COND:
                pred, true_g, false_g, operands = node.args
                ins = load_arg(list(operands))
                taken = self._subprop(true_g.target, ins)
                other = self._subprop(false_g.target, ins)
                result = taken if load_arg(pred) else other
            elif self._recurse and node.target is _commit_op():
                # ``commit(subgraph, *operands, ...)``: stamp the region (and
                # run its kernel, mutating the destination buffer) with operand
                # values.  The dependency / post semaphores are side effects the
                # oracle-disabled ShapeProp ignores.
                sub_g, *operands = node.args
                result = self._subprop(sub_g.target, load_arg(list(operands)))
            elif node.op == "call_function":
                result = node.target(
                    *load_arg(node.args), **load_arg(node.kwargs)
                )
            elif node.op == "call_method":
                self_obj, *rest = load_arg(node.args)
                result = getattr(self_obj, node.target)(
                    *rest, **load_arg(node.kwargs)
                )
            elif node.op == "call_module":
                if self._recurse:
                    result = self._subprop(node.target, load_arg(node.args))
                else:
                    result = self.modules[node.target](
                        *load_arg(node.args), **load_arg(node.kwargs)
                    )

            env[node.name] = result

            # A node nothing consumes is never retired below, so snapshot it now
            if node not in node_to_last_use:
                set_node_value(node, result)

            # Retire any nodes whose last use is this node
            for n in user_to_last_uses.get(node, []):
                set_node_value(n, env.pop(n.name))

        return load_arg(list(self.graph.nodes)[-1])
