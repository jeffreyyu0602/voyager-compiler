import logging
from typing import Dict, List, Optional

import torch
from torch.fx.graph import map_arg
from torch.fx.node import Node
from torch._subclasses.fake_tensor import FakeTensorMode

from ..pt2e_utils import fetch_attr, set_node_value

logger = logging.getLogger(__name__)

_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond


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
        if self._mode is not None:
            with self._mode:
                return self._propagate(*args)
        else:
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
