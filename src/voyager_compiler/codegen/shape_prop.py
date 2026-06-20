import logging
from typing import Dict, List, Optional

import torch
from torch.fx.graph import map_arg
from torch.fx.node import Node
from torch._subclasses.fake_tensor import FakeTensorMode

logger = logging.getLogger(__name__)


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod, mode: Optional[FakeTensorMode] = None):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self._mode = mode

    def propagate(self, *args):
        if self._mode is not None:
            with self._mode:
                return self._propagate(*args)
        else:
            return self._propagate(*args)

    def _propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use: Dict[Node, Node] = {}
        user_to_last_uses: Dict[Node, List[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(self.graph.nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        def store_value(node: Node, value):
            if isinstance(value, torch.Tensor):
                node.shape = value.shape
                node.value = value.cpu().clone()
            elif isinstance(value, (tuple, list)):
                # Tuples may mix tensors with scalars (e.g. integer loop counters
                # carried by a while_loop); keep non-tensor elements as-is.
                node.shape = tuple(
                    x.shape if isinstance(x, torch.Tensor) else None
                    for x in value
                )
                node.value = tuple(
                    x.cpu().clone() if isinstance(x, torch.Tensor) else x
                    for x in value
                )
            else:
                node.value = value
                logger.debug(f"Node {node} produced non-tensor output {value}")

        def delete_unused_values(user: Node):
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.

            We snapshot ``node.value`` *here*, at the node's last use, rather than
            when it was defined: a buffer (e.g. ``voyager.alloc``) is produced
            empty and filled *in place* by a later node — a ``while_loop`` body's
            tile DMA — so its definition-time value is still uninitialized.  By
            its last use ``env[n.name]`` holds the live, filled object.
            """
            for n in user_to_last_uses.get(user, []):
                store_value(n, env.pop(n.name))

        for node in self.graph.nodes:
            try:
                if node.op == "placeholder":
                    result = next(args_iter)
                elif node.op == "get_attr":
                    result = fetch_attr(node.target)
                elif node.op == "call_function":
                    result = node.target(
                        *load_arg(node.args), **load_arg(node.kwargs)
                    )
                elif node.op == "call_method":
                    self_obj, *args = load_arg(node.args)
                    kwargs = load_arg(node.kwargs)
                    result = getattr(self_obj, node.target)(*args, **kwargs)
                elif node.op == "call_module":
                    result = self.modules[node.target](
                        *load_arg(node.args), **load_arg(node.kwargs)
                    )
                elif node.op == "output":
                    result = load_arg(node.args[0])
            except:
                self.graph.print_tabular()
                print(f"Error in node {node}")
                raise

            env[node.name] = result

            # A node nothing consumes is never retired below, so snapshot it now
            if node not in node_to_last_use:
                store_value(node, result)

            delete_unused_values(node)

        return load_arg(list(self.graph.nodes)[-1])
