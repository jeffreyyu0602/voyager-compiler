"""Execute an FX graph and record each node's shape / dtype.

``ShapeProp`` walks a whole ``GraphModule``; ``propagate_shape`` is its
single-node twin, used by the passes that build nodes one at a time.  Both
resolve ``get_attr`` through ``fetch_attr`` and stamp their result with
``set_node_value``, so all four live together.

Lowering reads shapes and dtypes, never activation contents, so a pass
runs the graph on fake tensors: the callers fake the model inputs
(``fake_like``), an input-free generator (``arange``, ``full``,
``voyager.alloc``) is fake, and a node is real only when every input it
has is real -- a parameter, or a value computed from parameters alone,
which a pass may bake into a buffer.  Nothing past the fake frontier is
allocated, whatever its size.  A propagation given real inputs is a real
run (``materialized_values``): generators run for real and a constant the
folder deferred is computed from its producer and installed, so the
emitter's tensor dump and a numeric check see real values.
"""

import logging
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule
from torch.fx.graph import map_arg
from torch.fx.node import Node
from torch.utils._pytree import tree_flatten

logger = logging.getLogger(__name__)

__all__ = [
    "ShapeProp",
    "constant_value",
    "fake_like",
    "fetch_attr",
    "materialized_values",
    "propagate_shape",
    "run_op",
    "set_node_value",
    "written_buffers",
]

# One mode for every fake value: fakes of two modes cannot meet in an op.
# ``allow_non_fake_inputs`` lets a fake operand combine with a real one, as
# happens whenever a tile of a real weight meets a fake activation.
_FAKE_MODE = FakeTensorMode(allow_non_fake_inputs=True)

_materialize = False


@contextmanager
def materialized_values():
    """Run everything for real inside the block: generators too, and the
    constants the folder deferred."""
    global _materialize
    saved, _materialize = _materialize, True
    try:
        yield
    finally:
        _materialize = saved


def fake_like(value):
    """A storage-free stand-in for a real tensor, of its shape, dtype and
    strides; a fake tensor or a non-tensor is returned as it is.  The
    callers that propagate a model's inputs fake them with this."""
    if isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor):
        return _FAKE_MODE.from_tensor(value)
    return value


_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond
_HOPS = (_WHILE_LOOP, _COND)
_LOCAL_SCALAR = torch.ops.aten._local_scalar_dense.default


def run_op(fn, args, kwargs):
    """Call ``fn`` on ``args`` / ``kwargs`` and return its result.  An op
    with a tensor operand is called directly: real operands make a real
    result, and a fake one makes a fake result through its own dispatch,
    while any real tensor the op touches on the side (a module's enable
    flag, a parameter) stays real.  An op with no tensor operand is a
    generator and runs under the fake mode, so its result is fake -- except
    inside ``materialized_values``, where everything runs for real.  A
    scalar read off a fake buffer (``_local_scalar_dense``, a bufferized
    nest's datapath-to-control boundary) is a zero of the buffer's dtype:
    no lowering decision reads it."""
    if fn is _LOCAL_SCALAR and isinstance(args[0], FakeTensor):
        return torch.zeros((), dtype=args[0].dtype).item()
    if _materialize or any(
        isinstance(t, torch.Tensor) for t in tree_flatten((args, kwargs))[0]
    ):
        return fn(*args, **kwargs)
    with _FAKE_MODE:
        return fn(*args, **kwargs)


def written_buffers(graph) -> set:
    """Targets of the ``get_attr`` buffers the graph writes in place: the
    destination of an in-place ATen op (a KV-cache ``index_copy_``), or an
    operand of a ``cond`` / ``while_loop`` region, whose body may write it.
    Such a buffer must not be replaced by a compile-time copy."""
    written = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if target in _HOPS or target is _commit_op():
            operands = node.all_input_nodes
        elif (
            isinstance(target, torch._ops.OpOverload)
            and target._schema.name.endswith("_")
            and node.args
        ):
            operands = [node.args[0]]
        else:
            continue
        for operand in operands:
            if isinstance(operand, Node) and operand.op == "get_attr":
                written.add(operand.target)
    return written


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


def _record(value: torch.Tensor) -> torch.Tensor:
    """The tensor stored for ``value``: itself, on CPU.  A fake tensor has
    no storage to move."""
    return value if isinstance(value, FakeTensor) else value.cpu()


def set_node_value(node: Node, value):
    """Record ``value`` on ``node`` as ``.value`` (plus ``.shape``)."""
    if isinstance(value, torch.Tensor):
        node.shape = value.shape
        node.value = _record(value)
    elif isinstance(value, (tuple, list)):
        # A tuple may mix tensors with scalars (e.g. the integer loop counters
        # carried by a while_loop); keep non-tensor elements as-is.
        node.shape = tuple(
            x.shape if isinstance(x, torch.Tensor) else None for x in value
        )
        node.value = tuple(
            _record(x) if isinstance(x, torch.Tensor) else x for x in value
        )
    else:
        node.value = value


def constant_value(module, node: Node):
    """The real tensor a ``get_attr`` node resolves to, for a pass that must
    read a constant's contents: a constant the folder deferred is computed
    from its producer and installed in place of its fake stand-in.  A fake
    with no producer is returned as it is."""
    value = fetch_attr(module, node.target)
    producer = node.meta.get("producer")
    if producer is not None and isinstance(value, FakeTensor):
        value = producer()
        *path, name = node.target.split(".")
        parent = fetch_attr(module, ".".join(path)) if path else module
        setattr(parent, name, value)
    return value


def _attr_value(module, node: Node):
    """The tensor a ``get_attr`` node resolves to: in a real run, the real
    one."""
    if _materialize:
        return constant_value(module, node)
    return fetch_attr(module, node.target)


def propagate_shape(node: Node, model: GraphModule = None):
    """Run a single ``node`` on its operands' recorded values and stamp it."""

    def load_arg(a):
        return map_arg(a, lambda n: getattr(n, "value", n.meta.get("val")))

    modules = dict(model.named_modules()) if model is not None else {}

    if node.op == "get_attr":
        result = _attr_value(model, node)
    elif node.op == "call_function":
        result = run_op(node.target, load_arg(node.args), load_arg(node.kwargs))
    elif node.op == "call_method":
        self_obj, *args = load_arg(node.args)
        result = run_op(
            getattr(self_obj, node.target), args, load_arg(node.kwargs)
        )
    elif node.op == "call_module":
        result = run_op(
            modules[node.target], load_arg(node.args), load_arg(node.kwargs)
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

    The module's own state is left as found: a buffer the graph writes in
    place (a KV cache) is run on a copy, and a ``get_attr`` is stamped with
    the value it holds when the graph starts, before any write of the step.

    Fake inputs make a fake run, the passes' mode; inputs with no fake
    tensor among them make a real run (see ``materialized_values``).
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
        real = not any(isinstance(t, FakeTensor) for t in tree_flatten(args)[0])
        with self._mode or nullcontext():
            with materialized_values() if real else nullcontext():
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

        written = written_buffers(self.graph)

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = _attr_value(self.mod, node)
                if node.target in written and isinstance(result, torch.Tensor):
                    result = result.clone()
                set_node_value(node, result)
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
                result = run_op(
                    node.target, load_arg(node.args), load_arg(node.kwargs)
                )
            elif node.op == "call_method":
                self_obj, *rest = load_arg(node.args)
                result = run_op(
                    getattr(self_obj, node.target), rest, load_arg(node.kwargs)
                )
            elif node.op == "call_module":
                if self._recurse:
                    result = self._subprop(node.target, load_arg(node.args))
                else:
                    result = run_op(
                        self.modules[node.target],
                        load_arg(node.args),
                        load_arg(node.kwargs),
                    )

            env[node.name] = result

            # A node nothing consumes is never retired below, so snapshot it now
            if node not in node_to_last_use and node.op != "get_attr":
                set_node_value(node, result)

            # Retire any nodes whose last use is this node.  A ``get_attr``
            # was stamped at its definition, with the step's input state.
            for n in user_to_last_uses.get(node, []):
                value = env.pop(n.name)
                if n.op != "get_attr":
                    set_node_value(n, value)

        return load_arg(list(self.graph.nodes)[-1])
