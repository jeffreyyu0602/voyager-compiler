"""Lower single-element arithmetic to control-processor scalars.

A decode graph does its index bookkeeping -- the cache position, the slot a
token lands in, the chunk it belongs to -- as arithmetic on one-element
``int64`` tensors, because that is how PyTorch spells it; a training step
computes its optimizer's per-step factors and its loss's final division on
one-element float tensors.  Lowered as tensor ops, each becomes a
vector-unit kernel with a DMA in and out for a single number.  This pass
rewrites such a chain into what the accelerator's control processor does
natively: one read of each source tensor (``_local_scalar_dense``),
Python-scalar arithmetic on the result (emitted as ``cpu`` ops), and, where
a consumer needs a tensor again, a host-written one (``full`` for a single
element, ``arange`` for an index vector shifted by a scalar).
"""

import logging
import math
import operator

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.shape_prop import (
    constant_value,
    fetch_attr,
    propagate_shape,
)

logger = logging.getLogger(__name__)

__all__ = ["scalarize_index_arithmetic"]

aten = torch.ops.aten


# The scalar ops below are emitted by name, as ``cpu`` ops.
def truediv(a, b):
    """``a / b`` as a tensor divides it: by zero, an infinity or a NaN."""
    if b == 0:
        if a == 0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def rsub(a, b):
    """``b - a``, as ``aten.rsub`` computes it."""
    return b - a


def reciprocal(a):
    """``1 / a``, as ``aten.reciprocal`` computes it."""
    return truediv(1.0, a)


def sqrt(a):
    """The square root of ``a`` as a tensor takes it: of a negative, NaN."""
    return math.sqrt(a) if a >= 0 else math.nan

# Elementwise ops that mean the same on Python scalars.  ``floor_divide``
# and ``remainder`` follow Python's floor semantics, as ``//`` and ``%`` do.
_SCALAR_OPS = {
    aten.add.Tensor: operator.add,
    aten.add.Scalar: operator.add,
    aten.sub.Tensor: operator.sub,
    aten.sub.Scalar: operator.sub,
    aten.mul.Tensor: operator.mul,
    aten.mul.Scalar: operator.mul,
    aten.floor_divide.default: operator.floordiv,
    aten.floor_divide.Scalar: operator.floordiv,
    aten.remainder.Tensor: operator.mod,
    aten.remainder.Scalar: operator.mod,
    aten.neg.default: operator.neg,
    aten.eq.Tensor: operator.eq,
    aten.eq.Scalar: operator.eq,
    aten.ne.Tensor: operator.ne,
    aten.ne.Scalar: operator.ne,
    aten.lt.Tensor: operator.lt,
    aten.lt.Scalar: operator.lt,
    aten.le.Tensor: operator.le,
    aten.le.Scalar: operator.le,
    aten.gt.Tensor: operator.gt,
    aten.gt.Scalar: operator.gt,
    aten.ge.Tensor: operator.ge,
    aten.ge.Scalar: operator.ge,
    aten.div.Tensor: truediv,
    aten.div.Scalar: truediv,
    aten.rsub.Scalar: rsub,
    aten.pow.Scalar: operator.pow,
    aten.pow.Tensor_Scalar: operator.pow,
    aten.pow.Tensor_Tensor: operator.pow,
    aten.sqrt.default: sqrt,
    aten.reciprocal.default: reciprocal,
}

_INTEGER_DTYPES = (torch.int64, torch.int32, torch.bool)


def _single_element(node) -> bool:
    """Whether ``node`` is a one-element tensor."""
    value = getattr(node, "value", None)
    return isinstance(value, torch.Tensor) and value.numel() == 1


def _index_vector(model: GraphModule, node):
    """``(first, step, count)`` if ``node`` is a constant 1-D integer tensor
    with a uniform stride (an ``arange``), else ``None``."""
    if not isinstance(node, Node) or node.op != "get_attr":
        return None
    value = fetch_attr(model, node.target)
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() < 2
        or value.dtype not in _INTEGER_DTYPES
    ):
        return None
    value = constant_value(model, node)
    steps = value[1:] - value[:-1]
    if not bool((steps == steps[0]).all()):
        return None
    return int(value[0]), int(steps[0]), value.numel()


def scalarize_index_arithmetic(model: GraphModule) -> GraphModule:
    """Rewrite arithmetic on one-element tensors into scalar ops.

    Walking in program order, an elementwise op in ``_SCALAR_OPS`` whose
    tensor operands are all single-element tensors becomes the
    matching Python operator on scalars: a constant operand folds to its
    value, a computed one is read once with ``_local_scalar_dense``, and an
    already scalarized one is used as is.  A consumer that needs a tensor
    gets one back with ``full`` in the op's original shape.  An ``add`` of
    such a scalar to a constant index vector (the chunk offsets ``c * R +
    arange(R)``) becomes an ``arange`` starting at the scalar, so the vector
    is written by the host rather than computed.

    Args:
        model: The graph module to rewrite in place, shape-propagated.

    Returns:
        ``model``, rewritten in place.
    """
    graph = model.graph
    scalar_of = {}  # tensor node -> the scalar node standing in for it
    reads = {}  # tensor node -> its _local_scalar_dense read

    def scalar(operand, before):
        """``operand`` as a scalar: a literal, a scalar node, or a read."""
        if not isinstance(operand, Node):
            return operand
        if operand in scalar_of:
            return scalar_of[operand]
        if operand.op == "get_attr":
            return constant_value(model, operand).item()
        if operand not in reads:
            with graph.inserting_before(before):
                reads[operand] = graph.call_function(
                    aten._local_scalar_dense.default, (operand,)
                )
            propagate_shape(reads[operand], model)
        return reads[operand]

    def materialize(node, value, before):
        """A tensor holding scalar ``value``, in ``node``'s shape and dtype,
        written by the host."""
        original = node.value
        with graph.inserting_before(before):
            tensor = graph.call_function(
                aten.full.default,
                (list(original.shape), value),
                {"dtype": original.dtype, "device": original.device},
            )
        propagate_shape(tensor, model)
        return tensor

    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _SCALAR_OPS:
            continue
        tensors = [a for a in node.all_input_nodes]
        if not tensors:
            continue
        vector = None
        if (
            node.target in (aten.add.Tensor, aten.add.Scalar)
            and len(node.args) == 2
        ):
            for index, other in ((0, 1), (1, 0)):
                if (
                    _index_vector(model, node.args[index]) is not None
                    and node.args[other] in scalar_of
                ):
                    vector = (node.args[index], node.args[other])
        if vector is None and not (
            _single_element(node)
            and all(_single_element(a) or a in scalar_of for a in tensors)
        ):
            continue

        if vector is not None:
            constant, shift = vector
            first, step, size = _index_vector(model, constant)
            with graph.inserting_before(node):
                start = graph.call_function(
                    operator.add, (scalar_of[shift], first)
                )
                end = graph.call_function(operator.add, (start, step * size))
                new = graph.call_function(
                    aten.arange.start_step,
                    (start, end, step),
                    {"dtype": node.value.dtype, "device": node.value.device},
                )
            for n in (start, end, new):
                propagate_shape(n, model)
            node.replace_all_uses_with(new)
            graph.erase_node(node)
            count += 1
            continue

        args = [scalar(a, node) for a in node.args]
        with graph.inserting_before(node):
            new = graph.call_function(_SCALAR_OPS[node.target], tuple(args))
        propagate_shape(new, model)
        scalar_of[node] = new
        # Consumers that still want the tensor read a host-written one.
        tensor = None
        for user in list(node.users):
            # A consumer scalarized in its turn, one element or an index
            # vector shifted by this scalar, reads the scalar.
            if user.target in _SCALAR_OPS and (
                _single_element(user)
                or (
                    user.target in (aten.add.Tensor, aten.add.Scalar)
                    and any(_index_vector(model, a) for a in user.args)
                )
            ):
                continue
            if tensor is None:
                tensor = materialize(node, new, user)
            user.replace_input_with(node, tensor)
        count += 1

    # A scalarized op whose consumers were all scalarized is now unused, as
    # is a tensor op feeding only reads that a read replaced.
    for node in reversed(list(graph.nodes)):
        if (
            node.op == "call_function"
            and not node.users
            and node.target in _SCALAR_OPS
        ):
            graph.erase_node(node)
    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info(f"[transform] scalarized {count} index ops")
    return model
