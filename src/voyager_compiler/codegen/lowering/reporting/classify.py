"""Map a bufferized FX node to a scheduling ``kind``.

Reuses the dialect's own target handles and predicates so the estimator stays
in lock-step with the lowering: the control / DMA op set is bufferization's
``_NON_COMPUTE``; the compute test mirrors bufferization's ``_is_compute``
(``call_module`` fused groups, or a bare tensor-producing op that is not
control / NOP).  ``copy_tile`` is *not* a transfer here — it is the
destination-passing write of a compute result into its buffer, so it is
zero-time control.
"""

import torch
from torch.fx import Node

from ..bufferization import _NON_COMPUTE, _produces_tensor
from ..codegen import COND, WHILE_LOOP
from ...mapping_utils import (
    is_indexing_or_concatenation_op,
    is_memory_op,
    is_nop,
    is_reshape_op,
)

ASYNC_COPY = torch.ops.voyager.async_copy.default
ASYNC_WAIT = torch.ops.voyager.async_wait.default
COPY_TILE = torch.ops.voyager.copy_tile.default


def classify(node: Node) -> str:
    """One of: ``while_loop``, ``cond``, ``compute``, ``async_copy``,
    ``async_wait``, ``copy_tile``, or ``control`` (everything zero-time:
    allocs, zeros, index math, selects, getitems)."""
    if node.op == "call_module":
        return "compute"
    if node.op != "call_function":
        return "control"
    t = node.target
    if t is WHILE_LOOP:
        return "while_loop"
    if t is COND:
        return "cond"
    if t is ASYNC_COPY:
        return "async_copy"
    if t is ASYNC_WAIT:
        return "async_wait"
    if t is COPY_TILE:
        return "copy_tile"
    # A bare tile-compute op (e.g. the no-accumulate ``aten.linear`` in a
    # reduction loop's first step): produces a tensor and is not control / NOP
    # / data movement (memory / reshape / indexing ops are kept bare and cost
    # no systolic time).
    if (
        _produces_tensor(node)
        and t not in _NON_COMPUTE
        and not is_nop(node)
        and not is_memory_op(node)
        and not is_reshape_op(node)
        and not is_indexing_or_concatenation_op(node)
    ):
        return "compute"
    return "control"
