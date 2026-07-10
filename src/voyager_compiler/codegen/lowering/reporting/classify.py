"""Map a bufferized FX node to a scheduling ``kind``.

Reuses the dialect's own target handles and predicates so the estimator stays
in lock-step with the lowering: the compute test mirrors bufferization's
``_is_compute`` (``call_module`` fused groups, or a bare tensor-producing op
that ``is_compute_op`` accepts).  ``insert`` is *not* a transfer here — it is
the destination-passing write of a compute result into its buffer, so it is
zero-time control.
"""

import torch
from torch.fx import Node

from ..bufferization import _produces_tensor
from ..codegen import COND, WHILE_LOOP
from ...mapping_utils import is_compute_op

ASYNC_COPY = torch.ops.voyager.async_copy.default
ASYNC_WAIT = torch.ops.voyager.async_wait.default
INSERT = torch.ops.voyager.insert.default


def classify(node: Node) -> str:
    """One of: ``while_loop``, ``cond``, ``compute``, ``async_copy``,
    ``async_wait``, ``insert``, or ``control`` (everything zero-time:
    allocs, zeros, index math, selects, getitems)."""
    if node.op == "call_module":
        return "compute"
    t = node.target
    if t is WHILE_LOOP:
        return "while_loop"
    if t is COND:
        return "cond"
    if t is ASYNC_COPY:
        return "async_copy"
    if t is ASYNC_WAIT:
        return "async_wait"
    if t is INSERT:
        return "insert"
    if _produces_tensor(node) and is_compute_op(node):
        return "compute"
    return "control"
