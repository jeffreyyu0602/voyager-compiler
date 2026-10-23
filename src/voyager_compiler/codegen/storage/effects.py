# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Reading a storage relation out of what an operation means.

The relation between a value and its backing is not a separate fact somebody
records. It follows from the operator: a reshape shares bytes, a select
windows them, an in-place add writes into its operand's, and a convolution
writes bytes of its own. This module reads that off the operator schema and
the compiler's existing view predicates, in one pass, and stops there.

Two boundaries it deliberately does not cross.

**It does not decide addresses.** Nothing here knows where anything lands.
That keeps the derivation checkable on its own: a relation graph can be
validated for consistency before an allocator has run, which is where a
contradiction is cheap to find.

**It does not decide target policy.** Alpha's concatenation slots and its
vector-scale components are relations the *target* declares, on top of these.
What the generic derivation must do is be right about the operators every
target shares, and leave a seam rather than guess.

The order below is a precedence, and it is the order it is for a reason: a
mutable schema outranks a view because ``add_`` both aliases and writes, and
reading it as an alias would lose the write. An operator that matches nothing
materializes, which is the safe default -- claiming a backing of your own is
never *wrong*, only sometimes wasteful, while claiming somebody else's when
you should not is a correctness bug.
"""

import logging
import operator
from typing import Optional

import torch
from torch.fx import Node

from voyager_compiler.codegen.node_info import is_nop
from voyager_compiler.codegen.storage.types import (
    StorageRelation,
    StorageRelationKind,
    ValueRef,
    ViewTransform,
    recorded_value,
)

logger = logging.getLogger(__name__)

__all__ = [
    "generic_storage_relation",
    "is_address_only_view",
    "mutable_storage_source",
]

#: Views whose result addresses a sub-rectangle of their source. They are
#: *address-only* when the window is contiguous in the source's layout; when
#: it is not, the same operator is a gather and has to copy.
_WINDOWING = {
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
}

#: The bufferized path's own vocabulary. These matter because the relations
#: that make a *loop* work are all here: a slot window, and the two ways a
#: region hands a buffer back to its caller.
_SUBVIEW = torch.ops.voyager.subview.default
_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond
_COMMIT = torch.ops.higher_order.commit
#: Regions that bind their operands to a subgraph's placeholders.
_REGIONS = (_WHILE_LOOP, _COND, _COMMIT)

#: Operators whose result is the source's bytes under a different shape, and
#: which ``is_nop`` does not already name. It names reshape, view, flatten,
#: squeeze, unsqueeze, as_strided, contiguous, lift_fresh_copy, a ``select``
#: over an extent-1 dimension, and everything ``is_prunable_op`` accepts -- so
#: this set exists only for what it misses.
#:
#: Consulting ``is_nop`` rather than restating it is the point. A duplicate of
#: a predicate the compiler already owns drifts, and this one had: it left out
#: ``contiguous`` (filed here with ``permute`` as a reorder, which is right for
#: a permute and wrong on an IR whose layout is normalized) and the extent-1
#: ``select``, and the second of those merged two blocks on a llama graph.
_RESHAPING = {
    torch.ops.aten.alias.default,
}

#: Operators that reorder bytes. They alias only if something downstream
#: consumes them through an address generator that can walk the permutation;
#: by themselves they are a copy, and calling them an alias is the mistake
#: that produces a tensor whose strides nobody honours.
#: ``contiguous`` is deliberately absent: ``is_nop`` calls it the same bytes,
#: and it is consulted first. A permute and a transpose really do reorder.
_REORDERING = {
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
}


def mutable_storage_source(node: Node) -> Optional[int]:
    """The argument index an in-place operator writes, or ``None``.

    Read from the operator schema rather than from a name list: ``add_``,
    ``relu_`` and ``copy_`` are recognizable by their trailing underscore, and
    an operator that mutates without one would slip past a name check while
    the schema still declares it.
    """
    schema = getattr(getattr(node, "target", None), "_schema", None)
    if schema is None:
        return None
    written = [
        index
        for index, argument in enumerate(schema.arguments)
        if getattr(argument, "alias_info", None) is not None
        and argument.alias_info.is_write
    ]
    if len(written) != 1:
        # No mutation, or more than one written operand -- which no operator
        # in this compiler's vocabulary has, and which this model could not
        # express as a single relation anyway.
        return None
    return written[0]


def is_address_only_view(node: Node, source: Node) -> bool:
    """Whether ``node``'s window can be reached by address arithmetic alone.

    A slice along the outermost dimension is a contiguous run and needs only
    a base offset. A slice along an inner one is strided, which the address
    generator can still walk. What it cannot walk is a *step* other than one:
    that is a gather, and gathers copy.
    """
    if node.target is torch.ops.aten.slice.Tensor:
        step = node.args[4] if len(node.args) > 4 else 1
        return step in (None, 1)
    if node.target is torch.ops.aten.select.int:
        return True
    del source
    return False


def _value(node: Node):
    return recorded_value(node)


def _same_bytes(node: Node, source: Node) -> bool:
    """Whether two values hold the same number of the same-width elements."""
    left, right = _value(node), _value(source)
    if left is None or right is None:
        return True  # unknown; the reshape rules already restrict the callers
    if not hasattr(left, "numel") or not hasattr(right, "numel"):
        return True
    return left.numel() == right.numel() and left.dtype == right.dtype


def generic_storage_relation(
    result: ValueRef,
    node: Node,
) -> StorageRelation:
    """The relation ``node``'s result has to its operands' storage.

    Precedence, and each step is a fact about the operator rather than a
    heuristic:

    1. a **parameter or input** materializes -- it is where bytes come from;
    2. an operator whose **schema declares a written operand** writes into it;
    3. an explicitly declared **physical copy** owns new bytes copied from
       its source;
    4. a **NOP** is the same bytes under another shape -- and which operators
       those are is asked of ``node_info.is_nop``, not restated here;
    5. a **``voyager.subview``** is a window, and the only kind that aliases;
    6. an ``aten`` **window** (``slice``, ``select``) owns its bytes: on this
       IR a slot is a subview, so one of these is the model slicing a tensor;
    7. a **getitem** names one result of a multi-output operator;
    8. a **reorder** copies -- it is the one case where the tempting answer
       (an alias, since no arithmetic happens) is wrong;
    9. anything else materializes.

    Rules 4 and 6 were both learned by disagreeing with ``_buffer_identity``
    on a llama graph, in opposite directions and one at a time.
    """
    if node.op in ("placeholder", "get_attr"):
        return StorageRelation(result, StorageRelationKind.MATERIALIZE)

    written = mutable_storage_source(node)
    if written is not None and written < len(node.args):
        source = node.args[written]
        if isinstance(source, Node):
            return StorageRelation(
                result,
                StorageRelationKind.WRITE_INTO,
                source=ValueRef(source),
                transform=ViewTransform("in_place"),
            )

    target = getattr(node, "target", None)
    first = node.args[0] if node.args else None

    # A target memory rewrite may expose movement as an ordinary FX owner.
    # The marker is an explicit semantic declaration, not a target-policy
    # inference: the result owns distinct bytes copied from its input.  Keep
    # this ahead of the NOP rules because a value-preserving copy can look
    # reshape-like while still requiring its own backing until the target
    # binds the final destination relation.
    if node.meta.get("physical_copy") is True and isinstance(first, Node):
        return StorageRelation(
            result,
            StorageRelationKind.COPY,
            source=ValueRef(first),
        )

    if (is_nop(node) or target in _RESHAPING) and isinstance(first, Node):
        if _same_bytes(node, first):
            return StorageRelation(
                result,
                StorageRelationKind.ALIAS,
                source=ValueRef(first),
            )
        # A reshape that does not preserve the byte count is not a reshape of
        # this tensor's storage, whatever it is called.
        logger.debug(
            "%s changes its element count; treating it as a copy", node.name
        )
        return StorageRelation(
            result, StorageRelationKind.COPY, source=ValueRef(first)
        )

    if target in _WINDOWING and isinstance(first, Node):
        # A window *owns its bytes*, even when address arithmetic could reach
        # it. On this compiler's IR the window vocabulary is
        # ``voyager.subview``; an ``aten.select`` or ``aten.slice`` that
        # survives to here is the **model** slicing a tensor, not a slot, and
        # ``_viewed_buffer`` says the same in as many words.
        #
        # Read as a view instead, several slices of one tensor collapse into
        # one backing and drag everything reading them along: on a llama
        # decode graph that merged six loop-body placeholders into two blocks
        # that ``_buffer_identity`` keeps apart -- a claim on somebody else's
        # storage, which is the one direction that is a correctness bug rather
        # than waste.
        #
        # ``is_address_only_view`` stays, because the geometry is still true
        # and a *target* may declare a window on top of it. What is not safe
        # is assuming it here, for every graph.
        return StorageRelation(
            result,
            StorageRelationKind.COPY,
            source=ValueRef(first),
            transform=_window_transform(node),
        )

    if target is _SUBVIEW and isinstance(first, Node):
        # A window onto a buffer -- the slot a pipeline step reads or writes.
        # Address-only by construction: it exists precisely so a step can name
        # part of a buffer without moving anything.
        return StorageRelation(
            result,
            StorageRelationKind.VIEW,
            source=ValueRef(first),
            transform=ViewTransform("subview"),
        )

    if target is operator.getitem and isinstance(first, Node):
        index = node.args[1]
        if isinstance(index, int):
            region = _region_result(result, first, index)
            if region is not None:
                return region
            producer_value = recorded_value(first)
            if isinstance(producer_value, (tuple, list)):
                length = len(producer_value)
                if index < 0 and -length <= index:
                    index += length
            # The path names which result, so the relation is to the producer
            # as a whole. A getitem that itself returns an aggregate has one
            # relation per tensor leaf, so retain the leaf's relative path.
            return StorageRelation(
                result,
                StorageRelationKind.ALIAS,
                source=ValueRef(first, (index, *result.path)),
            )

    if target in _REORDERING and isinstance(first, Node):
        return StorageRelation(
            result, StorageRelationKind.COPY, source=ValueRef(first)
        )

    return StorageRelation(result, StorageRelationKind.MATERIALIZE)


def _region_result(result: ValueRef, producer: Node, index: int):
    """What one result of a control-flow region actually is.

    Both cases are places where a *new name* appears for a buffer that already
    existed, and reading them as fresh allocations is how a loop's accumulator
    ends up co-live with itself:

    ``while_loop``
        The loop writes each carried buffer in place and hands it back. So
        result *i* is the *i*-th carried operand -- not a copy of it, and not
        a value of the loop's own. An accumulator threaded
        ``alloc -> body arg -> add -> getitem`` is one buffer with one
        lifetime, and only this makes it so.
    ``cond``
        The branches compute into the destination their ``insert`` names, so
        the conditional owns no storage of its own; its result is the region.

    Returns ``None`` for anything that is not one of those, so an ordinary
    multi-output operator keeps its own answer.
    """
    if producer.target is _WHILE_LOOP:
        carried = producer.args[2] if len(producer.args) > 2 else ()
        if index < len(carried) and isinstance(carried[index], Node):
            return StorageRelation(
                result=result,
                kind=StorageRelationKind.WRITE_INTO,
                source=ValueRef(carried[index]),
                transform=ViewTransform("loop_carried"),
            )
    if producer.target is _COND:
        return StorageRelation(
            result=result,
            kind=StorageRelationKind.ALIAS,
            source=ValueRef(producer),
        )
    return None


def _window_transform(node: Node) -> ViewTransform:
    """The geometry a windowing operator applies, as named parameters."""
    if node.target is torch.ops.aten.select.int:
        return ViewTransform(
            "select",
            (("dim", node.args[1]), ("index", node.args[2])),
        )
    dim = node.args[1] if len(node.args) > 1 else 0
    start = node.args[2] if len(node.args) > 2 else None
    end = node.args[3] if len(node.args) > 3 else None
    return ViewTransform(
        "slice", (("dim", dim), ("start", start), ("end", end))
    )
