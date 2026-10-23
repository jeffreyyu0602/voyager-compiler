# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Explicit non-accelerator boundaries.

An accelerator may support an operation in general and still be unable to run
one particular *instance* of it -- a pad the datapath cannot absorb, an add
whose operands land in the wrong domain, a conv whose accumulator would
overflow. That is not "this op is unsupported"; it is "this call is a
boundary, for this reason".

The compiler records that as a structured declaration on the node rather than
as a bare flag, because a refusal nobody can explain is indistinguishable from
a bug. ``emit`` serializes a declared call with ``op: "unsupported"``, which is
how a downstream consumer learns the accelerator will not execute it.

Only the *compiler* may declare this. A converter or runtime that meets a call
it cannot translate has hit a gap, not a boundary, and must fail rather than
decide the question for the compiler.
"""

from typing import Mapping, Optional, Sequence

from torch.fx import Node

#: Where the declaration lives on the node.
UNSUPPORTED_OPERATION_META = "unsupported_operation"
#: The value ``emit`` writes into ``PrimOp.op``.
SERIALIZED_UNSUPPORTED_OPERATION = "unsupported"
#: Marks a record as a deliberate declaration rather than incidental meta.
DECLARED_UNSUPPORTED_DISPOSITION = "declared_unsupported"

_SCHEMA_VERSION = 1
_REQUIRED_TEXT = ("reason_code", "detail", "stage")

__all__ = [
    "DECLARED_UNSUPPORTED_DISPOSITION",
    "SERIALIZED_UNSUPPORTED_OPERATION",
    "UNSUPPORTED_OPERATION_META",
    "declare_unsupported_operation",
    "is_declared_unsupported_operation",
    "propagate_unsupported_operation",
    "validate_unsupported_operation_declaration",
]


def _source_operation(node: Node) -> dict:
    """A stable description of what was refused, independent of node naming."""
    target = node.target
    return {
        "name": node.name,
        "op": node.op,
        "target": getattr(target, "_name", None) or str(target),
    }


def declare_unsupported_operation(
    node: Node,
    *,
    reason_code: str,
    detail: str,
    stage: str,
    facts: Optional[Mapping] = None,
) -> dict:
    """Declare one call an explicit non-accelerator boundary.

    Args:
        node: The call being refused. Only a call owner may be declared: a
            placeholder or an output is not something the accelerator runs.
        reason_code: Stable machine-readable cause, e.g.
            ``"tiling_does_not_fit"``. Meant to be grepped and counted.
        detail: One human-readable sentence saying what the hardware could
            not do.
        stage: Which pass decided, so a reader knows where to look.
        facts: Optional evidence -- shapes, capacities, the numbers that made
            the call impossible.

    Returns:
        The normalized declaration.

    Raises:
        ValueError: If the node is not a call, a required field is missing, or
            the node already carries a *different* declaration. Two passes
            disagreeing about why a call is a boundary is a bug, not a merge.
    """
    if node.op not in {"call_function", "call_module"}:
        raise ValueError(
            f"{node.name}: only call owners may be declared unsupported"
        )
    for field, value in (
        ("reason_code", reason_code),
        ("detail", detail),
        ("stage", stage),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{node.name}: unsupported declaration needs {field}"
            )
    if facts is not None and not isinstance(facts, Mapping):
        raise TypeError("unsupported declaration facts must be a mapping")

    declaration = {
        "schema_version": _SCHEMA_VERSION,
        "disposition": DECLARED_UNSUPPORTED_DISPOSITION,
        "reason_code": reason_code,
        "detail": detail,
        "stage": stage,
        "source_operations": (_source_operation(node),),
        "facts": dict(facts or {}),
    }

    previous = node.meta.get(UNSUPPORTED_OPERATION_META)
    if previous is not None:
        previous = validate_unsupported_operation_declaration(node)
        if previous != declaration:
            raise ValueError(
                f"{node.name}: conflicting unsupported-operation declaration "
                f"({previous['reason_code']} vs {reason_code})"
            )
        return previous

    node.meta[UNSUPPORTED_OPERATION_META] = declaration
    return validate_unsupported_operation_declaration(node)


def validate_unsupported_operation_declaration(node: Node) -> dict:
    """Validate the declaration on ``node`` and return it normalized."""
    declaration = node.meta.get(UNSUPPORTED_OPERATION_META)
    if not isinstance(declaration, Mapping):
        raise TypeError(
            f"{node.name}: invalid unsupported-operation declaration"
        )
    if declaration.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"{node.name}: unsupported declaration has an unknown schema"
        )
    if declaration.get("disposition") != DECLARED_UNSUPPORTED_DISPOSITION:
        raise ValueError(
            f"{node.name}: unsupported declaration has invalid disposition"
        )
    for field in _REQUIRED_TEXT:
        value = declaration.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{node.name}: unsupported declaration needs {field}"
            )
    sources = declaration.get("source_operations")
    if not isinstance(sources, (tuple, list)) or not sources:
        raise TypeError(
            f"{node.name}: unsupported declaration needs source_operations"
        )
    if not isinstance(declaration.get("facts"), Mapping):
        raise TypeError(f"{node.name}: unsupported declaration needs facts")
    return dict(declaration)


def is_declared_unsupported_operation(node: Node) -> bool:
    """Whether ``node`` carries a valid explicit declaration."""
    if UNSUPPORTED_OPERATION_META not in node.meta:
        return False
    validate_unsupported_operation_declaration(node)
    return True


def propagate_unsupported_operation(
    nodes: Sequence[Node],
    owner: Node,
) -> None:
    """Move members' declarations onto the fused call that replaces them.

    Once operations fuse into one call, the boundary belongs to the call: a
    consumer sees the fused op, not its members. Members that agree collapse
    to one declaration; members that disagree are a bug.
    """
    declarations = [
        validate_unsupported_operation_declaration(node)
        for node in nodes
        if UNSUPPORTED_OPERATION_META in node.meta
    ]
    if not declarations:
        return

    merged = dict(declarations[0])
    sources = []
    for declaration in declarations:
        if declaration["reason_code"] != merged["reason_code"]:
            raise ValueError(
                f"{owner.name}: fused members disagree on why they are "
                f"unsupported ({merged['reason_code']} vs "
                f"{declaration['reason_code']})"
            )
        sources.extend(declaration["source_operations"])
    merged["source_operations"] = tuple(sources)
    owner.meta[UNSUPPORTED_OPERATION_META] = merged
    validate_unsupported_operation_declaration(owner)
