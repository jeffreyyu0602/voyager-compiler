# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Which unit runs a call, and what happens when none does.

``AcceleratorConfig.compute_units`` declares what each execution resource
owns.  Nothing read that declaration, so a target had exactly one state --
whatever it bufferized, it claimed -- and an operation none of its units can
issue was asserted as accelerator work anyway.  A machine with one matrix
unit ended up owning standalone elementwise chains its datapath has no way
to start.

This pass reads the declaration.  A call whose operation class no unit
accepts is declared a non-accelerator boundary: the host runs it, the
accelerator earns no coverage for it, and ``emit`` serializes it as
``unsupported`` rather than as a call the array will make.

A target that declares no units declares nothing, so every call stays
accelerator work -- the untargeted behaviour, unchanged.

The pass only ever *refuses*.  It never moves a call onto a unit, because
which unit runs an accepted call is a scheduling question and this runs long
before scheduling.  Refusal is the part that has to happen early: it decides
whether a nest is built at all.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import torch

from torch.fx import GraphModule, Node

from voyager_compiler.codegen.aten_classifier import is_elementwise_op
from voyager_compiler.codegen.node_info import (
    get_anchor_node,
    is_conv2d,
    is_depthwise_conv,
    is_gemm_op,
    is_pooling,
)
from voyager_compiler.codegen.unsupported import (
    declare_unsupported_operation,
    is_declared_unsupported_operation,
)

logger = logging.getLogger(__name__)

#: Why a call was refused, as it appears in the declaration and in reports.
NO_ACCEPTING_UNIT = "no_unit_accepts_operation"

__all__ = [
    "NO_ACCEPTING_UNIT",
    "assign_compute_units",
    "issuable_group",
    "operation_class",
]


def operation_class(anchor: Optional[Node]) -> Optional[str]:
    """The class ``anchor`` belongs to, in ``accepted_operations`` vocabulary.

    ``None`` for anything this pass cannot classify.  An unclassified call is
    never refused: a name we do not recognize is ignorance, not a hardware
    limit, and refusing on it would cost dispatch for no stated reason.
    """
    if anchor is None:
        return None
    if is_conv2d(anchor):
        return "depthwise_conv2d" if is_depthwise_conv(anchor) else "gemm"
    if is_gemm_op(anchor):
        return "gemm"
    if is_pooling(anchor):
        return "pool2d"
    if is_elementwise_op(anchor):
        return "elementwise"
    return None


def _accepts(units: Sequence, op_class: str) -> bool:
    """Whether any declared unit owns ``op_class``.

    A unit whose ``accepted_operations`` is empty claims whatever no other
    unit named, so it accepts everything -- that is the documented meaning of
    the empty set, and it is what keeps a partially-described machine from
    refusing work it can do.
    """
    for unit in units:
        if not unit.accepted_operations:
            return True
        if op_class in unit.accepted_operations:
            return True
    return False


def _group_anchor(nodes: Sequence[Node]) -> Optional[Node]:
    """The compute anchor of a candidate fusion group, before it is a call.

    Mirrors ``get_anchor_node`` -- which can only answer once the group is
    already a fused ``call_module`` -- so the same operation classifies the
    same way whether it is asked before or after fusion.
    """
    for node in nodes:
        if is_gemm_op(node):
            return node
    anchor = None
    for node in nodes:
        if node.op == "call_function" and (
            anchor is None
            or anchor.target is torch.ops.quantized_ops.dequantize.default
        ):
            anchor = node
    return anchor


def issuable_group(config):
    """Whether a set of operations may be fused into one accelerator call.

    Fusion states that the datapath runs the group as a single instruction.
    That is only true if the machine can issue such a call at all, so a group
    whose class no declared unit accepts must not be formed: the operations
    stay separate, which is what they are on the host that will run them.

    Returns ``None`` when the machine declares no units -- no declaration, no
    narrowing, and ``fuse_operator`` keeps its untargeted behaviour.
    """
    units = tuple(getattr(config, "compute_units", None) or ())
    if not units:
        return None

    def accepts_group(nodes: List[Node]) -> bool:
        op_class = operation_class(_group_anchor(nodes))
        return op_class is None or _accepts(units, op_class)

    return accepts_group


def assign_compute_units(model: GraphModule, config) -> Tuple[Node, ...]:
    """Declare every call no declared unit accepts a non-accelerator boundary.

    Returns the nodes declared, so a caller can report coverage rather than
    having to re-derive it.
    """
    units = tuple(getattr(config, "compute_units", None) or ())
    if not units:
        return ()

    owned = sorted({op for unit in units for op in unit.accepted_operations})
    declared = []
    for node in list(model.graph.nodes):
        if node.op not in ("call_module", "call_function"):
            continue
        if is_declared_unsupported_operation(node):
            continue  # an earlier pass already answered for this call

        op_class = operation_class(get_anchor_node(node))
        if op_class is None or _accepts(units, op_class):
            continue

        detail = (
            f"no compute unit accepts a {op_class!r} call; "
            f"the declared units own {owned or ['(unrestricted)']}"
        )
        facts = {
            "operation_class": op_class,
            "accepted_operations": owned,
            "units": [unit.name for unit in units],
        }
        declare_unsupported_operation(
            node,
            reason_code=NO_ACCEPTING_UNIT,
            detail=detail,
            stage="unit-assignment",
            facts=facts,
        )
        declared.append(node)

    if declared:
        logger.info(
            "[unit-assignment] %d call(s) have no accepting unit: %s",
            len(declared),
            ", ".join(n.name for n in declared[:8])
            + (" ..." if len(declared) > 8 else ""),
        )
    return tuple(declared)
