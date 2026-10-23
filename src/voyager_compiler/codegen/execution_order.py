# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Two compile-wide choices about *when* and *where* tensors live.

They are stated together because they are not independent, and separating them
is how a compile ends up asking for something no allocator can deliver.

**Scratchpad mapping** -- what the on-chip memory is asked to hold.

``per_operation``
    Each operation stages its own operands in and its result out. The
    allocator only has to fit one operation's working set, and a buffer's
    address is reused as soon as its last reader is done. This is what the
    bufferized path does today, under a different name: a lifetime-based
    greedy best-fit over the whole graph.
``end_to_end``
    Every tensor the model touches is resident for its whole lifetime, and
    nothing is staged per call. That turns the fit from a per-operation
    question into a whole-model one, and it is what makes a concat slot or a
    channel shuffle an address operation rather than a copy.

**Execution order** -- the order the operations run in.

``fx``
    Graph order, as exported. Every schedule is a valid one, so nothing has to
    be solved.
``dp``
    The order that minimises the peak resident footprint, solved over the
    storage plan. It is only a question worth asking under ``end_to_end``:
    when operands are staged per call, the peak is set by the largest single
    operation and reordering cannot move it.

That last sentence is the rule below. It is stated once, here, rather than at
each of the three places that need it -- the command line, the compile entry
point, and memory mapping -- because a rule restated three times is a rule
that can be enforced two ways.
"""

from dataclasses import dataclass

from torch.fx import GraphModule, Node

PER_OPERATION_SCRATCHPAD_MAPPING = "per_operation"
END_TO_END_SCRATCHPAD_MAPPING = "end_to_end"
SCRATCHPAD_MAPPING_MODES = (
    PER_OPERATION_SCRATCHPAD_MAPPING,
    END_TO_END_SCRATCHPAD_MAPPING,
)
DEFAULT_SCRATCHPAD_MAPPING = PER_OPERATION_SCRATCHPAD_MAPPING

FX_EXECUTION_ORDER_MODE = "fx"
DP_EXECUTION_ORDER_MODE = "dp"
EXECUTION_ORDER_MODES = (FX_EXECUTION_ORDER_MODE, DP_EXECUTION_ORDER_MODE)
DEFAULT_EXECUTION_ORDER_MODE = FX_EXECUTION_ORDER_MODE

__all__ = [
    "DEFAULT_EXECUTION_MODES",
    "DEFAULT_EXECUTION_ORDER_MODE",
    "DEFAULT_SCRATCHPAD_MAPPING",
    "DP_EXECUTION_ORDER_MODE",
    "END_TO_END_SCRATCHPAD_MAPPING",
    "EXECUTION_ORDER_MODES",
    "ExecutionOrder",
    "FX_EXECUTION_ORDER_MODE",
    "PER_OPERATION_SCRATCHPAD_MAPPING",
    "SCRATCHPAD_MAPPING_MODES",
    "require_execution_mode",
    "execution_order_for_model",
    "execution_positions",
    "fx_execution_order",
    "validate_execution_order",
    "validate_execution_order_mode",
    "validate_scratchpad_mapping",
]


@dataclass(frozen=True)
class ExecutionOrder:
    """One immutable topological permutation of a final FX graph."""

    nodes: tuple[Node, ...]

    def __post_init__(self):
        if not isinstance(self.nodes, tuple):
            raise TypeError("ExecutionOrder.nodes must be a tuple")
        if not all(isinstance(node, Node) for node in self.nodes):
            raise TypeError("ExecutionOrder may contain only FX nodes")


def fx_execution_order(model: GraphModule) -> ExecutionOrder:
    """Capture the graph's current linked-list order."""
    return ExecutionOrder(tuple(model.graph.nodes))


def execution_positions(order: ExecutionOrder) -> dict[Node, int]:
    """Derive node positions for one execution-order consumer."""
    if not isinstance(order, ExecutionOrder):
        raise TypeError("expected an ExecutionOrder")
    return {node: position for position, node in enumerate(order.nodes)}


def validate_execution_order(
    model: GraphModule,
    order: ExecutionOrder,
) -> None:
    """Require an exact topological permutation of the current FX graph."""
    if not isinstance(order, ExecutionOrder):
        raise TypeError("expected an ExecutionOrder")
    if len(set(order.nodes)) != len(order.nodes):
        raise ValueError("ExecutionOrder contains duplicate nodes")

    graph_nodes = tuple(model.graph.nodes)
    graph_node_set = set(graph_nodes)
    ordered_node_set = set(order.nodes)
    foreign = ordered_node_set - graph_node_set
    missing = graph_node_set - ordered_node_set
    if foreign or missing:
        details = []
        if missing:
            details.append(
                "missing " + ", ".join(sorted(node.name for node in missing))
            )
        if foreign:
            details.append(
                "foreign " + ", ".join(sorted(node.name for node in foreign))
            )
        raise ValueError(
            "ExecutionOrder is not an exact graph permutation: "
            + "; ".join(details)
        )

    positions = execution_positions(order)
    for node in order.nodes:
        for predecessor in node.all_input_nodes:
            if positions[predecessor] >= positions[node]:
                raise ValueError(
                    f"ExecutionOrder places {node.name} before dependency "
                    f"{predecessor.name}"
                )


def execution_order_for_model(model: GraphModule) -> ExecutionOrder:
    """Return the plan's selected order, or current FX order before planning."""
    plan = getattr(model, "meta", {}).get("storage_plan")
    if plan is None:
        return fx_execution_order(model)
    order = getattr(plan, "execution_order", None)
    if order is None:
        raise ValueError("storage plan has no execution order")
    validate_execution_order(model, order)
    return order


def validate_scratchpad_mapping(scratchpad_mapping):
    """Normalize one scratchpad mapping mode, or reject it."""
    if scratchpad_mapping is None:
        return DEFAULT_SCRATCHPAD_MAPPING
    if scratchpad_mapping not in SCRATCHPAD_MAPPING_MODES:
        raise ValueError(
            f"unknown scratchpad mapping {scratchpad_mapping!r}; "
            f"accepted: {', '.join(SCRATCHPAD_MAPPING_MODES)}"
        )
    return scratchpad_mapping


def validate_execution_order_mode(execution_order, scratchpad_mapping):
    """Normalize the pair, and reject a combination nothing can honour.

    A solved order needs a whole-model residency to solve *for*. Asked for
    without one, it would run the solver against a plan whose peak it cannot
    influence and report an objective that means nothing -- which is worse
    than refusing, because it looks like it worked.
    """
    mapping = validate_scratchpad_mapping(scratchpad_mapping)
    if execution_order is None:
        return DEFAULT_EXECUTION_ORDER_MODE, mapping
    if execution_order not in EXECUTION_ORDER_MODES:
        raise ValueError(
            f"unknown execution order {execution_order!r}; "
            f"accepted: {', '.join(EXECUTION_ORDER_MODES)}"
        )
    if (
        execution_order == DP_EXECUTION_ORDER_MODE
        and mapping != END_TO_END_SCRATCHPAD_MAPPING
    ):
        raise ValueError(
            f"--execution_order {DP_EXECUTION_ORDER_MODE} requires "
            f"--scratchpad_mapping {END_TO_END_SCRATCHPAD_MAPPING}: a solved "
            f"order minimises a whole-model resident footprint, and "
            f"{mapping} has none to minimise"
        )
    return execution_order, mapping


#: What a codegen path honours when it declares nothing: operands staged per
#: call, in graph order. It is what the bufferized path does, so every target
#: that fits that path is described by it without saying anything.
DEFAULT_EXECUTION_MODES = frozenset(
    {(PER_OPERATION_SCRATCHPAD_MAPPING, FX_EXECUTION_ORDER_MODE)}
)


def require_execution_mode(execution_order, scratchpad_mapping, supported):
    """Refuse a mode pair the codegen path that will run does not honour.

    The pair is validated for internal consistency first -- that is a fact
    about the two flags and holds whoever runs them. This is the second
    question, and a different one: *this* path, for *this* target, either
    implements the pair or does not.

    It is a declaration rather than a branch on a path name because the two
    answers move independently. A target's terminal codegen grows support for
    a mode without the bufferized path acquiring it, and stating that as a set
    means the compiler refuses on what a path has *not implemented* rather
    than on which path it happens to be.

    Args:
        execution_order: The validated execution-order mode.
        scratchpad_mapping: The validated scratchpad mapping.
        supported: The ``(mapping, order)`` pairs the path honours.

    Raises:
        NotImplementedError: If the pair is not among them. Naming the
            declared set rather than one flag: it is the *combination* that
            is or is not implemented, and reporting half of it sends the
            reader to change the wrong flag.
    """
    if (scratchpad_mapping, execution_order) in supported:
        return
    declared = ", ".join(
        f"{mapping}/{order}" for mapping, order in sorted(supported)
    )
    raise NotImplementedError(
        f"--scratchpad_mapping {scratchpad_mapping} --execution_order "
        f"{execution_order} is not implemented by the codegen path this "
        f"compile takes; it declares {declared or '(nothing)'}"
    )
