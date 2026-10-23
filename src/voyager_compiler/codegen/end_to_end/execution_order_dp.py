# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Exact bounded execution-order search for model-wide memory planning.

The caller owns target policy and supplies the authoritative outer executable
nodes.  This module only reasons about the finalized FX graph and one
provisional physical storage plan.  It never mutates either object.
"""

from __future__ import annotations

import heapq
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from torch.fx import GraphModule, Node

from voyager_compiler.codegen.execution_order import (
    ExecutionOrder,
    execution_positions,
    validate_execution_order,
)
from voyager_compiler.codegen.storage import (
    GRAPH_SCOPE_OUTER_FX,
    PhysicalTensorId,
    StoragePlan,
)

DEFAULT_DP_STATE_LIMIT = 250_000
_READ = 1
_WRITE = 2
_OPERATION_KINDS = frozenset({"call_function", "call_module", "call_method"})


class DPExecutionOrderError(RuntimeError):
    """The exact execution-order problem is invalid or cannot be solved."""


class DPExecutionOrderLimitError(DPExecutionOrderError):
    """The exact search exceeded its deterministic state budget."""

    def __init__(self, explored_states: int, state_limit: int):
        self.explored_states = explored_states
        self.state_limit = state_limit
        super().__init__(
            "Exact execution-order DP exceeded its state limit: "
            f"explored={explored_states}, limit={state_limit}"
        )


@dataclass(frozen=True)
class DPExecutionOrderResult:
    """A successful order plus diagnostics that do not pollute the value."""

    order: ExecutionOrder
    peak_live_bytes: int
    source_candidate_peak_live_bytes: int
    explored_states: int
    state_limit: int
    candidate_count: int


@dataclass(frozen=True)
class _LiveProfile:
    prefix: int
    live: int
    live_bytes: int


@dataclass(frozen=True)
class _Problem:
    graph_nodes: tuple[Node, ...]
    source_positions: dict[Node, int]
    candidates: tuple[Node, ...]
    candidate_index: dict[Node, int]
    candidate_predecessors: tuple[int, ...]
    unit_members: tuple[tuple[Node, ...], ...]
    preamble: tuple[Node, ...]
    structural: tuple[Node, ...]
    candidate_structural_predecessors: tuple[int, ...]
    outputs: tuple[Node, ...]
    constraint_edges: frozenset[tuple[Node, Node]]


def _plane_name(plane) -> str:
    return str(getattr(plane, "value", plane))


def _bit_indices(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _validate_candidates(
    graph_nodes: tuple[Node, ...],
    executable_nodes: tuple[Node, ...],
) -> None:
    if not isinstance(executable_nodes, tuple):
        raise TypeError("executable_nodes must be a tuple")
    if not all(isinstance(node, Node) for node in executable_nodes):
        raise TypeError("executable_nodes may contain only FX nodes")
    if len(set(executable_nodes)) != len(executable_nodes):
        raise DPExecutionOrderError("Executable candidates contain duplicates")
    graph_node_set = set(graph_nodes)
    foreign = set(executable_nodes) - graph_node_set
    if foreign:
        raise DPExecutionOrderError(
            "Executable candidates are not in the graph: "
            + ", ".join(sorted(node.name for node in foreign))
        )
    invalid = [
        node for node in executable_nodes if node.op not in _OPERATION_KINDS
    ]
    if invalid:
        raise DPExecutionOrderError(
            "Executable candidates must be outer operations: "
            + ", ".join(node.name for node in invalid)
        )


def _normalize_owned_nodes(
    graph_nodes: tuple[Node, ...],
    candidates: tuple[Node, ...],
    owned_structural_nodes: Mapping[Node, Sequence[Node]] | None,
    source_positions: dict[Node, int],
) -> tuple[dict[Node, Node], tuple[tuple[Node, ...], ...]]:
    candidate_set = set(candidates)
    graph_node_set = set(graph_nodes)
    owner_for = {candidate: candidate for candidate in candidates}
    declared_owner: dict[Node, Node] = {}
    owned_structural_nodes = (
        {} if owned_structural_nodes is None else owned_structural_nodes
    )
    unknown_owners = set(owned_structural_nodes) - candidate_set
    if unknown_owners:
        raise DPExecutionOrderError(
            "Owned structural nodes name non-candidate owners: "
            + ", ".join(sorted(node.name for node in unknown_owners))
        )

    for owner, declared_members in owned_structural_nodes.items():
        if not isinstance(declared_members, Sequence):
            raise TypeError("owned structural members must be a sequence")
        for member in declared_members:
            if not isinstance(member, Node):
                raise TypeError("owned structural members must be FX nodes")
            if member not in graph_node_set:
                raise DPExecutionOrderError(
                    f"{member.name}: owned structural node is not in graph"
                )
            if member in candidate_set:
                raise DPExecutionOrderError(
                    f"{member.name}: one executable cannot belong to "
                    "another executable unit"
                )
            previous = declared_owner.get(member)
            if previous is not None and previous is not owner:
                raise DPExecutionOrderError(
                    f"{member.name}: structural node has multiple owners"
                )
            if member.op in {"placeholder", "get_attr", "output"}:
                raise DPExecutionOrderError(
                    f"{member.name}: preamble/output nodes cannot be owned"
                )
            declared_owner[member] = owner
            owner_for[member] = owner

    unit_members = []
    for candidate in candidates:
        members = [candidate]
        members.extend(
            node for node, owner in declared_owner.items() if owner is candidate
        )
        members.sort(key=source_positions.__getitem__)
        unit_members.append(tuple(members))
    return owner_for, tuple(unit_members)


def _physical_modes(
    graph_nodes: tuple[Node, ...],
    storage_plan: StoragePlan,
    plane: str,
) -> dict[PhysicalTensorId, tuple[tuple[Node, int], ...]]:
    modes: dict[Node, dict[PhysicalTensorId, int]] = {
        node: {} for node in graph_nodes
    }

    def add(node: Node, physical: PhysicalTensorId, mode: int) -> None:
        modes[node][physical] = modes[node].get(physical, 0) | mode

    for node in graph_nodes:
        for logical in storage_plan.index.node_inputs.get(node, ()):
            add(node, storage_plan.physical_id(logical, plane), _READ)
        for physical in storage_plan.produced_physical_outputs(node, plane):
            add(node, physical, _WRITE)

    for physical, tensor in storage_plan.physical_tensors[plane].items():
        for access in tensor.accesses:
            add(
                access.operation,
                physical,
                _READ if access.kind == "read" else _WRITE,
            )

    by_physical: dict[PhysicalTensorId, list[tuple[Node, int]]] = defaultdict(
        list
    )
    for node, node_modes in modes.items():
        for physical, mode in node_modes.items():
            by_physical[physical].append((node, mode))
    return {physical: tuple(events) for physical, events in by_physical.items()}


def _validate_read_completeness(
    storage_plan: StoragePlan,
    plane: str,
    source_positions: dict[Node, int],
    by_physical: dict[PhysicalTensorId, tuple[tuple[Node, int], ...]],
) -> None:
    for physical in storage_plan.physical_tensors[plane]:
        tensor = storage_plan.physical_tensors[plane][physical]
        initially_available = physical.owner.node.op in {
            "placeholder",
            "get_attr",
        } or any(
            logical.node.op in {"placeholder", "get_attr"}
            for logical in tensor.logical_tensors
        )
        available = initially_available
        events = sorted(
            by_physical.get(physical, ()),
            key=lambda event: source_positions[event[0]],
        )
        for operation, mode in events:
            if mode & _READ and not available:
                raise DPExecutionOrderError(
                    f"{operation.name}: reads {physical.name} before an "
                    "initial preamble backing or physical write"
                )
            if mode & _WRITE:
                available = True


def _constraint_edges(
    graph_nodes: tuple[Node, ...],
    source_positions: dict[Node, int],
    unit_members: tuple[tuple[Node, ...], ...],
    by_physical: dict[PhysicalTensorId, tuple[tuple[Node, int], ...]],
) -> set[tuple[Node, Node]]:
    edges: set[tuple[Node, Node]] = set()

    def add(before: Node | None, after: Node | None) -> None:
        if before is None or after is None or before is after:
            return
        if source_positions[before] >= source_positions[after]:
            raise DPExecutionOrderError(
                "Execution constraint contradicts source topology: "
                f"{before.name} -> {after.name}"
            )
        edges.add((before, after))

    for node in graph_nodes:
        for predecessor in node.all_input_nodes:
            add(predecessor, node)

    for members in unit_members:
        for before, after in zip(members, members[1:]):
            add(before, after)

    operations = [node for node in graph_nodes if node.op in _OPERATION_KINDS]
    for impure in operations:
        if not impure.is_impure():
            continue
        impure_position = source_positions[impure]
        for operation in operations:
            operation_position = source_positions[operation]
            if operation_position < impure_position:
                add(operation, impure)
            elif operation_position > impure_position:
                add(impure, operation)

    for events in by_physical.values():
        ordered = sorted(
            events,
            key=lambda event: source_positions[event[0]],
        )
        last_write: Node | None = None
        readers: set[Node] = set()
        for operation, mode in ordered:
            if mode & _READ:
                add(last_write, operation)
            if mode & _WRITE:
                add(last_write, operation)
                for reader in readers:
                    add(reader, operation)
                readers.clear()
                last_write = operation
            elif mode & _READ:
                readers.add(operation)
    return edges


def _stable_entity_topology(
    entities: set[Node],
    predecessors: dict[Node, set[Node]],
    successors: dict[Node, set[Node]],
    rank: dict[Node, int],
) -> tuple[Node, ...]:
    indegree = {
        entity: len(predecessors.get(entity, ())) for entity in entities
    }
    ready = [
        (rank[entity], id(entity), entity)
        for entity, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    result = []
    while ready:
        _rank, _identity, entity = heapq.heappop(ready)
        result.append(entity)
        for successor in successors.get(entity, ()):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    ready,
                    (rank[successor], id(successor), successor),
                )
    if len(result) != len(entities):
        cyclic = sorted(
            (entity.name for entity, degree in indegree.items() if degree),
        )
        raise DPExecutionOrderError(
            "Execution constraints are cyclic after executable-unit "
            "contraction: " + ", ".join(cyclic)
        )
    return tuple(result)


def _build_problem(
    model: GraphModule,
    storage_plan: StoragePlan,
    plane: str,
    executable_nodes: tuple[Node, ...],
    owned_structural_nodes: Mapping[Node, Sequence[Node]] | None,
) -> _Problem:
    graph_nodes = tuple(model.graph.nodes)
    _validate_candidates(graph_nodes, executable_nodes)
    source_positions = {
        node: position for position, node in enumerate(graph_nodes)
    }
    candidates = tuple(
        sorted(executable_nodes, key=source_positions.__getitem__)
    )
    candidate_index = {
        candidate: index for index, candidate in enumerate(candidates)
    }
    owner_for, unit_members = _normalize_owned_nodes(
        graph_nodes,
        candidates,
        owned_structural_nodes,
        source_positions,
    )
    by_physical = _physical_modes(
        graph_nodes,
        storage_plan,
        plane,
    )
    _validate_read_completeness(
        storage_plan,
        plane,
        source_positions,
        by_physical,
    )
    edges = _constraint_edges(
        graph_nodes,
        source_positions,
        unit_members,
        by_physical,
    )

    entity_for = {node: owner_for.get(node, node) for node in graph_nodes}
    entities = set(entity_for.values())
    entity_members: dict[Node, tuple[Node, ...]] = {
        entity: (entity,) for entity in entities
    }
    for candidate, members in zip(candidates, unit_members):
        entity_members[candidate] = members
    entity_rank = {
        entity: min(source_positions[node] for node in entity_members[entity])
        for entity in entities
    }
    entity_predecessors: dict[Node, set[Node]] = defaultdict(set)
    entity_successors: dict[Node, set[Node]] = defaultdict(set)
    for before, after in edges:
        before_entity = entity_for[before]
        after_entity = entity_for[after]
        if before_entity is after_entity:
            continue
        entity_successors[before_entity].add(after_entity)
        entity_predecessors[after_entity].add(before_entity)

    topology = _stable_entity_topology(
        entities,
        entity_predecessors,
        entity_successors,
        entity_rank,
    )
    candidate_set = set(candidates)
    preamble = tuple(
        node for node in graph_nodes if node.op in {"placeholder", "get_attr"}
    )
    preamble_set = set(preamble)
    outputs = tuple(node for node in graph_nodes if node.op == "output")
    output_set = set(outputs)
    owned_nodes = set(owner_for) - candidate_set

    for node in preamble:
        entity = entity_for[node]
        incoming = entity_predecessors.get(entity, ())
        if incoming:
            raise DPExecutionOrderError(
                f"{node.name}: fixed preamble node has execution "
                "predecessors"
            )
    for node in outputs:
        entity = entity_for[node]
        outgoing = entity_successors.get(entity, ())
        if outgoing:
            raise DPExecutionOrderError(
                f"{node.name}: graph output is not terminal"
            )

    ancestor_candidates: dict[Node, int] = {}
    for entity in topology:
        mask = 0
        for predecessor in entity_predecessors.get(entity, ()):
            if predecessor in candidate_index:
                mask |= 1 << candidate_index[predecessor]
            else:
                mask |= ancestor_candidates[predecessor]
        ancestor_candidates[entity] = mask

    candidate_predecessors = tuple(
        ancestor_candidates[candidate] for candidate in candidates
    )
    structural_entities = tuple(
        entity
        for entity in topology
        if (
            entity not in candidate_set
            and entity not in preamble_set
            and entity not in output_set
            and entity not in owned_nodes
        )
    )
    if any(len(entity_members[entity]) != 1 for entity in structural_entities):
        raise DPExecutionOrderError(
            "Only executable units may contain multiple FX nodes"
        )
    structural = tuple(
        entity_members[entity][0] for entity in structural_entities
    )
    structural_index = {
        entity: index for index, entity in enumerate(structural_entities)
    }
    ancestor_structural: dict[Node, int] = {}
    for entity in topology:
        mask = 0
        for predecessor in entity_predecessors.get(entity, ()):
            mask |= ancestor_structural[predecessor]
            if predecessor in structural_index:
                mask |= 1 << structural_index[predecessor]
        ancestor_structural[entity] = mask
    candidate_structural_predecessors = tuple(
        ancestor_structural[candidate] for candidate in candidates
    )

    problem = _Problem(
        graph_nodes=graph_nodes,
        source_positions=source_positions,
        candidates=candidates,
        candidate_index=candidate_index,
        candidate_predecessors=candidate_predecessors,
        unit_members=unit_members,
        preamble=preamble,
        structural=structural,
        candidate_structural_predecessors=(candidate_structural_predecessors),
        outputs=outputs,
        constraint_edges=frozenset(edges),
    )
    return problem


def solve_execution_order_dp(
    model: GraphModule,
    storage_plan: StoragePlan,
    executable_nodes: tuple[Node, ...],
    *,
    storage_plane,
    resident_physical_ids: tuple[PhysicalTensorId, ...],
    owned_structural_nodes: Mapping[Node, Sequence[Node]] | None = None,
    state_limit: int = DEFAULT_DP_STATE_LIMIT,
) -> DPExecutionOrderResult:
    """Return the exact minimum-peak legal outer-operation order.

    ``executable_nodes`` is an authoritative target-provided membership set.
    ``owned_structural_nodes`` optionally makes fused prelude/epilogue nodes
    atomic with an executable while preserving their source FX order.
    """
    if not isinstance(model, GraphModule):
        raise TypeError("model must be a GraphModule")
    if not isinstance(storage_plan, StoragePlan):
        raise TypeError("storage_plan must be a StoragePlan")
    if storage_plan.index.model is not model:
        raise DPExecutionOrderError(
            "Storage plan was built for a different FX graph"
        )
    if storage_plan.index.scope != GRAPH_SCOPE_OUTER_FX:
        raise DPExecutionOrderError(
            "Exact execution-order DP requires an outer-FX storage plan; "
            f"got scope {storage_plan.index.scope!r}"
        )
    if isinstance(state_limit, bool) or not isinstance(state_limit, int):
        raise TypeError("state_limit must be an integer")
    if state_limit <= 0:
        raise ValueError("state_limit must be positive")
    plane = _plane_name(storage_plane)
    if plane not in storage_plan.physical_tensors:
        raise DPExecutionOrderError(
            f"Storage plan has no {plane!r} physical plane"
        )

    if not isinstance(resident_physical_ids, tuple):
        raise TypeError("resident_physical_ids must be a tuple")
    if not all(
        isinstance(physical, PhysicalTensorId)
        for physical in resident_physical_ids
    ):
        raise TypeError(
            "resident_physical_ids may contain only physical tensor IDs"
        )
    if len(set(resident_physical_ids)) != len(resident_physical_ids):
        raise DPExecutionOrderError(
            "Resident physical tensor IDs contain duplicates"
        )
    declared_physicals = storage_plan.physical_tensors[plane]
    invalid_residents = [
        physical
        for physical in resident_physical_ids
        if (physical.plane != plane or physical not in declared_physicals)
    ]
    if invalid_residents:
        raise DPExecutionOrderError(
            "Resident physical tensor IDs are not declared in the selected "
            "plane: "
            + ", ".join(sorted(physical.name for physical in invalid_residents))
        )
    hidden_physicals = {
        physical
        for physical, tensor in declared_physicals.items()
        if tensor.accesses
    }
    missing_hidden = hidden_physicals - set(resident_physical_ids)
    if missing_hidden:
        raise DPExecutionOrderError(
            "Resident physical tensor IDs omit hidden-access storage: "
            + ", ".join(sorted(physical.name for physical in missing_hidden))
        )

    problem = _build_problem(
        model,
        storage_plan,
        plane,
        executable_nodes,
        owned_structural_nodes,
    )
    candidate_count = len(problem.candidates)
    if candidate_count + 64 >= sys.getrecursionlimit():
        raise DPExecutionOrderError(
            "Exact execution-order DP exceeds the safe recursion depth: "
            f"candidates={candidate_count}, "
            f"limit={sys.getrecursionlimit()}"
        )
    full_state = (1 << candidate_count) - 1

    # Residency is target policy supplied by the caller.  The solver refuses
    # to infer it from generic logical/component declarations because those
    # also describe compiler-only and nonresident serialized values.
    physicals = tuple(
        sorted(
            resident_physical_ids,
            key=lambda physical: physical.name,
        )
    )
    physical_index = {
        physical: index for index, physical in enumerate(physicals)
    }
    sizes = []
    birth_masks = []
    lifetime_masks = []
    persistent_mask = 0
    node_index = {node: index for index, node in enumerate(problem.graph_nodes)}
    graph_node_set = set(problem.graph_nodes)
    births_by_node: dict[Node, list[int]] = defaultdict(list)
    lifetime_by_node: dict[Node, list[int]] = defaultdict(list)

    for physical in physicals:
        index = physical_index[physical]
        try:
            size = storage_plan.backing_size_bytes(physical, plane)
        except (KeyError, TypeError, ValueError) as error:
            raise DPExecutionOrderError(
                f"{physical.name}: exact physical backing size is required"
            ) from error
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DPExecutionOrderError(
                f"{physical.name}: invalid physical backing size {size!r}"
            )
        sizes.append(size)
        tensor = storage_plan.physical_tensors[plane][physical]
        birth_nodes = {physical.owner.node}
        lifetime_nodes = {physical.owner.node}
        persistent = False
        for logical in tensor.logical_tensors:
            birth_nodes.add(logical.node)
            lifetime_nodes.add(logical.node)
            persistent |= logical.node.op == "get_attr"
            lifetime_nodes.update(storage_plan.logical_tensors[logical].users)
        for access in tensor.accesses:
            birth_nodes.add(access.operation)
            lifetime_nodes.add(access.operation)
        foreign = (birth_nodes | lifetime_nodes) - graph_node_set
        if foreign:
            raise DPExecutionOrderError(
                f"{physical.name}: physical lifetime references foreign "
                "FX nodes"
            )
        birth_mask = 0
        lifetime_mask = 0
        for node in birth_nodes:
            birth_mask |= 1 << node_index[node]
            births_by_node[node].append(index)
        for node in lifetime_nodes:
            lifetime_mask |= 1 << node_index[node]
            lifetime_by_node[node].append(index)
        birth_masks.append(birth_mask)
        lifetime_masks.append(lifetime_mask)
        if persistent:
            persistent_mask |= 1 << index

    preamble_mask = 0
    for node in problem.preamble:
        preamble_mask |= 1 << node_index[node]
    unit_node_masks = []
    for members in problem.unit_members:
        mask = 0
        for node in members:
            mask |= 1 << node_index[node]
        unit_node_masks.append(mask)

    @lru_cache(maxsize=None)
    def required_structural(
        state: int,
    ) -> tuple[tuple[Node, ...], int]:
        """Return structural ancestors demanded by completed candidates."""
        structural_mask = 0
        for candidate_index_value in _bit_indices(state):
            structural_mask |= problem.candidate_structural_predecessors[
                candidate_index_value
            ]
        nodes = tuple(
            node
            for index, node in enumerate(problem.structural)
            if structural_mask & (1 << index)
        )
        graph_mask = 0
        for node in nodes:
            graph_mask |= 1 << node_index[node]
        return nodes, graph_mask

    @lru_cache(maxsize=None)
    def prefix_mask(state: int) -> int:
        mask = preamble_mask | required_structural(state)[1]
        for candidate_index in _bit_indices(state):
            mask |= unit_node_masks[candidate_index]
        return mask

    @lru_cache(maxsize=None)
    def profile(state: int) -> _LiveProfile:
        prefix = prefix_mask(state)
        live = 0
        live_bytes = 0
        for index, (birth, lifetime, size) in enumerate(
            zip(birth_masks, lifetime_masks, sizes)
        ):
            if not birth & prefix:
                continue
            if persistent_mask & (1 << index) or lifetime & ~prefix:
                live |= 1 << index
                live_bytes += size
        return _LiveProfile(prefix, live, live_bytes)

    def simulate_block(
        start: _LiveProfile,
        nodes: Sequence[Node],
    ) -> tuple[int, _LiveProfile]:
        done = start.prefix
        live = start.live
        live_bytes = start.live_bytes
        peak = live_bytes
        for node in nodes:
            node_bit = 1 << node_index[node]
            if done & node_bit:
                raise DPExecutionOrderError(
                    f"{node.name}: execution-order placement emitted a node "
                    "twice"
                )
            for physical_index_value in births_by_node.get(node, ()):
                physical_bit = 1 << physical_index_value
                if live & physical_bit:
                    continue
                live |= physical_bit
                live_bytes += sizes[physical_index_value]
            peak = max(peak, live_bytes)
            done |= node_bit
            for physical_index_value in lifetime_by_node.get(node, ()):
                physical_bit = 1 << physical_index_value
                if not live & physical_bit:
                    continue
                if persistent_mask & physical_bit:
                    continue
                if lifetime_masks[physical_index_value] & ~done:
                    continue
                live &= ~physical_bit
                live_bytes -= sizes[physical_index_value]
        return peak, _LiveProfile(done, live, live_bytes)

    initial_nodes = problem.preamble
    initial_peak, initial_profile = simulate_block(
        _LiveProfile(0, 0, 0),
        initial_nodes,
    )
    if initial_profile != profile(0):
        raise DPExecutionOrderError(
            "Initial fixed preamble does not match physical liveness"
        )

    @lru_cache(maxsize=None)
    def transition(state: int, candidate_index_value: int) -> tuple[int, int]:
        next_state = state | (1 << candidate_index_value)
        old_structural_mask = required_structural(state)[1]
        new_structural = tuple(
            node
            for node in required_structural(next_state)[0]
            if not old_structural_mask & (1 << node_index[node])
        )
        block = new_structural + problem.unit_members[candidate_index_value]
        peak, actual = simulate_block(profile(state), block)
        expected = profile(next_state)
        if actual != expected:
            raise DPExecutionOrderError(
                "Executable transition does not match deterministic "
                f"structural placement at state {state:#x}"
            )
        return peak, next_state

    def ready_candidates(state: int) -> tuple[int, ...]:
        return tuple(
            index
            for index, predecessors in enumerate(problem.candidate_predecessors)
            if (not state & (1 << index) and not predecessors & ~state)
        )

    final_structural_mask = required_structural(full_state)[1]
    terminal_nodes = (
        tuple(
            node
            for node in problem.structural
            if not final_structural_mask & (1 << node_index[node])
        )
        + problem.outputs
    )
    terminal_peak, terminal_profile = simulate_block(
        profile(full_state),
        terminal_nodes,
    )
    expected_terminal_mask = (1 << len(problem.graph_nodes)) - 1
    if terminal_profile.prefix != expected_terminal_mask:
        missing = [
            node.name
            for node in problem.graph_nodes
            if not terminal_profile.prefix & (1 << node_index[node])
        ]
        raise DPExecutionOrderError(
            "Structural placement did not cover the complete FX graph: "
            + ", ".join(missing)
        )

    def materialize_candidate_order(
        selected_indices: Sequence[int],
    ) -> tuple[Node, ...]:
        ordered_nodes = list(problem.preamble)
        state = 0
        for candidate_index_value in selected_indices:
            old_structural_mask = required_structural(state)[1]
            state |= 1 << candidate_index_value
            ordered_nodes.extend(
                node
                for node in required_structural(state)[0]
                if not old_structural_mask & (1 << node_index[node])
            )
            ordered_nodes.extend(problem.unit_members[candidate_index_value])
        ordered_nodes.extend(terminal_nodes)
        return tuple(ordered_nodes)

    # Keep source order as the legal incumbent and equal-cost winner. If an
    # explicitly declared atomic unit is interleaved in raw FX, materialize
    # the source candidate sequence through the same unit-preserving policy.
    preamble_set = set(problem.preamble)
    normalized_source_nodes = problem.preamble + tuple(
        node for node in problem.graph_nodes if node not in preamble_set
    )
    normalized_source_positions = {
        node: position for position, node in enumerate(normalized_source_nodes)
    }
    source_units_are_contiguous = all(
        tuple(normalized_source_positions[node] for node in members)
        == tuple(
            range(
                min(normalized_source_positions[node] for node in members),
                min(normalized_source_positions[node] for node in members)
                + len(members),
            )
        )
        for members in problem.unit_members
    )
    source_incumbent_indices = list(range(candidate_count))
    if not source_units_are_contiguous:
        source_incumbent_indices = []
        source_state = 0
        while source_state != full_state:
            ready = ready_candidates(source_state)
            if not ready:
                raise DPExecutionOrderError(
                    "Atomic source candidate topology has no ready "
                    "operation before completion"
                )
            selected = ready[0]
            source_incumbent_indices.append(selected)
            source_state |= 1 << selected
    source_incumbent_nodes = (
        normalized_source_nodes
        if source_units_are_contiguous
        else materialize_candidate_order(source_incumbent_indices)
    )
    source_order = ExecutionOrder(source_incumbent_nodes)
    validate_execution_order(model, source_order)
    source_positions = execution_positions(source_order)
    for before, after in problem.constraint_edges:
        if source_positions[before] >= source_positions[after]:
            raise DPExecutionOrderError(
                "Normalized source order violates execution constraint: "
                f"{before.name} -> {after.name}"
            )
    source_candidate_peak, source_profile = simulate_block(
        _LiveProfile(0, 0, 0),
        source_incumbent_nodes,
    )
    if source_profile.prefix != expected_terminal_mask:
        raise DPExecutionOrderError("Normalized source order is incomplete")

    memo: dict[int, int] = {}
    explored_states = 0

    def suffix_peak(state: int) -> int:
        nonlocal explored_states
        known = memo.get(state)
        if known is not None:
            return known
        if explored_states >= state_limit:
            raise DPExecutionOrderLimitError(
                explored_states,
                state_limit,
            )
        explored_states += 1
        if state == full_state:
            memo[state] = terminal_peak
            return terminal_peak
        ready = ready_candidates(state)
        if not ready:
            raise DPExecutionOrderError(
                "Execution DAG has no ready operation before completion"
            )
        best = None
        for candidate_index_value in ready:
            local_peak, next_state = transition(
                state,
                candidate_index_value,
            )
            value = max(local_peak, suffix_peak(next_state))
            if best is None or value < best:
                best = value
        assert best is not None
        memo[state] = best
        return best

    try:
        dp_optimum = max(initial_peak, suffix_peak(0))
    except RecursionError as error:
        raise DPExecutionOrderError(
            "Exact execution-order DP exhausted recursion depth"
        ) from error
    use_source_incumbent = source_candidate_peak <= dp_optimum
    optimum = min(source_candidate_peak, dp_optimum)

    if use_source_incumbent:
        selected_indices = list(source_incumbent_indices)
        ordered_nodes = list(source_incumbent_nodes)
    else:
        selected_indices = []
        state = 0
        while state != full_state:
            selected = None
            for candidate_index_value in ready_candidates(state):
                local_peak, next_state = transition(
                    state,
                    candidate_index_value,
                )
                # Candidate indices are FX ordered, so the first optimum
                # transition is the stable source-order tie.
                if max(local_peak, memo[next_state]) <= optimum:
                    selected = candidate_index_value
                    break
            if selected is None:
                raise DPExecutionOrderError(
                    "Could not reconstruct the exact optimum"
                )
            selected_indices.append(selected)
            state |= 1 << selected

        ordered_nodes = list(materialize_candidate_order(selected_indices))
    order = ExecutionOrder(tuple(ordered_nodes))
    validate_execution_order(model, order)
    positions = execution_positions(order)
    for before, after in problem.constraint_edges:
        if positions[before] >= positions[after]:
            raise DPExecutionOrderError(
                "Selected order violates execution constraint: "
                f"{before.name} -> {after.name}"
            )
    selected_candidates = tuple(
        node for node in order.nodes if node in problem.candidate_index
    )
    expected_candidates = tuple(
        problem.candidates[index] for index in selected_indices
    )
    if selected_candidates != expected_candidates:
        raise DPExecutionOrderError(
            "Selected full order changed executable membership"
        )
    measured_peak, measured_profile = simulate_block(
        _LiveProfile(0, 0, 0),
        order.nodes,
    )
    if measured_profile.prefix != expected_terminal_mask:
        raise DPExecutionOrderError("Selected full order is incomplete")
    if measured_peak != optimum:
        raise DPExecutionOrderError(
            "Reconstructed order does not attain the exact DP objective: "
            f"measured={measured_peak}, optimum={optimum}"
        )
    return DPExecutionOrderResult(
        order=order,
        peak_live_bytes=optimum,
        source_candidate_peak_live_bytes=source_candidate_peak,
        explored_states=explored_states,
        state_limit=state_limit,
        candidate_count=candidate_count,
    )


__all__ = [
    "DEFAULT_DP_STATE_LIMIT",
    "DPExecutionOrderError",
    "DPExecutionOrderLimitError",
    "DPExecutionOrderResult",
    "solve_execution_order_dp",
]
