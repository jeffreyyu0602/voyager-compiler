# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Resolving relations into a plan, and checking that it is consistent.

:mod:`.effects` says what relation each operation has. This module does the
two things that turn a pile of relations into something an allocator can act
on:

**Resolve.** Follow each logical tensor's relation chain until it reaches a
value that materializes; that value's identity, in this plane, is the backing.
Chains are followed rather than flattened as they are built, because a target
declares its own relations *after* the generic ones and a chain's end can move.

**Validate.** Then check the things a chain cannot check locally. Every
logical tensor in a plane has exactly one backing; no backing spans two
planes; every component a relation names is declared; every transfer's
endpoints exist. Each of those is a contradiction that is cheap to find here
and expensive to find as a wrong address later.

The plan holds no addresses, and that is the point: it can be checked before
an allocator has run, which is when a contradiction is still a compiler bug
rather than a backend mismatch.
"""

import logging
import math
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import dtype_byte_size
from voyager_compiler.codegen.storage.effects import generic_storage_relation
from voyager_compiler.codegen.storage.types import (
    LogicalTensor,
    PhysicalAccess,
    PhysicalPayloadRef,
    PhysicalTensor,
    PhysicalTensorComponent,
    PhysicalTensorId,
    SerializedTensorArtifact,
    SerializedTensorRef,
    StorageInterval,
    StoragePlane,
    StorageRelation,
    StorageRelationKind,
    StorageTransfer,
    ValueRef,
    recorded_value,
)

logger = logging.getLogger(__name__)

GRAPH_SCOPE_RECURSIVE = "recursive"
GRAPH_SCOPE_OUTER_FX = "outer_fx"
GRAPH_SCOPES = (GRAPH_SCOPE_RECURSIVE, GRAPH_SCOPE_OUTER_FX)

__all__ = [
    "GRAPH_SCOPE_OUTER_FX",
    "GRAPH_SCOPE_RECURSIVE",
    "GRAPH_SCOPES",
    "GraphIndex",
    "StorageGraph",
    "StoragePlan",
]

_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond
_COMMIT = torch.ops.higher_order.commit


@dataclass(frozen=True)
class _PhysicalComponentDeclaration:
    """One target representation awaiting relation-chain resolution."""

    logical_tensor: ValueRef
    component: PhysicalTensorComponent
    backing_size_bytes: int
    layout: str


@dataclass(frozen=True)
class _SerializedArtifactDeclaration:
    """A nonresident target encoding awaiting physical-ID resolution."""

    logical_tensor: ValueRef
    name: str
    shape: tuple[int, ...]
    dtype: object
    bits: int
    byte_size: int
    value: torch.Tensor


def _subgraph(graph_module, target):
    """The ``GraphModule`` a region's function argument names."""
    module = getattr(graph_module, str(target), None)
    return module if hasattr(module, "graph") else None


def _region_bindings(graph_module, node):
    """``(body, operands)`` for each subgraph a node hands its operands to.

    Four shapes, and they differ only in where the operand list sits:
    ``while_loop(cond, body, carried, additional)`` binds both bodies over
    ``carried + additional``; ``cond(pred, true, false, operands)`` binds each
    branch over one shared list; ``commit(body, *operands)`` and a fused
    ``call_module`` bind their positional arguments.
    """
    if node.op == "call_module":
        body = _subgraph(graph_module, node.target)
        return [(body, list(node.args))] if body is not None else []
    if node.op != "call_function":
        return []
    if node.target is _WHILE_LOOP:
        carried = list(node.args[2]) if len(node.args) > 2 else []
        extra = list(node.args[3]) if len(node.args) > 3 else []
        operands = carried + extra
        return [
            (body, operands)
            for body in (
                _subgraph(graph_module, node.args[0].target),
                _subgraph(graph_module, node.args[1].target),
            )
            if body is not None
        ]
    if node.target is _COND:
        operands = list(node.args[3]) if len(node.args) > 3 else []
        return [
            (body, operands)
            for body in (
                _subgraph(graph_module, node.args[1].target),
                _subgraph(graph_module, node.args[2].target),
            )
            if body is not None
        ]
    if node.target is _COMMIT:
        body = _subgraph(graph_module, node.args[0].target)
        return [(body, list(node.args[1:]))] if body is not None else []
    return []


#: Relations whose result lives in somebody else's backing. The chain
#: continues through these; anything else ends it.
_SHARING = {
    StorageRelationKind.ALIAS,
    StorageRelationKind.VIEW,
    StorageRelationKind.WRITE_INTO,
    StorageRelationKind.COMPONENT,
}


def _tensor_value_paths(value, path=()):
    """Yield exact paths to tensor leaves in an FX result value."""
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_value_paths(item, path + (index,))


def _normalize_getitem_index(
    producer: Node,
    index: int,
    producer_outputs,
) -> int:
    """Normalize one Python sequence index against an aggregate result."""
    value = recorded_value(producer)
    if isinstance(value, (tuple, list)):
        length = len(value)
    else:
        top_level_indices = {
            output.path[0] for output in producer_outputs if output.path
        }
        length = max(top_level_indices, default=-1) + 1
    if index < 0 and -length <= index:
        return length + index
    return index


def _trace_storage_relations(
    logical,
    relation_lookup: Callable[[ValueRef], StorageRelation],
    current: ValueRef,
    *,
    follow_kinds=None,
    follow_copy=None,
    follow_write_input=None,
):
    sharing_kinds = set(_SHARING if follow_kinds is None else follow_kinds)
    chain = []
    visited = set()
    while True:
        if current in visited:
            names = " -> ".join(ref.name for ref in (*chain, current))
            raise ValueError(f"Storage-relation cycle: {names}")
        visited.add(current)
        chain.append(current)
        relation = relation_lookup(current)
        if (
            relation.kind is StorageRelationKind.WRITE_INTO
            and follow_write_input is not None
            and follow_write_input(current)
        ):
            inputs = logical[current].inputs
            if len(inputs) != 1:
                break
            current = inputs[0]
            continue
        follows_source = relation.kind in sharing_kinds or (
            relation.kind is StorageRelationKind.COPY
            and follow_copy is not None
            and follow_copy(current)
        )
        if not follows_source or relation.source is None:
            break
        current = relation.source
    return tuple(chain)


class GraphIndex:
    """Stable logical-tensor and topology index for an FX graph scope.

    ``recursive`` preserves the storage graph's historical behavior by
    indexing region bodies as well as the outer graph. ``outer_fx`` indexes
    only the final graph whose nodes can appear in an ``ExecutionOrder``;
    terminal non-bufferized targets opt into that scope explicitly.
    """

    def __init__(
        self,
        model: GraphModule,
        *,
        scope: str = GRAPH_SCOPE_RECURSIVE,
    ):
        if scope not in GRAPH_SCOPES:
            raise ValueError(
                f"unknown storage graph scope {scope!r}; accepted: "
                + ", ".join(GRAPH_SCOPES)
            )
        self.model = model
        self.scope = scope
        self.graph_modules = self._collect_graph_modules()
        nodes = tuple(
            node
            for graph_module in self.graph_modules
            for node in graph_module.graph.nodes
        )
        self.source_positions = {
            node: position for position, node in enumerate(nodes)
        }
        self.node_outputs: Dict[Node, tuple[ValueRef, ...]] = {}
        self.node_inputs: Dict[Node, tuple[ValueRef, ...]] = {}
        self.logical_tensors: Dict[ValueRef, LogicalTensor] = {}
        self._build(nodes)

    def _collect_graph_modules(self):
        modules = []
        visited = set()

        def visit(graph_module):
            identity = id(graph_module)
            if identity in visited:
                return
            visited.add(identity)
            modules.append(graph_module)
            if self.scope == GRAPH_SCOPE_OUTER_FX:
                return
            for node in graph_module.graph.nodes:
                for body, _operands in _region_bindings(graph_module, node):
                    visit(body)

        visit(self.model)
        return tuple(modules)

    @staticmethod
    def _outputs_for(node: Node) -> tuple[ValueRef, ...]:
        if node.op == "output":
            return ()
        value = recorded_value(node)
        paths = tuple(path for path, _ in _tensor_value_paths(value))
        if paths:
            return tuple(ValueRef(node, path) for path in paths)
        # Preserve the old analysis's conservative fallback for nodes whose
        # propagated value is unavailable or not tensor-like.
        if isinstance(value, (tuple, list)):
            return tuple(
                ValueRef(node, (index,)) for index in range(len(value))
            )
        return (ValueRef(node),)

    def _input_refs(self, node: Node) -> tuple[ValueRef, ...]:
        if (
            node.target is operator.getitem
            and node.args
            and isinstance(node.args[0], Node)
            and len(node.args) > 1
            and isinstance(node.args[1], int)
        ):
            producer = node.args[0]
            outputs = self.node_outputs.get(producer, ())
            index = _normalize_getitem_index(
                producer,
                node.args[1],
                outputs,
            )
            selected = tuple(
                output for output in outputs if output.path[:1] == (index,)
            )
            if selected:
                return selected

        refs = []
        seen = set()
        for input_node in node.all_input_nodes:
            for ref in self.node_outputs.get(input_node, ()):
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
        return tuple(refs)

    @staticmethod
    def _shape_dtype(ref: ValueRef):
        declared_dtype = ref.node.meta.get("dtype")
        if ref.path and isinstance(declared_dtype, (tuple, list)):
            try:
                for index in ref.path:
                    declared_dtype = declared_dtype[index]
            except (IndexError, TypeError):
                declared_dtype = None

        value = recorded_value(ref.node)
        for path, tensor in _tensor_value_paths(value):
            if path == ref.path:
                dtype = (
                    declared_dtype
                    if declared_dtype is not None
                    else tensor.dtype
                )
                return tuple(tensor.shape), dtype

        shape = getattr(ref.node, "shape", ())
        return tuple(shape or ()), declared_dtype

    def _build(self, nodes) -> None:
        for node in nodes:
            self.node_outputs[node] = self._outputs_for(node)
        for node in nodes:
            self.node_inputs[node] = self._input_refs(node)

        users = defaultdict(set)
        for node, inputs in self.node_inputs.items():
            for ref in inputs:
                users[ref].add(node)

        for node in nodes:
            inputs = self.node_inputs[node]
            for ref in self.node_outputs[node]:
                shape, dtype = self._shape_dtype(ref)
                self.logical_tensors[ref] = LogicalTensor(
                    ref=ref,
                    shape=shape,
                    dtype=dtype,
                    inputs=inputs,
                    users=tuple(
                        sorted(
                            users.get(ref, ()),
                            key=self.source_positions.__getitem__,
                        )
                    ),
                )

    def ref(self, node: Node, path=()) -> ValueRef:
        candidate = ValueRef(node, path)
        if candidate not in self.logical_tensors:
            raise KeyError(f"{candidate.name} is not a logical tensor")
        return candidate


class StorageGraph:
    """Every logical tensor in a model, and its relation, per plane.

    A target may declare a relation of its own for any value. That is the
    seam Alpha's concatenation slots and vector-scale components arrive
    through, and it *replaces* the generic answer rather than being merged
    with it -- two relations for one value is a contradiction, not a refinement,
    so declaring one twice differently is refused.
    """

    def __init__(
        self,
        model,
        planes: Iterable[str] = (StoragePlane.DRAM,),
        *,
        scope: str = GRAPH_SCOPE_RECURSIVE,
    ):
        self.model = model
        self.scope = scope
        self.index = GraphIndex(model, scope=scope)
        self.planes = tuple(StoragePlane(plane).value for plane in planes)
        self.logical: Dict[ValueRef, LogicalTensor] = {}
        self._generic: Dict[ValueRef, StorageRelation] = {}
        self._declared: Dict[str, Dict[ValueRef, StorageRelation]] = {
            plane: {} for plane in self.planes
        }
        self._components: Dict[
            PhysicalTensorId, List[PhysicalTensorComponent]
        ] = {}
        self._auxiliary = {plane: set() for plane in self.planes}
        self._auxiliary_representations: Dict[
            PhysicalTensorId, tuple[int, str]
        ] = {}
        self._component_declarations: Dict[
            str, List[_PhysicalComponentDeclaration]
        ] = {plane: [] for plane in self.planes}
        self._serialized_artifact_declarations: Dict[
            str, List[_SerializedArtifactDeclaration]
        ] = {plane: [] for plane in self.planes}
        self._accesses: Dict[PhysicalTensorId, List[PhysicalAccess]] = {}
        self._access_declarations = {plane: [] for plane in self.planes}
        self._transfers: List[StorageTransfer] = []
        self._build()

    # -- construction ------------------------------------------------------

    def _build(self) -> None:
        self._walk(self.model)

    def _walk(self, graph_module) -> None:
        """Record every value, descending into control-flow regions.

        A region's body is not a separate program. Its placeholders are other
        names for the buffers the region's operands name, and its own
        operations read and write those same buffers -- so a walk that stopped
        at the region boundary would see a loop body's accumulator as a value
        with no producer and no storage, and the buffer it accumulates into as
        dead the moment the loop starts.
        """
        for node in graph_module.graph.nodes:
            for ref in self._results(node):
                self.logical[ref] = self.index.logical_tensors[ref]
                self._generic[ref] = generic_storage_relation(ref, node)
            if self.scope == GRAPH_SCOPE_RECURSIVE:
                self._bind_region(graph_module, node)

    def _bind_region(self, graph_module, node) -> None:
        """Tie a region's placeholders to the operands it was given.

        Program order does the work: a region appears after the values it is
        handed, so by the time its body is walked every operand already has a
        relation, and one pass suffices -- no fixpoint.

        The body is walked **first** and the binding written after, and the
        order is the whole point. ``_walk`` gives every placeholder the
        generic rule, and for a placeholder that rule is ``MATERIALIZE`` -- a
        backing of its own. Binding first and walking second therefore
        overwrites each binding with the very answer it exists to correct, and
        the result still looks plausible: a body that reads and writes buffers
        nobody else names. It only shows up as a partition far finer than the
        one ``_buffer_identity`` builds.
        """
        for body, operands in _region_bindings(graph_module, node):
            self._walk(body)
            placeholders = [
                inner for inner in body.graph.nodes if inner.op == "placeholder"
            ]
            for placeholder, operand in zip(placeholders, operands):
                if not isinstance(operand, Node):
                    continue
                self._generic[ValueRef(placeholder)] = StorageRelation(
                    result=ValueRef(placeholder),
                    kind=StorageRelationKind.ALIAS,
                    source=ValueRef(operand),
                )

    def _results(self, node) -> Iterable[ValueRef]:
        """The value or values one node produces.

        A multi-output operation is several logical tensors, and each needs an
        identity of its own -- otherwise the four results of a ``chunk`` share
        one placement, which is exactly the thing that has to be decidable
        separately.
        """
        return self.index.node_outputs.get(node, ())

    def ref(self, value, path=()) -> ValueRef:
        """Return and validate one exact logical-tensor reference."""
        if isinstance(value, ValueRef):
            if value not in self.logical:
                raise KeyError(f"{value.name} is not a logical tensor")
            return value
        return self.index.ref(value, path)

    def output_refs(self, node: Node) -> tuple[ValueRef, ...]:
        """Return the exact logical tensor leaves produced by ``node``."""
        return self.index.node_outputs.get(node, ())

    # -- target declarations ----------------------------------------------

    def declare(self, plane: str, relation: StorageRelation) -> None:
        """Record a target's relation for one value, in one plane."""
        plane = StoragePlane(plane).value
        if plane not in self.planes:
            raise ValueError(f"{plane} is not a plane of this graph")
        existing = self._declared[plane].get(relation.result)
        if existing is not None and existing != relation:
            raise ValueError(
                f"{relation.result.name}: two different storage relations "
                f"declared in {plane} -- {existing.kind.value} and "
                f"{relation.kind.value}"
            )
        self._declared[plane][relation.result] = relation

    def declare_component(
        self, identifier: PhysicalTensorId, component: PhysicalTensorComponent
    ) -> None:
        """Declare a component against an already-known physical identity.

        Kept for callers that deliberately construct an auxiliary identity.
        A target component attached to an FX value should use
        :meth:`register_physical_component`, because relation chains can move
        that value onto another backing before finalization.
        """
        if identifier.plane not in self.planes:
            raise ValueError(
                f"{identifier.name}: component plane is not in this graph"
            )
        self._components.setdefault(identifier, []).append(component)

    def add_auxiliary(self, owner, name: str, *, plane: str) -> PhysicalTensorId:
        """Create target-owned storage with no logical FX result of its own.

        This mirrors the fork's storage boundary.  For example, an Alpha
        channel shuffle has one FX result but two independently addressed
        output backings; the second is an auxiliary physical identity owned by
        the shuffle, not a synthetic logical node.
        """
        plane_name = StoragePlane(plane).value
        if plane_name not in self.planes:
            raise ValueError(f"{plane_name} is not a plane of this graph")
        identifier = PhysicalTensorId(
            plane=plane_name,
            owner=self.ref(owner),
            component=str(name),
        )
        self._auxiliary[plane_name].add(identifier)
        return identifier

    def bind_physical(
        self,
        result,
        physical: PhysicalTensorId,
        *,
        plane: str,
        kind=StorageRelationKind.VIEW,
        transform=None,
    ) -> StorageRelation:
        """Bind a logical result to a previously declared auxiliary backing."""
        plane_name = StoragePlane(plane).value
        if physical.plane != plane_name:
            raise ValueError(
                "a logical tensor and its physical storage must use the same "
                "plane"
            )
        if physical not in self._auxiliary.get(plane_name, ()):
            raise ValueError(
                "physical storage must be declared with add_auxiliary() "
                "before binding a logical tensor"
            )
        return self.set_relation(
            result,
            kind,
            plane=plane_name,
            physical_source=physical,
            transform=transform,
        )

    def register_physical_component(
        self,
        logical_tensor,
        value: torch.Tensor,
        *,
        plane: str,
        component: str = "primary",
        dtype=None,
        bits: Optional[int] = None,
        backing_size_bytes: int,
        layout: str,
    ) -> None:
        """Declare target-owned bytes before their physical ID is known.

        The logical value is resolved through all final alias, view,
        write-into, and component relations when the plan is built. Every
        declaration that lands on one backing must agree on its total extent
        and layout. This keeps target packing policy out of the shared
        relation vocabulary while still giving allocation one authoritative
        representation.
        """
        plane = StoragePlane(plane).value
        if plane not in self.planes:
            raise ValueError(f"{plane} is not a plane of this graph")
        if isinstance(logical_tensor, PhysicalTensorId):
            identifier = logical_tensor
            if identifier.plane != plane:
                raise ValueError(
                    f"{identifier.name}: component plane does not match "
                    f"{plane}"
                )
            if identifier not in self._auxiliary[plane]:
                raise ValueError(
                    f"{identifier.name}: physical component may only target "
                    "a declared auxiliary identity"
                )
            ref = identifier.owner
        else:
            identifier = None
            ref = self.ref(logical_tensor)
        if ref not in self.logical:
            raise ValueError(f"{ref.name}: no logical tensor in this graph")
        if not isinstance(value, torch.Tensor):
            raise TypeError("physical component value must be a tensor")
        if not component:
            raise ValueError("physical component name must not be empty")
        if not layout:
            raise ValueError("physical component layout must not be empty")
        dtype = value.dtype if dtype is None else dtype
        bits = round(dtype_byte_size(dtype) * 8) if bits is None else int(bits)
        if bits <= 0:
            raise ValueError("physical component width must be positive")
        byte_size = math.ceil(value.numel() * bits / 8)
        backing_size_bytes = int(backing_size_bytes)
        if backing_size_bytes < byte_size:
            raise ValueError(
                f"{ref.name}: {component} needs {byte_size} bytes, but its "
                f"backing declares {backing_size_bytes}"
            )
        representation = PhysicalTensorComponent(
            name=component,
            logical_tensor=ref,
            shape=tuple(int(dimension) for dimension in value.shape),
            dtype=dtype,
            bits=bits,
            byte_size=byte_size,
            value=value,
        )
        if identifier is not None:
            physical_representation = (backing_size_bytes, str(layout))
            existing = self._auxiliary_representations.get(identifier)
            if existing is not None and existing != physical_representation:
                raise ValueError(
                    f"{identifier.name}: conflicting backing representation"
                )
            self._auxiliary_representations[identifier] = (
                physical_representation
            )
            declared = self._components.setdefault(identifier, [])
            if representation not in declared:
                declared.append(representation)
            return
        declaration = _PhysicalComponentDeclaration(
            logical_tensor=ref,
            component=representation,
            backing_size_bytes=backing_size_bytes,
            layout=str(layout),
        )
        self._component_declarations[plane].append(declaration)

    def register_serialized_artifact(
        self,
        logical_tensor,
        value: torch.Tensor,
        *,
        plane: str,
        name: str,
        dtype=None,
        bits: Optional[int] = None,
    ) -> None:
        """Register a target encoding that is not resident in its backing.

        The logical value, rather than a guessed physical ID, is recorded here
        because later target relations may move it onto another backing.  The
        artifact is resolved only when the storage plan is finalized.
        """
        plane_name = StoragePlane(plane).value
        if plane_name not in self.planes:
            raise ValueError(f"{plane_name} is not a plane of this graph")
        logical_ref = self.ref(logical_tensor)
        if not isinstance(value, torch.Tensor):
            raise TypeError("serialized artifact value must be a tensor")
        if not name:
            raise ValueError("serialized artifact name must not be empty")
        dtype = value.dtype if dtype is None else dtype
        bits = (
            round(dtype_byte_size(dtype) * 8) if bits is None else int(bits)
        )
        if bits <= 0:
            raise ValueError("serialized artifact width must be positive")
        declaration = _SerializedArtifactDeclaration(
            logical_tensor=logical_ref,
            name=name,
            shape=tuple(int(dimension) for dimension in value.shape),
            dtype=dtype,
            bits=bits,
            byte_size=math.ceil(value.numel() * bits / 8),
            value=value,
        )
        self._serialized_artifact_declarations[plane_name].append(declaration)

    def declare_access(
        self, identifier: PhysicalTensorId, access: PhysicalAccess
    ) -> None:
        self._accesses.setdefault(identifier, []).append(access)

    def add_access(
        self,
        storage,
        user: Node,
        *,
        plane,
        role,
        kind: str,
    ) -> None:
        """Record a physical access hidden from ordinary FX dataflow.

        A logical declaration is resolved only after all target relations are
        known. ``declare_access`` remains available for callers that already
        hold an exact physical identity.
        """
        plane_name = StoragePlane(plane).value
        if plane_name not in self.planes:
            raise ValueError(f"{plane_name} is not a plane of this graph")
        if user not in self.index.source_positions:
            raise KeyError(f"{user.name} is not in the indexed graph scope")
        if isinstance(storage, PhysicalTensorId):
            if storage.plane != plane_name:
                raise ValueError(
                    "a physical access must use the selected storage plane"
                )
            source = storage
        else:
            source = self.ref(storage)
        declaration = (
            source,
            PhysicalAccess(operation=user, role=role, kind=kind),
        )
        if declaration not in self._access_declarations[plane_name]:
            self._access_declarations[plane_name].append(declaration)

    def declare_transfer(self, transfer: StorageTransfer) -> None:
        self._transfers.append(transfer)

    def relation(self, plane: str, ref: ValueRef) -> StorageRelation:
        """Return one relation, accepting old and value-first query order.

        Existing callers use ``relation(plane, ref)``. Fork-style target
        preparation uses ``relation(value, plane)``; both resolve here.
        """
        if isinstance(plane, (Node, ValueRef)):
            plane, ref = ref, plane
        plane = StoragePlane(plane).value
        ref = self.ref(ref)
        declared = self._declared[plane].get(ref)
        return declared if declared is not None else self._generic[ref]

    def set_relation(
        self,
        result,
        kind,
        *,
        plane,
        source=None,
        physical_source=None,
        transform=None,
        component=None,
    ) -> StorageRelation:
        """Construct and conservatively declare a target relation."""
        relation = StorageRelation(
            result=self.ref(result),
            kind=kind,
            source=None if source is None else self.ref(source),
            physical_source=physical_source,
            transform=transform,
            component=component,
        )
        self.declare(plane, relation)
        return relation

    def trace_storage_chain(
        self,
        value,
        plane,
        *,
        follow_kinds=None,
        follow_copy=None,
        follow_write_input=None,
    ) -> tuple[ValueRef, ...]:
        """Trace a logical value to the owner of its current backing."""
        plane_name = StoragePlane(plane).value
        return _trace_storage_relations(
            self.logical,
            lambda ref: self.relation(plane_name, ref),
            self.ref(value),
            follow_kinds=follow_kinds,
            follow_copy=follow_copy,
            follow_write_input=follow_write_input,
        )

    def finalize(self, execution_order=None, end_to_end=False) -> "StoragePlan":
        """Resolve the graph into a plan.

        ``end_to_end`` is the scratchpad-residency mode: everything the program
        is handed lives in the scratchpad for the whole run, so each input is
        loaded once up front rather than staged per operation. The loads are
        derived here rather than declared by a caller because they are not a
        policy -- given the two planes, they are the only way a value that
        arrives in DRAM can be read by an array that only reaches scratchpad.
        """
        return StoragePlan(
            self, execution_order=execution_order, end_to_end=end_to_end
        )


def _is_instruction_parameter(dtype) -> bool:
    """Whether a value is encoded into an instruction rather than staged.

    Floating point says so: by the time a graph reaches this point every value
    the array reads has been quantized, so anything still floating is a
    parameter the serializer will fold into an operand field.

    An unknown dtype answers ``False`` -- load it. The two errors are not
    symmetric: a load too many wastes scratchpad, a load too few means the
    array reads a buffer nothing ever filled.
    """
    return bool(getattr(dtype, "is_floating_point", False))


def _shape(node, ref: ValueRef):
    value = recorded_value(node)
    for index in ref.path:
        if isinstance(value, (list, tuple)):
            value = value[index]
    return tuple(getattr(value, "shape", ()) or ())


def _dtype(node):
    return node.meta.get("dtype") or getattr(
        recorded_value(node), "dtype", None
    )


class StoragePlan:
    """Which backing every logical tensor has, in every plane."""

    def __init__(
        self, graph: StorageGraph, execution_order=None, end_to_end=False
    ):
        from voyager_compiler.codegen.execution_order import (
            ExecutionOrder,
            validate_execution_order,
        )

        self.graph = graph
        self.index = graph.index
        self.planes = graph.planes
        self.logical_tensors = dict(graph.logical)
        self.relations = {
            plane: {ref: graph.relation(plane, ref) for ref in graph.logical}
            for plane in self.planes
        }
        self.physical_of: Dict[str, Dict[ValueRef, PhysicalTensorId]] = {}
        self.physical: Dict[str, Dict[PhysicalTensorId, PhysicalTensor]] = {}
        # Fork-style names are aliases, not a second representation.
        self.physical_tensors = self.physical
        self.serialized_artifacts: Dict[
            str,
            Dict[tuple[PhysicalTensorId, str], SerializedTensorArtifact],
        ] = {}
        self.transfers = tuple(graph._transfers)
        self.intervals: Dict[PhysicalTensorId, StorageInterval] = {}
        self.intervals_by_plane = {plane: {} for plane in self.planes}
        self._release_after = {plane: {} for plane in self.planes}
        self._accesses_by_plane = {plane: {} for plane in self.planes}
        if execution_order is None:
            self.execution_order = None
            ordered_nodes = None
        elif isinstance(execution_order, ExecutionOrder):
            validate_execution_order(graph.model, execution_order)
            self.execution_order = execution_order
            ordered_nodes = execution_order.nodes
        else:
            self.execution_order = ExecutionOrder(tuple(execution_order))
            validate_execution_order(graph.model, self.execution_order)
            ordered_nodes = self.execution_order.nodes

        for plane in self.planes:
            resolved = self._resolve_plane(plane)
            self.physical_of[plane] = resolved
            self.serialized_artifacts[plane] = (
                self._resolve_serialized_artifacts(
                    plane,
                    resolved,
                    graph._serialized_artifact_declarations[plane],
                )
            )
            representations = self._resolve_representations(
                plane, resolved
            )
            accesses = self._resolve_accesses(plane, resolved)
            self._accesses_by_plane[plane] = accesses
            self.physical[plane] = self._build_plane(
                plane, representations, accesses
            )
        if end_to_end:
            self._add_entry_loads()
        if ordered_nodes is not None:
            self._build_intervals(ordered_nodes)
        self.validate()

    # -- resolution --------------------------------------------------------

    def _resolve_plane(self, plane: str) -> Dict[ValueRef, PhysicalTensorId]:
        resolved: Dict[ValueRef, PhysicalTensorId] = {}
        for ref in self.graph.logical:
            resolved[ref] = self._chase(plane, ref, resolved, set())
        return resolved

    def _chase(self, plane, ref, resolved, seen) -> PhysicalTensorId:
        if ref in resolved:
            return resolved[ref]
        if ref in seen:
            raise ValueError(
                f"{ref.name}: its storage relation is a cycle, so nothing in "
                f"it owns bytes"
            )
        seen = seen | {ref}

        relation = self.graph.relation(plane, ref)
        if relation.physical_source is not None:
            return relation.physical_source
        if relation.kind not in _SHARING or relation.source is None:
            return PhysicalTensorId(plane, ref)

        source = relation.source
        if source not in self.graph.logical:
            # A relation naming a value this plane does not describe -- a
            # parameter in a scratchpad-only graph, say. The result owns its
            # bytes here, which is the honest answer rather than an error.
            return PhysicalTensorId(plane, ref)

        backing = self._chase(plane, source, resolved, seen)
        # A component is another typed interpretation *inside the same
        # allocation*. ``relation.component`` selects that interpretation;
        # making it part of the physical ID would split the VSQ payload and
        # its fine scale into two allocations and defeat the relation.
        return backing

    def _logical_component(
        self, ref: ValueRef, name: str = "primary"
    ) -> Optional[PhysicalTensorComponent]:
        logical = self.graph.logical[ref]
        value = recorded_value(ref.node)
        try:
            for index in ref.path:
                value = value[index]
        except (IndexError, TypeError):
            value = None
        value = value if isinstance(value, torch.Tensor) else None
        dtype = logical.dtype or getattr(value, "dtype", None)
        try:
            bits = round(dtype_byte_size(dtype) * 8)
            byte_size = math.ceil(math.prod(logical.shape) * bits / 8)
        except (TypeError, ValueError):
            return None
        return PhysicalTensorComponent(
            name=name,
            logical_tensor=ref,
            shape=logical.shape,
            dtype=dtype,
            bits=bits,
            byte_size=byte_size,
            value=(
                value
                if value is not None and tuple(value.shape) == logical.shape
                else None
            ),
        )

    @staticmethod
    def _components_match(first, second) -> bool:
        if first != second:
            return False
        if first.value is None or second.value is None:
            return first.value is second.value
        return torch.equal(
            first.value.detach().cpu(), second.value.detach().cpu()
        )

    @staticmethod
    def _artifact_declarations_match(first, second) -> bool:
        metadata_matches = (
            first.shape == second.shape
            and str(first.dtype) == str(second.dtype)
            and first.bits == second.bits
            and first.byte_size == second.byte_size
        )
        if not metadata_matches:
            return False
        return torch.equal(
            first.value.detach().cpu(), second.value.detach().cpu()
        )

    def _resolve_serialized_artifacts(
        self,
        plane,
        resolved,
        declarations,
    ) -> Dict[tuple[PhysicalTensorId, str], SerializedTensorArtifact]:
        """Resolve artifact declarations through finalized storage relations."""
        by_key = defaultdict(list)
        for declaration in declarations:
            physical = resolved[declaration.logical_tensor]
            by_key[(physical, declaration.name)].append(declaration)

        artifacts = {}
        for (physical, name), candidates in by_key.items():
            first = candidates[0]
            if any(
                not self._artifact_declarations_match(first, candidate)
                for candidate in candidates[1:]
            ):
                consumers = ", ".join(
                    candidate.logical_tensor.name for candidate in candidates
                )
                raise ValueError(
                    f"{physical.name}: conflicting serialized artifact "
                    f"{name!r} from {consumers}"
                )
            artifacts[(physical, name)] = SerializedTensorArtifact(
                physical_tensor=physical,
                name=name,
                logical_tensor=first.logical_tensor,
                shape=first.shape,
                dtype=first.dtype,
                bits=first.bits,
                byte_size=first.byte_size,
                value=first.value,
            )
        return artifacts

    def _resolve_representations(self, plane, resolved):
        """Resolve target declarations onto the final shared backing."""
        members = defaultdict(list)
        for ref, identifier in resolved.items():
            members[identifier].append(ref)

        identifiers = set(members)
        identifiers.update(
            self.graph._auxiliary.get(plane, ())
        )

        declarations = defaultdict(list)
        for item in self.graph._component_declarations[plane]:
            identifier = resolved[item.logical_tensor]
            relation = self.graph.relation(plane, item.logical_tensor)
            if (
                relation.kind is StorageRelationKind.COMPONENT
                and relation.component != item.component.name
            ):
                raise ValueError(
                    f"{item.logical_tensor.name}: component "
                    f"{item.component.name!r} conflicts with relation "
                    f"{relation.component!r}"
                )
            declarations[identifier].append(item)

        result = {}
        for identifier in identifiers:
            refs = members.get(identifier, ())
            exact = list(self.graph._components.get(identifier, ()))
            target = declarations.get(identifier, ())
            auxiliary = self.graph._auxiliary_representations.get(identifier)
            backing_size = None if auxiliary is None else auxiliary[0]
            layout = None if auxiliary is None else auxiliary[1]
            if target:
                sizes = {item.backing_size_bytes for item in target}
                layouts = {item.layout for item in target}
                if len(sizes) != 1 or len(layouts) != 1:
                    details = ", ".join(
                        f"{item.logical_tensor.name}:{item.component.name}"
                        f"[{item.backing_size_bytes} B, {item.layout}]"
                        for item in target
                    )
                    raise ValueError(
                        f"{identifier.name}: conflicting backing "
                        f"representations: {details}"
                    )
                backing_size = next(iter(sizes))
                layout = next(iter(layouts))
                exact.extend(item.component for item in target)

            by_name = defaultdict(list)
            for item in exact:
                by_name[item.name].append(item)
            components = []
            for name, candidates in sorted(by_name.items()):
                first = candidates[0]
                if any(
                    not self._components_match(first, candidate)
                    for candidate in candidates[1:]
                ):
                    raise ValueError(
                        f"{identifier.name}: conflicting component {name!r}"
                    )
                components.append(first)

            required = {
                relation.component
                for ref in refs
                for relation in (self.graph.relation(plane, ref),)
                if relation.kind is StorageRelationKind.COMPONENT
            }
            declared = {item.name for item in components}
            if required - declared:
                raise ValueError(
                    f"{identifier.name}: missing physical components "
                    f"{sorted(required - declared)}"
                )
            has_primary = any(
                self.graph.relation(plane, ref).kind
                is not StorageRelationKind.COMPONENT
                for ref in refs
            )
            if target and has_primary and "primary" not in declared:
                raise ValueError(
                    f"{identifier.name}: target representation has no "
                    "primary component"
                )

            if not components:
                if not refs:
                    raise ValueError(
                        f"{identifier.name}: auxiliary backing has no "
                        "physical components"
                    )
                owner = (
                    identifier.owner if identifier.owner in refs else refs[0]
                )
                dense = self._logical_component(owner)
                if dense is not None:
                    components = [dense]
                    backing_size = dense.byte_size
                    layout = "dense"
            elif backing_size is None:
                backing_size = max(item.byte_size for item in components)

            result[identifier] = (
                tuple(components),
                backing_size,
                layout,
            )
        return result

    def _resolve_accesses(self, plane, resolved):
        """Resolve old physical and new logical access declarations."""
        physical_accesses = defaultdict(list)

        def add(identifier, access):
            if identifier.plane != plane:
                raise ValueError("a physical access crossed storage planes")
            if access not in physical_accesses[identifier]:
                physical_accesses[identifier].append(access)

        for identifier, accesses in self.graph._accesses.items():
            if identifier.plane != plane:
                continue
            for access in accesses:
                add(identifier, access)
        for storage, access in self.graph._access_declarations[plane]:
            identifier = (
                storage
                if isinstance(storage, PhysicalTensorId)
                else resolved[storage]
            )
            add(identifier, access)
        return {
            identifier: tuple(accesses)
            for identifier, accesses in physical_accesses.items()
        }

    def _build_plane(
        self, plane, representations, accesses
    ) -> Dict[PhysicalTensorId, PhysicalTensor]:
        members: Dict[PhysicalTensorId, List[ValueRef]] = {}
        relations: Dict[PhysicalTensorId, List[StorageRelation]] = {}
        for ref, identifier in self.physical_of[plane].items():
            members.setdefault(identifier, []).append(ref)
            relations.setdefault(identifier, []).append(
                self.graph.relation(plane, ref)
            )
        return {
            identifier: PhysicalTensor(
                id=identifier,
                logical_tensors=tuple(members.get(identifier, ())),
                relations=tuple(relations.get(identifier, ())),
                components=representations[identifier][0],
                backing_size_bytes=representations[identifier][1],
                layout=representations[identifier][2],
                accesses=accesses.get(identifier, ()),
            )
            for identifier in representations
        }

    def _build_intervals(self, execution_order) -> None:
        """When each backing is live, half-open over the given order.

        A backing is born where its earliest member is produced and dies after
        its latest reader, taken over *every* member -- which is the whole
        point of the relation: two names for one backing keep it alive between
        them.

        A value the graph returns lives to the end. The ``output`` node is not
        an execution position, so a backing read only by it would otherwise
        look dead one step after it was written, and an allocator would be
        free to hand its address to something else while the caller still
        needs it.
        """
        order = list(execution_order)
        position = {node: index for index, node in enumerate(order)}
        end = len(order)
        returned = {
            ref
            for node in order
            if getattr(node, "op", None) == "output"
            for ref in self.index.node_inputs.get(node, ())
        }
        for plane in self.planes:
            for identifier, tensor in self.physical[plane].items():
                births, deaths = [], []
                for ref in tensor.logical_tensors:
                    at = position.get(ref.node)
                    if at is None:
                        continue
                    births.append(at)
                    deaths.append(at + 1)
                    if ref in returned:
                        deaths.append(end)
                    for user in self.graph.logical[ref].users:
                        used = position.get(user)
                        if used is not None:
                            deaths.append(used + 1)
                    if (
                        self.graph.scope == GRAPH_SCOPE_OUTER_FX
                        and ref.node.op == "get_attr"
                    ):
                        # A flat target's constant remains resident through
                        # the plan. The extra terminal position deliberately
                        # has no release node.
                        deaths.append(end + 1)
                for access in self._accesses_by_plane[plane].get(
                    identifier, ()
                ):
                    used = position.get(access.operation)
                    if used is not None:
                        births.append(used)
                        deaths.append(used + 1)
                if (
                    not births
                    and identifier in self.graph._auxiliary.get(plane, ())
                ):
                    # Match the fork's conservative fallback: a target-owned
                    # auxiliary is at least live at its owning operation even
                    # before a target supplies more precise hidden accesses.
                    owned_at = position.get(identifier.owner.node)
                    if owned_at is not None:
                        births.append(owned_at)
                        deaths.append(owned_at + 1)
                if not births:
                    continue
                interval = StorageInterval(identifier, min(births), max(deaths))
                self.intervals[identifier] = interval
                self.intervals_by_plane[plane][identifier] = interval
        self._index_releases(position)

    def _index_releases(self, position) -> None:
        by_position = {value: node for node, value in position.items()}
        for plane, intervals in self.intervals_by_plane.items():
            releases = defaultdict(list)
            for identifier, interval in intervals.items():
                node = by_position.get(interval.death - 1)
                if node is not None:
                    releases[node].append(identifier)
            self._release_after[plane] = {
                node: tuple(sorted(items, key=lambda item: item.name))
                for node, items in releases.items()
            }

    # -- queries -----------------------------------------------------------

    def _add_entry_loads(self) -> None:
        """One DRAM -> scratchpad load for every value the program is handed.

        A *live* one: an input nothing reads needs no load, and emitting one
        would reserve scratchpad for a value that is never read -- which in
        end-to-end mode, where the residency lasts the whole run, is capacity
        taken from something that needs it.

        Only inputs. A value an operation computes is already produced into
        the plane its consumer reads from, so a load for it would be movement
        that never happens.

        And only inputs the array reads as *data*. A floating-point input is a
        requantization parameter -- a multiplier, a shift, a bias -- that the
        serializer encodes into an instruction field, so it never occupies the
        scratchpad and loading it would reserve capacity for something the
        array never reads from memory. An integer input is already in its
        device representation.

        That split is read off the reference rather than declared: of its 420
        live inputs, the 122 it loads are exactly the ones whose dtype is not
        floating point (36 int4, 36 uint8, 50 int8), and the 298 it does not
        are exactly the float32 ones (66 `_ppu_primary_multiplier`, 66
        `_ppu_bias`, 163 `_scale`, 3 others). The match is exact both ways.
        """
        dram = StoragePlane.DRAM.value
        scratchpad = StoragePlane.SCRATCHPAD.value
        if dram not in self.planes or scratchpad not in self.planes:
            raise ValueError(
                "end-to-end residency needs both planes; this graph has "
                f"{sorted(self.planes)}"
            )
        loads = []
        for ref, logical in self.graph.logical.items():
            if getattr(ref.node, "op", None) not in ("placeholder", "get_attr"):
                continue
            if not logical.users:
                continue
            if _is_instruction_parameter(logical.dtype):
                continue
            loads.append(
                StorageTransfer(
                    logical_tensor=ref,
                    source=self.physical_of[dram][ref],
                    destination=self.physical_of[scratchpad][ref],
                )
            )
        self.transfers = self.transfers + tuple(loads)

    def ref(self, value, path=()) -> ValueRef:
        """Return and validate one exact logical-tensor reference."""
        if isinstance(value, ValueRef):
            if value not in self.logical_tensors:
                raise KeyError(f"{value.name} is not a logical tensor")
            return value
        return self.index.ref(value, path)

    def relation(self, plane, ref) -> StorageRelation:
        """Return one frozen relation in old or value-first query order."""
        if isinstance(plane, (Node, ValueRef)):
            plane, ref = ref, plane
        plane_name = StoragePlane(plane).value
        return self.relations[plane_name][self.ref(ref)]

    def trace_storage_chain(
        self,
        value,
        plane,
        *,
        follow_kinds=None,
        follow_copy=None,
        follow_write_input=None,
    ) -> tuple[ValueRef, ...]:
        """Trace immutable plan relations to their physical owner."""
        plane_name = StoragePlane(plane).value
        return _trace_storage_relations(
            self.logical_tensors,
            lambda ref: self.relations[plane_name][ref],
            self.ref(value),
            follow_kinds=follow_kinds,
            follow_copy=follow_copy,
            follow_write_input=follow_write_input,
        )

    def output_refs(self, node: Node) -> tuple[ValueRef, ...]:
        """Return the exact logical tensor leaves produced by ``node``."""
        return self.index.node_outputs.get(node, ())

    def has_logical_value(self, value, path=()) -> bool:
        ref = value if isinstance(value, ValueRef) else ValueRef(value, path)
        return ref in self.logical_tensors

    def physical_id(self, plane, ref: ValueRef) -> PhysicalTensorId:
        """Resolve storage in old ``(plane, ref)`` or value-first order."""
        if isinstance(plane, (Node, ValueRef)):
            plane, ref = ref, plane
        plane_name = StoragePlane(plane).value
        return self.physical_of[plane_name][self.ref(ref)]

    def physical_tensor(self, plane, identifier) -> PhysicalTensor:
        """Return a backing in old or value-first query order."""
        if isinstance(plane, (Node, ValueRef, PhysicalTensorId)):
            plane, identifier = identifier, plane
        plane_name = StoragePlane(plane).value
        if not isinstance(identifier, PhysicalTensorId):
            identifier = self.physical_id(plane_name, identifier)
        return self.physical[plane_name][identifier]

    def physical_component(
        self,
        value,
        plane,
        component: str = "primary",
    ) -> PhysicalTensorComponent:
        return self.physical_tensor(value, plane).component(component)

    def backing_size_bytes(self, value, plane) -> int:
        tensor = self.physical_tensor(value, plane)
        if tensor.backing_size_bytes is None:
            raise ValueError(
                f"{tensor.id.name}: physical backing size is unknown"
            )
        return tensor.backing_size_bytes

    def logical_size_bytes(self, value) -> int:
        ref = self.ref(value)
        component = self._logical_component(ref, "logical")
        if component is None:
            raise ValueError(f"{ref.name}: logical tensor size is unknown")
        return component.byte_size

    def payload_ref(
        self,
        value,
        plane,
        component: str,
        segment,
    ) -> PhysicalPayloadRef:
        """Bind one finalized physical component to an explicit placement."""
        physical = self.storage_key(value, plane)
        representation = self.physical_component(
            physical,
            plane,
            component,
        )
        tensor = self.physical_tensor(physical, plane)
        if tensor.backing_size_bytes is None:
            raise ValueError(
                f"{physical.name}: physical backing size is unknown"
            )
        placement_size = segment.end - segment.start
        if placement_size < tensor.backing_size_bytes:
            raise ValueError(
                f"{physical.name}: placement is smaller than its "
                "physical backing"
            )
        return PhysicalPayloadRef(
            physical_tensor=physical,
            representation=representation,
            segment=segment,
        )

    def serialized_artifact(
        self,
        value,
        plane,
        name: str,
    ) -> SerializedTensorArtifact:
        """Return one finalized nonresident target encoding."""
        physical = self.storage_key(value, plane)
        plane_name = StoragePlane(plane).value
        try:
            return self.serialized_artifacts[plane_name][(physical, name)]
        except KeyError as error:
            raise KeyError(
                f"{physical.name}: serialized artifact {name!r} "
                "is not declared"
            ) from error

    def serialized_ref(
        self,
        value,
        plane,
        name: str,
    ) -> SerializedTensorRef:
        """Return a finalized reference to a nonresident target encoding."""
        return SerializedTensorRef(
            artifact=self.serialized_artifact(value, plane, name)
        )

    def storage_key(self, value, plane) -> PhysicalTensorId:
        plane_name = StoragePlane(plane).value
        identifier = (
            value
            if isinstance(value, PhysicalTensorId)
            else self.physical_id(value, plane_name)
        )
        if identifier.plane != plane_name:
            raise ValueError(
                f"{identifier.name}: physical storage crossed planes"
            )
        if identifier not in self.physical[plane_name]:
            raise KeyError(
                f"{identifier.name} is not declared in the storage plan"
            )
        return identifier

    def produced_physical_outputs(self, node: Node, plane):
        """Return distinct physical roots produced by one exact FX node."""
        plane_name = StoragePlane(plane).value
        produced = []
        for output in self.output_refs(node):
            identifier = self.physical_id(output, plane_name)
            relation = self.relations[plane_name][output]
            if (
                relation.kind
                in {
                    StorageRelationKind.MATERIALIZE,
                    StorageRelationKind.COPY,
                    StorageRelationKind.WRITE_INTO,
                }
                or identifier.owner.node is node
            ):
                produced.append(identifier)
        return tuple(dict.fromkeys(produced))

    def accesses(self, value, plane) -> tuple[PhysicalAccess, ...]:
        identifier = self.storage_key(value, plane)
        return self._accesses_by_plane[StoragePlane(plane).value].get(
            identifier, ()
        )

    def release_after(self, node: Node, plane):
        return self._release_after[StoragePlane(plane).value].get(node, ())

    def backings(self, plane) -> Dict[PhysicalTensorId, PhysicalTensor]:
        return self.physical[StoragePlane(plane).value]

    def trace(self, plane, ref: ValueRef) -> List[StorageRelation]:
        """The chain of relations from ``ref`` to the value that owns bytes."""
        chain, seen = [], set()
        current = ref
        while current is not None and current not in seen:
            seen.add(current)
            relation = self.graph.relation(plane, current)
            chain.append(relation)
            if relation.kind not in _SHARING:
                break
            current = relation.source
        return chain

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        for plane in self.planes:
            resolved = self.physical_of[plane]
            for ref in self.graph.logical:
                if ref not in resolved:
                    raise ValueError(
                        f"{ref.name}: no storage resolved in {plane}"
                    )
                identifier = resolved[ref]
                if identifier.plane != plane:
                    raise ValueError(
                        f"{ref.name}: resolved to {identifier.plane} storage "
                        f"while being planned in {plane}"
                    )
                relation = self.graph.relation(plane, ref)
                if relation.kind is StorageRelationKind.COMPONENT:
                    tensor = self.physical[plane].get(identifier)
                    declared = {
                        c.name for c in (tensor.components if tensor else ())
                    }
                    if relation.component not in declared:
                        raise ValueError(
                            f"{ref.name}: names component "
                            f"{relation.component!r} of {identifier.name}, "
                            f"which declares {sorted(declared)}"
                        )
            unknown_accesses = set(self._accesses_by_plane[plane]) - set(
                self.physical[plane]
            )
            if unknown_accesses:
                names = ", ".join(
                    sorted(identifier.name for identifier in unknown_accesses)
                )
                raise ValueError(
                    f"{plane}: accesses reference undeclared storage: {names}"
                )
            for identifier, _name in self.serialized_artifacts[plane]:
                if identifier not in self.physical[plane]:
                    raise ValueError(
                        f"{identifier.name}: serialized artifact references "
                        "undeclared storage"
                    )
        for transfer in self.transfers:
            for end in (transfer.source, transfer.destination):
                if end.plane not in self.planes:
                    raise ValueError(
                        f"{transfer.logical_tensor.name}: transfer touches "
                        f"{end.plane}, which is not planned"
                    )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        """A readable projection, for inspection only.

        Deliberately not a source: the authoritative placement is the address
        the emitted collateral carries, and a second copy that could disagree
        with it is worth less than no copy.
        """
        return {
            "logical_tensors": [
                {
                    "name": ref.name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "inputs": [item.name for item in tensor.inputs],
                    "users": [user.name for user in tensor.users],
                }
                for ref, tensor in self.graph.logical.items()
            ],
            "physical_tensors": {
                plane: [
                    {
                        "name": identifier.name,
                        "layout": tensor.layout,
                        "backing_size_bytes": tensor.backing_size_bytes,
                        "logical_tensors": [
                            ref.name for ref in tensor.logical_tensors
                        ],
                        "components": [
                            {
                                "name": component.name,
                                "logical_tensor": component.logical_tensor.name,
                                "shape": list(component.shape),
                                "dtype": str(component.dtype),
                                "bits": component.bits,
                                "byte_size": component.byte_size,
                            }
                            for component in tensor.components
                        ],
                        "serialized_artifacts": [
                            {
                                "name": artifact.name,
                                "logical_tensor": artifact.logical_tensor.name,
                                "shape": list(artifact.shape),
                                "dtype": str(artifact.dtype),
                                "bits": artifact.bits,
                                "byte_size": artifact.byte_size,
                            }
                            for (
                                (artifact_physical, _artifact_name),
                                artifact,
                            ) in self.serialized_artifacts[plane].items()
                            if artifact_physical == identifier
                        ],
                        "relations": [
                            {
                                "result": relation.result.name,
                                "kind": relation.kind.value,
                                "source": (
                                    relation.source.name
                                    if relation.source is not None
                                    else None
                                ),
                                "component": relation.component,
                                "transform": (
                                    {
                                        "kind": relation.transform.kind,
                                        "parameters": dict(
                                            relation.transform.parameters
                                        ),
                                    }
                                    if relation.transform is not None
                                    else None
                                ),
                            }
                            for relation in tensor.relations
                        ],
                        "interval": (
                            {
                                "birth": self.intervals[identifier].birth,
                                "death": self.intervals[identifier].death,
                            }
                            if identifier in self.intervals
                            else None
                        ),
                    }
                    for identifier, tensor in sorted(
                        self.physical[plane].items(), key=lambda i: i[0].name
                    )
                ]
                for plane in self.planes
            },
            "transfers": [
                {
                    "logical_tensor": transfer.logical_tensor.name,
                    "source": transfer.source.name,
                    "destination": transfer.destination.name,
                    "kind": transfer.kind,
                }
                for transfer in self.transfers
            ],
        }
