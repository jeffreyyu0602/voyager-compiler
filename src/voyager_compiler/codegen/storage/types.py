# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Logical values, physical backings, and the relations between them.

The compiler already distinguishes these two things -- ``_buffer_identity`` in
the memory planner answers "which buffer does this FX node name", and the
emitted IR distinguishes a ``TensorBox`` from a windowed ``TensorBoxRef``. What
it does not do is give the relation a *name*. ``buffer_of`` is a
``Dict[Node, Node]``, so every way a value can come to share a backing looks
alike: a reshape, a slot window, a concatenation slot written in place, and a
narrow payload sharing bytes with its own scale are all "the same buffer".

They are not alike, and three of them are things the planner currently cannot
represent at all:

``WRITE_INTO``
    Several producers writing disjoint parts of one backing. A concatenation
    on a machine whose ``cat`` is an address calculation *is* this, and
    ``bind()`` refuses it today -- it raises when a node names two buffers,
    which is exactly the legal case.
``COMPONENT``
    One backing holding more than one typed payload. A vector-scale quantized
    tensor is an INT4 payload and a UINT8 fine scale in one allocation, and a
    model with one shape and one dtype per buffer cannot say so.
``COPY``
    A relation that is *not* sharing. Naming it matters as much as the others:
    an alias that should have been a copy is a correctness bug, and without
    the vocabulary there is nowhere to write down which one was decided.

Two ideas the rest of this rests on:

**A logical tensor is a value; a physical tensor is a backing, in one plane.**
The same value has a DRAM identity and a scratchpad identity, and they are
different physical tensors even when the allocator later gives them the same
integer address. Equal addresses in different planes are not equal storage.

**A component describes bytes, not shapes.** It may have a padded shape or a
narrower dtype than the logical tensor that asked for it, because that is what
the datapath stores.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

import torch

from voyager_compiler.hardware_config import TensorRole

__all__ = [
    "LogicalTensor",
    "PhysicalAccess",
    "PhysicalPayloadRef",
    "PhysicalTensor",
    "PhysicalTensorComponent",
    "PhysicalTensorId",
    "SerializedTensorArtifact",
    "SerializedTensorRef",
    "StorageInterval",
    "StoragePlane",
    "StorageRelation",
    "StorageRelationKind",
    "StorageTransfer",
    "ValueRef",
    "ViewTransform",
]


class StoragePlane(str, Enum):
    """A storage space a value can have an identity in.

    The values are the names the emitted IR uses for the two placements a
    tensor carries, so a plane never has to be translated on the way out.
    """

    DRAM = "memory"
    SCRATCHPAD = "scratchpad"


@dataclass(frozen=True)
class ValueRef:
    """One logical tensor: an FX result, and a path into it if it is a tuple.

    The path is what makes a multi-output operation addressable. Without it a
    ``chunk`` would be one value, and the four things downstream reads from it
    would have no identity to carry a placement on.
    """

    node: Any
    path: Tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "path", tuple(int(i) for i in self.path))

    @property
    def name(self) -> str:
        return self.node.name + "".join(f"[{i}]" for i in self.path)


@dataclass(frozen=True)
class PhysicalTensorId:
    """The identity of one backing, within one plane.

    Deliberately *not* an allocation: it says which storage this is, not
    where it ended up. Whether it is resident, and at what address, is the
    allocator's answer and is recorded separately -- so a plan can be checked
    for consistency before any address exists.
    """

    plane: str
    owner: ValueRef
    component: str = "primary"

    def __post_init__(self):
        object.__setattr__(self, "plane", StoragePlane(self.plane).value)
        if not self.component:
            raise ValueError("a physical tensor needs a component name")

    @property
    def name(self) -> str:
        suffix = "" if self.component == "primary" else f".{self.component}"
        return f"{self.plane}:{self.owner.name}{suffix}"


@dataclass(frozen=True)
class PhysicalTensorComponent:
    """One typed payload inside a backing.

    ``shape`` and ``dtype`` are the *stored* ones. A component may be padded
    to a lane count the logical tensor does not have, or held in a narrower
    type than the graph's -- an INT4 payload beside its UINT8 scale in one
    allocation is two components of one backing, and neither has the logical
    tensor's dtype.
    """

    name: str
    logical_tensor: ValueRef
    shape: Tuple[int, ...]
    dtype: Any
    bits: int
    byte_size: int
    value: Optional[torch.Tensor] = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        if not self.name:
            raise ValueError("a component needs a name")
        if self.bits <= 0:
            raise ValueError(f"{self.name}: component width must be positive")
        if self.byte_size < 0:
            raise ValueError(f"{self.name}: component size must not be negative")
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        if self.value is not None:
            if not isinstance(self.value, torch.Tensor):
                raise TypeError(f"{self.name}: component value must be a tensor")
            if tuple(self.value.shape) != self.shape:
                raise ValueError(
                    f"{self.name}: component value shape "
                    f"{tuple(self.value.shape)} does not match {self.shape}"
                )


@dataclass(frozen=True)
class PhysicalPayloadRef:
    """A finalized physical component and its explicit placement.

    The storage plan owns the physical identity and representation.  The
    placement is deliberately structural: both flat target allocators and the
    legacy allocator expose ``start``, ``end``, and ``memory_space`` without
    making shared storage depend on either allocator's concrete type.
    """

    physical_tensor: PhysicalTensorId
    representation: PhysicalTensorComponent
    segment: Any

    def __post_init__(self):
        if not all(
            hasattr(self.segment, attribute)
            for attribute in ("start", "end", "memory_space")
        ):
            raise TypeError(
                "physical payload placement needs start, end, and memory_space"
            )
        placement_size = self.segment.end - self.segment.start
        if placement_size < self.representation.byte_size:
            raise ValueError(
                f"{self.physical_tensor.name}: component "
                f"{self.representation.name} exceeds its placement"
            )

    @property
    def component(self) -> str:
        return self.representation.name

    @property
    def name(self) -> str:
        """A stable serialization name unique to one physical payload."""
        physical = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in self.physical_tensor.name
        )
        component = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in self.component
        )
        return f"{physical}__{component}"


@dataclass(frozen=True)
class SerializedTensorArtifact:
    """One plan-owned tensor encoding that is not resident in a backing."""

    physical_tensor: PhysicalTensorId
    name: str
    logical_tensor: ValueRef
    shape: Tuple[int, ...]
    dtype: Any
    bits: int
    byte_size: int
    value: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self):
        if not self.name:
            raise ValueError("serialized artifact name must not be empty")
        if self.bits <= 0 or self.byte_size < 0:
            raise ValueError("serialized artifact has invalid storage size")
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        if not isinstance(self.value, torch.Tensor):
            raise TypeError("serialized artifact value must be a tensor")
        if tuple(self.value.shape) != self.shape:
            raise ValueError(
                "serialized artifact value does not match its shape"
            )


@dataclass(frozen=True)
class SerializedTensorRef:
    """A finalized reference to one plan-owned nonresident artifact."""

    artifact: SerializedTensorArtifact

    @property
    def physical_tensor(self) -> PhysicalTensorId:
        return self.artifact.physical_tensor

    @property
    def name(self) -> str:
        physical = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in self.physical_tensor.name
        )
        artifact = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in self.artifact.name
        )
        return f"{physical}__artifact__{artifact}"


class StorageRelationKind(str, Enum):
    """How a logical tensor comes to have storage."""

    #: It owns a backing of its own.
    MATERIALIZE = "materialize"
    #: Another name for the whole of one, with the same geometry.
    ALIAS = "alias"
    #: A window onto one, addressable without moving bytes.
    VIEW = "view"
    #: Its own backing, filled from another. The one relation that is *not*
    #: sharing, and naming it is what makes the others checkable.
    COPY = "copy"
    #: A part of a backing several producers write disjointly. This is what a
    #: concatenation is when the concatenation is an address calculation.
    WRITE_INTO = "write_into"
    #: One typed payload of a backing that holds more than one.
    COMPONENT = "component"


@dataclass(frozen=True)
class ViewTransform:
    """The addressing a view applies, normalized and named.

    A kind plus named parameters rather than a subclass per view: address
    lowering reads the parameters it knows and a target can declare a
    transform the compiler has never heard of -- a channel-offset write into a
    concatenated backing -- without a shared type having to grow a field.
    """

    kind: str
    parameters: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("a view transform needs a kind")
        parameters = tuple(
            (str(name), value) for name, value in self.parameters
        )
        names = tuple(name for name, _ in parameters)
        if len(set(names)) != len(names):
            raise ValueError(f"{self.kind}: parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)

    def parameter(self, name: str, default=None):
        return next(
            (value for key, value in self.parameters if key == name), default
        )


@dataclass(frozen=True)
class StorageRelation:
    """One logical tensor's claim on storage."""

    result: ValueRef
    kind: StorageRelationKind
    source: Optional[ValueRef] = None
    physical_source: Optional[PhysicalTensorId] = None
    transform: Optional[ViewTransform] = None
    component: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "kind", StorageRelationKind(self.kind))
        if self.source is not None and self.physical_source is not None:
            raise ValueError(
                f"{self.result.name}: a relation names a logical source or a "
                f"physical one, not both"
            )
        shares = self.kind in {
            StorageRelationKind.ALIAS,
            StorageRelationKind.VIEW,
            StorageRelationKind.WRITE_INTO,
            StorageRelationKind.COMPONENT,
        }
        if shares and self.source is None and self.physical_source is None:
            raise ValueError(
                f"{self.result.name}: {self.kind.value} needs a source to "
                f"share with"
            )
        if self.kind is StorageRelationKind.COPY and self.source is None:
            raise ValueError(f"{self.result.name}: a copy needs a source")
        if (
            self.kind is StorageRelationKind.COMPONENT
            and self.component is None
        ):
            raise ValueError(
                f"{self.result.name}: a component relation names which one"
            )


@dataclass(frozen=True)
class LogicalTensor:
    """One graph value, before any storage decision."""

    ref: ValueRef
    shape: Tuple[int, ...]
    dtype: Any
    inputs: Tuple[ValueRef, ...] = ()
    users: Tuple[Any, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))


@dataclass(frozen=True)
class PhysicalAccess:
    """A role-typed read or write that ordinary dataflow does not show.

    A matrix call reads its weight through a port the FX edge does not name,
    and an allocator that only saw the edges would place that weight anywhere.
    Recording the access is what lets a placement be checked against what the
    memory actually accepts.
    """

    operation: Any
    role: TensorRole
    kind: str

    def __post_init__(self):
        object.__setattr__(self, "role", TensorRole(self.role))
        if self.kind not in {"read", "write"}:
            raise ValueError("a physical access is a read or a write")


@dataclass(frozen=True)
class PhysicalTensor:
    """One backing, and everything that shares it."""

    id: PhysicalTensorId
    logical_tensors: Tuple[ValueRef, ...] = ()
    relations: Tuple[StorageRelation, ...] = ()
    components: Tuple[PhysicalTensorComponent, ...] = ()
    backing_size_bytes: Optional[int] = None
    layout: Optional[str] = None
    accesses: Tuple[PhysicalAccess, ...] = ()

    def __post_init__(self):
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.id.name}: component names are not unique")
        if self.backing_size_bytes is not None:
            if self.backing_size_bytes < 0:
                raise ValueError(f"{self.id.name}: backing size is negative")
            for component in self.components:
                if component.byte_size > self.backing_size_bytes:
                    raise ValueError(
                        f"{self.id.name}: component {component.name} needs "
                        f"{component.byte_size} B of a {self.backing_size_bytes}"
                        f" B backing"
                    )

    def component(self, name: str = "primary") -> PhysicalTensorComponent:
        matches = [c for c in self.components if c.name == name]
        if len(matches) != 1:
            raise KeyError(f"{self.id.name}: no component {name!r}")
        return matches[0]


@dataclass(frozen=True)
class StorageTransfer:
    """One movement of a value between planes."""

    logical_tensor: ValueRef
    source: PhysicalTensorId
    destination: PhysicalTensorId
    kind: str = "load"

    def __post_init__(self):
        if self.source.plane == self.destination.plane:
            raise ValueError(
                f"{self.logical_tensor.name}: a transfer crosses planes"
            )


@dataclass(frozen=True)
class StorageInterval:
    """When a backing is live, as half-open positions in execution order.

    Half-open because that is what makes two intervals' disjointness the same
    question as whether their addresses may be reused: a backing that dies at
    position *n* and one born at *n* do not overlap.
    """

    physical_tensor: PhysicalTensorId
    birth: int
    death: int

    def __post_init__(self):
        if self.death < self.birth:
            raise ValueError(
                f"{self.physical_tensor.name}: dies before it is born"
            )

    def overlaps(self, other: "StorageInterval") -> bool:
        return self.birth < other.death and other.birth < self.death


def recorded_value(node):
    """The value a node carries, from ``.value`` or ``meta['val']``.

    Written as an explicit ``is None`` test rather than ``a or b``: the value
    is usually a tensor, and ``or`` asks a multi-element tensor for its truth,
    which raises.  The seductive one-liner works on every scalar and every
    unset node, so it survives a test suite and fails on the first real graph.
    """
    value = getattr(node, "value", None)
    if value is None:
        value = node.meta.get("val")
    return value
