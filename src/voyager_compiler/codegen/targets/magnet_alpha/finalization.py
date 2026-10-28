# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable hand-off from Alpha allocation to flat serialization.

This module deliberately performs no graph transformation, allocation, or
protobuf construction.  It freezes the target ABI occurrences and the answers
earlier stages have already made, then validates that a later serializer can
look them up without chasing storage relations or consulting mutable FX
metadata.

The lookup key is an occurrence, not merely a logical tensor.  That distinction
is required by Alpha channel-shuffle copies: the same FX value may be read from
one backing in one record and another backing in the next record.
"""

import operator
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch.fx import Node

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.storage import (
    PhysicalPayloadRef,
    PhysicalTensorId,
    SerializedTensorRef,
    StoragePlane,
    StorageRelationKind,
    ValueRef,
    ViewTransform,
)
from voyager_compiler.codegen.storage.types import recorded_value
from voyager_compiler.codegen.targets.magnet_alpha.address import (
    alpha_entry_bytes,
    alpha_place,
)
from voyager_compiler.codegen.targets.magnet_alpha.allocation import (
    AlphaPlacement,
)
from voyager_compiler.codegen.targets.magnet_alpha.dram_allocation import (
    AlphaDramPlacement,
    validate_alpha_dram_placements,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import GLOBAL
from voyager_compiler.codegen.targets.magnet_alpha.memory_plan import (
    ALPHA_COPY_PORT_TRANSFORM_KINDS,
    AlphaEgressBoundary,
    AlphaGemmStorageBinding,
    AlphaLogicalViewBinding,
    AlphaNmpInputView,
    AlphaNmpStorageBinding,
    AlphaPayloadUse,
    AlphaPreparedStorage,
    AlphaResidentWriteBinding,
    AlphaSerializedUse,
    AlphaShuffleMovement,
    AlphaShufflePlan,
)


class AlphaFinalizationError(RuntimeError):
    """Final Alpha state is incomplete or internally contradictory."""


_ALPHA_PPU = torch.ops.magnet_alpha.ppu_fixed_relu.default
_ALPHA_PVS = torch.ops.magnet_alpha.pvs_quantize.default


def _require_nonnegative_integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must not be negative")
    return value


def _is_storage_field(value: Any) -> bool:
    """Whether ``value`` can be frozen as a scalar protobuf field.

    Lists and dictionaries are intentionally absent.  Repeated protobuf fields
    are stored as tuples here and materialized only by the serializer.
    """
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        return all(_is_storage_field(item) for item in value)
    return False


@dataclass(frozen=True)
class AlphaRecordSite:
    """One exact location in Alpha's flat collateral.

    ``owner`` is the outer FX node that becomes a serialized operation.
    ``member`` is the exact inner operation carrying the field; it is the owner
    itself for a standalone record and ``None`` for an operation-envelope or
    model-inventory field.  ``occurrence`` distinguishes repeated tensor leaves
    within one normalized argument.
    """

    owner: Node
    member: Optional[Node]
    field: str
    occurrence: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.owner, Node):
            raise TypeError("Alpha record-site owner must be an FX node")
        if self.member is not None and not isinstance(self.member, Node):
            raise TypeError(
                "Alpha record-site member must be an FX node or None"
            )
        if not isinstance(self.field, str) or not self.field:
            raise ValueError(
                "Alpha record-site field must be a non-empty string"
            )
        try:
            occurrence = tuple(self.occurrence)
        except TypeError as error:
            raise TypeError(
                "Alpha record-site occurrence must be iterable"
            ) from error
        try:
            hash(occurrence)
        except TypeError as error:
            raise TypeError(
                "Alpha record-site occurrence must contain hashable values"
            ) from error
        object.__setattr__(self, "occurrence", occurrence)

    @property
    def label(self) -> str:
        member = "@operation" if self.member is None else self.member.name
        occurrence = "" if not self.occurrence else repr(self.occurrence)
        return f"{self.owner.name}/{member}.{self.field}{occurrence}"


@dataclass(frozen=True)
class AlphaPlacedView:
    """One fully encoded logical view into an allocated physical backing.

    This has the placement attributes consumed by Alpha serialization, while
    retaining the physical/component provenance needed for validation.  Offset
    calculation and Alpha address encoding happen before this object is made;
    serialization only copies ``memory_space``, ``address``, and
    ``memory_instance``.
    """

    physical_tensor: PhysicalTensorId
    component: str
    allocation_id: int
    memory_space: int
    start: int
    end: int
    address: int
    memory_instance: str

    def __post_init__(self) -> None:
        if not isinstance(self.physical_tensor, PhysicalTensorId):
            raise TypeError("Alpha placed view needs a physical tensor ID")
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("Alpha placed view needs a component name")
        _require_nonnegative_integer(self.allocation_id, "allocation_id")
        _require_nonnegative_integer(self.memory_space, "memory_space")
        _require_nonnegative_integer(self.start, "start")
        _require_nonnegative_integer(self.end, "end")
        _require_nonnegative_integer(self.address, "address")
        if self.end <= self.start:
            raise ValueError("Alpha placed view must have a positive byte span")
        if (
            not isinstance(self.memory_instance, str)
            or not self.memory_instance
        ):
            raise ValueError("Alpha placed view needs a memory instance")

    @property
    def plane(self) -> str:
        return self.physical_tensor.plane

    @classmethod
    def from_placement(
        cls,
        placement: AlphaPlacement | AlphaDramPlacement,
        *,
        component: str = "primary",
        start: Optional[int] = None,
        end: Optional[int] = None,
        address: Optional[int] = None,
    ) -> "AlphaPlacedView":
        """Freeze a base placement or an explicitly precomputed sub-view.

        Overrides are values, not offsets.  In particular this method never
        adds a byte count to an Alpha encoded address.
        """
        if not isinstance(placement, (AlphaPlacement, AlphaDramPlacement)):
            raise TypeError("expected an Alpha scratchpad or DRAM placement")
        return cls(
            physical_tensor=placement.physical_tensor,
            component=component,
            allocation_id=placement.allocation_id,
            memory_space=placement.memory_space,
            start=placement.start if start is None else start,
            end=placement.end if end is None else end,
            address=placement.address if address is None else address,
            memory_instance=placement.memory_instance,
        )


@dataclass(frozen=True)
class AlphaRecordBinding:
    """One finalized value projected at one exact record site.

    Logical tensors may carry both storage planes.  Physical payload references
    already contain their full-backing placement, serialized artifacts are
    deliberately unplaced, and target-derived scalar fields are recursively
    immutable tuples/scalars.
    """

    site: AlphaRecordSite
    value: Any
    memory: Optional[AlphaPlacedView] = None
    scratchpad: Optional[AlphaPlacedView] = None

    def __post_init__(self) -> None:
        if not isinstance(self.site, AlphaRecordSite):
            raise TypeError("Alpha record binding needs an AlphaRecordSite")
        for label, view in (
            ("memory", self.memory),
            ("scratchpad", self.scratchpad),
        ):
            if view is not None and not isinstance(view, AlphaPlacedView):
                raise TypeError(
                    f"{self.site.label}: {label} view must be an "
                    "AlphaPlacedView"
                )
        if isinstance(self.value, ValueRef):
            if (
                self.memory is not None
                and self.memory.plane != StoragePlane.DRAM.value
            ):
                raise ValueError(
                    f"{self.site.label}: memory view is not in DRAM"
                )
            if (
                self.scratchpad is not None
                and self.scratchpad.plane != StoragePlane.SCRATCHPAD.value
            ):
                raise ValueError(
                    f"{self.site.label}: scratchpad view is not on chip"
                )
            return
        if self.memory is not None or self.scratchpad is not None:
            raise ValueError(
                f"{self.site.label}: only logical tensors carry placed views"
            )
        if isinstance(self.value, (PhysicalPayloadRef, SerializedTensorRef)):
            return
        if not _is_storage_field(self.value):
            raise TypeError(
                f"{self.site.label}: unsupported finalized field value "
                f"{type(self.value).__name__}"
            )

    @property
    def kind(self) -> str:
        if isinstance(self.value, ValueRef):
            return "logical"
        if isinstance(self.value, PhysicalPayloadRef):
            return "physical_payload"
        if isinstance(self.value, SerializedTensorRef):
            return "serialized_artifact"
        return "storage_field"


def _duplicates(values: Iterable[Any]) -> tuple[Any, ...]:
    values = tuple(values)
    seen = set()
    repeated = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return tuple(repeated)


_COPY_PAYLOAD_FIELDS = frozenset(
    {
        "input",
        "output_payload",
        "copy_source_payload",
        "copy_destination_payload",
    }
)
_COPY_SERIALIZED_FIELDS = frozenset({"copy_block_payload"})
_ALPHA_SHUFFLE = torch.ops.magnet_alpha.channel_shuffle.default
_ALPHA_SHUFFLE_COPY = torch.ops.magnet_alpha.channel_shuffle_copy.default


def _validated_alpha_shuffle_copy_refs(
    prepared: AlphaPreparedStorage,
) -> frozenset[ValueRef]:
    """Validate semantic shuffle-copy relations against their frozen plans."""
    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha shuffle-copy validation needs prepared storage")
    plans = tuple(prepared.shuffle_plans)
    if not all(isinstance(plan, AlphaShufflePlan) for plan in plans):
        raise TypeError("shuffle plans must be AlphaShufflePlan values")
    duplicate_shuffles = _duplicates(plan.shuffle for plan in plans)
    if duplicate_shuffles:
        raise AlphaFinalizationError(
            "Alpha shuffle plans repeat owners: "
            + ", ".join(node.name for node in duplicate_shuffles)
        )
    copies = tuple(copy for plan in plans for copy in plan.copies)
    duplicate_copies = _duplicates(copies)
    if duplicate_copies:
        raise AlphaFinalizationError(
            "Alpha shuffle plans repeat copies: "
            + ", ".join(node.name for node in duplicate_copies)
        )
    planned_copy_nodes = frozenset(copies)
    graph_copy_nodes = frozenset(
        node
        for node in prepared.graph.model.graph.nodes
        if node.op == "call_function" and node.target is _ALPHA_SHUFFLE_COPY
    )
    executable_copy_nodes = frozenset(
        node
        for node in prepared.executable_owners
        if node.op == "call_function" and node.target is _ALPHA_SHUFFLE_COPY
    )
    if (
        graph_copy_nodes != planned_copy_nodes
        or executable_copy_nodes != planned_copy_nodes
    ):
        raise AlphaFinalizationError(
            "Alpha shuffle-copy owner coverage differs: "
            f"graph={sorted(node.name for node in graph_copy_nodes)}, "
            f"plans={sorted(node.name for node in planned_copy_nodes)}, "
            f"executable={sorted(node.name for node in executable_copy_nodes)}"
        )

    indexed = set()
    storage_plan = prepared.plan
    plane = StoragePlane.SCRATCHPAD
    for shuffle_plan in plans:
        movements = tuple(shuffle_plan.movements)
        if len(shuffle_plan.copies) != len(movements):
            raise AlphaFinalizationError(
                f"{shuffle_plan.shuffle.name}: copy/movement coverage differs"
            )
        try:
            shuffle_ref = storage_plan.ref(shuffle_plan.shuffle)
            shuffle_outputs = tuple(
                storage_plan.output_refs(shuffle_plan.shuffle)
            )
            shuffle_physical = storage_plan.physical_id(plane, shuffle_ref)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{shuffle_plan.shuffle.name}: frozen shuffle storage is "
                "unavailable"
            ) from error
        if (
            shuffle_outputs != (shuffle_ref,)
            or shuffle_physical != shuffle_plan.primary
        ):
            raise AlphaFinalizationError(
                f"{shuffle_plan.shuffle.name}: frozen shuffle primary changed"
            )

        for producer, successor in zip(
            (shuffle_plan.shuffle, *shuffle_plan.copies[:-1]),
            shuffle_plan.copies,
        ):
            users = tuple(producer.users)
            if len(users) != 1 or users[0] is not successor:
                raise AlphaFinalizationError(
                    f"{producer.name}: an intermediate Alpha shuffle stage "
                    f"must feed only {successor.name}"
                )

        for index, (copy, movement) in enumerate(
            zip(shuffle_plan.copies, movements),
            1,
        ):
            if not isinstance(movement, AlphaShuffleMovement):
                raise TypeError(
                    "shuffle movements must be AlphaShuffleMovement"
                )
            if movement.copy is not copy:
                raise AlphaFinalizationError(
                    f"{copy.name}: copy/movement ordering disagrees"
                )
            try:
                copy_ref = storage_plan.ref(copy)
                copy_outputs = tuple(storage_plan.output_refs(copy))
                relation = storage_plan.relation(plane, copy_ref)
                physical = storage_plan.physical_id(plane, copy_ref)
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaFinalizationError(
                    f"{copy.name}: frozen shuffle-copy storage is unavailable"
                ) from error
            expected_transform = ViewTransform(
                "alpha_shuffle_copy",
                parameters=(("copy_index", index),),
            )
            if (
                copy_outputs != (copy_ref,)
                or relation.kind is not StorageRelationKind.WRITE_INTO
                or relation.source != shuffle_ref
                or relation.physical_source is not None
                or relation.component is not None
                or relation.transform != expected_transform
                or physical != shuffle_plan.primary
            ):
                raise AlphaFinalizationError(
                    f"{copy.name}: frozen semantic shuffle-copy relation "
                    "changed"
                )
            expected_source = (
                shuffle_plan.primary
                if movement.source_output == 0
                else shuffle_plan.auxiliary
                if movement.source_output == 1
                else None
            )
            if (
                expected_source is None
                or movement.source_physical != expected_source
                or movement.destination_physical != shuffle_plan.primary
            ):
                raise AlphaFinalizationError(
                    f"{copy.name}: frozen shuffle-copy endpoints changed"
                )
            indexed.add(copy_ref)

    declared = frozenset(
        logical
        for logical, relation in storage_plan.relations[plane.value].items()
        if relation.transform is not None
        and relation.transform.kind == "alpha_shuffle_copy"
    )
    indexed = frozenset(indexed)
    if declared != indexed:
        missing = sorted(ref.name for ref in indexed - declared)
        orphan = sorted(ref.name for ref in declared - indexed)
        raise AlphaFinalizationError(
            "Alpha semantic shuffle-copy relation coverage differs: "
            f"missing={missing}, orphan={orphan}"
        )
    return indexed


class _AlphaLogicalPlacementResolver:
    """Resolve one semantic tensor to its already-decided Alpha placement.

    Logical views are deliberately separate from shuffle-copy port views.  A
    copy record writes one entry even though its FX result denotes the complete
    tensor after that write; using that one-entry occurrence as a later GEMM
    operand would silently truncate the operand.  General record builders come
    through this resolver, while :class:`_AlphaShuffleBindingBuilder` continues
    to consume its occurrence-specific ``AlphaPayloadUse`` values directly.
    """

    _NORMALIZED_TRANSFORMS = frozenset(
        {"alpha_channel_split", "alpha_shuffle_output"}
    )
    _COPY_PORT_TRANSFORMS = ALPHA_COPY_PORT_TRANSFORM_KINDS

    def __init__(
        self,
        prepared: AlphaPreparedStorage,
        scratchpad_placements: Iterable[AlphaPlacement],
        dram_placements: Iterable[AlphaDramPlacement] = (),
    ) -> None:
        if not isinstance(prepared, AlphaPreparedStorage):
            raise TypeError("Alpha logical placement needs prepared storage")
        scratchpad = tuple(scratchpad_placements)
        dram = tuple(dram_placements)
        if not all(isinstance(item, AlphaPlacement) for item in scratchpad):
            raise TypeError("scratchpad placements must be AlphaPlacement values")
        if not all(isinstance(item, AlphaDramPlacement) for item in dram):
            raise TypeError("DRAM placements must be AlphaDramPlacement values")

        self.prepared = prepared
        self.plan = prepared.plan
        self._scratchpad = self._index_placements(
            scratchpad,
            "scratchpad",
        )
        self._dram = self._index_placements(dram, "DRAM")
        validate_alpha_dram_placements(self.plan, dram)
        self._shuffle_copy_refs = _validated_alpha_shuffle_copy_refs(prepared)
        self._logical_views = self._index_logical_views()
        self._resident_by_ref = self._index_resident_writes()

    @staticmethod
    def _index_placements(placements, label: str):
        physicals = tuple(item.physical_tensor for item in placements)
        duplicates = _duplicates(physicals)
        if duplicates:
            raise AlphaFinalizationError(
                f"Alpha {label} placements repeat: "
                + ", ".join(physical.name for physical in duplicates)
            )
        return {item.physical_tensor: item for item in placements}

    def _index_logical_views(self) -> dict[ValueRef, AlphaLogicalViewBinding]:
        views = tuple(self.prepared.logical_views)
        if not all(isinstance(view, AlphaLogicalViewBinding) for view in views):
            raise TypeError("logical views must be AlphaLogicalViewBinding values")
        logicals = tuple(view.logical_tensor for view in views)
        duplicates = _duplicates(logicals)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha logical views repeat tensors: "
                + ", ".join(logical.name for logical in duplicates)
            )
        indexed = dict(zip(logicals, views))
        for logical, view in indexed.items():
            try:
                logical = self.plan.ref(logical)
                physical = self.plan.physical_id(
                    StoragePlane.SCRATCHPAD,
                    logical,
                )
                self.plan.physical_component(
                    physical,
                    StoragePlane.SCRATCHPAD,
                    view.component,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaFinalizationError(
                    f"{view.logical_tensor.name}: logical view is absent from "
                    "final Alpha storage"
                ) from error
            if physical != view.physical_tensor:
                raise AlphaFinalizationError(
                    f"{logical.name}: logical view names the wrong Alpha backing"
                )
            kinds = self._view_transform_kinds(view)
            if kinds & self._COPY_PORT_TRANSFORMS:
                raise AlphaFinalizationError(
                    f"{logical.name}: shuffle-copy port views belong in "
                    "AlphaPayloadUse"
                )
            try:
                chain = self.plan.trace_storage_chain(
                    logical,
                    StoragePlane.SCRATCHPAD,
                )
                transforms = tuple(
                    relation.transform
                    for item in chain
                    for relation in (
                        self.plan.relation(StoragePlane.SCRATCHPAD, item),
                    )
                    if relation.transform is not None
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaFinalizationError(
                    f"{logical.name}: logical-view relation chain is unavailable"
                ) from error
            if view.transforms != transforms:
                raise AlphaFinalizationError(
                    f"{logical.name}: frozen logical-view transforms are stale"
                )
            if (
                view.byte_offset is not None
                and not kinds & self._NORMALIZED_TRANSFORMS
            ):
                raise AlphaFinalizationError(
                    f"{logical.name}: offset view has no normalized Alpha "
                    "transform"
                )
        return indexed

    def _index_resident_writes(
        self,
    ) -> dict[ValueRef, AlphaResidentWriteBinding]:
        writes = tuple(self.prepared.resident_writes)
        if not all(isinstance(write, AlphaResidentWriteBinding) for write in writes):
            raise TypeError(
                "resident writes must be AlphaResidentWriteBinding values"
            )
        owners = tuple(write.owner for write in writes)
        duplicates = _duplicates(owners)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha resident writes repeat owners: "
                + ", ".join(owner.name for owner in duplicates)
            )
        indexed = {}
        for write in writes:
            for logical in (write.logical_tensor, *write.path):
                previous = indexed.get(logical)
                if previous is not None and previous != write:
                    raise AlphaFinalizationError(
                        f"{logical.name}: logical path belongs to two Alpha "
                        "resident writes"
                    )
                indexed[logical] = write
        return indexed

    def _scratchpad_placement(
        self,
        physical: PhysicalTensorId,
    ) -> AlphaPlacement:
        placement = self._scratchpad.get(physical)
        if placement is None:
            raise AlphaFinalizationError(
                f"{physical.name}: missing Alpha scratchpad placement"
            )
        try:
            request = self.prepared.request_for(physical)
            self.plan.storage_key(physical, StoragePlane.SCRATCHPAD)
            memory = GLOBAL.instance(placement.memory_instance)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}: scratchpad placement is absent from "
                "prepared Alpha storage"
            ) from error
        if placement.request is not request:
            raise AlphaFinalizationError(
                f"{physical.name}: scratchpad placement has a stale request"
            )
        if placement.memory_space != GLOBAL.instances.index(memory):
            raise AlphaFinalizationError(
                f"{physical.name}: scratchpad memory-space index is stale"
            )
        try:
            encoded = alpha_place(
                memory,
                placement.start,
                cluster=placement.cluster,
                operand_bits=placement.operand_bits,
            ).encode()
        except ValueError as error:
            raise AlphaFinalizationError(
                f"{physical.name}: scratchpad placement cannot be encoded"
            ) from error
        if placement.address != encoded:
            raise AlphaFinalizationError(
                f"{physical.name}: scratchpad base address is stale"
            )
        return placement

    def _chain(self, logical: ValueRef, plane: StoragePlane):
        try:
            chain = self.plan.trace_storage_chain(logical, plane)
            physical = self.plan.physical_id(plane, logical)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{logical.name}: cannot trace finalized {plane.value} storage"
            ) from error
        for item in chain:
            try:
                item_physical = self.plan.physical_id(plane, item)
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaFinalizationError(
                    f"{item.name}: relation chain left finalized storage"
                ) from error
            if item_physical != physical:
                raise AlphaFinalizationError(
                    f"{logical.name}: relation chain changes physical backing"
                )
        return chain, physical

    @staticmethod
    def _positions(plan, logical: ValueRef, label: str) -> int:
        try:
            shape = tuple(int(value) for value in plan.logical_tensors[logical].shape)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{logical.name}: {label} shape is unavailable"
            ) from error
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise AlphaFinalizationError(
                f"{logical.name}: {label} must be positive rank-four NCHW"
            )
        return shape[0] * shape[2] * shape[3]

    def _entry_geometry(
        self,
        physical: PhysicalTensorId,
        placement: AlphaPlacement,
    ) -> tuple[int, int]:
        try:
            entry_bytes = alpha_entry_bytes(
                GLOBAL.instance(placement.memory_instance),
                placement.cluster,
            )
            size = self.plan.backing_size_bytes(
                physical,
                StoragePlane.SCRATCHPAD,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}: Alpha backing geometry is unavailable"
            ) from error
        entries, remainder = divmod(size, entry_bytes)
        if remainder or entries <= 0:
            raise AlphaFinalizationError(
                f"{physical.name}: Alpha backing is not entry aligned"
            )
        if placement.end - placement.start < size:
            raise AlphaFinalizationError(
                f"{physical.name}: placement is smaller than its backing"
            )
        return entries, entry_bytes

    def _encoded_view(
        self,
        physical: PhysicalTensorId,
        placement: AlphaPlacement,
        *,
        component: str,
        start: int,
        end: int,
    ) -> AlphaPlacedView:
        backing_entries, entry_bytes = self._entry_geometry(
            physical,
            placement,
        )
        backing_end = placement.start + backing_entries * entry_bytes
        if start < placement.start or end <= start or end > backing_end:
            raise AlphaFinalizationError(
                f"{physical.name}.{component}: logical view exceeds its backing"
            )
        try:
            self.plan.physical_component(
                physical,
                StoragePlane.SCRATCHPAD,
                component,
            )
            address = alpha_place(
                GLOBAL.instance(placement.memory_instance),
                start,
                cluster=placement.cluster,
                operand_bits=placement.operand_bits,
            ).encode()
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}.{component}: logical view cannot be encoded"
            ) from error
        return AlphaPlacedView.from_placement(
            placement,
            component=component,
            start=start,
            end=end,
            address=address,
        )

    def view_for_payload_use(self, use: AlphaPayloadUse) -> AlphaPlacedView:
        """Place one storage-owned, occurrence-specific scratchpad view."""
        if not isinstance(use, AlphaPayloadUse):
            raise TypeError("Alpha payload view needs an AlphaPayloadUse")
        if use.physical_tensor.plane != StoragePlane.SCRATCHPAD.value:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload use is not in "
                "scratchpad"
            )
        try:
            logical = self.plan.ref(use.logical_tensor)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload use has no "
                "logical Alpha tensor"
            ) from error
        if logical != use.logical_tensor:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload logical tensor "
                "is not canonical"
            )
        entry_offset = _require_nonnegative_integer(
            use.entry_offset,
            f"{use.operation.name}.{use.field} entry offset",
        )
        if use.span_entries is None:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload use has no entry "
                "span"
            )
        span_entries = _require_nonnegative_integer(
            use.span_entries,
            f"{use.operation.name}.{use.field} entry span",
        )
        if span_entries == 0:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload entry span must "
                "be positive"
            )
        if use.row_pitch_entries is None:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload use has no row "
                "pitch"
            )
        row_pitch_entries = _require_nonnegative_integer(
            use.row_pitch_entries,
            f"{use.operation.name}.{use.field} row pitch",
        )
        if row_pitch_entries == 0:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload row pitch must "
                "be positive"
            )
        placement = self._scratchpad_placement(use.physical_tensor)
        backing_entries, entry_bytes = self._entry_geometry(
            use.physical_tensor,
            placement,
        )
        if entry_offset + span_entries > backing_entries:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: payload view exceeds its "
                "physical backing"
            )
        start = placement.start + entry_offset * entry_bytes
        return self._encoded_view(
            use.physical_tensor,
            placement,
            component=use.component,
            start=start,
            end=start + span_entries * entry_bytes,
        )

    @staticmethod
    def _view_transform_kinds(view: AlphaLogicalViewBinding) -> frozenset[str]:
        return frozenset(transform.kind for transform in view.transforms)

    def _resident_match(
        self,
        chain: tuple[ValueRef, ...],
    ) -> Optional[AlphaResidentWriteBinding]:
        matches = []
        for logical in chain:
            match = self._resident_by_ref.get(logical)
            if match is not None and match not in matches:
                matches.append(match)
        if len(matches) > 1:
            raise AlphaFinalizationError(
                f"{chain[0].name}: relation chain crosses Alpha resident slots"
            )
        return matches[0] if matches else None

    def _validate_resident(
        self,
        write: AlphaResidentWriteBinding,
        physical: PhysicalTensorId,
    ) -> None:
        if write.physical_tensor != physical:
            raise AlphaFinalizationError(
                f"{write.logical_tensor.name}: resident write names the wrong "
                "physical backing"
            )
        try:
            relation = self.plan.relation(
                StoragePlane.SCRATCHPAD,
                write.logical_tensor,
            )
            backing_physical = self.plan.physical_id(
                StoragePlane.SCRATCHPAD,
                write.backing,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{write.logical_tensor.name}: resident relation is unavailable"
            ) from error
        expected_kind = (
            StorageRelationKind.MATERIALIZE
            if write.logical_tensor == write.backing
            else StorageRelationKind.WRITE_INTO
        )
        expected_source = (
            None if write.logical_tensor == write.backing else write.backing
        )
        parameters = {
            "channel_offset": write.channel_offset,
            "slot_index": write.slot_index,
            "slot_channels": write.slot_channels,
            "resident_channels": write.resident_channels,
        }
        if write.concat_channel_index is not None:
            parameters["concat_channel_index"] = write.concat_channel_index
        if (
            backing_physical != physical
            or relation.kind is not expected_kind
            or relation.source != expected_source
            or relation.transform is None
            or relation.transform.kind != "alpha_resident_write"
            or dict(relation.transform.parameters) != parameters
        ):
            raise AlphaFinalizationError(
                f"{write.logical_tensor.name}: frozen resident relation is stale"
            )

    def _resident_position(
        self,
        write: AlphaResidentWriteBinding,
        physical: PhysicalTensorId,
        placement: AlphaPlacement,
    ):
        self._validate_resident(write, physical)
        _backing_entries, entry_bytes = self._entry_geometry(
            physical,
            placement,
        )
        entry_bits = entry_bytes * 8
        channels_per_entry, remainder = divmod(
            entry_bits,
            placement.operand_bits,
        )
        if remainder or channels_per_entry <= 0:
            raise AlphaFinalizationError(
                f"{physical.name}: resident channel geometry is not integral"
            )
        values = (
            write.channel_offset,
            write.slot_channels,
            write.resident_channels,
        )
        converted = tuple(divmod(value, channels_per_entry) for value in values)
        if any(remainder for _entries, remainder in converted):
            raise AlphaFinalizationError(
                f"{write.logical_tensor.name}: resident channel geometry is not "
                "entry aligned"
            )
        entry_offset, active_entries, row_pitch = tuple(
            entries for entries, _remainder in converted
        )
        if active_entries <= 0 or row_pitch < entry_offset + active_entries:
            raise AlphaFinalizationError(
                f"{write.logical_tensor.name}: resident entry window is invalid"
            )
        positions = self._positions(
            self.plan,
            write.logical_tensor,
            "resident writer",
        )
        footprint = (positions - 1) * row_pitch + active_entries
        start = placement.start + entry_offset * entry_bytes
        view = self._encoded_view(
            physical,
            placement,
            component="primary",
            start=start,
            end=start + footprint * entry_bytes,
        )
        return view, entry_offset, active_entries, row_pitch

    def _normalized_position(
        self,
        view: AlphaLogicalViewBinding,
        physical: PhysicalTensorId,
        placement: AlphaPlacement,
    ):
        if (
            view.physical_tensor != physical
            or view.component != "primary"
            or view.byte_offset is None
            or view.span_entries is None
            or view.row_pitch_entries is None
        ):
            raise AlphaFinalizationError(
                f"{view.logical_tensor.name}: normalized Alpha view is incomplete"
            )
        _backing_entries, entry_bytes = self._entry_geometry(
            physical,
            placement,
        )
        try:
            logical_size = self.plan.logical_size_bytes(view.logical_tensor)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{view.logical_tensor.name}: normalized logical size is "
                "unavailable"
            ) from error
        if view.span_entries * entry_bytes != logical_size:
            raise AlphaFinalizationError(
                f"{view.logical_tensor.name}: normalized Alpha span differs "
                "from its logical size"
            )
        if view.byte_offset != view.entry_offset * entry_bytes:
            raise AlphaFinalizationError(
                f"{view.logical_tensor.name}: normalized Alpha byte and entry "
                "offsets disagree"
            )
        positions = self._positions(
            self.plan,
            view.logical_tensor,
            "normalized logical view",
        )
        active_entries, remainder = divmod(view.span_entries, positions)
        if remainder or active_entries <= 0:
            raise AlphaFinalizationError(
                f"{view.logical_tensor.name}: normalized Alpha span is not a "
                "whole NCHW view"
            )
        row_pitch = view.row_pitch_entries
        if row_pitch < view.entry_offset + active_entries:
            raise AlphaFinalizationError(
                f"{view.logical_tensor.name}: normalized Alpha view exceeds its "
                "row pitch"
            )
        footprint = (positions - 1) * row_pitch + active_entries
        start = placement.start + view.byte_offset
        placed = self._encoded_view(
            physical,
            placement,
            component="primary",
            start=start,
            end=start + footprint * entry_bytes,
        )
        return placed, view.entry_offset, active_entries, row_pitch

    def _validate_scratchpad_chain(
        self,
        chain: tuple[ValueRef, ...],
        *,
        mode: str,
        has_resident: bool,
    ) -> None:
        for logical in chain:
            relation = self.plan.relation(StoragePlane.SCRATCHPAD, logical)
            transform = relation.transform
            kind = None if transform is None else transform.kind
            if relation.kind is StorageRelationKind.MATERIALIZE:
                if kind is None:
                    continue
                if kind == "alpha_resident_write" and has_resident:
                    continue
            elif relation.kind is StorageRelationKind.COPY:
                if kind is None:
                    continue
            elif relation.kind is StorageRelationKind.ALIAS:
                if kind is None:
                    continue
            elif relation.kind is StorageRelationKind.COMPONENT:
                if relation.component == "fine_scale" and relation.source is not None:
                    continue
            elif relation.kind is StorageRelationKind.VIEW:
                if (
                    kind in {"identity_view", "alpha_address_only"}
                    and relation.source is not None
                    and relation.component is None
                ):
                    # These fork-authored markers change neither the backing
                    # nor its origin.  Any view with real geometry remains a
                    # separate, explicitly normalized case below.
                    continue
                if kind in self._NORMALIZED_TRANSFORMS and mode == "normalized":
                    continue
                if kind == "alpha_resident_view" and has_resident:
                    continue
            elif relation.kind is StorageRelationKind.WRITE_INTO:
                if kind == "alpha_resident_write" and has_resident:
                    continue
                if (
                    kind == "alpha_shuffle_copy"
                    and logical in self._shuffle_copy_refs
                    and mode in {"full", "resident", "normalized"}
                ):
                    # The position is supplied by a separate frozen resident or
                    # split view.  The copy relation contributes identity only.
                    continue
            raise AlphaFinalizationError(
                f"{logical.name}: unsupported {relation.kind.value} Alpha "
                f"placement transform {kind!r}"
            )

    def _position_for(
        self,
        logical: ValueRef,
        physical: PhysicalTensorId,
        placement: AlphaPlacement,
    ) -> tuple[AlphaPlacedView, str, bool]:
        chain, chained_physical = self._chain(logical, StoragePlane.SCRATCHPAD)
        if chained_physical != physical:
            raise AlphaFinalizationError(
                f"{logical.name}: logical carrier changed Alpha backing"
            )
        normalized = []
        for item in chain:
            view = self._logical_views.get(item)
            if view is None:
                continue
            if view.byte_offset is not None:
                normalized.append(view)
        resident = self._resident_match(chain)

        if normalized:
            selected = normalized[0]
            placed, offset, active, pitch = self._normalized_position(
                selected,
                physical,
                placement,
            )
            if resident is not None:
                _resident_view, resident_offset, resident_active, resident_pitch = (
                    self._resident_position(resident, physical, placement)
                )
                if (
                    pitch != resident_pitch
                    or offset < resident_offset
                    or offset + active > resident_offset + resident_active
                ):
                    raise AlphaFinalizationError(
                        f"{logical.name}: normalized view disagrees with its "
                        "resident slot"
                    )
            self._validate_scratchpad_chain(
                chain,
                mode="normalized",
                has_resident=resident is not None,
            )
            return placed, "normalized", resident is not None

        if resident is not None:
            placed, _offset, _active, _pitch = self._resident_position(
                resident,
                physical,
                placement,
            )
            self._validate_scratchpad_chain(
                chain,
                mode="resident",
                has_resident=True,
            )
            return placed, "resident", True

        self._validate_scratchpad_chain(
            chain,
            mode="full",
            has_resident=False,
        )
        self.plan.physical_component(
            physical,
            StoragePlane.SCRATCHPAD,
            "primary",
        )
        return (
            AlphaPlacedView.from_placement(
                placement,
                component="primary",
            ),
            "full",
            False,
        )

    def _scratchpad_view(self, logical: ValueRef) -> AlphaPlacedView:
        chain, physical = self._chain(logical, StoragePlane.SCRATCHPAD)
        placement = self._scratchpad_placement(physical)
        component_relations = tuple(
            (index, item, self.plan.relation(StoragePlane.SCRATCHPAD, item))
            for index, item in enumerate(chain)
            if self.plan.relation(StoragePlane.SCRATCHPAD, item).kind
            is StorageRelationKind.COMPONENT
        )
        if len(component_relations) > 1:
            raise AlphaFinalizationError(
                f"{logical.name}: nested Alpha physical components are unsupported"
            )
        if not component_relations:
            if any(
                view.component != "primary"
                for item in chain
                for view in (self._logical_views.get(item),)
                if view is not None
            ):
                raise AlphaFinalizationError(
                    f"{logical.name}: component view has no component relation"
                )
            return self._position_for(logical, physical, placement)[0]

        index, component_ref, relation = component_relations[0]
        if relation.component != "fine_scale" or relation.source is None:
            raise AlphaFinalizationError(
                f"{component_ref.name}: unsupported Alpha physical component"
            )
        confirmations = tuple(
            view
            for item in chain[: index + 1]
            for view in (self._logical_views.get(item),)
            if view is not None and view.component == "fine_scale"
        )
        if len(confirmations) != 1 or confirmations[0].byte_offset is not None:
            raise AlphaFinalizationError(
                f"{component_ref.name}: embedded fine scale has no exact frozen "
                "component view"
            )
        confirmation = confirmations[0]
        if confirmation.physical_tensor != physical:
            raise AlphaFinalizationError(
                f"{component_ref.name}: fine-scale view changed physical backing"
            )
        carrier = relation.source
        placed, mode, has_resident = self._position_for(
            carrier,
            physical,
            placement,
        )
        self._validate_scratchpad_chain(
            chain,
            mode=mode,
            has_resident=has_resident,
        )
        self.plan.physical_component(
            physical,
            StoragePlane.SCRATCHPAD,
            "fine_scale",
        )
        return AlphaPlacedView.from_placement(
            placement,
            component="fine_scale",
            start=placed.start,
            end=placed.end,
            address=placed.address,
        )

    def _dram_view(self, logical: ValueRef) -> Optional[AlphaPlacedView]:
        chain, physical = self._chain(logical, StoragePlane.DRAM)
        placement = self._dram.get(physical)
        if placement is None:
            return None
        transfers = tuple(
            transfer
            for transfer in self.plan.transfers
            if transfer.logical_tensor == logical
            and transfer.source.plane == StoragePlane.DRAM.value
            and transfer.destination.plane == StoragePlane.SCRATCHPAD.value
        )
        if not transfers:
            # DRAM allocation is per physical root, but a record operand is a
            # logical occurrence.  An alias of a loaded root is not itself a
            # load and therefore must not inherit that root's DRAM address.
            return None
        if len(transfers) != 1:
            raise AlphaFinalizationError(
                f"{logical.name}: finalized Alpha storage repeats its exact "
                "DRAM transfer"
            )
        try:
            scratchpad_physical = self.plan.physical_id(
                StoragePlane.SCRATCHPAD,
                logical,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{logical.name}: DRAM transfer has no scratchpad destination"
            ) from error
        transfer = transfers[0]
        if (
            transfer.kind != "load"
            or transfer.source != physical
            or transfer.destination != scratchpad_physical
        ):
            raise AlphaFinalizationError(
                f"{logical.name}: finalized Alpha DRAM load names the wrong "
                "source or scratchpad backing"
            )
        for item in chain:
            relation = self.plan.relation(StoragePlane.DRAM, item)
            transform = relation.transform
            kind = None if transform is None else transform.kind
            valid = (
                relation.kind in {
                    StorageRelationKind.MATERIALIZE,
                    StorageRelationKind.COPY,
                }
                and kind is None
            ) or (
                relation.kind is StorageRelationKind.ALIAS and kind is None
            ) or (
                relation.kind is StorageRelationKind.VIEW
                and kind in {"identity_view", "alpha_address_only"}
            )
            if not valid:
                raise AlphaFinalizationError(
                    f"{item.name}: unsupported DRAM {relation.kind.value} "
                    f"transform {kind!r}"
                )
        try:
            self.plan.physical_component(
                physical,
                StoragePlane.DRAM,
                "primary",
            )
            size = self.plan.backing_size_bytes(physical, StoragePlane.DRAM)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}: DRAM placement is absent from final storage"
            ) from error
        if placement.end - placement.start < size:
            raise AlphaFinalizationError(
                f"{physical.name}: DRAM placement is smaller than its backing"
            )
        return AlphaPlacedView.from_placement(placement, component="primary")

    def view_for(self, value, plane) -> Optional[AlphaPlacedView]:
        """Return ``value``'s exact view in ``plane``, if that plane is placed."""
        try:
            logical = self.plan.ref(value)
            plane = StoragePlane(plane)
        except (KeyError, TypeError, ValueError) as error:
            name = getattr(value, "name", repr(value))
            raise AlphaFinalizationError(
                f"{name}: invalid Alpha logical placement query"
            ) from error
        if plane is StoragePlane.SCRATCHPAD:
            return self._scratchpad_view(logical)
        if plane is StoragePlane.DRAM:
            return self._dram_view(logical)
        raise AlphaFinalizationError(
            f"{logical.name}: Alpha has no placement resolver for {plane.value}"
        )


_ALPHA_GETITEM = operator.getitem
_ALPHA_CAT = torch.ops.aten.cat.default
_ALPHA_CHUNK = torch.ops.aten.chunk.default
_ALPHA_CONTIGUOUS = torch.ops.aten.contiguous.default
_ALPHA_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_ALPHA_PERMUTE = torch.ops.aten.permute.default
_ALPHA_RESHAPE = torch.ops.aten.reshape.default
_ALPHA_SELECT = torch.ops.aten.select.int
_ALPHA_SLICE = torch.ops.aten.slice.Tensor
_ALPHA_STACK = torch.ops.aten.stack.default
_ALPHA_VIEW = torch.ops.aten.view.default

# This is the measured Alpha flat-record vocabulary.  It is repeated here
# deliberately rather than imported from ``serialization``: finalization is
# an input to that module, not a client of it.  A frozen egress role below
# takes precedence over this set, because the terminal permute/contiguous pair
# are real host-layout operations even though those targets can also be
# address-only elsewhere in the graph.
_ALPHA_ADDRESS_ONLY_TARGETS = frozenset(
    {
        _ALPHA_GETITEM,
        _ALPHA_CAT,
        _ALPHA_CHUNK,
        _ALPHA_CONTIGUOUS,
        _ALPHA_PERMUTE,
        _ALPHA_RESHAPE,
        _ALPHA_SELECT,
        _ALPHA_SLICE,
        _ALPHA_STACK,
        _ALPHA_VIEW,
    }
)

_STRUCTURAL_SCRATCHPAD = "scratchpad"
_STRUCTURAL_UNPLACED = "unplaced"
_STRUCTURAL_SCALAR = "scalar"


@dataclass(frozen=True)
class _AlphaStructuralRequirement:
    """One canonical tensor/scalar occurrence in a non-executable record."""

    site: AlphaRecordSite
    value: Any
    placement: str
    address_only: bool

    def __post_init__(self) -> None:
        if self.placement not in {
            _STRUCTURAL_SCRATCHPAD,
            _STRUCTURAL_UNPLACED,
            _STRUCTURAL_SCALAR,
        }:
            raise ValueError(
                f"{self.site.label}: unknown structural placement contract"
            )
        if self.placement == _STRUCTURAL_SCALAR:
            if not _is_storage_field(self.value):
                raise TypeError(
                    f"{self.site.label}: structural scalar is not immutable"
                )
        elif not isinstance(self.value, ValueRef):
            raise TypeError(
                f"{self.site.label}: structural tensor needs a ValueRef"
            )


def _alpha_single_output_ref(
    prepared: AlphaPreparedStorage,
    node: Node,
    *,
    label: str,
) -> ValueRef:
    outputs = tuple(prepared.graph.output_refs(node))
    if len(outputs) != 1:
        raise AlphaFinalizationError(
            f"{node.name}: {label} must name one tensor result, got "
            f"{len(outputs)}"
        )
    return outputs[0]


def _alpha_tensor_argument_ref(
    prepared: AlphaPreparedStorage,
    value,
    *,
    owner: Node,
    field: str,
) -> ValueRef:
    if not isinstance(value, Node):
        raise AlphaFinalizationError(
            f"{owner.name}.{field}: Alpha structural operand is not an FX node"
        )
    return _alpha_single_output_ref(
        prepared,
        value,
        label=f"structural operand {field!r}",
    )


def _alpha_getitem_argument_ref(
    prepared: AlphaPreparedStorage,
    node: Node,
) -> ValueRef:
    source = get_arg_value(node, 0, "input")
    index = get_arg_value(node, 1, "index")
    value = recorded_value(source) if isinstance(source, Node) else None
    if (
        not isinstance(source, Node)
        or isinstance(index, bool)
        or not isinstance(index, int)
        or not isinstance(value, (tuple, list))
        or not value
    ):
        raise AlphaFinalizationError(
            f"{node.name}: Alpha getitem NOP needs a tensor tuple and integer "
            "index"
        )
    normalized = index % len(value)
    try:
        return prepared.plan.ref(ValueRef(source, (normalized,)))
    except (KeyError, TypeError, ValueError) as error:
        raise AlphaFinalizationError(
            f"{node.name}: selected getitem tensor component is absent from "
            "final storage"
        ) from error


def _alpha_record_output_requirements(
    prepared: AlphaPreparedStorage,
    node: Node,
    *,
    address_only: bool,
) -> tuple[_AlphaStructuralRequirement, ...]:
    outputs = tuple(prepared.graph.output_refs(node))
    value = recorded_value(node)
    if isinstance(value, (tuple, list)):
        if len(outputs) != len(value) or any(
            output.path != (index,)
            for index, output in enumerate(outputs)
        ):
            raise AlphaFinalizationError(
                f"{node.name}: structural tuple outputs are not flat tensor "
                "components"
            )
        return tuple(
            _AlphaStructuralRequirement(
                site=AlphaRecordSite(
                    owner=node,
                    member=None,
                    field="outputs",
                    occurrence=("tensor_list", index),
                ),
                value=output,
                placement=_STRUCTURAL_SCRATCHPAD,
                address_only=address_only,
            )
            for index, output in enumerate(outputs)
        )
    if len(outputs) != 1 or outputs[0].path:
        raise AlphaFinalizationError(
            f"{node.name}: structural record needs one tensor output"
        )
    return (
        _AlphaStructuralRequirement(
            site=AlphaRecordSite(
                owner=node,
                member=None,
                field="output",
            ),
            value=outputs[0],
            placement=_STRUCTURAL_SCRATCHPAD,
            address_only=address_only,
        ),
    )


def _alpha_address_only_requirements(
    prepared: AlphaPreparedStorage,
    node: Node,
) -> tuple[_AlphaStructuralRequirement, ...]:
    requirements = []
    if node.target is _ALPHA_GETITEM:
        requirements.append(
            _AlphaStructuralRequirement(
                site=AlphaRecordSite(
                    owner=node,
                    member=node,
                    field="args",
                    occurrence=(0,),
                ),
                value=_alpha_getitem_argument_ref(prepared, node),
                placement=_STRUCTURAL_SCRATCHPAD,
                address_only=True,
            )
        )
    elif node.target in {_ALPHA_CAT, _ALPHA_STACK}:
        tensors = get_arg_value(node, 0, "tensors")
        if not isinstance(tensors, (tuple, list)) or not tensors:
            raise AlphaFinalizationError(
                f"{node.name}: Alpha {node.target} NOP needs a non-empty "
                "tensor list"
            )
        for index, tensor in enumerate(tensors):
            requirements.append(
                _AlphaStructuralRequirement(
                    site=AlphaRecordSite(
                        owner=node,
                        member=node,
                        field="tensors",
                        occurrence=("tensor_list", index),
                    ),
                    value=_alpha_tensor_argument_ref(
                        prepared,
                        tensor,
                        owner=node,
                        field=f"tensors[{index}]",
                    ),
                    placement=_STRUCTURAL_SCRATCHPAD,
                    address_only=True,
                )
            )
    else:
        source = get_arg_value(node, 0, "input")
        requirements.append(
            _AlphaStructuralRequirement(
                site=AlphaRecordSite(
                    owner=node,
                    member=node,
                    field="input",
                ),
                value=_alpha_tensor_argument_ref(
                    prepared,
                    source,
                    owner=node,
                    field="input",
                ),
                placement=_STRUCTURAL_SCRATCHPAD,
                address_only=True,
            )
        )
    requirements.extend(
        _alpha_record_output_requirements(
            prepared,
            node,
            address_only=True,
        )
    )
    return tuple(requirements)


def _alpha_l2_requirement(
    node: Node,
    expected: tuple[int, ...],
) -> _AlphaStructuralRequirement:
    declared = node.meta.get("l2_tiling")
    if declared is not None:
        try:
            declared = tuple(declared)
        except TypeError as error:
            raise AlphaFinalizationError(
                f"{node.name}: terminal Alpha l2_tiling is not a sequence"
            ) from error
        if declared != expected or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in declared
        ):
            raise AlphaFinalizationError(
                f"{node.name}: terminal Alpha l2_tiling must be {expected}"
            )
    return _AlphaStructuralRequirement(
        site=AlphaRecordSite(
            owner=node,
            member=node,
            field="l2_tiling",
        ),
        # Alpha's frozen end-to-end egress is one full resident tensor, so
        # these are target decisions rather than a serializer-side tiling
        # inference.  Preserve an identical earlier answer when present and
        # fail above if some pass claimed a different one.
        value=expected,
        placement=_STRUCTURAL_SCALAR,
        address_only=False,
    )


def _alpha_boundary_record_requirements(
    prepared: AlphaPreparedStorage,
    node: Node,
    role: str,
) -> tuple[_AlphaStructuralRequirement, ...]:
    source = _alpha_tensor_argument_ref(
        prepared,
        get_arg_value(node, 0, "input"),
        owner=node,
        field="input",
    )
    requirements = [
        _AlphaStructuralRequirement(
            site=AlphaRecordSite(node, node, "input"),
            value=source,
            placement=_STRUCTURAL_SCRATCHPAD,
            address_only=False,
        )
    ]
    if role == "dequantize":
        scale = _alpha_tensor_argument_ref(
            prepared,
            get_arg_value(node, 1, "scale"),
            owner=node,
            field="scale",
        )
        dtype = prepared.plan.logical_tensors[scale].dtype
        if str(dtype).split(".")[-1] != "float32":
            raise AlphaFinalizationError(
                f"{node.name}.scale: terminal Alpha scale must be float32"
            )
        requirements.append(
            _AlphaStructuralRequirement(
                site=AlphaRecordSite(node, node, "scale"),
                value=scale,
                placement=_STRUCTURAL_UNPLACED,
                address_only=False,
            )
        )
        requirements.append(_alpha_l2_requirement(node, (1, 1, 1, 1)))
    elif role == "permute":
        requirements.append(
            _alpha_l2_requirement(node, (1, 1, 1, 1, 1))
        )
    elif role != "contiguous":
        raise AlphaFinalizationError(
            f"{node.name}: unknown frozen Alpha boundary role {role!r}"
        )
    requirements.extend(
        _alpha_record_output_requirements(
            prepared,
            node,
            address_only=False,
        )
    )
    return tuple(requirements)


def _alpha_egress_roles(
    prepared: AlphaPreparedStorage,
) -> dict[Node, str]:
    positions = {
        node: index
        for index, node in enumerate(prepared.execution_order.nodes)
    }
    executable = set(prepared.executable_owners)
    roles = {}
    for boundary in prepared.egress_boundaries:
        if not isinstance(boundary, AlphaEgressBoundary):
            raise TypeError("Alpha egress boundaries must be frozen records")
        if (
            len(boundary.shared_views) != 1
            or len(boundary.host_layouts) != 2
            or boundary.dequantize.target is not _ALPHA_DEQUANTIZE
            or boundary.padding_slice.target is not _ALPHA_SLICE
            or boundary.shared_views[0].target is not _ALPHA_VIEW
            or boundary.host_layouts[0].target is not _ALPHA_PERMUTE
            or boundary.host_layouts[1].target is not _ALPHA_CONTIGUOUS
            or boundary.output.op != "output"
        ):
            raise AlphaFinalizationError(
                f"{boundary.dequantize.name}: frozen Alpha egress is not the "
                "canonical YOLO suffix"
            )
        if not boundary.owners <= executable:
            raise AlphaFinalizationError(
                f"{boundary.dequantize.name}: frozen Alpha egress names an "
                "unknown executable owner"
            )
        source = _alpha_tensor_argument_ref(
            prepared,
            get_arg_value(boundary.dequantize, 0, "input"),
            owner=boundary.dequantize,
            field="input",
        )
        if source != boundary.source:
            raise AlphaFinalizationError(
                f"{boundary.dequantize.name}: frozen Alpha egress source "
                "changed"
            )
        path = (
            boundary.source.node,
            *boundary.storage_path,
            boundary.output,
        )
        try:
            path_positions = tuple(positions[item] for item in path)
        except KeyError as error:
            missing = error.args[0]
            raise AlphaFinalizationError(
                f"{missing.name}: frozen Alpha egress is outside the selected "
                "execution order"
            ) from error
        if any(
            left >= right
            for left, right in zip(path_positions, path_positions[1:])
        ):
            raise AlphaFinalizationError(
                f"{boundary.dequantize.name}: frozen Alpha egress order changed"
            )
        boundary_roles = {
            boundary.dequantize: "dequantize",
            boundary.padding_slice: "address_only",
            boundary.shared_views[0]: "address_only",
            boundary.host_layouts[0]: "permute",
            boundary.host_layouts[1]: "contiguous",
        }
        repeated = set(roles) & set(boundary_roles)
        if repeated:
            raise AlphaFinalizationError(
                "Alpha egress boundaries overlap at "
                + ", ".join(sorted(node.name for node in repeated))
            )
        roles.update(boundary_roles)
    return roles


def _alpha_structural_requirements(
    prepared: AlphaPreparedStorage,
) -> tuple[_AlphaStructuralRequirement, ...]:
    """Derive the complete canonical non-executable record inventory once."""
    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha structural bindings need prepared storage")
    egress_roles = _alpha_egress_roles(prepared)
    executable = set(prepared.executable_owners)
    requirements = []
    seen_egress = set()
    for node in prepared.execution_order.nodes:
        role = egress_roles.get(node)
        if role is not None:
            seen_egress.add(node)
            if role == "address_only":
                requirements.extend(
                    _alpha_address_only_requirements(prepared, node)
                )
            else:
                requirements.extend(
                    _alpha_boundary_record_requirements(
                        prepared,
                        node,
                        role,
                    )
                )
            continue
        if node in executable or node.op in {"placeholder", "get_attr", "output"}:
            continue
        if (
            node.op == "call_function"
            and node.target in _ALPHA_ADDRESS_ONLY_TARGETS
        ):
            requirements.extend(
                _alpha_address_only_requirements(prepared, node)
            )
            continue
        if node.op in {"call_function", "call_module", "call_method"}:
            raise AlphaFinalizationError(
                f"{node.name}: selected Alpha execution order contains an "
                "unclassified non-executable operation"
            )
    missing_egress = set(egress_roles) - seen_egress
    if missing_egress:
        raise AlphaFinalizationError(
            "frozen Alpha egress nodes are missing from execution order: "
            + ", ".join(sorted(node.name for node in missing_egress))
        )
    sites = tuple(requirement.site for requirement in requirements)
    duplicates = _duplicates(sites)
    if duplicates:
        raise AlphaFinalizationError(
            "canonical Alpha structural sites repeat: "
            + ", ".join(site.label for site in duplicates)
        )
    return tuple(requirements)


def _validate_alpha_structural_binding_inventory(
    prepared: AlphaPreparedStorage,
    bindings: Iterable[AlphaRecordBinding],
) -> None:
    """Require every structural occurrence once, with its frozen placement."""
    requirements = _alpha_structural_requirements(prepared)
    required = {requirement.site: requirement for requirement in requirements}
    structural_owners = {site.owner for site in required}
    actual = {
        binding.site: binding
        for binding in bindings
        if binding.site.owner in structural_owners
    }
    missing = tuple(site for site in required if site not in actual)
    unexpected = tuple(site for site in actual if site not in required)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing " + ", ".join(site.label for site in missing)
            )
        if unexpected:
            details.append(
                "unexpected " + ", ".join(site.label for site in unexpected)
            )
        raise AlphaFinalizationError(
            "Alpha structural binding inventory is incomplete: "
            + "; ".join(details)
        )
    for site, requirement in required.items():
        binding = actual[site]
        if binding.value != requirement.value:
            raise AlphaFinalizationError(
                f"{site.label}: structural binding value changed"
            )
        if requirement.placement == _STRUCTURAL_SCRATCHPAD:
            valid = binding.memory is None and binding.scratchpad is not None
        elif requirement.placement == _STRUCTURAL_UNPLACED:
            valid = binding.memory is None and binding.scratchpad is None
        else:
            valid = binding.memory is None and binding.scratchpad is None
        if not valid:
            raise AlphaFinalizationError(
                f"{site.label}: structural binding placement changed"
            )


def _alpha_zero_copy_shuffle_split_plan(
    prepared: AlphaPreparedStorage,
    resolver: _AlphaLogicalPlacementResolver,
    node: Node,
) -> Optional[AlphaShufflePlan]:
    """Return the exact fork-proven two-backing address-only split, if any."""
    if node.target is not _ALPHA_CHUNK:
        return None
    source = get_arg_value(node, 0, "input")
    if not isinstance(source, Node):
        return None
    shuffle_plan = prepared.shuffle_for(source)
    if shuffle_plan is None or shuffle_plan.copies or shuffle_plan.movements:
        return None

    storage_plan = prepared.plan
    plane = StoragePlane.SCRATCHPAD
    try:
        source_ref = storage_plan.ref(source)
        graph_outputs = tuple(prepared.graph.output_refs(node))
        outputs = tuple(storage_plan.output_refs(node))
        source_physical = storage_plan.physical_id(plane, source_ref)
        output_physicals = tuple(
            storage_plan.physical_id(plane, output) for output in outputs
        )
        relations = tuple(
            storage_plan.relation(plane, output) for output in outputs
        )
        source_shape = tuple(storage_plan.logical_tensors[source_ref].shape)
        output_shapes = tuple(
            tuple(storage_plan.logical_tensors[output].shape)
            for output in outputs
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AlphaFinalizationError(
            f"{node.name}: zero-copy shuffle split storage is unavailable"
        ) from error

    block_channels = shuffle_plan.block_channels
    output_channels = shuffle_plan.output_channels
    pitch_entries = shuffle_plan.pitch_entries
    span_entries = shuffle_plan.span_entries
    primary_entry_offset = shuffle_plan.primary_entry_offset
    integer_geometry = all(
        not isinstance(value, bool) and isinstance(value, int)
        for value in (
            block_channels,
            output_channels,
            pitch_entries,
            span_entries,
            primary_entry_offset,
        )
    )
    positions = (
        source_shape[0] * source_shape[2] * source_shape[3]
        if len(source_shape) == 4
        else None
    )
    active_entries = (
        output_channels // block_channels
        if integer_geometry and block_channels > 0
        else None
    )
    try:
        source_chain, chained_physical = resolver._chain(source_ref, plane)
        resident_write = resolver._resident_match(source_chain)
        primary_placement = resolver._scratchpad_placement(
            shuffle_plan.primary
        )
        auxiliary_placement = resolver._scratchpad_placement(
            shuffle_plan.auxiliary
        )
        primary_entries, primary_entry_bytes = resolver._entry_geometry(
            shuffle_plan.primary,
            primary_placement,
        )
        auxiliary_entries, auxiliary_entry_bytes = resolver._entry_geometry(
            shuffle_plan.auxiliary,
            auxiliary_placement,
        )
        resident_geometry = (
            resolver._resident_position(
                resident_write,
                shuffle_plan.primary,
                primary_placement,
            )
            if resident_write is not None
            else None
        )
    except AlphaFinalizationError as error:
        raise AlphaFinalizationError(
            f"{node.name}: zero-copy shuffle split backing is unavailable"
        ) from error
    standalone_backing = (
        resident_write is None
        and primary_entry_offset == 0
        and pitch_entries == active_entries
        and primary_entries == span_entries
    )
    resident_backing = (
        resident_geometry is not None
        and chained_physical == shuffle_plan.primary
        and resident_geometry[1:] == (
            primary_entry_offset,
            active_entries,
            pitch_entries,
        )
        and resident_geometry[0].end - resident_geometry[0].start
        == span_entries * primary_entry_bytes
    )
    valid_geometry = (
        integer_geometry
        and block_channels > 0
        and output_channels == 2 * block_channels
        and active_entries == 2
        and pitch_entries >= active_entries
        and primary_entry_offset >= 0
        and primary_entry_offset + active_entries <= pitch_entries
        and span_entries > 0
        and positions is not None
        and positions > 0
        and span_entries
        == (positions - 1) * pitch_entries + active_entries
        and primary_entry_offset + span_entries <= primary_entries
        and auxiliary_entries == span_entries
        and primary_entry_bytes == auxiliary_entry_bytes
        and (standalone_backing or resident_backing)
        and len(source_shape) == 4
        and source_shape[0] == 1
        and source_shape[1] == output_channels
        and len(output_shapes) == 2
        and all(
            shape
            == (
                source_shape[0],
                block_channels,
                source_shape[2],
                source_shape[3],
            )
            for shape in output_shapes
        )
    )
    expected_primary = ViewTransform(
        "alpha_channel_split",
        parameters=(
            ("channel_offset", 0),
            ("channel_size", block_channels),
        ),
    )
    expected_secondary = ViewTransform(
        "alpha_shuffle_output",
        parameters=(("output", 1),),
    )
    chunks = get_arg_value(node, 1, "chunks")
    dimension = get_arg_value(node, 2, "dim", 0)
    valid_arguments = (
        not isinstance(chunks, bool)
        and isinstance(chunks, int)
        and chunks == 2
        and not isinstance(dimension, bool)
        and isinstance(dimension, int)
        and dimension == 1
    )
    valid_relations = (
        len(relations) == 2
        and relations[0].kind is StorageRelationKind.VIEW
        and relations[0].source == source_ref
        and relations[0].physical_source is None
        and relations[0].component is None
        and relations[0].transform == expected_primary
        and relations[1].kind is StorageRelationKind.VIEW
        and relations[1].source is None
        and relations[1].physical_source == shuffle_plan.auxiliary
        and relations[1].component is None
        and relations[1].transform == expected_secondary
    )
    view_by_logical = {
        view.logical_tensor: view for view in prepared.logical_views
    }
    output_views = tuple(view_by_logical.get(output) for output in outputs)
    valid_views = (
        len(output_views) == 2
        and all(view is not None for view in output_views)
        and tuple(view.physical_tensor for view in output_views)
        == (shuffle_plan.primary, shuffle_plan.auxiliary)
        and tuple(view.entry_offset for view in output_views)
        == (primary_entry_offset, 0)
        and tuple(view.byte_offset for view in output_views)
        == (primary_entry_offset * primary_entry_bytes, 0)
        and tuple(view.span_entries for view in output_views)
        == (positions, positions)
        and all(
            view.component == "primary"
            and view.row_pitch_entries == pitch_entries
            for view in output_views
        )
    )
    if not (
        node.op == "call_function"
        and source.op == "call_function"
        and source.target is _ALPHA_SHUFFLE
        and source is shuffle_plan.shuffle
        and source_ref == ValueRef(shuffle_plan.shuffle)
        and valid_arguments
        and graph_outputs == outputs
        and len(outputs) == 2
        and tuple(output.path for output in outputs) == ((0,), (1,))
        and shuffle_plan.primary != shuffle_plan.auxiliary
        and source_physical == shuffle_plan.primary
        and output_physicals
        == (shuffle_plan.primary, shuffle_plan.auxiliary)
        and valid_geometry
        and valid_relations
        and valid_views
    ):
        raise AlphaFinalizationError(
            f"{node.name}: frozen zero-copy shuffle split contract changed"
        )
    return shuffle_plan


def build_alpha_structural_bindings(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
    dram_placements: Iterable[AlphaDramPlacement] = (),
) -> tuple[AlphaRecordBinding, ...]:
    """Freeze all Alpha NOP and terminal-boundary record occurrences.

    The selected execution order is the sole ordering source.  Address-only
    records receive scratchpad views for every input and output occurrence;
    terminal DQ/permute/contiguous records receive their materialized views,
    while each DQ scale is explicitly represented as an unplaced instruction
    parameter.  No DRAM placement is projected into these records.
    """
    scratchpad = tuple(scratchpad_placements)
    dram = tuple(dram_placements)
    resolver = _AlphaLogicalPlacementResolver(prepared, scratchpad, dram)
    scratchpad_by_physical = {
        placement.physical_tensor: placement for placement in scratchpad
    }
    if len(scratchpad_by_physical) != len(scratchpad):
        raise AlphaFinalizationError(
            "Alpha structural scratchpad placements repeat physical tensors"
        )
    requirements = _alpha_structural_requirements(prepared)
    address_only_owners = tuple(
        dict.fromkeys(
            requirement.site.owner
            for requirement in requirements
            if requirement.address_only
        )
    )
    zero_copy_splits = {
        node: shuffle_plan
        for node in address_only_owners
        for shuffle_plan in (
            _alpha_zero_copy_shuffle_split_plan(prepared, resolver, node),
        )
        if shuffle_plan is not None
    }
    structural_owners = {requirement.site.owner for requirement in requirements}
    structural_payload_uses = tuple(
        use
        for use in prepared.payload_uses
        if use.operation in structural_owners
    )
    structural_payload_sites = tuple(
        (use.operation, use.field) for use in structural_payload_uses
    )
    duplicate_payload_sites = _duplicates(structural_payload_sites)
    if duplicate_payload_sites:
        raise AlphaFinalizationError(
            "Alpha structural payload uses repeat operation fields: "
            + ", ".join(
                f"{operation.name}.{field}"
                for operation, field in duplicate_payload_sites
            )
        )
    payload_by_site = dict(
        zip(structural_payload_sites, structural_payload_uses)
    )
    expected_payload_sites = {
        (node, "input") for node in zero_copy_splits
    }
    actual_payload_sites = set(payload_by_site)
    if actual_payload_sites != expected_payload_sites:
        missing = sorted(
            f"{operation.name}.{field}"
            for operation, field in (
                expected_payload_sites - actual_payload_sites
            )
        )
        unexpected = sorted(
            f"{operation.name}.{field}"
            for operation, field in (
                actual_payload_sites - expected_payload_sites
            )
        )
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AlphaFinalizationError(
            "Alpha structural payload-use coverage differs: "
            + "; ".join(details)
        )
    for node, shuffle_plan in zero_copy_splits.items():
        use = payload_by_site[(node, "input")]
        if (
            use.logical_tensor != ValueRef(shuffle_plan.shuffle)
            or use.physical_tensor != shuffle_plan.primary
            or use.component != "primary"
            or use.entry_offset != shuffle_plan.primary_entry_offset
            or use.span_entries != shuffle_plan.span_entries
            or use.row_pitch_entries != shuffle_plan.pitch_entries
        ):
            raise AlphaFinalizationError(
                f"{node.name}.input: zero-copy split payload use disagrees "
                "with its shuffle plan"
            )
    bindings = []
    nop_physicals: dict[Node, set[PhysicalTensorId]] = {}
    for requirement in requirements:
        scratchpad_view = None
        if requirement.placement == _STRUCTURAL_SCRATCHPAD:
            payload_use = payload_by_site.get(
                (requirement.site.member, requirement.site.field)
            )
            if payload_use is None:
                scratchpad_view = resolver.view_for(
                    requirement.value,
                    StoragePlane.SCRATCHPAD,
                )
            else:
                if payload_use.logical_tensor != requirement.value:
                    raise AlphaFinalizationError(
                        f"{requirement.site.label}: payload use names the "
                        "wrong logical tensor"
                    )
                scratchpad_view = resolver.view_for_payload_use(payload_use)
                semantic_view = resolver.view_for(
                    requirement.value,
                    StoragePlane.SCRATCHPAD,
                )
                if (
                    semantic_view is None
                    or scratchpad_view.physical_tensor
                    != semantic_view.physical_tensor
                    or scratchpad_view.component != semantic_view.component
                    or scratchpad_view.start != semantic_view.start
                    or scratchpad_view.end != semantic_view.end
                    or scratchpad_view.address != semantic_view.address
                ):
                    raise AlphaFinalizationError(
                        f"{requirement.site.label}: payload-use anchor "
                        "disagrees with its semantic Alpha view"
                    )
            if scratchpad_view is None:
                raise AlphaFinalizationError(
                    f"{requirement.site.label}: structural tensor has no "
                    "scratchpad view"
                )
            if requirement.address_only:
                nop_physicals.setdefault(
                    requirement.site.owner,
                    set(),
                ).add(scratchpad_view.physical_tensor)
        elif requirement.placement == _STRUCTURAL_UNPLACED:
            scratchpad_physical = prepared.plan.physical_id(
                StoragePlane.SCRATCHPAD,
                requirement.value,
            )
            if scratchpad_physical in scratchpad_by_physical:
                raise AlphaFinalizationError(
                    f"{requirement.site.label}: terminal DQ scale unexpectedly "
                    "occupies scratchpad"
                )
            if resolver.view_for(
                requirement.value,
                StoragePlane.DRAM,
            ) is not None:
                raise AlphaFinalizationError(
                    f"{requirement.site.label}: terminal DQ scale unexpectedly "
                    "has a DRAM record placement"
                )
        bindings.append(
            AlphaRecordBinding(
                site=requirement.site,
                value=requirement.value,
                scratchpad=scratchpad_view,
            )
        )
    invalid_nops = tuple(
        node
        for node, physicals in nop_physicals.items()
        if len(physicals) != 1
        and not (
            node in zero_copy_splits
            and physicals
            == {
                zero_copy_splits[node].primary,
                zero_copy_splits[node].auxiliary,
            }
        )
    )
    if invalid_nops:
        raise AlphaFinalizationError(
            "Alpha address-only records change physical backing: "
            + ", ".join(node.name for node in invalid_nops)
        )
    result = tuple(bindings)
    _validate_alpha_structural_binding_inventory(prepared, result)
    return result


class _AlphaShuffleBindingBuilder:
    """Strictly freeze the storage-owned fields of Alpha shuffle records."""

    def __init__(
        self,
        prepared: AlphaPreparedStorage,
        placements: Iterable[AlphaPlacement],
    ) -> None:
        if not isinstance(prepared, AlphaPreparedStorage):
            raise TypeError("Alpha shuffle binding needs prepared storage")
        placements = tuple(placements)
        if not all(isinstance(item, AlphaPlacement) for item in placements):
            raise TypeError(
                "Alpha shuffle bindings need AlphaPlacement values"
            )
        self.prepared = prepared
        self.placements = placements
        self.bindings: list[AlphaRecordBinding] = []
        self._sites: set[AlphaRecordSite] = set()

        self._validate_plan_collections()
        self._placement_by_physical = self._index_placements()
        self._request_by_physical = self._index_requests()
        self._payload_by_site = self._index_payload_uses()
        self._serialized_by_site = self._index_serialized_uses()

    @staticmethod
    def _integer(value, label: str, *, positive: bool = False) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AlphaFinalizationError(f"{label} must be an integer")
        if value < (1 if positive else 0):
            qualifier = "positive" if positive else "nonnegative"
            raise AlphaFinalizationError(f"{label} must be {qualifier}")
        return value

    def _validate_plan_collections(self) -> None:
        plans = tuple(self.prepared.shuffle_plans)
        if not all(isinstance(plan, AlphaShufflePlan) for plan in plans):
            raise TypeError("shuffle plans must be AlphaShufflePlan values")
        duplicate_shuffles = _duplicates(plan.shuffle for plan in plans)
        if duplicate_shuffles:
            raise AlphaFinalizationError(
                "Alpha shuffle plans repeat owners: "
                + ", ".join(node.name for node in duplicate_shuffles)
            )
        copies = tuple(copy for plan in plans for copy in plan.copies)
        duplicate_copies = _duplicates(copies)
        if duplicate_copies:
            raise AlphaFinalizationError(
                "Alpha shuffle plans repeat copies: "
                + ", ".join(node.name for node in duplicate_copies)
            )
        executable = tuple(self.prepared.executable_owners)
        duplicate_owners = _duplicates(executable)
        if duplicate_owners:
            raise AlphaFinalizationError(
                "Alpha executable owners repeat: "
                + ", ".join(node.name for node in duplicate_owners)
            )
        self._planned_shuffles = frozenset(plan.shuffle for plan in plans)
        self._planned_copies = frozenset(copies)
        _validated_alpha_shuffle_copy_refs(self.prepared)

    def _index_placements(self) -> dict[PhysicalTensorId, AlphaPlacement]:
        physicals = tuple(
            placement.physical_tensor for placement in self.placements
        )
        duplicates = _duplicates(physicals)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha scratchpad placements repeat: "
                + ", ".join(physical.name for physical in duplicates)
            )
        return {
            placement.physical_tensor: placement
            for placement in self.placements
        }

    def _index_requests(self):
        requests = tuple(self.prepared.allocation_requests)
        physicals = tuple(request.physical_tensor for request in requests)
        duplicates = _duplicates(physicals)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha allocation requests repeat: "
                + ", ".join(physical.name for physical in duplicates)
            )
        return {request.physical_tensor: request for request in requests}

    def _index_payload_uses(
        self,
    ) -> dict[tuple[Node, str], AlphaPayloadUse]:
        uses = tuple(self.prepared.payload_uses)
        if not all(isinstance(use, AlphaPayloadUse) for use in uses):
            raise TypeError("payload uses must be AlphaPayloadUse values")
        sites = tuple((use.operation, use.field) for use in uses)
        duplicates = _duplicates(sites)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha payload uses repeat operation fields: "
                + ", ".join(
                    f"{operation.name}.{field}"
                    for operation, field in duplicates
                )
            )
        for use in uses:
            if use.operation in self._planned_shuffles:
                raise AlphaFinalizationError(
                    f"{use.operation.name}.{use.field}: shuffle payload view "
                    "must come from its AlphaShufflePlan"
                )
            if (
                use.field.startswith("copy_")
                and use.operation not in self._planned_copies
            ):
                raise AlphaFinalizationError(
                    f"{use.operation.name}.{use.field}: orphan Alpha copy use"
                )
        return dict(zip(sites, uses))

    def _index_serialized_uses(
        self,
    ) -> dict[tuple[Node, str], AlphaSerializedUse]:
        uses = tuple(self.prepared.serialized_uses)
        if not all(isinstance(use, AlphaSerializedUse) for use in uses):
            raise TypeError(
                "serialized uses must be AlphaSerializedUse values"
            )
        sites = tuple((use.operation, use.field) for use in uses)
        duplicates = _duplicates(sites)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha serialized uses repeat operation fields: "
                + ", ".join(
                    f"{operation.name}.{field}"
                    for operation, field in duplicates
                )
            )
        for use in uses:
            if (
                use.field == "copy_block_payload"
                and use.operation not in self._planned_copies
            ):
                raise AlphaFinalizationError(
                    f"{use.operation.name}.{use.field}: orphan Alpha copy use"
                )
        return dict(zip(sites, uses))

    def _site(self, operation: Node, field: str) -> AlphaRecordSite:
        matches = []
        model = self.prepared.graph.model
        for owner in self.prepared.executable_owners:
            if operation is owner:
                matches.append(owner)
                continue
            if owner.op != "call_module":
                continue
            try:
                submodule = model.get_submodule(str(owner.target))
            except (AttributeError, KeyError):
                continue
            if operation.graph is getattr(submodule, "graph", None):
                matches.append(owner)
        if len(matches) != 1:
            detail = "missing" if not matches else "ambiguous"
            raise AlphaFinalizationError(
                f"{operation.name}: {detail} Alpha record owner"
            )
        owner = matches[0]
        if owner not in set(self.prepared.execution_order.nodes):
            raise AlphaFinalizationError(
                f"{operation.name}: record owner is outside execution order"
            )
        return AlphaRecordSite(owner=owner, member=operation, field=field)

    def _add(
        self,
        operation: Node,
        field: str,
        value,
        *,
        scratchpad: Optional[AlphaPlacedView] = None,
    ) -> None:
        site = self._site(operation, field)
        if site in self._sites:
            raise AlphaFinalizationError(
                f"{site.label}: Alpha shuffle binding was emitted twice"
            )
        self._sites.add(site)
        self.bindings.append(
            AlphaRecordBinding(site, value, scratchpad=scratchpad)
        )

    def _placement(self, physical: PhysicalTensorId) -> AlphaPlacement:
        if not isinstance(physical, PhysicalTensorId):
            raise TypeError("Alpha shuffle storage needs physical tensor IDs")
        if physical.plane != StoragePlane.SCRATCHPAD.value:
            raise AlphaFinalizationError(
                f"{physical.name}: shuffle backing is not in scratchpad"
            )
        placement = self._placement_by_physical.get(physical)
        if placement is None:
            raise AlphaFinalizationError(
                f"{physical.name}: missing Alpha scratchpad placement"
            )
        request = self._request_by_physical.get(physical)
        if request is None:
            raise AlphaFinalizationError(
                f"{physical.name}: missing Alpha allocation request"
            )
        if placement.request != request:
            raise AlphaFinalizationError(
                f"{physical.name}: placement does not match prepared request"
            )
        try:
            self.prepared.plan.storage_key(
                physical,
                StoragePlane.SCRATCHPAD,
            )
            memory = GLOBAL.instance(placement.memory_instance)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}: placement is absent from Alpha storage"
            ) from error
        if placement.memory_instance != GLOBAL.output_memory:
            raise AlphaFinalizationError(
                f"{physical.name}: shuffle backing is not in Alpha output "
                "memory"
            )
        if placement.memory_space != GLOBAL.instances.index(memory):
            raise AlphaFinalizationError(
                f"{physical.name}: placement memory-space index is stale"
            )
        try:
            expected_address = alpha_place(
                memory,
                placement.start,
                cluster=placement.cluster,
                operand_bits=placement.operand_bits,
            ).encode()
        except ValueError as error:
            raise AlphaFinalizationError(
                f"{physical.name}: placement cannot be Alpha encoded"
            ) from error
        if placement.address != expected_address:
            raise AlphaFinalizationError(
                f"{physical.name}: base address is not encoded from bytes"
            )
        return placement

    def _backing_entries(
        self,
        physical: PhysicalTensorId,
        placement: AlphaPlacement,
    ) -> tuple[int, int]:
        memory = GLOBAL.instance(placement.memory_instance)
        entry_bytes = alpha_entry_bytes(memory, placement.cluster)
        try:
            size = self.prepared.plan.backing_size_bytes(
                physical,
                StoragePlane.SCRATCHPAD,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}: shuffle backing size is unavailable"
            ) from error
        entries, remainder = divmod(size, entry_bytes)
        if remainder or entries <= 0:
            raise AlphaFinalizationError(
                f"{physical.name}: shuffle backing is not entry aligned"
            )
        if placement.end - placement.start < size:
            raise AlphaFinalizationError(
                f"{physical.name}: placement is smaller than its backing"
            )
        return entries, entry_bytes

    def _view(
        self,
        physical: PhysicalTensorId,
        *,
        component: str,
        entry_offset: int,
        span_entries: int,
        through_base_end: bool = False,
    ) -> AlphaPlacedView:
        entry_offset = self._integer(
            entry_offset,
            f"{physical.name} entry offset",
        )
        span_entries = self._integer(
            span_entries,
            f"{physical.name} entry span",
            positive=True,
        )
        placement = self._placement(physical)
        backing_entries, entry_bytes = self._backing_entries(
            physical,
            placement,
        )
        if entry_offset + span_entries > backing_entries:
            raise AlphaFinalizationError(
                f"{physical.name}: Alpha view exceeds its backing"
            )
        start = placement.start + entry_offset * entry_bytes
        end = (
            placement.end
            if through_base_end
            else start + span_entries * entry_bytes
        )
        try:
            address = alpha_place(
                GLOBAL.instance(placement.memory_instance),
                start,
                cluster=placement.cluster,
                operand_bits=placement.operand_bits,
            ).encode()
            self.prepared.plan.physical_component(
                physical,
                StoragePlane.SCRATCHPAD,
                component,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{physical.name}.{component}: Alpha view cannot be encoded"
            ) from error
        return AlphaPlacedView.from_placement(
            placement,
            component=component,
            start=start,
            end=end,
            address=address,
        )

    def _payload_use(self, copy: Node, field: str) -> AlphaPayloadUse:
        try:
            return self._payload_by_site[(copy, field)]
        except KeyError:
            raise AlphaFinalizationError(
                f"{copy.name}.{field}: missing Alpha payload use"
            ) from None

    def _serialized_use(self, copy: Node) -> AlphaSerializedUse:
        field = "copy_block_payload"
        try:
            return self._serialized_by_site[(copy, field)]
        except KeyError:
            raise AlphaFinalizationError(
                f"{copy.name}.{field}: missing Alpha serialized use"
            ) from None

    def _validate_copy_use_fields(self, copy: Node) -> None:
        payload_fields = {
            field
            for operation, field in self._payload_by_site
            if operation is copy
        }
        serialized_fields = {
            field
            for operation, field in self._serialized_by_site
            if operation is copy
        }
        if payload_fields != _COPY_PAYLOAD_FIELDS:
            missing = sorted(_COPY_PAYLOAD_FIELDS - payload_fields)
            unexpected = sorted(payload_fields - _COPY_PAYLOAD_FIELDS)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise AlphaFinalizationError(
                f"{copy.name}: copy payload-use coverage differs: "
                + "; ".join(details)
            )
        if serialized_fields != _COPY_SERIALIZED_FIELDS:
            missing = sorted(_COPY_SERIALIZED_FIELDS - serialized_fields)
            unexpected = sorted(
                serialized_fields - _COPY_SERIALIZED_FIELDS
            )
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise AlphaFinalizationError(
                f"{copy.name}: copy artifact-use coverage differs: "
                + "; ".join(details)
            )

    @staticmethod
    def _single_output(prepared: AlphaPreparedStorage, node: Node) -> ValueRef:
        outputs = tuple(prepared.plan.output_refs(node))
        if len(outputs) != 1:
            raise AlphaFinalizationError(
                f"{node.name}: Alpha shuffle record needs one tensor output"
            )
        return outputs[0]

    @staticmethod
    def _validate_transform(
        copy: Node,
        movement: AlphaShuffleMovement,
        *,
        source_entry: int,
        destination_entry: int,
        pitch_entries: int,
    ) -> None:
        expected_source = {
            "output": movement.source_output,
            "entry": source_entry,
            "pitch_entries": pitch_entries,
        }
        expected_destination = {
            "entry": destination_entry,
            "pitch_entries": pitch_entries,
        }
        if (
            movement.source_view.kind != "alpha_shuffle_copy_source"
            or dict(movement.source_view.parameters) != expected_source
            or movement.destination_view.kind
            != "alpha_shuffle_copy_destination"
            or dict(movement.destination_view.parameters)
            != expected_destination
        ):
            raise AlphaFinalizationError(
                f"{copy.name}: shuffle movement views are stale"
            )

    def _validate_entry_use(
        self,
        use: AlphaPayloadUse,
        *,
        logical: ValueRef,
        physical: PhysicalTensorId,
        entry: int,
        pitch_entries: int,
    ) -> None:
        if (
            use.logical_tensor != logical
            or use.physical_tensor != physical
            or use.component != "primary"
            or use.entry_offset != entry
            or use.span_entries != 1
            or use.row_pitch_entries != pitch_entries
        ):
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: entry view disagrees "
                "with its shuffle movement"
            )

    def _validate_full_use(
        self,
        use: AlphaPayloadUse,
        *,
        logical: ValueRef,
        physical: PhysicalTensorId,
        pitch_entries: int,
    ) -> AlphaPlacement:
        placement = self._placement(physical)
        backing_entries, _entry_bytes = self._backing_entries(
            physical,
            placement,
        )
        if (
            use.logical_tensor != logical
            or use.physical_tensor != physical
            or use.component != "primary"
            or use.entry_offset != 0
            or use.span_entries != backing_entries
            or use.row_pitch_entries != pitch_entries
        ):
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: physical payload is not "
                "the full shuffle backing"
            )
        return placement

    def _build_copy(
        self,
        plan: AlphaShufflePlan,
        movement: AlphaShuffleMovement,
        copy: Node,
        *,
        active_entries: int,
    ) -> None:
        if not isinstance(movement, AlphaShuffleMovement):
            raise TypeError("shuffle movements must be AlphaShuffleMovement")
        if movement.copy is not copy:
            raise AlphaFinalizationError(
                f"{copy.name}: copy/movement ordering disagrees"
            )
        self._validate_copy_use_fields(copy)
        source_output = self._integer(
            movement.source_output,
            f"{copy.name} source output",
        )
        if source_output not in (0, 1):
            raise AlphaFinalizationError(
                f"{copy.name}: source output must be zero or one"
            )
        source_entry = self._integer(
            movement.source_entry,
            f"{copy.name} source entry",
        )
        destination_entry = self._integer(
            movement.destination_entry,
            f"{copy.name} destination entry",
        )
        if source_entry >= active_entries or destination_entry >= active_entries:
            raise AlphaFinalizationError(
                f"{copy.name}: movement entry exceeds active shuffle width"
            )
        if movement.block_channels != plan.block_channels:
            raise AlphaFinalizationError(
                f"{copy.name}: movement block width disagrees with shuffle"
            )
        metadata = copy.meta.get("magnet_alpha")
        declared_width = (
            metadata.get("block_channels")
            if isinstance(metadata, dict)
            else None
        )
        if declared_width != plan.block_channels:
            raise AlphaFinalizationError(
                f"{copy.name}: rewrite block width disagrees with shuffle"
            )
        expected_source_physical = (
            plan.primary if source_output == 0 else plan.auxiliary
        )
        if (
            movement.source_physical != expected_source_physical
            or movement.destination_physical != plan.primary
        ):
            raise AlphaFinalizationError(
                f"{copy.name}: movement physical endpoints are stale"
            )
        absolute_source_entry = source_entry + (
            plan.primary_entry_offset if source_output == 0 else 0
        )
        absolute_destination_entry = (
            plan.primary_entry_offset + destination_entry
        )
        self._validate_transform(
            copy,
            movement,
            source_entry=absolute_source_entry,
            destination_entry=absolute_destination_entry,
            pitch_entries=plan.pitch_entries,
        )

        if not copy.args or not isinstance(copy.args[0], Node):
            raise AlphaFinalizationError(
                f"{copy.name}: copy has no logical FX input"
            )
        try:
            logical_input = self.prepared.plan.ref(copy.args[0])
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{copy.name}: logical FX input is absent from storage"
            ) from error
        logical_output = self._single_output(self.prepared, copy)
        input_use = self._payload_use(copy, "input")
        output_use = self._payload_use(copy, "output_payload")
        self._validate_entry_use(
            input_use,
            logical=logical_input,
            physical=expected_source_physical,
            entry=absolute_source_entry,
            pitch_entries=plan.pitch_entries,
        )
        self._validate_entry_use(
            output_use,
            logical=logical_output,
            physical=plan.primary,
            entry=absolute_destination_entry,
            pitch_entries=plan.pitch_entries,
        )

        input_view = self._view(
            input_use.physical_tensor,
            component=input_use.component,
            entry_offset=input_use.entry_offset,
            span_entries=input_use.span_entries,
        )
        output_view = self._view(
            output_use.physical_tensor,
            component=output_use.component,
            entry_offset=output_use.entry_offset,
            span_entries=output_use.span_entries,
        )
        self._add(
            copy,
            "input",
            input_use.logical_tensor,
            scratchpad=input_view,
        )
        self._add(
            copy,
            "output_payload",
            output_use.logical_tensor,
            scratchpad=output_view,
        )

        source_use = self._payload_use(copy, "copy_source_payload")
        destination_use = self._payload_use(
            copy,
            "copy_destination_payload",
        )
        source_placement = self._validate_full_use(
            source_use,
            logical=logical_input,
            physical=expected_source_physical,
            pitch_entries=plan.pitch_entries,
        )
        destination_placement = self._validate_full_use(
            destination_use,
            logical=logical_output,
            physical=plan.primary,
            pitch_entries=plan.pitch_entries,
        )
        try:
            source_payload = self.prepared.plan.payload_ref(
                source_use.physical_tensor,
                StoragePlane.SCRATCHPAD,
                source_use.component,
                source_placement,
            )
            destination_payload = self.prepared.plan.payload_ref(
                destination_use.physical_tensor,
                StoragePlane.SCRATCHPAD,
                destination_use.component,
                destination_placement,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{copy.name}: copy payload backing cannot be finalized"
            ) from error
        self._add(copy, "copy_source_payload", source_payload)
        self._add(copy, "copy_destination_payload", destination_payload)

        artifact_use = self._serialized_use(copy)
        expected_artifact = f"{copy.name}.copy_block_payload"
        if (
            artifact_use.logical_tensor != logical_output
            or artifact_use.artifact != expected_artifact
        ):
            raise AlphaFinalizationError(
                f"{copy.name}: copy-block artifact declaration is stale"
            )
        try:
            artifact = self.prepared.plan.serialized_ref(
                artifact_use.logical_tensor,
                StoragePlane.SCRATCHPAD,
                artifact_use.artifact,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{copy.name}: copy-block artifact is not finalized"
            ) from error
        if artifact.physical_tensor != plan.primary:
            raise AlphaFinalizationError(
                f"{copy.name}: copy-block artifact has the wrong backing"
            )
        shape = tuple(artifact.artifact.shape)
        if len(shape) != 4 or shape[1] != plan.block_channels:
            raise AlphaFinalizationError(
                f"{copy.name}: copy-block artifact has the wrong shape"
            )
        self._add(copy, "copy_block_payload", artifact)

        for field, value in (
            ("block_channels", plan.block_channels),
            ("concat_total_size_0", plan.output_channels),
            ("copy_destination_entry", destination_entry),
            ("copy_source_entry", source_entry),
            ("copy_source_output", source_output),
            ("is_to_concat", True),
            ("nmp_c2", plan.copy_nmp_c2),
            ("nmp_k2", plan.copy_nmp_k2),
        ):
            self._add(copy, field, value)

    def _build_shuffle(self, plan: AlphaShufflePlan) -> None:
        block_channels = self._integer(
            plan.block_channels,
            f"{plan.shuffle.name} block channels",
            positive=True,
        )
        output_channels = self._integer(
            plan.output_channels,
            f"{plan.shuffle.name} output channels",
            positive=True,
        )
        active_entries, remainder = divmod(output_channels, block_channels)
        if remainder:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: output is not block aligned"
            )
        pitch_entries = self._integer(
            plan.pitch_entries,
            f"{plan.shuffle.name} pitch entries",
            positive=True,
        )
        span_entries = self._integer(
            plan.span_entries,
            f"{plan.shuffle.name} span entries",
            positive=True,
        )
        primary_entry_offset = self._integer(
            plan.primary_entry_offset,
            f"{plan.shuffle.name} primary entry offset",
        )
        if pitch_entries < active_entries:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: pitch is narrower than the output"
            )
        if plan.primary == plan.auxiliary:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: shuffle backings are not distinct"
            )
        logical_output = self._single_output(self.prepared, plan.shuffle)
        try:
            logical = self.prepared.plan.logical_tensors[logical_output]
            physical = self.prepared.plan.storage_key(
                logical_output,
                StoragePlane.SCRATCHPAD,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: output storage is not finalized"
            ) from error
        shape = tuple(logical.shape)
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[1] != output_channels
        ):
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: output geometry disagrees with plan"
            )
        positions = shape[0] * shape[2] * shape[3]
        expected_span = (positions - 1) * pitch_entries + active_entries
        if span_entries != expected_span:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: shuffle span disagrees with row pitch"
            )
        if physical != plan.primary:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: primary backing is stale"
            )

        primary = self._placement(plan.primary)
        auxiliary = self._placement(plan.auxiliary)
        primary_entries, _entry_bytes = self._backing_entries(
            plan.primary,
            primary,
        )
        auxiliary_entries, _entry_bytes = self._backing_entries(
            plan.auxiliary,
            auxiliary,
        )
        if primary_entry_offset + span_entries > primary_entries:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: primary view exceeds its backing"
            )
        if auxiliary_entries != span_entries:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: auxiliary extent is stale"
            )
        if primary.memory_instance != auxiliary.memory_instance:
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: shuffle outputs cross memories"
            )

        output_view = self._view(
            plan.primary,
            component="primary",
            entry_offset=primary_entry_offset,
            span_entries=span_entries,
            through_base_end=True,
        )
        self._add(
            plan.shuffle,
            "output_payload",
            logical_output,
            scratchpad=output_view,
        )
        self._add(
            plan.shuffle,
            "output_addresses",
            (primary.address, auxiliary.address),
        )
        self._add(
            plan.shuffle,
            "output0_memory_instance",
            primary.memory_instance,
        )
        self._add(
            plan.shuffle,
            "output1_memory_instance",
            auxiliary.memory_instance,
        )
        pitch_gap = pitch_entries - active_entries
        if pitch_gap:
            self._add(plan.shuffle, "nmp_k2", pitch_gap)
        self._add(plan.shuffle, "padding_c", 0)
        self._add(plan.shuffle, "padding_k", 0)

        movements = tuple(plan.movements)
        if len(plan.copies) != len(movements):
            raise AlphaFinalizationError(
                f"{plan.shuffle.name}: copy/movement coverage differs"
            )
        previous = plan.shuffle
        for copy, movement in zip(plan.copies, movements):
            if not copy.args or copy.args[0] is not previous:
                raise AlphaFinalizationError(
                    f"{copy.name}: logical copy chain is stale"
                )
            self._build_copy(
                plan,
                movement,
                copy,
                active_entries=active_entries,
            )
            previous = copy

    def build(self) -> tuple[AlphaRecordBinding, ...]:
        for plan in self.prepared.shuffle_plans:
            self._build_shuffle(plan)
        return tuple(self.bindings)


def build_alpha_shuffle_bindings(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
) -> tuple[AlphaRecordBinding, ...]:
    """Finalize fork-compatible shuffle/copy storage record fields.

    This target-local handoff consumes only frozen storage facts and allocator
    placements.  Every logical entry view is re-encoded from its byte offset;
    an encoded base address is never used as an arithmetic quantity.
    """
    return _AlphaShuffleBindingBuilder(
        prepared,
        scratchpad_placements,
    ).build()


def build_alpha_resident_bindings(
    prepared: AlphaPreparedStorage,
) -> tuple[AlphaRecordBinding, ...]:
    """Freeze producer-side concat fields from resident-write decisions.

    A channel offset changes the logical view selected from the backing; it
    is not a protobuf scalar.  Only a resident write explicitly marked as a
    concat emits the fork's five concat fields.  Pitch-only writes emit none.
    """
    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha resident bindings need prepared storage")
    plan = prepared.plan
    bindings = []
    sites = set()
    owners = tuple(write.owner for write in prepared.resident_writes)
    duplicates = _duplicates(owners)
    if duplicates:
        raise AlphaFinalizationError(
            "Alpha resident writes repeat owners: "
            + ", ".join(owner.name for owner in duplicates)
        )

    for write in prepared.resident_writes:
        if not isinstance(write, AlphaResidentWriteBinding):
            raise TypeError(
                "resident writes must be AlphaResidentWriteBinding values"
            )
        constraint = prepared.constraint_for(write.owner)
        if constraint is None or constraint.output != write.logical_tensor:
            raise AlphaFinalizationError(
                f"{write.owner.name}: resident write has no matching Alpha "
                "operation output"
            )
        try:
            physical = plan.physical_id(
                StoragePlane.SCRATCHPAD,
                write.logical_tensor,
            )
            relation = plan.relation(
                StoragePlane.SCRATCHPAD,
                write.logical_tensor,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{write.owner.name}: resident output is absent from final "
                "Alpha storage"
            ) from error
        expected_kind = (
            StorageRelationKind.MATERIALIZE
            if write.logical_tensor == write.backing
            else StorageRelationKind.WRITE_INTO
        )
        expected_source = (
            None if write.logical_tensor == write.backing else write.backing
        )
        parameters = {
            "channel_offset": write.channel_offset,
            "slot_index": write.slot_index,
            "slot_channels": write.slot_channels,
            "resident_channels": write.resident_channels,
        }
        if write.concat_channel_index is not None:
            parameters["concat_channel_index"] = write.concat_channel_index
        if (
            physical != write.physical_tensor
            or relation.kind is not expected_kind
            or relation.source != expected_source
            or relation.transform is None
            or relation.transform.kind != "alpha_resident_write"
            or dict(relation.transform.parameters) != parameters
        ):
            raise AlphaFinalizationError(
                f"{write.owner.name}: frozen Alpha resident write is stale"
            )
        if write.concat_channel_index is None:
            continue

        fields = (
            ("concat_channel_index_0", write.concat_channel_index),
            ("concat_idx_0", write.slot_index),
            ("concat_size_0", write.slot_channels),
            ("concat_total_size_0", write.resident_channels),
            ("is_to_concat", True),
        )
        for field, value in fields:
            site = AlphaRecordSite(
                owner=write.owner,
                member=constraint.terminal,
                field=field,
            )
            if site in sites:
                raise AlphaFinalizationError(
                    f"{site.label}: Alpha resident binding was emitted twice"
                )
            sites.add(site)
            bindings.append(AlphaRecordBinding(site=site, value=value))
    return tuple(bindings)


def build_alpha_model_bindings(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
    dram_placements: Iterable[AlphaDramPlacement],
) -> tuple[AlphaRecordBinding, ...]:
    """Bind exactly the entry tensors the storage plan loads once.

    This deliberately follows ``StorageTransfer`` rather than
    ``require_allocation``.  Float instruction parameters are consequently
    absent, while every model input/parameter the endpoint must preload has
    both its persistent DRAM placement and its resident scratchpad placement.
    """
    resolver = _AlphaLogicalPlacementResolver(
        prepared,
        scratchpad_placements,
        dram_placements,
    )
    bindings = []
    sites = set()
    seen = set()
    residents = set(prepared.resident_physical_ids)
    for transfer in prepared.plan.transfers:
        logical = transfer.logical_tensor
        if (
            transfer.kind != "load"
            or transfer.source.plane != StoragePlane.DRAM.value
            or transfer.destination.plane != StoragePlane.SCRATCHPAD.value
            or transfer.destination not in residents
            or logical in seen
        ):
            continue
        seen.add(logical)
        node = logical.node
        if node.op not in {"placeholder", "get_attr"}:
            raise AlphaFinalizationError(
                f"{logical.name}: Alpha entry load is not a model input or "
                "parameter"
            )
        memory = resolver.view_for(logical, StoragePlane.DRAM)
        scratchpad = resolver.view_for(logical, StoragePlane.SCRATCHPAD)
        if memory is None or scratchpad is None:
            raise AlphaFinalizationError(
                f"{logical.name}: Alpha entry load has incomplete placement"
            )
        if (
            memory.physical_tensor != transfer.source
            or scratchpad.physical_tensor != transfer.destination
        ):
            raise AlphaFinalizationError(
                f"{logical.name}: Alpha entry load placement is stale"
            )
        site = AlphaRecordSite(
            owner=node,
            member=None,
            field="inputs" if node.op == "placeholder" else "parameters",
            occurrence=logical.path,
        )
        if site in sites:
            raise AlphaFinalizationError(
                f"{site.label}: Alpha model binding was emitted twice"
            )
        sites.add(site)
        bindings.append(
            AlphaRecordBinding(
                site=site,
                value=logical,
                memory=memory,
                scratchpad=scratchpad,
            )
        )
    return tuple(bindings)


_GEMM_PAYLOAD_FIELDS = frozenset(
    {"weight_payload", "materialized_input_payload"}
)
_GEMM_SERIALIZED_FIELDS = frozenset(
    {"input_vector_scale_payload", "weight_vector_scale_payload"}
)


class _AlphaGemmBindingBuilder:
    """Freeze GEMM storage fields at their exact flat-record members."""

    def __init__(
        self,
        prepared: AlphaPreparedStorage,
        scratchpad_placements: Iterable[AlphaPlacement],
        dram_placements: Iterable[AlphaDramPlacement],
    ) -> None:
        if not isinstance(prepared, AlphaPreparedStorage):
            raise TypeError("Alpha GEMM binding needs prepared storage")
        self.prepared = prepared
        self.plan = prepared.plan
        self.resolver = _AlphaLogicalPlacementResolver(
            prepared,
            scratchpad_placements,
            dram_placements,
        )
        self.bindings: list[AlphaRecordBinding] = []
        self._sites: set[AlphaRecordSite] = set()
        self._payload_by_site = self._index_payload_uses()
        self._serialized_by_site = self._index_serialized_uses()
        self._consumed_payload_sites: set[tuple[Node, str]] = set()
        self._consumed_serialized_sites: set[tuple[Node, str]] = set()

    def _index_payload_uses(self) -> dict[tuple[Node, str], AlphaPayloadUse]:
        uses = tuple(self.prepared.payload_uses)
        if not all(isinstance(use, AlphaPayloadUse) for use in uses):
            raise TypeError("payload uses must be AlphaPayloadUse values")
        sites = tuple((use.operation, use.field) for use in uses)
        duplicates = _duplicates(sites)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha payload uses repeat operation fields: "
                + ", ".join(
                    f"{operation.name}.{field}"
                    for operation, field in duplicates
                )
            )
        return dict(zip(sites, uses))

    def _index_serialized_uses(
        self,
    ) -> dict[tuple[Node, str], AlphaSerializedUse]:
        uses = tuple(self.prepared.serialized_uses)
        if not all(isinstance(use, AlphaSerializedUse) for use in uses):
            raise TypeError("serialized uses must be AlphaSerializedUse values")
        sites = tuple((use.operation, use.field) for use in uses)
        duplicates = _duplicates(sites)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha serialized uses repeat operation fields: "
                + ", ".join(
                    f"{operation.name}.{field}"
                    for operation, field in duplicates
                )
            )
        return dict(zip(sites, uses))

    def _add(
        self,
        owner: Node,
        member: Node,
        field: str,
        value,
        *,
        memory: Optional[AlphaPlacedView] = None,
        scratchpad: Optional[AlphaPlacedView] = None,
    ) -> None:
        site = AlphaRecordSite(owner=owner, member=member, field=field)
        if site in self._sites:
            raise AlphaFinalizationError(
                f"{site.label}: Alpha GEMM binding was emitted twice"
            )
        self._sites.add(site)
        self.bindings.append(
            AlphaRecordBinding(
                site=site,
                value=value,
                memory=memory,
                scratchpad=scratchpad,
            )
        )

    def _add_logical(
        self,
        owner: Node,
        member: Node,
        field: str,
        logical: ValueRef,
        *,
        expected_scratchpad: PhysicalTensorId,
        expected_component: str = "primary",
    ) -> None:
        scratchpad = self.resolver.view_for(
            logical,
            StoragePlane.SCRATCHPAD,
        )
        if (
            scratchpad is None
            or scratchpad.physical_tensor != expected_scratchpad
            or scratchpad.component != expected_component
        ):
            raise AlphaFinalizationError(
                f"{logical.name}: {field} resolved to the wrong Alpha "
                "scratchpad backing or component"
            )
        memory = self.resolver.view_for(logical, StoragePlane.DRAM)
        self._add(
            owner,
            member,
            field,
            logical,
            memory=memory,
            scratchpad=scratchpad,
        )

    def _payload_use(self, operation: Node, field: str) -> AlphaPayloadUse:
        site = (operation, field)
        try:
            use = self._payload_by_site[site]
        except KeyError:
            raise AlphaFinalizationError(
                f"{operation.name}.{field}: missing Alpha GEMM payload use"
            ) from None
        self._consumed_payload_sites.add(site)
        return use

    def _serialized_use(
        self,
        operation: Node,
        field: str,
    ) -> AlphaSerializedUse:
        site = (operation, field)
        try:
            use = self._serialized_by_site[site]
        except KeyError:
            raise AlphaFinalizationError(
                f"{operation.name}.{field}: missing Alpha GEMM serialized use"
            ) from None
        self._consumed_serialized_sites.add(site)
        return use

    def _physical_payload(
        self,
        use: AlphaPayloadUse,
        *,
        logical: ValueRef,
        physical: PhysicalTensorId,
        component: str,
    ) -> PhysicalPayloadRef:
        if (
            use.logical_tensor != logical
            or use.physical_tensor != physical
            or use.component != component
            or use.entry_offset != 0
            or use.span_entries is not None
            or use.row_pitch_entries is not None
        ):
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: Alpha GEMM payload use "
                "is not the complete frozen backing"
            )
        placement = self.resolver._scratchpad_placement(physical)
        try:
            return self.plan.payload_ref(
                physical,
                StoragePlane.SCRATCHPAD,
                component,
                placement,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: physical payload is absent "
                "from final Alpha storage"
            ) from error

    def _serialized_artifact(
        self,
        use: AlphaSerializedUse,
        *,
        logical: ValueRef,
    ) -> SerializedTensorRef:
        if use.logical_tensor != logical:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: serialized Alpha GEMM "
                "artifact names the wrong logical tensor"
            )
        try:
            return self.plan.serialized_ref(
                logical,
                StoragePlane.SCRATCHPAD,
                use.artifact,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{use.operation.name}.{use.field}: serialized Alpha GEMM "
                "artifact is absent from the final plan"
            ) from error

    def _output_scratchpad_view(
        self,
        owner: Node,
        logical: ValueRef,
        physical: PhysicalTensorId,
    ) -> AlphaPlacedView:
        """Return the record writer's view of its output backing.

        A direct resident writer receives the backing base plus concat slot
        scalars.  Offset views are for later consumers.  Shuffle is a
        specialized two-backing writer and is finalized by its own builder.
        """
        write = self.prepared.resident_write_for(owner)
        if write is None:
            view = self.resolver.view_for(logical, StoragePlane.SCRATCHPAD)
            if view is None:
                raise AlphaFinalizationError(
                    f"{owner.name}: GEMM output has no scratchpad view"
                )
            return view
        if write.logical_tensor != logical or write.physical_tensor != physical:
            raise AlphaFinalizationError(
                f"{owner.name}: resident GEMM output binding is stale"
            )
        self.resolver._validate_resident(write, physical)
        return AlphaPlacedView.from_placement(
            self.resolver._scratchpad_placement(physical),
            component="primary",
        )

    def _build_gemm(self, binding: AlphaGemmStorageBinding) -> None:
        if not isinstance(binding, AlphaGemmStorageBinding):
            raise TypeError("GEMM bindings must be AlphaGemmStorageBinding values")
        constraint = self.prepared.constraint_for(binding.owner)
        if (
            constraint is None
            or constraint.kind != "gemm"
            or constraint.implementation is not binding.implementation
            or constraint.output != binding.output
        ):
            raise AlphaFinalizationError(
                f"{binding.owner.name}: frozen GEMM binding has no matching "
                "operation constraint"
            )
        owner = binding.owner
        head = binding.implementation
        terminal = constraint.terminal

        expected = (
            (binding.input, binding.input_physical, "input"),
            (binding.weight, binding.weight_physical, "weight"),
            (binding.output, binding.output_physical, "output"),
        )
        for logical, physical, label in expected:
            try:
                actual = self.plan.physical_id(
                    StoragePlane.SCRATCHPAD,
                    logical,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaFinalizationError(
                    f"{owner.name}: GEMM {label} is absent from final storage"
                ) from error
            if actual != physical:
                raise AlphaFinalizationError(
                    f"{owner.name}: frozen GEMM {label} physical ID is stale"
                )

        self._add_logical(
            owner,
            head,
            "input",
            binding.input,
            expected_scratchpad=binding.input_physical,
        )
        self._add_logical(
            owner,
            head,
            "weight",
            binding.weight,
            expected_scratchpad=binding.weight_physical,
            expected_component=binding.weight_component,
        )

        if binding.weight_scale is not None:
            input_scale_use = self._serialized_use(
                owner,
                "input_vector_scale_payload",
            )
            weight_scale_use = self._serialized_use(
                owner,
                "weight_vector_scale_payload",
            )
            input_scale = input_scale_use.logical_tensor
            weight_scale = binding.weight_scale
            if weight_scale_use.logical_tensor != weight_scale:
                raise AlphaFinalizationError(
                    f"{owner.name}: GEMM weight-scale artifacts disagree"
                )
            self._add_logical(
                owner,
                head,
                "input_scale",
                input_scale,
                expected_scratchpad=binding.input_physical,
                expected_component="fine_scale",
            )
            self._add_logical(
                owner,
                head,
                "weight_scale",
                weight_scale,
                expected_scratchpad=binding.weight_physical,
                expected_component="fine_scale",
            )
            self._add(
                owner,
                terminal,
                "input_vector_scale_payload",
                self._serialized_artifact(
                    input_scale_use,
                    logical=input_scale,
                ),
            )
            self._add(
                owner,
                terminal,
                "weight_vector_scale_payload",
                self._serialized_artifact(
                    weight_scale_use,
                    logical=weight_scale,
                ),
            )
            self._add(
                owner,
                terminal,
                "weight_scale_valid_output_channels",
                binding.weight_scale_valid_output_channels,
            )

        self._add(
            owner,
            terminal,
            "output_payload",
            binding.output,
            memory=self.resolver.view_for(binding.output, StoragePlane.DRAM),
            scratchpad=self._output_scratchpad_view(
                owner,
                binding.output,
                binding.output_physical,
            ),
        )
        weight_use = self._payload_use(owner, "weight_payload")
        self._add(
            owner,
            terminal,
            "weight_payload",
            self._physical_payload(
                weight_use,
                logical=binding.weight,
                physical=binding.weight_physical,
                component=binding.weight_component,
            ),
        )

        if binding.materialized_input_physical is not None:
            use = self._payload_use(head, "materialized_input_payload")
            self._add(
                owner,
                head,
                "materialized_input_payload",
                self._physical_payload(
                    use,
                    logical=binding.input,
                    physical=binding.materialized_input_physical,
                    component=binding.materialized_input_component,
                ),
            )
            self._add(
                owner,
                head,
                "materialized_input_padding",
                binding.materialized_input_padding,
            )
            if binding.materialized_input_valid_channels is not None:
                self._add(
                    owner,
                    head,
                    "materialized_input_valid_channels",
                    binding.materialized_input_valid_channels,
                )
                self._add(
                    owner,
                    head,
                    "materialized_input_resident_channels",
                    binding.materialized_input_resident_channels,
                )

        scalar_fields = (
            ("pad_h_up", binding.pad_h_up),
            ("pad_h_down", binding.pad_h_down),
            ("pad_w_left", binding.pad_w_left),
            ("pad_w_right", binding.pad_w_right),
            ("padding_c", binding.padding_c),
            ("padding_k", binding.padding_k),
            ("weight_valid_input_channels", binding.valid_input_channels),
            ("weight_valid_output_channels", binding.valid_output_channels),
        )
        for field, value in scalar_fields:
            self._add(owner, terminal, field, value)

    @staticmethod
    def _coverage_error(label: str, expected, actual) -> None:
        missing = sorted(
            f"{operation.name}.{field}"
            for operation, field in expected - actual
        )
        unexpected = sorted(
            f"{operation.name}.{field}"
            for operation, field in actual - expected
        )
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AlphaFinalizationError(
            f"Alpha GEMM {label} coverage differs: " + "; ".join(details)
        )

    def _validate_use_coverage(self) -> None:
        payload_sites = {
            site
            for site, use in self._payload_by_site.items()
            if use.field in _GEMM_PAYLOAD_FIELDS
        }
        if payload_sites != self._consumed_payload_sites:
            self._coverage_error(
                "payload-use",
                payload_sites,
                self._consumed_payload_sites,
            )
        serialized_sites = {
            site
            for site, use in self._serialized_by_site.items()
            if use.field in _GEMM_SERIALIZED_FIELDS
        }
        if serialized_sites != self._consumed_serialized_sites:
            self._coverage_error(
                "serialized-use",
                serialized_sites,
                self._consumed_serialized_sites,
            )

    def build(self) -> tuple[AlphaRecordBinding, ...]:
        gemms = tuple(self.prepared.gemm_bindings)
        owners = tuple(binding.owner for binding in gemms)
        duplicates = _duplicates(owners)
        if duplicates:
            raise AlphaFinalizationError(
                "Alpha GEMM bindings repeat owners: "
                + ", ".join(owner.name for owner in duplicates)
            )
        constrained = tuple(
            constraint.owner
            for constraint in self.prepared.operation_constraints
            if constraint.kind == "gemm"
        )
        if owners != constrained:
            expected = set(constrained)
            actual = set(owners)
            missing = sorted(owner.name for owner in expected - actual)
            unexpected = sorted(owner.name for owner in actual - expected)
            raise AlphaFinalizationError(
                "Alpha GEMM owner coverage differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for binding in gemms:
            self._build_gemm(binding)
        self._validate_use_coverage()
        return tuple(self.bindings)


def build_alpha_gemm_bindings(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
    dram_placements: Iterable[AlphaDramPlacement] = (),
) -> tuple[AlphaRecordBinding, ...]:
    """Finalize fork-compatible GEMM storage and extent record fields."""
    return _AlphaGemmBindingBuilder(
        prepared,
        scratchpad_placements,
        dram_placements,
    ).build()


def build_alpha_input_view_bindings(
    prepared: AlphaPreparedStorage,
) -> tuple[AlphaRecordBinding, ...]:
    """Project frozen split/resident input views onto exact record sites.

    The view geometry is decided by storage preparation.  This builder only
    applies the two operation-specific split-index overrides present in the
    fork and assigns fields that do not belong to an ordinary NMP descriptor.
    """
    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha input-view bindings need prepared storage")
    views = tuple(prepared.nmp_input_views)
    if not all(isinstance(view, AlphaNmpInputView) for view in views):
        raise TypeError("NMP input views must be AlphaNmpInputView values")
    owners = tuple(view.owner for view in views)
    duplicates = _duplicates(owners)
    if duplicates:
        raise AlphaFinalizationError(
            "Alpha NMP input views repeat owners: "
            + ", ".join(owner.name for owner in duplicates)
        )

    bindings = []
    sites = set()

    def add(owner: Node, member: Node, field: str, value) -> None:
        site = AlphaRecordSite(owner=owner, member=member, field=field)
        if site in sites:
            raise AlphaFinalizationError(
                f"{site.label}: Alpha input-view binding was emitted twice"
            )
        sites.add(site)
        bindings.append(AlphaRecordBinding(site=site, value=value))

    ordinary = {
        binding.owner: binding for binding in prepared.nmp_bindings
    }
    if len(ordinary) != len(prepared.nmp_bindings):
        raise AlphaFinalizationError("ordinary Alpha NMP bindings repeat owners")

    for view in views:
        constraint = prepared.constraint_for(view.owner)
        if constraint is None:
            raise AlphaFinalizationError(
                f"{view.owner.name}: Alpha NMP input view has no constraint"
            )
        matching_operands = tuple(
            operand
            for operand in constraint.operands
            if operand.logical_tensor == view.logical_tensor
        )
        if len(matching_operands) != 1:
            raise AlphaFinalizationError(
                f"{view.owner.name}: Alpha NMP input view does not identify "
                "one operation operand"
            )
        split_index = view.split_index
        if view.source_kind == "channel_split":
            if constraint.kind == "gemm":
                split_index = 0
            elif constraint.kind == "channel_shuffle":
                split_index = 1

        for field, value in (
            ("is_from_split", True),
            ("split_channel_idx", view.channel_index),
            ("split_cur_size", view.current_channels),
            ("split_idx", split_index),
            ("split_scale", view.scale),
            ("split_total_size", view.total_channels),
        ):
            add(view.owner, constraint.terminal, field, value)

        if constraint.kind == "channel_shuffle":
            if view.pitch_gap_words:
                add(
                    view.owner,
                    constraint.terminal,
                    "nmp_c2",
                    view.pitch_gap_words,
                )
        elif constraint.kind == "gemm":
            if (
                view.source_kind == "resident"
                and view.resident_group_count is not None
                and view.resident_group_count > 1
            ):
                add(
                    view.owner,
                    constraint.terminal,
                    "nmp_groups",
                    view.resident_group_count,
                )
                if view.uniform_group_channels is not None:
                    add(
                        view.owner,
                        constraint.terminal,
                        "nmp_channels_per_group",
                        view.uniform_group_channels,
                    )
        elif constraint.kind in {
            "depthwise_conv2d",
            "pool2d",
            "upsample_nearest2d",
        }:
            binding = ordinary.get(view.owner)
            if binding is None or binding.input_view != view:
                raise AlphaFinalizationError(
                    f"{view.owner.name}: ordinary NMP descriptor does not "
                    "consume its frozen input view"
                )
        else:
            raise AlphaFinalizationError(
                f"{view.owner.name}: unsupported Alpha input-view owner "
                f"kind {constraint.kind!r}"
            )
    return tuple(bindings)


class _AlphaNmpBindingBuilder:
    """Finalize ordinary NMP placements and their frozen scalar contract."""

    _KINDS = frozenset(
        {"depthwise_conv2d", "pool2d", "upsample_nearest2d"}
    )

    def __init__(
        self,
        prepared: AlphaPreparedStorage,
        scratchpad_placements: Iterable[AlphaPlacement],
        dram_placements: Iterable[AlphaDramPlacement],
    ) -> None:
        if not isinstance(prepared, AlphaPreparedStorage):
            raise TypeError("Alpha NMP bindings need prepared storage")
        self.prepared = prepared
        self.plan = prepared.plan
        self.resolver = _AlphaLogicalPlacementResolver(
            prepared,
            scratchpad_placements,
            dram_placements,
        )
        self.bindings: list[AlphaRecordBinding] = []
        self.sites: set[AlphaRecordSite] = set()

    def _add(
        self,
        owner: Node,
        member: Node,
        field: str,
        value,
        *,
        memory: Optional[AlphaPlacedView] = None,
        scratchpad: Optional[AlphaPlacedView] = None,
    ) -> None:
        site = AlphaRecordSite(owner=owner, member=member, field=field)
        if site in self.sites:
            raise AlphaFinalizationError(
                f"{site.label}: Alpha NMP binding was emitted twice"
            )
        self.sites.add(site)
        self.bindings.append(
            AlphaRecordBinding(
                site=site,
                value=value,
                memory=memory,
                scratchpad=scratchpad,
            )
        )

    def _add_logical(
        self,
        owner: Node,
        member: Node,
        field: str,
        logical: ValueRef,
        physical: PhysicalTensorId,
    ) -> None:
        scratchpad = self.resolver.view_for(
            logical, StoragePlane.SCRATCHPAD
        )
        if (
            scratchpad is None
            or scratchpad.physical_tensor != physical
            or scratchpad.component != "primary"
        ):
            raise AlphaFinalizationError(
                f"{logical.name}: {field} resolved to the wrong Alpha NMP "
                "scratchpad backing"
            )
        self._add(
            owner,
            member,
            field,
            logical,
            memory=self.resolver.view_for(logical, StoragePlane.DRAM),
            scratchpad=scratchpad,
        )

    def _output_view(
        self,
        owner: Node,
        logical: ValueRef,
        physical: PhysicalTensorId,
    ) -> AlphaPlacedView:
        write = self.prepared.resident_write_for(owner)
        if write is None:
            view = self.resolver.view_for(logical, StoragePlane.SCRATCHPAD)
            if view is None:
                raise AlphaFinalizationError(
                    f"{owner.name}: ordinary NMP output has no placement"
                )
            if view.physical_tensor != physical or view.component != "primary":
                raise AlphaFinalizationError(
                    f"{owner.name}: ordinary NMP output placement is stale"
                )
            return view
        if write.logical_tensor != logical or write.physical_tensor != physical:
            raise AlphaFinalizationError(
                f"{owner.name}: resident NMP output binding is stale"
            )
        self.resolver._validate_resident(write, physical)
        return AlphaPlacedView.from_placement(
            self.resolver._scratchpad_placement(physical),
            component="primary",
        )

    def _validate_physical(
        self,
        owner: Node,
        label: str,
        logical: ValueRef,
        expected: PhysicalTensorId,
    ) -> None:
        try:
            actual = self.plan.physical_id(
                StoragePlane.SCRATCHPAD, logical
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{owner.name}: Alpha NMP {label} is absent from final storage"
            ) from error
        if actual != expected:
            raise AlphaFinalizationError(
                f"{owner.name}: frozen Alpha NMP {label} physical ID is stale"
            )

    def _build_ordinary(self, binding: AlphaNmpStorageBinding) -> None:
        constraint = self.prepared.constraint_for(binding.owner)
        if (
            constraint is None
            or constraint.kind != binding.kind
            or constraint.kind not in self._KINDS
            or constraint.implementation is not binding.implementation
            or constraint.terminal is not binding.terminal
            or constraint.output != binding.output
        ):
            raise AlphaFinalizationError(
                f"{binding.owner.name}: frozen Alpha NMP binding has no "
                "matching operation constraint"
            )
        expected_view = self.prepared.nmp_input_view_for(binding.owner)
        if binding.input_view != expected_view:
            raise AlphaFinalizationError(
                f"{binding.owner.name}: frozen Alpha NMP input view is stale"
            )

        self._validate_physical(
            binding.owner,
            "input",
            binding.input,
            binding.input_physical,
        )
        self._validate_physical(
            binding.owner,
            "output",
            binding.output,
            binding.output_physical,
        )
        self._add_logical(
            binding.owner,
            binding.implementation,
            "input",
            binding.input,
            binding.input_physical,
        )
        if binding.weight is not None:
            self._validate_physical(
                binding.owner,
                "weight",
                binding.weight,
                binding.weight_physical,
            )
            self._add_logical(
                binding.owner,
                binding.implementation,
                "weight",
                binding.weight,
                binding.weight_physical,
            )
        self._add(
            binding.owner,
            binding.terminal,
            "output_payload",
            binding.output,
            memory=self.resolver.view_for(binding.output, StoragePlane.DRAM),
            scratchpad=self._output_view(
                binding.owner,
                binding.output,
                binding.output_physical,
            ),
        )

        if binding.pads is not None:
            for field, value in zip(
                ("pad_h_up", "pad_h_down", "pad_w_left", "pad_w_right"),
                binding.pads,
            ):
                self._add(binding.owner, binding.terminal, field, value)
        for field in ("padding_c", "padding_k"):
            value = getattr(binding, field)
            if value is not None:
                self._add(binding.owner, binding.terminal, field, value)
        for field in ("nmp_c2", "nmp_k2"):
            value = getattr(binding, field)
            if value:
                self._add(binding.owner, binding.terminal, field, value)
        for field in ("nmp_groups", "nmp_channels_per_group"):
            value = getattr(binding, field)
            if value is not None:
                self._add(binding.owner, binding.terminal, field, value)
        if binding.is_pool_1bank is not None:
            self._add(
                binding.owner,
                binding.terminal,
                "is_pool_1bank",
                binding.is_pool_1bank,
            )

    def _build_shuffle_inputs(self) -> None:
        for constraint in self.prepared.operation_constraints:
            if constraint.kind != "channel_shuffle":
                continue
            for field in ("input1", "input2"):
                operand = constraint.operand(field)
                if operand is None:
                    raise AlphaFinalizationError(
                        f"{constraint.owner.name}: Alpha shuffle {field} did "
                        "not resolve"
                    )
                try:
                    physical = self.plan.physical_id(
                        StoragePlane.SCRATCHPAD,
                        operand.logical_tensor,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise AlphaFinalizationError(
                        f"{constraint.owner.name}: Alpha shuffle {field} has "
                        "no final storage"
                    ) from error
                self._add_logical(
                    constraint.owner,
                    constraint.implementation,
                    field,
                    operand.logical_tensor,
                    physical,
                )

    def build(self) -> tuple[AlphaRecordBinding, ...]:
        values = tuple(self.prepared.nmp_bindings)
        if not all(isinstance(value, AlphaNmpStorageBinding) for value in values):
            raise TypeError(
                "ordinary NMP bindings must be AlphaNmpStorageBinding values"
            )
        owners = tuple(value.owner for value in values)
        duplicates = _duplicates(owners)
        if duplicates:
            raise AlphaFinalizationError(
                "ordinary Alpha NMP bindings repeat owners: "
                + ", ".join(owner.name for owner in duplicates)
            )
        expected = tuple(
            constraint.owner
            for constraint in self.prepared.operation_constraints
            if constraint.kind in self._KINDS
        )
        if owners != expected:
            raise AlphaFinalizationError(
                "ordinary Alpha NMP owner coverage differs: "
                f"expected={[owner.name for owner in expected]}, "
                f"actual={[owner.name for owner in owners]}"
            )
        for value in values:
            self._build_ordinary(value)
        self._build_shuffle_inputs()
        return tuple(self.bindings)


def build_alpha_nmp_bindings(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
    dram_placements: Iterable[AlphaDramPlacement] = (),
) -> tuple[AlphaRecordBinding, ...]:
    """Finalize DWC/pool/upsample fields and shuffle semantic inputs."""
    return _AlphaNmpBindingBuilder(
        prepared,
        scratchpad_placements,
        dram_placements,
    ).build()


def build_alpha_pvs_bindings(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
    dram_placements: Iterable[AlphaDramPlacement] = (),
) -> tuple[AlphaRecordBinding, ...]:
    """Bind terminal PVS fine-scale views and their serialized artifacts.

    PVS output storage is unit-independent: both GEMM and depthwise records
    expose the fine scale as a component of the output payload backing and
    carry a separate unplaced byte artifact for the converter.  Both facts
    were declared and frozen by storage preparation; this function only
    resolves the component's allocated base and assigns the two exact record
    sites.
    """
    if not isinstance(prepared, AlphaPreparedStorage):
        raise TypeError("Alpha PVS bindings need prepared storage")
    resolver = _AlphaLogicalPlacementResolver(
        prepared,
        scratchpad_placements,
        dram_placements,
    )
    uses = tuple(
        use
        for use in prepared.serialized_uses
        if use.field == "output_vector_scale_payload"
    )
    operations = tuple(use.operation for use in uses)
    duplicates = _duplicates(operations)
    if duplicates:
        raise AlphaFinalizationError(
            "Alpha PVS artifacts repeat owners: "
            + ", ".join(owner.name for owner in duplicates)
        )

    bindings = []
    sites = set()
    for use in uses:
        if not isinstance(use, AlphaSerializedUse):
            raise TypeError("PVS artifacts must be AlphaSerializedUse values")
        constraint = prepared.constraint_for(use.operation)
        if constraint is None:
            raise AlphaFinalizationError(
                f"{use.operation.name}: terminal PVS artifact has no "
                "operation constraint"
            )
        fine_scale = use.logical_tensor
        try:
            fine_physical = prepared.plan.physical_id(
                StoragePlane.SCRATCHPAD,
                fine_scale,
            )
            output_physical = prepared.plan.physical_id(
                StoragePlane.SCRATCHPAD,
                constraint.output,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{use.operation.name}: terminal PVS output is absent from "
                "final Alpha storage"
            ) from error
        fine_view = resolver.view_for(
            fine_scale,
            StoragePlane.SCRATCHPAD,
        )
        if (
            fine_view is None
            or fine_physical != output_physical
            or fine_view.physical_tensor != output_physical
            or fine_view.component != "fine_scale"
        ):
            raise AlphaFinalizationError(
                f"{use.operation.name}: terminal PVS fine scale is not the "
                "frozen output component"
            )
        try:
            artifact = prepared.plan.serialized_ref(
                fine_scale,
                StoragePlane.SCRATCHPAD,
                use.artifact,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{use.operation.name}: terminal PVS artifact is absent from "
                "the final plan"
            ) from error

        for field, value, scratchpad in (
            ("output_vector_scale", fine_scale, fine_view),
            ("output_vector_scale_payload", artifact, None),
        ):
            site = AlphaRecordSite(
                owner=use.operation,
                member=constraint.terminal,
                field=field,
            )
            if site in sites:
                raise AlphaFinalizationError(
                    f"{site.label}: Alpha PVS binding was emitted twice"
                )
            sites.add(site)
            bindings.append(
                AlphaRecordBinding(
                    site=site,
                    value=value,
                    scratchpad=scratchpad,
                )
            )
    return tuple(bindings)


@dataclass(frozen=True)
class AlphaFinalizedSerialization:
    """Complete immutable Alpha state accepted by a projection-only writer."""

    prepared: AlphaPreparedStorage
    scratchpad_placements: tuple[AlphaPlacement, ...]
    dram_placements: tuple[AlphaDramPlacement, ...]
    bindings: tuple[AlphaRecordBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, AlphaPreparedStorage):
            raise TypeError("final Alpha serialization needs prepared storage")
        scratchpad = tuple(self.scratchpad_placements)
        dram = tuple(self.dram_placements)
        bindings = tuple(self.bindings)
        object.__setattr__(self, "scratchpad_placements", scratchpad)
        object.__setattr__(self, "dram_placements", dram)
        object.__setattr__(self, "bindings", bindings)

        if not all(
            isinstance(item, AlphaPlacement) for item in scratchpad
        ):
            raise TypeError(
                "scratchpad placements must be AlphaPlacement values"
            )
        if not all(isinstance(item, AlphaDramPlacement) for item in dram):
            raise TypeError("DRAM placements must be AlphaDramPlacement values")
        if not all(isinstance(item, AlphaRecordBinding) for item in bindings):
            raise TypeError("Alpha bindings must be AlphaRecordBinding values")

        self._validate_prepared_state()
        self._validate_scratchpad_coverage()
        validate_alpha_dram_placements(self.prepared.plan, dram)
        self._validate_binding_sites()
        _validate_alpha_structural_binding_inventory(
            self.prepared,
            self.bindings,
        )
        self._validate_bindings()

    @property
    def execution_order(self):
        return self.prepared.execution_order

    @property
    def sites(self) -> tuple[AlphaRecordSite, ...]:
        return tuple(binding.site for binding in self.bindings)

    def _validate_prepared_state(self) -> None:
        prepared = self.prepared
        if prepared.provisional_plan.graph is not prepared.graph:
            raise AlphaFinalizationError(
                "provisional storage plan belongs to another graph"
            )
        if prepared.plan.graph is not prepared.graph:
            raise AlphaFinalizationError(
                "final storage plan belongs to another graph"
            )
        if prepared.plan.execution_order != prepared.execution_order:
            raise AlphaFinalizationError(
                "final storage plan does not carry the selected execution order"
            )
        for constraint in prepared.operation_constraints:
            if constraint.kind != "gemm":
                continue
            terminal = constraint.terminal
            if terminal.target is _ALPHA_PPU:
                ppu = terminal
            elif terminal.target is _ALPHA_PVS:
                ppu = get_arg_value(terminal, 0, "input", None)
            else:
                ppu = None
            if (
                not isinstance(ppu, Node)
                or ppu.target is not _ALPHA_PPU
                or (
                    get_arg_value(ppu, 0, "accumulator", None)
                    is not constraint.implementation
                )
                or "integer_accumulator" not in ppu.kwargs
                or ppu.kwargs["integer_accumulator"] is not True
            ):
                raise AlphaFinalizationError(
                    f"{constraint.owner.name}: Alpha GEMM PPU must explicitly "
                    "declare integer_accumulator=True"
                )

    def _validate_scratchpad_coverage(self) -> None:
        expected = tuple(self.prepared.resident_physical_ids)
        requested = tuple(
            request.physical_tensor
            for request in self.prepared.allocation_requests
        )
        actual = tuple(
            placement.physical_tensor
            for placement in self.scratchpad_placements
        )
        for label, physicals in (
            ("resident physical IDs", expected),
            ("allocation requests", requested),
            ("scratchpad placements", actual),
        ):
            duplicates = _duplicates(physicals)
            if duplicates:
                raise AlphaFinalizationError(
                    f"{label} repeat: "
                    + ", ".join(physical.name for physical in duplicates)
                )
        expected_set = set(expected)
        if set(requested) != expected_set:
            self._raise_physical_coverage(
                "allocation requests", expected_set, set(requested)
            )
        if set(actual) != expected_set:
            self._raise_physical_coverage(
                "scratchpad placements", expected_set, set(actual)
            )

        request_by_physical = {
            request.physical_tensor: request
            for request in self.prepared.allocation_requests
        }
        allocation_ids = tuple(
            placement.allocation_id for placement in self.scratchpad_placements
        )
        if _duplicates(allocation_ids):
            raise AlphaFinalizationError(
                "scratchpad placement allocation IDs are not unique"
            )
        for placement in self.scratchpad_placements:
            expected_request = request_by_physical[placement.physical_tensor]
            if placement.request != expected_request:
                raise AlphaFinalizationError(
                    f"{placement.physical_tensor.name}: placement request "
                    "differs from prepared storage"
                )

    @staticmethod
    def _raise_physical_coverage(label, expected, actual) -> None:
        missing = sorted(
            (physical.name for physical in expected - actual),
        )
        unexpected = sorted(
            (physical.name for physical in actual - expected),
        )
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise AlphaFinalizationError(
            f"{label} do not match resident physical IDs: " + "; ".join(details)
        )

    def _validate_binding_sites(self) -> None:
        duplicate_sites = _duplicates(self.sites)
        if duplicate_sites:
            raise AlphaFinalizationError(
                "Alpha serialization sites repeat: "
                + ", ".join(site.label for site in duplicate_sites)
            )

        ordered_nodes = set(self.execution_order.nodes)
        model = self.prepared.graph.model
        for site in self.sites:
            if site.owner not in ordered_nodes:
                raise AlphaFinalizationError(
                    f"{site.label}: owner is outside the selected execution "
                    "order"
                )
            member = site.member
            if member is None or member is site.owner:
                continue
            if site.owner.op != "call_module":
                raise AlphaFinalizationError(
                    f"{site.label}: a non-module owner cannot name an inner "
                    "member"
                )
            try:
                submodule = model.get_submodule(str(site.owner.target))
            except (AttributeError, KeyError) as error:
                raise AlphaFinalizationError(
                    f"{site.label}: fused owner has no graph submodule"
                ) from error
            if member.graph is not getattr(submodule, "graph", None):
                raise AlphaFinalizationError(
                    f"{site.label}: member is outside the owner's fused graph"
                )

    def _placements_by_physical(self):
        return {
            **{
                placement.physical_tensor: placement
                for placement in self.scratchpad_placements
            },
            **{
                placement.physical_tensor: placement
                for placement in self.dram_placements
            },
        }

    def _validate_view(self, site, view: AlphaPlacedView) -> None:
        placements = self._placements_by_physical()
        base = placements.get(view.physical_tensor)
        if base is None:
            raise AlphaFinalizationError(
                f"{site.label}: view has no finalized base placement "
                f"for {view.physical_tensor.name}"
            )
        if (
            view.allocation_id != base.allocation_id
            or view.memory_space != base.memory_space
            or view.memory_instance != base.memory_instance
        ):
            raise AlphaFinalizationError(
                f"{site.label}: view disagrees with its base allocation"
            )
        if view.start < base.start or view.end > base.end:
            raise AlphaFinalizationError(
                f"{site.label}: view lies outside its base allocation"
            )
        if view.plane == StoragePlane.SCRATCHPAD.value:
            if not isinstance(base, AlphaPlacement):
                raise AlphaFinalizationError(
                    f"{site.label}: scratchpad view has no scratchpad base"
                )
            try:
                expected_address = alpha_place(
                    GLOBAL.instance(base.memory_instance),
                    view.start,
                    cluster=base.cluster,
                    operand_bits=base.operand_bits,
                ).encode()
            except ValueError as error:
                raise AlphaFinalizationError(
                    f"{site.label}: scratchpad view cannot be encoded"
                ) from error
            if view.address != expected_address:
                raise AlphaFinalizationError(
                    f"{site.label}: scratchpad view address is not the "
                    "encoding of its byte offset"
                )
        if view.plane == StoragePlane.DRAM.value and view.address != view.start:
            raise AlphaFinalizationError(
                f"{site.label}: DRAM view address must equal its byte offset"
            )
        plan = self.prepared.plan
        try:
            plan.storage_key(view.physical_tensor, view.plane)
            plan.physical_component(
                view.physical_tensor,
                view.plane,
                view.component,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{site.label}: view is not represented by the storage plan"
            ) from error

    def _expected_logical_view_identity(
        self,
        binding: AlphaRecordBinding,
        view: AlphaPlacedView,
    ) -> tuple[PhysicalTensorId, str]:
        """Return the frozen physical/component identity for one binding.

        An exact ``AlphaPayloadUse`` is occurrence-specific: a shuffle-copy
        port may select the auxiliary backing, and a zero-copy split input
        names the primary anchor of a two-backed semantic value.  Logical
        fields without such a declaration follow the storage relation chain.
        """
        plan = self.prepared.plan
        logical = binding.value
        if view.plane == StoragePlane.SCRATCHPAD.value:
            payload_uses = tuple(
                use
                for use in self.prepared.payload_uses
                if use.operation is binding.site.member
                and use.field == binding.site.field
                and use.logical_tensor == logical
            )
            if len(payload_uses) > 1:
                raise AlphaFinalizationError(
                    f"{binding.site.label}: logical view has ambiguous frozen "
                    "payload uses"
                )
            if payload_uses:
                use = payload_uses[0]
                return use.physical_tensor, use.component

        try:
            physical = plan.physical_id(view.plane, logical)
            chain = plan.trace_storage_chain(logical, view.plane)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{binding.site.label}: logical view has no finalized storage "
                "identity"
            ) from error
        components = tuple(
            relation.component
            for item in chain
            for relation in (plan.relation(view.plane, item),)
            if relation.kind is StorageRelationKind.COMPONENT
        )
        if len(components) > 1:
            raise AlphaFinalizationError(
                f"{binding.site.label}: logical view crosses nested physical "
                "components"
            )
        return physical, components[0] if components else "primary"

    def _validate_logical_view_binding(
        self,
        binding: AlphaRecordBinding,
        view: AlphaPlacedView,
    ) -> None:
        expected_physical, expected_component = (
            self._expected_logical_view_identity(binding, view)
        )
        if (
            view.physical_tensor != expected_physical
            or view.component != expected_component
        ):
            raise AlphaFinalizationError(
                f"{binding.site.label}: logical tensor is attached to the "
                "wrong finalized physical backing or component"
            )
        if view.plane != StoragePlane.DRAM.value:
            return
        try:
            scratchpad_physical = self.prepared.plan.physical_id(
                StoragePlane.SCRATCHPAD,
                binding.value,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaFinalizationError(
                f"{binding.site.label}: DRAM view has no scratchpad target"
            ) from error
        transfers = tuple(
            transfer
            for transfer in self.prepared.plan.transfers
            if transfer.logical_tensor == binding.value
            and transfer.source == expected_physical
            and transfer.destination == scratchpad_physical
            and transfer.kind == "load"
        )
        if len(transfers) != 1:
            raise AlphaFinalizationError(
                f"{binding.site.label}: DRAM view has no unique exact Alpha "
                "load"
            )

    def _validate_bindings(self) -> None:
        plan = self.prepared.plan
        placements = self._placements_by_physical()
        for binding in self.bindings:
            site = binding.site
            if isinstance(binding.value, ValueRef):
                try:
                    plan.ref(binding.value)
                except (KeyError, ValueError) as error:
                    raise AlphaFinalizationError(
                        f"{site.label}: logical tensor is absent from the plan"
                    ) from error
                for view in (binding.memory, binding.scratchpad):
                    if view is not None:
                        self._validate_view(site, view)
                        self._validate_logical_view_binding(binding, view)
                continue

            if isinstance(binding.value, PhysicalPayloadRef):
                payload = binding.value
                base = placements.get(payload.physical_tensor)
                if base is None:
                    raise AlphaFinalizationError(
                        f"{site.label}: physical payload has no base placement"
                    )
                segment_fields = (
                    "allocation_id",
                    "memory_space",
                    "start",
                    "end",
                    "address",
                    "memory_instance",
                )
                if any(
                    getattr(payload.segment, field, None)
                    != getattr(base, field, None)
                    for field in segment_fields
                ):
                    raise AlphaFinalizationError(
                        f"{site.label}: physical payload is not the full base "
                        "placement"
                    )
                try:
                    expected = plan.physical_component(
                        payload.physical_tensor,
                        payload.physical_tensor.plane,
                        payload.component,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise AlphaFinalizationError(
                        f"{site.label}: physical payload is absent from the "
                        "plan"
                    ) from error
                if payload.representation is not expected:
                    raise AlphaFinalizationError(
                        f"{site.label}: physical payload representation is not "
                        "the finalized plan component"
                    )
                continue

            if isinstance(binding.value, SerializedTensorRef):
                artifact = binding.value.artifact
                plane = artifact.physical_tensor.plane
                expected = plan.serialized_artifacts.get(plane, {}).get(
                    (artifact.physical_tensor, artifact.name)
                )
                if artifact is not expected:
                    raise AlphaFinalizationError(
                        f"{site.label}: serialized artifact is not from the "
                        "finalized plan"
                    )

    def placement_for(self, physical: PhysicalTensorId):
        """Return ``physical``'s base allocation or raise ``KeyError``."""
        if not isinstance(physical, PhysicalTensorId):
            raise TypeError("placement lookup needs a physical tensor ID")
        matches = tuple(
            placement
            for placement in (
                *self.scratchpad_placements,
                *self.dram_placements,
            )
            if placement.physical_tensor == physical
        )
        if len(matches) != 1:
            raise KeyError(f"{physical.name}: no finalized Alpha placement")
        return matches[0]

    def binding_for(self, site: AlphaRecordSite) -> AlphaRecordBinding:
        """Return the unique binding at one exact site or raise ``KeyError``."""
        if not isinstance(site, AlphaRecordSite):
            raise TypeError("binding lookup needs an AlphaRecordSite")
        try:
            return next(
                binding for binding in self.bindings if binding.site == site
            )
        except StopIteration:
            raise KeyError(
                f"{site.label}: no finalized Alpha binding"
            ) from None

    def bindings_for_owner(self, owner: Node) -> tuple[AlphaRecordBinding, ...]:
        """Return bindings owned by one outer serialized operation, in order."""
        return tuple(
            binding for binding in self.bindings if binding.site.owner is owner
        )

    def bindings_for_member(
        self,
        owner: Node,
        member: Optional[Node],
    ) -> tuple[AlphaRecordBinding, ...]:
        """Return bindings for one exact standalone/fused record member."""
        return tuple(
            binding
            for binding in self.bindings
            if binding.site.owner is owner and binding.site.member is member
        )

    def require_exact_site_coverage(
        self,
        consumed_sites: Iterable[AlphaRecordSite],
    ) -> tuple[AlphaRecordBinding, ...]:
        """Require a serializer to consume every binding exactly once.

        The returned bindings follow the serializer's supplied order, making
        this useful both as a final coverage assertion and as an exact lookup.
        """
        consumed = tuple(consumed_sites)
        if not all(isinstance(site, AlphaRecordSite) for site in consumed):
            raise TypeError("consumed sites must be AlphaRecordSite values")
        duplicates = _duplicates(consumed)
        if duplicates:
            raise AlphaFinalizationError(
                "serializer consumed Alpha sites more than once: "
                + ", ".join(site.label for site in duplicates)
            )
        expected_set = set(self.sites)
        actual_set = set(consumed)
        missing = tuple(site for site in self.sites if site not in actual_set)
        unexpected = tuple(
            site for site in consumed if site not in expected_set
        )
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    "missing " + ", ".join(site.label for site in missing)
                )
            if unexpected:
                details.append(
                    "unexpected "
                    + ", ".join(site.label for site in unexpected)
                )
            raise AlphaFinalizationError(
                "serializer site coverage differs from finalized state: "
                + "; ".join(details)
            )
        return tuple(self.binding_for(site) for site in consumed)


def build_alpha_finalized_serialization(
    prepared: AlphaPreparedStorage,
    scratchpad_placements: Iterable[AlphaPlacement],
    dram_placements: Iterable[AlphaDramPlacement],
) -> AlphaFinalizedSerialization:
    """Compose Alpha's complete immutable flat-serialization handoff.

    Every binding family is invoked here, even when that family is empty for
    a particular graph.  Keeping the inventory in one target-owned function
    prevents a terminal caller from silently omitting one class of decided
    storage fields while still constructing an otherwise valid handoff.
    """
    scratchpad = tuple(scratchpad_placements)
    dram = tuple(dram_placements)
    bindings = (
        *build_alpha_model_bindings(prepared, scratchpad, dram),
        *build_alpha_structural_bindings(prepared, scratchpad, dram),
        *build_alpha_gemm_bindings(prepared, scratchpad, dram),
        *build_alpha_input_view_bindings(prepared),
        *build_alpha_nmp_bindings(prepared, scratchpad, dram),
        *build_alpha_pvs_bindings(prepared, scratchpad, dram),
        *build_alpha_resident_bindings(prepared),
        *build_alpha_shuffle_bindings(prepared, scratchpad),
    )
    return AlphaFinalizedSerialization(
        prepared=prepared,
        scratchpad_placements=scratchpad,
        dram_placements=dram,
        bindings=bindings,
    )


__all__ = [
    "AlphaFinalizationError",
    "AlphaFinalizedSerialization",
    "AlphaPlacedView",
    "AlphaRecordBinding",
    "AlphaRecordSite",
    "build_alpha_finalized_serialization",
    "build_alpha_gemm_bindings",
    "build_alpha_input_view_bindings",
    "build_alpha_model_bindings",
    "build_alpha_nmp_bindings",
    "build_alpha_pvs_bindings",
    "build_alpha_resident_bindings",
    "build_alpha_shuffle_bindings",
    "build_alpha_structural_bindings",
]
