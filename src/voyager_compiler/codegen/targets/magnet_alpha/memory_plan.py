# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Frozen records produced by Magnet Alpha storage-policy analysis.

This module intentionally contains no graph recognition and no allocation
algorithm.  :mod:`.storage` freezes the final FX facts into these records;
:mod:`.allocation` consumes the resulting requests, and serialization may
only project their eventual placements.
"""

from dataclasses import dataclass
from typing import Optional

from torch.fx import Node

from scheduler.types import LocalScheduleResult, OperationWorkload
from voyager_compiler.codegen.execution_order import ExecutionOrder
from voyager_compiler.codegen.storage import (
    PhysicalTensorId,
    StorageGraph,
    StoragePlan,
    ValueRef,
    ViewTransform,
)
from voyager_compiler.codegen.targets.magnet_alpha.allocation import (
    ALPHA_BANK_CLUSTERS,
    AlphaAllocationRequest,
)
from voyager_compiler.hardware_config import TensorRole


class AlphaStoragePlanError(RuntimeError):
    """The final Alpha graph has no conservative physical interpretation."""


ALPHA_COPY_PORT_TRANSFORM_KINDS = frozenset(
    {
        "alpha_shuffle_copy_destination",
        "alpha_shuffle_copy_source",
    }
)


@dataclass(frozen=True)
class AlphaOperandBinding:
    """One final FX tensor consumed through an Alpha hardware port."""

    port: str
    logical_tensor: ValueRef
    role: TensorRole
    bits: int
    compute_unit: str

    def __post_init__(self):
        object.__setattr__(self, "role", TensorRole(self.role))
        if not self.port:
            raise ValueError("an Alpha operand binding needs a port")
        if self.bits not in {4, 8}:
            raise ValueError("an Alpha resident operand must be 4 or 8 bits")


@dataclass(frozen=True)
class AlphaOperationConstraint:
    """Authoritative execution owner plus its resolved physical operands."""

    owner: Node
    implementation: Node
    terminal: Node
    unit: str
    workload: OperationWorkload
    schedule: LocalScheduleResult
    operands: tuple[AlphaOperandBinding, ...]
    output: ValueRef
    output_component: str
    output_bits: int

    def __post_init__(self):
        if not all(
            isinstance(node, Node)
            for node in (self.owner, self.implementation, self.terminal)
        ):
            raise TypeError(
                "Alpha operation owner, implementation, and terminal must "
                "be FX nodes"
            )
        if not isinstance(self.schedule, LocalScheduleResult):
            raise TypeError(
                "an Alpha operation constraint needs a LocalScheduleResult"
            )

    @property
    def kind(self) -> str:
        return self.workload.operation.kind

    def operands_for_port(self, port: str) -> tuple[AlphaOperandBinding, ...]:
        return tuple(operand for operand in self.operands if operand.port == port)

    def operand(self, port: str) -> Optional[AlphaOperandBinding]:
        operands = self.operands_for_port(port)
        if len(operands) > 1:
            raise AlphaStoragePlanError(
                f"{self.owner.name}: multiple Alpha operands named {port!r}"
            )
        return operands[0] if operands else None


@dataclass(frozen=True)
class AlphaClusterConstraint:
    """One target-decided bank width bound to finalized physical storage."""

    physical_tensor: PhysicalTensorId
    cluster: int

    def __post_init__(self):
        if not isinstance(self.physical_tensor, PhysicalTensorId):
            raise TypeError(
                "an Alpha cluster constraint needs a physical tensor ID"
            )
        if self.physical_tensor.plane != "scratchpad":
            raise ValueError(
                f"{self.physical_tensor.name}: Alpha cluster constraints "
                "must name scratchpad storage"
            )
        if (
            isinstance(self.cluster, bool)
            or not isinstance(self.cluster, int)
            or self.cluster not in ALPHA_BANK_CLUSTERS
        ):
            raise ValueError(
                f"{self.physical_tensor.name}: unsupported Alpha bank "
                f"cluster {self.cluster!r}"
            )


@dataclass(frozen=True)
class AlphaEgressBoundary:
    """One fully proved terminal Alpha-to-host layout suffix.

    The final FX graph remains the semantic instruction carrier.  This record
    only freezes which executable owners feed the suffix and which graph nodes
    make up its address-only and host-layout portions; it neither allocates
    storage nor invents a serializer-side graph walk.
    """

    owners: frozenset[Node]
    source: ValueRef
    dequantize: Node
    padding_slice: Node
    shared_views: tuple[Node, ...]
    host_layouts: tuple[Node, ...]
    output: Node

    def __post_init__(self):
        owners = frozenset(self.owners)
        if not owners or any(not isinstance(owner, Node) for owner in owners):
            raise TypeError(
                "an Alpha egress boundary needs executable FX owners"
            )
        object.__setattr__(self, "owners", owners)
        if not isinstance(self.source, ValueRef):
            raise TypeError("an Alpha egress boundary source must be a ValueRef")
        if not all(
            isinstance(node, Node)
            for node in (
                self.dequantize,
                self.padding_slice,
                *self.shared_views,
                *self.host_layouts,
                self.output,
            )
        ):
            raise TypeError("Alpha egress boundary members must be FX nodes")
        object.__setattr__(self, "shared_views", tuple(self.shared_views))
        object.__setattr__(self, "host_layouts", tuple(self.host_layouts))
        if len(set(self.storage_path)) != len(self.storage_path):
            raise ValueError("Alpha egress storage paths must not repeat nodes")

    @property
    def storage_path(self) -> tuple[Node, ...]:
        return (
            self.dequantize,
            self.padding_slice,
            *self.shared_views,
            *self.host_layouts,
        )

    @property
    def address_only_nodes(self) -> tuple[Node, ...]:
        return (self.padding_slice, *self.shared_views)

    @property
    def boundary_nodes(self) -> tuple[Node, ...]:
        """Ordinary records accepted only as this complete boundary."""
        return (self.dequantize, *self.host_layouts)


@dataclass(frozen=True)
class AlphaGemmStorageBinding:
    """One finalized Alpha matrix record's storage and padding contract.

    All logical, resident, and physical extents are kept distinct.  The
    serializer consumes these frozen facts; it must not recover valid extents
    from mutable FX metadata or infer padding from payload contents.
    """

    owner: Node
    implementation: Node
    input: ValueRef
    weight: ValueRef
    output: ValueRef
    input_physical: PhysicalTensorId
    weight_physical: PhysicalTensorId
    output_physical: PhysicalTensorId
    weight_component: str
    input_bits: int
    weight_bits: int
    output_bits: int
    valid_input_channels: int
    resident_input_channels: int
    physical_weight_input_channels: int
    valid_output_channels: int
    resident_output_channels: int
    physical_weight_output_channels: int
    pad_h_up: int
    pad_h_down: int
    pad_w_left: int
    pad_w_right: int
    padding_c: int
    padding_k: int
    materialized_input_physical: Optional[PhysicalTensorId] = None
    materialized_input_component: Optional[str] = None
    materialized_input_shape: Optional[tuple[int, ...]] = None
    materialized_input_padding: Optional[tuple[int, int, int, int]] = None
    materialized_input_valid_channels: Optional[int] = None
    materialized_input_resident_channels: Optional[int] = None
    weight_scale: Optional[ValueRef] = None
    weight_scale_physical: Optional[PhysicalTensorId] = None
    weight_scale_valid_output_channels: Optional[int] = None
    weight_scale_encoded_output_channels: Optional[int] = None

    def __post_init__(self):
        for name in ("owner", "implementation"):
            if not isinstance(getattr(self, name), Node):
                raise TypeError(f"Alpha GEMM {name} must be an FX node")
        for name in ("input", "weight", "output"):
            if not isinstance(getattr(self, name), ValueRef):
                raise TypeError(f"Alpha GEMM {name} must be a ValueRef")
        for name in (
            "input_physical",
            "weight_physical",
            "output_physical",
        ):
            physical = getattr(self, name)
            if not isinstance(physical, PhysicalTensorId):
                raise TypeError(f"Alpha GEMM {name} must be physical storage")
            if physical.plane != "scratchpad":
                raise ValueError(f"Alpha GEMM {name} must be in scratchpad")
        if not self.weight_component:
            raise ValueError("Alpha GEMM weight component must not be empty")
        for name in ("input_bits", "weight_bits", "output_bits"):
            if getattr(self, name) not in {4, 8}:
                raise ValueError(f"Alpha GEMM {name} must be 4 or 8")
        for prefix in ("input", "output"):
            valid = getattr(self, f"valid_{prefix}_channels")
            resident = getattr(self, f"resident_{prefix}_channels")
            physical = getattr(
                self, f"physical_weight_{prefix}_channels"
            )
            if not 0 < valid <= resident <= physical:
                raise ValueError(
                    f"Alpha GEMM {prefix} channels must satisfy "
                    "0 < valid <= resident <= physical"
                )
        for name in (
            "pad_h_up",
            "pad_h_down",
            "pad_w_left",
            "pad_w_right",
            "padding_c",
            "padding_k",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Alpha GEMM {name} must be nonnegative")

        materialized = (
            self.materialized_input_physical,
            self.materialized_input_component,
            self.materialized_input_shape,
            self.materialized_input_padding,
        )
        if any(value is not None for value in materialized):
            if any(value is None for value in materialized):
                raise ValueError(
                    "Alpha GEMM materialized input facts must be complete"
                )
            if self.materialized_input_physical != self.input_physical:
                raise ValueError(
                    "Alpha GEMM materialized input changed physical identity"
                )
            shape = tuple(self.materialized_input_shape)
            padding = tuple(self.materialized_input_padding)
            if not shape or any(dimension <= 0 for dimension in shape):
                raise ValueError(
                    "Alpha GEMM materialized input shape must be positive"
                )
            if len(padding) != 4 or any(value < 0 for value in padding):
                raise ValueError(
                    "Alpha GEMM materialized input padding must have four "
                    "nonnegative values"
                )
            object.__setattr__(self, "materialized_input_shape", shape)
            object.__setattr__(self, "materialized_input_padding", padding)
        elif any(
            value is not None
            for value in (
                self.materialized_input_valid_channels,
                self.materialized_input_resident_channels,
            )
        ):
            raise ValueError(
                "Alpha GEMM materialized channel facts need a payload"
            )
        channels = (
            self.materialized_input_valid_channels,
            self.materialized_input_resident_channels,
        )
        if any(value is not None for value in channels):
            if any(value is None for value in channels):
                raise ValueError(
                    "Alpha GEMM materialized channel facts must be paired"
                )
            if not 0 < channels[0] <= channels[1]:
                raise ValueError(
                    "Alpha GEMM materialized channels are inconsistent"
                )

        scale = (
            self.weight_scale,
            self.weight_scale_physical,
            self.weight_scale_valid_output_channels,
            self.weight_scale_encoded_output_channels,
        )
        if any(value is not None for value in scale):
            if any(value is None for value in scale):
                raise ValueError("Alpha GEMM weight-scale facts must be complete")
            if not isinstance(self.weight_scale, ValueRef):
                raise TypeError("Alpha GEMM weight scale must be a ValueRef")
            if not isinstance(self.weight_scale_physical, PhysicalTensorId):
                raise TypeError(
                    "Alpha GEMM weight scale must have physical storage"
                )
            if self.weight_scale_physical != self.weight_physical:
                raise ValueError(
                    "Alpha VSQ weight and fine scale must share one backing"
                )
            if not (
                0 < self.weight_scale_valid_output_channels
                <= self.weight_scale_encoded_output_channels
            ):
                raise ValueError(
                    "Alpha GEMM weight-scale output extents are inconsistent"
                )
            if (
                self.weight_scale_valid_output_channels
                != self.valid_output_channels
            ):
                raise ValueError(
                    "Alpha GEMM weight-scale valid output channels must "
                    "match the weight"
                )

    @property
    def issued_input_channels(self) -> int:
        return self.resident_input_channels + self.padding_c

    @property
    def issued_output_channels(self) -> int:
        return self.resident_output_channels + self.padding_k


@dataclass(frozen=True)
class AlphaResidentWriteBinding:
    """One finalized direct-write slot or pitch-only resident output.

    Storage recognition freezes these fields once.  Allocation and emission
    consume the decided physical identity and channel geometry without
    walking back through concatenations or split users.
    """

    owner: Node
    logical_tensor: ValueRef
    physical_tensor: PhysicalTensorId
    backing: ValueRef
    path: tuple[ValueRef, ...]
    slot_index: int
    channel_offset: int
    slot_channels: int
    logical_backing_channels: int
    resident_channels: int
    concat_channel_index: Optional[int]

    def __post_init__(self):
        path = tuple(self.path)
        if any(not isinstance(ref, ValueRef) for ref in path):
            raise TypeError("Alpha resident alias paths must contain ValueRefs")
        if len(set(path)) != len(path):
            raise ValueError("Alpha resident alias paths must not repeat values")
        object.__setattr__(self, "path", path)
        if self.slot_index < 0 or self.channel_offset < 0:
            raise ValueError("Alpha resident slot indices must not be negative")
        if min(
            self.slot_channels,
            self.logical_backing_channels,
            self.resident_channels,
        ) <= 0:
            raise ValueError("Alpha resident channel extents must be positive")
        if self.channel_offset + self.slot_channels > self.resident_channels:
            raise ValueError("Alpha resident slot exceeds its row pitch")
        if self.concat_channel_index not in {None, 0, 1}:
            raise ValueError("Alpha concat channel index must be 0, 1, or None")


@dataclass(frozen=True)
class AlphaNmpInputView:
    """One frozen logical channel view into a wider physical NMP row.

    The name follows the endpoint fields, but GEMM records consume the same
    split descriptor.  ``source_kind`` retains why the view exists so final
    ABI projection can apply the fork's operation-specific split-index and
    pitch rules without retracing storage relations.
    """

    owner: Node
    logical_tensor: ValueRef
    source_kind: str
    channel_index: int
    split_index: int
    current_channels: int
    total_channels: int
    pitch_gap_words: int
    resident_group_count: Optional[int] = None
    uniform_group_channels: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.owner, Node):
            raise TypeError("Alpha NMP input-view owner must be an FX node")
        if not isinstance(self.logical_tensor, ValueRef):
            raise TypeError("Alpha NMP input view must name a ValueRef")
        if self.source_kind not in {
            "channel_split",
            "shuffle_output",
            "resident",
        }:
            raise ValueError(
                f"unsupported Alpha NMP input-view source {self.source_kind!r}"
            )
        for name in (
            "channel_index",
            "split_index",
            "pitch_gap_words",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Alpha NMP input-view {name} must be nonnegative")
        if not 0 < self.current_channels <= self.total_channels:
            raise ValueError(
                "Alpha NMP input-view channels must satisfy "
                "0 < current <= total"
            )
        for name in ("resident_group_count", "uniform_group_channels"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"Alpha NMP input-view {name} must be positive when set"
                )
        if (
            self.uniform_group_channels is not None
            and self.resident_group_count is None
        ):
            raise ValueError(
                "Alpha uniform resident-group channels need a group count"
            )

    @property
    def scale(self) -> float:
        return self.total_channels / self.current_channels


@dataclass(frozen=True)
class AlphaNmpStorageBinding:
    """Final storage and address-generator facts for an ordinary NMP call."""

    owner: Node
    implementation: Node
    terminal: Node
    kind: str
    input: ValueRef
    output: ValueRef
    input_physical: PhysicalTensorId
    output_physical: PhysicalTensorId
    weight: Optional[ValueRef] = None
    weight_physical: Optional[PhysicalTensorId] = None
    input_view: Optional[AlphaNmpInputView] = None
    pads: Optional[tuple[int, int, int, int]] = None
    padding_c: Optional[int] = None
    padding_k: Optional[int] = None
    nmp_c2: int = 0
    nmp_k2: int = 0
    nmp_groups: Optional[int] = None
    nmp_channels_per_group: Optional[int] = None
    is_pool_1bank: Optional[bool] = None

    def __post_init__(self):
        if not all(
            isinstance(node, Node)
            for node in (self.owner, self.implementation, self.terminal)
        ):
            raise TypeError("Alpha NMP storage owners must be FX nodes")
        if self.kind not in {
            "depthwise_conv2d",
            "pool2d",
            "upsample_nearest2d",
        }:
            raise ValueError(f"unsupported ordinary Alpha NMP kind {self.kind!r}")
        for name in ("input", "output"):
            if not isinstance(getattr(self, name), ValueRef):
                raise TypeError(f"Alpha NMP {name} must be a ValueRef")
        for name in ("input_physical", "output_physical"):
            physical = getattr(self, name)
            if not isinstance(physical, PhysicalTensorId):
                raise TypeError(f"Alpha NMP {name} must have physical storage")
            if physical.plane != "scratchpad":
                raise ValueError(f"Alpha NMP {name} must be in scratchpad")
        if (self.weight is None) != (self.weight_physical is None):
            raise ValueError("Alpha NMP weight logical/physical facts must pair")
        if self.weight is not None:
            if not isinstance(self.weight, ValueRef):
                raise TypeError("Alpha NMP weight must be a ValueRef")
            if not isinstance(self.weight_physical, PhysicalTensorId):
                raise TypeError("Alpha NMP weight must have physical storage")
            if self.weight_physical.plane != "scratchpad":
                raise ValueError("Alpha NMP weight must be in scratchpad")
        if self.kind == "depthwise_conv2d" and self.weight is None:
            raise ValueError("Alpha depthwise storage needs a weight")
        if self.kind != "depthwise_conv2d" and self.weight is not None:
            raise ValueError("only Alpha depthwise storage has a weight")
        if self.input_view is not None:
            if not isinstance(self.input_view, AlphaNmpInputView):
                raise TypeError("Alpha NMP input view has the wrong type")
            if (
                self.input_view.owner is not self.owner
                or self.input_view.logical_tensor != self.input
            ):
                raise ValueError("Alpha NMP input view belongs to another call")
        if self.pads is not None:
            pads = tuple(self.pads)
            if len(pads) != 4 or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in pads
            ):
                raise ValueError("Alpha NMP padding must be four nonnegative ints")
            object.__setattr__(self, "pads", pads)
        for name in (
            "padding_c",
            "padding_k",
            "nmp_c2",
            "nmp_k2",
            "nmp_groups",
            "nmp_channels_per_group",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"Alpha NMP {name} must be nonnegative")
        if self.nmp_groups == 0 or self.nmp_channels_per_group == 0:
            raise ValueError("Alpha NMP group extents must be positive")
        if (
            self.nmp_channels_per_group is not None
            and self.nmp_groups is None
        ):
            raise ValueError("Alpha NMP channels per group need a group count")
        if self.is_pool_1bank is not None and not isinstance(
            self.is_pool_1bank, bool
        ):
            raise TypeError("Alpha pool one-bank fact must be bool")


@dataclass(frozen=True)
class AlphaPayloadUse:
    """One target protobuf role awaiting the allocator's placement.

    The physical identity and the view are frozen here.  Serialization may
    combine them with a finalized :class:`AlphaPlacement`, but must not chase
    FX relations or rediscover a channel offset after allocation.  This is
    also the sole record-site carrier of occurrence-specific physical ports.
    A shuffle copy's one-entry ports do not redefine its complete logical FX
    value, and a zero-copy shuffle split's input port names the primary anchor
    without pretending that both semantic outputs share that backing.
    """

    operation: Node
    field: str
    logical_tensor: ValueRef
    physical_tensor: PhysicalTensorId
    component: str = "primary"
    entry_offset: int = 0
    span_entries: Optional[int] = None
    row_pitch_entries: Optional[int] = None


@dataclass(frozen=True)
class AlphaSerializedUse:
    """One protobuf field bound to a plan-owned, unplaced tensor artifact."""

    operation: Node
    field: str
    logical_tensor: ValueRef
    artifact: str


@dataclass(frozen=True)
class AlphaLogicalViewBinding:
    """One logical Alpha value projected onto a decided physical backing.

    ``transforms`` preserves the complete semantic relation chain while the
    normalized offsets are the target address facts downstream consumers
    need.  An embedded component (for example a VSQ fine scale) names its
    component and intentionally has no invented byte offset into the payload.
    Instruction-local shuffle-copy ports belong in :class:`AlphaPayloadUse`,
    never in this semantic collection.
    """

    logical_tensor: ValueRef
    physical_tensor: PhysicalTensorId
    component: str = "primary"
    transforms: tuple[ViewTransform, ...] = ()
    entry_offset: int = 0
    byte_offset: Optional[int] = None
    span_entries: Optional[int] = None
    row_pitch_entries: Optional[int] = None

    def __post_init__(self):
        transforms = tuple(self.transforms)
        if not all(
            isinstance(transform, ViewTransform) for transform in transforms
        ):
            raise TypeError(
                "Alpha logical view transforms must be ViewTransform values"
            )
        if any(
            transform.kind in ALPHA_COPY_PORT_TRANSFORM_KINDS
            for transform in transforms
        ):
            raise ValueError(
                "Alpha shuffle-copy port views belong in AlphaPayloadUse, not "
                "AlphaLogicalViewBinding"
            )
        object.__setattr__(self, "transforms", transforms)
        if self.entry_offset < 0:
            raise ValueError("Alpha logical view entry offset must not be negative")
        if self.byte_offset is not None and self.byte_offset < 0:
            raise ValueError("Alpha logical view byte offset must not be negative")
        for name in ("span_entries", "row_pitch_entries"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Alpha logical view {name} must be positive")


@dataclass(frozen=True)
class AlphaShuffleMovement:
    """One exact NMP block movement within a two-backing shuffle."""

    copy: Node
    source_output: int
    source_entry: int
    destination_entry: int
    block_channels: int
    source_physical: PhysicalTensorId
    destination_physical: PhysicalTensorId
    source_view: ViewTransform
    destination_view: ViewTransform


@dataclass(frozen=True)
class AlphaShufflePlan:
    """The primary and auxiliary roots one channel shuffle spans."""

    shuffle: Node
    primary: PhysicalTensorId
    auxiliary: PhysicalTensorId
    copies: tuple[Node, ...]
    movements: tuple[AlphaShuffleMovement, ...]
    block_channels: int
    output_channels: int
    pitch_entries: int
    span_entries: int
    primary_entry_offset: int

    @property
    def copy_nmp_c2(self) -> int:
        return self.pitch_entries - 1

    @property
    def copy_nmp_k2(self) -> int:
        return self.pitch_entries - 1


@dataclass(frozen=True)
class AlphaPreparedStorage:
    """Immutable hand-off from Alpha policy to ordering/allocation/emission."""

    graph: StorageGraph
    provisional_plan: StoragePlan
    plan: StoragePlan
    execution_order: ExecutionOrder
    executable_owners: tuple[Node, ...]
    operation_constraints: tuple[AlphaOperationConstraint, ...]
    resident_physical_ids: tuple[PhysicalTensorId, ...]
    allocation_requests: tuple[AlphaAllocationRequest, ...]
    cluster_constraints: tuple[AlphaClusterConstraint, ...] = ()
    egress_boundaries: tuple[AlphaEgressBoundary, ...] = ()
    gemm_bindings: tuple[AlphaGemmStorageBinding, ...] = ()
    resident_writes: tuple[AlphaResidentWriteBinding, ...] = ()
    nmp_input_views: tuple[AlphaNmpInputView, ...] = ()
    nmp_bindings: tuple[AlphaNmpStorageBinding, ...] = ()
    shuffle_plans: tuple[AlphaShufflePlan, ...] = ()
    logical_views: tuple[AlphaLogicalViewBinding, ...] = ()
    payload_uses: tuple[AlphaPayloadUse, ...] = ()
    serialized_uses: tuple[AlphaSerializedUse, ...] = ()
    order_result: object = None

    def constraint_for(self, owner: Node) -> Optional[AlphaOperationConstraint]:
        return next(
            (
                constraint
                for constraint in self.operation_constraints
                if constraint.owner is owner
            ),
            None,
        )

    def shuffle_for(self, node: Node) -> Optional[AlphaShufflePlan]:
        return next(
            (plan for plan in self.shuffle_plans if plan.shuffle is node),
            None,
        )

    def cluster_for(self, physical: PhysicalTensorId) -> Optional[int]:
        matches = tuple(
            constraint.cluster
            for constraint in self.cluster_constraints
            if constraint.physical_tensor == physical
        )
        if len(matches) > 1:
            raise AlphaStoragePlanError(
                f"{physical.name}: multiple frozen Alpha cluster constraints"
            )
        return matches[0] if matches else None

    def gemm_for(self, owner: Node) -> Optional[AlphaGemmStorageBinding]:
        matches = tuple(
            binding for binding in self.gemm_bindings if binding.owner is owner
        )
        if len(matches) > 1:
            raise AlphaStoragePlanError(
                f"{owner.name}: multiple frozen Alpha GEMM bindings"
            )
        return matches[0] if matches else None

    def egress_for(self, node: Node) -> Optional[AlphaEgressBoundary]:
        matches = tuple(
            boundary
            for boundary in self.egress_boundaries
            if node in boundary.storage_path
        )
        if len(matches) > 1:
            raise AlphaStoragePlanError(
                f"{node.name}: multiple frozen Alpha egress boundaries"
            )
        return matches[0] if matches else None

    def resident_write_for(
        self,
        owner: Node,
    ) -> Optional[AlphaResidentWriteBinding]:
        matches = tuple(
            write for write in self.resident_writes if write.owner is owner
        )
        if len(matches) > 1:
            raise AlphaStoragePlanError(
                f"{owner.name}: multiple frozen Alpha resident writes"
            )
        return matches[0] if matches else None

    def nmp_input_view_for(self, owner: Node) -> Optional[AlphaNmpInputView]:
        matches = tuple(
            view for view in self.nmp_input_views if view.owner is owner
        )
        if len(matches) > 1:
            raise AlphaStoragePlanError(
                f"{owner.name}: multiple frozen Alpha NMP input views"
            )
        return matches[0] if matches else None

    def nmp_for(self, owner: Node) -> Optional[AlphaNmpStorageBinding]:
        matches = tuple(
            binding for binding in self.nmp_bindings if binding.owner is owner
        )
        if len(matches) > 1:
            raise AlphaStoragePlanError(
                f"{owner.name}: multiple frozen ordinary Alpha NMP bindings"
            )
        return matches[0] if matches else None

    def request_for(self, physical: PhysicalTensorId) -> AlphaAllocationRequest:
        try:
            return next(
                request
                for request in self.allocation_requests
                if request.physical_tensor == physical
            )
        except StopIteration:
            raise KeyError(f"{physical.name}: no Alpha allocation request") from None

    def logical_view_for(self, logical: ValueRef) -> AlphaLogicalViewBinding:
        try:
            return next(
                binding
                for binding in self.logical_views
                if binding.logical_tensor == logical
            )
        except StopIteration:
            raise KeyError(f"{logical.name}: no Alpha logical view") from None


__all__ = [
    "AlphaClusterConstraint",
    "AlphaEgressBoundary",
    "AlphaGemmStorageBinding",
    "AlphaOperandBinding",
    "AlphaLogicalViewBinding",
    "AlphaNmpInputView",
    "AlphaNmpStorageBinding",
    "AlphaOperationConstraint",
    "AlphaPayloadUse",
    "AlphaPreparedStorage",
    "AlphaResidentWriteBinding",
    "AlphaSerializedUse",
    "AlphaShuffleMovement",
    "AlphaShufflePlan",
    "AlphaStoragePlanError",
]
