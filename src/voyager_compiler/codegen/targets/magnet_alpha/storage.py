# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha's non-bufferized graph-wide storage policy.

The shared storage graph owns identity and liveness.  This module contributes
only the Alpha flat-record facts: embedded VSQ components, resident channel
views, two-output shuffle storage, and global-memory placement requests.  It
is deliberately absent from the bufferized pipeline.
"""

from __future__ import annotations

import math
import operator
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch.fx import GraphModule, Node

from scheduler.types import LocalScheduleResult, OperationWorkload
from scheduler.voyager import (
    LOCAL_SCHEDULE_META,
    LOCAL_SCHEDULER_WORKLOAD_META,
)
from voyager_compiler.codegen.end_to_end import solve_execution_order_dp

from voyager_compiler.codegen.execution_order import (
    DP_EXECUTION_ORDER_MODE,
    END_TO_END_SCRATCHPAD_MAPPING,
    FX_EXECUTION_ORDER_MODE,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    fx_execution_order,
    require_execution_mode,
    validate_execution_order_mode,
)
from voyager_compiler.codegen.node_info import (
    HARDWARE_PADDING_SLICE,
    dtype_byte_size,
    get_arg_value,
    get_hardware_logical_shape,
    is_nop,
    node_operand_bits,
    require_allocation,
)
from voyager_compiler.codegen.storage import (
    GRAPH_SCOPE_OUTER_FX,
    PhysicalTensorId,
    StorageGraph,
    StoragePlane,
    StorageRelationKind,
    ValueRef,
    ViewTransform,
)
from voyager_compiler.codegen.storage.types import recorded_value
from voyager_compiler.codegen.targets.magnet_alpha.allocation import (
    ALPHA_BANK_CLUSTERS,
    AlphaAllocationRequest,
)
from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
    GLOBAL,
    MATRIX_UNIT,
    NMP_UNIT,
    OUTPUT_LANES,
    PARTIAL_MATRIX_INPUT_CHANNELS,
    VSQ_BLOCK_VALUES,
    operand_mode,
)
from voyager_compiler.codegen.targets.magnet_alpha.memory_plan import (
    AlphaClusterConstraint,
    AlphaEgressBoundary,
    AlphaGemmStorageBinding,
    AlphaLogicalViewBinding,
    AlphaNmpInputView,
    AlphaNmpStorageBinding,
    AlphaOperandBinding,
    AlphaOperationConstraint,
    AlphaPayloadUse,
    AlphaPreparedStorage,
    AlphaResidentWriteBinding,
    AlphaSerializedUse,
    AlphaShuffleMovement,
    AlphaShufflePlan,
    AlphaStoragePlanError,
)
from voyager_compiler.codegen.targets.magnet_alpha.scheduling import (
    build_alpha_workload,
)
from voyager_compiler.codegen.targets.magnet_alpha.serialization import (
    address_only_target,
    record_target,
)
from voyager_compiler.hardware_config import TensorRole

EXECUTION_MODES = frozenset(
    {
        (PER_OPERATION_SCRATCHPAD_MAPPING, FX_EXECUTION_ORDER_MODE),
        (END_TO_END_SCRATCHPAD_MAPPING, DP_EXECUTION_ORDER_MODE),
    }
)

_SCRATCHPAD = StoragePlane.SCRATCHPAD.value
_DRAM = StoragePlane.DRAM.value
_ALPHA_DENSE_LAYOUT = "alpha_dense"
_ALPHA_PADDED_INPUT_LAYOUT = "alpha_padded_input"
_ALPHA_VSQ_LAYOUT = "alpha_vsq_embedded"
_ALPHA_SHUFFLE_SECONDARY_LAYOUT = "alpha_shuffle_secondary"
_ALPHA_META = "magnet_alpha"
_PHYSICAL_COPY_META = "physical_copy"

_CONV2D_INT8 = torch.ops.magnet_alpha.conv2d_int8_accumulator.default
_CONV2D_VSQ = torch.ops.magnet_alpha.conv2d_vsq.default
_PVS = torch.ops.magnet_alpha.pvs_quantize.default
_SHUFFLE = torch.ops.magnet_alpha.channel_shuffle.default
_SHUFFLE_COPY = torch.ops.magnet_alpha.channel_shuffle_copy.default
_CAT = torch.ops.aten.cat.default
_STACK = torch.ops.aten.stack.default
_RESHAPE = torch.ops.aten.reshape.default
_VIEW = torch.ops.aten.view.default
_CONTIGUOUS = torch.ops.aten.contiguous.default
_VSQ_LINEAR_VIEWS = frozenset({_RESHAPE, _VIEW, _CONTIGUOUS})
_CHUNK = torch.ops.aten.chunk.default
_SPLIT = torch.ops.aten.split.Tensor
_SPLIT_WITH_SIZES = torch.ops.aten.split_with_sizes.default
_CHANNEL_SPLITS = frozenset({_CHUNK, _SPLIT, _SPLIT_WITH_SIZES})
_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_SLICE = torch.ops.aten.slice.Tensor
_PERMUTE = torch.ops.aten.permute.default


@dataclass(frozen=True)
class _ShuffleCopyFact:
    index: int
    source_output: int
    source_entry: int
    destination_entry: int


@dataclass(frozen=True)
class _ShuffleDraft:
    shuffle: Node
    copies: tuple[Node, ...]
    movements: tuple[_ShuffleCopyFact, ...]
    block_channels: int
    auxiliary: PhysicalTensorId
    active_entries: int
    pitch_entries: int
    span_entries: int
    entry_bytes: int
    primary_entry_offset: int
    zero_copy_splits: tuple[tuple[ValueRef, ValueRef], ...]


@dataclass(frozen=True)
class _ShuffleStorageFacts:
    """Post-finalization facts consumed by allocation and serialization."""

    plans: tuple[AlphaShufflePlan, ...]
    payload_uses: tuple[AlphaPayloadUse, ...]
    serialized_uses: tuple[AlphaSerializedUse, ...]


@dataclass(frozen=True)
class _SplitDraft:
    """One legal Alpha channel split, before physical finalization."""

    split: Node
    source: ValueRef
    outputs: tuple[ValueRef, ...]
    channel_sizes: tuple[int, ...]
    operand_bits: int
    cluster: int
    channels_per_entry: int
    entry_bytes: int
    deferred_secondary: Optional[ValueRef] = None


@dataclass(frozen=True)
class _ClusterDraft:
    """One logical Alpha cluster decision awaiting physical finalization."""

    logical_tensor: ValueRef
    cluster: int


@dataclass(frozen=True)
class _ResidentWrite:
    backing: Node
    owner: Node
    path: tuple[ValueRef, ...]
    slot_index: int
    channel_offset: int
    slot_channels: int
    logical_backing_channels: int
    resident_channels: int
    concat_channel_index: Optional[int]


@dataclass(frozen=True)
class _ConcatLayout:
    inputs: tuple[Node, ...]
    shapes: tuple[tuple[int, ...], ...]
    channel_axis: int
    aliases: tuple[Node, ...]


@dataclass(frozen=True)
class _ConcatPlan:
    layout: _ConcatLayout
    writes: tuple[_ResidentWrite, ...]


@dataclass(frozen=True)
class _VsqPair:
    """One proved Alpha INT4 payload and its UINT8 companion scale."""

    fine_scale: ValueRef
    payload: ValueRef


@dataclass(frozen=True)
class _VsqBlockGeometry:
    """The one axis on which a VSQ payload expands its companion scale."""

    axis: int
    normalized_shape: tuple[int, ...]


@dataclass(frozen=True)
class _VsqArtifactDraft:
    """One scale tensor whose serialized bytes are decided before finalize."""

    operation: Node
    field: str
    logical_tensor: ValueRef
    encoded_axis0_extent: Optional[int] = None


@dataclass(frozen=True)
class _VsqDraft:
    """Immutable pre-finalization VSQ relation and artifact declarations."""

    pairs: tuple[_VsqPair, ...]
    artifacts: tuple[_VsqArtifactDraft, ...]


@dataclass(frozen=True)
class _VsqStorageFacts:
    """Post-finalization VSQ views and serializer-only payload references."""

    logical_views: tuple[AlphaLogicalViewBinding, ...]
    serialized_uses: tuple[AlphaSerializedUse, ...]


@dataclass(frozen=True)
class _MaterializedInputDraft:
    """One host-prepared matrix input representation before finalization."""

    logical_tensor: ValueRef
    component: str
    physical_shape: tuple[int, ...]
    padding: tuple[int, int, int, int]
    valid_channels: int
    resident_channels: int
    serialize_channels: bool


@dataclass(frozen=True)
class _GemmDraft:
    """Storage and padding facts for one GEMM before physical IDs exist."""

    constraint: AlphaOperationConstraint
    input: ValueRef
    weight: ValueRef
    output: ValueRef
    weight_component: str
    input_bits: int
    weight_bits: int
    valid_input_channels: int
    resident_input_channels: int
    physical_weight_input_channels: int
    valid_output_channels: int
    resident_output_channels: int
    physical_weight_output_channels: int
    pads: tuple[int, int, int, int]
    padding_c: int
    padding_k: int
    materialized_input: Optional[_MaterializedInputDraft] = None
    weight_scale: Optional[ValueRef] = None
    weight_scale_valid_output_channels: Optional[int] = None
    weight_scale_encoded_output_channels: Optional[int] = None


@dataclass(frozen=True)
class _GemmStorageFacts:
    """Post-finalization GEMM facts consumed by allocation/serialization."""

    bindings: tuple[AlphaGemmStorageBinding, ...]
    payload_uses: tuple[AlphaPayloadUse, ...]


def _value(node: Node):
    return recorded_value(node)


def _shape(node: Node) -> tuple[int, ...]:
    value = _value(node)
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = getattr(node, "shape", None)
    if shape is None:
        shape = getattr(node.meta.get("tensor_meta"), "shape", None)
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return ()


def _logical_value(ref: ValueRef):
    value = _value(ref.node)
    try:
        for index in ref.path:
            value = value[index]
    except (IndexError, TypeError):
        return None
    return value


def _bits_for_ref(graph: StorageGraph, ref: ValueRef) -> int:
    dtype = graph.logical[ref].dtype
    if dtype is not None:
        try:
            bits = round(dtype_byte_size(dtype) * 8)
        except (TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"{ref.name}: invalid storage dtype {dtype!r}"
            ) from error
        if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
            raise AlphaStoragePlanError(
                f"{ref.name}: invalid {bits!r}-bit storage width"
            )
        return bits
    try:
        bits = node_operand_bits(ref.node)
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"{ref.name}: no storage width is available"
        ) from error
    if isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0:
        raise AlphaStoragePlanError(
            f"{ref.name}: invalid {bits!r}-bit storage width"
        )
    return bits


def _single_ref(graph: StorageGraph, node: Node) -> ValueRef:
    outputs = graph.output_refs(node)
    if len(outputs) != 1:
        raise AlphaStoragePlanError(
            f"{node.name}: expected one logical tensor, found {len(outputs)}"
        )
    return outputs[0]


def _payload_ref(graph: StorageGraph, node: Node) -> ValueRef:
    outputs = {ref.path: ref for ref in graph.output_refs(node)}
    if (2,) in outputs:
        return outputs[(2,)]
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    raise AlphaStoragePlanError(
        f"{node.name}: Alpha output has no unique payload result"
    )


def _submodule(model: GraphModule, owner: Node):
    if owner.op != "call_module":
        return None
    try:
        candidate = model.get_submodule(str(owner.target))
    except AttributeError:
        return None
    return candidate if isinstance(candidate, GraphModule) else None


def _record_nodes(model: GraphModule, owner: Node) -> tuple[Node, ...]:
    submodule = _submodule(model, owner)
    candidates = tuple(submodule.graph.nodes) if submodule is not None else (owner,)
    return tuple(node for node in candidates if record_target(node) is not None)


def _outer_operand(model: GraphModule, owner: Node, candidate) -> Node:
    """Resolve a fused-body tensor argument to its outer FX binding."""
    if not isinstance(candidate, Node):
        raise AlphaStoragePlanError(
            f"{owner.name}: Alpha tensor operand did not resolve to FX"
        )
    submodule = _submodule(model, owner)
    if submodule is None or candidate.graph is not submodule.graph:
        return candidate

    placeholders = tuple(
        node for node in submodule.graph.nodes if node.op == "placeholder"
    )
    bindings = {
        placeholder: argument
        for placeholder, argument in zip(placeholders, owner.args)
        if isinstance(argument, Node)
    }
    current = candidate
    visited = set()
    while current.graph is submodule.graph:
        if current in visited:
            raise AlphaStoragePlanError(
                f"{owner.name}: fused operand path is cyclic"
            )
        visited.add(current)
        if current.op == "placeholder":
            result = bindings.get(current)
            if not isinstance(result, Node):
                raise AlphaStoragePlanError(
                    f"{owner.name}: fused placeholder {current.name} has no "
                    "outer tensor binding"
                )
            return result
        inputs = tuple(current.all_input_nodes)
        if len(inputs) != 1 or address_only_target(current) is None:
            raise AlphaStoragePlanError(
                f"{owner.name}: {current.name} is not a transparent fused "
                "operand path"
            )
        current = inputs[0]
    return current


def _implementation_operand(
    model: GraphModule,
    owner: Node,
    implementation: Node,
    position: int,
    name: str,
    default=None,
):
    candidate = get_arg_value(implementation, position, name, default)
    return None if candidate is None else _outer_operand(model, owner, candidate)


def _collect_operation_constraints(
    model: GraphModule,
    graph: StorageGraph,
) -> tuple[AlphaOperationConstraint, ...]:
    constraints = []
    for owner in model.graph.nodes:
        workload = build_alpha_workload(model, owner)
        if workload is None:
            continue
        result = owner.meta.get(LOCAL_SCHEDULE_META)
        bound_workload = owner.meta.get(LOCAL_SCHEDULER_WORKLOAD_META)
        if not isinstance(result, LocalScheduleResult):
            raise AlphaStoragePlanError(
                f"{owner.name}: executable Alpha owner has no LocalScheduleResult"
            )
        if not isinstance(bound_workload, OperationWorkload):
            raise AlphaStoragePlanError(
                f"{owner.name}: executable Alpha owner has no bound workload"
            )
        if bound_workload != workload:
            raise AlphaStoragePlanError(
                f"{owner.name}: bound Alpha workload is stale"
            )

        records = _record_nodes(model, owner)
        if not records:
            raise AlphaStoragePlanError(
                f"{owner.name}: executable workload has no implementation"
            )
        implementation = records[0]
        kind = workload.operation.kind
        unit = MATRIX_UNIT.name if kind == "gemm" else NMP_UNIT.name
        positions = (
            {"input": 0, "weight": 1}
            if kind in {"gemm", "depthwise_conv2d"}
            else {"input1": 0, "input2": 1}
            if kind == "channel_shuffle"
            else {"input": 0}
        )
        operands = []
        for scheduled in workload.operands:
            position = positions.get(scheduled.port)
            if position is None:
                raise AlphaStoragePlanError(
                    f"{owner.name}: unsupported Alpha port {scheduled.port!r}"
                )
            operand_node = _implementation_operand(
                model, owner, implementation, position, scheduled.port
            )
            operand_ref = _payload_ref(graph, operand_node)
            if tuple(scheduled.shape or ()) != tuple(graph.logical[operand_ref].shape):
                raise AlphaStoragePlanError(
                    f"{owner.name}: {scheduled.port} changed after scheduling"
                )
            bits = int(scheduled.bits or 0)
            if bits not in {4, 8}:
                raise AlphaStoragePlanError(
                    f"{owner.name}: unsupported {bits}-bit Alpha operand"
                )
            if require_allocation(operand_node):
                operands.append(
                    AlphaOperandBinding(
                        port=scheduled.port,
                        logical_tensor=operand_ref,
                        role=(
                            TensorRole.WEIGHT
                            if scheduled.port == "weight"
                            else TensorRole.ACTIVATION
                        ),
                        bits=bits,
                        compute_unit=unit,
                    )
                )
        output = _payload_ref(graph, owner)
        output_bits = int(workload.output.bits or 0)
        if output_bits not in {4, 8}:
            raise AlphaStoragePlanError(
                f"{owner.name}: unsupported {output_bits}-bit Alpha output"
            )
        constraints.append(
            AlphaOperationConstraint(
                owner=owner,
                implementation=implementation,
                terminal=records[-1],
                unit=unit,
                workload=workload,
                schedule=result,
                operands=tuple(operands),
                output=output,
                output_component="primary",
                output_bits=output_bits,
            )
        )
    return tuple(constraints)


def _single_user(node: Node) -> Optional[Node]:
    if len(node.users) != 1:
        return None
    return next(iter(node.users))


def _alpha_node_width_is(
    graph: StorageGraph,
    node: Node,
    width: int,
) -> bool:
    try:
        return _bits_for_ref(graph, _single_ref(graph, node)) == width
    except (AlphaStoragePlanError, KeyError, TypeError, ValueError):
        return False


def _alpha_egress_owners(
    source: Node,
    operation_owners: frozenset[Node],
) -> Optional[frozenset[Node]]:
    """Resolve a terminal value through only declared serialized NOPs."""
    pending = [source]
    visited = set()
    owners = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in operation_owners:
            owners.add(current)
            continue
        transparent = current.target is operator.getitem or is_nop(current)
        if current.target is _CONTIGUOUS:
            input_node = current.args[0] if current.args else None
            input_value = (
                _value(input_node) if isinstance(input_node, Node) else None
            )
            transparent = bool(
                isinstance(input_value, torch.Tensor)
                and input_value.is_contiguous()
            )
        if (
            current.op not in {"call_function", "call_module"}
            or not transparent
            or address_only_target(current) is None
        ):
            return None
        inputs = tuple(current.all_input_nodes)
        if not inputs:
            return None
        pending.extend(inputs)
    return frozenset(owners) if owners else None


def _alpha_terminal_padding_slice(
    graph: StorageGraph,
    node: Node,
    dequantize: Node,
) -> bool:
    """Prove one target-inserted crop of padded output channels."""
    if (
        node.target is not _SLICE
        or not node.meta.get(HARDWARE_PADDING_SLICE, False)
        or not node.args
        or node.args[0] is not dequantize
    ):
        return False
    source_shape = _shape(dequantize)
    result_shape = _shape(node)
    if (
        not source_shape
        or len(source_shape) != len(result_shape)
        or len(source_shape) < 2
    ):
        return False
    dim = get_arg_value(node, 1, "dim", 0)
    start = get_arg_value(node, 2, "start", 0)
    end = get_arg_value(node, 3, "end")
    step = get_arg_value(node, 4, "step", 1)
    normalized_dim = (
        dim + len(source_shape)
        if isinstance(dim, int) and dim < 0
        else dim
    )
    return bool(
        dim in (1, -3)
        and normalized_dim == 1
        and start in (None, 0)
        and step == 1
        and isinstance(end, int)
        and source_shape[1] == 64
        and end == 48
        and 0 < end < source_shape[1]
        and result_shape[1] == end
        and source_shape[:1] == result_shape[:1]
        and source_shape[2:] == result_shape[2:]
        and _alpha_node_width_is(graph, node, 32)
    )


def _alpha_contiguous_requires_copy(source: Node, result: Node) -> bool:
    """Mirror the fork's stride-aware contiguous storage predicate."""
    source_value = _value(source)
    result_value = _value(result)
    return bool(
        isinstance(source_value, torch.Tensor)
        and isinstance(result_value, torch.Tensor)
        and source_value.dtype == result_value.dtype
        and source_value.numel() == result_value.numel()
        and not source_value.is_contiguous()
        and result_value.is_contiguous()
    )


def _match_alpha_yolo_egress(
    graph: StorageGraph,
    dequantize: Node,
    operation_owners: frozenset[Node],
) -> Optional[AlphaEgressBoundary]:
    """Match the fork's complete YOLO terminal host-layout suffix."""
    if dequantize.op != "call_function" or dequantize.target is not _DEQUANTIZE:
        return None
    source = get_arg_value(dequantize, 0, "input")
    scale = get_arg_value(dequantize, 1, "scale")
    if not isinstance(source, Node) or not isinstance(scale, Node):
        return None
    owners = _alpha_egress_owners(source, operation_owners)
    if (
        owners is None
        or not _alpha_node_width_is(graph, source, 8)
        or not _alpha_node_width_is(graph, dequantize, 32)
        or _shape(source) != _shape(dequantize)
    ):
        return None

    padding_slice = _single_user(dequantize)
    if not isinstance(padding_slice, Node) or not _alpha_terminal_padding_slice(
        graph,
        padding_slice,
        dequantize,
    ):
        return None

    view = _single_user(padding_slice)
    if (
        not isinstance(view, Node)
        or view.target is not _VIEW
        or not view.args
        or view.args[0] is not padding_slice
        or address_only_target(view) != "view"
    ):
        return None
    slice_shape = _shape(padding_slice)
    view_shape = _shape(view)
    declared_shape = get_arg_value(view, 1, "size")
    try:
        declared_shape = tuple(int(size) for size in declared_shape)
    except (TypeError, ValueError):
        return None
    if (
        len(slice_shape) != 4
        or slice_shape[1] != 48
        or view_shape
        != (
            slice_shape[0],
            3,
            16,
            slice_shape[2],
            slice_shape[3],
        )
        or declared_shape != view_shape
        or not _alpha_node_width_is(graph, view, 32)
    ):
        return None

    permute = _single_user(view)
    if (
        not isinstance(permute, Node)
        or permute.target is not _PERMUTE
        or not permute.args
        or permute.args[0] is not view
    ):
        return None
    dims = get_arg_value(permute, 1, "dims")
    try:
        dims = tuple(int(dim) for dim in dims)
    except (TypeError, ValueError):
        return None
    expected_shape = (
        view_shape[0],
        view_shape[1],
        view_shape[3],
        view_shape[4],
        view_shape[2],
    )
    if (
        dims != (0, 1, 3, 4, 2)
        or _shape(permute) != expected_shape
        or not _alpha_node_width_is(graph, permute, 32)
    ):
        return None

    contiguous = _single_user(permute)
    if (
        not isinstance(contiguous, Node)
        or contiguous.target is not _CONTIGUOUS
        or not contiguous.args
        or contiguous.args[0] is not permute
    ):
        return None
    memory_format = get_arg_value(contiguous, 1, "memory_format")
    if (
        memory_format not in (None, torch.contiguous_format)
        or _shape(contiguous) != expected_shape
        or not _alpha_node_width_is(graph, contiguous, 32)
        or not _alpha_contiguous_requires_copy(permute, contiguous)
    ):
        return None
    output = _single_user(contiguous)
    if not isinstance(output, Node) or output.op != "output":
        return None
    return AlphaEgressBoundary(
        owners=owners,
        source=_single_ref(graph, source),
        dequantize=dequantize,
        padding_slice=padding_slice,
        shared_views=(view,),
        host_layouts=(permute, contiguous),
        output=output,
    )


def _relation_matches(
    relation,
    kind: StorageRelationKind,
    source: Optional[ValueRef],
    *,
    transform: Optional[ViewTransform] = None,
    compare_transform: bool = False,
) -> bool:
    return bool(
        relation.kind is kind
        and relation.source == source
        and relation.physical_source is None
        and relation.component is None
        and (not compare_transform or relation.transform == transform)
    )


def _preflight_alpha_egress_relations(
    graph: StorageGraph,
    boundary: AlphaEgressBoundary,
    *,
    plane: str,
) -> None:
    """Check every affected relation before declaring either override."""
    dequantize_ref = _single_ref(graph, boundary.dequantize)
    slice_ref = _single_ref(graph, boundary.padding_slice)
    view_ref = _single_ref(graph, boundary.shared_views[0])
    permute_ref = _single_ref(graph, boundary.host_layouts[0])
    contiguous_ref = _single_ref(graph, boundary.host_layouts[1])
    address_transform = ViewTransform("alpha_address_only")
    checks = (
        (
            dequantize_ref,
            _relation_matches(
                graph.relation(plane, dequantize_ref),
                StorageRelationKind.MATERIALIZE,
                None,
            ),
        ),
        (
            slice_ref,
            _relation_matches(
                graph.relation(plane, slice_ref),
                StorageRelationKind.COPY,
                dequantize_ref,
            )
            or _relation_matches(
                graph.relation(plane, slice_ref),
                StorageRelationKind.VIEW,
                dequantize_ref,
                transform=address_transform,
                compare_transform=True,
            ),
        ),
        (
            view_ref,
            _relation_matches(
                graph.relation(plane, view_ref),
                StorageRelationKind.ALIAS,
                slice_ref,
            ),
        ),
        (
            permute_ref,
            _relation_matches(
                graph.relation(plane, permute_ref),
                StorageRelationKind.COPY,
                view_ref,
            ),
        ),
        (
            contiguous_ref,
            _relation_matches(
                graph.relation(plane, contiguous_ref),
                StorageRelationKind.ALIAS,
                permute_ref,
            )
            or _relation_matches(
                graph.relation(plane, contiguous_ref),
                StorageRelationKind.COPY,
                permute_ref,
            ),
        ),
    )
    invalid = tuple(ref.name for ref, valid in checks if not valid)
    if invalid:
        raise AlphaStoragePlanError(
            "Alpha terminal egress conflicts with existing storage relations: "
            + ", ".join(invalid)
        )


def _declare_alpha_egress_storage(
    model: GraphModule,
    graph: StorageGraph,
    constraints: tuple[AlphaOperationConstraint, ...],
    *,
    plane: str = _SCRATCHPAD,
) -> tuple[AlphaEgressBoundary, ...]:
    """Declare only complete fork-proven terminal YOLO storage suffixes."""
    plane = StoragePlane(plane).value
    operation_owners = frozenset(
        constraint.owner for constraint in constraints
    )
    boundaries = tuple(
        boundary
        for node in model.graph.nodes
        for boundary in (
            _match_alpha_yolo_egress(graph, node, operation_owners),
        )
        if boundary is not None
    )
    storage_nodes = tuple(
        node for boundary in boundaries for node in boundary.storage_path
    )
    if len(set(storage_nodes)) != len(storage_nodes):
        raise AlphaStoragePlanError(
            "Alpha terminal egress boundaries overlap in storage"
        )
    for boundary in boundaries:
        _preflight_alpha_egress_relations(graph, boundary, plane=plane)
    for boundary in boundaries:
        graph.set_relation(
            boundary.padding_slice,
            StorageRelationKind.VIEW,
            plane=plane,
            source=boundary.dequantize,
            transform=ViewTransform("alpha_address_only"),
        )
        # The fork's stride-aware generic analysis materializes contiguous()
        # after this non-contiguous permute.  The shared upstream predicate is
        # intentionally broader, so reproduce that proven boundary locally.
        graph.set_relation(
            boundary.host_layouts[1],
            StorageRelationKind.COPY,
            plane=plane,
            source=boundary.host_layouts[0],
        )
    return boundaries


def _freeze_alpha_egress_storage(
    plan,
    boundaries: tuple[AlphaEgressBoundary, ...],
    *,
    plane: str = _SCRATCHPAD,
) -> tuple[AlphaEgressBoundary, ...]:
    """Validate the declared egress relations against the finalized plan."""
    plane = StoragePlane(plane).value
    for boundary in boundaries:
        dequantize_ref = plan.ref(boundary.dequantize)
        slice_ref = plan.ref(boundary.padding_slice)
        view_ref = plan.ref(boundary.shared_views[0])
        permute_ref = plan.ref(boundary.host_layouts[0])
        contiguous_ref = plan.ref(boundary.host_layouts[1])
        expected = (
            (dequantize_ref, StorageRelationKind.MATERIALIZE, None, None),
            (
                slice_ref,
                StorageRelationKind.VIEW,
                dequantize_ref,
                ViewTransform("alpha_address_only"),
            ),
            (view_ref, StorageRelationKind.ALIAS, slice_ref, None),
            (permute_ref, StorageRelationKind.COPY, view_ref, None),
            (contiguous_ref, StorageRelationKind.COPY, permute_ref, None),
        )
        for ref, kind, source, transform in expected:
            relation = plan.relation(plane, ref)
            if not _relation_matches(
                relation,
                kind,
                source,
                transform=transform,
                compare_transform=transform is not None,
            ):
                raise AlphaStoragePlanError(
                    f"{ref.name}: finalized Alpha egress relation changed"
                )
        if (
            plan.physical_id(plane, slice_ref)
            != plan.physical_id(plane, dequantize_ref)
            or plan.physical_id(plane, view_ref)
            != plan.physical_id(plane, dequantize_ref)
            or plan.physical_id(plane, permute_ref)
            == plan.physical_id(plane, view_ref)
            or plan.physical_id(plane, contiguous_ref)
            == plan.physical_id(plane, permute_ref)
        ):
            raise AlphaStoragePlanError(
                f"{boundary.dequantize.name}: finalized Alpha egress "
                "physical identities changed"
            )
    return tuple(boundaries)


def _shuffle_copy_movements(
    channel_blocks: int,
) -> tuple[_ShuffleCopyFact, ...]:
    """Return the fork's exact in-place movement order."""
    if channel_blocks < 2 or channel_blocks % 2:
        raise AlphaStoragePlanError(
            "Alpha shuffle requires a positive even channel-block count"
        )

    half = channel_blocks // 2
    movements = []
    for source_entry in range(half - 1, 0, -1):
        movements.append(
            _ShuffleCopyFact(
                index=len(movements) + 1,
                source_output=0,
                source_entry=source_entry,
                destination_entry=2 * source_entry,
            )
        )
    for source_entry in range(half - 1):
        movements.append(
            _ShuffleCopyFact(
                index=len(movements) + 1,
                source_output=1,
                source_entry=source_entry,
                destination_entry=2 * source_entry + 1,
            )
        )
    return tuple(movements)


def _vsq_tensor_ref(graph: StorageGraph, value, label: str) -> ValueRef:
    """Resolve one VSQ operand without guessing a tuple component."""
    if isinstance(value, ValueRef):
        return graph.ref(value)
    if not isinstance(value, Node):
        raise AlphaStoragePlanError(f"Alpha VSQ {label} is not an FX tensor")
    try:
        return _single_ref(graph, value)
    except AlphaStoragePlanError as error:
        raise AlphaStoragePlanError(
            f"{value.name}: Alpha VSQ {label} is not one tensor"
        ) from error


def _validate_alpha_vsq_pair(
    graph: StorageGraph,
    fine_scale: ValueRef,
    payload: ValueRef,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require Alpha's measured UINT8-per-32-values INT4 representation."""
    fine_value = _logical_value(fine_scale)
    payload_value = _logical_value(payload)
    if not isinstance(fine_value, torch.Tensor):
        raise AlphaStoragePlanError(
            f"{fine_scale.name}: Alpha VSQ fine scale has no tensor value"
        )
    if not isinstance(payload_value, torch.Tensor):
        raise AlphaStoragePlanError(
            f"{payload.name}: Alpha VSQ payload has no tensor value"
        )
    if fine_value.dtype is not torch.uint8:
        raise AlphaStoragePlanError(
            f"{fine_scale.name}: Alpha VSQ fine scale must contain UINT8 "
            f"codes, not {fine_value.dtype}"
        )
    fine_dtype = str(graph.logical[fine_scale].dtype).removeprefix("torch.")
    payload_dtype = str(graph.logical[payload].dtype).removeprefix("torch.")
    if fine_dtype != "uint8":
        raise AlphaStoragePlanError(
            f"{fine_scale.name}: Alpha VSQ fine scale has no UINT8 storage "
            "contract"
        )
    if payload_dtype != "int4":
        raise AlphaStoragePlanError(
            f"{payload.name}: Alpha VSQ payload has no INT4 storage contract"
        )
    _alpha_vsq_block_geometry(fine_value, payload_value, fine_scale.name)
    fine_static = fine_scale.node.op == "get_attr"
    payload_static = payload.node.op == "get_attr"
    if fine_static != payload_static:
        raise AlphaStoragePlanError(
            f"{fine_scale.name}: Alpha VSQ payload and fine scale must have "
            "the same static/runtime lifetime"
        )
    return fine_value, payload_value


def _alpha_vsq_block_geometry(
    fine_value: torch.Tensor,
    payload_value: torch.Tensor,
    label: str,
) -> _VsqBlockGeometry:
    """Prove one exact, shape-preserving 32-value VSQ block axis."""
    fine_shape = tuple(int(extent) for extent in fine_value.shape)
    payload_shape = tuple(int(extent) for extent in payload_value.shape)
    scaled_axes = []
    if (
        not fine_shape
        or len(fine_shape) != len(payload_shape)
        or any(extent <= 0 for extent in (*fine_shape, *payload_shape))
    ):
        scaled_axes = []
    else:
        for axis, (fine_extent, payload_extent) in enumerate(
            zip(fine_shape, payload_shape)
        ):
            if payload_extent == fine_extent:
                continue
            if payload_extent == fine_extent * VSQ_BLOCK_VALUES:
                scaled_axes.append(axis)
                continue
            scaled_axes = []
            break
    if len(scaled_axes) != 1:
        raise AlphaStoragePlanError(
            f"{label}: Alpha VSQ requires one UINT8 fine scale per "
            f"{VSQ_BLOCK_VALUES} INT4 payload values on exactly one axis; "
            f"got fine shape {fine_shape} and payload shape {payload_shape}"
        )
    return _VsqBlockGeometry(
        axis=scaled_axes[0],
        normalized_shape=fine_shape,
    )


def _set_alpha_vsq_relation(
    graph: StorageGraph,
    result: ValueRef,
    kind: StorageRelationKind,
    *,
    plane: str,
    source: ValueRef,
    component: Optional[str] = None,
) -> None:
    """Declare one immutable VSQ relation, accepting only an exact repeat."""
    previous = graph.relation(plane, result)
    if (
        previous.kind is kind
        and previous.source == source
        and previous.physical_source is None
        and previous.transform is None
        and previous.component == component
    ):
        return
    try:
        graph.set_relation(
            result,
            kind,
            plane=plane,
            source=source,
            component=component,
        )
    except ValueError as error:
        raise AlphaStoragePlanError(
            f"{result.name}: conflicting Alpha VSQ storage relation"
        ) from error


def _alpha_matrix_output_lanes(config, input_bits: int) -> int:
    """Resolve the matrix K encoding width from the selected machine."""
    if config is None:
        lanes = MATRIX_UNIT.unroll("OC", input_bits)
    else:
        output_lanes = getattr(config, "output_lanes", None)
        if not callable(output_lanes):
            raise AlphaStoragePlanError(
                "Alpha configuration has no matrix output_lanes contract"
            )
        try:
            lanes = output_lanes(input_bits)
        except (TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"Alpha configuration cannot resolve {input_bits}-bit "
                "matrix output lanes"
            ) from error
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
        raise AlphaStoragePlanError(
            "Alpha matrix output lanes must be a positive integer, got "
            f"{lanes!r}"
        )
    return lanes


def _declare_alpha_vsq_relations(
    model: GraphModule,
    graph: StorageGraph,
    operation_constraints: tuple[AlphaOperationConstraint, ...],
    *,
    config=None,
    plane: str = _SCRATCHPAD,
) -> _VsqDraft:
    """Recognize exact physical VSQ pairs, then declare them atomically.

    Discovery is deliberately closed over the paths observed in the Alpha
    graph: terminal PVS tuple results, corresponding linear views, and paired
    ordered cat/stack aggregates.  No graph relation is changed until every
    candidate has passed shape, dtype, lifetime, and bijection checks.
    """
    plane = StoragePlane(plane).value
    if plane not in graph.planes:
        raise AlphaStoragePlanError(
            f"{plane}: Alpha VSQ storage is not a plane of this graph"
        )

    constraints = {
        constraint.owner: constraint for constraint in operation_constraints
    }
    if len(constraints) != len(operation_constraints):
        raise AlphaStoragePlanError(
            "Alpha VSQ operation constraints repeat an owner"
        )

    records_by_owner = {}
    proved_pvs_outputs = {}
    pair_seeds = []
    coarse_aliases = []
    artifacts = []
    artifact_by_field = {}

    def add_pair_seed(fine, payload) -> None:
        pair_seeds.append(
            (
                _vsq_tensor_ref(graph, fine, "fine scale"),
                _vsq_tensor_ref(graph, payload, "payload"),
            )
        )

    def add_artifact(
        operation: Node,
        field: str,
        logical_tensor: ValueRef,
        *,
        encoded_axis0_extent: Optional[int] = None,
    ) -> None:
        key = (operation, field)
        draft = _VsqArtifactDraft(
            operation=operation,
            field=field,
            logical_tensor=logical_tensor,
            encoded_axis0_extent=encoded_axis0_extent,
        )
        previous = artifact_by_field.get(key)
        if previous is not None:
            if previous != draft:
                raise AlphaStoragePlanError(
                    f"{operation.name}: conflicting Alpha VSQ artifact {field}"
                )
            return
        artifact_by_field[key] = draft
        artifacts.append(draft)

    # ``dtype`` describes physical payload overrides, not merely the dtype of
    # the tensors returned by the executable reference operator.  PVS output
    # 0 passes its floating coarse-scale operand through unchanged, so Alpha
    # deliberately leaves that component undeclared (``None``).  Outputs 1
    # and 2 are the physical uint8 mantissa and INT4 payload.  The storage
    # index independently recovers output 0's logical float32 dtype from the
    # recorded tuple value.
    expected_pvs_declared_dtypes = (None, "uint8", "int4")
    expected_pvs_logical_dtypes = ("float32", "uint8", "int4")
    for owner in model.graph.nodes:
        records = _record_nodes(model, owner)
        records_by_owner[owner] = records
        pvs_records = tuple(
            record for record in records if record.target is _PVS
        )
        if len(pvs_records) > 1:
            raise AlphaStoragePlanError(
                f"{owner.name}: Alpha record contains multiple PVS stages"
            )
        if not pvs_records:
            continue

        pvs = pvs_records[0]
        submodule = _submodule(model, owner)
        if owner.target is _PVS:
            terminal = len(records) == 1 and records[0] is owner
        else:
            outputs = (
                tuple(
                    node
                    for node in submodule.graph.nodes
                    if node.op == "output"
                )
                if submodule is not None
                else ()
            )
            terminal = (
                bool(records)
                and records[-1] is pvs
                and len(outputs) == 1
                and bool(outputs[0].args)
                and outputs[0].args[0] is pvs
            )
        if not terminal:
            raise AlphaStoragePlanError(
                f"{owner.name}: Alpha PVS must be the exact terminal fused "
                "result"
            )

        output_refs = {ref.path: ref for ref in graph.output_refs(owner)}
        if set(output_refs) != {(0,), (1,), (2,)}:
            raise AlphaStoragePlanError(
                f"{owner.name}: PVS record must expose exact output paths "
                "0, 1, and 2"
            )
        declared_dtypes = owner.meta.get("dtype")
        normalized_dtypes = (
            tuple(
                None
                if dtype is None
                else str(dtype).removeprefix("torch.")
                for dtype in declared_dtypes
            )
            if isinstance(declared_dtypes, (tuple, list))
            else ()
        )
        logical_dtypes = tuple(
            str(graph.logical[output_refs[(index,)]].dtype).removeprefix(
                "torch."
            )
            for index in range(3)
        )
        if (
            normalized_dtypes != expected_pvs_declared_dtypes
            or logical_dtypes != expected_pvs_logical_dtypes
        ):
            raise AlphaStoragePlanError(
                f"{owner.name}: Alpha PVS outputs must declare dtype tuple "
                "(None, uint8, int4) with logical tensor dtypes "
                "(float32, uint8, int4); got declared "
                f"{normalized_dtypes!r} and logical {logical_dtypes!r}"
            )

        precision = get_arg_value(pvs, 2, "output_precision")
        block_size = get_arg_value(pvs, 3, "block_size")
        if precision != 4 or block_size != VSQ_BLOCK_VALUES:
            raise AlphaStoragePlanError(
                f"{owner.name}: Alpha PVS storage requires INT4 output "
                f"and block size {VSQ_BLOCK_VALUES}, got "
                f"{precision!r}/{block_size!r}"
            )
        if submodule is not None:
            constraint = constraints.get(owner)
            if (
                constraint is None
                or constraint.output != output_refs[(2,)]
                or constraint.output_bits != 4
            ):
                raise AlphaStoragePlanError(
                    f"{owner.name}: terminal fused PVS has no matching "
                    "4-bit operation constraint"
                )

        coarse_source = _implementation_operand(
            model,
            owner,
            pvs,
            1,
            "coarse_scale",
        )
        coarse_source_ref = _vsq_tensor_ref(
            graph, coarse_source, "coarse scale source"
        )
        if (
            graph.relation(plane, output_refs[(2,)]).kind
            is not StorageRelationKind.MATERIALIZE
        ):
            raise AlphaStoragePlanError(
                f"{output_refs[(2,)].name}: PVS payload does not begin as a "
                "materialization"
            )
        proved_pvs_outputs[owner] = tuple(
            output_refs[(index,)] for index in range(3)
        )
        coarse_aliases.append((output_refs[(0,)], coarse_source_ref))
        add_pair_seed(output_refs[(1,)], output_refs[(2,)])
        add_artifact(
            owner,
            "output_vector_scale_payload",
            output_refs[(1,)],
        )

    for owner in model.graph.nodes:
        records = records_by_owner[owner]
        conv_records = tuple(
            record for record in records if record.target is _CONV2D_VSQ
        )
        if len(conv_records) > 1:
            raise AlphaStoragePlanError(
                f"{owner.name}: Alpha record contains multiple VSQ convolutions"
            )
        if not conv_records:
            continue
        conv = conv_records[0]
        constraint = constraints.get(owner)
        if constraint is None or constraint.implementation is not conv:
            raise AlphaStoragePlanError(
                f"{owner.name}: VSQ convolution has no matching operation "
                "constraint"
            )

        input_payload = _implementation_operand(
            model, owner, conv, 0, "input"
        )
        weight_payload = _implementation_operand(
            model, owner, conv, 1, "weight"
        )
        weight_scale = _implementation_operand(
            model, owner, conv, 2, "weight_scale"
        )
        input_scale = _implementation_operand(
            model, owner, conv, 7, "input_scale", None
        )

        weight_scale_ref = _vsq_tensor_ref(
            graph, weight_scale, "weight fine scale"
        )
        weight_payload_ref = _vsq_tensor_ref(
            graph, weight_payload, "weight payload"
        )
        add_pair_seed(weight_scale_ref, weight_payload_ref)
        weight_scale_value = _logical_value(weight_scale_ref)
        if (
            not isinstance(weight_scale_value, torch.Tensor)
            or weight_scale_value.ndim != 4
        ):
            raise AlphaStoragePlanError(
                f"{weight_scale_ref.name}: Alpha weight fine scale must be "
                "rank four"
            )
        try:
            input_bits = constraint.workload.require_operand("input").bits
        except ValueError as error:
            raise AlphaStoragePlanError(
                f"{owner.name}: VSQ convolution has no scheduled input width"
            ) from error
        if (
            isinstance(input_bits, bool)
            or not isinstance(input_bits, int)
            or input_bits <= 0
        ):
            raise AlphaStoragePlanError(
                f"{owner.name}: invalid VSQ input width {input_bits!r}"
            )
        output_lanes = _alpha_matrix_output_lanes(config, input_bits)
        encoded_k = (
            math.ceil(weight_scale_value.shape[0] / output_lanes)
            * output_lanes
        )
        add_artifact(
            owner,
            "weight_vector_scale_payload",
            weight_scale_ref,
            encoded_axis0_extent=encoded_k,
        )

        if input_scale is not None:
            input_scale_ref = _vsq_tensor_ref(
                graph, input_scale, "input fine scale"
            )
            input_payload_ref = _vsq_tensor_ref(
                graph, input_payload, "input payload"
            )
            add_pair_seed(input_scale_ref, input_payload_ref)
            add_artifact(
                owner,
                "input_vector_scale_payload",
                input_scale_ref,
            )

    canonical_pairs = []
    pair_by_refs = {}
    payload_by_fine = {}
    fine_by_payload = {}
    resolving = set()

    def bind_canonical_pair(pair: _VsqPair) -> _VsqPair:
        previous_payload = payload_by_fine.get(pair.fine_scale)
        if previous_payload is not None and previous_payload != pair.payload:
            raise AlphaStoragePlanError(
                f"{pair.fine_scale.name}: Alpha VSQ fine scale names two "
                "payloads"
            )
        previous_fine = fine_by_payload.get(pair.payload)
        if previous_fine is not None and previous_fine != pair.fine_scale:
            raise AlphaStoragePlanError(
                f"{pair.payload.name}: Alpha VSQ payload names two fine scales"
            )
        payload_by_fine[pair.fine_scale] = pair.payload
        fine_by_payload[pair.payload] = pair.fine_scale
        if pair not in canonical_pairs:
            canonical_pairs.append(pair)
        return pair

    def aggregate_dim(node: Node, rank: int) -> int:
        dim = get_arg_value(node, 1, "dim", 0)
        if isinstance(dim, bool) or not isinstance(dim, int) or rank <= 0:
            raise AlphaStoragePlanError(
                f"{node.name}: Alpha VSQ aggregate has no static dimension"
            )
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise AlphaStoragePlanError(
                f"{node.name}: Alpha VSQ aggregate dimension is out of range"
            )
        return dim

    def view_shape(node: Node, ref: ValueRef) -> tuple[int, ...]:
        argument_name = "shape" if node.target is _RESHAPE else "size"
        shape = get_arg_value(node, 1, argument_name, None)
        if (
            not isinstance(shape, (tuple, list, torch.Size))
            or any(
                isinstance(extent, bool) or not isinstance(extent, int)
                for extent in shape
            )
        ):
            raise AlphaStoragePlanError(
                f"{node.name}: paired Alpha VSQ view has no static shape"
            )
        shape = tuple(int(extent) for extent in shape)
        if shape != tuple(graph.logical[ref].shape):
            raise AlphaStoragePlanError(
                f"{node.name}: paired Alpha VSQ view shape is stale"
            )
        allowed_kwargs = {argument_name}
        if len(node.args) > 2 or set(node.kwargs) - allowed_kwargs:
            raise AlphaStoragePlanError(
                f"{node.name}: paired Alpha VSQ view has unsupported arguments"
            )
        return shape

    def resolve_pair(
        fine_ref: ValueRef,
        payload_ref: ValueRef,
        *,
        allow_leaf: bool = False,
    ) -> _VsqPair:
        key = (fine_ref, payload_ref)
        previous = pair_by_refs.get(key)
        if previous is not None:
            return previous
        if key in resolving:
            raise AlphaStoragePlanError(
                f"{fine_ref.name}: cyclic Alpha VSQ paired transform"
            )
        resolving.add(key)
        try:
            fine_value, payload_value = _validate_alpha_vsq_pair(
                graph, fine_ref, payload_ref
            )
            geometry = _alpha_vsq_block_geometry(
                fine_value, payload_value, fine_ref.name
            )
            fine_relation = graph.relation(plane, fine_ref)
            payload_relation = graph.relation(plane, payload_ref)
            if (
                fine_relation.kind is StorageRelationKind.COPY
                or payload_relation.kind is StorageRelationKind.COPY
                or fine_ref.node.meta.get(_PHYSICAL_COPY_META) is True
                or payload_ref.node.meta.get(_PHYSICAL_COPY_META) is True
            ):
                raise AlphaStoragePlanError(
                    f"{fine_ref.name}: copied values cannot form an embedded "
                    "Alpha VSQ pair"
                )

            if (
                fine_ref.node is payload_ref.node
                and fine_ref.path == (1,)
                and payload_ref.path == (2,)
                and fine_ref.node in proved_pvs_outputs
            ):
                pair = _VsqPair(fine_ref, payload_ref)
                result = bind_canonical_pair(pair)
                pair_by_refs[key] = result
                return result

            fine_node = fine_ref.node
            payload_node = payload_ref.node
            exact_pvs_getitems = (
                not fine_ref.path
                and not payload_ref.path
                and fine_node.target is operator.getitem
                and payload_node.target is operator.getitem
                and len(fine_node.args) == 2
                and len(payload_node.args) == 2
                and fine_node.args[0] is payload_node.args[0]
                and fine_node.args[1] == 1
                and payload_node.args[1] == 2
                and isinstance(fine_node.args[0], Node)
                and fine_node.args[0] in proved_pvs_outputs
            )
            if exact_pvs_getitems:
                outputs = proved_pvs_outputs[fine_node.args[0]]
                result = resolve_pair(outputs[1], outputs[2])
                pair_by_refs[key] = result
                return result

            fine_target = getattr(fine_node, "target", None)
            payload_target = getattr(payload_node, "target", None)
            if (
                fine_target in {_CAT, _STACK}
                or payload_target in {_CAT, _STACK}
            ):
                if (
                    fine_target is not payload_target
                    or fine_target not in {_CAT, _STACK}
                    or fine_ref.path
                    or payload_ref.path
                ):
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: Alpha VSQ aggregate is not paired"
                    )
                if payload_relation.kind is not StorageRelationKind.MATERIALIZE:
                    raise AlphaStoragePlanError(
                        f"{payload_ref.name}: Alpha VSQ aggregate payload "
                        "does not begin as a materialization"
                    )
                fine_inputs = fine_node.args[0] if fine_node.args else ()
                payload_inputs = (
                    payload_node.args[0] if payload_node.args else ()
                )
                if not (
                    isinstance(fine_inputs, (tuple, list))
                    and isinstance(payload_inputs, (tuple, list))
                    and bool(fine_inputs)
                    and len(fine_inputs) == len(payload_inputs)
                    and all(isinstance(node, Node) for node in fine_inputs)
                    and all(isinstance(node, Node) for node in payload_inputs)
                ):
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: paired Alpha VSQ aggregate inputs "
                        "disagree"
                    )
                if (
                    len(fine_node.args) > 2
                    or len(payload_node.args) > 2
                    or set(fine_node.kwargs) - {"dim"}
                    or set(payload_node.kwargs) - {"dim"}
                ):
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: paired Alpha VSQ aggregate has "
                        "unsupported arguments"
                    )
                fine_dim = aggregate_dim(
                    fine_node, len(geometry.normalized_shape)
                )
                payload_dim = aggregate_dim(
                    payload_node, len(geometry.normalized_shape)
                )
                if fine_dim != payload_dim:
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: paired Alpha VSQ aggregate "
                        "dimensions disagree"
                    )
                for fine_input, payload_input in zip(
                    fine_inputs, payload_inputs
                ):
                    resolve_pair(
                        _vsq_tensor_ref(graph, fine_input, "fine scale"),
                        _vsq_tensor_ref(graph, payload_input, "payload"),
                    )
                result = bind_canonical_pair(_VsqPair(fine_ref, payload_ref))
                pair_by_refs[key] = result
                return result

            if (
                fine_target in _VSQ_LINEAR_VIEWS
                or payload_target in _VSQ_LINEAR_VIEWS
            ):
                if (
                    fine_target is not payload_target
                    or fine_target not in _VSQ_LINEAR_VIEWS
                    or fine_ref.path
                    or payload_ref.path
                ):
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: Alpha VSQ linear view is not paired"
                    )
                fine_source = fine_node.args[0] if fine_node.args else None
                payload_source = (
                    payload_node.args[0] if payload_node.args else None
                )
                if not isinstance(fine_source, Node) or not isinstance(
                    payload_source, Node
                ):
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: paired Alpha VSQ view has no "
                        "tensor source"
                    )
                if fine_target in {_RESHAPE, _VIEW}:
                    view_shape(fine_node, fine_ref)
                    view_shape(payload_node, payload_ref)
                else:
                    for node in (fine_node, payload_node):
                        kwargs = dict(node.kwargs)
                        memory_format = kwargs.pop(
                            "memory_format", torch.contiguous_format
                        )
                        if (
                            len(node.args) != 1
                            or kwargs
                            or memory_format != torch.contiguous_format
                        ):
                            raise AlphaStoragePlanError(
                                f"{fine_ref.name}: paired Alpha VSQ "
                                "contiguous requires the default memory format"
                            )
                fine_source_ref = _vsq_tensor_ref(
                    graph, fine_source, "fine scale"
                )
                payload_source_ref = _vsq_tensor_ref(
                    graph, payload_source, "payload"
                )
                if (
                    fine_relation.kind is not StorageRelationKind.ALIAS
                    or fine_relation.source != fine_source_ref
                    or payload_relation.kind is not StorageRelationKind.ALIAS
                    or payload_relation.source != payload_source_ref
                ):
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: paired Alpha VSQ view is not a "
                        "linear storage alias"
                    )
                source_fine, source_payload = _validate_alpha_vsq_pair(
                    graph, fine_source_ref, payload_source_ref
                )
                source_geometry = _alpha_vsq_block_geometry(
                    source_fine, source_payload, fine_source_ref.name
                )
                if source_geometry != geometry:
                    raise AlphaStoragePlanError(
                        f"{fine_ref.name}: paired Alpha VSQ view changes its "
                        "normalized block shape"
                    )
                result = resolve_pair(
                    fine_source_ref,
                    payload_source_ref,
                    allow_leaf=allow_leaf,
                )
                pair_by_refs[key] = result
                return result

            if (
                fine_target is operator.getitem
                or payload_target is operator.getitem
            ):
                raise AlphaStoragePlanError(
                    f"{fine_ref.name}: Alpha VSQ getitems are not the exact "
                    "fine/payload results of one terminal PVS"
                )
            if (
                fine_relation.kind is not StorageRelationKind.MATERIALIZE
                or payload_relation.kind is not StorageRelationKind.MATERIALIZE
                or fine_node.op not in {"get_attr", "placeholder"}
                or payload_node.op not in {"get_attr", "placeholder"}
                or not allow_leaf
            ):
                raise AlphaStoragePlanError(
                    f"{fine_ref.name}: unsupported Alpha VSQ producer or "
                    "unpaired transform"
                )
            result = bind_canonical_pair(_VsqPair(fine_ref, payload_ref))
            pair_by_refs[key] = result
            return result
        finally:
            resolving.remove(key)

    for fine_ref, payload_ref in pair_seeds:
        resolve_pair(fine_ref, payload_ref, allow_leaf=True)

    intended_relations = []

    def preflight_relation(
        result: ValueRef,
        kind: StorageRelationKind,
        source: ValueRef,
        component: Optional[str] = None,
    ) -> None:
        intended = (result, kind, source, component)
        previous_intended = next(
            (
                candidate
                for candidate in intended_relations
                if candidate[0] == result
            ),
            None,
        )
        if previous_intended is not None:
            if previous_intended != intended:
                raise AlphaStoragePlanError(
                    f"{result.name}: conflicting discovered Alpha VSQ relations"
                )
            return
        previous = graph.relation(plane, result)
        exact = (
            previous.kind is kind
            and previous.source == source
            and previous.physical_source is None
            and previous.transform is None
            and previous.component == component
        )
        if not exact and previous.kind is not StorageRelationKind.MATERIALIZE:
            raise AlphaStoragePlanError(
                f"{result.name}: conflicting Alpha VSQ storage relation"
            )
        intended_relations.append(intended)

    for coarse, source in coarse_aliases:
        preflight_relation(coarse, StorageRelationKind.ALIAS, source)
    for pair in canonical_pairs:
        preflight_relation(
            pair.fine_scale,
            StorageRelationKind.COMPONENT,
            pair.payload,
            "fine_scale",
        )
    for result, kind, source, component in intended_relations:
        _set_alpha_vsq_relation(
            graph,
            result,
            kind,
            plane=plane,
            source=source,
            component=component,
        )

    return _VsqDraft(
        pairs=tuple(canonical_pairs),
        artifacts=tuple(artifacts),
    )


def _register_alpha_vsq_backings(
    graph: StorageGraph,
    draft: _VsqDraft,
    *,
    plane: str = _SCRATCHPAD,
) -> None:
    """Register each final VSQ root once, after all sharing decisions."""
    plane = StoragePlane(plane).value
    registered = set()
    for pair in draft.pairs:
        relation = graph.relation(plane, pair.payload)
        if relation.kind is not StorageRelationKind.MATERIALIZE:
            continue
        fine_value, payload_value = _validate_alpha_vsq_pair(
            graph, pair.fine_scale, pair.payload
        )
        if pair.payload in registered:
            raise AlphaStoragePlanError(
                f"{pair.payload.name}: duplicate canonical Alpha VSQ backing"
            )
        registered.add(pair.payload)
        backing_size = math.ceil(payload_value.numel() * 4 / 8)
        try:
            graph.register_physical_component(
                pair.payload,
                payload_value,
                plane=plane,
                component="primary",
                dtype="int4",
                bits=4,
                backing_size_bytes=backing_size,
                layout=_ALPHA_VSQ_LAYOUT,
            )
            graph.register_physical_component(
                pair.fine_scale,
                fine_value,
                plane=plane,
                component="fine_scale",
                dtype="uint8",
                bits=8,
                backing_size_bytes=backing_size,
                layout=_ALPHA_VSQ_LAYOUT,
            )
        except (TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"{pair.payload.name}: invalid Alpha VSQ backing declaration"
            ) from error

    missing = []
    for pair in draft.pairs:
        root = graph.trace_storage_chain(pair.payload, plane)[-1]
        if root not in registered:
            missing.append((pair.fine_scale, pair.payload, root))
    if missing:
        details = ", ".join(
            f"{fine.name}->{payload.name} (root {root.name})"
            for fine, payload, root in missing
        )
        raise AlphaStoragePlanError(
            "Alpha VSQ relations have no materializing pair: " + details
        )


def _pad_vsq_artifact_axis0(
    value: torch.Tensor,
    extent: Optional[int],
) -> torch.Tensor:
    if value.ndim != 4:
        raise AlphaStoragePlanError(
            "Alpha vector scale artifact must be rank four"
        )
    if extent is None or extent == value.shape[0]:
        return value.contiguous()
    if extent < value.shape[0]:
        raise AlphaStoragePlanError("Alpha vector scale artifact cannot shrink")
    shape = list(value.shape)
    shape[0] = extent
    result = value.new_zeros(shape)
    result[: value.shape[0]] = value
    return result.contiguous()


def _declare_alpha_vsq_artifacts(
    graph: StorageGraph,
    draft: _VsqDraft,
    *,
    plane: str = _SCRATCHPAD,
) -> None:
    """Declare unplaced scale bytes before the first storage finalization."""
    plane = StoragePlane(plane).value
    for artifact in draft.artifacts:
        value = _logical_value(artifact.logical_tensor)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype is not torch.uint8
        ):
            raise AlphaStoragePlanError(
                f"{artifact.logical_tensor.name}: Alpha vector scale artifact "
                "must contain UINT8 codes"
            )
        encoded = _pad_vsq_artifact_axis0(
            value,
            artifact.encoded_axis0_extent,
        )
        name = f"{artifact.operation.name}.{artifact.field}"
        try:
            graph.register_serialized_artifact(
                artifact.logical_tensor,
                encoded,
                plane=plane,
                name=name,
                dtype="uint8",
                bits=8,
            )
        except (TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"{artifact.logical_tensor.name}: invalid Alpha vector scale "
                "artifact declaration"
            ) from error


def _freeze_alpha_vsq_storage(
    graph: StorageGraph,
    plan,
    draft: _VsqDraft,
    *,
    plane: str = _SCRATCHPAD,
) -> _VsqStorageFacts:
    """Freeze component identities without inventing an embedded byte offset."""
    plane = StoragePlane(plane).value
    if plan.graph is not graph:
        raise AlphaStoragePlanError(
            "Alpha VSQ facts must be frozen from their declaring graph"
        )

    logical_views = []
    pair_roots = set()
    for pair in draft.pairs:
        relation = plan.relation(plane, pair.fine_scale)
        if (
            relation.kind is not StorageRelationKind.COMPONENT
            or relation.source != pair.payload
            or relation.component != "fine_scale"
        ):
            raise AlphaStoragePlanError(
                f"{pair.fine_scale.name}: finalized Alpha VSQ relation changed"
            )
        fine_physical = plan.physical_id(plane, pair.fine_scale)
        payload_physical = plan.physical_id(plane, pair.payload)
        if fine_physical != payload_physical:
            raise AlphaStoragePlanError(
                f"{pair.fine_scale.name}: Alpha VSQ components were separated"
            )
        pair_roots.add(payload_physical)
        tensor = plan.physical_tensor(plane, payload_physical)
        try:
            primary = tensor.component("primary")
            fine = tensor.component("fine_scale")
        except KeyError as error:
            raise AlphaStoragePlanError(
                f"{payload_physical.name}: Alpha VSQ backing lost a component"
            ) from error
        if (
            tensor.layout != _ALPHA_VSQ_LAYOUT
            or {component.name for component in tensor.components}
            != {"primary", "fine_scale"}
            or primary.bits != 4
            or fine.bits != 8
            or str(primary.dtype) != "int4"
            or str(fine.dtype) != "uint8"
            or tensor.backing_size_bytes != primary.byte_size
        ):
            raise AlphaStoragePlanError(
                f"{payload_physical.name}: finalized Alpha VSQ representation "
                "does not match the target contract"
            )
        chain = plan.trace_storage_chain(pair.fine_scale, plane)
        transforms = tuple(
            transform
            for ref in chain
            for transform in (plan.relation(plane, ref).transform,)
            if transform is not None
        )
        logical_views.append(
            AlphaLogicalViewBinding(
                logical_tensor=pair.fine_scale,
                physical_tensor=payload_physical,
                component="fine_scale",
                transforms=transforms,
                byte_offset=None,
            )
        )

    layout_roots = {
        physical
        for physical, tensor in plan.physical_tensors[plane].items()
        if tensor.layout == _ALPHA_VSQ_LAYOUT
    }
    if pair_roots != layout_roots:
        missing = sorted(root.name for root in pair_roots - layout_roots)
        extra = sorted(root.name for root in layout_roots - pair_roots)
        raise AlphaStoragePlanError(
            "Alpha VSQ root coverage changed during finalization: "
            f"missing={missing}, extra={extra}"
        )

    serialized_uses = []
    for artifact in draft.artifacts:
        name = f"{artifact.operation.name}.{artifact.field}"
        try:
            frozen = plan.serialized_artifact(
                artifact.logical_tensor,
                plane,
                name,
            )
        except KeyError as error:
            raise AlphaStoragePlanError(
                f"{artifact.logical_tensor.name}: Alpha VSQ artifact was not "
                "finalized"
            ) from error
        if frozen.bits != 8 or str(frozen.dtype) != "uint8":
            raise AlphaStoragePlanError(
                f"{artifact.logical_tensor.name}: finalized Alpha VSQ artifact "
                "is not UINT8"
            )
        serialized_uses.append(
            AlphaSerializedUse(
                operation=artifact.operation,
                field=artifact.field,
                logical_tensor=artifact.logical_tensor,
                artifact=name,
            )
        )

    return _VsqStorageFacts(
        logical_views=tuple(logical_views),
        serialized_uses=tuple(serialized_uses),
    )


def _alpha_metadata(node: Node) -> dict:
    metadata = node.meta.get(_ALPHA_META)
    if not isinstance(metadata, dict):
        raise AlphaStoragePlanError(
            f"{node.name}: missing Magnet Alpha declaration metadata"
        )
    return metadata


def _resident_channels(
    graph: StorageGraph,
    shuffle_ref: ValueRef,
    *,
    plane: str,
    logical_channels: int,
) -> tuple[int, int, bool]:
    """Return resident channels, channel offset, and whether it was decided."""
    relation = graph.relation(plane, shuffle_ref)
    transform = relation.transform
    if transform is None or transform.kind != "alpha_resident_write":
        return logical_channels, 0, False
    resident = transform.parameter("resident_channels")
    offset = transform.parameter("channel_offset")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (resident, offset)
    ):
        raise AlphaStoragePlanError(
            f"{shuffle_ref.name}: resident channel pitch/offset is incomplete"
        )
    if offset < 0 or resident < logical_channels or offset + logical_channels > resident:
        raise AlphaStoragePlanError(
            f"{shuffle_ref.name}: resident channel window [{offset}, "
            f"{offset + logical_channels}) exceeds pitch {resident}"
        )
    return resident, offset, True


def _shuffle_copies_by_parent(model: GraphModule) -> dict[str, tuple[Node, ...]]:
    copies = defaultdict(list)
    for node in model.graph.nodes:
        if node.target is not _SHUFFLE_COPY:
            continue
        metadata = _alpha_metadata(node)
        if node.meta.get(_PHYSICAL_COPY_META) is not True:
            raise AlphaStoragePlanError(
                f"{node.name}: shuffle copy is not declared physical"
            )
        if metadata.get("kind") != "channel_shuffle_copy":
            raise AlphaStoragePlanError(
                f"{node.name}: invalid shuffle-copy kind declaration"
            )
        parent = metadata.get("logical_parent")
        if not isinstance(parent, str) or not parent:
            raise AlphaStoragePlanError(
                f"{node.name}: shuffle copy has no logical parent"
            )
        copies[parent].append(node)
    return {parent: tuple(nodes) for parent, nodes in copies.items()}


def _channel_split_layout(
    split: Node,
) -> Optional[tuple[Node, tuple[torch.Tensor, ...], tuple[int, ...]]]:
    """Return the static batch-one NCHW geometry of one channel split.

    A split that cannot be proved to preserve every non-channel extent is not
    an Alpha address calculation.  Returning ``None`` deliberately leaves its
    generic materializing relations intact.
    """
    if split.target not in _CHANNEL_SPLITS or not split.args:
        return None
    source = split.args[0]
    source_value = _value(source) if isinstance(source, Node) else None
    outputs = _value(split)
    if not isinstance(source_value, torch.Tensor):
        return None
    if not isinstance(outputs, (tuple, list)) or not outputs:
        return None
    if not all(isinstance(output, torch.Tensor) for output in outputs):
        return None

    dim = get_arg_value(split, 2, "dim", 0)
    if isinstance(dim, bool) or not isinstance(dim, int):
        return None
    source_shape = tuple(int(dimension) for dimension in source_value.shape)
    if len(source_shape) != 4:
        return None
    dim %= len(source_shape)
    if source_shape[0] != 1 or dim != 1:
        return None

    output_shapes = tuple(
        tuple(int(dimension) for dimension in output.shape)
        for output in outputs
    )
    non_channel_shape = source_shape[:dim] + source_shape[dim + 1 :]
    if any(
        shape[:dim] + shape[dim + 1 :] != non_channel_shape
        for shape in output_shapes
    ):
        return None
    channel_sizes = tuple(shape[dim] for shape in output_shapes)
    if any(size <= 0 for size in channel_sizes):
        return None
    if sum(channel_sizes) != source_shape[dim]:
        return None
    return source, tuple(outputs), channel_sizes


def _declare_alpha_split_storage(
    model: GraphModule,
    graph: StorageGraph,
    *,
    plane: str = _SCRATCHPAD,
    block_channels: int = OUTPUT_LANES,
    cluster_constraints: Optional[Mapping[ValueRef, int]] = None,
) -> tuple[_SplitDraft, ...]:
    """Share only channel splits addressable at an Alpha entry boundary.

    The split remains a logical multi-result operation.  Each result names a
    cumulative channel window of the source backing; no allocation or encoded
    address is decided here.  Unsupported widths and non-entry-aligned output
    sizes retain the shared storage graph's conservative materialization.  A
    previously decided target cluster may be supplied explicitly; otherwise
    the hardware's ordinary operand-width mode is authoritative.
    """
    plane = StoragePlane(plane).value
    if plane not in graph.planes:
        raise AlphaStoragePlanError(
            f"{plane}: split storage is not a plane of this graph"
        )
    if (
        isinstance(block_channels, bool)
        or not isinstance(block_channels, int)
        or block_channels <= 0
    ):
        raise AlphaStoragePlanError(
            "Alpha split block width must be a positive integer"
        )

    memory = GLOBAL.instance(GLOBAL.output_memory)
    drafts = []
    for split in model.graph.nodes:
        layout = _channel_split_layout(split)
        if layout is None:
            continue
        source, outputs, channel_sizes = layout
        # A get_attr can be a weight or a scale.  Static geometry alone cannot
        # prove which Alpha address contract applies, so keep independent
        # materializations exactly as the fork does.
        if source.op == "get_attr":
            continue

        source_refs = graph.output_refs(source)
        split_outputs = graph.output_refs(split)
        if len(source_refs) != 1 or len(split_outputs) != len(outputs):
            continue
        source_ref = source_refs[0]
        try:
            operand_bits = node_operand_bits(source)
            default_cluster = int(
                operand_mode(operand_bits)["banks_per_value"]
            )
        except (TypeError, ValueError):
            continue
        override = None
        if cluster_constraints is not None:
            override = cluster_constraints.get(source_ref)
        if override is not None and (
            isinstance(override, bool)
            or not isinstance(override, int)
            or override <= 0
            or int(memory.bank_count) % override
        ):
            raise AlphaStoragePlanError(
                f"{source.name}: invalid Alpha split cluster {override!r}"
            )
        cluster = default_cluster if override is None else override
        logical_entry_bits = cluster * int(memory.bits_per_bank_entry)
        if logical_entry_bits % operand_bits:
            continue
        channels_per_entry = logical_entry_bits // operand_bits
        entry_bytes = logical_entry_bits // 8
        if channels_per_entry <= 0 or any(
            channel_size % channels_per_entry
            for channel_size in channel_sizes
        ):
            continue

        source_shape = tuple(graph.logical[source_ref].shape)
        source_metadata = source.meta.get(_ALPHA_META)
        has_injected_copy = any(
            node.target is _SHUFFLE_COPY
            and (
                (node.args and node.args[0] is source)
                or (
                    isinstance(node.meta.get(_ALPHA_META), dict)
                    and node.meta[_ALPHA_META].get("logical_parent")
                    == source.name
                )
            )
            for node in model.graph.nodes
        )
        deferred_secondary = None
        if (
            source.target is _SHUFFLE
            and isinstance(source_metadata, dict)
            and source_metadata.get("kind") == "channel_shuffle"
            and source_metadata.get("groups") == 2
            and not has_injected_copy
            and len(split_outputs) == 2
            and channel_sizes == (block_channels, block_channels)
            and len(source_shape) == 4
            and source_shape[1] == 2 * block_channels
        ):
            # A two-entry shuffle has no injected copy operations.  Its
            # second split result is subsequently proven to live in the
            # shuffle's auxiliary backing.  Leave exactly that one result at
            # generic materialization so the shared StorageGraph sees one
            # target decision, not an overwritten decision.
            deferred_secondary = split_outputs[1]

        expected = []
        channel_offset = 0
        for output_ref, channel_size in zip(split_outputs, channel_sizes):
            transform = ViewTransform(
                "alpha_channel_split",
                parameters=(
                    ("channel_offset", channel_offset),
                    ("channel_size", channel_size),
                ),
            )
            previous = graph.relation(plane, output_ref)
            same_relation = (
                previous.kind is StorageRelationKind.VIEW
                and previous.source == source_ref
                and previous.physical_source is None
                and previous.transform == transform
            )
            if (
                previous.kind is not StorageRelationKind.MATERIALIZE
                and not same_relation
            ):
                raise AlphaStoragePlanError(
                    f"{output_ref.name}: conflicting Alpha split relation"
                )
            expected.append(
                (
                    output_ref,
                    transform,
                    same_relation,
                    output_ref == deferred_secondary,
                )
            )
            channel_offset += channel_size

        for output_ref, transform, same_relation, deferred in expected:
            if not same_relation and not deferred:
                graph.set_relation(
                    output_ref,
                    StorageRelationKind.VIEW,
                    plane=plane,
                    source=source_ref,
                    transform=transform,
                )
        drafts.append(
            _SplitDraft(
                split=split,
                source=source_ref,
                outputs=split_outputs,
                channel_sizes=channel_sizes,
                operand_bits=operand_bits,
                cluster=cluster,
                channels_per_entry=channels_per_entry,
                entry_bytes=entry_bytes,
                deferred_secondary=deferred_secondary,
            )
        )
    return tuple(drafts)


def _concat_downstream_aliases(
    cat: Node,
    graph: StorageGraph,
    *,
    plane: str,
) -> tuple[Node, ...]:
    """Return the fork-supported single-user address chain after a concat."""

    def successor(current: Node) -> Optional[Node]:
        if len(current.users) != 1:
            return None
        user = next(iter(current.users))
        current_outputs = graph.output_refs(current)
        user_outputs = graph.output_refs(user)
        if len(current_outputs) == 1 and len(user_outputs) == 1:
            relation = graph.relation(plane, user_outputs[0])
            if (
                relation.kind
                in {StorageRelationKind.ALIAS, StorageRelationKind.VIEW}
                and relation.source == current_outputs[0]
            ):
                return user
        if user.target is _STACK:
            tensors = user.args[0] if user.args else None
            dim = get_arg_value(user, 1, "dim", 0)
            if (
                isinstance(tensors, (tuple, list))
                and len(tensors) == 1
                and tensors[0] is current
                and dim == 0
            ):
                return user
        return None

    aliases = []
    visited = set()
    current = cat
    while current is not None:
        if current in visited:
            names = " -> ".join(node.name for node in (*aliases, current))
            raise AlphaStoragePlanError(f"value-relation cycle: {names}")
        visited.add(current)
        aliases.append(current)
        current = successor(current)
    return tuple(aliases)


def _concat_channel_axis(
    node: Node,
    shapes: tuple[tuple[int, ...], ...],
    aliases: tuple[Node, ...],
) -> Optional[int]:
    """Recognize a batch-one NCHW channel concat or its stack/view form."""
    rank = len(shapes[0])
    dim = get_arg_value(node, 1, "dim", 0)
    if isinstance(dim, bool) or not isinstance(dim, int):
        return None
    if node.target is _CAT:
        dim %= rank
        if rank == 4 and dim == 1 and all(shape[0] == 1 for shape in shapes):
            return 1
        return None

    if dim < 0:
        dim += rank + 1
    if (
        rank != 4
        or dim != 0
        or shapes[0][0] != 1
        or any(shape != shapes[0] for shape in shapes[1:])
        or len(aliases) == 1
    ):
        return None
    batch, channels, height, width = shapes[0]
    expected = (batch, len(shapes) * channels, height, width)
    return 1 if _shape(aliases[-1]) == expected else None


def _concat_layout(
    node: Node,
    graph: StorageGraph,
    *,
    plane: str,
) -> Optional[_ConcatLayout]:
    if node.target not in {_CAT, _STACK} or not node.args:
        return None
    inputs = node.args[0]
    if (
        not isinstance(inputs, (tuple, list))
        or len(inputs) < 2
        or not all(isinstance(input_node, Node) for input_node in inputs)
    ):
        return None
    inputs = tuple(inputs)
    shapes = tuple(_shape(input_node) for input_node in inputs)
    if (
        not shapes
        or any(not shape for shape in shapes)
        or any(len(shape) != len(shapes[0]) for shape in shapes)
        or any(shape[1] <= 0 for shape in shapes if len(shape) > 1)
    ):
        return None
    aliases = _concat_downstream_aliases(node, graph, plane=plane)
    channel_axis = _concat_channel_axis(node, shapes, aliases)
    if channel_axis is None:
        return None
    reference = shapes[0]
    if any(
        extent != reference[index]
        for shape in shapes[1:]
        for index, extent in enumerate(shape)
        if index != channel_axis
    ):
        return None
    try:
        output_shape = tuple(graph.logical[_single_ref(graph, node)].shape)
    except (AlphaStoragePlanError, KeyError, TypeError, ValueError):
        return None
    if node.target is _CAT:
        expected_output = list(reference)
        expected_output[channel_axis] = sum(
            shape[channel_axis] for shape in shapes
        )
        expected_output = tuple(expected_output)
    else:
        # _concat_channel_axis admits only a dim-0 stack here.
        expected_output = (len(inputs), *reference)
    if output_shape != expected_output:
        return None
    return _ConcatLayout(
        inputs=inputs,
        shapes=shapes,
        channel_axis=channel_axis,
        aliases=aliases,
    )


def _resident_channel_granularity(
    graph: StorageGraph,
    logical: ValueRef,
    *,
    block_channels: int,
    vsq_payloads: frozenset[ValueRef] = frozenset(),
    cluster_constraints: Optional[Mapping[ValueRef, int]] = None,
) -> Optional[tuple[int, int]]:
    """Return ``(bits, channels)`` for one Alpha-resident channel slot."""
    if logical in vsq_payloads:
        return 4, VSQ_BLOCK_VALUES
    try:
        operand_bits = _bits_for_ref(graph, logical)
        default_cluster = int(operand_mode(operand_bits)["banks_per_value"])
    except (AlphaStoragePlanError, KeyError, TypeError, ValueError):
        return None
    memory = GLOBAL.instance(GLOBAL.output_memory)
    override = None
    if cluster_constraints is not None:
        override = cluster_constraints.get(logical)
    if override is not None and (
        isinstance(override, bool)
        or not isinstance(override, int)
        or override not in ALPHA_BANK_CLUSTERS
        or int(memory.bank_count) % override
    ):
        raise AlphaStoragePlanError(
            f"{logical.name}: invalid Alpha resident cluster {override!r}"
        )
    cluster = default_cluster if override is None else override
    logical_entry_bits = cluster * int(memory.bits_per_bank_entry)
    if logical_entry_bits % operand_bits:
        return None
    entry_channels = logical_entry_bits // operand_bits
    if operand_bits == 8 and cluster == 1:
        entry_channels = max(entry_channels, block_channels)
    return operand_bits, entry_channels


def _trace_resident_write_owner(
    graph: StorageGraph,
    input_node: Node,
    *,
    plane: str,
    expected_shape: tuple[int, ...],
) -> Optional[tuple[Node, ValueRef, tuple[ValueRef, ...]]]:
    """Resolve a producer through exact batch-one NCHW storage aliases."""
    if (
        len(expected_shape) != 4
        or expected_shape[0] != 1
        or any(extent <= 0 for extent in expected_shape)
    ):
        return None
    try:
        start = _payload_ref(graph, input_node)
        chain = graph.trace_storage_chain(
            start,
            plane,
            follow_kinds={
                StorageRelationKind.ALIAS,
                StorageRelationKind.VIEW,
                StorageRelationKind.COMPONENT,
            },
            follow_copy=lambda ref: ref.node.target is _SHUFFLE_COPY,
        )
    except (AlphaStoragePlanError, KeyError, ValueError):
        return None
    if not chain:
        return None
    try:
        chain_shapes = tuple(
            tuple(graph.logical[ref].shape) for ref in chain
        )
    except (KeyError, TypeError, ValueError):
        return None
    if any(shape != expected_shape for shape in chain_shapes):
        # A direct resident writer may be reached only through aliases that
        # preserve the complete producer geometry.  In particular, equal-
        # numel reshapes are not evidence that an NCHW channel slot remains
        # addressable with the producer's row pitch.
        return None
    owner_ref = chain[-1]
    owner = owner_ref.node
    try:
        if _payload_ref(graph, owner) != owner_ref:
            return None
    except AlphaStoragePlanError:
        return None
    return owner, owner_ref, chain[:-1]


def _set_alpha_cluster_draft(
    drafts: dict[ValueRef, _ClusterDraft],
    logical: ValueRef,
    cluster: int,
) -> None:
    """Record one logical cluster decision without permitting replacement."""
    if not isinstance(logical, ValueRef):
        raise TypeError("Alpha logical cluster constraints require ValueRefs")
    if (
        isinstance(cluster, bool)
        or not isinstance(cluster, int)
        or cluster not in ALPHA_BANK_CLUSTERS
    ):
        raise AlphaStoragePlanError(
            f"{logical.name}: unsupported Alpha bank cluster {cluster!r}"
        )
    previous = drafts.get(logical)
    if previous is not None and previous.cluster != cluster:
        raise AlphaStoragePlanError(
            f"{logical.name}: conflicting Alpha bank clusters "
            f"{previous.cluster} and {cluster}"
        )
    if previous is None:
        drafts[logical] = _ClusterDraft(logical, cluster)


def _alpha_matrix_input_lanes(config, input_bits: int) -> int:
    if config is None:
        lanes = MATRIX_UNIT.unroll("IC", input_bits)
    else:
        input_lanes = getattr(config, "input_lanes", None)
        if not callable(input_lanes):
            raise AlphaStoragePlanError(
                "Alpha configuration has no matrix input_lanes contract"
            )
        try:
            lanes = input_lanes(input_bits)
        except (TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"Alpha configuration cannot resolve {input_bits}-bit "
                "matrix input lanes"
            ) from error
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
        raise AlphaStoragePlanError(
            "Alpha matrix input lanes must be a positive integer, got "
            f"{lanes!r}"
        )
    return lanes


def _supports_partial_matrix_output(
    config,
    channels: int,
    input_bits: int,
) -> bool:
    """Query the selected machine for one compact resident output."""
    if config is None:
        lanes = _alpha_matrix_output_lanes(None, input_bits)
        try:
            alignment = int(operand_mode(input_bits)["channel_alignment"])
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"Alpha has no {input_bits}-bit output alignment"
            ) from error
        return 0 < channels < lanes and channels % alignment == 0

    supports = getattr(config, "supports_partial_matrix_output", None)
    if not callable(supports):
        raise AlphaStoragePlanError(
            "Alpha configuration has no partial matrix-output contract"
        )
    try:
        return bool(supports(channels, input_bits))
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"Alpha configuration cannot validate compact K={channels} at "
            f"{input_bits} bits"
        ) from error


def _compact_matrix_output_cluster(
    graph: StorageGraph,
    constraint: AlphaOperationConstraint,
    config,
) -> Optional[int]:
    """Return the fork's compact INT8 matrix-output bank width."""
    if constraint.kind != "gemm" or constraint.output_bits != 8:
        return None
    output_shape = tuple(graph.logical[constraint.output].shape)
    if len(output_shape) != 4:
        return None
    try:
        scheduled_input = constraint.workload.require_operand("input")
    except ValueError:
        return None
    input_bits = scheduled_input.bits
    if (
        isinstance(input_bits, bool)
        or not isinstance(input_bits, int)
        or input_bits != 8
    ):
        return None
    channels = int(output_shape[1])
    lanes = _alpha_matrix_output_lanes(config, input_bits)
    if channels <= 0 or channels >= lanes:
        return None
    if not _supports_partial_matrix_output(config, channels, input_bits):
        return None

    memory = GLOBAL.instance(GLOBAL.output_memory)
    cluster, remainder = divmod(
        channels * constraint.output_bits,
        int(memory.bits_per_bank_entry),
    )
    if remainder or cluster not in ALPHA_BANK_CLUSTERS:
        return None
    return cluster


def _is_host_matrix_input(
    model: GraphModule,
    logical: ValueRef,
    owner: Node,
    by_owner: Mapping[Node, AlphaOperationConstraint],
) -> bool:
    """Prove that a value is host-originated and used only as a GEMM input."""
    if logical.path:
        return False
    users = tuple(logical.node.users)
    if owner not in users:
        return False
    for user in users:
        constraint = by_owner.get(user)
        input_binding = (
            constraint.operand("input") if constraint is not None else None
        )
        if (
            constraint is None
            or constraint.kind != "gemm"
            or input_binding is None
            or input_binding.logical_tensor != logical
        ):
            return False

    pending = [logical.node]
    visited = set()
    reaches_placeholder = False
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op == "placeholder":
            reaches_placeholder = True
            continue
        if node.op == "get_attr":
            continue
        if node in by_owner:
            return False
        pending.extend(node.all_input_nodes)
    return reaches_placeholder


def _static_pair(value, default: int) -> Optional[tuple[int, int]]:
    if value is None:
        return (default, default)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            pair = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        return pair if all(item >= 0 for item in pair) else None
    return None


def _convolution_geometry_arg(
    implementation: Node,
    name: str,
    default,
):
    positions = (
        {"stride": 2, "padding": 3, "dilation": 4, "groups": 5}
        if implementation.target is _CONV2D_INT8
        else {"stride": 3, "padding": 4, "dilation": 5, "groups": 6}
        if implementation.target is _CONV2D_VSQ
        else None
    )
    if positions is None or name not in positions:
        raise AlphaStoragePlanError(
            f"{implementation.name}: no Alpha convolution field {name!r}"
        )
    return get_arg_value(implementation, positions[name], name, default)


def _host_matrix_input_cluster(
    model: GraphModule,
    graph: StorageGraph,
    constraint: AlphaOperationConstraint,
    by_owner: Mapping[Node, AlphaOperationConstraint],
    config,
) -> Optional[ValueRef]:
    """Recognize the retained no-repacking host input contract."""
    if constraint.kind != "gemm" or constraint.implementation.target is not _CONV2D_INT8:
        return None
    input_binding = constraint.operand("input")
    weight_binding = constraint.operand("weight")
    if (
        input_binding is None
        or weight_binding is None
        or input_binding.bits != 8
        or weight_binding.bits != 8
        or not _is_host_matrix_input(
            model,
            input_binding.logical_tensor,
            constraint.owner,
            by_owner,
        )
    ):
        return None

    input_shape = tuple(graph.logical[input_binding.logical_tensor].shape)
    weight_shape = tuple(graph.logical[weight_binding.logical_tensor].shape)
    output_shape = tuple(graph.logical[constraint.output].shape)
    if any(len(shape) != 4 for shape in (input_shape, weight_shape, output_shape)):
        return None
    implementation = constraint.implementation
    stride = _static_pair(
        _convolution_geometry_arg(implementation, "stride", None), 1
    )
    padding = _static_pair(
        _convolution_geometry_arg(implementation, "padding", None), 0
    )
    dilation = _static_pair(
        _convolution_geometry_arg(implementation, "dilation", None), 1
    )
    if None in {stride, padding, dilation}:
        return None
    if (
        int(input_shape[1]) == _alpha_matrix_input_lanes(config, 8)
        and int(input_shape[2]) == 1
        and int(output_shape[2]) == 1
        and tuple(weight_shape[-2:]) == (1, 1)
        and stride == (1, 1)
        and padding == (0, 0)
        and dilation == (1, 1)
    ):
        return input_binding.logical_tensor
    return None


def _supports_partial_matrix_input(config, channels: int) -> bool:
    if config is None:
        return channels in PARTIAL_MATRIX_INPUT_CHANNELS
    supports = getattr(config, "supports_partial_matrix_input", None)
    if not callable(supports):
        raise AlphaStoragePlanError(
            "Alpha configuration has no partial matrix-input contract"
        )
    try:
        return bool(supports(channels))
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"Alpha configuration cannot validate compact C={channels}"
        ) from error


def _alpha_matrix_input_resident_channels(
    config,
    channels: int,
    operand_bits: int,
) -> int:
    """Return Alpha's resident matrix-input extent without changing logic."""
    _alpha_matrix_input_lanes(config, operand_bits)
    resolver = (
        None
        if config is None
        else getattr(config, "matrix_input_resident_channels", None)
    )
    try:
        resident = (
            resolver(channels, operand_bits)
            if callable(resolver)
            else 8
            if operand_bits == 8 and channels == 3
            else channels
        )
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"Alpha configuration cannot resolve resident C={channels} at "
            f"{operand_bits} bits"
        ) from error
    if (
        isinstance(resident, bool)
        or not isinstance(resident, int)
        or resident < channels
    ):
        raise AlphaStoragePlanError(
            f"invalid Alpha resident input channels {resident!r} for "
            f"C={channels}"
        )
    return resident


def _alpha_matrix_weight_physical_channels(
    config,
    resident_channels: int,
    operand_bits: int,
) -> int:
    """Return the fork's target-owned matrix-weight channel stride."""
    resolver = (
        None
        if config is None
        else getattr(config, "matrix_weight_physical_channels", None)
    )
    try:
        if callable(resolver):
            physical = resolver(resident_channels, operand_bits)
        else:
            physical = resident_channels
            if operand_bits == 8:
                physical = (
                    8
                    if resident_channels == 3
                    else math.ceil(resident_channels / 4) * 4
                )
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            "Alpha configuration cannot resolve matrix-weight physical "
            f"channels for C={resident_channels} at {operand_bits} bits"
        ) from error
    if (
        isinstance(physical, bool)
        or not isinstance(physical, int)
        or physical < resident_channels
    ):
        raise AlphaStoragePlanError(
            f"invalid Alpha physical weight channels {physical!r} for "
            f"resident C={resident_channels}"
        )
    return physical


def _pad_tensor_axis(
    value: torch.Tensor,
    axis: int,
    extent: int,
) -> torch.Tensor:
    """Zero-pad one tensor axis while preserving all existing values."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("Alpha physical payload must be a tensor")
    if axis < 0 or axis >= value.ndim:
        raise ValueError("Alpha physical payload axis is outside the tensor")
    current = int(value.shape[axis])
    if extent < current:
        raise ValueError("Alpha physical payload cannot shrink")
    if extent == current:
        return value.contiguous()
    shape = list(value.shape)
    shape[axis] = extent
    result = value.new_zeros(shape)
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, current)
    result[tuple(slices)] = value
    return result.contiguous()


def _physical_component_name(
    graph: StorageGraph,
    logical: ValueRef,
    plane: str,
) -> str:
    relation = graph.relation(plane, logical)
    return relation.component or "primary"


def _rank_four_logical_shape(
    graph: StorageGraph,
    logical: ValueRef,
    label: str,
) -> tuple[int, int, int, int]:
    # A path names one tensor *within* a tuple-valued FX result; it does not
    # make the ValueRef itself tuple-valued.  Node-level logical-shape metadata
    # cannot describe such a component, so use the StorageGraph's path-aware
    # logical tensor entry.  Scalar-result nodes may still carry a preserved
    # pre-padding extent, which remains authoritative for that case.
    if logical.path:
        shape = graph.logical[logical].shape
    else:
        shape = get_hardware_logical_shape(logical.node)
        if not shape:
            shape = graph.logical[logical].shape
    try:
        normalized = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"{logical.name}: {label} has a non-concrete logical shape"
        ) from error
    if len(normalized) != 4 or any(dimension <= 0 for dimension in normalized):
        raise AlphaStoragePlanError(
            f"{logical.name}: {label} must have a positive rank-four shape"
        )
    return normalized


def _register_alpha_gemm_component(
    graph: StorageGraph,
    logical: ValueRef,
    value: torch.Tensor,
    *,
    bits: int,
    layout: str,
    plane: str,
) -> str:
    component = _physical_component_name(graph, logical, plane)
    backing_size = math.ceil(value.numel() * bits / 8)
    try:
        graph.register_physical_component(
            logical,
            value,
            plane=plane,
            component=component,
            dtype=graph.logical[logical].dtype,
            bits=bits,
            backing_size_bytes=backing_size,
            layout=layout,
        )
    except (TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"{logical.name}: invalid Alpha GEMM {layout} representation"
        ) from error
    return component


def _declare_host_matrix_input(
    model: GraphModule,
    graph: StorageGraph,
    constraint: AlphaOperationConstraint,
    by_owner: Mapping[Node, AlphaOperationConstraint],
    input_ref: ValueRef,
    input_bits: int,
    weight_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    config,
    *,
    plane: str,
) -> Optional[_MaterializedInputDraft]:
    """Declare one exact fork-supported host input representation."""
    if input_bits != 8 or not _is_host_matrix_input(
        model,
        input_ref,
        constraint.owner,
        by_owner,
    ):
        return None
    input_shape = tuple(int(value) for value in graph.logical[input_ref].shape)
    if len(input_shape) != 4 or any(value <= 0 for value in input_shape):
        raise AlphaStoragePlanError(
            f"{input_ref.name}: host matrix input must be positive rank four"
        )
    implementation = constraint.implementation
    kernel = tuple(weight_shape[-2:])
    stride = _static_pair(
        _convolution_geometry_arg(implementation, "stride", None), 1
    )
    padding = _static_pair(
        _convolution_geometry_arg(implementation, "padding", None), 0
    )
    dilation = _static_pair(
        _convolution_geometry_arg(implementation, "dilation", None), 1
    )
    if None in {stride, padding, dilation}:
        raise AlphaStoragePlanError(
            f"{constraint.owner.name}: host GEMM geometry is not static"
        )
    valid_channels = int(input_shape[1])
    value = _logical_value(input_ref)

    if (
        valid_channels == _alpha_matrix_input_lanes(config, input_bits)
        and input_shape[2] == 1
        and output_shape[2] == 1
        and kernel == (1, 1)
        and stride == (1, 1)
        and padding == (0, 0)
        and dilation == (1, 1)
    ):
        if not isinstance(value, torch.Tensor):
            raise AlphaStoragePlanError(
                f"{input_ref.name}: host matrix input has no tensor value"
            )
        _register_alpha_gemm_component(
            graph,
            input_ref,
            value.contiguous(),
            bits=input_bits,
            layout=_ALPHA_DENSE_LAYOUT,
            plane=plane,
        )
        return None

    resident_channels = _alpha_matrix_input_resident_channels(
        config,
        valid_channels,
        input_bits,
    )
    if resident_channels != valid_channels:
        groups = _convolution_geometry_arg(implementation, "groups", 1)
        if (
            not _supports_partial_matrix_input(config, valid_channels)
            or weight_shape[1] != valid_channels
            or isinstance(groups, bool)
            or groups != 1
        ):
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: invalid compact Alpha matrix "
                f"input contract for C={valid_channels}"
            )
        if not isinstance(value, torch.Tensor):
            raise AlphaStoragePlanError(
                f"{input_ref.name}: host matrix input has no tensor value"
            )
        physical = _pad_tensor_axis(value, 1, resident_channels)
        component = _register_alpha_gemm_component(
            graph,
            input_ref,
            physical,
            bits=input_bits,
            layout=_ALPHA_PADDED_INPUT_LAYOUT,
            plane=plane,
        )
        return _MaterializedInputDraft(
            logical_tensor=input_ref,
            component=component,
            physical_shape=tuple(int(dimension) for dimension in physical.shape),
            padding=(0, 0, 0, 0),
            valid_channels=valid_channels,
            resident_channels=resident_channels,
            serialize_channels=True,
        )

    if (
        valid_channels != 1
        or not _supports_partial_matrix_input(config, valid_channels)
        or kernel != (3, 3)
        or stride != (2, 2)
        or dilation != (1, 1)
    ):
        return None
    effective_width = dilation[1] * (kernel[1] - 1) + 1
    needed_width = max(
        (output_shape[3] - 1) * stride[1]
        + effective_width
        - input_shape[3],
        0,
    )
    pad_left = min(padding[1], needed_width)
    pad_right = needed_width - pad_left
    if (pad_left, pad_right) != (1, 0):
        return None
    if not isinstance(value, torch.Tensor):
        raise AlphaStoragePlanError(
            f"{input_ref.name}: host matrix input has no tensor value"
        )
    physical = torch.nn.functional.pad(
        value,
        (pad_left, 0, 0, 0),
        mode="constant",
        value=0,
    ).contiguous()
    component = _register_alpha_gemm_component(
        graph,
        input_ref,
        physical,
        bits=input_bits,
        layout=_ALPHA_PADDED_INPUT_LAYOUT,
        plane=plane,
    )
    return _MaterializedInputDraft(
        logical_tensor=input_ref,
        component=component,
        physical_shape=tuple(int(dimension) for dimension in physical.shape),
        padding=(pad_left, 0, 0, 0),
        valid_channels=valid_channels,
        resident_channels=valid_channels,
        serialize_channels=False,
    )


def _declare_alpha_gemm_storage(
    model: GraphModule,
    graph: StorageGraph,
    operation_constraints: tuple[AlphaOperationConstraint, ...],
    vsq: _VsqDraft,
    *,
    config=None,
    plane: str = _SCRATCHPAD,
) -> tuple[_GemmDraft, ...]:
    """Declare exact matrix payload representations before finalization."""
    plane = StoragePlane(plane).value
    by_owner = {
        constraint.owner: constraint for constraint in operation_constraints
    }
    if len(by_owner) != len(operation_constraints):
        raise AlphaStoragePlanError(
            "Alpha GEMM storage received duplicate executable owners"
        )
    vsq_payloads = frozenset(pair.payload for pair in vsq.pairs)
    vsq_scales = {pair.payload: pair.fine_scale for pair in vsq.pairs}
    artifacts = {
        (artifact.operation, artifact.field): artifact for artifact in vsq.artifacts
    }
    drafts = []
    for constraint in operation_constraints:
        if constraint.kind != "gemm":
            continue
        if constraint.implementation.target not in {_CONV2D_INT8, _CONV2D_VSQ}:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: unsupported Alpha GEMM implementation"
            )
        input_binding = constraint.operand("input")
        weight_binding = constraint.operand("weight")
        if input_binding is None or weight_binding is None:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: GEMM operands did not resolve"
            )
        input_ref = input_binding.logical_tensor
        weight_ref = weight_binding.logical_tensor
        weight_value = _logical_value(weight_ref)
        if not isinstance(weight_value, torch.Tensor) or weight_value.ndim != 4:
            raise AlphaStoragePlanError(
                f"{weight_ref.name}: Alpha GEMM weight must be rank four"
            )
        resident_shape = tuple(int(dimension) for dimension in weight_value.shape)
        logical_shape = _rank_four_logical_shape(
            graph,
            weight_ref,
            "logical Alpha GEMM weight",
        )
        valid_k, valid_c = logical_shape[:2]
        resident_k, resident_c = resident_shape[:2]
        physical_c = _alpha_matrix_weight_physical_channels(
            config,
            resident_c,
            weight_binding.bits,
        )
        physical_k = resident_k
        if not (
            0 < valid_c <= resident_c <= physical_c
            and 0 < valid_k <= resident_k <= physical_k
        ):
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: invalid logical/resident/physical "
                "Alpha GEMM weight extents: "
                f"C={valid_c}/{resident_c}/{physical_c}, "
                f"K={valid_k}/{resident_k}/{physical_k}"
            )
        component = _physical_component_name(graph, weight_ref, plane)
        if weight_binding.bits == 8:
            physical_weight = _pad_tensor_axis(weight_value, 1, physical_c)
            component = _register_alpha_gemm_component(
                graph,
                weight_ref,
                physical_weight,
                bits=weight_binding.bits,
                layout=_ALPHA_DENSE_LAYOUT,
                plane=plane,
            )
        elif weight_binding.bits == 4:
            if weight_ref not in vsq_payloads or physical_c != resident_c:
                raise AlphaStoragePlanError(
                    f"{weight_ref.name}: INT4 GEMM weight is not one exact "
                    "Alpha VSQ backing"
                )
        else:
            raise AlphaStoragePlanError(
                f"{weight_ref.name}: unsupported Alpha GEMM weight width "
                f"{weight_binding.bits}"
            )

        output_shape = tuple(int(value) for value in graph.logical[constraint.output].shape)
        if len(output_shape) != 4 or any(value <= 0 for value in output_shape):
            raise AlphaStoragePlanError(
                f"{constraint.output.name}: Alpha GEMM output must be rank four"
            )
        materialized = _declare_host_matrix_input(
            model,
            graph,
            constraint,
            by_owner,
            input_ref,
            input_binding.bits,
            resident_shape,
            output_shape,
            config,
            plane=plane,
        )
        physical_input_shape = (
            tuple(graph.logical[input_ref].shape)
            if materialized is None
            else materialized.physical_shape
        )
        issued_resident_c = (
            resident_c
            if materialized is None
            else materialized.resident_channels
        )
        if len(physical_input_shape) != 4:
            raise AlphaStoragePlanError(
                f"{input_ref.name}: Alpha GEMM input must be rank four"
            )
        stride = _static_pair(
            _convolution_geometry_arg(
                constraint.implementation, "stride", None
            ),
            1,
        )
        torch_padding = _static_pair(
            _convolution_geometry_arg(
                constraint.implementation, "padding", None
            ),
            0,
        )
        dilation = _static_pair(
            _convolution_geometry_arg(
                constraint.implementation, "dilation", None
            ),
            1,
        )
        if None in {stride, torch_padding, dilation}:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: convolution geometry is not static"
            )
        pads = []
        kernel = resident_shape[-2:]
        for axis in range(2):
            effective = dilation[axis] * (kernel[axis] - 1) + 1
            needed = max(
                (output_shape[axis + 2] - 1) * stride[axis]
                + effective
                - int(physical_input_shape[axis + 2]),
                0,
            )
            before = min(torch_padding[axis], needed)
            pads.extend((before, needed - before))
        input_lanes = _alpha_matrix_input_lanes(config, input_binding.bits)
        output_lanes = _alpha_matrix_output_lanes(config, input_binding.bits)

        weight_scale_ref = None
        scale_valid_k = None
        scale_encoded_k = None
        if constraint.implementation.target is _CONV2D_VSQ:
            scale_node = _implementation_operand(
                model,
                constraint.owner,
                constraint.implementation,
                2,
                "weight_scale",
            )
            weight_scale_ref = _vsq_tensor_ref(
                graph,
                scale_node,
                "weight fine scale",
            )
            if vsq_scales.get(weight_ref) != weight_scale_ref:
                raise AlphaStoragePlanError(
                    f"{constraint.owner.name}: GEMM weight and fine scale "
                    "do not share the declared VSQ pair"
                )
            scale_value = _logical_value(weight_scale_ref)
            scale_shape = _rank_four_logical_shape(
                graph,
                weight_scale_ref,
                "logical Alpha GEMM weight scale",
            )
            if not isinstance(scale_value, torch.Tensor) or scale_value.ndim != 4:
                raise AlphaStoragePlanError(
                    f"{weight_scale_ref.name}: Alpha GEMM weight scale must "
                    "be rank four"
                )
            scale_valid_k = scale_shape[0]
            scale_encoded_k = math.ceil(scale_value.shape[0] / output_lanes) * output_lanes
            artifact = artifacts.get(
                (constraint.owner, "weight_vector_scale_payload")
            )
            if artifact is None or artifact.encoded_axis0_extent != scale_encoded_k:
                raise AlphaStoragePlanError(
                    f"{constraint.owner.name}: GEMM weight-scale artifact "
                    "does not match its encoded extent"
                )

        drafts.append(
            _GemmDraft(
                constraint=constraint,
                input=input_ref,
                weight=weight_ref,
                output=constraint.output,
                weight_component=component,
                input_bits=input_binding.bits,
                weight_bits=weight_binding.bits,
                valid_input_channels=valid_c,
                resident_input_channels=issued_resident_c,
                physical_weight_input_channels=physical_c,
                valid_output_channels=valid_k,
                resident_output_channels=resident_k,
                physical_weight_output_channels=physical_k,
                pads=tuple(pads),
                padding_c=(-issued_resident_c) % input_lanes,
                padding_k=(-resident_k) % output_lanes,
                materialized_input=materialized,
                weight_scale=weight_scale_ref,
                weight_scale_valid_output_channels=scale_valid_k,
                weight_scale_encoded_output_channels=scale_encoded_k,
            )
        )
    return tuple(drafts)


def _freeze_alpha_gemm_storage(
    graph: StorageGraph,
    plan,
    drafts: tuple[_GemmDraft, ...],
    *,
    plane: str = _SCRATCHPAD,
) -> _GemmStorageFacts:
    """Bind prepared GEMM facts to final shared physical identities."""
    plane = StoragePlane(plane).value
    if plan.graph is not graph:
        raise AlphaStoragePlanError(
            "Alpha GEMM facts must be frozen from their declaring graph"
        )
    bindings = []
    payload_uses = []
    for draft in drafts:
        input_physical = plan.physical_id(plane, draft.input)
        weight_physical = plan.physical_id(plane, draft.weight)
        output_physical = plan.physical_id(plane, draft.output)
        weight_tensor = plan.physical_tensor(plane, weight_physical)
        try:
            weight_component = weight_tensor.component(draft.weight_component)
        except KeyError as error:
            raise AlphaStoragePlanError(
                f"{weight_physical.name}: Alpha GEMM weight component was lost"
            ) from error
        expected_weight_shape = (
            draft.physical_weight_output_channels,
            draft.physical_weight_input_channels,
            *tuple(graph.logical[draft.weight].shape[-2:]),
        )
        expected_layout = (
            _ALPHA_VSQ_LAYOUT if draft.weight_bits == 4 else _ALPHA_DENSE_LAYOUT
        )
        if (
            tuple(weight_component.shape) != expected_weight_shape
            or weight_component.bits != draft.weight_bits
            or weight_tensor.layout != expected_layout
        ):
            raise AlphaStoragePlanError(
                f"{weight_physical.name}: finalized Alpha GEMM weight "
                "representation changed"
            )

        materialized_physical = None
        materialized_component = None
        materialized_shape = None
        materialized_padding = None
        materialized_valid = None
        materialized_resident = None
        if draft.materialized_input is not None:
            materialized = draft.materialized_input
            materialized_physical = plan.physical_id(
                plane,
                materialized.logical_tensor,
            )
            input_tensor = plan.physical_tensor(plane, materialized_physical)
            try:
                input_component = input_tensor.component(materialized.component)
            except KeyError as error:
                raise AlphaStoragePlanError(
                    f"{materialized_physical.name}: materialized Alpha GEMM "
                    "input component was lost"
                ) from error
            if (
                input_tensor.layout != _ALPHA_PADDED_INPUT_LAYOUT
                or tuple(input_component.shape) != materialized.physical_shape
                or input_component.bits != draft.input_bits
            ):
                raise AlphaStoragePlanError(
                    f"{materialized_physical.name}: materialized Alpha GEMM "
                    "input representation changed"
                )
            materialized_component = materialized.component
            materialized_shape = materialized.physical_shape
            materialized_padding = materialized.padding
            if materialized.serialize_channels:
                materialized_valid = materialized.valid_channels
                materialized_resident = materialized.resident_channels
            payload_uses.append(
                AlphaPayloadUse(
                    operation=draft.constraint.implementation,
                    field="materialized_input_payload",
                    logical_tensor=materialized.logical_tensor,
                    physical_tensor=materialized_physical,
                    component=materialized.component,
                )
            )

        weight_scale_physical = None
        if draft.weight_scale is not None:
            weight_scale_physical = plan.physical_id(plane, draft.weight_scale)
            if weight_scale_physical != weight_physical:
                raise AlphaStoragePlanError(
                    f"{draft.weight_scale.name}: finalized Alpha VSQ scale "
                    "does not share its GEMM weight backing"
                )
            artifact = plan.serialized_artifact(
                draft.weight_scale,
                plane,
                f"{draft.constraint.owner.name}.weight_vector_scale_payload",
            )
            if artifact.shape[0] != draft.weight_scale_encoded_output_channels:
                raise AlphaStoragePlanError(
                    f"{draft.weight_scale.name}: finalized Alpha GEMM scale "
                    "encoding changed"
                )

        payload_uses.append(
            AlphaPayloadUse(
                operation=draft.constraint.owner,
                field="weight_payload",
                logical_tensor=draft.weight,
                physical_tensor=weight_physical,
                component=draft.weight_component,
            )
        )
        bindings.append(
            AlphaGemmStorageBinding(
                owner=draft.constraint.owner,
                implementation=draft.constraint.implementation,
                input=draft.input,
                weight=draft.weight,
                output=draft.output,
                input_physical=input_physical,
                weight_physical=weight_physical,
                output_physical=output_physical,
                weight_component=draft.weight_component,
                input_bits=draft.input_bits,
                weight_bits=draft.weight_bits,
                output_bits=draft.constraint.output_bits,
                valid_input_channels=draft.valid_input_channels,
                resident_input_channels=draft.resident_input_channels,
                physical_weight_input_channels=(
                    draft.physical_weight_input_channels
                ),
                valid_output_channels=draft.valid_output_channels,
                resident_output_channels=draft.resident_output_channels,
                physical_weight_output_channels=(
                    draft.physical_weight_output_channels
                ),
                pad_h_up=draft.pads[0],
                pad_h_down=draft.pads[1],
                pad_w_left=draft.pads[2],
                pad_w_right=draft.pads[3],
                padding_c=draft.padding_c,
                padding_k=draft.padding_k,
                materialized_input_physical=materialized_physical,
                materialized_input_component=materialized_component,
                materialized_input_shape=materialized_shape,
                materialized_input_padding=materialized_padding,
                materialized_input_valid_channels=materialized_valid,
                materialized_input_resident_channels=materialized_resident,
                weight_scale=draft.weight_scale,
                weight_scale_physical=weight_scale_physical,
                weight_scale_valid_output_channels=(
                    draft.weight_scale_valid_output_channels
                ),
                weight_scale_encoded_output_channels=(
                    draft.weight_scale_encoded_output_channels
                ),
            )
        )
    return _GemmStorageFacts(tuple(bindings), tuple(payload_uses))


def _one_bank_pool_operands(
    graph: StorageGraph,
    constraint: AlphaOperationConstraint,
) -> Optional[tuple[ValueRef, ValueRef]]:
    """Return the exact INT8 pool input/output that fit one Alpha bank."""
    if constraint.kind != "pool2d" or constraint.output_bits != 8:
        return None
    input_binding = constraint.operand("input")
    if input_binding is None or input_binding.bits != 8:
        return None
    input_shape = tuple(graph.logical[input_binding.logical_tensor].shape)
    output_shape = tuple(graph.logical[constraint.output].shape)
    if len(input_shape) != 4 or len(output_shape) != 4:
        return None
    memory = GLOBAL.instance(GLOBAL.input_memory)
    channels_per_bank, remainder = divmod(
        int(memory.bits_per_bank_entry),
        input_binding.bits,
    )
    if remainder or channels_per_bank <= 0:
        return None
    if (
        0 < int(input_shape[1]) <= channels_per_bank
        and 0 < int(output_shape[1]) <= channels_per_bank
    ):
        return input_binding.logical_tensor, constraint.output
    return None


def _declare_alpha_cluster_constraints(
    model: GraphModule,
    graph: StorageGraph,
    operation_constraints: tuple[AlphaOperationConstraint, ...],
    *,
    config=None,
    plane: str = _SCRATCHPAD,
    initial_constraints: Optional[Mapping[ValueRef, int]] = None,
) -> tuple[_ClusterDraft, ...]:
    """Decide logical bank widths before split and resident declarations."""
    plane = StoragePlane(plane).value
    if plane not in graph.planes:
        raise AlphaStoragePlanError(
            f"{plane}: Alpha cluster constraints are not in this graph"
        )
    by_owner = {
        constraint.owner: constraint for constraint in operation_constraints
    }
    if len(by_owner) != len(operation_constraints):
        raise AlphaStoragePlanError(
            "Alpha cluster constraints received duplicate executable owners"
        )

    drafts = {}
    for logical, cluster in (initial_constraints or {}).items():
        if not isinstance(logical, ValueRef):
            raise TypeError(
                "initial Alpha cluster constraints must use ValueRef keys"
            )
        try:
            logical = graph.ref(logical)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"{logical.name}: initial Alpha cluster value is not in the "
                "storage graph"
            ) from error
        _set_alpha_cluster_draft(drafts, logical, cluster)
    for constraint in operation_constraints:
        compact_cluster = _compact_matrix_output_cluster(
            graph,
            constraint,
            config,
        )
        if compact_cluster is not None:
            _set_alpha_cluster_draft(
                drafts,
                constraint.output,
                compact_cluster,
            )

        host_input = _host_matrix_input_cluster(
            model,
            graph,
            constraint,
            by_owner,
            config,
        )
        if host_input is not None:
            _set_alpha_cluster_draft(drafts, host_input, 1)

        pool = _one_bank_pool_operands(graph, constraint)
        if pool is not None:
            for logical in pool:
                _set_alpha_cluster_draft(drafts, logical, 1)

    # Match the fork's one-pass propagation from only the initially constrained
    # operations.  A newly constrained aggregate must not recursively seed a
    # later aggregate in this same pass.
    constrained = frozenset(drafts)
    for node in model.graph.nodes:
        layout = _concat_layout(node, graph, plane=plane)
        if layout is None:
            continue
        traced = tuple(
            _trace_resident_write_owner(
                graph,
                input_node,
                plane=plane,
                expected_shape=shape,
            )
            for input_node, shape in zip(layout.inputs, layout.shapes)
        )
        if any(item is None for item in traced):
            continue
        resolved = tuple(item for item in traced if item is not None)
        if not any(
            owner_ref in constrained
            for _owner, owner_ref, _path in resolved
        ):
            continue
        try:
            if any(_bits_for_ref(graph, owner_ref) != 8 for _, owner_ref, _ in resolved):
                continue
        except AlphaStoragePlanError:
            continue
        try:
            backing = _single_ref(graph, node)
        except AlphaStoragePlanError:
            continue
        _set_alpha_cluster_draft(drafts, backing, 1)
        for _owner, owner_ref, path in resolved:
            _set_alpha_cluster_draft(drafts, owner_ref, 1)
            for path_ref in path:
                _set_alpha_cluster_draft(drafts, path_ref, 1)

    return tuple(drafts.values())


def _concat_writes_for(
    cat: Node,
    graph: StorageGraph,
    *,
    plane: str,
    block_channels: int,
    executable_owners: frozenset[Node],
    vsq_payloads: frozenset[ValueRef] = frozenset(),
    cluster_constraints: Optional[Mapping[ValueRef, int]] = None,
) -> Optional[_ConcatPlan]:
    try:
        backing_ref = _single_ref(graph, cat)
    except AlphaStoragePlanError:
        return None
    if (
        graph.relation(plane, backing_ref).kind
        is StorageRelationKind.COMPONENT
    ):
        # A joined companion scale follows the payload aggregate. It never
        # owns a second resident concat or allocation.
        return None
    layout = _concat_layout(cat, graph, plane=plane)
    if layout is None:
        return None
    geometry = _resident_channel_granularity(
        graph,
        backing_ref,
        block_channels=block_channels,
        vsq_payloads=vsq_payloads,
        cluster_constraints=cluster_constraints,
    )
    if geometry is None:
        return None
    operand_bits, channel_granularity = geometry
    logical_sizes = tuple(
        shape[layout.channel_axis] for shape in layout.shapes
    )
    physical_sizes = tuple(
        math.ceil(size / channel_granularity) * channel_granularity
        for size in logical_sizes
    )
    logical_total_channels = sum(logical_sizes)
    total_channels = sum(physical_sizes)

    writes = []
    channel_offset = 0
    for input_index, (input_node, input_shape, physical_size) in enumerate(
        zip(layout.inputs, layout.shapes, physical_sizes)
    ):
        traced = _trace_resident_write_owner(
            graph,
            input_node,
            plane=plane,
            expected_shape=input_shape,
        )
        if traced is None:
            return None
        owner, owner_ref, path = traced
        if owner not in executable_owners or owner.op not in {
            "call_function",
            "call_module",
        }:
            return None
        try:
            if _bits_for_ref(graph, owner_ref) != operand_bits:
                return None
        except AlphaStoragePlanError:
            return None
        writes.append(
            _ResidentWrite(
                backing=cat,
                owner=owner,
                path=path,
                slot_index=input_index,
                channel_offset=channel_offset,
                slot_channels=physical_size,
                logical_backing_channels=logical_total_channels,
                resident_channels=total_channels,
                concat_channel_index=1,
            )
        )
        channel_offset += physical_size
    if len({write.owner for write in writes}) != len(writes):
        return None
    return _ConcatPlan(layout=layout, writes=tuple(writes))


def _shuffle_input_write_for(
    shuffle: Node,
    graph: StorageGraph,
    existing: Mapping[Node, _ResidentWrite],
    *,
    plane: str,
    block_channels: int,
    executable_owners: frozenset[Node],
    vsq_payloads: frozenset[ValueRef] = frozenset(),
    cluster_constraints: Optional[Mapping[ValueRef, int]] = None,
) -> Optional[_ResidentWrite]:
    if shuffle.target is not _SHUFFLE:
        return None
    inputs = tuple(shuffle.args[:2])
    if len(inputs) != 2 or not all(isinstance(node, Node) for node in inputs):
        return None
    shapes = tuple(_shape(node) for node in (*inputs, shuffle))
    if (
        any(len(shape) != 4 or shape[0] != 1 for shape in shapes)
        or shapes[0][2:] != shapes[1][2:]
        or shapes[0][2:] != shapes[2][2:]
        or shapes[2][1] != shapes[0][1] + shapes[1][1]
    ):
        return None
    traced = _trace_resident_write_owner(
        graph,
        inputs[1],
        plane=plane,
        expected_shape=shapes[1],
    )
    if traced is None:
        return None
    owner, owner_ref, path = traced
    if owner not in executable_owners:
        return None
    geometry = _resident_channel_granularity(
        graph,
        owner_ref,
        block_channels=block_channels,
        vsq_payloads=vsq_payloads,
        cluster_constraints=cluster_constraints,
    )
    if geometry is None:
        return None
    operand_bits, channel_granularity = geometry
    try:
        input_bits = tuple(
            _bits_for_ref(graph, _payload_ref(graph, node)) for node in inputs
        )
    except AlphaStoragePlanError:
        return None
    if operand_bits != 8 or input_bits != (operand_bits, operand_bits):
        return None
    physical_sizes = tuple(
        math.ceil(shape[1] / channel_granularity) * channel_granularity
        for shape in shapes[:2]
    )
    total_channels = sum(physical_sizes)
    dense_channels = physical_sizes[1]
    if total_channels <= dense_channels:
        return None

    try:
        input1_chain = graph.trace_storage_chain(inputs[0], plane)
    except (KeyError, ValueError):
        input1_chain = ()
    completes_split = any(
        relation.transform is not None
        and relation.transform.kind == "alpha_channel_split"
        for ref in input1_chain
        for relation in (graph.relation(plane, ref),)
    )
    expected_concat_index = 0 if completes_split else None
    previous = existing.get(owner)
    if previous is not None:
        if (
            previous.slot_index == 0
            and previous.channel_offset == 0
            and previous.slot_channels == dense_channels
            and previous.resident_channels == total_channels
            and previous.concat_channel_index == expected_concat_index
        ):
            return previous
        raise AlphaStoragePlanError(
            f"{owner.name}: shuffle input pitch conflicts with an existing "
            "concat write"
        )
    return _ResidentWrite(
        backing=owner,
        owner=owner,
        path=path,
        slot_index=0,
        channel_offset=0,
        slot_channels=dense_channels,
        logical_backing_channels=shapes[1][1],
        resident_channels=total_channels,
        concat_channel_index=expected_concat_index,
    )


def _resident_transform(write: _ResidentWrite) -> ViewTransform:
    parameters = (
        ("channel_offset", write.channel_offset),
        ("slot_index", write.slot_index),
        ("slot_channels", write.slot_channels),
        ("resident_channels", write.resident_channels),
    )
    if write.concat_channel_index is not None:
        parameters += (("concat_channel_index", write.concat_channel_index),)
    return ViewTransform("alpha_resident_write", parameters=parameters)


def _set_resident_relation(
    graph: StorageGraph,
    result: ValueRef,
    kind: StorageRelationKind,
    *,
    plane: str,
    source: Optional[ValueRef],
    transform: ViewTransform,
    component: Optional[str] = None,
) -> None:
    previous = graph.relation(plane, result)
    same_relation = (
        previous.kind is kind
        and previous.source == source
        and previous.physical_source is None
        and previous.transform == transform
        and previous.component == component
    )
    if same_relation:
        return
    chain = graph.trace_storage_chain(result, plane)
    compatible_split = (
        kind is StorageRelationKind.VIEW
        and source is not None
        and previous.kind
        in {StorageRelationKind.ALIAS, StorageRelationKind.VIEW}
        and source in chain[1:]
        and any(
            relation.transform is not None
            and relation.transform.kind == "alpha_channel_split"
            for ref in chain
            for relation in (graph.relation(plane, ref),)
        )
    )
    if compatible_split:
        return
    if previous.transform is not None and previous.transform.kind.startswith(
        "alpha_"
    ):
        raise AlphaStoragePlanError(
            f"{result.name}: conflicting Alpha resident storage relation"
        )
    try:
        graph.set_relation(
            result,
            kind,
            plane=plane,
            source=source,
            transform=transform,
            component=component,
        )
    except ValueError as error:
        raise AlphaStoragePlanError(
            f"{result.name}: conflicting resident storage declaration"
        ) from error


def _register_resident_backings(
    graph: StorageGraph,
    writes: tuple[_ResidentWrite, ...],
    *,
    plane: str,
    vsq_payloads: frozenset[ValueRef] = frozenset(),
) -> None:
    declarations = {}
    for write in writes:
        backing = _payload_ref(graph, write.backing)
        value = _logical_value(backing)
        if not isinstance(value, torch.Tensor):
            raise AlphaStoragePlanError(
                f"{backing.name}: resident backing has no tensor value"
            )
        logical_numel = value.numel()
        if logical_numel % write.logical_backing_channels:
            raise AlphaStoragePlanError(
                f"{backing.name}: resident backing shape is not divisible "
                "by its logical channel extent"
            )
        operand_bits = _bits_for_ref(graph, backing)
        resident_numel = (
            logical_numel
            // write.logical_backing_channels
            * write.resident_channels
        )
        backing_size = math.ceil(resident_numel * operand_bits / 8)
        declaration = (
            write.logical_backing_channels,
            write.resident_channels,
            operand_bits,
            backing_size,
        )
        previous = declarations.get(backing)
        if previous is not None and previous != declaration:
            raise AlphaStoragePlanError(
                f"{backing.name}: inconsistent resident backing declarations"
            )
        declarations[backing] = declaration

    for backing, (
        logical_channels,
        resident_channels,
        operand_bits,
        backing_size,
    ) in declarations.items():
        if logical_channels == resident_channels:
            continue
        if backing in vsq_payloads:
            raise AlphaStoragePlanError(
                f"{backing.name}: padded Alpha resident storage cannot "
                "preserve an embedded VSQ companion"
            )
        if backing.path or len(graph.output_refs(backing.node)) != 1:
            raise AlphaStoragePlanError(
                f"{backing.name}: padded resident storage with components "
                "is unsupported"
            )
        value = _logical_value(backing)
        graph.register_physical_component(
            backing,
            value,
            plane=plane,
            component="primary",
            dtype=graph.logical[backing].dtype,
            bits=operand_bits,
            backing_size_bytes=backing_size,
            layout=_ALPHA_DENSE_LAYOUT,
        )


def _declare_alpha_resident_storage(
    model: GraphModule,
    graph: StorageGraph,
    operation_constraints: tuple[AlphaOperationConstraint, ...],
    *,
    plane: str = _SCRATCHPAD,
    block_channels: int = OUTPUT_LANES,
    vsq_payloads: frozenset[ValueRef] = frozenset(),
    cluster_constraints: Optional[Mapping[ValueRef, int]] = None,
) -> tuple[_ResidentWrite, ...]:
    """Declare fork-compatible concat slots and shuffle-input row pitches."""
    plane = StoragePlane(plane).value
    if plane not in graph.planes:
        raise AlphaStoragePlanError(
            f"{plane}: resident storage is not a plane of this graph"
        )
    if (
        isinstance(block_channels, bool)
        or not isinstance(block_channels, int)
        or block_channels <= 0
    ):
        raise AlphaStoragePlanError(
            "Alpha resident channel width must be a positive integer"
        )
    executable_owners = frozenset(
        constraint.owner for constraint in operation_constraints
    )
    writes_by_owner = {}
    for node in model.graph.nodes:
        plan = _concat_writes_for(
            node,
            graph,
            plane=plane,
            block_channels=block_channels,
            executable_owners=executable_owners,
            vsq_payloads=vsq_payloads,
            cluster_constraints=cluster_constraints,
        )
        if plan is None:
            continue
        if any(write.owner in writes_by_owner for write in plan.writes):
            continue
        for write in plan.writes:
            writes_by_owner[write.owner] = write

    for shuffle in model.graph.nodes:
        write = _shuffle_input_write_for(
            shuffle,
            graph,
            writes_by_owner,
            plane=plane,
            block_channels=block_channels,
            executable_owners=executable_owners,
            vsq_payloads=vsq_payloads,
            cluster_constraints=cluster_constraints,
        )
        if write is not None:
            writes_by_owner[write.owner] = write

    writes = tuple(writes_by_owner.values())
    for write in writes:
        owner_ref = _payload_ref(graph, write.owner)
        backing_ref = _payload_ref(graph, write.backing)
        transform = _resident_transform(write)
        if owner_ref == backing_ref:
            previous = graph.relation(plane, owner_ref)
            if previous.kind is not StorageRelationKind.MATERIALIZE:
                raise AlphaStoragePlanError(
                    f"{owner_ref.name}: shuffle pitch requires an owning "
                    "materialization"
                )
            _set_resident_relation(
                graph,
                owner_ref,
                previous.kind,
                plane=plane,
                source=previous.source,
                transform=transform,
                component=previous.component,
            )
        else:
            _set_resident_relation(
                graph,
                owner_ref,
                StorageRelationKind.WRITE_INTO,
                plane=plane,
                source=backing_ref,
                transform=transform,
            )
        for path_ref in write.path:
            if path_ref.node.target is _SHUFFLE_COPY:
                continue
            _set_resident_relation(
                graph,
                path_ref,
                StorageRelationKind.VIEW,
                plane=plane,
                source=backing_ref,
                transform=ViewTransform("alpha_resident_view"),
            )
    _register_resident_backings(
        graph,
        writes,
        plane=plane,
        vsq_payloads=vsq_payloads,
    )
    return writes


def _freeze_alpha_resident_storage(
    graph: StorageGraph,
    plan,
    writes: tuple[_ResidentWrite, ...],
    *,
    plane: str = _SCRATCHPAD,
) -> tuple[AlphaResidentWriteBinding, ...]:
    """Freeze resident identities and slot geometry after finalization."""
    plane = StoragePlane(plane).value
    bindings = []
    for write in writes:
        logical = _payload_ref(graph, write.owner)
        backing = _payload_ref(graph, write.backing)
        relation = plan.relation(plane, logical)
        expected_kind = (
            StorageRelationKind.MATERIALIZE
            if logical == backing
            else StorageRelationKind.WRITE_INTO
        )
        expected_source = None if logical == backing else backing
        if (
            relation.kind is not expected_kind
            or relation.source != expected_source
            or relation.transform != _resident_transform(write)
        ):
            raise AlphaStoragePlanError(
                f"{logical.name}: finalized resident relation changed"
            )
        physical = plan.physical_id(plane, logical)
        if physical != plan.physical_id(plane, backing):
            raise AlphaStoragePlanError(
                f"{logical.name}: resident writer and backing were separated"
            )
        for path_ref in write.path:
            try:
                path_physical = plan.physical_id(plane, path_ref)
            except KeyError as error:
                raise AlphaStoragePlanError(
                    f"{path_ref.name}: resident alias path was not finalized"
                ) from error
            if path_physical != physical:
                raise AlphaStoragePlanError(
                    f"{path_ref.name}: resident alias path was separated "
                    f"from {backing.name}"
                )
        bindings.append(
            AlphaResidentWriteBinding(
                owner=write.owner,
                logical_tensor=logical,
                physical_tensor=physical,
                backing=backing,
                path=write.path,
                slot_index=write.slot_index,
                channel_offset=write.channel_offset,
                slot_channels=write.slot_channels,
                logical_backing_channels=write.logical_backing_channels,
                resident_channels=write.resident_channels,
                concat_channel_index=write.concat_channel_index,
            )
        )
    return tuple(bindings)


def _zero_copy_shuffle_splits(
    graph: StorageGraph,
    shuffle: Node,
    shape: tuple[int, ...],
) -> tuple[tuple[ValueRef, ValueRef], ...]:
    """Recognize exact two-way channel splits fed directly by a shuffle."""
    expected = (shape[0], shape[1] // 2, shape[2], shape[3])
    matches = []
    for user in shuffle.users:
        if (
            user.target not in _CHANNEL_SPLITS
            or not user.args
            or user.args[0] is not shuffle
        ):
            continue
        if get_arg_value(user, 2, "dim", 0) != 1:
            raise AlphaStoragePlanError(
                f"{user.name}: zero-copy shuffle split is not along channels"
            )
        outputs = graph.output_refs(user)
        if len(outputs) != 2 or any(
            tuple(graph.logical[output].shape) != expected for output in outputs
        ):
            raise AlphaStoragePlanError(
                f"{user.name}: zero-copy shuffle requires an exact two-way "
                "channel split"
            )
        matches.append((outputs[0], outputs[1]))
    return tuple(matches)


def _declare_alpha_shuffle_storage(
    model: GraphModule,
    graph: StorageGraph,
    *,
    plane: str = _SCRATCHPAD,
    block_channels: int = OUTPUT_LANES,
    split_drafts: tuple[_SplitDraft, ...] = (),
) -> tuple[_ShuffleDraft, ...]:
    """Declare only the fork-proven two-backing shuffle storage relations.

    Copy insertion is a separate terminal memory rewrite.  This pass consumes
    its declarations and refuses to repair or infer incomplete copy chains.
    It intentionally runs after resident-write decisions, because those decide
    the row pitch used by both shuffle backings.
    """
    plane = StoragePlane(plane).value
    if plane not in graph.planes:
        raise AlphaStoragePlanError(
            f"{plane}: shuffle storage is not a plane of this graph"
        )
    if (
        isinstance(block_channels, bool)
        or not isinstance(block_channels, int)
        or block_channels <= 0
    ):
        raise AlphaStoragePlanError(
            "Alpha shuffle block width must be a positive integer"
        )

    deferred_splits = {
        draft.deferred_secondary: draft
        for draft in split_drafts
        if draft.deferred_secondary is not None
    }
    if len(deferred_splits) != sum(
        draft.deferred_secondary is not None for draft in split_drafts
    ):
        raise AlphaStoragePlanError(
            "one Alpha split output was deferred more than once"
        )
    consumed_deferred = set()

    copies_by_parent = _shuffle_copies_by_parent(model)
    shuffle_names = {
        node.name
        for node in model.graph.nodes
        if node.target is _SHUFFLE
    }
    orphans = set(copies_by_parent) - shuffle_names
    if orphans:
        raise AlphaStoragePlanError(
            "shuffle copies name missing logical parents: "
            + ", ".join(sorted(orphans))
        )
    if not shuffle_names:
        return ()

    memory = GLOBAL.instance(GLOBAL.output_memory)
    entry_bits = block_channels * 8
    cluster, cluster_remainder = divmod(
        entry_bits,
        int(memory.bits_per_bank_entry),
    )
    if (
        cluster_remainder
        or cluster <= 0
        or int(memory.bank_count) % cluster
    ):
        raise AlphaStoragePlanError(
            f"configured {block_channels}-channel INT8 NMP words do not map "
            f"to {memory.bits_per_bank_entry}-bit Alpha bank entries"
        )
    entry_bytes = entry_bits // 8
    drafts = []
    for shuffle in model.graph.nodes:
        if shuffle.target is not _SHUFFLE:
            continue
        metadata = _alpha_metadata(shuffle)
        groups = get_arg_value(shuffle, 2, "groups")
        if (
            metadata.get("kind") != "channel_shuffle"
            or metadata.get("groups") != groups
            or groups != 2
        ):
            raise AlphaStoragePlanError(
                f"{shuffle.name}: Alpha shuffle requires matching two-group "
                "operator and metadata declarations"
            )

        shuffle_ref = _single_ref(graph, shuffle)
        shape = tuple(graph.logical[shuffle_ref].shape)
        value = _logical_value(shuffle_ref)
        if (
            not isinstance(value, torch.Tensor)
            or len(shape) != 4
            or shape[0] != 1
        ):
            raise AlphaStoragePlanError(
                f"{shuffle.name}: Alpha shuffle requires a concrete "
                "batch-one NCHW payload"
            )
        if _bits_for_ref(graph, shuffle_ref) != 8:
            raise AlphaStoragePlanError(
                f"{shuffle.name}: Alpha NMP shuffle requires an INT8 payload"
            )

        inputs = tuple(shuffle.args[:2])
        if len(inputs) != 2 or not all(isinstance(node, Node) for node in inputs):
            raise AlphaStoragePlanError(
                f"{shuffle.name}: Alpha shuffle requires two tensor inputs"
            )
        input_shapes = tuple(_shape(node) for node in inputs)
        if (
            any(
                len(input_shape) != 4
                or input_shape[0] != 1
                or input_shape[1] <= 0
                for input_shape in input_shapes
            )
            or any(
                input_shape[index] != shape[index]
                for input_shape in input_shapes
                for index in (0, 2, 3)
            )
            or sum(input_shape[1] for input_shape in input_shapes) != shape[1]
        ):
            raise AlphaStoragePlanError(
                f"{shuffle.name}: shuffle input/output extents disagree"
            )

        channel_blocks, remainder = divmod(shape[1], block_channels)
        if remainder:
            raise AlphaStoragePlanError(
                f"{shuffle.name}: {shape[1]} channels are not divisible by "
                f"the configured {block_channels}-channel NMP width"
            )
        movements = _shuffle_copy_movements(channel_blocks)
        copies = copies_by_parent.get(shuffle.name, ())
        indexed = []
        for copy in copies:
            copy_metadata = _alpha_metadata(copy)
            index = copy_metadata.get("copy_index")
            declared_width = copy_metadata.get("block_channels")
            if isinstance(index, bool) or not isinstance(index, int):
                raise AlphaStoragePlanError(
                    f"{copy.name}: shuffle-copy index is not an integer"
                )
            if declared_width != block_channels:
                raise AlphaStoragePlanError(
                    f"{copy.name}: shuffle-copy metadata and configured "
                    "NMP widths disagree"
                )
            indexed.append((index, copy))
        indexed.sort(key=lambda item: item[0])
        if tuple(index for index, _ in indexed) != tuple(
            movement.index for movement in movements
        ):
            raise AlphaStoragePlanError(
                f"{shuffle.name}: expected {len(movements)} indexed Alpha "
                f"shuffle copies, found {[index for index, _ in indexed]}"
            )
        copies = tuple(copy for _, copy in indexed)
        previous = shuffle
        for copy in copies:
            if not copy.args or copy.args[0] is not previous:
                raise AlphaStoragePlanError(
                    f"{copy.name}: shuffle-copy chain does not follow "
                    f"{previous.name}"
                )
            previous = copy
        for producer, successor in zip((shuffle, *copies[:-1]), copies):
            users = tuple(producer.users)
            if len(users) != 1 or users[0] is not successor:
                raise AlphaStoragePlanError(
                    f"{producer.name}: an intermediate Alpha shuffle stage "
                    f"must feed only {successor.name}"
                )

        resident_channels, channel_offset, has_resident_write = _resident_channels(
            graph,
            shuffle_ref,
            plane=plane,
            logical_channels=shape[1],
        )
        pitch_entries, pitch_remainder = divmod(
            resident_channels, block_channels
        )
        if pitch_remainder or pitch_entries < channel_blocks:
            raise AlphaStoragePlanError(
                f"{shuffle.name}: invalid {resident_channels}-channel "
                "resident pitch"
            )
        primary_entry_offset, offset_remainder = divmod(
            channel_offset, block_channels
        )
        if offset_remainder:
            raise AlphaStoragePlanError(
                f"{shuffle.name}: resident channel offset {channel_offset} "
                f"is not aligned to {block_channels} NMP lanes"
            )
        positions = math.prod((shape[0], shape[2], shape[3]))
        span_entries = (positions - 1) * pitch_entries + channel_blocks
        backing_size = span_entries * entry_bytes

        # A resident/concat relation owns the full primary representation.
        # Its backing is rows * pitch (and may begin before this shuffle's
        # channel window), whereas the auxiliary owns only the accessed span.
        if not has_resident_write:
            graph.register_physical_component(
                shuffle_ref,
                value,
                plane=plane,
                component="primary",
                dtype=value.dtype,
                bits=8,
                backing_size_bytes=backing_size,
                layout=_ALPHA_DENSE_LAYOUT,
            )
        auxiliary = graph.add_auxiliary(
            shuffle_ref,
            "output[1]",
            plane=plane,
        )
        graph.register_physical_component(
            auxiliary,
            value,
            plane=plane,
            component="primary",
            dtype=value.dtype,
            bits=8,
            backing_size_bytes=backing_size,
            layout=_ALPHA_SHUFFLE_SECONDARY_LAYOUT,
        )
        graph.add_access(
            auxiliary,
            shuffle,
            plane=plane,
            role=TensorRole.OUTPUT,
            kind="write",
        )

        zero_copy_splits = ()
        if not movements:
            zero_copy_splits = _zero_copy_shuffle_splits(graph, shuffle, shape)
            for primary_output, secondary_output in zero_copy_splits:
                split_draft = deferred_splits.get(secondary_output)
                expected_primary = ViewTransform(
                    "alpha_channel_split",
                    parameters=(
                        ("channel_offset", 0),
                        ("channel_size", shape[1] // 2),
                    ),
                )
                primary_relation = graph.relation(plane, primary_output)
                secondary_relation = graph.relation(plane, secondary_output)
                if not (
                    split_draft is not None
                    and split_draft.source == shuffle_ref
                    and split_draft.outputs
                    == (primary_output, secondary_output)
                    and split_draft.channel_sizes
                    == (shape[1] // 2, shape[1] // 2)
                    and primary_relation.kind is StorageRelationKind.VIEW
                    and primary_relation.source == shuffle_ref
                    and primary_relation.physical_source is None
                    and primary_relation.transform == expected_primary
                    and secondary_relation.kind
                    is StorageRelationKind.MATERIALIZE
                ):
                    raise AlphaStoragePlanError(
                        f"{secondary_output.name}: zero-copy shuffle output "
                        "was not deferred by the Alpha split pass"
                    )
                graph.bind_physical(
                    secondary_output,
                    auxiliary,
                    plane=plane,
                    transform=ViewTransform(
                        "alpha_shuffle_output",
                        parameters=(("output", 1),),
                    ),
                )
                consumed_deferred.add(secondary_output)

        for copy, movement in zip(copies, movements):
            copy_ref = _single_ref(graph, copy)
            graph.set_relation(
                copy_ref,
                StorageRelationKind.WRITE_INTO,
                plane=plane,
                source=shuffle_ref,
                transform=ViewTransform(
                    "alpha_shuffle_copy",
                    parameters=(("copy_index", movement.index),),
                ),
            )
            if movement.source_output == 1:
                graph.add_access(
                    auxiliary,
                    copy,
                    plane=plane,
                    role=TensorRole.ACTIVATION,
                    kind="read",
                )

            destination_channel = movement.destination_entry * block_channels
            block = value[
                :,
                destination_channel : destination_channel + block_channels,
                :,
                :,
            ].contiguous()
            if tuple(block.shape) != (
                shape[0],
                block_channels,
                shape[2],
                shape[3],
            ):
                raise AlphaStoragePlanError(
                    f"{copy.name}: destination block is outside shuffle output"
                )
            graph.register_serialized_artifact(
                copy_ref,
                block,
                plane=plane,
                name=f"{copy.name}.copy_block_payload",
                dtype=block.dtype,
                bits=8,
            )

        drafts.append(
            _ShuffleDraft(
                shuffle=shuffle,
                copies=copies,
                movements=movements,
                block_channels=block_channels,
                auxiliary=auxiliary,
                active_entries=channel_blocks,
                pitch_entries=pitch_entries,
                span_entries=span_entries,
                entry_bytes=entry_bytes,
                primary_entry_offset=primary_entry_offset,
                zero_copy_splits=zero_copy_splits,
            )
        )
    unconsumed = set(deferred_splits) - consumed_deferred
    if unconsumed:
        raise AlphaStoragePlanError(
            "deferred Alpha shuffle splits were not consumed: "
            + ", ".join(sorted(output.name for output in unconsumed))
        )
    return tuple(drafts)


def _freeze_alpha_split_storage(
    graph: StorageGraph,
    plan,
    drafts: tuple[_SplitDraft, ...],
    *,
    plane: str = _SCRATCHPAD,
    shuffle_plans: tuple[AlphaShufflePlan, ...] = (),
) -> tuple[AlphaLogicalViewBinding, ...]:
    """Freeze legal split windows into projection-only placement facts."""
    plane = StoragePlane(plane).value
    if plan.graph is not graph:
        raise AlphaStoragePlanError(
            "Alpha split facts must be frozen from their declaring graph"
        )

    pitch_by_physical = {}
    for shuffle_plan in shuffle_plans:
        for physical in (shuffle_plan.primary, shuffle_plan.auxiliary):
            previous = pitch_by_physical.get(physical)
            if previous is not None and previous != shuffle_plan.pitch_entries:
                raise AlphaStoragePlanError(
                    f"{physical.name}: conflicting Alpha shuffle row pitches"
                )
            pitch_by_physical[physical] = shuffle_plan.pitch_entries

    bindings = []
    bound_outputs = set()
    for draft in drafts:
        if node_operand_bits(draft.source.node) != draft.operand_bits:
            raise AlphaStoragePlanError(
                f"{draft.source.name}: split operand width changed after "
                "declaration"
            )
        source_shape = tuple(plan.logical_tensors[draft.source].shape)
        if len(source_shape) != 4 or source_shape[0] != 1:
            raise AlphaStoragePlanError(
                f"{draft.source.name}: finalized Alpha split source is not "
                "batch-one NCHW"
            )
        positions = math.prod((source_shape[0], source_shape[2], source_shape[3]))
        if positions <= 0:
            raise AlphaStoragePlanError(
                f"{draft.source.name}: finalized Alpha split has no positions"
            )

        local_channel_offset = 0
        for output, channel_size in zip(draft.outputs, draft.channel_sizes):
            if output in bound_outputs:
                raise AlphaStoragePlanError(
                    f"{output.name}: Alpha split view was frozen twice"
                )
            relation = plan.relations[plane][output]
            if output == draft.deferred_secondary:
                valid_relation = (
                    relation.kind is StorageRelationKind.VIEW
                    and relation.source is None
                    and relation.physical_source is not None
                    and relation.transform is not None
                    and relation.transform.kind == "alpha_shuffle_output"
                    and relation.transform.parameter("output") == 1
                )
            else:
                valid_relation = (
                    relation.kind is StorageRelationKind.VIEW
                    and relation.source == draft.source
                    and relation.physical_source is None
                    and relation.transform
                    == ViewTransform(
                        "alpha_channel_split",
                        parameters=(
                            ("channel_offset", local_channel_offset),
                            ("channel_size", channel_size),
                        ),
                    )
                )
            if not valid_relation:
                raise AlphaStoragePlanError(
                    f"{output.name}: finalized Alpha split relation changed"
                )

            output_shape = tuple(plan.logical_tensors[output].shape)
            expected_shape = (
                source_shape[0],
                channel_size,
                source_shape[2],
                source_shape[3],
            )
            if output_shape != expected_shape:
                raise AlphaStoragePlanError(
                    f"{output.name}: finalized Alpha split shape changed"
                )

            physical = plan.physical_id(plane, output)
            chain = plan.trace_storage_chain(output, plane)
            transforms = tuple(
                transform
                for ref in chain
                for transform in (plan.relations[plane][ref].transform,)
                if transform is not None
            )
            cumulative_channels = 0
            for transform in transforms:
                if transform.kind not in {
                    "alpha_channel_split",
                    "alpha_resident_view",
                    "alpha_resident_write",
                }:
                    continue
                offset = transform.parameter("channel_offset", 0)
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or offset < 0
                ):
                    raise AlphaStoragePlanError(
                        f"{output.name}: Alpha split chain has an invalid "
                        "channel offset"
                    )
                cumulative_channels += offset
            entry_offset, offset_remainder = divmod(
                cumulative_channels, draft.channels_per_entry
            )
            if offset_remainder:
                raise AlphaStoragePlanError(
                    f"{output.name}: cumulative Alpha split offset is not "
                    "entry aligned"
                )

            backing_size = plan.backing_size_bytes(physical, plane)
            backing_entries, backing_remainder = divmod(
                backing_size, draft.entry_bytes
            )
            if backing_remainder or backing_entries <= 0:
                raise AlphaStoragePlanError(
                    f"{physical.name}: Alpha split backing is not entry aligned"
                )
            row_pitch_entries = pitch_by_physical.get(physical)
            if row_pitch_entries is None:
                row_pitch_entries, pitch_remainder = divmod(
                    backing_entries, positions
                )
                if pitch_remainder or row_pitch_entries <= 0:
                    raise AlphaStoragePlanError(
                        f"{physical.name}: Alpha split row pitch is ambiguous"
                    )
            active_entries = channel_size // draft.channels_per_entry
            if entry_offset + active_entries > row_pitch_entries:
                raise AlphaStoragePlanError(
                    f"{output.name}: Alpha split window exceeds its row pitch"
                )

            logical_size = plan.logical_size_bytes(output)
            span_entries, span_remainder = divmod(
                logical_size, draft.entry_bytes
            )
            if span_remainder or span_entries != positions * active_entries:
                raise AlphaStoragePlanError(
                    f"{output.name}: Alpha split logical span is not entry "
                    "aligned"
                )
            bindings.append(
                AlphaLogicalViewBinding(
                    logical_tensor=output,
                    physical_tensor=physical,
                    transforms=transforms,
                    entry_offset=entry_offset,
                    byte_offset=entry_offset * draft.entry_bytes,
                    span_entries=span_entries,
                    row_pitch_entries=row_pitch_entries,
                )
            )
            bound_outputs.add(output)
            local_channel_offset += channel_size
    return tuple(bindings)


def _physical_row_pitch_channels(
    plan,
    logical: ValueRef,
    operand_bits: int,
    *,
    plane: str = _SCRATCHPAD,
) -> int:
    """Return the finalized NCHW channel pitch of one dense backing."""
    plane = StoragePlane(plane).value
    shape = tuple(plan.logical_tensors[logical].shape)
    if len(shape) != 4:
        raise AlphaStoragePlanError(
            f"{logical.name}: NMP physical pitch requires rank four"
        )
    physical = plan.physical_id(plane, logical)
    tensor = plan.physical_tensor(plane, physical)
    relation = plan.relation(plane, logical)
    component_name = relation.component or "primary"
    dense_vsq_primary = (
        tensor.layout == _ALPHA_VSQ_LAYOUT and component_name == "primary"
    )
    if tensor.layout not in {"dense", _ALPHA_DENSE_LAYOUT} and not dense_vsq_primary:
        raise AlphaStoragePlanError(
            f"{logical.name}: NMP pitch cannot be inferred from physical "
            f"layout {tensor.layout!r}"
        )
    try:
        component = tensor.component(component_name)
    except KeyError as error:
        raise AlphaStoragePlanError(
            f"{logical.name}: NMP backing has no {component_name!r} component"
        ) from error
    if component.bits != operand_bits:
        raise AlphaStoragePlanError(
            f"{logical.name}: physical component width {component.bits} "
            f"does not match {operand_bits}-bit NMP"
        )
    rows = int(shape[0]) * int(shape[2]) * int(shape[3])
    row_divisor = rows * operand_bits
    backing_bits = plan.backing_size_bytes(physical, plane) * 8
    if rows <= 0 or backing_bits % row_divisor:
        raise AlphaStoragePlanError(
            f"{logical.name}: physical backing is not an integral NCHW "
            "channel pitch"
        )
    return backing_bits // row_divisor


def _alpha_nmp_input_view(
    plan,
    owner: Node,
    operand: AlphaOperandBinding,
    *,
    nmp_lanes: int,
    include_resident: bool,
    plane: str = _SCRATCHPAD,
) -> Optional[AlphaNmpInputView]:
    """Freeze one split/resident input descriptor from final relations."""
    plane = StoragePlane(plane).value
    logical = operand.logical_tensor
    try:
        chain = plan.trace_storage_chain(logical, plane)
    except (KeyError, TypeError, ValueError):
        return None

    split_ref = None
    split_relation = None
    shuffle_output_relation = None
    resident_parameters = None
    for ref in chain:
        relation = plan.relation(plane, ref)
        transform = relation.transform
        if transform is None:
            continue
        if transform.kind == "alpha_resident_write":
            resident_parameters = dict(transform.parameters)
        if (
            transform.kind == "alpha_channel_split"
            and relation.source is not None
        ):
            split_ref = ref
            split_relation = relation
            break
        if transform.kind == "alpha_shuffle_output":
            shuffle_output_relation = relation

    shape = tuple(plan.logical_tensors[logical].shape)
    if len(shape) != 4:
        return None
    logical_channels = int(shape[1])
    resident_group_count = None
    uniform_group_channels = None

    if split_relation is not None:
        physical_channels = _physical_row_pitch_channels(
            plan,
            logical,
            operand.bits,
            plane=plane,
        )
        parameters = dict(split_relation.transform.parameters)
        channel_size = int(parameters["channel_size"])
        if channel_size != logical_channels:
            raise AlphaStoragePlanError(
                f"{logical.name}: split view shape does not match its "
                "storage relation"
            )
        channel_offset = int(parameters["channel_offset"])
        split_index = (
            int(split_ref.path[0])
            if split_ref is not None and split_ref.path
            else channel_offset // channel_size
        )
        source_kind = "channel_split"
    elif shuffle_output_relation is not None:
        physical = plan.physical_id(plane, logical)
        tensor = plan.physical_tensor(plane, physical)
        relation = plan.relation(plane, logical)
        try:
            component = tensor.component(relation.component or "primary")
        except KeyError as error:
            raise AlphaStoragePlanError(
                f"{logical.name}: shuffle output has no physical component"
            ) from error
        if len(component.shape) != 4:
            return None
        physical_channels = int(component.shape[1])
        split_index = 0
        source_kind = "shuffle_output"
    elif include_resident and resident_parameters is not None:
        physical_channels = _physical_row_pitch_channels(
            plan,
            logical,
            operand.bits,
            plane=plane,
        )
        if physical_channels <= logical_channels:
            return None
        split_index = int(resident_parameters["slot_index"])
        source_kind = "resident"
        physical = plan.physical_id(plane, logical)
        tensor = plan.physical_tensor(plane, physical)
        slots = {}
        for resident_logical in tensor.logical_tensors:
            relation = plan.relation(plane, resident_logical)
            transform = relation.transform
            if transform is None or transform.kind != "alpha_resident_write":
                continue
            parameters = dict(transform.parameters)
            slots[int(parameters["slot_index"])] = int(
                parameters["slot_channels"]
            )
        if slots:
            resident_group_count = len(slots)
            widths = set(slots.values())
            if len(widths) == 1:
                uniform_group_channels = next(iter(widths))
    else:
        return None

    pitch_gap_channels = physical_channels - logical_channels
    if pitch_gap_channels < 0 or pitch_gap_channels % nmp_lanes:
        raise AlphaStoragePlanError(
            f"{logical.name}: split row-pitch gap {pitch_gap_channels} is not "
            f"representable in {nmp_lanes}-channel NMP words"
        )
    return AlphaNmpInputView(
        owner=owner,
        logical_tensor=logical,
        source_kind=source_kind,
        channel_index=1,
        split_index=split_index,
        current_channels=logical_channels,
        total_channels=physical_channels,
        pitch_gap_words=pitch_gap_channels // nmp_lanes,
        resident_group_count=resident_group_count,
        uniform_group_channels=uniform_group_channels,
    )


def _freeze_alpha_nmp_input_views(
    plan,
    constraints: tuple[AlphaOperationConstraint, ...],
    *,
    nmp_lanes: int,
) -> tuple[AlphaNmpInputView, ...]:
    """Freeze at most one endpoint split descriptor per executable owner."""
    bindings = []
    for constraint in constraints:
        candidates = []
        for operand in constraint.operands:
            candidate = _alpha_nmp_input_view(
                plan,
                constraint.owner,
                operand,
                nmp_lanes=nmp_lanes,
                include_resident=constraint.kind
                in {"gemm", "depthwise_conv2d"},
            )
            if candidate is not None:
                candidates.append(candidate)
        unique = {}
        for candidate in candidates:
            key = (
                candidate.source_kind,
                candidate.channel_index,
                candidate.split_index,
                candidate.current_channels,
                candidate.total_channels,
                candidate.pitch_gap_words,
                candidate.resident_group_count,
                candidate.uniform_group_channels,
            )
            unique.setdefault(key, candidate)
        if len(unique) > 1:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: multiple split storage views "
                "cannot share one NMP descriptor"
            )
        if unique:
            bindings.append(next(iter(unique.values())))
    return tuple(bindings)


def _nmp_pitch_gap_words(
    plan,
    logical: ValueRef,
    operand_bits: int,
    *,
    nmp_lanes: int,
    round_logical: bool = False,
    plane: str = _SCRATCHPAD,
) -> int:
    """Freeze one physical-minus-logical NMP row-pitch increment.

    Most NMP records require the physical channel gap to be an exact number
    of channel words.  MaxPool is the fork's one exception: both logical and
    physical channel extents are independently rounded to NMP words because
    a partial logical word is itself a legal pool row.
    """
    plane = StoragePlane(plane).value
    try:
        shape = tuple(plan.logical_tensors[logical].shape)
    except (KeyError, TypeError, ValueError) as error:
        raise AlphaStoragePlanError(
            f"{logical.name}: NMP logical shape is unavailable"
        ) from error
    if len(shape) != 4 or int(shape[1]) <= 0:
        raise AlphaStoragePlanError(
            f"{logical.name}: NMP pitch requires positive rank-four NCHW"
        )
    logical_channels = int(shape[1])
    physical_channels = _physical_row_pitch_channels(
        plan,
        logical,
        operand_bits,
        plane=plane,
    )
    if physical_channels < logical_channels:
        raise AlphaStoragePlanError(
            f"{logical.name}: physical NMP pitch C={physical_channels} "
            f"cannot contain logical C={logical_channels}"
        )
    if round_logical:
        return (
            math.ceil(physical_channels / nmp_lanes)
            - math.ceil(logical_channels / nmp_lanes)
        )
    gap = physical_channels - logical_channels
    words, remainder = divmod(gap, nmp_lanes)
    if remainder:
        raise AlphaStoragePlanError(
            f"{logical.name}: NMP pitch gap C={gap} is not representable "
            f"in {nmp_lanes}-channel words"
        )
    return words


def _freeze_alpha_nmp_storage(
    graph: StorageGraph,
    plan,
    constraints: tuple[AlphaOperationConstraint, ...],
    input_views: tuple[AlphaNmpInputView, ...],
    *,
    nmp_lanes: int,
    plane: str = _SCRATCHPAD,
) -> tuple[AlphaNmpStorageBinding, ...]:
    """Freeze ordinary NMP storage and address-generator facts once.

    This is intentionally upstream of allocation and serialization.  The
    latter may attach addresses to the named physical tensors, but it must not
    recover padding, pitch, split, or group geometry from mutable FX metadata.
    """
    plane = StoragePlane(plane).value
    if plan.graph is not graph:
        raise AlphaStoragePlanError(
            "Alpha NMP facts must be frozen from their declaring graph"
        )
    by_owner = {view.owner: view for view in input_views}
    if len(by_owner) != len(input_views):
        raise AlphaStoragePlanError("Alpha NMP input views repeat owners")

    bindings = []
    ordinary_kinds = {
        "depthwise_conv2d",
        "pool2d",
        "upsample_nearest2d",
    }
    for constraint in constraints:
        if constraint.kind not in ordinary_kinds:
            continue
        input_binding = constraint.operand("input")
        if input_binding is None:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: ordinary Alpha NMP input did not "
                "resolve"
            )
        input_ref = input_binding.logical_tensor
        output_ref = constraint.output
        try:
            input_physical = plan.physical_id(plane, input_ref)
            output_physical = plan.physical_id(plane, output_ref)
            plan.physical_tensor(plane, input_physical)
            plan.physical_tensor(plane, output_physical)
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: ordinary Alpha NMP storage is "
                "incomplete"
            ) from error

        input_view = by_owner.get(constraint.owner)
        if input_view is not None and input_view.logical_tensor != input_ref:
            raise AlphaStoragePlanError(
                f"{constraint.owner.name}: frozen NMP view does not describe "
                "the input port"
            )

        weight_ref = None
        weight_physical = None
        pads = None
        padding_c = None
        padding_k = None
        nmp_c2 = 0
        nmp_k2 = 0
        nmp_groups = None
        nmp_channels_per_group = None
        is_pool_1bank = None

        if constraint.kind == "depthwise_conv2d":
            weight_binding = constraint.operand("weight")
            if weight_binding is None:
                raise AlphaStoragePlanError(
                    f"{constraint.owner.name}: Alpha depthwise weight did "
                    "not resolve"
                )
            weight_ref = weight_binding.logical_tensor
            try:
                weight_physical = plan.physical_id(plane, weight_ref)
                input_shape = _rank_four_logical_shape(
                    graph, input_ref, "Alpha depthwise input"
                )
                weight_shape = _rank_four_logical_shape(
                    graph, weight_ref, "Alpha depthwise weight"
                )
                output_shape = _rank_four_logical_shape(
                    graph, output_ref, "Alpha depthwise output"
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AlphaStoragePlanError(
                    f"{constraint.owner.name}: Alpha depthwise geometry is "
                    "unavailable"
                ) from error
            stride = _static_pair(
                _convolution_geometry_arg(
                    constraint.implementation, "stride", None
                ),
                1,
            )
            torch_padding = _static_pair(
                _convolution_geometry_arg(
                    constraint.implementation, "padding", None
                ),
                0,
            )
            dilation = _static_pair(
                _convolution_geometry_arg(
                    constraint.implementation, "dilation", None
                ),
                1,
            )
            if None in {stride, torch_padding, dilation}:
                raise AlphaStoragePlanError(
                    f"{constraint.owner.name}: Alpha depthwise geometry is "
                    "not static"
                )
            spatial_pads = []
            for axis in range(2):
                effective = (
                    dilation[axis] * (weight_shape[axis + 2] - 1) + 1
                )
                needed = max(
                    (output_shape[axis + 2] - 1) * stride[axis]
                    + effective
                    - input_shape[axis + 2],
                    0,
                )
                before = min(torch_padding[axis], needed)
                spatial_pads.append((before, needed - before))
            pads = (
                spatial_pads[0][0],
                spatial_pads[0][1],
                spatial_pads[1][0],
                spatial_pads[1][1],
            )
            padding_c = 0
            padding_k = 0
            if input_view is not None:
                nmp_c2 = input_view.pitch_gap_words
            if (
                input_view is not None
                and input_view.source_kind == "resident"
                and input_view.resident_group_count is not None
                and input_view.resident_group_count > 1
            ):
                nmp_groups = input_view.resident_group_count
                nmp_channels_per_group = input_view.uniform_group_channels
            else:
                logical_weight_shape = get_hardware_logical_shape(
                    weight_ref.node
                )
                if not logical_weight_shape:
                    logical_weight_shape = weight_shape
                nmp_groups = int(logical_weight_shape[0])

        elif constraint.kind == "pool2d":
            is_pool_1bank = (
                _one_bank_pool_operands(graph, constraint) is not None
            )
            nmp_k2 = _nmp_pitch_gap_words(
                plan,
                output_ref,
                constraint.output_bits,
                nmp_lanes=nmp_lanes,
                round_logical=True,
                plane=plane,
            )
            if input_view is not None:
                nmp_c2 = input_view.pitch_gap_words

        else:
            padding_c = 0
            padding_k = 0
            nmp_c2 = _nmp_pitch_gap_words(
                plan,
                input_ref,
                input_binding.bits,
                nmp_lanes=nmp_lanes,
                plane=plane,
            )
            nmp_k2 = _nmp_pitch_gap_words(
                plan,
                output_ref,
                constraint.output_bits,
                nmp_lanes=nmp_lanes,
                plane=plane,
            )
            if input_view is not None:
                # A split descriptor is the final input pitch authority in the
                # fork, even when the operation has an ordinary backing gap.
                nmp_c2 = input_view.pitch_gap_words

        bindings.append(
            AlphaNmpStorageBinding(
                owner=constraint.owner,
                implementation=constraint.implementation,
                terminal=constraint.terminal,
                kind=constraint.kind,
                input=input_ref,
                output=output_ref,
                input_physical=input_physical,
                output_physical=output_physical,
                weight=weight_ref,
                weight_physical=weight_physical,
                input_view=input_view,
                pads=pads,
                padding_c=padding_c,
                padding_k=padding_k,
                nmp_c2=nmp_c2,
                nmp_k2=nmp_k2,
                nmp_groups=nmp_groups,
                nmp_channels_per_group=nmp_channels_per_group,
                is_pool_1bank=is_pool_1bank,
            )
        )
    return tuple(bindings)


def _freeze_alpha_shuffle_storage(
    graph: StorageGraph,
    plan,
    drafts: tuple[_ShuffleDraft, ...],
    *,
    plane: str = _SCRATCHPAD,
) -> _ShuffleStorageFacts:
    """Resolve shuffle drafts to exact physical roots and per-port views."""
    plane = StoragePlane(plane).value
    if plan.graph is not graph:
        raise AlphaStoragePlanError(
            "Alpha shuffle facts must be frozen from their declaring graph"
        )

    plans = []
    payload_uses = []
    serialized_uses = []
    for draft in drafts:
        shuffle_ref = _single_ref(graph, draft.shuffle)
        primary = plan.physical_id(plane, shuffle_ref)
        # Querying both backings here is deliberate: an auxiliary declaration
        # without a representation or an accidentally detached primary fails
        # before any allocator or serializer sees a partial shuffle record.
        plan.physical_tensor(plane, primary)
        plan.physical_tensor(plane, draft.auxiliary)

        def backing_entries(physical: PhysicalTensorId) -> int:
            size = plan.backing_size_bytes(physical, plane)
            entries, remainder = divmod(size, draft.entry_bytes)
            if remainder or entries <= 0:
                raise AlphaStoragePlanError(
                    f"{physical.name}: shuffle backing is not entry aligned"
                )
            return entries

        primary_entries = backing_entries(primary)
        auxiliary_entries = backing_entries(draft.auxiliary)
        if auxiliary_entries != draft.span_entries:
            raise AlphaStoragePlanError(
                f"{draft.shuffle.name}: auxiliary shuffle extent changed "
                "during finalization"
            )
        if draft.primary_entry_offset + draft.span_entries > primary_entries:
            raise AlphaStoragePlanError(
                f"{draft.shuffle.name}: primary shuffle window exceeds its "
                "resident backing"
            )

        for primary_output, secondary_output in draft.zero_copy_splits:
            if plan.physical_id(plane, primary_output) != primary:
                raise AlphaStoragePlanError(
                    f"{primary_output.name}: zero-copy shuffle output is not "
                    "bound to its primary backing"
                )
            primary_relation = plan.relations[plane][primary_output]
            expected_primary = ViewTransform(
                "alpha_channel_split",
                parameters=(
                    ("channel_offset", 0),
                    ("channel_size", draft.block_channels),
                ),
            )
            if (
                primary_relation.kind is not StorageRelationKind.VIEW
                or primary_relation.source != shuffle_ref
                or primary_relation.physical_source is not None
                or primary_relation.component is not None
                or primary_relation.transform != expected_primary
            ):
                raise AlphaStoragePlanError(
                    f"{primary_output.name}: zero-copy primary Alpha split "
                    "relation changed"
                )
            if plan.physical_id(plane, secondary_output) != draft.auxiliary:
                raise AlphaStoragePlanError(
                    f"{secondary_output.name}: zero-copy shuffle output is "
                    "not bound to its auxiliary backing"
                )
            secondary_relation = plan.relations[plane][secondary_output]
            expected_secondary = ViewTransform(
                "alpha_shuffle_output",
                parameters=(("output", 1),),
            )
            if (
                secondary_relation.kind is not StorageRelationKind.VIEW
                or secondary_relation.source is not None
                or secondary_relation.physical_source != draft.auxiliary
                or secondary_relation.component is not None
                or secondary_relation.transform != expected_secondary
            ):
                raise AlphaStoragePlanError(
                    f"{secondary_output.name}: zero-copy secondary Alpha "
                    "auxiliary relation changed"
                )
            if primary_output.node is not secondary_output.node:
                raise AlphaStoragePlanError(
                    f"{draft.shuffle.name}: zero-copy split outputs have "
                    "different record owners"
                )
            payload_uses.append(
                AlphaPayloadUse(
                    operation=primary_output.node,
                    field="input",
                    logical_tensor=shuffle_ref,
                    physical_tensor=primary,
                    entry_offset=draft.primary_entry_offset,
                    span_entries=draft.span_entries,
                    row_pitch_entries=draft.pitch_entries,
                )
            )
            # Split outputs are full logical tensor views, not one-entry NMP
            # copy ports.  _freeze_alpha_split_storage owns their normalized
            # spans and row pitches so they appear exactly once downstream.

        frozen_movements = []
        for copy, movement in zip(draft.copies, draft.movements):
            input_node = copy.args[0]
            input_ref = _single_ref(graph, input_node)
            output_ref = _single_ref(graph, copy)
            relation = plan.relation(plane, output_ref)
            expected_transform = ViewTransform(
                "alpha_shuffle_copy",
                parameters=(("copy_index", movement.index),),
            )
            if (
                relation.kind is not StorageRelationKind.WRITE_INTO
                or relation.source != shuffle_ref
                or relation.physical_source is not None
                or relation.component is not None
                or relation.transform != expected_transform
                or plan.physical_id(plane, output_ref) != primary
            ):
                raise AlphaStoragePlanError(
                    f"{copy.name}: finalized semantic shuffle-copy relation "
                    "changed"
                )
            source_physical = (
                primary
                if movement.source_output == 0
                else draft.auxiliary
            )
            source_entry = movement.source_entry + (
                draft.primary_entry_offset
                if movement.source_output == 0
                else 0
            )
            destination_entry = (
                draft.primary_entry_offset + movement.destination_entry
            )
            source_view = ViewTransform(
                "alpha_shuffle_copy_source",
                parameters=(
                    ("output", movement.source_output),
                    ("entry", source_entry),
                    ("pitch_entries", draft.pitch_entries),
                ),
            )
            destination_view = ViewTransform(
                "alpha_shuffle_copy_destination",
                parameters=(
                    ("entry", destination_entry),
                    ("pitch_entries", draft.pitch_entries),
                ),
            )
            frozen_movements.append(
                AlphaShuffleMovement(
                    copy=copy,
                    source_output=movement.source_output,
                    source_entry=movement.source_entry,
                    destination_entry=movement.destination_entry,
                    block_channels=draft.block_channels,
                    source_physical=source_physical,
                    destination_physical=primary,
                    source_view=source_view,
                    destination_view=destination_view,
                )
            )
            payload_uses.extend(
                (
                    AlphaPayloadUse(
                        operation=copy,
                        field="input",
                        logical_tensor=input_ref,
                        physical_tensor=source_physical,
                        entry_offset=source_entry,
                        span_entries=1,
                        row_pitch_entries=draft.pitch_entries,
                    ),
                    AlphaPayloadUse(
                        operation=copy,
                        field="output_payload",
                        logical_tensor=output_ref,
                        physical_tensor=primary,
                        entry_offset=destination_entry,
                        span_entries=1,
                        row_pitch_entries=draft.pitch_entries,
                    ),
                    AlphaPayloadUse(
                        operation=copy,
                        field="copy_source_payload",
                        logical_tensor=input_ref,
                        physical_tensor=source_physical,
                        entry_offset=0,
                        span_entries=(
                            primary_entries
                            if source_physical == primary
                            else auxiliary_entries
                        ),
                        row_pitch_entries=draft.pitch_entries,
                    ),
                    AlphaPayloadUse(
                        operation=copy,
                        field="copy_destination_payload",
                        logical_tensor=output_ref,
                        physical_tensor=primary,
                        entry_offset=0,
                        span_entries=primary_entries,
                        row_pitch_entries=draft.pitch_entries,
                    ),
                )
            )
            artifact = f"{copy.name}.copy_block_payload"
            plan.serialized_artifact(output_ref, plane, artifact)
            serialized_uses.append(
                AlphaSerializedUse(
                    operation=copy,
                    field="copy_block_payload",
                    logical_tensor=output_ref,
                    artifact=artifact,
                )
            )

        plans.append(
            AlphaShufflePlan(
                shuffle=draft.shuffle,
                primary=primary,
                auxiliary=draft.auxiliary,
                copies=draft.copies,
                movements=tuple(frozen_movements),
                block_channels=draft.block_channels,
                output_channels=_shape(draft.shuffle)[1],
                pitch_entries=draft.pitch_entries,
                span_entries=draft.span_entries,
                primary_entry_offset=draft.primary_entry_offset,
            )
        )
    return _ShuffleStorageFacts(
        plans=tuple(plans),
        payload_uses=tuple(payload_uses),
        serialized_uses=tuple(serialized_uses),
    )


def _nmp_lanes(config) -> int:
    """Resolve Alpha's NMP channel width from the selected machine."""
    unit = config.unit("nmp") if config is not None else None
    if unit is None:
        lanes = NMP_UNIT.unroll("CHANNEL") or NMP_UNIT.lanes
    else:
        lanes = unit.unroll("CHANNEL") or unit.lanes
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
        raise AlphaStoragePlanError(
            f"Alpha NMP lanes must be a positive integer, not {lanes!r}"
        )
    return lanes


def _resident_physical_ids(
    plan,
    operation_constraints,
    egress_boundaries,
) -> tuple[PhysicalTensorId, ...]:
    """Return the exact scratchpad backings the flat Alpha path allocates."""
    resident = set()
    resident.update(
        plan.physical_id(_SCRATCHPAD, logical)
        for constraint in operation_constraints
        for logical in (
            *(operand.logical_tensor for operand in constraint.operands),
            constraint.output,
        )
    )
    resident.update(
        physical
        for physical, tensor in plan.physical_tensors[_SCRATCHPAD].items()
        if tensor.accesses
    )
    resident.update(
        plan.physical_id(_SCRATCHPAD, plan.ref(node))
        for boundary in egress_boundaries
        for node in boundary.boundary_nodes
    )
    return tuple(sorted(resident, key=lambda physical: physical.name))


def _freeze_alpha_cluster_constraints(
    graph: StorageGraph,
    plan,
    drafts: tuple[_ClusterDraft, ...],
    resident_physical_ids: tuple[PhysicalTensorId, ...],
    *,
    plane: str = _SCRATCHPAD,
) -> tuple[AlphaClusterConstraint, ...]:
    """Normalize logical bank decisions onto exact finalized backings."""
    plane = StoragePlane(plane).value
    if plan.graph is not graph:
        raise AlphaStoragePlanError(
            "Alpha cluster constraints must be frozen from their declaring graph"
        )
    residents = set(resident_physical_ids)
    normalized = {}
    for draft in drafts:
        logical = draft.logical_tensor
        try:
            if _bits_for_ref(graph, logical) != 8:
                raise AlphaStoragePlanError(
                    f"{logical.name}: Alpha one-bank constraint is not INT8"
                )
            physical = plan.physical_id(plane, logical)
            tensor = plan.physical_tensor(plane, physical)
            primary = tensor.component("primary")
        except (KeyError, TypeError, ValueError) as error:
            raise AlphaStoragePlanError(
                f"{logical.name}: Alpha cluster constraint has no exact "
                "physical backing"
            ) from error
        if physical not in residents:
            # A structural value can receive a provisional cluster hint before
            # its unused backing is folded out of the resident set. With no
            # executable access, there is no allocation to constrain.
            continue
        if primary.bits != 8:
            raise AlphaStoragePlanError(
                f"{physical.name}: Alpha one-bank backing is not INT8"
            )
        previous = normalized.get(physical)
        if previous is not None and previous != draft.cluster:
            raise AlphaStoragePlanError(
                f"{physical.name}: logical aliases require conflicting Alpha "
                f"bank clusters {previous} and {draft.cluster}"
            )
        normalized[physical] = draft.cluster

    return tuple(
        AlphaClusterConstraint(physical, cluster)
        for physical, cluster in sorted(
            normalized.items(), key=lambda item: item[0].name
        )
    )


def _peak_resident_bytes(
    plan,
    resident_physical_ids: tuple[PhysicalTensorId, ...],
) -> int:
    """Recompute the DP objective from one refinalized storage plan."""
    residents = set(resident_physical_ids)
    intervals = plan.intervals_by_plane[_SCRATCHPAD]
    return max(
        (
            sum(
                plan.backing_size_bytes(physical, _SCRATCHPAD)
                for physical, interval in intervals.items()
                if (
                    physical in residents
                    and interval.birth <= position < interval.death
                )
            )
            for position in range(len(plan.execution_order.nodes))
        ),
        default=0,
    )


def _prepare_alpha_end_to_end_storage(
    model: GraphModule,
    config,
    order_mode: str,
) -> AlphaPreparedStorage:
    """Prepare Alpha's complete pre-allocation end-to-end storage handoff.

    This helper owns relation, residency, and execution-order completeness.
    Allocation and serialization consume only the immutable result returned
    here.
    """
    normalized_order, mapping_mode = validate_execution_order_mode(
        order_mode,
        END_TO_END_SCRATCHPAD_MAPPING,
    )
    if mapping_mode != END_TO_END_SCRATCHPAD_MAPPING or normalized_order not in {
        FX_EXECUTION_ORDER_MODE,
        DP_EXECUTION_ORDER_MODE,
    }:
        raise AlphaStoragePlanError(
            f"unsupported Alpha end-to-end order {order_mode!r}"
        )

    lanes = _nmp_lanes(config)
    source_order = fx_execution_order(model)
    graph = StorageGraph(
        model,
        planes=(StoragePlane.DRAM, StoragePlane.SCRATCHPAD),
        scope=GRAPH_SCOPE_OUTER_FX,
    )
    operation_constraints = _collect_operation_constraints(model, graph)
    executable_owners = tuple(
        constraint.owner for constraint in operation_constraints
    )
    if len(set(executable_owners)) != len(executable_owners):
        raise AlphaStoragePlanError(
            "Alpha executable owners contain duplicates"
        )
    egress_drafts = _declare_alpha_egress_storage(
        model,
        graph,
        operation_constraints,
    )

    vsq_draft = _declare_alpha_vsq_relations(
        model,
        graph,
        operation_constraints,
        config=config,
    )
    gemm_drafts = _declare_alpha_gemm_storage(
        model,
        graph,
        operation_constraints,
        vsq_draft,
        config=config,
    )
    vsq_payloads = frozenset(pair.payload for pair in vsq_draft.pairs)
    gemm_input_clusters = {
        draft.materialized_input.logical_tensor: 1
        for draft in gemm_drafts
        if draft.materialized_input is not None
    }
    cluster_drafts = _declare_alpha_cluster_constraints(
        model,
        graph,
        operation_constraints,
        config=config,
        initial_constraints=gemm_input_clusters,
    )
    logical_clusters = {
        draft.logical_tensor: draft.cluster for draft in cluster_drafts
    }
    split_drafts = _declare_alpha_split_storage(
        model,
        graph,
        block_channels=lanes,
        cluster_constraints=logical_clusters,
    )
    resident_drafts = _declare_alpha_resident_storage(
        model,
        graph,
        operation_constraints,
        block_channels=lanes,
        vsq_payloads=vsq_payloads,
        cluster_constraints=logical_clusters,
    )
    shuffle_drafts = _declare_alpha_shuffle_storage(
        model,
        graph,
        block_channels=lanes,
        split_drafts=split_drafts,
    )
    _register_alpha_vsq_backings(graph, vsq_draft)
    _declare_alpha_vsq_artifacts(graph, vsq_draft)
    provisional_plan = graph.finalize(
        execution_order=source_order,
        end_to_end=True,
    )
    resident_physical_ids = _resident_physical_ids(
        provisional_plan,
        operation_constraints,
        egress_drafts,
    )

    order_result = None
    final_plan = provisional_plan
    selected_order = source_order
    if normalized_order == DP_EXECUTION_ORDER_MODE:
        order_result = solve_execution_order_dp(
            model,
            provisional_plan,
            executable_owners,
            storage_plane=StoragePlane.SCRATCHPAD,
            resident_physical_ids=resident_physical_ids,
        )
        selected_order = order_result.order
        final_plan = graph.finalize(
            execution_order=selected_order,
            end_to_end=True,
        )
        final_residents = _resident_physical_ids(
            final_plan,
            operation_constraints,
            egress_drafts,
        )
        if final_residents != resident_physical_ids:
            raise AlphaStoragePlanError(
                "selected Alpha execution order changed resident physical "
                "storage membership"
            )
        measured_peak = _peak_resident_bytes(
            final_plan,
            resident_physical_ids,
        )
        if measured_peak != order_result.peak_live_bytes:
            raise AlphaStoragePlanError(
                "selected Alpha storage-plan liveness disagrees with the DP "
                f"objective: plan={measured_peak}, "
                f"solver={order_result.peak_live_bytes}"
            )

    cluster_constraints = _freeze_alpha_cluster_constraints(
        graph,
        final_plan,
        cluster_drafts,
        resident_physical_ids,
    )
    egress_boundaries = _freeze_alpha_egress_storage(
        final_plan,
        egress_drafts,
    )
    resident_writes = _freeze_alpha_resident_storage(
        graph,
        final_plan,
        resident_drafts,
    )
    shuffle_facts = _freeze_alpha_shuffle_storage(
        graph,
        final_plan,
        shuffle_drafts,
    )
    split_views = _freeze_alpha_split_storage(
        graph,
        final_plan,
        split_drafts,
        shuffle_plans=shuffle_facts.plans,
    )
    nmp_input_views = _freeze_alpha_nmp_input_views(
        final_plan,
        operation_constraints,
        nmp_lanes=lanes,
    )
    nmp_bindings = _freeze_alpha_nmp_storage(
        graph,
        final_plan,
        operation_constraints,
        nmp_input_views,
        nmp_lanes=lanes,
    )
    vsq_facts = _freeze_alpha_vsq_storage(
        graph,
        final_plan,
        vsq_draft,
    )
    gemm_facts = _freeze_alpha_gemm_storage(
        graph,
        final_plan,
        gemm_drafts,
    )
    logical_views = (
        vsq_facts.logical_views
        + split_views
    )
    logical_view_refs = tuple(
        binding.logical_tensor for binding in logical_views
    )
    if len(set(logical_view_refs)) != len(logical_view_refs):
        raise AlphaStoragePlanError(
            "Alpha logical storage views contain duplicate values"
        )

    return AlphaPreparedStorage(
        graph=graph,
        provisional_plan=provisional_plan,
        plan=final_plan,
        execution_order=selected_order,
        executable_owners=executable_owners,
        operation_constraints=operation_constraints,
        resident_physical_ids=resident_physical_ids,
        allocation_requests=(),
        cluster_constraints=cluster_constraints,
        egress_boundaries=egress_boundaries,
        gemm_bindings=gemm_facts.bindings,
        resident_writes=resident_writes,
        nmp_input_views=nmp_input_views,
        nmp_bindings=nmp_bindings,
        shuffle_plans=shuffle_facts.plans,
        logical_views=logical_views,
        payload_uses=(
            gemm_facts.payload_uses + shuffle_facts.payload_uses
        ),
        serialized_uses=(
            vsq_facts.serialized_uses + shuffle_facts.serialized_uses
        ),
        order_result=order_result,
    )


def plan_alpha_storage(
    model: GraphModule,
    *,
    config=None,
    scratchpad_mapping=None,
    execution_order=None,
):
    """Build Alpha's selected flat-path storage handoff.

    The legacy plan remains attached immediately for its existing structural
    serializer.  End-to-end state is returned without publication so the
    terminal pipeline can publish only the allocation-complete handoff after
    strict serialization succeeds.
    """
    order_mode, mapping_mode = validate_execution_order_mode(
        execution_order, scratchpad_mapping
    )
    require_execution_mode(order_mode, mapping_mode, EXECUTION_MODES)
    if mapping_mode == END_TO_END_SCRATCHPAD_MAPPING:
        return _prepare_alpha_end_to_end_storage(
            model,
            config,
            order_mode,
        )
    order = fx_execution_order(model)
    plan = StorageGraph(
        model,
        planes=(StoragePlane.DRAM,),
        scope=GRAPH_SCOPE_OUTER_FX,
    ).finalize(execution_order=order)
    model.meta["storage_plan"] = plan
    return plan


__all__ = ["EXECUTION_MODES", "plan_alpha_storage"]
