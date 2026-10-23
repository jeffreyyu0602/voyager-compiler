# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Project finalized Alpha state into flat ``model.txt`` collateral.

Alpha's output is not the loop-nest IR. It is one record per call, with
addresses resolved before serialization in the ``param.proto`` form. Planning
belongs to :mod:`.pipeline`; this module only validates and projects decisions
already bound to the graph.

It is being built in the order its parts can be *checked*, which is not the
order they appear in the file:

1. **structure** -- which records the graph would emit, and how many of each.
   That is the 144 gate, and it can be measured before a single address
   exists. It is what this module does today.
2. **fields** -- the semantic kwargs, from the operators and the verified
   encoders.
3. **placement** -- `output_payload`, addresses, memory instances, from the
   storage plan and Alpha's allocation. Pure projection helpers live here;
   binding the finalized plan to records remains the pipeline's job.

Structure comes first because a missing record can otherwise leave a
well-formed but incomplete file. The explicit 144-record inventory makes that
failure visible at the compiler boundary.
"""

import collections
import logging
import operator
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from scheduler.voyager import LOCAL_SCHEDULE_META
from voyager_compiler.codegen import param_pb2
from voyager_compiler.codegen.node_info import (
    get_arg_value,
    get_hardware_logical_shape,
    require_allocation,
)
from voyager_compiler.codegen.storage import (
    PhysicalPayloadRef,
    SerializedTensorRef,
    StoragePlane,
    ValueRef,
)
from voyager_compiler.codegen.storage.types import recorded_value
from voyager_compiler.codegen.targets.magnet_alpha.emit import (
    attach_record_output,
    attach_record_role,
    build_argument,
    build_fused_record,
    build_memory,
    build_model,
    build_operation,
    build_single_record,
    build_tensor,
)
from voyager_compiler.codegen.targets.magnet_alpha.operators import (
    execution_kind,
)
from voyager_compiler.codegen.targets.magnet_alpha.traits import (
    NON_ALLOCATING_OPERAND_META,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "address_only_target",
    "emitted_inventory",
    "layer_table",
    "mishead_count",
    "record_sequences",
    "record_target",
    "serialize_finalized_model",
    "serialize_model",
]

#: FX target -> the name a record carries. Alpha's own operators keep their
#: namespace's name; the two near-memory operations the reference leaves as
#: ATen keep theirs, because that is what a reference compile emits.
_RECORD_TARGETS: Dict[object, str] = {
    torch.ops.magnet_alpha.conv2d_int8_accumulator.default: (
        "conv2d_int8_accumulator"
    ),
    torch.ops.magnet_alpha.conv2d_vsq.default: "conv2d_vsq",
    torch.ops.magnet_alpha.ppu_fixed_relu.default: "ppu_fixed_relu",
    torch.ops.magnet_alpha.nmp_fixed_requantize.default: (
        "nmp_fixed_requantize"
    ),
    torch.ops.magnet_alpha.pvs_quantize.default: "pvs_quantize",
    torch.ops.magnet_alpha.channel_shuffle.default: "channel_shuffle",
    torch.ops.magnet_alpha.channel_shuffle_copy.default: (
        "channel_shuffle_copy"
    ),
    torch.ops.aten.max_pool2d.default: "max_pool2d",
    # Both overloads, because which one a graph carries is decided by how the
    # model asked for the resize, not by the target. ``F.interpolate`` with a
    # ``scale_factor`` exports as ``.vec``, and that is what this workload has
    # -- ``quantization.regions`` roots its domain on ``.vec`` for the same
    # reason. Keying only ``.default`` costs two records, and costs them
    # *silently*: an unrecognized node is simply not a record, so the layer
    # count drops and nothing reports why.
    torch.ops.aten.upsample_nearest2d.default: "upsample_nearest2d",
    torch.ops.aten.upsample_nearest2d.vec: "upsample_nearest2d",
    torch.ops.magnet_alpha.upsample_nearest2d.default: (
        "upsample_nearest2d"
    ),
}


#: What a record calls each operand, where that differs from the operator's
#: own parameter name.
#:
#: These names are an **ABI**, not a style: the converter binds operands by
#: keyword (``missing=input,weight`` is what it says when it cannot find
#: them), so an operand under the wrong key is not a renamed operand -- it is
#: an absent one, and the record it belongs to emits no layer.
#:
#: Read off a reference compile rather than transcribed: for each executable
#: target, the keys every one of its records carries. Only the disagreements
#: are listed; everything else already matches, which is why the table is
#: short and why a new operator normally needs no entry at all.
#:
#: ``conv2d_vsq`` used to need two entries here, for scales its operator
#: called ``fine_scale`` / ``input_fine_scale``. It does not any more: the
#: operator was renamed to the record's own names, which is better than
#: translating them -- a table can drift from the schema it translates, and a
#: drifted entry produces an operand the converter reports as *missing*.
_RECORD_OPERAND_NAMES: Dict[object, Dict[str, str]] = {
    torch.ops.magnet_alpha.ppu_fixed_relu.default: {
        "accumulator": "input",
        "bias": "bias_payload",
    },
    # ATen names a unary operation's operand ``self``; a record calls it
    # ``input``, as it does for every other operation.
    torch.ops.aten.max_pool2d.default: {"self": "input"},
    torch.ops.aten.upsample_nearest2d.default: {"self": "input"},
    # These are ordinary records only inside a fully frozen terminal egress
    # boundary.  Their ABI still follows Alpha's universal unary-port name.
    torch.ops.aten.permute.default: {"self": "input"},
    torch.ops.aten.contiguous.default: {"self": "input"},
}


#: What the endpoint fetches for each semantic operand, by record and key.
#:
#: **A record's operand dtype is a property of the record, not of the graph
#: node.** A convolution weight stays a float ``get_attr`` all the way through
#: lowering -- deliberately, because quantizing it there would be a second
#: opinion about a tensor the quantizer already decided -- and the record is
#: where it is stated as the int8 or int4 the array actually reads. The graph
#: node has no dtype to read *because nothing has narrowed it yet*, so a
#: serializer that reports what the node says reports ``float32``, and the
#: converter refuses it: ``unsupported-target-precision``, 66 of them, one per
#: matrix record.
#:
#: Declaring it is sound rather than convenient, and the reason is measured:
#: across a whole reference compile, every one of these keys carries exactly
#: one dtype -- all 49 ``conv2d_int8_accumulator`` records read an int8 input
#: and an int8 weight, all 36 ``conv2d_vsq`` records read int4 under a uint8
#: scale, and no record crosses over. The head is what decides, and the head
#: is what picks the target here.
#:
#: Only the **semantic** operands. Placement operands -- ``output_payload``,
#: ``weight_payload``, the vector-scale payloads -- name storage rather than
#: values, and they arrive from allocation with their own widths.
_RECORD_OPERAND_DTYPES: Dict[object, Dict[str, str]] = {
    torch.ops.magnet_alpha.conv2d_int8_accumulator.default: {
        "input": "int8",
        "weight": "int8",
    },
    torch.ops.magnet_alpha.conv2d_vsq.default: {
        "input": "int4",
        "weight": "int4",
        "folded_coarse_scale": "float32",
        "semantic_input_coarse_scale": "float32",
        "semantic_weight_coarse_scale": "float32",
        # ``input_scale`` and ``weight_scale`` are deliberately **not**
        # declared. They are separate tensors -- uint8 mantissas -- not a
        # reinterpretation of the float the graph holds, so declaring uint8
        # here would state the right dtype over the wrong payload, and the
        # collateral would convert while its tensor file contradicted its own
        # field. Left undeclared, the record reports whatever
        # ``finalize_alpha_vsq_operands`` actually produced: uint8 when the
        # mantissa is there, and float32 -- refused loudly by the converter --
        # when it is not.
    },
    torch.ops.magnet_alpha.ppu_fixed_relu.default: {
        # The accumulator, at the matrix unit's own width.
        "input": "int32",
        "bias_payload": "float32",
        "primary_multiplier": "float32",
    },
    torch.ops.magnet_alpha.nmp_fixed_requantize.default: {
        "accumulator": "int32",
        "semantic_accumulator_scale": "float32",
    },
    torch.ops.magnet_alpha.pvs_quantize.default: {
        # Wider than the epilogue's own container, and the reference says so
        # on all 38: this stage reads the epilogue's *pre-truncation* value,
        # which is why it can narrow it.
        "input": "int64",
        "coarse_scale": "float32",
    },
    torch.ops.magnet_alpha.channel_shuffle.default: {
        "input1": "int8",
        "input2": "int8",
    },
    torch.ops.magnet_alpha.channel_shuffle_copy.default: {"input": "int8"},
    torch.ops.aten.max_pool2d.default: {"input": "int8"},
    torch.ops.aten.upsample_nearest2d.default: {"input": "int8"},
    torch.ops.aten.upsample_nearest2d.vec: {"input": "int8"},
    torch.ops.magnet_alpha.upsample_nearest2d.default: {"input": "int8"},
}


def _operand_names(target) -> List[str]:
    """The record keys this target's positional operands take, in order.

    From the operator's own schema, so a signature change cannot leave the
    serializer describing the old one, with the declared aliases applied on
    top. An operator with no schema keeps the positional fallback -- there is
    none in ``_RECORD_TARGETS`` today, and inventing a name for one would be
    a guess the converter would silently reject.
    """
    schema = getattr(target, "_schema", None)
    if schema is None:
        return []
    aliases = _RECORD_OPERAND_NAMES.get(target, {})
    return [
        aliases.get(argument.name, argument.name)
        for argument in schema.arguments
    ]


def record_target(node: Node):
    """The record name ``node`` would carry, or ``None``.

    ``None`` is not an error: most nodes in a transformed graph are movement
    and layout, and a reference compile emits those as NOPs.
    """
    return _RECORD_TARGETS.get(getattr(node, "target", None))


def _submodule(model: GraphModule, node: Node):
    if node.op != "call_module":
        return None
    return getattr(model, str(node.target), None)


def record_sequences(model: GraphModule) -> List[Tuple[str, ...]]:
    """The target sequence of every record the graph would emit.

    A fused ``call_module`` contributes the sequence of its body; anything
    else contributes itself. Sequences are what the converter dispatches on --
    see ``operators.execution_kind`` -- so this is the graph expressed in the
    only vocabulary that decides whether a layer appears.
    """
    sequences = []
    for node in model.graph.nodes:
        submodule = _submodule(model, node)
        if submodule is not None and hasattr(submodule, "graph"):
            inner = tuple(
                name
                for name in (
                    record_target(child) for child in submodule.graph.nodes
                )
                if name is not None
            )
            if inner:
                sequences.append(inner)
            continue
        name = record_target(node)
        if name is not None:
            sequences.append((name,))
    return sequences


def emitted_inventory(model: GraphModule) -> Dict[str, int]:
    """How many Alpha calls of each execution kind this graph would produce.

    The number to compare against a reference compile: 66 `gemm_convolution`,
    19 `nmp_depthwise_convolution`, 40 `nmp_copy`, 16 `nmp_channel_shuffle`,
    2 `nmp_nearest_upsample`, 1 `nmp_max_pool` -- 144.

    A sequence outside the registered Alpha vocabulary is counted under
    ``unemitted``, deliberately and by name, so it cannot disappear silently.
    """
    counts: Dict[str, int] = collections.Counter()
    for sequence in record_sequences(model):
        kind = execution_kind(sequence)
        counts[kind if kind is not None else "unemitted"] += 1
    return dict(counts)


def mishead_count(model: GraphModule) -> int:
    """Accumulators emitted under a head their quantizer did not ask for.

    The lowering stamps each accumulator with the head the *quantizer's* own
    per-layer decision calls for (``lowering.HEAD_META``). Until the VSQ
    fine-scale operand is materialized, a boundary annotated ``int4`` is still
    emitted as ``conv2d_int8_accumulator``: the record count stays right and
    the arithmetic does not.

    This counts them, so the gap is a number a test can assert on rather than
    a sentence in a document. It must reach zero before any claim of Alpha
    numerical correctness, and it does *not* have to be zero for the 144-record
    structural gate -- which is exactly why it needs its own name.
    """
    from voyager_compiler.codegen.targets.magnet_alpha.lowering import (
        HEAD_META,
        VSQ_HEAD,
    )

    mishead = 0
    for node in model.graph.nodes:
        submodule = _submodule(model, node)
        children = (
            list(submodule.graph.nodes)
            if submodule is not None and hasattr(submodule, "graph")
            else [node]
        )
        for child in children:
            if record_target(child) != "conv2d_int8_accumulator":
                continue
            if child.meta.get(HEAD_META) == VSQ_HEAD:
                mishead += 1
    return mishead


def _is_nop(operation) -> bool:
    """Whether an operation issues nothing on the endpoint.

    A record whose every member is a ``nop`` is address arithmetic: a view, a
    concatenation Alpha performs by writing its branches where they belong, a
    slice that restores a logical extent. It occupies a place in the model and
    no time on the machine.
    """
    kind = operation.WhichOneof("op_type")
    if kind == "fused_op":
        members = operation.fused_op.op_list
        return bool(members) and all(member.op == "nop" for member in members)
    if kind == "op":
        return operation.op.op == "nop"
    return False


def layer_table(model, params) -> str:
    """The ordered list of layers this collateral runs.

    One bare name per line, in execution order -- not the shared writer's
    five-column table, which describes a *bufferized* nest: its start and end
    ops, the equivalence class it was built from, and its trip count. A flat
    record has none of those, and the shared writer reads ``op.name`` from
    each emitted operation, which ``param_pb2.Operation`` does not have. So it
    would not merely describe the wrong thing here, it would raise.

    Consumers read this file as an *order*: ``convert.py`` writes its own
    beside the converted collateral, and downstream tools treat it as the
    authoritative operation order. So what matters is that the sequence is
    the one the model.txt has, which is why it is derived from ``params``
    rather than walked from the graph a second time.

    Nops are excluded, and that is the rule the reference follows exactly: for
    the YOLOv5 workload, the non-nop operation names *are* its 153
    lines, in order, with nothing else.

    Args:
        model: The transformed graph. Unused -- the order is the emitted
            model's, and re-deriving it from the graph would be a second
            opinion about the same sequence.
        params: The emitted ``param_pb2.Model``.

    Returns:
        The layer names, newline separated.
    """
    del model
    names = []
    for operation in params.ops:
        if _is_nop(operation):
            continue
        kind = operation.WhichOneof("op_type")
        if kind is None:
            continue
        names.append(getattr(operation, kind).name)
    return "\n".join(names)


# -----------------------------------------------------------------------------
# The terminal path
# -----------------------------------------------------------------------------


def _dtype_name(dtype) -> str:
    """The protobuf spelling of a torch or already-normalized dtype."""
    text = str(dtype)
    return text.removeprefix("torch.")


def _memory_for_placement(placement):
    """Project one already-decided placement to collateral memory metadata.

    Alpha's allocator distinguishes the arena offset (``start``) from the
    hardware-encoded address (``address``).  The converter consumes the
    latter.  This helper intentionally has no fallback to ``start``: a
    placement without a finalized encoded address is not serializable.
    """
    required = ("memory_space", "address", "memory_instance")
    missing = [name for name in required if not hasattr(placement, name)]
    if missing:
        raise TypeError(
            "finalized placement is missing " + ", ".join(missing)
        )

    partition = placement.memory_space
    address = placement.address
    instance = placement.memory_instance
    if isinstance(partition, bool) or not isinstance(partition, int):
        raise TypeError("placement memory_space must be an integer")
    if partition < 0:
        raise ValueError("placement memory_space must not be negative")
    if isinstance(address, bool) or not isinstance(address, int):
        raise TypeError("placement address must be an integer")
    if address < 0:
        raise ValueError("placement address must not be negative")
    if not isinstance(instance, str) or not instance:
        raise ValueError(
            "placement memory_instance must be a non-empty string"
        )
    return build_memory(
        partition=partition,
        address=address,
        instance=instance,
    )


def _write_tensor_payload(value: torch.Tensor, filename: str) -> None:
    """Synchronously write the converter's deterministic float32 payload.

    A tensor name is a file identity.  Re-projecting the same bytes is
    idempotent; projecting different bytes to an existing identity is a
    compiler error rather than a last-writer-wins placement decision.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError("serialized tensor payload must be a torch.Tensor")
    payload = (
        value.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
    )
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "rb") as stream:
            existing = stream.read()
        if existing != payload:
            raise ValueError(f"conflicting tensor payload for {filename}")
        return
    with open(filename, "wb") as stream:
        stream.write(payload)


def _tensor_from_physical_payload(
    payload: PhysicalPayloadRef,
    *,
    tensor_dir: Optional[str] = None,
):
    """Project one physical component and only its named storage plane."""
    if not isinstance(payload, PhysicalPayloadRef):
        raise TypeError("payload must be a PhysicalPayloadRef")

    placement = _memory_for_placement(payload.segment)
    plane = StoragePlane(payload.physical_tensor.plane)
    kwargs = {plane.value: placement}
    tensor = build_tensor(
        payload.name,
        payload.representation.shape,
        _dtype_name(payload.representation.dtype),
        **kwargs,
    )
    if tensor_dir is not None:
        if payload.representation.value is None:
            raise ValueError(f"{payload.name}: physical payload has no value")
        _write_tensor_payload(
            payload.representation.value,
            os.path.join(tensor_dir, payload.name + ".bin"),
        )
    return tensor


def _tensor_from_serialized_ref(
    reference: SerializedTensorRef,
    *,
    tensor_dir: Optional[str] = None,
):
    """Project a nonresident artifact; it deliberately has no placement."""
    if not isinstance(reference, SerializedTensorRef):
        raise TypeError("reference must be a SerializedTensorRef")
    artifact = reference.artifact
    tensor = build_tensor(
        reference.name,
        artifact.shape,
        _dtype_name(artifact.dtype),
    )
    if tensor_dir is not None:
        _write_tensor_payload(
            artifact.value,
            os.path.join(tensor_dir, reference.name + ".bin"),
        )
    return tensor


def _tensor_for_logical(
    node,
    *,
    dtype_override=None,
    memory_placement=None,
    scratchpad_placement=None,
):
    """Project one logical FX tensor plus explicitly supplied placements.

    The caller owns relation tracing, offset calculation, and allocation.
    Supplying neither placement therefore preserves the pre-storage
    serializer behavior exactly.

    Shape and dtype come from the traced value. A placement is included only
    when the caller supplies its finalized answer; absent does not mean
    address zero, because zero is a valid address.
    """
    value = getattr(node, "value", None)
    shape = tuple(getattr(value, "shape", ()) or ())
    dtype = dtype_override or node.meta.get("dtype")
    if dtype is None and value is not None:
        dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
    memory = (
        _memory_for_placement(memory_placement)
        if memory_placement is not None
        else None
    )
    scratchpad = (
        _memory_for_placement(scratchpad_placement)
        if scratchpad_placement is not None
        else None
    )
    return build_tensor(
        node.name,
        shape,
        dtype or "",
        memory=memory,
        scratchpad=scratchpad,
    )


def _tensor_for(node, *, dtype_override=None):
    """Backward-compatible unplaced logical-tensor projection."""
    return _tensor_for_logical(node, dtype_override=dtype_override)


def _logical_abi_name(logical: ValueRef) -> str:
    """Return the flat collateral name for one logical tensor leaf."""
    suffix = "".join(f"_{index}" for index in logical.path)
    return f"{logical.node.name}{suffix}"


def _logical_value(logical: ValueRef):
    value = recorded_value(logical.node)
    try:
        for index in logical.path:
            value = value[index]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            f"{logical.name}: finalized logical value path is unavailable"
        ) from error
    return value


class _AlphaFinalizedProjector:
    """Project immutable Alpha bindings and account for every exact site."""

    def __init__(self, finalized, *, tensor_dir=None):
        from voyager_compiler.codegen.targets.magnet_alpha.finalization import (
            AlphaFinalizedSerialization,
        )

        if not isinstance(finalized, AlphaFinalizedSerialization):
            raise TypeError(
                "finalized Alpha serialization needs "
                "AlphaFinalizedSerialization"
            )
        self.finalized = finalized
        self.plan = finalized.prepared.plan
        self.tensor_dir = tensor_dir
        self.consumed = []

    def _logical_tensor(self, binding):
        logical = binding.value
        try:
            description = self.plan.logical_tensors[logical]
        except KeyError as error:
            raise ValueError(
                f"{binding.site.label}: logical tensor is absent from the plan"
            ) from error
        tensor = build_tensor(
            _logical_abi_name(logical),
            description.shape,
            _dtype_name(description.dtype),
            memory=(
                _memory_for_placement(binding.memory)
                if binding.memory is not None
                else None
            ),
            scratchpad=(
                _memory_for_placement(binding.scratchpad)
                if binding.scratchpad is not None
                else None
            ),
        )
        return tensor

    def _project(self, binding):
        if isinstance(binding.value, ValueRef):
            return self._logical_tensor(binding)
        if isinstance(binding.value, PhysicalPayloadRef):
            return _tensor_from_physical_payload(
                binding.value,
                tensor_dir=self.tensor_dir,
            )
        if isinstance(binding.value, SerializedTensorRef):
            return _tensor_from_serialized_ref(
                binding.value,
                tensor_dir=self.tensor_dir,
            )
        return binding.value

    def model_tensor(self, node: Node, field: str):
        bindings = tuple(
            binding
            for binding in self.finalized.bindings_for_member(node, None)
            if binding.site.field == field
        )
        if not bindings:
            return None
        if len(bindings) != 1:
            raise ValueError(
                f"{node.name}: finalized model tensor binding is ambiguous"
            )
        binding = bindings[0]
        value = self._project(binding)
        self.consumed.append(binding.site)
        return value

    @staticmethod
    def _require_projected_tensor(binding, value):
        if not isinstance(value, param_pb2.Tensor):
            raise ValueError(
                f"{binding.site.label}: tensor occurrence did not project "
                "to a Tensor"
            )
        return value

    def _patch_member_argument(self, binding, record, value) -> None:
        """Patch one already-built member argument at its exact occurrence."""
        site = binding.site
        occurrence = site.occurrence
        if site.field == "args":
            if (
                len(occurrence) != 1
                or isinstance(occurrence[0], bool)
                or not isinstance(occurrence[0], int)
            ):
                raise ValueError(
                    f"{site.label}: positional binding needs one integer "
                    "occurrence"
                )
            index = occurrence[0]
            if index < 0 or index >= len(record.args):
                raise ValueError(
                    f"{site.label}: positional occurrence is out of range"
                )
            record.args[index].CopyFrom(build_argument(value))
            return

        if not occurrence:
            # Target-owned derived fields (padding, pitch, L2 tiling, and
            # placement roles) are deliberately absent from the semantic FX
            # schema.  Their exact finalized sites are the authority that
            # permits creating the corresponding record kwarg here.
            record.kwargs[site.field].CopyFrom(build_argument(value))
            return
        if site.field not in record.kwargs:
            raise ValueError(
                f"{site.label}: occurrence field is absent from the record "
                "skeleton"
            )
        argument = record.kwargs[site.field]
        if (
            len(occurrence) != 2
            or occurrence[0] != "tensor_list"
            or isinstance(occurrence[1], bool)
            or not isinstance(occurrence[1], int)
        ):
            raise ValueError(
                f"{site.label}: unsupported finalized argument occurrence"
            )
        index = occurrence[1]
        if argument.WhichOneof("arg_type") != "tensor_list":
            raise ValueError(
                f"{site.label}: occurrence does not name a tensor-list field"
            )
        if index < 0 or index >= len(argument.tensor_list.tensors):
            raise ValueError(
                f"{site.label}: tensor-list occurrence is out of range"
            )
        argument.tensor_list.tensors[index].CopyFrom(
            self._require_projected_tensor(binding, value)
        )

    def apply_record(self, owner: Node, members) -> dict:
        """Overlay exact member fields and return projections by exact site."""
        projected = {}
        for binding in self.finalized.bindings_for_owner(owner):
            member = binding.site.member
            if member is None:
                continue
            try:
                record = members[member]
            except KeyError:
                raise ValueError(
                    f"{binding.site.label}: finalized member was not emitted"
                ) from None
            value = self._project(binding)
            self._patch_member_argument(binding, record, value)
            projected[binding.site] = value
            self.consumed.append(binding.site)
        return projected

    def apply_envelope(self, owner: Node, output, outputs):
        """Overlay exact operation-return bindings on a built skeleton."""
        for binding in self.finalized.bindings_for_member(owner, None):
            site = binding.site
            if site.field in {"inputs", "parameters"}:
                continue
            value = self._require_projected_tensor(
                binding,
                self._project(binding),
            )
            if site.field == "output":
                if site.occurrence or output is None or outputs is not None:
                    raise ValueError(
                        f"{site.label}: binding does not name a single "
                        "operation return"
                    )
                output = value
            elif site.field == "outputs":
                occurrence = site.occurrence
                if (
                    len(occurrence) != 2
                    or occurrence[0] != "tensor_list"
                    or isinstance(occurrence[1], bool)
                    or not isinstance(occurrence[1], int)
                    or output is not None
                    or outputs is None
                ):
                    raise ValueError(
                        f"{site.label}: binding does not name a tensor-list "
                        "operation return"
                    )
                index = occurrence[1]
                if index < 0 or index >= len(outputs):
                    raise ValueError(
                        f"{site.label}: output occurrence is out of range"
                    )
                outputs[index] = value
            else:
                raise ValueError(
                    f"{site.label}: unknown operation-envelope field"
                )
            self.consumed.append(site)
        return output, outputs

    def finish(self) -> None:
        self.finalized.require_exact_site_coverage(self.consumed)


class _AlphaPayloadLedger:
    """Emit only outer-FX values and explicit plan-owned payload artifacts."""

    def __init__(self, plan, *, tensor_dir=None):
        self.plan = plan
        self.tensor_dir = tensor_dir
        self._logical_by_name = {}

    def emit(self, logical: ValueRef) -> None:
        if self.tensor_dir is None:
            return
        try:
            logical = self.plan.ref(logical)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{getattr(logical, 'name', logical)!s}: payload is absent "
                "from the finalized plan"
            ) from error
        name = _logical_abi_name(logical)
        previous = self._logical_by_name.get(name)
        if previous is not None:
            if previous != logical:
                raise ValueError(
                    f"{name}: two finalized logical tensors share a payload "
                    "identity"
                )
            return
        value = _logical_value(logical)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"{logical.name}: outer finalized tensor has no payload value"
            )
        _write_tensor_payload(
            value,
            os.path.join(self.tensor_dir, name + ".bin"),
        )
        self._logical_by_name[name] = logical

    def emit_node_outputs(self, node: Node) -> None:
        for logical in self.plan.output_refs(node):
            self.emit(logical)


#: What a record *writes*, by the member that terminates it.
#:
#: The record's own dtype, for the same reason its operands have one: the
#: graph value is whatever the traced Python produced -- a pool over a float
#: activation is float32 -- while the record describes what the endpoint
#: stores. Measured across a reference compile, every ``output_payload`` is
#: int8 except a per-vector stage's, which is int4.
_RECORD_OUTPUT_DTYPES: Dict[object, str] = {
    torch.ops.magnet_alpha.pvs_quantize.default: "int4",
    torch.ops.magnet_alpha.ppu_fixed_relu.default: "int8",
    torch.ops.magnet_alpha.nmp_fixed_requantize.default: "int8",
    torch.ops.magnet_alpha.channel_shuffle.default: "int8",
    torch.ops.magnet_alpha.channel_shuffle_copy.default: "int8",
    torch.ops.aten.max_pool2d.default: "int8",
    torch.ops.aten.upsample_nearest2d.default: "int8",
    torch.ops.aten.upsample_nearest2d.vec: "int8",
    torch.ops.magnet_alpha.upsample_nearest2d.default: "int8",
}

#: The three values a per-vector stage returns, and what a record calls them.
#:
#: ``(coarse scale, companion scale, payload)`` -- the order the operator
#: returns and the order a reference record lists them, named ``<record>_0``,
#: ``_1``, ``_2``. The names matter because the ``getitem`` records downstream
#: name them, so the numbering is the graph's edge, not a label.
_PVS_RESULT_DTYPES = ("float32", "uint8", "int4")


def _pvs_results(node, record) -> List[object]:
    """The three tensors a per-vector record returns, or ``None``.

    ``None`` for every other record, which returns one value.

    This exists because a record ending in ``pvs_quantize`` returns a *tuple*,
    and reading a shape off a tuple gives ``()``. The converter says so
    exactly -- ``DWC output: shape=(); expected positive rank-4 NCHW``, on 18
    of the 19 depthwise records, which are precisely the ones with a
    per-vector stage.
    """
    terminal = record.op_list[-1] if hasattr(record, "op_list") else record
    if terminal.target != "pvs_quantize":
        return None
    value = getattr(node, "value", None)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    return [
        build_tensor(
            f"{node.name}_{index}",
            tuple(getattr(component, "shape", ()) or ()),
            dtype,
        )
        for index, (component, dtype) in enumerate(
            zip(value, _PVS_RESULT_DTYPES)
        )
    ]


#: What can head a matrix record. Both drain through an epilogue that names
#: the array's operand under ``weight_payload``, and the converter requires it
#: on every one -- 66 of them, and refused as ``missing-serialized-tensor-role``
#: without it.
_MATRIX_HEADS = (
    torch.ops.magnet_alpha.conv2d_int8_accumulator.default,
    torch.ops.magnet_alpha.conv2d_vsq.default,
)


def _weight_payload(children):
    """The tensor a matrix record's array actually reads, or ``None``.

    The same node the head names as its ``weight``, restated under the role
    the converter looks the operand up by. It is *not* a second tensor: a
    reference compile names the weight's scratchpad backing here
    (``scratchpad_<weight>__primary``) because by then it has one, and until
    an allocator exists the honest name for the same bytes is the operand's
    own. Inventing the backing's name before the backing exists would put a
    node in the collateral that nothing wrote.

    ``None`` for a movement record, which has no array and no weight.
    """
    for child in children:
        target = getattr(child, "target", None)
        if target not in _MATRIX_HEADS:
            continue
        weight = get_arg_value(child, 1, "weight")
        if not isinstance(weight, Node):
            return None
        return _tensor_for(
            weight,
            dtype_override=_RECORD_OPERAND_DTYPES.get(target, {}).get(
                "weight"
            ),
        )
    return None


def _weight_valid_extents(children) -> Dict[str, int]:
    """How much of a padded weight the layer actually owns.

    The datapath holds a padded extent and the record says which part of it
    is the model's: ``weight_valid_input_channels`` and
    ``weight_valid_output_channels``. The reference's ``model_0_stem_1_conv``
    declares 9 valid input channels against a 16-wide resident weight, and 48
    valid output channels against 64 on each detection head.

    Read from the extent the padding pass *recorded* before it widened the
    operand, not from the operand's shape -- the shape is the padded one by
    construction, so reading it would always answer "all of it" and the field
    would be inert exactly where it matters.
    """
    for child in children:
        target = getattr(child, "target", None)
        if target not in _MATRIX_HEADS:
            continue
        weight = get_arg_value(child, 1, "weight")
        if not isinstance(weight, Node):
            return {}
        logical = get_hardware_logical_shape(weight)
        if not logical:
            value = getattr(weight, "value", None)
            logical = tuple(getattr(value, "shape", ()) or ())
        if len(logical) < 2:
            return {}
        return {
            "weight_valid_output_channels": int(logical[0]),
            "weight_valid_input_channels": int(logical[1]),
        }
    return {}


#: The movement vocabulary Alpha performs by addressing, and the name a
#: record carries for each.
#:
#: A ``nop`` record asserts two things at once -- the endpoint issues no
#: instruction, **and** the bytes do not move. Both have to be true, which is
#: why this is a declared list rather than "whatever ``record_target`` did not
#: claim". The difference is not pedantic. *Measured on this graph:* 33
#: ``aten.relu.default`` nodes survive with no record to perform them (their
#: producing record carries ``ppu_fixed_relu`` with ``relu=False``, against
#: the reference's 3 of 66), and 7 more nodes are a real requantization
#: between two different scales. A ``nop`` for any of those converts cleanly,
#: reports a connected program, and drops arithmetic the model specifies --
#: the exact failure this module's header calls "wrong in a way nothing
#: downstream notices".
#:
#: The strings are the reference's short names, not ``str(node.target)``.
#: They are an external ABI: ``convert.py``'s ``_nop_skip_reason`` reads
#: ``view`` *by name* to report it as ``shared_address_only``, so
#: ``aten.view.default`` would be filed as an ordinary address-only record
#: instead. ``getitem`` has no schema and no overload packet, and the
#: reference writes exactly ``str(operator.getitem)``.
_ADDRESS_ONLY_TARGETS: Dict[object, str] = {
    operator.getitem: str(operator.getitem),
    torch.ops.aten.cat.default: "cat",
    torch.ops.aten.chunk.default: "chunk",
    torch.ops.aten.contiguous.default: "contiguous",
    torch.ops.aten.permute.default: "permute",
    torch.ops.aten.reshape.default: "reshape",
    torch.ops.aten.select.int: "select",
    torch.ops.aten.slice.Tensor: "slice",
    torch.ops.aten.stack.default: "stack",
    torch.ops.aten.view.default: "view",
}


def address_only_target(node: Node) -> Optional[str]:
    """The ``nop`` name ``node`` would carry, or ``None``.

    ``None`` means "not known to be address-only", which is the honest answer
    for everything this table does not name -- including nodes that really are
    free but have not been measured to be. An unemitted node costs an island;
    a wrongly emitted one costs the arithmetic.
    """
    return _ADDRESS_ONLY_TARGETS.get(getattr(node, "target", None))


def _tensor_like(tensor, node_name: str, shape=None):
    """Restate one tensor under a different name.

    A ``nop`` may change *shape* -- a view, a slice and a concatenation all
    do, and the reference's own records confirm it: 12 of its 14 ``reshape``
    NOPs take rank 5 to rank 4. It may never change **dtype**, on any of the
    reference's 153. So the dtype is carried across rather than re-read from
    the traced value, which is a float for every quantized activation in this
    graph and would declare a storage the endpoint does not use.
    """
    return build_tensor(
        node_name,
        tuple(tensor.shape) if shape is None else shape,
        tensor.dtype,
    )


def _nop_operand(node, produced, emitted):
    """The tensor a movement node names for one input, resolved.

    Two resolutions, both measured against the reference:

    * A ``getitem`` off a record that returns a tuple names the **element**,
      ``<producer>_<index>``, not the record -- 102 of 102 of the reference's
      ``getitem`` NOPs do, with the index repeated in the positional argument.
      Naming the bare record instead names a tensor nothing produces, and the
      converter's adjacency is by name, so the island would not close.
    * The operand's dtype comes from the producing record's *emitted* output,
      never from the traced value. For a per-vector stage those differ on
      purpose: the graph traced ``float32`` for all three results and the
      record stores ``('float32', 'uint8', 'int4')``.
    """
    if not isinstance(node, Node):
        return None
    name = node.name
    if produced is not None:
        name = produced
    tensor = emitted.get(name)
    if tensor is not None:
        return _tensor_like(tensor, name)
    return _tensor_for(node)


def _nop_record_for(node, emitted, tuple_outputs):
    """One ``nop`` record and the tensors it returns, or ``None``.

    Returns ``(record, output, outputs)`` with exactly one of the two arms
    populated, matching ``build_operation``'s contract.
    """
    target = address_only_target(node)
    if target is None:
        return None

    is_getitem = node.target is operator.getitem
    operands: List[object] = []
    positional: List[object] = []
    index = None
    if is_getitem:
        # The only NOP target the reference gives positional arguments, and
        # the only one needing index resolution. Both are the same fact: the
        # index selects which of the producer's outputs is being named.
        source = node.args[0] if node.args else None
        index = node.args[1] if len(node.args) > 1 else None
        if not isinstance(source, Node) or not isinstance(index, int):
            return None
        element = tuple_outputs.get(source.name)
        resolved = (
            element[index]
            if element is not None and 0 <= index < len(element)
            else source.name
        )
        operand = _nop_operand(source, resolved, emitted)
        if operand is None:
            return None
        positional = [operand, index]
    else:
        # The reference writes ``self`` as ``input`` on every movement target
        # that has one -- reshape, view, chunk, select, slice -- and leaves
        # ``tensors`` alone on cat and stack. Measured off its own 153 NOPs.
        # The converter binds by keyword, so this is the operand's name, not
        # a label.
        names = [
            "input" if name == "self" else name
            for name in _operand_names(node.target)
        ]
        for position, argument in enumerate(node.args):
            key = (
                names[position]
                if position < len(names)
                else f"arg{position}"
            )
            if isinstance(argument, Node):
                operand = _nop_operand(argument, None, emitted)
                if operand is not None:
                    operands.append((key, operand))
            elif isinstance(argument, (list, tuple)) and any(
                isinstance(value, Node) for value in argument
            ):
                members = [
                    _nop_operand(value, None, emitted)
                    for value in argument
                    if isinstance(value, Node)
                ]
                if any(member is None for member in members):
                    return None
                operands.append((key, members))
            elif isinstance(argument, (int, float, bool, str)):
                operands.append((key, argument))
            elif isinstance(argument, (list, tuple)) and all(
                isinstance(value, (int, float, bool)) for value in argument
            ):
                operands.append((key, list(argument)))
        for key, argument in node.kwargs.items():
            key = "input" if key == "self" else key
            if isinstance(argument, Node):
                operand = _nop_operand(argument, None, emitted)
                if operand is not None:
                    operands.append((key, operand))
            elif isinstance(argument, (int, float, bool, str)):
                operands.append((key, argument))

    # The identity this record restates. A NOP has no dtype of its own; it
    # carries its input's, because that is the storage it is re-addressing.
    source_tensor = None
    for value in ([positional[0]] if positional else
                  [operand for _, operand in operands]):
        if isinstance(value, list):
            value = value[0] if value else None
        if value is not None and hasattr(value, "dtype"):
            source_tensor = value
            break
    if source_tensor is None:
        return None

    record = build_single_record(
        node.name,
        target,
        op="nop",
        args=positional,
        kwargs=dict(operands),
    )

    value = getattr(node, "value", None)
    if isinstance(value, (tuple, list)):
        outputs = [
            _tensor_like(
                source_tensor,
                f"{node.name}_{position}",
                tuple(getattr(component, "shape", ()) or ()),
            )
            for position, component in enumerate(value)
        ]
        return record, None, outputs
    return (
        record,
        _tensor_like(
            source_tensor,
            node.name,
            tuple(getattr(value, "shape", ()) or ()),
        ),
        None,
    )


def _record_for(node, name):
    """The ``OpOverload`` one call contributes, or ``None``.

    ``None`` for movement: a view, a concatenation Alpha performs by writing
    its branches where they belong. The reference emits those as ``nop``
    records, which is a different thing from omitting them -- they hold a
    place in the order -- but a nop carries no operand the endpoint reads, so
    nothing here has to invent one.
    """
    operands = {}
    names = _operand_names(node.target)
    dtypes = _RECORD_OPERAND_DTYPES.get(node.target, {})
    for index, argument in enumerate(node.args):
        # Positional on the node, keyword in the record. A record has no
        # positional operands at all -- the reference emits every one under a
        # key -- so the schema's name is not a convenience here, it is the
        # only thing that makes the operand findable.
        key = names[index] if index < len(names) else f"arg{index}"
        if isinstance(argument, Node):
            operands[key] = _tensor_for(
                argument, dtype_override=dtypes.get(key)
            )
        elif isinstance(argument, (int, float, bool, str)):
            operands[key] = argument
        elif isinstance(argument, (list, tuple)) and all(
            isinstance(v, (int, float, bool)) for v in argument
        ):
            operands[key] = list(argument)
    aliases = _RECORD_OPERAND_NAMES.get(node.target, {})
    for key, argument in node.kwargs.items():
        key = aliases.get(key, key)
        if isinstance(argument, Node):
            operands[key] = _tensor_for(
                argument, dtype_override=dtypes.get(key)
            )
        elif isinstance(argument, (int, float, bool, str)):
            operands[key] = argument
        elif isinstance(argument, (list, tuple)) and all(
            isinstance(v, (int, float, bool)) for v in argument
        ):
            operands[key] = list(argument)
    return build_single_record(node.name, name, kwargs=operands)


def _attach_record_schedule(record, owner: Node) -> None:
    """Project one already-selected schedule into Alpha's named ABI fields.

    This is intentionally a copy, not a derivation: loop mapping and factor
    selection belong to the provider, and the immutable named factors on its
    ``LocalScheduleResult`` are the only values serialized here.
    """
    result = owner.meta.get(LOCAL_SCHEDULE_META)
    if result is None:
        return
    if not result.named_factors:
        raise ValueError(f"{owner.name}: scheduled record has no named factors")
    for name, value in result.named_factors:
        attach_record_role(record, f"schedule_factor_{name}", value)


def _tiling_from_schedule(owner: Node, result) -> param_pb2.Tiling:
    """Project one frozen local mapping into Alpha's diagnostic tiling.

    Level zero is the implicit PE placement.  The endpoint consumes the
    target-named ``schedule_factor_*`` kwargs instead; this envelope preserves
    the remaining loop order/blocking and optional access counts exactly as
    the fork did, without deriving either decision during serialization.
    """
    mapping = result.mapping
    tiling = param_pb2.Tiling(name=owner.name)
    for level in range(1, mapping.num_levels):
        level_tiling = tiling.level_tilings.add()
        loop_order = 0
        while loop_order < len(mapping.loop_orders):
            match = next(
                (
                    loop
                    for loop in range(len(mapping.loop_orders))
                    if mapping.loop_orders[loop][level] == loop_order
                ),
                None,
            )
            if match is None:
                break
            bound = level_tiling.loop_bounds.add()
            bound.loop = match
            bound.bound = mapping.loop_blockings[match][level]
            loop_order += 1
        if level < len(result.access_counts):
            counts = result.access_counts[level]
            projected = tiling.level_access_counts.add()
            projected.input_access_count = counts.input
            projected.output_access_count = counts.output
            projected.weight_access_count = counts.weight
    return tiling


def _emit_movement_record(
    node, emitted, tuple_outputs, operations, unrepresented
) -> None:
    """Append the ``nop`` for one movement node, or count that there is none.

    The two outcomes are one line of collateral apart and opposite claims
    about the machine, so the second is tallied by target rather than passed
    over: ``gen_code`` names the total before it returns. A node nothing
    represents costs an execution island, and the converter reports that as a
    disconnected program -- which says nothing about which node it was.
    """
    built = _nop_record_for(node, emitted, tuple_outputs)
    if built is None:
        unrepresented[str(getattr(node, "target", node.op))] += 1
        return
    record, output, outputs = built
    operations.append(
        build_operation(record, output=output)
        if outputs is None
        else build_operation(record, outputs=outputs)
    )
    if outputs is None:
        emitted[output.node] = output
    else:
        tuple_outputs[node.name] = [tensor.node for tensor in outputs]
        for tensor in outputs:
            emitted[tensor.node] = tensor


def _finalized_record_target(prepared, node: Node) -> Optional[str]:
    """Return the strict record target, honoring a frozen egress boundary."""
    boundary = prepared.egress_for(node)
    if boundary is not None:
        if node is boundary.dequantize:
            return "dequantize"
        if node in boundary.host_layouts:
            if node.target is torch.ops.aten.permute.default:
                return "permute"
            if node.target is torch.ops.aten.contiguous.default:
                return "contiguous"
            raise ValueError(
                f"{node.name}: frozen Alpha egress has an unknown host "
                "layout target"
            )
        if node not in boundary.address_only_nodes:
            raise ValueError(
                f"{node.name}: frozen Alpha egress node has no record role"
            )
        return None
    return record_target(node)


def _projected_tensor_field(projected, field: str):
    matches = tuple(
        value
        for site, value in projected.items()
        if site.field == field and not site.occurrence
    )
    if len(matches) > 1:
        raise ValueError(f"finalized Alpha field {field!r} is ambiguous")
    if not matches:
        return None
    value = matches[0]
    if not isinstance(value, param_pb2.Tensor):
        raise ValueError(f"finalized Alpha field {field!r} is not a tensor")
    return value


def _emit_finalized_movement_record(
    node,
    emitted,
    tuple_outputs,
    operations,
    projector,
    payloads,
) -> None:
    """Emit one strict address-only record from its frozen exact bindings."""
    built = _nop_record_for(node, emitted, tuple_outputs)
    if built is None:
        raise ValueError(
            f"{node.name}: finalized Alpha operation is neither executable "
            "nor a proved address-only record"
        )
    record, output, outputs = built
    projector.apply_record(node, {node: record})
    output, outputs = projector.apply_envelope(node, output, outputs)
    operations.append(
        build_operation(record, output=output)
        if outputs is None
        else build_operation(record, outputs=outputs)
    )
    if outputs is None:
        emitted[output.node] = output
    else:
        tuple_outputs[node.name] = [tensor.node for tensor in outputs]
        for tensor in outputs:
            emitted[tensor.node] = tensor
    payloads.emit_node_outputs(node)


def serialize_model(model: GraphModule, *, tensor_dir=None):
    """Project one finalized Alpha graph into flat collateral.

    **What this emits and what it does not.** Structure and the semantic
    operands: one record per call, in order, with the shapes and dtypes the
    graph knows. Not placement -- no address, no memory instance, no bank.
    Those are the allocator's, there is no Alpha allocator yet, and a record
    with ``address: 0`` would not be an incomplete answer but a wrong one,
    since zero is a legal address. Absent is the honest encoding of unknown,
    and the converter refuses it loudly rather than launching against the
    base of the arena.

    So the collateral this produces is measurable -- the 144-record inventory
    reads it -- and not yet convertible. Those are different milestones and
    conflating them is how a green record count comes to stand for a working
    lane.

    Args:
        model: The transformed graph after shape propagation, schedule binding,
            and storage planning.
        tensor_dir: Where to write operand payloads, or ``None``.

    Returns:
        The ``param_pb2.Model`` the endpoint's converter reads.
    """
    if tensor_dir is not None:
        os.makedirs(tensor_dir, exist_ok=True)

    inputs, parameters, operations = [], [], []
    # What every record has already produced, by tensor name, and which
    # records returned a tuple. A movement record names its producer's
    # *emitted* operand -- its dtype is the storage the endpoint uses, which
    # for a per-vector stage is not the dtype the graph traced.
    emitted: Dict[str, object] = {}
    tuple_outputs: Dict[str, List[str]] = {}
    unrepresented: Dict[str, int] = collections.Counter()
    for node in model.graph.nodes:
        if node.op == "placeholder":
            tensor = _tensor_for(node)
            inputs.append(tensor)
            emitted[tensor.node] = tensor
            continue
        if node.op == "get_attr":
            # Only what the endpoint fetches from an address. A quantization
            # lookup table, a scalar scale and a non-tensor are *programmed
            # into* the instruction that reads them, so listing them as
            # parameters would describe storage nothing occupies -- and it
            # showed: before this filter the collateral declared 439
            # parameters against the reference's 121, the difference being
            # exactly the qmaps and the single-element scales.
            marked = any(
                node.meta.get(key) for key in NON_ALLOCATING_OPERAND_META
            )
            if require_allocation(node) and not marked:
                tensor = _tensor_for(node)
                parameters.append(tensor)
                emitted[tensor.node] = tensor
            continue
        if node.op != "call_function" and node.op != "call_module":
            continue

        submodule = _submodule(model, node)
        if submodule is not None and hasattr(submodule, "graph"):
            children = [
                child
                for child in submodule.graph.nodes
                if record_target(child) is not None
            ]
            members = [
                _record_for(child, record_target(child)) for child in children
            ]
            if not members:
                _emit_movement_record(
                    node, emitted, tuple_outputs, operations, unrepresented
                )
                continue
            record = build_fused_record(node.name, members)
        else:
            name = record_target(node)
            if name is None:
                _emit_movement_record(
                    node, emitted, tuple_outputs, operations, unrepresented
                )
                continue
            children = [node]
            record = _record_for(node, name)
        # Every role goes on before the operation is built: ``build_operation``
        # *copies* the record into the message, so a field attached afterwards
        # lands on a record nothing reads.
        terminal = children[-1] if children else node
        results = _pvs_results(node, record)
        if results is not None:
            # Three outputs, and the payload is the one the record writes.
            attach_record_output(record, results[2])
            attach_record_role(record, "output_vector_scale", results[1])
        else:
            results = None
            produced = _tensor_for(
                node,
                dtype_override=_RECORD_OUTPUT_DTYPES.get(
                    getattr(terminal, "target", None)
                ),
            )
            attach_record_output(record, produced)
        weight_payload = _weight_payload(children)
        if weight_payload is not None:
            attach_record_role(record, "weight_payload", weight_payload)
        for key, value in _weight_valid_extents(children).items():
            attach_record_role(record, key, value)
        _attach_record_schedule(record, node)
        schedule = node.meta.get(LOCAL_SCHEDULE_META)
        tiling = (
            _tiling_from_schedule(node, schedule)
            if schedule is not None
            else None
        )
        operations.append(
            build_operation(record, outputs=results, tiling=tiling)
            if results is not None
            else build_operation(record, output=produced, tiling=tiling)
        )
        if results is not None:
            tuple_outputs[node.name] = [tensor.node for tensor in results]
            for tensor in results:
                emitted[tensor.node] = tensor
        else:
            emitted[produced.node] = produced

    if unrepresented:
        # Named, not counted. Each one is a node the endpoint will not
        # perform: it costs an execution island, and the converter reports
        # that as a disconnected program rather than as this.
        LOGGER.warning(
            "%d graph nodes are neither a record nor known to be "
            "address-only, so the collateral does not represent them: %s",
            sum(unrepresented.values()),
            dict(sorted(unrepresented.items())),
        )

    return build_model(operations, inputs=inputs, parameters=parameters)


def serialize_finalized_model(finalized, *, tensor_dir=None):
    """Project a fully finalized Alpha end-to-end compilation.

    The legacy :func:`serialize_model` remains the per-operation structural
    path.  This entry point is deliberately stricter: record order comes from
    the prepared execution order, schedules come from frozen operation
    constraints, and every allocator-owned field must be supplied and
    consumed through an exact ``AlphaRecordBinding`` site.
    """
    projector = _AlphaFinalizedProjector(finalized, tensor_dir=tensor_dir)
    prepared = finalized.prepared
    model = prepared.graph.model
    payloads = _AlphaPayloadLedger(prepared.plan, tensor_dir=tensor_dir)
    if tensor_dir is not None:
        os.makedirs(tensor_dir, exist_ok=True)

    inputs, parameters, operations = [], [], []
    emitted: Dict[str, object] = {}
    tuple_outputs: Dict[str, List[str]] = {}

    for node in finalized.execution_order.nodes:
        if node.op == "placeholder":
            tensor = projector.model_tensor(node, "inputs")
            if tensor is not None:
                inputs.append(tensor)
                emitted[tensor.node] = tensor
            payloads.emit_node_outputs(node)
            continue
        if node.op == "get_attr":
            tensor = projector.model_tensor(node, "parameters")
            if tensor is not None:
                parameters.append(tensor)
                emitted[tensor.node] = tensor
            payloads.emit_node_outputs(node)
            continue
        if node.op == "output":
            continue
        if node.op not in {"call_function", "call_module", "call_method"}:
            raise ValueError(
                f"{node.name}: unsupported node kind {node.op!r} in the "
                "finalized Alpha execution order"
            )

        submodule = _submodule(model, node)
        if submodule is not None and hasattr(submodule, "graph"):
            children = [
                child
                for child in submodule.graph.nodes
                if record_target(child) is not None
            ]
            records = [
                _record_for(child, record_target(child))
                for child in children
            ]
            if not records:
                _emit_finalized_movement_record(
                    node,
                    emitted,
                    tuple_outputs,
                    operations,
                    projector,
                    payloads,
                )
                continue
            record = build_fused_record(node.name, records)
            member_records = dict(zip(children, record.op_list))
        else:
            name = _finalized_record_target(prepared, node)
            if name is None:
                _emit_finalized_movement_record(
                    node,
                    emitted,
                    tuple_outputs,
                    operations,
                    projector,
                    payloads,
                )
                continue
            children = [node]
            record = _record_for(node, name)
            member_records = {node: record}

        constraint = prepared.constraint_for(node)
        boundary = prepared.egress_for(node)
        is_boundary_record = (
            boundary is not None and node in boundary.boundary_nodes
        )
        if constraint is None and not is_boundary_record:
            raise ValueError(
                f"{node.name}: ordinary finalized Alpha record is neither "
                "executable nor a frozen terminal boundary"
            )

        terminal = children[-1] if children else node
        results = _pvs_results(node, record)
        if results is not None:
            attach_record_output(record, results[2])
            attach_record_role(record, "output_vector_scale", results[1])
            produced = None
        else:
            produced = _tensor_for(
                node,
                dtype_override=_RECORD_OUTPUT_DTYPES.get(
                    getattr(terminal, "target", None)
                ),
            )
            if not is_boundary_record:
                attach_record_output(record, produced)

        tiling = None
        if constraint is not None:
            try:
                schedule_member = member_records[constraint.terminal]
            except KeyError:
                raise ValueError(
                    f"{node.name}: frozen Alpha schedule terminal was not "
                    "emitted"
                ) from None
            if not constraint.schedule.named_factors:
                raise ValueError(
                    f"{node.name}: frozen Alpha schedule has no named factors"
                )
            for name, value in constraint.schedule.named_factors:
                schedule_member.kwargs[
                    f"schedule_factor_{name}"
                ].CopyFrom(build_argument(value))
            tiling = _tiling_from_schedule(node, constraint.schedule)

        projected = projector.apply_record(node, member_records)
        projected_output = _projected_tensor_field(
            projected,
            "output_payload",
        )
        if results is not None:
            if projected_output is not None:
                results[2] = projected_output
            projected_scale = _projected_tensor_field(
                projected,
                "output_vector_scale",
            )
            if projected_scale is not None:
                results[1] = projected_scale
        elif projected_output is not None:
            produced = projected_output

        produced, results = projector.apply_envelope(
            node,
            produced,
            results,
        )

        operations.append(
            build_operation(record, outputs=results, tiling=tiling)
            if results is not None
            else build_operation(record, output=produced, tiling=tiling)
        )
        if results is not None:
            tuple_outputs[node.name] = [tensor.node for tensor in results]
            for tensor in results:
                emitted[tensor.node] = tensor
        else:
            emitted[produced.node] = produced
        payloads.emit_node_outputs(node)

    projector.finish()
    return build_model(operations, inputs=inputs, parameters=parameters)
