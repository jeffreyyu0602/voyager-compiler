# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Build Alpha's flat ``param.proto`` representation for ``model.txt``.

This is the writer. Nothing else in the tree writes ``param_pb2``: the shared
path emits ``voyager_ir.proto`` from ``bufferize/emit.py``, and Alpha's form is
a different one -- flat records, addresses resolved, no loop nests.

Every function here takes plain Python and returns a message, so the writer can
be checked by parsing ``model.txt`` and rebuilding records from their fields.
That check needs no graph, schedule, allocator, or external runtime.

Two serialization conventions are load-bearing:

*A record's terminal op owns the record-wide fields.* Schedule factors and
placement sit on the *last* inner op of a fused record, not the first -- and
``bias_payload`` sits on the **middle** op of a three-op matrix record.

*Absence is meaningful.* ``is_none`` is a real field, and a ``conv2d_vsq`` that
carries a ``bias`` key at all is rejected outright.
"""

from typing import Any, Iterable, Mapping, Optional, Sequence

from voyager_compiler.codegen import param_pb2

__all__ = [
    "attach_record_output",
    "attach_record_role",
    "build_argument",
    "build_fused_record",
    "build_memory",
    "build_model",
    "build_operation",
    "build_single_record",
    "build_tensor",
]


def build_memory(
    partition: Optional[int] = None,
    address: Optional[int] = None,
    instance: Optional[str] = None,
) -> param_pb2.Memory:
    """One placement. Every field is optional because a tensor may be placed
    in one memory and not the other."""
    memory = param_pb2.Memory()
    if partition is not None:
        memory.partition = int(partition)
    if address is not None:
        memory.address = int(address)
    if instance is not None:
        memory.instance = str(instance)
    return memory


def build_tensor(
    node: str,
    shape: Sequence[int] = (),
    dtype: str = "",
    *,
    memory: Optional[param_pb2.Memory] = None,
    scratchpad: Optional[param_pb2.Memory] = None,
    tiled_shape: Sequence[int] = (),
    tile_strides: Sequence[int] = (),
    is_none: bool = False,
) -> param_pb2.Tensor:
    """One operand.

    ``node`` is the FX value's name and is what ties a record to a
    ``tensor_files`` payload, so it is required even when nothing else is.
    """
    tensor = param_pb2.Tensor(node=str(node), dtype=str(dtype))
    tensor.shape.extend(int(v) for v in shape)
    tensor.tiled_shape.extend(int(v) for v in tiled_shape)
    tensor.tile_strides.extend(int(v) for v in tile_strides)
    if memory is not None:
        tensor.memory.CopyFrom(memory)
    if scratchpad is not None:
        tensor.scratchpad.CopyFrom(scratchpad)
    if is_none:
        tensor.is_none = True
    return tensor


def build_argument(value: Any) -> param_pb2.Argument:
    """One keyword operand, in the oneof arm its Python type implies.

    The order of the checks is the contract, not a convenience: ``bool`` is a
    subclass of ``int`` in Python, so testing ``int`` first would serialize
    every ``relu=True`` as ``int_value: 1``. The converter reads
    ``bool_value``, so that record would convert and mean something else.
    """
    argument = param_pb2.Argument()
    if isinstance(value, param_pb2.Tensor):
        argument.tensor.CopyFrom(value)
    elif isinstance(value, param_pb2.Argument):
        argument.CopyFrom(value)
    elif isinstance(value, bool):
        argument.bool_value = value
    elif isinstance(value, int):
        argument.int_value = value
    elif isinstance(value, float):
        argument.float_value = value
    elif isinstance(value, str):
        argument.str_value = value
    elif isinstance(value, (list, tuple)):
        argument.CopyFrom(_build_sequence(value))
    else:
        raise TypeError(
            f"no param.proto argument arm for {type(value).__name__}"
        )
    return argument


def _build_sequence(values: Sequence[Any]) -> param_pb2.Argument:
    """Which list arm a sequence takes, decided by its contents.

    An empty sequence is an ``IntList``: the schema has no empty arm, and
    every empty operand the reference emits is a shape or a stride.
    """
    argument = param_pb2.Argument()
    if values and all(isinstance(v, param_pb2.Tensor) for v in values):
        argument.tensor_list.tensors.extend(values)
    elif values and all(isinstance(v, bool) for v in values):
        argument.bool_list.values.extend(values)
    elif values and any(isinstance(v, float) for v in values):
        argument.scalar_list.values.extend(float(v) for v in values)
    else:
        argument.int_list.values.extend(int(v) for v in values)
    return argument


def build_single_record(
    name: str,
    target: str,
    *,
    op: str = "call_function",
    args: Iterable[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> param_pb2.OpOverload:
    """One operation call."""
    record = param_pb2.OpOverload(name=str(name), op=str(op), target=str(target))
    for value in args:
        record.args.append(build_argument(value))
    for key, value in (kwargs or {}).items():
        record.kwargs[key].CopyFrom(build_argument(value))
    return record


def build_fused_record(
    name: str, ops: Sequence[param_pb2.OpOverload]
) -> param_pb2.OpOverloadList:
    """A fused record: one name, an ordered list of inner calls.

    The order is load-bearing twice over -- the converter dispatches on the
    target *sequence*, and the record's own fields live on particular members
    of it rather than on the record.
    """
    fused = param_pb2.OpOverloadList(name=str(name))
    fused.op_list.extend(ops)
    return fused


def attach_record_output(record, tensor) -> None:
    """Name where a record writes, on the member that owns record-wide fields.

    ``output_payload`` is a *role*: the converter looks for a kwarg of that
    name on any inner operation of the record and checks that the tensor it
    names is one of the record's own outputs
    (``serialized_tensor_role(..., output_only=True)``). A record without one
    is refused as ``missing-serialized-tensor-role`` and emits no layer -- 59
    of them on this workload, every movement record.

    It carries **no address here**. The role and the placement are two
    separate things the converter checks separately, which is what lets the
    role be stated now and the address when there is an allocator to state
    it.

    The *terminal* member, following the same rule as the schedule factors and
    the placement: a record's fields sit on its last inner op, not its first.
    That convention is the converter's and is not symmetric -- see this
    module's own header on ``bias_payload``, which sits on the middle op of a
    three-op matrix record and silently drops thirty payload files when it
    does not.

    Args:
        record: An ``OpOverload`` or ``OpOverloadList``.
        tensor: The record's output, as ``build_tensor`` made it.
    """
    attach_record_role(record, "output_payload", tensor)


def attach_record_role(record, role: str, tensor) -> None:
    """Put one serialized role on the member that owns record-wide fields.

    The converter resolves a role by scanning *every* inner operation for the
    key and requiring exactly one, so which member carries it is not forced by
    the lookup -- it is forced by consistency with the fields that *are*
    positional, and by the reference, which puts them on the last.
    """
    if isinstance(record, param_pb2.OpOverloadList):
        if not record.op_list:
            raise ValueError(f"{record.name}: a record has no members")
        terminal = record.op_list[-1]
    else:
        terminal = record
    terminal.kwargs[role].CopyFrom(build_argument(tensor))


def build_operation(
    record,
    *,
    output=None,
    outputs: Optional[Sequence[param_pb2.Tensor]] = None,
    tiling: Optional[param_pb2.Tiling] = None,
) -> param_pb2.Operation:
    """One entry of the model: a record plus what it returns."""
    operation = param_pb2.Operation()
    if isinstance(record, param_pb2.OpOverloadList):
        operation.fused_op.CopyFrom(record)
    elif isinstance(record, param_pb2.OpOverload):
        operation.op.CopyFrom(record)
    elif isinstance(record, param_pb2.Loop):
        operation.loop.CopyFrom(record)
    else:
        raise TypeError(f"not a record: {type(record).__name__}")
    if output is not None and outputs is not None:
        raise ValueError(
            f"{getattr(record, 'name', '?')}: an operation returns one tensor "
            f"or a list, not both"
        )
    if output is not None:
        operation.output.CopyFrom(output)
    if outputs is not None:
        operation.outputs.tensors.extend(outputs)
    if tiling is not None:
        operation.tiling.CopyFrom(tiling)
    return operation


def build_model(
    operations: Iterable[param_pb2.Operation],
    *,
    inputs: Iterable[param_pb2.Tensor] = (),
    parameters: Iterable[param_pb2.Tensor] = (),
) -> param_pb2.Model:
    """The whole collateral.

    The field is ``ops``, not ``operations`` -- worth stating because the
    latter is what the name of this argument suggests, and protobuf would
    accept neither silently.
    """
    model = param_pb2.Model()
    model.inputs.extend(inputs)
    model.parameters.extend(parameters)
    model.ops.extend(operations)
    return model
