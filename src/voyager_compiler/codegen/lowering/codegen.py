import operator
import os
import logging
import torch

import voyager_compiler.codegen.param_pb2 as pb

from .ir import Module, Operation, FusedOp, Loops, Value, IRNode, IndexValue
from ..shape_prop import ShapeProp
from ..mapping_utils import save_tensor, is_nop, is_addressing_op


logger = logging.getLogger(__name__)


MEM_SPACE_TO_INDEX = {
    "DRAM": 0,
    "Scratchpad": 1,
}


def set_tensor_field(field, value, output_dir=None, index=None):
    if isinstance(value, IndexValue):
        field.node = value.name
        field.shape.append(1)
        field.dtype = "int32"
        return

    node = value.producer_op.origin_node

    field.node = node.name
    field.shape.extend(value.shape or [1])

    if isinstance(value.dtype, torch.dtype):
        field.dtype = str(value.dtype).split(".")[1]
    else:
        field.dtype = value.dtype

    field.memory.partition = MEM_SPACE_TO_INDEX[value.space]
    field.memory.address = int(value.address)

    if output_dir is not None:
        tensor = node.value[index] if index is not None else node.value
        save_tensor(tensor, os.path.join(output_dir, f"{field.node}.bin"))


def set_tensor_list_field(field, values, output_dir):
    for i, t in enumerate(values):
        tensor = pb.Tensor()
        set_tensor_field(tensor, t, output_dir, index=i)
        field.tensors.append(tensor)


def convert_arg(value, output_dir=None) -> pb.Argument:
    """
    Converts an argument (which could be a Tensor, list, int, float, etc.)
    into an Argument protobuf.
    """
    arg = pb.Argument()

    if isinstance(value, Value):
        set_tensor_field(arg.tensor, value, output_dir)
    elif isinstance(value, bool):
        arg.bool_value = value
    elif isinstance(value, int):
        arg.int_value = value
    elif isinstance(value, float):
        arg.float_value = value
    elif isinstance(value, str):
        arg.str_value = value
    elif isinstance(value, (
        torch.dtype, torch.layout, torch.device, torch.memory_format
    )):
        arg.str_value = str(value).split(".")[-1]
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, Value) or x is None for x in value):
            set_tensor_list_field(arg.tensor_list, value, output_dir)
        elif all(isinstance(x, bool) for x in value):
            arg.bool_list.values.extend(value)
        elif all(isinstance(x, int) for x in value):
            arg.int_list.values.extend(value)
        elif all(isinstance(x, (int, float, bool)) for x in value):
            arg.scalar_list.values.extend(value)
        else:
            raise TypeError(f"Unsupported list value: {value}")
    else:
        raise TypeError(f"Unsupported arg type: {type(value)}")

    return arg


def map_node(ir_node: IRNode, output_dir=None) -> pb.OpOverload:
    """
    Converts a torch.fx.Node into an OpOverload protobuf message.
    """
    node = ir_node.origin_node
    if hasattr(node.target, "_schema"):
        target = node.target._schema.name.split('::')[1]
    else:
        target = str(node.target)

    op_overload = pb.OpOverload(
        name=node.name,
        op=node.op,
        target=target,
    )

    if is_nop(node) or is_addressing_op(node) or node.target == operator.getitem:
        op_overload.op = "nop"

    if node.target == torch.ops.aten.pad.default:
        op_overload.op = "cpu"

    ssa_value_map = {inp.name: inp for inp in ir_node.inputs}

    # Convert keyword arguments
    for key, value in ir_node.kwargs.items():
        if not "qmap" in key and value is not None:
            if isinstance(value, str):
                value = ssa_value_map.get(value, value)
            elif isinstance(value, (list, tuple)):
                value = [
                    ssa_value_map.get(v, v) if isinstance(v, str) else v
                    for v in value
                ]
            op_overload.kwargs[key].CopyFrom(convert_arg(value, output_dir))

    return op_overload


def map_operation(ir_node: Loops, output_dir=None):
    body = []
    if isinstance(ir_node, Operation):
        op = pb.Operation(op=map_node(ir_node))
        set_tensor_list_field(op.outputs, ir_node.outputs, output_dir)
        return op
    elif isinstance(ir_node, FusedOp):
        node = ir_node.origin_node
        op = pb.Operation()
        op.fused_op.name = node.name

        for fused_op in ir_node.ops:
            op.fused_op.op_list.append(map_node(fused_op))

        set_tensor_list_field(op.outputs, ir_node.outputs, output_dir)
        return op
    elif isinstance(ir_node, Loops):
        for stmt in ir_node.body:
            body.append(map_operation(stmt, output_dir=output_dir))
        loop = pb.Loop(
            node=ir_node.index.name,
            start=ir_node.start,
            end=ir_node.end,
            step=1,
            body=body,
        )
        op = pb.Operation(loop=loop)
        return op
    else:
        print(f"Skipping non-operation stmt in loops: {ir_node}")
        return None


def generate_proto(module: Module, model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    ShapeProp(model).propagate(*args)
    model_params = pb.Model()

    for stmt in module.body:
        node = stmt.origin_node
        if node and node.op == 'placeholder':
            tensor = pb.Tensor()
            set_tensor_field(tensor, node, output_dir)
            model_params.inputs.append(tensor)
        elif node and node.op == 'get_attr':
            tensor = pb.Tensor()
            set_tensor_field(tensor, node, output_dir)
            if "memory" in node.meta:
                model_params.parameters.append(tensor)
        else:
            op = map_operation(stmt, output_dir=output_dir)
            model_params.ops.append(op)

    return model_params
