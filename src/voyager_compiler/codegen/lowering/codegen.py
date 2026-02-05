import os

import voyager_compiler.codegen.param_pb2 as pb

from .ir import FunctionIR, Operation, FusedOp, Loops
from ..mapping_utils import (
    map_node,
    set_output_field,
    set_tensor_field,
)
from ..shape_prop import ShapeProp


def map_operation(op: Operation, model, args, output_dir=None):
    assert isinstance(op, Operation), f"Expected Operation, got {type(op)}"

    node = op.origin_node
    if node is None:
        print(f"Skipping operation with no origin node: {op}")
        return None

    mapped_op = pb.Operation()
    mapped_op.op.CopyFrom(map_node(node))
    set_output_field(mapped_op, node, output_dir)

    return mapped_op


def map_nested_loops(op: Loops):
    assert isinstance(op, Loops), f"Expected Loops, got {type(op)}"

    for stmt in op.body:
        if isinstance(stmt, Operation):
            pass
        elif isinstance(stmt, Loops):
            map_nested_loops(stmt)
        else:
            print(f"Skipping non-operation stmt in loops: {stmt}")
            continue


def map_function_ir(func: FunctionIR, model, args, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    ShapeProp(model).propagate(*args)
    model_params = pb.Model()

    for stmt in func.body:
        node = stmt.origin_node

        if node is None:
            print(f"Skipping stmt with no origin node: {stmt}")
            continue

        if node.op == 'placeholder':
            tensor = pb.Tensor()
            set_tensor_field(tensor, node, output_dir)
            model_params.inputs.append(tensor)
        elif node.op == 'get_attr':
            tensor = pb.Tensor()
            set_tensor_field(tensor, node, output_dir)
            if "memory" in node.meta:
                model_params.parameters.append(tensor)
        elif node.op == 'call_function':
            op = pb.Operation()
            op.op.CopyFrom(map_node(node))
            set_output_field(op, node, output_dir)
            model_params.ops.append(op)
        elif isinstance(stmt, FusedOp):
            node = stmt.origin_node

            op = pb.Operation()
            op.fused_op.name = node.name

            for fused_op in stmt.ops:
                op.fused_op.op_list.append(map_node(fused_op.origin_node))

            set_output_field(op, node, output_dir)
            model_params.ops.append(op)
        elif isinstance(stmt, Loops):
            pass
        else:
            print(f"Skipping non-operation stmt: {stmt}")
            continue

    return model_params
