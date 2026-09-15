import unittest
from voyager_compiler.codegen import param_pb2


# Supply a tiny native matrix operation with one dequantization pass
def small_model():
    model = param_pb2.Model()
    operation = model.ops.add()
    operation.fused_op.name = 'dense'
    matrix = operation.fused_op.op_list.add(name='linear', target='linear')
    matrix.kwargs['input'].tensor.shape.extend([1, 16, 16])
    matrix.kwargs['input'].tensor.dtype = 'int8'
    matrix.kwargs['weight'].tensor.shape.extend([16, 16])
    matrix.kwargs['weight'].tensor.dtype = 'int8'
    dequantize = operation.fused_op.op_list.add(name='dequantize', target='dequantize')
    dequantize.kwargs['scale'].tensor.shape.extend([1])
    dequantize.kwargs['scale'].tensor.dtype = 'bfloat16'
    operation.output.shape.extend([1, 16, 16])
    operation.output.dtype = 'bfloat16'
    return model


# Supply the command generator's resource description for the tiny fused fixture
def small_epilogues():
    return {"dense": dict(direct=False, passes=[dict(source="matrix", stages=["", "", "", ""],
        dequantize=True, modes=[], transfers=[dict(resource="output", dtype="bfloat16")])])}


if __name__ == '__main__':
    unittest.main()
