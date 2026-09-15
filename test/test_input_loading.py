# Check shared input traffic against explicit small controller traversals
from dataclasses import replace
import unittest
from voyager_compiler.codegen import param_pb2
from test_mapping_boundary import prepare
from voyager_compiler.mapping.models.output import OutputOptions
from voyager_compiler.mapping.schedule import Schedule, TemporalLevel
from test_mapping_boundary import small_sa


# Make a 4x4 convolution whose 2x2 output tiles each store a 4x4 input halo
def convolution(dtype="int8"):
    model = param_pb2.Model()
    operation = model.ops.add()
    operation.op.name, operation.op.target = "conv", "conv2d"
    operation.op.kwargs["input"].tensor.shape.extend([1, 4, 4, 16])
    operation.op.kwargs["input"].tensor.dtype = dtype
    operation.op.kwargs["weight"].tensor.shape.extend([3, 3, 16, 8])
    operation.op.kwargs["weight"].tensor.dtype = "int8"
    operation.op.kwargs["padding"].int_list.values.extend([1, 1])
    operation.output.shape.extend([1, 4, 4, 8])
    operation.output.dtype = "int24"
    return operation


# Supply temporal factors without invoking a network search
def point():
    l1 = dict(OX=2, OY=2, IC=2, OC=1, FX=3, FY=3)
    l2 = dict(OX=2, OY=2, IC=1, OC=1, FX=1, FY=1)
    return Schedule(TemporalLevel.make(**l1), TemporalLevel.make(**l2))


# Check input traffic and bank capacity with padding and packed elements
class InputLoadingTests(unittest.TestCase):
    # Enumerate only a tiny four-bank fixture to independently count reads and writes
    def test_padding_and_precision(self):
        requested, writes = 0, 0
        for oy in (0, 2):
            for ox in (0, 2):
                for y in range(oy - 1, oy + 3):
                    for x in range(ox - 1, ox + 3):
                        writes += 2
                        requested += 2 * (0 <= x < 4 and 0 <= y < 4)
        for dtype, requests, fill, service in (("int4", requested // 2, 33, 32), ("int8", requested, 33, 32), ("int16", requested, 37, 36)):
            inputs = prepare(small_sa(), convolution(dtype), OutputOptions())
            summary, reasons = inputs.candidate_evaluator.model.input_loading(point())
            self.assertFalse(reasons)
            self.assertEqual((summary.requests, summary.writes), (requests, writes))
            self.assertEqual((summary.first_fill_cycles, summary.total_fill_cycles), (fill, 4 * service))

    # A tile's halo needs 32 bank words even though its output-based estimate is only eight
    def test_halo_capacity(self):
        inputs = prepare(replace(small_sa(), input_buffer_words=16), convolution(), OutputOptions())
        summary, reasons = inputs.candidate_evaluator.model.input_loading(point())
        self.assertIsNone(summary)
        self.assertIn("input tile including halo and padding exceeds one input-buffer bank", reasons)
