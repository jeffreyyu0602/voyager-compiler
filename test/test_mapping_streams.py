# Exercise shared vector service and effective buffering through both mapping backends
from dataclasses import replace
import unittest
from voyager_compiler.codegen import param_pb2
from voyager_compiler.mapping.models.cim import evaluate
from voyager_compiler.mapping.schedule import Schedule, TemporalLevel
from voyager_compiler.mapping.models.cim_timing import TimingOptions
from voyager_compiler.mapping.workload import Workload
from test_cim_timing import small_target


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


# Check mapping rank and output timing with declared buffer capacities
class MappingStreamTests(unittest.TestCase):
    # A fixed declared capacity penalizes long completion bursts without fitting RTL cycles
    def test_historical_mapping_ranking(self):
        target = replace(small_target(), ch_in=64, tile_output_axis_elements=8, b_sets=18,
                         input_buffer_words=1024, accum_buffer_words=1024, ic_port_bits=512, oc_port_bits=512,
                         vector_config=dict(lanes=64), output_storage=dict(matrix_results=512, accumulation_metadata=128,
                                             accumulation_writeback=64, matrix_output=512, vector_pipeline=0))
        options = TimingOptions(output_cycles_per_vector=2)
        workload = Workload(1024, 1, 128, 128, output_to_memory=False)
        old = evaluate(target, Schedule(TemporalLevel.make(OX=32, IC=2), TemporalLevel.make(OX=32, OC=2)), workload, options=options)
        new = evaluate(target, Schedule(TemporalLevel.make(OX=64, OC=2), TemporalLevel.make(order=('IC', 'OX'), IC=2, OX=16)), workload, options=options)
        self.assertTrue(old.legal, old.reasons)
        self.assertTrue(new.legal, new.reasons)
        self.assertEqual(old.timing.readiness['output_stall_cycles'], 0)
        self.assertEqual(new.timing.readiness['output_stall_cycles'], 16 * (128 - 2 * 19))
        self.assertGreater(new.runtime_cycles, old.runtime_cycles)


if __name__ == '__main__':
    unittest.main()
