# Exercise shared vector service and effective buffering through both mapping backends
from dataclasses import replace
import unittest
from voyager_compiler.codegen import param_pb2
from voyager_compiler.mapping.models.cim import evaluate
from voyager_compiler.mapping.schedule import Schedule, TemporalLevel
from voyager_compiler.mapping.models.cim_timing import TimingOptions
from voyager_compiler.mapping.workload import Workload
from voyager_compiler.mapping.driver import generate_tilings
from voyager_compiler.mapping.models.output import OutputOptions, output_timing, OutputPipeline
from voyager_compiler.mapping.target import SATarget
from voyager_compiler.mapping.operations import evaluate_operation_epilogue
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
        self.assertEqual(old.timing.readiness['output_stall_cycles'], 64 * (32 - 2 * 8))
        self.assertEqual(new.timing.readiness['output_stall_cycles'], 16 * (128 - 2 * 8))
        self.assertGreater(new.runtime_cycles, old.runtime_cycles)


    # Physical slots and delays use target geometry and do not double-count reduction buffers
    def test_profile_capacity_and_drain(self):
        target = small_target()
        profile = dict(elements_per_vector=target.n, vector_lanes=target.vector_config['lanes'],
                       accum_bits=target.accum_bits, output_bits=16, port_bits=target.oc_port_bits,
                       stages=[dict(name='fifo', elements='matrix_output', forward_cycles=1, feedback_cycles=1),
                               dict(name='extra_fifo', elements='vector_pipeline', forward_cycles=0),
                               dict(name='pipeline', elements=3 * target.n, forward_cycles=2, feedback_cycles=1)])
        options = OutputOptions(output_pipeline=profile)
        stream, report = output_timing(target, (('OX', 24),), 2, options=options)
        self.assertEqual(report['output_capacity_vectors'], 11)
        self.assertEqual(report['output_headroom_cycles'], 17)
        self.assertEqual((stream.producer_cycles, stream.consumer_cycles), (31, 51))
        extra = replace(target, output_storage=dict(target.output_storage, matrix_results=800))
        self.assertEqual(output_timing(extra, (('OX', 24),), 2, options=options)[0], stream)
        self.assertEqual(output_timing(target, (('OX', 24),), 2, direct=True, options=options),
                         output_timing(target, (('OX', 24),), 2, direct=True))

    # A new multi-pass epilogue keeps mapping through the unprofiled bandwidth path
    def test_profile_route_selection(self):
        target = small_target()
        profile = OutputPipeline(target.n, target.vector_config['lanes'], target.accum_bits, 16,
            target.oc_port_bits, (dict(name='fifo', elements='matrix_output', forward_cycles=1),
                                 dict(name='vector', elements='vector_pipeline', forward_cycles=0)))
        options = OutputOptions(profile)
        service = evaluate_operation_epilogue(target, small_model().ops[0], small_epilogues()['dense'])
        self.assertIs(options.for_vector(target, service), options)
        with self.assertRaises(ValueError):
            options.for_vector(replace(target, oc_port_bits=2 * target.oc_port_bits), service)
        for other in (replace(service, direct=True), replace(service, passes=service.passes * 2),
                      replace(service, output_element_bits=8)):
            self.assertIsNone(options.for_vector(target, other).output_pipeline)
        sa = SATarget(k=8, n=8, input_buffer_words=256, weight_buffer_words=256, accum_buffer_words=32,
                      double_buffered_accum=False, ic_port_bits=64, oc_port_bits=64, input_bits=8,
                      weight_bits=8, accum_bits=24, datatype='INT8', vector_config=dict(lanes=8),
                      output_storage=target.output_storage)
        for hardware in (target, sa):
            _, report = generate_tilings(small_model(), hardware, epilogues=small_epilogues(),
                                         timing_options=dict(output_pipeline=profile))
            layer = report['operations'][0]
            readiness = (layer['evaluation']['timing']['readiness'] if hardware.backend == 'cim'
                         else layer['output_timing'])
            self.assertTrue(readiness['output_pipeline_profiled'])


if __name__ == '__main__':
    unittest.main()
