# Check resource-only vector service with supplied lowered epilogues and explicit passes
from types import SimpleNamespace
import unittest
from voyager_compiler.codegen import param_pb2
from voyager_compiler.mapping.models.vector import Transfer, VectorHardware, VectorPass, evaluate_passes
from voyager_compiler.mapping.operations import evaluate_epilogue


# Check vector throughput, shared ports, repeated passes, and direct output
class VectorTimingTests(unittest.TestCase):
    # Use the same descriptor for either matrix backend
    def setUp(self):
        self.target = SimpleNamespace(n=64, accum_bits=24, oc_port_bits=512, hardware_options={}, vector_config=dict(lanes=64, output_fifo_packets=8))
        self.output = param_pb2.Tensor(dtype='bfloat16', shape=[1, 64])


    # Reusing the same engine accumulates work while independent ports overlap
    def test_explicit_passes(self):
        hardware = VectorHardware(64, 512)
        first = VectorPass('matrix')
        later = VectorPass('intermediate', transfers=(Transfer('output', 64, 16),))
        two = evaluate_passes(hardware, 64, 24, 16, (first, later))
        four = evaluate_passes(hardware, 64, 24, 16, (first, VectorPass('intermediate'), VectorPass('intermediate'), later))
        self.assertEqual(two.pass_cycles_per_vector, (1, 2))
        self.assertEqual(two.cycles_per_vector, 2)
        self.assertEqual(four.cycles_per_vector, 4)
        self.assertEqual(two.report()['passes'][1]['source'], 'intermediate')

    # Narrow lanes and a slower pipeline stage constrain throughput independently of output precision
    def test_lane_and_stage_rate(self):
        stage_pass = VectorPass('matrix', transfers=(Transfer('output', 64, 8),))
        service = evaluate_passes(VectorHardware(16, 512, (1, 2, 1, 1)), 64, 24, 8, (stage_pass,))
        self.assertEqual(service.cycles_per_vector, 8)
        self.assertEqual(service.port_cycles_per_vector, 1)

    # Distinct fetch ports overlap while two transfers sharing a port add demand
    def test_port_contention(self):
        service = evaluate_passes(VectorHardware(64, 512), 64, 24, 8,
            (VectorPass('matrix', transfers=(Transfer('fetch1', 64, 16), Transfer('fetch1', 64, 16), Transfer('output', 64, 8))),))
        self.assertEqual(service.cycles_per_vector, 4)


    # Direct accumulation output uses its native precision and bypasses vector arithmetic
    def test_direct_output(self):
        self.output.dtype = 'int24'
        service = evaluate_epilogue(self.target, [param_pb2.OpOverload(target='linear')], self.output)
        self.assertTrue(service.direct)
        self.assertEqual(service.cycles_per_vector, 3)
        self.assertNotIn('vector', dict(service.resource_cycles_per_vector))


if __name__ == '__main__':
    unittest.main()
