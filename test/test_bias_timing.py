# Check burst demand against small bias-feeder schedules shared by SA and CIM
import unittest
from voyager_compiler.mapping.models.bias import bias_completion
from test_cim_timing import small_target


# Separate first-reduction demand from layer-average transfer bandwidth
class BiasTimingTests(unittest.TestCase):
    # A three-cycle feeder inserts one bubble between two-position bias uses
    def test_first_reduction_burst(self):
        l2 = (("OX", 64), ("IC", 2), ("OC", 8))
        short = bias_completion((("OX", 2), ("OC", 1)), l2, 3)
        wider = bias_completion((("OX", 4), ("OC", 1)), (("OX", 32), ("IC", 2), ("OC", 8)), 3)
        self.assertEqual(short.stall_cycles, 8 * 63)
        self.assertEqual(short.producer_cycles, 2048 + 504)
        self.assertEqual(wider.stall_cycles, 0)
        self.assertEqual(wider.producer_cycles, 2048)

    # Moving a reduction changes available refill gaps without changing request count
    def test_reduction_gaps_and_bias_reuse(self):
        l1 = (("OX", 2), ("OC", 1))
        spaced = bias_completion(l1, (("IC", 2), ("OX", 64)), 3)
        self.assertEqual(spaced.stall_cycles, 0)
        reused = bias_completion((("OX", 128), ("OC", 8)), (("IC", 2),), 3)
        self.assertEqual(reused.stall_cycles, 0)
        self.assertEqual(reused.producer_cycles, 2048)


    # Both backend searches expose the same bias transfer service in their retained reports
    def test_small_search_both_backends(self):
        from test_mapping_streams import small_model, small_epilogues
        from test_mapping_boundary import small_sa
        from voyager_compiler.mapping.driver import generate_tilings
        model = small_model()
        bias = model.ops[0].fused_op.op_list[0].kwargs["bias"].tensor
        bias.shape.extend([16])
        bias.dtype = "int24"
        for target in (small_target(), small_sa()):
            _, report = generate_tilings(model, target, epilogues=small_epilogues())
            op = report["operations"][0]
            timing = op["evaluation"]["timing"]["readiness"] if target.backend == "cim" else op["bias_timing"]
            self.assertEqual(timing["bias_cycles_per_vector"], 3)


if __name__ == '__main__':
    unittest.main()
