# Check direct SRAM feedback with small 8-by-8 mappings
from dataclasses import replace
import unittest

from voyager_compiler.mapping.models.cim import evaluate
from voyager_compiler.mapping.target import CIMTarget
from voyager_compiler.mapping.schedule import Schedule, TemporalLevel
from voyager_compiler.mapping.models.cim_timing import TimingOptions
from voyager_compiler.mapping.workload import Workload


# Use native INT8 lanes and enough result slots for one result per cycle
def small_target(**changes):
    target = CIMTarget(
        datatype="INT8", input_bits=8, weight_bits=8, accum_bits=24, ch_in=8, ch_out=8, b_sets=8,
        base_a_width=8, base_b_width=8, base_c_width=24,
        write_ch_in=1, mac_latency=3, mode=0,
        tile_input_axis_elements=1, tile_output_axis_elements=1,
        input_axis_tiles=1, output_axis_tiles=1,
        a_port_tiles=1, b_port_tiles=1, c_port_tiles=1, c_beat_layout=1,
        result_slots_per_output_lane=8, local_accum_contexts=4,
        input_buffer_words=256, accum_buffer_words=32,
        double_buffered_accum=False, ic_port_bits=64, oc_port_bits=64,
        accumulation_policy="loop-lifetime-prefix")
    return replace(target, **changes)


# Check feedback throughput independently of SRAM access energy
class AccumulationTimingTests(unittest.TestCase):
    # Sixteen live outputs with eight terms issue 84 spill reads without dependency bubbles
    def test_spills_overlap_accumulation(self):
        workload = Workload(16, 1, 64, 8, output_to_memory=False)
        schedule = Schedule(TemporalLevel.make(OX=16, IC=8))
        spilled = evaluate(small_target(), schedule, workload)
        resident = evaluate(small_target(local_accum_contexts=16), schedule, workload)
        for result in (spilled, resident):
            self.assertTrue(result.legal, result.reasons)
            self.assertEqual(result.timing.resource_cycles["accumulation"], 128)
        self.assertEqual(spilled.traffic.buffer_accum_reads, 84)
        self.assertEqual(spilled.traffic.buffer_accum_intermediate_writes, 84)
        self.assertEqual(resident.traffic.buffer_accum_reads, 0)
        self.assertEqual(spilled.runtime_cycles, resident.runtime_cycles)

    # Keep genuine consumer and result-slot limits after removing dependency waits
    def test_backpressure_limits_remain(self):
        workload = Workload(16, 1, 64, 8, output_to_memory=False)
        schedule = Schedule(TemporalLevel.make(OX=16, IC=8))
        baseline = evaluate(small_target(), schedule, workload)
        slow_output = evaluate(small_target(), schedule, workload,
                               options=TimingOptions(output_cycles_per_vector=16))
        few_slots = evaluate(small_target(result_slots_per_output_lane=2), schedule, workload)
        self.assertGreater(slow_output.runtime_cycles, baseline.runtime_cycles)
        self.assertGreater(few_slots.runtime_cycles, baseline.runtime_cycles)
        self.assertEqual(few_slots.timing.resource_cycles["result_slots"], 512)


if __name__ == "__main__":
    unittest.main()
