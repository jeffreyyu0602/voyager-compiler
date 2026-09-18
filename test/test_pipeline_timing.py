# Check coupled timing with small explicit schedules and very large repeated bursts
import unittest
from voyager_compiler.mapping.timing.overlap import BufferSlots, OverlapTiming, MAX_BURST_STEPS


# Verify overlap independently of the mapping search and hardware signal names
class PipelineTimingTests(unittest.TestCase):
    # A slower consumer hides refill turnaround instead of adding both independent waits
    def test_output_backpressure_hides_turnaround(self):
        pipeline = OverlapTiming((BufferSlots(2, 4, ready_delay=1, release_delay=1),), 2, 1)

        # Produce four groups per resident-set use
        def use():
            slot = pipeline.acquire(0)
            pipeline.produce(4, 4)
            pipeline.release(0, slot)

        count = 10**12
        pipeline.repeat(count, use)
        self.assertEqual(pipeline.producer_at, 5 + 8 * count - 2)
        self.assertEqual(pipeline.consumer_at, 5 + 8 * count)
        self.assertEqual(pipeline.waits[0], 0)
        self.assertLess(pipeline.steps, 10)

    # Repeated summaries must agree with a short direct execution, including every wait counter
    def test_repeat_matches_small_expansion(self):
        results = []
        for accelerated in (False, True):
            pipeline = OverlapTiming((BufferSlots(2, 7, ready_delay=2, release_delay=1),
                                      BufferSlots(2, 9)), 3, 2)

            # Preserve a reduction gap followed by a completed-output burst
            def body():
                bank = pipeline.acquire(1)
                for final in (False, True):
                    slot = pipeline.acquire(0)
                    pipeline.produce(4, 4 if final else 0, requests=0 if final else 1, request_cycles=3)
                    pipeline.release(0, slot)
                pipeline.release(1, bank)

            if accelerated:
                pipeline.repeat(40, body)
            else:
                for _ in range(40):
                    body()
            results.append((pipeline.producer_at, pipeline.consumer_at, pipeline.waits))
        self.assertEqual(*results)

    # Separate resident phases each need a short warm-up before their steady state repeats
    def test_nested_resident_phases(self):
        from voyager_compiler.mapping.models.cim_timing import _completion
        timing = _completion(((2, True, False, False),),
                             ((128, False, True, False), (4, True, False, False),
                              (2, False, False, False)),
                             8, 1, 1, 2, 19, 8, 64, 3, 3, 1, 16, 17, 3)
        self.assertIsNotNone(timing)
        self.assertEqual(timing[1] + 3, 16519)
        self.assertLess(timing[3], MAX_BURST_STEPS)


if __name__ == '__main__':
    unittest.main()
