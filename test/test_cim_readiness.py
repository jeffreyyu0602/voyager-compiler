# Check collapsed resident timing against small event-by-event reference schedules
import unittest
from voyager_compiler.mapping.timing.buffers import buffer_completion


# Schedule a tiny fixed sequence with explicit load and replay events for reference only
def reference(sets, replays, groups, capacity, load, delay, compute, input_ready):
    free = [0] * capacity
    producer, consumer, slot = 0, input_ready, 0
    for group in range(groups):
        ready = []
        for index in range(sets):
            position = (slot + index) % capacity
            producer = max(producer, free[position]) + load
            ready.append(producer + delay)
        for replay in range(replays):
            for index in range(sets):
                consumer = max(consumer, ready[index]) + compute
                if replay == replays - 1:
                    free[(slot + index) % capacity] = consumer
        slot = (slot + sets) % capacity
    return consumer


# Cover cold loading, prefetch, finite residency, replay collapse, and fixed work bounds
class ReadinessTests(unittest.TestCase):
    # Compare small ring layouts and rate combinations without large RTL workloads
    def test_small_reference(self):
        for capacity in (1, 2, 3, 5):
            for sets in range(1, capacity + 1):
                for replays in (1, 3):
                    for load, compute in ((2, 7), (7, 2), (3, 3)):
                        args = (sets, replays, 19, capacity, load, 1, compute, 5)
                        with self.subTest(args=args):
                            finish, steps, bounded = buffer_completion(*args)
                            self.assertFalse(bounded)
                            self.assertEqual(finish, reference(*args))


if __name__ == "__main__":
    unittest.main()
