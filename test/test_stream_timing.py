# Check the backlog equation and algebraic repetition with small transparent examples
import unittest
from voyager_compiler.mapping.timing.backpressure import StreamSummary, burst, idle


# Check analytical composition against the equation applied only to a few whole bursts
class StreamTests(unittest.TestCase):
    # Pipeline transit and returning credits occupy storage without adding service work
    def test_credit_delay(self):
        plain = burst(8, 8, 2, 4).timing()
        delayed = burst(8, 8, 2, 4, credit_delay_cycles=3).timing()
        self.assertEqual((plain.producer_cycles, plain.consumer_cycles), (8, 16))
        self.assertEqual((delayed.producer_cycles, delayed.consumer_cycles), (11, 16))
        self.assertEqual(delayed.backlog_cycles, 5)
        self.assertEqual(burst(8, 8, 2, 4, credit_delay_cycles=20).timing().producer_cycles, 16)
        period = burst(8, 8, 2, 4, credit_delay_cycles=3).then(idle(10))
        self.assertEqual(period.repeated(1000).timing().producer_cycles, 21000)

    # Preserve burst boundaries, incoming backlog and service units through nested summaries
    def test_small_equation_reference(self):
        phases = ((9, 9), (0, 3), (4, 8), (0, 5))
        for capacity in (0, 3, 8):
            for service in (1, 2, 3):
                body = StreamSummary()
                for groups, cycles in phases:
                    body = body.then(burst(groups, cycles, service, capacity))
                for incoming in (0, capacity * service):
                    backlog, stalls = incoming, 0
                    for _ in range(7):
                        for groups, cycles in phases:
                            excess = backlog + groups * service - cycles
                            stall = max(0, excess - capacity * service)
                            stalls += stall
                            backlog = max(0, excess - stall)
                    timing = body.repeated(7).timing(incoming)
                    self.assertEqual((timing.stall_cycles, timing.backlog_cycles), (stalls, backlog))
                    self.assertEqual(body.repeated(7), body.repeated(1).then(body.repeated(6)))


if __name__ == '__main__':
    unittest.main()
