# Model bandwidth mismatch with one backlog, measured in consumer work cycles
# For B vectors produced in T cycles, s cycles/vector and capacity E vectors:
# headroom = max(0, E*s - L), where L is forward plus credit-return latency
# stall = max(0, backlog + B*s - T - headroom)
# next_backlog = max(0, backlog + B*s - T - stall)
from dataclasses import dataclass


# Report producer delay separately from the consumer work still queued at completion
@dataclass(frozen=True)
class StreamTiming:
    producer_cycles: int
    consumer_cycles: int
    stall_cycles: int
    backlog_cycles: int


# Summarize elapsed time and backlog without retaining individual bursts or packets
@dataclass(frozen=True)
class StreamSummary:
    work_cycles: int = 0
    # p is producer time and d = p + backlog is the consumer's busy-until time
    # Four offsets express p' = max(p+a, d+b), d' = max(p+c, d+e)
    offsets: tuple = (0, float('-inf'), float('-inf'), 0)

    # Substitute the two clock equations to compose adjacent pieces of work
    def then(self, other):
        a, b, c, d = self.offsets
        e, f, g, h = other.offsets
        return StreamSummary(self.work_cycles + other.work_cycles,
                             (max(a + e, c + f), max(b + e, d + f),
                              max(a + g, c + h), max(b + g, d + h)))

    # Double summarized loop bodies so work grows logarithmically with the loop bound
    def repeated(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('repeat count must be a nonnegative integer')
        result, body = StreamSummary(), self
        while count:
            if count & 1:
                result = result.then(body)
            count //= 2
            if count:
                body = body.then(body)
        return result

    # Evaluate the summary once, retaining any backlog inherited from preceding work
    def timing(self, backlog_cycles=0):
        a, b, c, d = self.offsets
        producer, consumer = max(a, backlog_cycles + b), max(c, backlog_cycles + d)
        return StreamTiming(producer, consumer, producer - self.work_cycles, consumer - producer)


# Express the backlog equation as two elapsed-time constraints for a constant-rate burst
def burst(vectors, cycles, cycles_per_vector, capacity_vectors, *, credit_delay_cycles=0):
    demand = vectors * cycles_per_vector
    capacity = max(0, capacity_vectors * cycles_per_vector - credit_delay_cycles)
    # The consumer clock tracks processing work; its forward latency belongs to final drain
    # p' = max(p+T, d+B*s-headroom), d' = max(p', d+B*s)
    return StreamSummary(cycles, (cycles, demand - capacity, cycles, demand))


# Drain queued work during producer computation that yields no completed results
def idle(cycles):
    return StreamSummary(cycles, (cycles, float('-inf'), cycles, 0))
