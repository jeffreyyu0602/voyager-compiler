# Couple reusable operand storage and output bandwidth at whole-burst boundaries
from .backpressure import burst

MAX_BURST_STEPS = 512
MAX_BUFFER_SLOTS = 64


# Stop an unrecognized transient before timing work grows with the workload
class TimingBudgetExceeded(Exception):
    pass


# Track fill fill_cycles and release/ready times for reusable operand slots
class BufferSlots:
    # Keep throughput separate from readiness and ownership delays
    def __init__(self, capacity, fill_cycles, *, ready_delay=0, release_delay=0,
                 load_start=0, first_fill_cycles=None):
        if capacity > MAX_BUFFER_SLOTS:
            raise TimingBudgetExceeded
        self.capacity, self.fill_cycles = capacity, fill_cycles
        self.ready_delay, self.release_delay = ready_delay, release_delay
        self.loader_at, self.position = load_start, 0
        self.first_fill_cycles = fill_cycles if first_fill_cycles is None else first_fill_cycles
        self.free, self.ready = [0] * capacity, [0] * capacity

    # Normalize irrelevant idle producer_at while retaining enough lead to prefill every slot
    def state(self, producer_at):
        self.loader_at = max(self.loader_at, producer_at - self.capacity * self.fill_cycles - self.ready_delay)
        self.free = [max(t, self.loader_at) for t in self.free]
        self.ready = [max(t, producer_at) for t in self.ready]
        return (self.position, self.first_fill_cycles, self.loader_at - producer_at,
                tuple(t - producer_at for t in self.free), tuple(t - producer_at for t in self.ready))

    # Translate clocks after skipping identical repetitions
    def shift(self, cycles):
        self.loader_at += cycles
        self.free = [t + cycles for t in self.free]
        self.ready = [t + cycles for t in self.ready]


# Compose producer bursts, operand waits and draining output backlog
class OverlapTiming:
    # Counters distinguish the waits remaining after their actual overlap
    def __init__(self, buffers, output_cycles_per_vector, output_capacity_vectors):
        self.buffers = buffers
        self.output_cycles_per_vector, self.output_capacity_vectors = output_cycles_per_vector, output_capacity_vectors
        self.producer_at = self.consumer_at = self.operand_ready_at = self.steps = 0
        self.started = False
        self.waits = [0] * (len(buffers) + 2)

    # Load or replay one resident slot without serializing independent loaders
    def acquire(self, resource, *, load=True):
        buffer = self.buffers[resource]
        slot = buffer.position
        buffer.position = (slot + 1) % buffer.capacity
        if load:
            fill = buffer.first_fill_cycles if buffer.first_fill_cycles is not None else buffer.fill_cycles
            buffer.first_fill_cycles = None
            buffer.loader_at = max(buffer.loader_at, buffer.free[slot]) + fill
            buffer.ready[slot] = buffer.loader_at + buffer.ready_delay
        wait = max(0, buffer.ready[slot] - self.producer_at)
        self.producer_at += wait
        if self.started:
            self.waits[resource] += wait
        return slot

    # Output stalls postpone the reuse of the operand that produced that burst
    def release(self, resource, slot):
        buffer = self.buffers[resource]
        buffer.free[slot] = self.producer_at + buffer.release_delay

    # Apply one bandwidth equation to a whole compute burst and its paced operand demand
    def produce(self, cycles, output_vectors, *, requests=0, request_cycles=0):
        if self.steps >= MAX_BURST_STEPS:
            raise TimingBudgetExceeded
        self.steps += 1
        start, work = self.producer_at, cycles
        if requests:
            spacing = cycles // requests
            start = max(start, self.operand_ready_at)
            work += (requests - 1) * max(0, request_cycles - spacing)
            self.waits[-2] += start - self.producer_at + work - cycles
        summary = burst(output_vectors, work, self.output_cycles_per_vector, self.output_capacity_vectors)
        timing = summary.timing(max(0, self.consumer_at - start))
        self.producer_at, self.consumer_at = start + timing.producer_cycles, start + timing.consumer_cycles
        self.waits[-1] += timing.stall_cycles
        if requests:
            last_request = start if requests == 1 else self.producer_at - spacing
            self.operand_ready_at = last_request + request_cycles
        self.started = True

    # Skip identical normalized repetitions instead of walking every use or tile
    def repeat(self, count, body):
        seen, index = {}, 0
        while index < count:
            self.consumer_at = max(self.consumer_at, self.producer_at)
            self.operand_ready_at = max(self.operand_ready_at, self.producer_at)
            state = (self.consumer_at - self.producer_at, self.operand_ready_at - self.producer_at, self.started,
                     tuple(buffer.state(self.producer_at) for buffer in self.buffers))
            if state in seen:
                previous, producer_at, waits = seen[state]
                repeats = (count - index) // (index - previous)
                if repeats:
                    elapsed = repeats * (self.producer_at - producer_at)
                    self.producer_at += elapsed
                    self.consumer_at += elapsed
                    self.operand_ready_at += elapsed
                    for buffer in self.buffers:
                        buffer.shift(elapsed)
                    self.waits = [now + repeats * (now - old) for now, old in zip(self.waits, waits)]
                    index += repeats * (index - previous)
                    continue
            else:
                seen[state] = index, self.producer_at, tuple(self.waits)
            body()
            index += 1
