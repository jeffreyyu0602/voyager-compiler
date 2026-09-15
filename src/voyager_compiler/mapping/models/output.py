# Project matrix loops into an analytical bandwidth-and-capacity model shared by SA and CIM
from dataclasses import dataclass
from functools import lru_cache
from ..timing.backpressure import burst, idle


# State effective storage in logical matrix-result groups without describing its implementation
@dataclass(frozen=True)
class OutputOptions:
    output_capacity_vectors: int = None
    output_elements_per_vector: int = 0

    # Bind an explicit capacity to its group width; omission uses the exported output FIFO
    def __post_init__(self):
        if type(self.output_elements_per_vector) is not int or self.output_elements_per_vector < 0:
            raise ValueError('output_elements_per_vector must be a nonnegative integer')
        if self.output_capacity_vectors is None:
            if self.output_elements_per_vector:
                raise ValueError('output group width requires an explicit capacity')
        elif type(self.output_capacity_vectors) is not int or self.output_capacity_vectors < 0 or not self.output_elements_per_vector:
            raise ValueError('output capacity must be nonnegative with an explicit group width')


# Preserve reduction gaps and output bursts by composing summaries, never individual tiles
@lru_cache(maxsize=4096)
def completion_timing(loops, cycles_per_vector, capacity_vectors, interval=1, prefix_at=None, prefix_cycles=0):
    summary = burst(1, interval, cycles_per_vector, capacity_vectors)
    for index, (loop, bound) in enumerate(loops):
        if index == prefix_at:
            summary = idle(prefix_cycles).then(summary)
        if loop in ('IC', 'FX', 'FY'):
            summary = idle((bound - 1) * summary.work_cycles).then(summary)
        else:
            summary = summary.repeated(bound)
    return summary.timing()


# Resolve explicit group capacity and expose the producer stall and remaining consumer work
def output_timing(target, options, loops, cycles_per_vector, *, direct=False, interval=1,
                  prefix_at=None, prefix_cycles=0):
    explicit = options.output_capacity_vectors is not None and not direct
    if explicit and options.output_elements_per_vector != target.n:
        raise ValueError('output capacity group width must match the matrix output lane count')
    if explicit and target.vector_config['lanes'] != target.n:
        raise ValueError('output capacity profile requires equal matrix and vector group widths')
    capacity = options.output_capacity_vectors if explicit else target.vector_config['output_fifo_packets']
    result = completion_timing(loops, cycles_per_vector, capacity, interval, prefix_at, prefix_cycles)
    return result, dict(output_stall_cycles=result.stall_cycles,
                        output_consumer_finish_cycles=result.consumer_cycles,
                        output_backlog_cycles=result.backlog_cycles,
                        output_capacity_vectors=capacity,
                        output_elements_per_vector=target.n,
                        output_cycles_per_vector=cycles_per_vector,
                        output_capacity_explicit=explicit)
