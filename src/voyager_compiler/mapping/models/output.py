# Project matrix loops into an analytical bandwidth-and-capacity model shared by SA and CIM
from dataclasses import dataclass
from functools import lru_cache
from ..timing.backpressure import burst, idle


# Preserve the shared options contract while hardware alone supplies storage capacity
@dataclass(frozen=True)
class OutputOptions:
    pass


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


# Normalize distinct payload storage by matrix group width without counting parallel operands
# Direct matrix output bypasses vector storage; consumer vector_timing already covers its active payload
def output_timing(target, loops, cycles_per_vector, *, direct=False, interval=1,
                  prefix_at=None, prefix_cycles=0):
    storage = {name: elements for name, elements in target.output_storage.items()
               if name != 'vector_pipeline' or not direct}
    capacity = sum(storage.values()) // target.n
    result = completion_timing(loops, cycles_per_vector, capacity, interval, prefix_at, prefix_cycles)
    return result, dict(output_stall_cycles=result.stall_cycles,
                        output_consumer_finish_cycles=result.consumer_cycles,
                        output_backlog_cycles=result.backlog_cycles,
                        output_capacity_vectors=capacity,
                        output_elements_per_vector=target.n,
                        output_cycles_per_vector=cycles_per_vector,
                        output_storage_elements=storage)
