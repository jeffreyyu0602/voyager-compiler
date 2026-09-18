# Model bias demand at first-reduction boundaries for both matrix backends
from functools import lru_cache
from ..timing.backpressure import burst, idle
from ..timing.transfer import transfer_cycles


# Keep bias reuse and reduction gaps while composing whole repeated loop bodies
@lru_cache(maxsize=4096)
def bias_completion(l1, l2, cycles_per_vector, interval=1, prefix_at=None, prefix_cycles=0):
    # WeightController's feeder can hold one assembled bias vector ahead of its use
    summary = burst(1, 0, cycles_per_vector, 1).then(idle(interval))
    oc_position = next(index for index, (loop, _) in enumerate(l1) if loop == "OC")
    for index, (loop, bound) in enumerate(l1 + l2):
        if index == prefix_at:
            summary = idle(prefix_cycles).then(summary)
        reused = index < oc_position and loop in ("OX", "OY")
        if loop in ("IC", "FX", "FY") or reused:
            summary = summary.then(idle((bound - 1) * summary.work_cycles))
        else:
            summary = summary.repeated(bound)
    return summary.timing()


# Derive bias bandwidth from actual accumulator width and external port width
# The first prefetched vector is covered by matrix startup, not charged on every tile
def bias_timing(target, l1, l2, *, interval=1, request_latency=0,
                prefix_at=None, prefix_cycles=0):
    cycles_per_vector = transfer_cycles(target.n * target.accum_bits, target.oc_port_bits, request_latency)
    timing = bias_completion(l1, l2, cycles_per_vector, interval, prefix_at, prefix_cycles)
    return timing, dict(bias_wait_cycles=timing.stall_cycles,
                        bias_cycles_per_vector=cycles_per_vector,
                        bias_prefetch_vectors=1)
