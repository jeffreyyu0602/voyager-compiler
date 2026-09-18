# Model reusable storage slots with bounded repeated-state timing
from functools import lru_cache

MAX_BUFFER_STATES = 64
MAX_SEQUENCE_STEPS = 96


# Return completion time, explicit sequence steps, and whether a serialized bound was used
@lru_cache(maxsize=4096)
def buffer_completion(slots_per_sequence, uses_per_sequence, sequence_count, capacity, fill_cycles, ready_delay, compute_cycles, input_ready,
                      *, release_delay=0, load_start=0):
    if min(slots_per_sequence, uses_per_sequence, sequence_count, capacity, fill_cycles, compute_cycles) <= 0 or slots_per_sequence > capacity:
        raise ValueError("buffer timing requires positive counts and a sequence fitting the available slots")
    capacity = min(capacity, slots_per_sequence * sequence_count)
    if capacity > MAX_BUFFER_STATES:
        # Bound unsupported ring sizes by serializing complete fill_cycles/compute_cycles sequences
        return input_ready + load_start + sequence_count * (slots_per_sequence * fill_cycles + ready_delay + release_delay
                                                     + slots_per_sequence * uses_per_sequence * compute_cycles), 0, True
    free = [0] * capacity
    producer, consumer, sequence, steps = load_start, input_ready, 0, 0
    seen = {}
    while sequence < sequence_count and steps < MAX_SEQUENCE_STEPS:
        # Past releases cannot constrain a future fill_cycles; ring rotation removes physical slot labels
        state = (consumer - producer, tuple(max(0, time - producer) for time in free))
        if state in seen:
            previous_sequence, previous_time = seen[state]
            period, elapsed = sequence - previous_sequence, producer - previous_time
            repeats = (sequence_count - sequence) // period
            if repeats:
                shift = repeats * elapsed
                producer += shift
                consumer += shift
                free = [time + shift for time in free]
                sequence += repeats * period
                if sequence == sequence_count:
                    break
        else:
            seen[state] = (sequence, producer)
        releases = []
        for index in range(slots_per_sequence):
            producer = max(producer, free[index]) + fill_cycles
            consumer = max(consumer, producer + ready_delay) + compute_cycles
            releases.append(consumer + release_delay)
        if uses_per_sequence > 1:
            # All slots are ready after the first pass; later passes take one arithmetic step
            releases = [consumer + ((uses_per_sequence - 2) * slots_per_sequence + index + 1) * compute_cycles + release_delay
                        for index in range(slots_per_sequence)]
            consumer += (uses_per_sequence - 1) * slots_per_sequence * compute_cycles
        free[:slots_per_sequence] = releases
        free = free[slots_per_sequence:] + free[:slots_per_sequence]
        sequence += 1
        steps += 1
    bounded = sequence < sequence_count
    if bounded:
        # Never expand an unrecognized long transient into per-sequence work
        consumer = max(consumer, producer + ready_delay) + (sequence_count - sequence) * (
            slots_per_sequence * fill_cycles + ready_delay + release_delay + slots_per_sequence * uses_per_sequence * compute_cycles)
    return consumer, steps, bounded
