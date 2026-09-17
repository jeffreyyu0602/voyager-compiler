# Model port transfers in bits independently of matrix backend or scalar precision
from math import gcd


# Compute an exact ceiling for payloads and signed coordinate projections
def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


# Count bus beats and declared per-request overhead without charging extra handoff cycles
def transfer_cycles(payload_bits, port_bits, request_latency=0):
    return ceil_div(payload_bits, port_bits) + request_latency


# Match dtype_fetch_config and Common.h's all-or-one channel packing
def packing_factor(vector_bits, port_bits, bound):
    factor = port_bits // gcd(vector_bits, port_bits)
    return factor if factor & (factor - 1) == 0 and bound % factor == 0 else 1


# Separate first-fill latency from steady fill duration for a reader feeding a storage writer
def stream_fill(source_items, read_cycles, destination_items, write_cycles, *, blocking=False):
    if blocking:
        first = source_items * read_cycles + destination_items * write_cycles
        return first, first
    first = read_cycles + (source_items - 1) * max(read_cycles, write_cycles)
    first += (destination_items - source_items + 1) * write_cycles
    return first, max(source_items * read_cycles, destination_items * write_cycles)
