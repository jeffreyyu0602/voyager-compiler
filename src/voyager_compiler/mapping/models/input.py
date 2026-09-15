# Count shared InputController traffic using compressed boundary tile classes
from collections import Counter
from dataclasses import dataclass
from itertools import product
from ..timing.transfer import ceil_div, transfer_cycles


# Separate first-fill latency from total bank-fill work
@dataclass(frozen=True)
class InputTraffic:
    requests: int
    writes: int
    total_fill_cycles: int
    first_fill_cycles: int
    fills: int
    max_fill_cycles: int
    min_fill_cycles: int


# Derive the input bank's halo extents from temporal factors
def input_tile_shape(level, workload):
    return tuple(level[output] * (workload.stride if level[kernel] != 1 else 1)
                 + level[kernel] - 1 for output, kernel in (("OX", "FX"), ("OY", "FY")))


# Return the contiguous inner indices whose projected coordinates are in bounds
def _valid_span(origin, step, extent, inputs):
    begin = min(extent, max(0, ceil_div(-origin, step)))
    end = max(0, min(extent, (inputs - 1 - origin) // step + 1))
    return (begin, end) if begin < end else (0, 0)


# Collapse interior and exterior tiles while enumerating only changing boundaries
def _axis_tiles(count, extent, inputs, offset, fetch_steps, writer_steps):
    cuts = {0, count}
    for outer, inner in (fetch_steps, writer_steps):
        for edge in (0, (extent - 1) * inner):
            cuts.add(min(count, max(0, ceil_div(-offset - edge, outer))))
            cuts.add(min(count, max(0, (inputs - 1 - offset - edge) // outer + 1)))
    # Project both address generators at an outer tile index
    def spans(index):
        return tuple(_valid_span(offset + index * outer, inner, extent, inputs)
                     for outer, inner in (fetch_steps, writer_steps))
    result = Counter()
    points = sorted(cuts)
    for begin, end in zip(points, points[1:]):
        first = spans(begin)
        if first == spans(end - 1):
            result[first] += end - begin
        else:
            result.update(spans(index) for index in range(begin, end))
    return result


# Count exact input traffic and estimate memory and unpacking cycles per bank
def input_bank_traffic(l1, l2, workload, *, lanes, element_bits, port_bits, pack, request_latency=0):
    width, height = input_tile_shape(l1, workload)
    stride, padding = workload.stride, workload.padding
    x_fetch = (l1["OX"] * stride, stride if l1["FX"] == 1 else 1)
    x_writer = (l1["OX"] * (stride if l1["FX"] != 1 else 1), 1)
    y_step = stride if l1["FY"] == 1 else 1
    y_fetch = (l1["OY"] * stride, y_step)
    y_writer = (l1["OY"] * (stride if l1["FY"] != 1 else 1), y_step)
    for count, extent, offset, rules in (
            (l2["OX"], width, -padding, (x_fetch, x_writer)),
            (l2["OY"], height, l2["FY"] - 1 - padding, (y_fetch, y_writer))):
        if max(offset + (count - 1) * outer + (extent - 1) * inner for outer, inner in rules) > 32767:
            return None, ["input coordinate exceeds signed 16-bit addressing"]
    xs = _axis_tiles(l2["OX"], width, workload.input_x, -padding, x_fetch, x_writer)
    ys = Counter()
    for fy in range(l2["FY"]):
        ys.update(_axis_tiles(l2["OY"], height, workload.input_y, fy - padding, y_fetch, y_writer))
    channels = l1["IC"] // pack
    writes = width * height * l1["IC"]
    memory_interval = transfer_cycles(lanes * pack * element_bits, port_bits, request_latency)
    requests = total_fill_cycles = max_fill = 0
    min_fill = None
    for ((xf, xw), nx), ((yf, yw), ny) in product(xs.items(), ys.items()):
        valid = (xf[1] - xf[0]) * (yf[1] - yf[0])
        writer_valid = (xw[1] - xw[0]) * (yw[1] - yw[0])
        if (valid or writer_valid) and (xf != xw or yf != yw):
            return None, ["input fetcher and writer disagree on padding at this stride"]
        packets = valid * channels
        requests += nx * ny * packets
        # Fetching the next bank overlaps the current bank's final writes
        fill_cycles = max(packets * memory_interval, writes)
        total_fill_cycles += nx * ny * fill_cycles
        max_fill = max(max_fill, fill_cycles)
        min_fill = fill_cycles if min_fill is None else min(min_fill, fill_cycles)
    first_x = _valid_span(-padding, x_fetch[1], width, workload.input_x)
    first_y = _valid_span(-padding, y_fetch[1], height, workload.input_y)
    first_packets = (first_x[1] - first_x[0]) * (first_y[1] - first_y[0]) * channels
    first_cycles = max(first_packets * memory_interval, writes) + (min(memory_interval, pack) if first_packets else 0)
    repetitions = l2["IC"] * l2["OC"]
    fills = l2["OX"] * l2["OY"] * l2["FY"] * repetitions
    return InputTraffic(requests * repetitions, writes * fills, total_fill_cycles * repetitions, first_cycles,
                        fills, max_fill, min_fill), []
