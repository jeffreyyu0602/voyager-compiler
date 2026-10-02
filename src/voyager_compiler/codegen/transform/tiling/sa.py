"""Systolic architecture, loop constraints, and matrix runtime estimates."""

import math
from typing import Optional, Tuple

import interstellar
from voyager_compiler.codegen.transform.tiling.cost import (
    BANK_SWITCH_CYCLES,
    _step_classes,
    _sweep_cycles,
    strided_bank_walk,
)
from voyager_compiler.codegen.transform.tiling.input import input_buffer_usage

le = interstellar.le

# Finished output vectors the matrix -> vector path holds before a
# single-buffered accumulator's array feels the tail's drain rate: the
# matrix processor's output FIFO (8), the vector pipeline's input FIFO (9)
# and the stages between.  Measured on the E4M3x16x16 SoC as 20 (MobileBERT
# GEMMs) to 40 (ResNet18) vectors; either way the error is under ~60 cycles
# per L3 step.
OUTPUT_SLACK = 24

# Cycles the SpMM unit spends per row of each PE-array-wide pass on top of
# the row's outliers: it drains and restarts its accumulator ring between
# rows.  Measured on the Sphinx SoC: 7.3-7.5.
SPMM_ROW_CYCLES = 8

# Block-scale rows the SpMM unit's weight-scale buffer holds
# (``DoubleBuffer<32>`` in ``SpMMUnit.h``, fixed in the Sphinx silicon).  A
# streaming weight tile takes one row per K block per inner OC pass, and the
# hardware wraps the address rather than checking it.
SPMM_SCALE_ROWS = 32


def spmm_scale_rows(mapping):
    """Weight-scale buffer rows ``mapping``'s tile takes in the SpMM unit:
    its K blocks (the L1 and L2 IC blockings; the PE level is the array)
    times its L1 OC passes -- ``C * K0`` in the toolchain's ``SpMM.h``."""
    ic, oc = mapping.loop_blockings[le.IC], mapping.loop_blockings[le.OC]
    return ic[1] * ic[2] * oc[1]


# The L3 loops each operand's tile spans.  A tile advances only when one of
# them turns, so an operand whose every entry is a single L3 step reads one
# tile for the whole sweep.
_IF_DIMS = (le.OX, le.OY, le.IC, le.ON)
_FL_DIMS = (le.OC, le.IC, le.FX, le.FY)
_OF_DIMS = (le.OC, le.OY, le.OX, le.ON)


def build_architecture(config, dram_access_cost):
    """Build the systolic resource hierarchy and loop-order constraints."""
    ic_dim, oc_dim = config.pe_array_size

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                config.input_buffer_size * ic_dim,
                config.accum_buffer_size * oc_dim,
                config.weight_buffer_size * oc_dim,
            ],
            [config.usable_scratchpad_size],
            [config.dram_size * 1024**3],  # GB -> bytes
        ],
        buf_access_cost_list=[
            [1, 1, 1],
            [10, 10, 10],
            [100],
            [dram_access_cost],
        ],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
        para_count_list=[ic_dim * oc_dim, 1, 1, 1],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        mac_capacity=0,
        partition_mode=[0, 0, 0, 0],
        invalid_underutilized=False,
        bank_size_list=[None, None, config.bank_size, None],
    )

    # L1 IC is outermost; the inner order is pinned to FY > FX > OY > OX.
    # OX/OY innermost is never slower -- the L1 sweep costs
    # ``max(loading, reused_tile) * remaining``, monotone in the reused
    # tile -- and the arrangement of the loops above them ties on both
    # runtime and energy, so FX/FY are fixed to the representative the
    # search's first-seen tie-break picked anyway.  FX=2/FY=3 assumes a
    # square kernel (equal FX/FY blocking); a non-square model may prefer
    # them swapped.
    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": ic_dim},
                "level1": {"order": -1},
                "level2": {"order": 0},
                "level3": {"order": 0},
            },
            "OC": {
                "level0": {"order": 0, "partitioning_size": oc_dim},
            },
            "OX": {
                "level1": {"order": 0},
            },
            "OY": {
                "level1": {"order": 1},
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 2},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 3},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
        }
    }
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint, 4
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return architecture, schedule


class RuntimeCalculator:
    """Runtime cost model for a 4-level hierarchy (PE / L1 / L2 / DRAM).

    A mapping is priced in cycles as its L3 grid sweep.  With a
    double-buffered L2 each step costs the slower of its DRAM transfers and
    its compute, otherwise their sum, and the sweep is framed by the
    un-overlapped first load and last store; a split reduction adds its
    accumulate steps and the tail pass that finishes each output tile.  A
    DRAM transfer costs its bytes at ``dram_bandwidth`` plus one access
    latency, block scales being a transfer of their own, and the batch loop
    outside the mapping shares a weight tile among the elements of one
    group.  Energy is interstellar's cost model, not this one.

    The compute of an L2 block is the busier of the matrix unit and the
    scratchpad bus.  The matrix unit charges the systolic passes, each weight
    tile costing the longer of its loading and the rows streamed through it,
    plus the back-pressure of a single-buffered accumulator against
    ``OUTPUT_SLACK`` of buffering.  The bus charges the words every operand
    role moves, summed per bank group of the planner's partition with the
    busiest bank setting the pace: input and weight rows in whole beats,
    packed as the toolchain packs them, with the read interface's lost cycle
    whenever an aligned request follows an unaligned one; block scales one
    word each; the output, the fused tail operands, the bias and the
    reduction scratch; and the round trip a stream pays when it changes bank
    on its bank-aligned tile buffer.  A weight tile held across the spatial
    loops is fetched once.  The SpMM unit adds a turnaround per row visit and
    the outlier rows it gathers at the block's K depth, most of them a bank
    switch when the weight tile spans several banks.  A tail pass of its own
    costs the vector unit's lane rate or its bank words, whichever is
    slower.  The ramps of a sweep, the first buffer fill, the systolic skew
    and the last drain, are spread over the ops the sweep dispatches.

    Not modeled are the per-op costs (parameter load, deserialisation, the
    start/done handshake, the drain between uncommitted ops, the host's
    dispatch time), so an op that runs alone pays its ramps in full where
    the spreading charges a share; the cycle-exact datapath, the systolic
    skew being a constant of the array dims; a stream-breaking
    ``quantize_mx`` tail, charged as a staged region rather than a drained
    pass; more than one port width, every operand moving at
    ``sram_bandwidth`` over one bus per bank; the block scales' bank
    switches; and the outlier density of an individual tile, priced at the
    layer's average.
    """

    def __init__(
        self,
        input_dtype_width: int,
        weight_dtype_width: int,
        output_dtype_width: int,
        accum_dtype_width: int,
        double_buffered_accum_buffer: bool,
        sram_bandwidth: int,
        dram_bandwidth: int,
        dram_access_latency_cycles: float,
        double_buffered_l2: bool = False,
        outlier_rate: float = 0.0,
        batch: int = 1,
        weight_batch: Optional[int] = None,
        has_tail: bool = False,
        single_k_tail_extra_pass: bool = False,
        split_k_tail_extra_pass: bool = False,
        tail_keeps_shape: bool = False,
        tail_specs=(),
        input_scale_width: int = 0,
        weight_scale_width: int = 0,
        output_scale_width: int = 0,
        scale_block_size: int = 1,
        bias_width: int = 0,
        stride: Tuple[int, int] = (1, 1),
        bank_size: Optional[int] = None,
        weight_transposed: bool = False,
        input_buffer_size: Optional[int] = None,
    ):
        self.input_dtype_width = input_dtype_width
        self.weight_dtype_width = weight_dtype_width
        self.output_dtype_width = output_dtype_width
        self.accum_dtype_width = accum_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.sram_bandwidth = sram_bandwidth
        self.dram_bandwidth = dram_bandwidth
        self.dram_access_latency_cycles = dram_access_latency_cycles
        self.double_buffered_l2 = double_buffered_l2
        self.outlier_rate = outlier_rate
        self.batch = batch
        self.weight_batch = batch if weight_batch is None else weight_batch
        self.has_tail = has_tail
        self.single_k_tail_extra_pass = single_k_tail_extra_pass
        self.split_k_tail_extra_pass = split_k_tail_extra_pass
        self.tail_keeps_shape = tail_keeps_shape
        self.tail_specs = tuple(tail_specs)
        self.input_scale_width = input_scale_width
        self.weight_scale_width = weight_scale_width
        self.output_scale_width = output_scale_width
        self.scale_block_size = scale_block_size
        self.bias_width = bias_width
        self.stride = stride
        self.bank_size = bank_size
        self.weight_transposed = weight_transposed
        self.input_buffer_size = input_buffer_size
        self.dram_bytes = {}

    def tail_tile_sizes(self, mapping):
        """DRAM bytes each fused tail operand streams for one output tile --
        one transfer apiece.  An operand is tiled along the output dims it is
        not broadcast over, so its tile is the output tile's extent there."""
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings
        sizes = []
        for dims, bits in self.tail_specs:
            count = 1
            for d in dims:
                count *= blockings[d][1] * blockings[d][2] * partitionings[d][0]
            sizes.append(count * bits / 8)
        return sizes

    def _bank_cycles(self, words, bank_groups):
        """Cycles the busiest scratchpad bank spends moving ``words`` -- bus
        words per operand role -- when the roles sharing a bank
        (``bank_groups``: the search's partition as role sets, ``None`` =
        nothing shares) queue on its single port.  A role the partition does
        not name keeps a port of its own.
        """
        placed = set()
        busiest = 0
        for roles in bank_groups or ():
            busiest = max(busiest, sum(words.get(role, 0) for role in roles))
            placed.update(roles)
        loose = [count for role, count in words.items() if role not in placed]
        return max([busiest, *loose])

    def _request_words(self, requests, row_elems, bits, loop_bound, pitch):
        """Bus words ``requests`` fetches of one ``row_elems``-element row
        take.  A request is served in whole beats, so a row that is not a
        multiple of the port costs more than its bytes.  The controller packs
        the rows that fill whole beats into one request when the L1 loop
        that walks them, of ``loop_bound`` steps, divides into that many
        (the toolchain's ``get_packing_factor``); a ``loop_bound`` of 0 never
        packs.  Consecutive requests are ``pitch`` bytes apart, the row
        pitch of the tile buffer.  The SoC's read interface flushes the tail
        of a request that started inside a beat in a cycle of its own and
        holds the next request's first beat when that one starts on a beat
        boundary, so a pitch that is not a multiple of the beat cycles the
        requests through the beat offsets and costs one cycle per aligned
        request that follows an unaligned one."""
        if not bits or requests <= 0:
            return 0
        row_bits = row_elems * bits
        pf = math.lcm(row_bits, self.sram_bandwidth) // row_bits
        rows_per_request = pf if loop_bound and loop_bound % pf == 0 else 1
        request_words = math.ceil(
            rows_per_request * row_bits / self.sram_bandwidth
        )
        count = math.ceil(requests / rows_per_request)
        beat = self.sram_bandwidth // 8
        misalignment = rows_per_request * pitch % beat
        flushes = 0
        if misalignment:
            flushes = count * math.gcd(misalignment, beat) / beat
        return count * request_words + flushes

    def _bus_words(self, count, bits, bandwidth=None):
        """Bus words ``count`` elements of ``bits`` occupy at the bank's full
        width, or at ``bandwidth`` bytes per cycle."""
        if not bits or count <= 0:
            return 0
        return math.ceil(
            count * bits / 8 / (bandwidth or self.sram_bandwidth / 8)
        )

    @staticmethod
    def _extent(mapping, loop, level):
        """Elements along ``loop`` in one tile at ``level``: the blockings
        through that level times the PE-array partition."""
        extent = mapping.loop_partitionings[loop][0]
        for blocking in mapping.loop_blockings[loop][1 : level + 1]:
            extent *= blocking
        return extent

    def _load_words(self, mapping):
        """Bus words one L1 tile's every-round operands take, per role: the
        input and its block scales, the weight and its scales.  The input
        and the weight arrive one PE-array row per request
        (``_request_words``), packed several rows per request when the L1
        loop that walks them allows it -- never for a transposed weight;
        the weight scales one PE-array row of scales per request, never
        packed.  An input scale is delivered one
        per bus word however narrow it is; every outlier in the input tile
        gathers one more weight row on top of the dense weight tile."""
        ext = lambda loop: self._extent(mapping, loop, 1)
        rows = ext(le.OX) * ext(le.OY) * ext(le.ON)
        depth = ext(le.IC)
        taps = ext(le.FX) * ext(le.FY)
        gathered_rows = rows * depth * self.outlier_rate
        blockings = mapping.loop_blockings
        ic_unroll = mapping.loop_partitionings[le.IC][0]
        oc_unroll = mapping.loop_partitionings[le.OC][0]
        weight_loop = 0 if self.weight_transposed else blockings[le.OC][1]
        # The input tile includes the convolution halo and stride.
        input_words, _ = input_buffer_usage(mapping, self.stride)
        # The tile buffers are [rows, IC] for the input and [IC, OC] for the
        # weight and its scales: a request walks one row of each.
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        input_pitch = ic3 * self.input_dtype_width // 8
        weight_pitch = oc3 * self.weight_dtype_width // 8
        scale_pitch = oc3 * self.weight_scale_width // 8
        words = {
            "input": self._request_words(
                input_words,
                ic_unroll,
                self.input_dtype_width,
                blockings[le.IC][1],
                input_pitch,
            ),
            "weight": self._request_words(
                (depth * taps + gathered_rows) * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_dtype_width,
                weight_loop,
                weight_pitch,
            ),
            "weight_scale": self._request_words(
                depth * taps / self.scale_block_size * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_scale_width,
                0,
                scale_pitch,
            ),
        }
        if self.input_scale_width:
            words["input_scale"] = math.ceil(
                input_words * ic_unroll / self.scale_block_size
            )
        return words

    def _tail_words(self, mapping, level):
        """Bus words the tail's operands take for one tile at ``level`` (1 =
        an L1 output tile, 2 = the whole L3 output tile), per role: the
        finished output with its block scales -- each scale leaves in a bus
        word of its own, however narrow, as the input scales arrive -- and
        each fused operand over the output dims it is tiled along.  The tail
        rides on the array's output one vector (a row of ``oc_dim`` values)
        at a time and fetches a fused operand in one unpacked request per
        vector, so an operand narrower than a bus word still costs a word
        per vector."""
        ext = lambda loop: self._extent(mapping, loop, level)
        out = ext(le.OC) * ext(le.OY) * ext(le.OX) * ext(le.ON)
        words = {"output": self._bus_words(out, self.output_dtype_width)}
        if self.output_scale_width:
            words["output"] += math.ceil(out / self.scale_block_size)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        oc2 = self._extent(mapping, le.OC, 2)
        for i, (dims, bits) in enumerate(self.tail_specs):
            if le.OC in dims:
                rows = math.prod(ext(dim) for dim in dims if dim != le.OC)
                vectors = rows * math.ceil(ext(le.OC) / oc_dim)
                words[("fused", i)] = self._request_words(
                    vectors, oc_dim, bits, 0, math.ceil(oc2 * bits / 8)
                )
            else:
                words[("fused", i)] = self._bus_words(
                    math.prod(ext(dim) for dim in dims), bits
                )
        return words

    def _stream_switches(self, mapping, key_loops, walk_of, held_loops=()):
        """Compose L1 bank walks in the emitted L2 loop order.

        ``walk_of(idx)`` returns ``(switches, first_bank, last_bank)`` for
        one L1 request nest. Key loops change its addresses; other loops
        repeat it, unless the controller holds the operand across them.
        Fold repeats at their actual nesting depth, including loops between
        two key loops, so both rewinds and inter-block transitions survive.
        """
        blockings, orders = mapping.loop_blockings, mapping.loop_orders
        nest = sorted(
            (
                i
                for i in range(le.NUM)
                if blockings[i][2] > 1 and i not in held_loops
            ),
            key=lambda i: -orders[i][2],
        )

        def walk(depth, idx):
            if depth == len(nest):
                return walk_of(idx)
            loop = nest[depth]
            count = blockings[loop][2]
            if loop not in key_loops:
                inner, start, end = walk(depth + 1, idx)
                return count * inner + (count - 1) * (start != end), start, end
            total = 0
            first = last = None
            for index in range(count):
                idx[loop] = index
                inner, start, end = walk(depth + 1, idx)
                total += inner + (last is not None and start != last)
                if first is None:
                    first = start
                last = end
            return total, first, last

        return walk(0, {})[0]

    def _bank_switch_cycles(self, mapping):
        """Cycles one L3 step's input and weight streams lose to bank
        switches, per role (``BANK_SWITCH_CYCLES`` each).

        Follow the mapping's L1 input order and the weight FY/FX/IC/OC scan.
        Packing combines adjacent channel groups into a request exactly
        when the toolchain permits it, matching _request_words. Buffers
        start on banks; input scales are not included.
        A weight tile held across spatial loops is not refetched by them.
        """
        if not self.bank_size:
            return {}
        b = mapping.loop_blockings
        orders = mapping.loop_orders
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        ic1 = self._extent(mapping, le.IC, 1)
        oc1 = self._extent(mapping, le.OC, 1)
        fy, fx = b[le.FY][1], b[le.FX][1]
        oy1, ox1 = b[le.OY][1], b[le.OX][1]
        hs, ws = self.stride
        y_in = (oy1 * b[le.OY][2] - 1) * hs + fy
        x_in = (ox1 * b[le.OX][2] - 1) * ws + fx
        pitch_in = ic3 * self.input_dtype_width / 8
        ic_dim = mapping.loop_partitionings[le.IC][0]
        oc_dim = mapping.loop_partitionings[le.OC][0]

        def packed_width(lanes, bits, count):
            row_bits = lanes * bits
            factor = math.lcm(row_bits, self.sram_bandwidth) // row_bits
            return lanes * (factor if count and count % factor == 0 else 1)

        input_chunk = packed_width(ic_dim, self.input_dtype_width, b[le.IC][1])
        input_order = sorted((le.IC, le.OY, le.OX), key=lambda i: -orders[i][1])

        def input_walk(idx):
            y0 = idx.get(le.OY, 0) * oy1 * hs
            x0 = idx.get(le.OX, 0) * ox1 * ws
            # Filter taps expand the spatial fetch; the controller disables
            # its separate L1 FX/FY/OC loops. Clip the last halo to the tile.
            sy, sx = (hs if fy == 1 else 1), (ws if fx == 1 else 1)
            ny = min(
                oy1 if fy == 1 else oy1 * hs + fy - 1, (y_in - 1 - y0) // sy + 1
            )
            nx = min(
                ox1 if fx == 1 else ox1 * ws + fx - 1, (x_in - 1 - x0) // sx + 1
            )
            width = input_chunk * self.input_dtype_width / 8
            scans = {
                le.IC: (ic1 // input_chunk, width),
                le.OY: (ny, sy * x_in * pitch_in),
                le.OX: (nx, sx * pitch_in),
            }
            loops = tuple(scans[i] for i in input_order)
            offset = (y0 * x_in + x0) * pitch_in
            offset += idx.get(le.IC, 0) * ic1 * self.input_dtype_width / 8
            return strided_bank_walk(loops, width, self.bank_size, offset)

        pitch_w = oc3 * self.weight_dtype_width / 8
        beat_w = oc1 * self.weight_dtype_width / 8
        weight_chunk = packed_width(
            oc_dim,
            self.weight_dtype_width,
            0 if self.weight_transposed else b[le.OC][1],
        )

        def weight_walk(idx):
            c0 = idx.get(le.IC, 0) * ic1
            k_off = idx.get(le.OC, 0) * beat_w
            width = weight_chunk * self.weight_dtype_width / 8
            loops = (
                (fy * fx, ic3 * pitch_w),
                (ic1, pitch_w),
                (oc1 // weight_chunk, width),
            )
            return strided_bank_walk(
                loops, width, self.bank_size, c0 * pitch_w + k_off
            )

        held = ()
        if b[le.IC][2] == 1:
            held = tuple(
                loop
                for loop in (le.OX, le.OY)
                if orders[loop][2] < orders[le.OC][2]
            )
        return {
            "input": BANK_SWITCH_CYCLES
            * self._stream_switches(mapping, (le.OX, le.OY, le.IC), input_walk),
            "weight": BANK_SWITCH_CYCLES
            * self._stream_switches(mapping, (le.OC, le.IC), weight_walk, held),
        }

    def _tail_bank_switch_cycles(self, mapping, tiled):
        """Fused-operand read latency, charged only on a finishing pass.

        A riding tail uses MatrixOps' filtered L2/L1 output loops. A
        separate vector pass scans the finished output tile in storage
        order. Broadcast dimensions have zero address stride. Multi-beat
        requests overlap part of the bank-switch drain (two beats lose
        seven cycles, versus eight for one), as in pool_bank_switch_cycles.
        """
        if not self.bank_size:
            return {}
        b, orders = mapping.loop_blockings, mapping.loop_orders
        dims_out = (le.ON, le.OY, le.OX, le.OC)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        l1_order = sorted(dims_out, key=lambda i: -orders[i][1])
        result = {}
        for i, (dims, bits) in enumerate(self.tail_specs):
            strides = {}
            pitch = bits / 8
            for dim in reversed(dims_out):
                strides[dim] = pitch if dim in dims else 0
                if dim in dims:
                    pitch *= self._extent(mapping, dim, 2)
            lanes = oc_dim if le.OC in dims else 1
            width = math.ceil(lanes * bits / 8)

            def tile_walk(idx):
                offset = sum(
                    idx.get(d, 0) * self._extent(mapping, d, 1) * strides[d]
                    for d in dims_out
                )
                loops = tuple(
                    (b[d][1], strides[d] * mapping.loop_partitionings[d][0])
                    for d in l1_order
                )
                return strided_bank_walk(loops, width, self.bank_size, offset)

            if tiled:
                switches = self._stream_switches(
                    mapping, dims, tile_walk, (le.IC, le.FX, le.FY)
                )
            else:
                loops = tuple(
                    (
                        b[d][1] * b[d][2],
                        strides[d] * mapping.loop_partitionings[d][0],
                    )
                    for d in dims_out
                )
                switches = strided_bank_walk(loops, width, self.bank_size)[0]
            beats = math.ceil(width * 8 / self.sram_bandwidth)
            result[("fused", i)] = switches * max(
                1, BANK_SWITCH_CYCLES + 1 - beats
            )
        return result

    def _tail_stall(self, mapping, bank_groups, words, compute, bank):
        """Cycles the array loses to the tail's burst on a bank that feeds
        one of its every-round buffers.

        Those buffers ping-pong per L1 sweep (``blockings[IC][2]`` sweeps
        to a block), so each sweep's fetch must land inside the sweep before
        it.  The tail's words on such a bank do not spread over the block:
        they arrive together, and the bank's round-robin port lets them
        through at the tail's beats per request round for every grant of a
        stream that is always pending, or in the sweep's free cycles when
        the stream idles between its requests.  The sweeps the burst lands
        on are priced one by one, and what they cost beyond the block's
        price is the stall.  A tail buffer meets the stream's bank only on
        the steps whose ping-pong slots agree: every step when the stream
        is refetched with it, else every other.
        """
        if not bank_groups:
            return 0
        sweeps = mapping.loop_blockings[le.IC][2]
        steps = self._l3_blocks(mapping)
        if sweeps == 1 or steps == 1:
            return 0
        sweep_compute = compute / sweeps
        vectors = 1
        for loop in [le.OC, le.OY, le.OX]:
            vectors *= mapping.loop_blockings[loop][1]
        stream_dims = {
            "input": _IF_DIMS,
            "input_scale": _IF_DIMS,
            "weight": _FL_DIMS,
            "weight_scale": _FL_DIMS,
        }
        stall = 0.0
        for roles in bank_groups:
            streams = [role for role in roles if role in stream_dims]
            tails = []
            for role in roles:
                is_fused = isinstance(role, tuple) and role[0] == "fused"
                if (role == "output" or is_fused) and words.get(role, 0):
                    tails.append(role)
            if not streams or not tails:
                continue
            others = [
                role
                for role in roles
                if role not in streams and role not in tails
            ]
            stream_words = [words.get(role, 0) / sweeps for role in streams]
            busiest = max(stream_words)
            pending = sum(stream_words)
            spread = sum(words.get(role, 0) for role in others) / sweeps
            tail_words = sum(words[role] for role in tails)
            # The output's beats and its scales leave on two requesters,
            # every beat and every scale a request of its own; a fused
            # operand is fetched once per output vector.
            oc_dim = mapping.loop_partitionings[le.OC][0]
            requests = []
            for role in tails:
                if role == "output":
                    out = vectors * oc_dim
                    stores = self._bus_words(out, self.output_dtype_width)
                    requests.append(stores)
                    if self.output_scale_width:
                        requests.append(math.ceil(out / self.scale_block_size))
                elif le.OC in self.tail_specs[role[1]][0]:
                    requests.append(vectors)
                else:
                    requests.append(words[role])
            beats_per_grant = tail_words / max(requests)
            remaining = tail_words
            priced = 0.0
            for sweep in range(sweeps):
                free = max(0.0, sweep_compute - pending - spread)
                landed = min(remaining, max(beats_per_grant * busiest, free))
                if sweep == sweeps - 1:
                    landed = remaining
                remaining -= landed
                priced += max(sweep_compute, pending + spread + landed)
            refetched = [
                self._l3_loads(mapping, stream_dims[role]) == steps
                for role in streams
            ]
            fraction = 1.0 if any(refetched) else 0.5
            excess = max(0.0, priced - max(compute, bank))
            stall = max(stall, fraction * excess)
        return stall

    def matrix_cycles(self, mapping, bank_groups):
        """Cycles of one L3 grid step: the L2 sweep of weight-reuse tiles,
        each costing the busier of the matrix unit -- its systolic passes,
        plus the back-pressure a single-buffered accumulator takes while the
        tail drains each finished tile -- and the scratchpad bank with the
        most to move for it -- the every-round operands of its L1 sub-tiles,
        the accumulator read back and rewritten while the reduction is
        split, the finished tile and the tail's operands when it is not,
        the round trips a stream idles for when it changes bank, each
        summed with whatever shares its bank -- plus the stall the tail's
        burst inflicts where it shares a bank with one of the array's
        every-round buffers (``_tail_stall``) -- and, for an outlier GEMM,
        the SpMM unit, which must deliver the block's sparse correction
        before the vector pipeline releases any of its rows: per 64-column
        pass it walks every row of the block, paying ``SPMM_ROW_CYCLES`` of
        turnaround plus that row's outliers, each gather a bank switch when
        the weight tile spans several banks -- plus the once-per-sweep
        overhead (buffer fill, systolic skew, the last parked tile's drain)
        spread over the ops a double-buffered L2 overlaps it with.  Also
        the reporting model's per-tile utilization denominator.

        Args:
            mapping: The interstellar mapping to price.
            bank_groups: Its bank partition as role sets (``bank_partition``),
                or ``None`` when nothing shares a bank.
        """
        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing ---
        sa_weight_loading_time = partitionings[le.IC][0]

        first_non_ox_oy_index = 6
        for i in range(le.NUM):
            if i == le.OX or i == le.OY:
                continue
            if orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = orders[i][1]

        weight_reuse_tile_size = 1
        for i in range(le.NUM):
            if orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= blockings[i][1]
        weight_reuse_tile_time = max(
            sa_weight_loading_time, weight_reuse_tile_size
        )

        num_remaining_l1_tiles = 1
        for i in range(le.NUM):
            if orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= blockings[i][1]
        num_remaining_l1_tiles *= blockings[le.IC][2]
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        # --- the finished L1 output tile: its vectors, and the bus beats the
        # tail spends on each -- storing it, and reading a fused operand
        # alongside when one is wider than the port ---
        num_k = blockings[le.IC][3]
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        oc_dim = partitionings[le.OC][0]
        output_width = (
            self.accum_dtype_width if num_k > 1 else self.output_dtype_width
        )
        store_cycles = math.ceil(output_width * oc_dim / self.sram_bandwidth)
        if num_k == 1 and self.output_scale_width:
            # The block scales leave on a requester of their own, one beat
            # per vector, on the output's bank.
            store_cycles += 1
        vector_beats = store_cycles
        if num_k == 1 and not self.single_k_tail_extra_pass:
            for dims, bits in self.tail_specs:
                vector_beats = max(
                    vector_beats,
                    math.ceil(
                        bits
                        * (oc_dim if le.OC in dims else 1)
                        / self.sram_bandwidth
                    ),
                )

        # Without a bank to park the finished tile in, its vectors leave the
        # array one per step during the last reduction pass of each weight
        # tile -- as one burst of the whole tile when the OC passes are
        # adjacent (no filter loops at L1), else as OC1 bursts of one spatial
        # tile -- and the tail drains them at ``vector_beats`` apiece.  The
        # path absorbs ``OUTPUT_SLACK`` of them; past that the array runs at
        # the tail's pace for the rest of the burst.  A double-buffered
        # accumulator parks a tile whose tail moves more than a bus word per
        # vector on some port (the toolchain's ``should_use_direct_path``);
        # a narrower tail rides the pass and pays like a single-buffered one.
        parked = self.double_buffered_accum_buffer
        if parked:
            widths = [output_width * oc_dim]
            for dims, bits in self.tail_specs:
                widths.append(bits * (oc_dim if le.OC in dims else 1))
            parked = max(widths) > self.sram_bandwidth
        if not parked:
            burst_vectors = weight_reuse_tile_size
            burst_cycles = weight_reuse_tile_time
            bursts = blockings[le.OC][1]
            if blockings[le.FX][1] * blockings[le.FY][1] == 1:
                burst_vectors *= bursts
                burst_cycles *= bursts
                bursts = 1
            computation_l1_time += bursts * max(
                0, burst_vectors * vector_beats - burst_cycles - OUTPUT_SLACK
            )

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        # --- bus traffic of one L2 output block: the loads of its L1
        # sub-tiles, then what the vector unit moves for the block itself ---
        loads = self._load_words(mapping)
        words = {
            role: count * blockings[le.IC][2] for role, count in loads.items()
        }
        # With the whole reduction inside the block, a weight tile whose L2
        # loop is outside the spatial ones is fetched once and held across
        # them (the input is refetched every block), so its words are spread
        # over the blocks that reuse it.
        if blockings[le.IC][2] == 1:
            held = 1
            for loop in [le.OX, le.OY]:
                if orders[loop][2] < orders[le.OC][2]:
                    held *= blockings[loop][2]
            words["weight"] /= held
            words["weight_scale"] /= held
        # The bias is read once per output tile: spread over its rounds.
        words["bias"] = (
            self._bus_words(self._extent(mapping, le.OC, 1), self.bias_width)
            / num_k
        )
        output_elems = 1
        for loop in [le.OC, le.OY, le.OX, le.ON]:
            output_elems *= self._extent(mapping, loop, 1)
        if num_k > 1:
            # A split reduction reads the running partial back through the
            # same single-ported bank it writes the new one to.
            words["scratch"] = 2 * self._bus_words(
                output_elems, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass and not self.tail_keeps_shape:
            # A staged single round parks the finished tile in scratch for
            # the tail's own pass (``vector_cycles``) to read back.
            words["scratch"] = self._bus_words(
                output_elems, self.output_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # An in-place pass reads the tile back from its output slot and
            # rewrites it: two bank visits beyond a riding tail's.
            words.update(self._tail_words(mapping, 1))
            words["output"] += 2 * self._bus_words(
                output_elems, self.output_dtype_width
            )
        else:
            words.update(self._tail_words(mapping, 1))
        # A stream that changes bank idles its port for a round trip each
        # time; spread the step's switches over its output blocks.
        switches = self._bank_switch_cycles(mapping)
        if num_k == 1 and (
            not self.single_k_tail_extra_pass or self.tail_keeps_shape
        ):
            switches.update(
                self._tail_bank_switch_cycles(
                    mapping, tiled=not self.single_k_tail_extra_pass
                )
            )
        for role, cycles in switches.items():
            words[role] += cycles / l2_blocks
        # The matrix unit and the vector unit are pipelined -- one drains a
        # tile while the other computes the next -- so a block costs the
        # busier of the two.
        bank = self._bank_cycles(words, bank_groups)
        block_time = max(computation_l1_time, bank)
        # The tail's words do not spread over the block: they land as a
        # burst on a few of its sweeps, and on a bank that feeds one of the
        # array's ping-pong buffers they can outrun the sweeps they land on.
        block_time += self._tail_stall(
            mapping, bank_groups, words, computation_l1_time, bank
        )

        # The SpMM unit runs the block alongside and the vector pipeline
        # waits for its correction on every output vector, so a block also
        # costs its pace: per PE-array-wide pass, every row's turnaround
        # plus its gathered weight rows.
        if self.outlier_rate:
            rows = 1
            for loop in [le.OX, le.OY, le.ON]:
                rows *= self._extent(mapping, loop, 1)
            k_block = self._extent(mapping, le.IC, 2)
            passes = blockings[le.OC][1]
            visits = passes * rows
            gathers = visits * k_block * self.outlier_rate
            # Gathers hit random K rows of the weight tile; when it spans
            # several banks most consecutive gathers change bank and pay the
            # read path's round trip, as the streams above do.
            switch = 0.0
            if self.bank_size:
                weight_tile_bytes = (
                    self._extent(mapping, le.OC, 2)
                    * k_block
                    * blockings[le.FX][1]
                    * blockings[le.FY][1]
                    * self.weight_dtype_width
                    / 8
                )
                banks = max(1, math.ceil(weight_tile_bytes / self.bank_size))
                switch = BANK_SWITCH_CYCLES * (1 - 1 / banks)
            spmm_block_time = visits * SPMM_ROW_CYCLES + gathers * (1 + switch)
            block_time = max(block_time, spmm_block_time)

        # The first tile's loads overlap nothing; the last parked tile's drain
        # is a whole vector pass, while a single-buffered accumulator's drain
        # is already in its blocks' own time.
        buffer_fill = self._bank_cycles(loads, bank_groups)
        skew = partitionings[le.IC][0] + partitionings[le.OC][0] - 2
        drain = output_size * vector_beats if parked else 0
        overhead = buffer_fill + skew + drain
        steady = l2_blocks * block_time

        if not self.double_buffered_l2:
            return steady + overhead

        # Every op the sweep dispatches -- the L3 steps and the batch elements
        # looped outside the mapping -- overlaps the ramps of its neighbours.
        steps = self._l3_blocks(mapping) if num_k == 1 else num_k
        return steady + overhead / (steps * self.batch)

    def vector_cycles(self, mapping, bank_groups):
        """Vector-unit cycles to finish one L3 output tile.  Charged on the grid
        step that ends a K sweep, after that step's accumulation.

        The busier of the unit's own rate -- one lane group per cycle, sized
        by the widest element the tail touches (the partial sum it reads, not
        the narrower value a ``quantize_mx`` writes) at ``dram_bandwidth``,
        which is what ``vector_op_utilization`` charges for the same tail in
        the reporting model -- and the busiest bank: the accumulator read
        back, the finished tile and its scales written, each tail operand
        read, summed wherever the partition puts them together.
        """
        blockings = mapping.loop_blockings
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1] * blockings[loop][2]
        oc_dim = mapping.loop_partitionings[le.OC][0]
        widths = [self.output_dtype_width, self.accum_dtype_width]
        widths += [bits for _, bits in self.tail_specs]
        lane_bytes = max(widths) / 8 * oc_dim
        lanes = output_size * math.ceil(lane_bytes / self.dram_bandwidth)
        words = self._tail_words(mapping, 2)
        for role, cycles in self._tail_bank_switch_cycles(
            mapping, tiled=False
        ).items():
            words[role] += cycles
        if blockings[le.IC][3] > 1 or (
            self.single_k_tail_extra_pass and not self.tail_keeps_shape
        ):
            words["scratch"] = self._bus_words(
                output_size * oc_dim, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # The in-place pass reads the finished tile from its output slot.
            words["output"] += self._bus_words(
                output_size * oc_dim, self.output_dtype_width
            )
        return max(lanes, self._bank_cycles(words, bank_groups))

    @staticmethod
    def _l3_blocks(mapping):
        """Total L3 (DRAM) grid steps, the IC reduction included: with IC
        innermost at L3 the grid is ``(output tiles) x num_k``, one input and
        weight load each.  Stores are ``num_k`` times fewer.
        """
        blockings = mapping.loop_blockings
        l3_blocks = 1
        for i in range(le.NUM):
            l3_blocks *= blockings[i][3]
        return l3_blocks

    @staticmethod
    def _l3_loads(mapping, dims):
        """How many times an operand spanning ``dims`` is fetched over the
        sweep.

        Order the nest outermost to innermost and let ``p`` be the position of
        the innermost loop the operand spans.  Every loop inside ``p`` re-reads
        the tile that is already there, so the operand is fetched once per
        iteration of the loops at or outside ``p``.  Ranks come off the mapping
        (``loop_orders[d][3]``, 0 = innermost), the same order the builders
        emit, so the two cannot disagree.

        A loop that is empty at L3 carries the sentinel rank and a blocking of
        1, so it can only ever multiply in as 1 -- including the case where the
        operand spans nothing tiled, which correctly gives a single fetch.
        """
        orders, blockings = mapping.loop_orders, mapping.loop_blockings
        innermost = min(orders[d][3] for d in dims)
        steps = 1
        for d in range(le.NUM):
            if orders[d][3] >= innermost:
                steps *= blockings[d][3]
        return steps

    def _batch_loads(self, mapping, dims, distinct):
        """Fetches of an operand spanning ``dims`` over the whole sweep, the
        batch loop the builder wraps the mapping in included.

        Diced inside a batch step, the operand re-reads in full on every one:
        the next step restarts the sequence, so its first block differs from
        the last one loaded and the guard never fires.  Held whole, the tile
        survives into the next step and only a change of block costs --
        ``distinct`` of them over the batch, which is fewer than ``batch``
        for an operand a group shares and 1 for one they all share.
        """
        per_step = self._l3_loads(mapping, dims) if dims else 1
        return per_step * (self.batch if per_step > 1 else distinct)

    def calculate_runtime(self, architecture, layer, mapping):
        if self.input_buffer_size is not None:
            _, reasons = input_buffer_usage(
                mapping, self.stride, self.input_buffer_size
            )
            if reasons:
                return math.inf
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings

        # Elements of one L3 tile: levels 0-2 only, since [3] is the grid trip
        # count, not part of the tile.
        input_elems = interstellar.cost_model.get_if_bank_size(
            [self._extent(mapping, loop, 2) for loop in range(le.NUM)], layer
        )
        weight_elems = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.FY][1]
            * blockings[le.FX][1]
        )
        output_elems = (
            partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
        )

        lat = self.dram_access_latency_cycles

        def transfer(*sizes):
            """Cycles to move each of ``sizes`` as its own DMA: one fixed
            access latency apiece plus the bytes.  A microscaling operand's
            block scales are such a DMA -- a few hundred bytes, a whole
            latency."""
            sizes = [s for s in sizes if s]
            return len(sizes) * lat + sum(sizes) / self.dram_bandwidth

        input_sizes = (
            input_elems * self.input_dtype_width / 8,
            input_elems / self.scale_block_size * self.input_scale_width / 8,
        )
        weight_sizes = (
            weight_elems * self.weight_dtype_width / 8,
            weight_elems / self.scale_block_size * self.weight_scale_width / 8,
        )
        output_sizes = (
            output_elems * self.output_dtype_width / 8,
            output_elems / self.scale_block_size * self.output_scale_width / 8,
        )
        input_load = transfer(*input_sizes)
        weight_load = transfer(*weight_sizes)
        store = transfer(*output_sizes)

        # A tail operand spans output dims alone, so its count already runs
        # over the output steps -- the only ones that read it.
        tail_sizes = self.tail_tile_sizes(mapping)
        tail_dmas = [
            (
                transfer(size),
                self._batch_loads(
                    mapping, dims, self.batch if le.ON in dims else 1
                ),
            )
            for (dims, _), size in zip(self.tail_specs, tail_sizes)
        ]

        bank_groups, scratch_slots = bank_partition(
            architecture, layer.size_fn, layer, mapping
        )
        matrix_cycles = self.matrix_cycles(mapping, bank_groups)
        vector_cycles = (
            self.vector_cycles(mapping, bank_groups) if self.has_tail else 0
        )

        input_steps = self._batch_loads(mapping, _IF_DIMS, self.batch)
        weight_steps = self._batch_loads(mapping, _FL_DIMS, self.weight_batch)

        # The mapping covers one batch element; the builder loops the rest.
        l3_blocks = self._l3_blocks(mapping) * self.batch
        num_k = blockings[le.IC][3]
        output_tiles = l3_blocks // num_k

        # Traffic the sweep moves, for a caller ranking by DRAM rather than by
        # time, and to check the reuse counts against a profile.  Every
        # candidate mapping is priced through here, so it describes the last
        # one scored -- read it straight after the call that priced the mapping
        # in question.
        self.dram_bytes = {
            "input": input_steps * sum(input_sizes),
            "weight": weight_steps * sum(weight_sizes),
            "output": output_tiles * sum(output_sizes),
            "tail": sum(t * s for (_, t), s in zip(tail_dmas, tail_sizes)),
        }

        dmas = [
            (store, output_tiles),
            (input_load, input_steps),
            (weight_load, weight_steps),
            *tail_dmas,
        ]

        if not self.double_buffered_l2:
            total_time = l3_blocks * matrix_cycles + sum(t * c for c, t in dmas)
            if num_k > 1 or self.single_k_tail_extra_pass:
                total_time += output_tiles * vector_cycles
            return total_time

        if num_k == 1:
            # Every step finishes a tile: one schedule covers the sweep.  A
            # riding tail drains inside the matrix pass, and an in-place one
            # overlaps the next tile's (its bank words are in the block);
            # a staged one is a pass of its own, serial on the single
            # scratch region.
            step = matrix_cycles
            if self.single_k_tail_extra_pass and not self.tail_keeps_shape:
                step += vector_cycles
            return _sweep_cycles(dmas, l3_blocks, step)

        load = input_load + weight_load
        # The sweep's last step has no tile after it to prefetch, so it costs
        # compute alone: hold it out of the count and let the epilogue charge
        # it, with the tail and store that drain behind it.
        accum_steps = l3_blocks - 2 * (output_tiles - 1) - 1
        classes = _step_classes(tail_dmas, output_tiles)
        # Hold out the first output step in the same way -- the prologue fetches
        # its tail, with nothing running yet to hide it behind.  Taking what
        # that step owed leaves every tail fetch counted exactly once.
        first_tail = classes[-1][0]
        classes[-1] = (classes[-1][0], classes[-1][1] - 1)

        total_time = load + first_tail + accum_steps * max(load, matrix_cycles)
        # One window per remaining tile, spanning two grid steps: the matrix
        # unit finishes this tile and starts the next, while DRAM fits that
        # tile's tail read, its store and the next prefetch into the same span.
        # The busier side sets the price, and only the tail differs from one
        # window to the next -- hence one price per class.
        for tail, count in classes:
            prefetch = load + tail
            if self.split_k_tail_extra_pass and scratch_slots == 1:
                # The bare pass holds the control stream through both the
                # matrix and the vector pass, so the window's loads are
                # issued only then and nothing hides them.
                total_time += count * (
                    matrix_cycles + vector_cycles + prefetch + matrix_cycles
                )
                continue
            compute = max(matrix_cycles, prefetch) + matrix_cycles
            dma = max(matrix_cycles + vector_cycles, prefetch) + store + load
            total_time += count * max(compute, dma)
        total_time += matrix_cycles + vector_cycles + store
        return total_time


def bank_partition(architecture, size_fn, layer, mapping):
    """The role partition ``mapping``'s L2 fit is checked with.

    Rebuilds interstellar's scratchpad-level ``size_fn`` invocation
    (``cost_model.get_block_size``): the element counts from the blocking /
    partitioning products through L2, the bank geometry from the
    architecture -- and replays ``size_fn``'s own group construction, so the
    partition is exactly the one the fit check priced.  The runtime model
    prices every candidate mapping through it, and the winner's partition is
    stamped for the memory planner and the reporting model.

    Returns:
        ``(partition, scratch_slots)``: a list of role sets, one per bank
        group -- plus a ``{"scratch"}`` entry when the search charged the
        reduction scratch its own regions -- and how many regions it
        charged, the slot count the scratch is allocated with.  The
        partition is ``None`` when the architecture has no banked level or
        there is no ``size_fn`` (nothing checked the fit, so nothing shares
        a bank).
    """
    if size_fn is None:
        return None, 1
    level = 2
    buf = architecture.buffer(level)
    bank_size = buf.bank_size
    if not bank_size:
        return None, 1
    capacity = buf.capacity
    if isinstance(capacity, list):
        capacity = capacity[0]
    num_banks = capacity // bank_size

    blocking_accum = []
    partitioning_accum = []
    for i in range(le.NUM):
        blocking_accum.append(math.prod(mapping.loop_blocking(i)[: level + 1]))
        partitioning_accum.append(
            math.prod(mapping.loop_partitioning(i)[: level + 1])
        )
    partitioning = list(zip(*mapping.loop_partitionings))[level]

    cost_model = interstellar.cost_model
    counts = (
        cost_model.get_if_size(
            blocking_accum, partitioning_accum, partitioning, layer
        ),
        cost_model.get_of_size(
            blocking_accum, partitioning_accum, partitioning
        ),
        cost_model.get_fl_size(
            blocking_accum, partitioning_accum, partitioning
        ),
    )
    groups, scratch, regions = size_fn.compute_groups(
        counts, mapping, level, partitioning_accum, bank_size, num_banks
    )
    if groups is None:
        return None, 1
    partition = [roles for _, _, _, roles in groups]
    if scratch:
        partition.append({"scratch"})
    return partition, regions
