"""Shared SRAM, vector, and DRAM runtime accounting for matrix mappings."""

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

# The L3 loops each operand's tile spans.  A tile advances only when one of
# them turns, so an operand whose every entry is a single L3 step reads one
# tile for the whole sweep.
_IF_DIMS = (le.OX, le.OY, le.IC, le.ON)
_FL_DIMS = (le.OC, le.IC, le.FX, le.FY)
_OF_DIMS = (le.OC, le.OY, le.OX, le.ON)


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


class BaseRuntimeCalculator:
    """Common memory and vector costs, with matrix compute supplied by a backend."""

    # A backend whose cost models cannot tell some loop orders apart gives
    # ``order_key(level, order, mapping)`` (see interstellar's
    # ``opt_get_loop_order_generator``); None searches every order.
    order_key = None

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

        bank_groups, scratch_slots = self.bank_partition(
            architecture, layer, mapping
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

    def bank_partition(self, architecture, layer, mapping):
        """``bank_partition`` of a scored mapping.  It reads no loop order, so
        a backend may reuse it across the orders of one blocking."""
        return bank_partition(architecture, layer.size_fn, layer, mapping)

    def matrix_cycles(self, mapping, bank_groups):
        """Backend compute cost for one L3 grid step."""
        raise NotImplementedError
