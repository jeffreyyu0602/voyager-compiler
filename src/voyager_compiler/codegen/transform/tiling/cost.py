"""DRAM-aware latency and traffic model for vector-unit L2 tiling.

``vector_tile_latency`` and ``gemv_tile_latency`` score one candidate tiling of
an op that runs on the vector unit -- an elementwise / reduction op, or a
matrix-vector GEMM -- so ``_search_tiling`` can rank the tiles that fit rather
than take the largest.  Both cost the schedule ``PipelinedKernel`` emits, via
``_sweep_cycles``.  ``attention_tile_latency`` scores a flash-attention tiling
the same way, against the FA3 schedule instead.

Kept independent of the ``reporting`` package on purpose: the tiling pass runs
inside ``transform()``, long before any reporting stage exists.  The dependency
runs the other way -- ``reporting/cost.op_utilization`` imports
``vector_op_utilization`` from here for its vector branch, so this is the single
copy of that formula (and of ``OP_PASSES``).
"""

import functools
import itertools
import math
from typing import Optional

import torch
from torch.fx import Node

import interstellar
from voyager_compiler.codegen.node_info import (
    _pair,
    bound_operands,
    dtype_byte_size,
    get_anchor_node,
    get_arg_value,
    get_node_to_key_map,
    is_fully_connected,
    is_gemm_op,
    is_pooling,
    require_allocation,
    weight_transforms,
)
from voyager_compiler.ops.layout import (
    NCHW_TO_NHWC,
    NHWC_OP_VARIANTS,
    unproject,
)

le = interstellar.le

# Passes an op makes over its data, each as what it streams in and what it
# writes per element: IN the op's input, MID the intermediate a normalization
# stages between passes, OUT its output, None a reduction that writes one
# value per row.  Single source of truth: reporting/cost.py imports this via
# ``vector_op_utilization``.
IN, MID, OUT = "in", "mid", "out"
_LAYER_NORM_PASSES = [(IN, None), (IN, None), (IN, MID), (MID, OUT)]
_SOFTMAX_PASSES = [(IN, None), (IN, None), (IN, OUT)]
OP_PASSES = {
    torch.ops.aten.layer_norm.default: _LAYER_NORM_PASSES,
    torch.ops.aten.softmax.int: _SOFTMAX_PASSES,
    torch.ops.quantized_ops.layer_norm.default: _LAYER_NORM_PASSES,
    torch.ops.quantized_ops.softmax.default: _SOFTMAX_PASSES,
}

# Cycles a pass costs the vector unit whatever it holds: params, instruction
# issue and pipeline fill/drain.  Fixed per pass, so a tile too small to
# amortise it is dominated by overhead.  96 on every layer_norm and softmax
# pass the Sphinx RTL measured.
KERNEL_LAUNCH_OVERHEAD = 96

# Cycles a read stream loses each time consecutive requests move to another
# scratchpad bank.  The SoC's read masters take their responses in order, so
# the fabric holds a request to a new bank until every outstanding response
# from the previous one has returned -- one round trip, measured 8 cycles on
# the Sphinx SoC (requests resume once the in-flight beats drain).
BANK_SWITCH_CYCLES = 8


def get_dtype_width(dtype) -> int:
    """Element width in bits, derived from the canonical ``dtype_byte_size``
    so the dtype-name parsing lives in exactly one place."""
    return round(dtype_byte_size(dtype) * 8)


def _node_dtype_bits(node, default: Optional[int] = None):
    """Element width in bits of ``node``'s tensor, read from the graph.

    Prefers the compiler's tracked storage dtype (``meta['dtype']``, e.g. an
    NF4 weight) over the traced tensor dtype.  A multi-output node --
    ``quantize_mx`` gives ``(scale, quantized)`` -- carries one tracked dtype
    per output and gives one width per output in that order; every other node
    gives a single width.

    Args:
        node: The FX node to size, or ``None`` for an absent operand.
        default: Width to return when there is no node to size; omit to raise.

    Returns:
        The element width in bits, or one per output for a multi-output node.

    Raises:
        ValueError: If ``node`` is not an FX node and ``default`` is ``None``.
    """
    if not isinstance(node, Node):
        if default is None:
            raise ValueError(f"node {node} has no dtype to size the operand")
        return default

    dtype = node.meta.get("dtype")
    value = getattr(node, "value", None)
    if value is None:
        value = node.meta.get("val")

    if isinstance(dtype, (list, tuple)):
        # One tracked dtype per output; a ``None`` entry is an output left at
        # the dtype it was traced with.
        return [
            get_dtype_width(d if d is not None else v.dtype)
            for d, v in zip(dtype, value)
        ]
    if dtype is not None:
        return get_dtype_width(dtype)
    if isinstance(value, (list, tuple)):
        return [get_dtype_width(v.dtype) for v in value]
    return get_dtype_width(value.dtype)


def _record_writes(record_bits: int, beat: int) -> float:
    """Write cycles per record of ``record_bits`` on a ``beat``-byte bus,
    averaged over the alignment cycle: one per beat-sized word, two where
    the record starts mid-beat and the bus splits the word."""
    nbytes = math.ceil(record_bits / 8)
    words = math.ceil(nbytes / beat)
    period = math.lcm(nbytes, beat) // nbytes
    split = sum(1 for g in range(period) if (g * nbytes) % beat)
    return words * (period + split) / period


def _input_bits(node, anchor) -> int:
    """Element width of the buffer the anchor's primary input is fetched
    from: through a fused dequantize prologue to the operand the call binds,
    so a packed input is sized as loaded, not as decoded."""
    src = anchor.args[0]
    if src.target is torch.ops.quantized_ops.dequantize.default:
        src = src.args[0]
    submodule = node.meta.get("submodule")
    if submodule is not None and src.graph is submodule.graph:
        src = bound_operands(node, submodule).get(src, src)
    bits = _node_dtype_bits(src)
    return max(bits) if isinstance(bits, list) else bits


def _output_writes(node, lanes: int, beat: int) -> float:
    """Write cycles per lane group of ``node``'s outputs: the data output's
    record, plus one per block scale a ``quantize_mx`` tail writes beside
    it."""
    widths = _node_dtype_bits(node)
    if not isinstance(widths, list):
        return _record_writes(lanes * widths, beat)
    values = getattr(node, "value", None)
    if values is None:
        values = node.meta.get("val")
    data = max(range(len(widths)), key=lambda i: values[i].numel())
    return _record_writes(lanes * widths[data], beat) + len(widths) - 1


def pool_taps(anchor) -> int:
    """Input lane groups a pool fetches per output group: its window's taps
    (test/toolchain/Pooling.h runs the fetch reduce_count = kH * kW times per
    output).  An adaptive pool's window is its input over its output."""
    if "adaptive" in str(anchor.target):
        return math.prod(anchor.args[0].shape) // math.prod(anchor.shape)
    kernel = get_arg_value(anchor, 1, "kernel_size")
    kh, kw = (kernel, kernel) if isinstance(kernel, int) else kernel
    return kh * kw


def _pool_window(anchor, in_hw, out_hw):
    """``(kernel, stride, dilation)`` pairs of a pool whose input and output
    span ``in_hw`` and ``out_hw``.  An adaptive pool's window and stride are
    its input extent over its output's."""
    if "adaptive" in str(anchor.target):
        kernel = (in_hw[0] // out_hw[0], in_hw[1] // out_hw[1])
        return kernel, kernel, (1, 1)
    kernel = _pair(get_arg_value(anchor, 1, "kernel_size"))
    stride = get_arg_value(anchor, 2, "stride", [])
    stride = _pair(stride) if stride else kernel
    dilation = (1, 1)
    if "max_pool" in str(anchor.target):
        dilation = _pair(get_arg_value(anchor, 4, "dilation", 1))
    return kernel, stride, dilation


def _element_strides(shape):
    """Element strides of a contiguous tensor of ``shape``, per dimension."""
    strides = []
    for dim in range(len(shape)):
        strides.append(math.prod(shape[dim + 1 :]))
    return tuple(strides)


def _pool_fetch_ranges(tile, strides, out_hw, window, chunk, bits):
    """Byte ranges a pool tile's fetches read, in the order the vector unit
    issues them: output position, then channel chunk, then the window row by
    row (``get_pool2d_tiling`` in test/common/Tiling.cc).  ``tile`` and
    ``strides`` are per logical ``(N, C, H, W)`` axis, the strides in
    elements of the tile's own layout.  A chunk is one read where channels
    are innermost (NHWC, the layout the hardware runs); in any other layout
    its channels are read one by one."""
    batch, channels, _, _ = tile
    s_n, s_c, s_h, s_w = strides
    h_out, w_out = out_hw
    (kh, kw), (sh, sw), (dh, dw) = window
    for n, oy, ox, c0, ky, kx in itertools.product(
        range(batch),
        range(h_out),
        range(w_out),
        range(0, channels, chunk),
        range(kh),
        range(kw),
    ):
        y = oy * sh + ky * dh
        x = ox * sw + kx * dw
        first = n * s_n + c0 * s_c + y * s_h + x * s_w
        width = min(chunk, channels - c0)
        if s_c == 1:
            reads = [(first, width)]
        else:
            reads = [(first + c * s_c, 1) for c in range(width)]
        for element, count in reads:
            lo = element * bits // 8
            yield lo, lo + math.ceil(count * bits / 8)


@functools.lru_cache(maxsize=None)
def _pool_bank_switches(tile, strides, out_hw, window, chunk, bits, bank_size):
    """Bank switches of ``_pool_fetch_ranges``, cached: the tiling search
    and the report price the same tile geometries again and again."""
    ranges = _pool_fetch_ranges(tile, strides, out_hw, window, chunk, bits)
    switches, _, _ = bank_walk(ranges, bank_size)
    return switches


def pool_bank_switch_cycles(node, anchor, in_tile, config):
    """Cycles a pool tile loses to scratchpad bank switches.  The vector unit
    fetches ``config.accumulator_lanes`` channels at a time and walks a
    chunk's whole window before the next chunk (``_pool_fetch_ranges``), so a
    window that straddles a bank boundary switches banks twice per chunk.
    ``in_tile`` is the input tile in the op's layout -- NHWC for the layout
    twin, NCHW for the aten op -- and starts on a bank, where the planner
    puts it.  A switch costs the ``BANK_SWITCH_CYCLES`` round trip less the
    fetch's beats after its first, which the drain overlaps: 7 on Sphinx, as
    measured.  0 without banking."""
    if not config.bank_size:
        return 0
    nhwc = anchor.target in NHWC_OP_VARIANTS.values()
    dims = NCHW_TO_NHWC if nhwc else None
    tile = unproject(tuple(in_tile), dims)
    strides = unproject(_element_strides(in_tile), dims)
    out_hw = unproject(tuple(anchor.shape), dims)[2:]
    kernel, stride, dilation = _pool_window(anchor, tile[2:], out_hw)
    h_out = (tile[2] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    w_out = (tile[3] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    bits = _input_bits(node, anchor)
    chunk = config.accumulator_lanes
    switches = _pool_bank_switches(
        tile,
        strides,
        (h_out, w_out),
        (kernel, stride, dilation),
        chunk,
        bits,
        config.bank_size,
    )
    beat = max(1, round(config.bytes_per_cycle))
    fetch_beats = math.ceil(chunk * bits / 8 / beat)
    per_switch = max(1, BANK_SWITCH_CYCLES + 1 - fetch_beats)
    return per_switch * switches


def gemv_weight_bits(node, anchor) -> int:
    """Element width of the buffer a fully-connected GEMM streams its weight
    from (``weight_transforms``): a packed cache reaches the bank packed,
    whatever a fused prologue decodes it into."""
    submodule = node.meta.get("submodule")
    bound = bound_operands(node, submodule)
    weight = anchor.args[1]
    if submodule is not None and weight.graph is submodule.graph:
        weight = weight_transforms(weight)[0]
    bits = _node_dtype_bits(bound.get(weight, weight))
    return max(bits) if isinstance(bits, list) else bits


def _logical_dtype(node):
    """The compiler's tracked storage dtype of ``node`` if it has one, else
    the dtype it was traced with."""
    dtype = node.meta.get("dtype")
    if dtype is not None:
        return dtype
    value = getattr(node, "value", None)
    if value is None:
        value = node.meta.get("val")
    return value.dtype


def gemv_lanes(anchor, config) -> int:
    """Elements per cycle of the unit a fully-connected GEMM streams through:
    the vector unit's lanes when its input is bfloat16 -- the toolchain maps
    such an op onto the vector pipeline -- else the matrix-vector unit's."""
    if str(_logical_dtype(anchor.args[0])).endswith("bfloat16"):
        return config.vector_lanes
    return config.matrix_vector_lanes


def bank_walk(ranges, bank_size):
    """Scratchpad bank switches a read stream makes walking ``ranges``, byte
    ``(lo, hi)`` pairs in fetch order: one whenever a range starts in a bank
    other than the one the previous range ended in, plus the boundaries a
    range straddles.  Returns ``(switches, first bank, last bank)``."""
    switches = 0
    first = last = None
    for lo, hi in ranges:
        lo_bank, hi_bank = int(lo // bank_size), int((hi - 1) // bank_size)
        switches += hi_bank - lo_bank
        if last is not None and lo_bank != last:
            switches += 1
        if first is None:
            first = lo_bank
        last = hi_bank
    return switches, first, last


def gemv_bank_switch_cycles(rows, reduction, weight_bits, chunk, config):
    """Cycles a matrix-vector tile loses to scratchpad bank switches,
    ``BANK_SWITCH_CYCLES`` each.  The unit walks one ``chunk``-element piece
    of every row of a ``pe_array_size[0]``-row block before the next chunk,
    so when the row stride of a ``reduction``-wide tile spreads a block's
    rows over several banks, every chunk column crosses them again.  The
    tile buffer starts on a bank, where the planner puts it.  0 without
    banking."""
    if not config.bank_size:
        return 0
    block = config.pe_array_size[0]
    row_bytes = reduction * weight_bits / 8
    chunk_bytes = chunk * weight_bits / 8
    ranges = []
    for k in range(0, rows, block):
        for c in range(math.ceil(reduction / chunk)):
            for r in range(k, min(k + block, rows)):
                lo = r * row_bytes + c * chunk_bytes
                ranges.append((lo, lo + chunk_bytes))
    switches, _, _ = bank_walk(ranges, config.bank_size)
    return BANK_SWITCH_CYCLES * switches


def vector_op_utilization(node, config, ideal_cycles=None, tile=None):
    """Fraction of peak a vector ``node`` sustains, bound by the bus.

    Peak is one ``config.vector_lanes``-wide lane group per cycle.  Each
    pass over the data (``OP_PASSES``; one for everything else) costs a lane
    group the greater of its read cycles, the bus beats its widest streamed
    operand is fetched in, and its write cycles, at ``config.bytes_per_cycle``
    per beat: a reduction pass writes one value per row, so it runs at the
    read rate, and a record that ends mid-beat is split by the bus, so a
    sub-byte output can take more write cycles than read cycles.  A
    fully-connected GEMM streams its weight once per output through the
    unit ``gemv_lanes`` names, a lane group being that unit's width, and is
    sized by the buffer the kernel loads (``weight_transforms``), not by
    what a fused prologue decodes it into.  A pool fetches one lane group
    per window tap, the taps being its callers' work, and its output leaves
    on a port of its own behind the fetch.  ``tile`` is the tile being
    priced, which ``node`` cannot name while a tiling is still being
    searched, and prices the scratchpad bank switches of the op's stream: a
    fully-connected GEMM's ``(rows, reduction)`` weight tile
    (``gemv_bank_switch_cycles``), a pool's input tile
    (``pool_bank_switch_cycles``); other ops ignore it.  Given
    ``ideal_cycles``, the per-pass ``KERNEL_LAUNCH_OVERHEAD`` and the
    switches are folded in so ``ideal_cycles / result`` is the tile's whole
    cost.  The single copy of the formula:
    ``reporting/cost.op_utilization`` calls it for its vector branch.
    """
    anchor = get_anchor_node(node) or node
    lanes = config.vector_lanes
    bytes_per_cycle = config.bytes_per_cycle
    beat = max(1, round(bytes_per_cycle))
    switch_cycles = 0
    cycles_per_group = 0.0
    if is_fully_connected(anchor):
        lanes = gemv_lanes(anchor, config)
        reads = {IN: gemv_weight_bits(node, anchor)}
        profile = [(IN, None)]
        rows, reduction = tile
        switch_cycles = gemv_bank_switch_cycles(
            rows, reduction, reads[IN], lanes, config
        )
    elif is_pooling(anchor):
        # One lane group fetched per window tap (test/toolchain/Pooling.h);
        # the output is written on another port while the fetch runs.
        profile = []
        cycles_per_group = math.ceil(
            lanes * _input_bits(node, anchor) / 8 / bytes_per_cycle
        )
        if tile is not None:
            switch_cycles = pool_bank_switch_cycles(node, anchor, tile, config)
    elif anchor.target in OP_PASSES:
        profile = OP_PASSES[anchor.target]
        reads = {IN: _input_bits(node, anchor), MID: _node_dtype_bits(anchor)}
    else:
        widths = [
            _node_dtype_bits(n)
            for n in node.all_input_nodes
            if require_allocation(n)
        ]
        reads = {IN: max(widths, default=16)}
        profile = [(IN, OUT)]
    for read, write in profile:
        read_cycles = math.ceil(lanes * reads[read] / 8 / bytes_per_cycle)
        if write is None:
            write_cycles = 0.0
        elif write is MID:
            write_cycles = _record_writes(lanes * reads[MID], beat)
        else:
            write_cycles = _output_writes(node, lanes, beat)
        cycles_per_group += max(read_cycles, write_cycles)
    util = min(1.0, 1.0 / cycles_per_group)
    if not ideal_cycles:
        return util
    overhead = max(1, len(profile)) * KERNEL_LAUNCH_OVERHEAD + switch_cycles
    return ideal_cycles / (ideal_cycles / util + overhead)


def _operand_bytes(shape, node):
    """Physical bytes of a tiled operand.  ``shape`` is a single tile shape, or
    a sequence of per-output shapes for a multi-output node, which then pays for
    each output at its own width."""
    bits = _node_dtype_bits(node)
    if isinstance(bits, int):
        return math.ceil(math.prod(shape) * bits / 8)
    return sum(math.ceil(math.prod(s) * w / 8) for s, w in zip(shape, bits))


def _tile_elems(shape):
    """Element count of a tile shape (largest output of a multi-output node)."""
    if shape and isinstance(shape[0], (tuple, list)):
        return max(math.prod(s) for s in shape)
    return math.prod(shape)


def _transfer_cost(shape, operand_bytes, lat, bpc):
    """Cycles one transfer of a tiled operand costs: an access latency per
    tensor -- a ``quantize_mx`` pair is two, its tile of scales paying a whole
    latency for a few hundred bytes -- plus the bytes at the DRAM rate."""
    multi = bool(shape) and isinstance(shape[0], (tuple, list))
    return (len(shape) if multi else 1) * lat + operand_bytes / bpc


def vector_tile_latency(node, tile_sizes, tiled_shapes, tiling, config):
    """Latency and DRAM traffic of a vector op under a candidate tiling.

    The grid is the tile counts over ``node``'s own output, and each input's
    dims right-align onto it, the way the tile shapes were built
    (``compute_tiled_shape``) -- so a dim an operand lacks, or broadcasts
    over, leaves its tile in place while that loop turns and
    ``_block_transfers`` prices the reuse.  A tile costs its element count
    spread over the lanes and de-rated by ``vector_op_utilization``, sized by
    the larger of the output tile and the widest input tile.  Double-buffered
    the two engines overlap (``_sweep_cycles``); single-buffered load, compute
    and store run back to back.

    Args:
        node: The op to cost -- a vector op, or a fused ``call_module`` around
            one.
        tile_sizes: Unused; part of the ``cost_fn`` contract ``_search_tiling``
            calls through, which ``gemv_tile_latency`` sizes its compute from.
        tiled_shapes: Operand FX node -> tile shape: ``node`` itself plus each
            allocated activation input (resident params are absent).
        tiling: Per-dim tile *count* over ``node``'s output.
        config (AcceleratorConfig): The hardware description.

    Returns:
        ``(cycles, DRAM bytes)``; the bytes break a latency tie in
        ``_search_tiling``.
    """
    num_tiles = math.prod(tiling)

    out_shape = tiled_shapes[node]
    out_bytes = _operand_bytes(out_shape, node)
    out_ndim = len(tiling)

    in_elems = max(
        (_tile_elems(s) for n, s in tiled_shapes.items() if n is not node),
        default=0,
    )
    anchor = get_anchor_node(node) or node
    tile = None
    if is_pooling(anchor):
        tile_ops = _tile_elems(out_shape) * pool_taps(anchor)
        # The halo the kernel loads, keyed the way ``_pool_shapes`` keys it.
        bound = bound_operands(node, node.meta.get("submodule"))
        halo = bound.get(anchor.args[0], anchor.args[0])
        tile = tiled_shapes.get(halo)
    else:
        tile_ops = max(_tile_elems(out_shape), in_elems)
    lanes = config.vector_lanes
    bpc = config.bytes_per_cycle
    ideal = math.ceil(tile_ops / lanes)
    util = vector_op_utilization(node, config, ideal, tile)
    compute = math.ceil(ideal / util)

    lat = config.access_latency_cycles
    # ``_sweep_cycles`` reads the store off the front.  An operand's dims
    # right-align onto the output's, the way the tile shapes were built
    # (``compute_tiled_shape``).
    dmas = [(_transfer_cost(out_shape, out_bytes, lat, bpc), num_tiles)]
    traffic = num_tiles * out_bytes
    for n, shp in tiled_shapes.items():
        if n is node:
            continue
        offset = out_ndim - len(n.shape)
        spans = {
            offset + j: 1
            for j, size in enumerate(n.shape)
            if offset + j >= 0 and size != 1
        }
        transfers = _block_transfers(spans, tiling)
        n_bytes = _operand_bytes(shp, n)
        dmas.append((_transfer_cost(shp, n_bytes, lat, bpc), transfers))
        traffic += transfers * n_bytes

    if config.double_buffered_l2:
        latency = _sweep_cycles(dmas, num_tiles, compute)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += num_tiles * compute
    return latency, traffic


# The trailing grid dims each operand role is diced by, named as the
# interstellar tiler names them (``_GEMM_LOOP``: M is ``OX``, N is ``OC``, and
# the reduction is ``IC``).  ``build_gemm`` loops the batch, then M, then the
# output block, then the reduction innermost -- so a role that misses the
# reduction holds its tile across it, and one that misses the output block as
# well holds it for the entire kernel.  Whatever a fused tail brings of its own
# has no role and is diced by the output block, like the output.
def operand_roles(node) -> dict:
    """Each buffer the kernel loads for an anchor operand -> ``(role,
    operand)``, keyed by the outer FX node the memory planner allocates.

    A fused ``call_module``'s operands take their roles from the anchor, which
    reads them through the prologue's expand / transpose / dequantize; the
    kernel loads the placeholder beneath (``weight_transforms``), bound to an
    outer node (``bound_operands``).  ``operand`` is the anchor-side node,
    which keeps the expand ``_operand_spans`` prices reuse by.  What the tail
    or prologue brings of its own has no role and takes a bank of its own.
    """
    anchor = get_anchor_node(node)
    if not is_gemm_op(anchor):
        return {}
    submod = node.meta.get("submodule")
    bound = bound_operands(node, submod)
    roles = {}
    for n, role in get_node_to_key_map(anchor, bound).items():
        if n is anchor:
            continue
        loaded = n
        if submod is not None and n.graph is submod.graph:
            source = weight_transforms(n)[0]
            loaded = bound.get(source, source)
        roles[loaded] = (role, n)
    # The output is named on the outer node -- what the shape map keys it by --
    # in place of the anchor's own entry, which is inside the submodule.
    roles[node] = ("output", node)
    return roles


_GEMV_GRID_DIMS = {
    "input": (le.OX, le.IC),
    "input_scale": (le.OX, le.IC),
    "weight": (le.OC, le.IC),
    "other": (le.OC, le.IC),
    "weight_scale": (le.OC, le.IC),
    "bias": (le.OC,),
}
_OUTPUT_GRID_DIMS = (le.OX, le.OC)
_GEMV_DIM_POS = {le.OX: 0, le.OC: 1, le.IC: 2}


def _operand_spans(node, dims, nb):
    """Grid dim -> consecutive steps ``node`` holds one tile for, over the dims
    it is diced by.

    Mirrors the specs ``build_gemm`` builds: the operand's own batch dims
    right-align onto the output's, a size-1 one broadcasting (pinned to block
    0, so it is not diced at all) and a repeated one holding its tile for the
    whole group -- eight KV heads reaching an attention matmul as thirty-two
    are addressed as head ``h // 4``, so four consecutive heads read one tile
    and the expand never runs.  ``dims`` names the trailing dims its role is
    diced by.
    """
    shape = tuple(node.shape)
    own = shape[: max(0, len(shape) - 2)]
    repeat = weight_transforms(node)[2]

    spans = {}
    for j, size in enumerate(own):
        g = nb - len(own) + j
        if g >= 0 and size != 1:
            spans[g] = repeat[j] if repeat else 1
    for d in dims:
        spans[nb + _GEMV_DIM_POS[d]] = 1
    return spans


def _block_transfers(spans, grid):
    """DMAs an operand issues over the whole grid.

    Every load is guarded on the tile's block index changing
    (``_BufferedRef.copy_in``), so the operand is fetched once per *run* of
    steps addressing the same tile.  The innermost dim it is diced by sets that
    run: every dim inner to it re-reads the tile, and so does a dim it merely
    repeats over.  Diced by nothing, it is read once and held.
    """
    tiled = [g for g in spans if grid[g] > 1]
    if not tiled:
        return 1
    inner = max(tiled)
    return math.prod(grid[:inner]) * max(1, grid[inner] // spans[inner])


def _step_classes(dmas, steps):
    """``(cycles, count)`` per class of step, grouped by what reloads on it.

    A DMA issued ``t`` times over ``steps`` steps lands every ``steps // t`` of
    them, so what a step owes is whatever happens to recur on it.  Those
    periods nest -- each is a suffix-product of the grid times a repeat on a
    batch dim -- so the longest one dividing a step fixes what it owes and the
    class sizes come out by subtraction.  Non-nested periods (a repeat
    somewhere other than a batch dim) have no such structure and are counted
    directly.

    Cheapest class first, and the classes always cover every step: a DMA that
    recurs on all of them may be absent, and then the steps nothing lands on
    are a class of their own owing nothing.

    Args:
        dmas: ``(cycles, transfers)`` per operand over the whole sweep.
        steps: Steps to spread them over.

    Returns:
        ``(cycles, count)`` pairs whose counts sum to ``steps``.
    """
    by_period = {1: 0.0}
    for cycles, transfers in dmas:
        period = steps // transfers
        by_period[period] = by_period.get(period, 0.0) + cycles

    periods = sorted(by_period)
    if any(b % a for a, b in zip(periods, periods[1:])):
        counts = {}
        for s in range(steps):
            owed = sum(c for p, c in by_period.items() if s % p == 0)
            counts[owed] = counts.get(owed, 0) + 1
        return sorted(counts.items())

    classes, owed = [], 0.0
    for j, period in enumerate(periods):
        owed += by_period[period]
        inner = steps // periods[j + 1] if j + 1 < len(periods) else 0
        classes.append((owed, steps // period - inner))
    return classes


def _sweep_cycles(dmas, steps, compute):
    """Cycles a double-buffered sweep of ``steps`` grid steps takes.

    A guarded load makes the DRAM stream uneven, so ``_step_classes`` sorts the
    steps by what reloads on each.  The two engines run concurrently, so a step
    costs whichever of them is busier.  Comparing the *totals* instead would
    let the DRAM idle of a cheap step pay for the load of an expensive one, and
    no amount of buffering can move time backwards like that.

    A tile is loaded while its predecessor computes, so the two engines are
    offset by a step and the ends do not line up: the first load overlaps
    nothing, the step computing tile 0 has no store to make yet, the one
    computing the last tile has no load left to issue, and the last store
    waits on it.  Those four are priced on their own, in place of the two
    steps the per-step charge would have spent on them.

    Args:
        dmas: ``(cycles, transfers)`` per operand over the whole grid, the
            output store first.
        steps: Grid steps the sweep runs.
        compute: Cycles one tile costs the compute engine.

    Returns:
        Cycles the sweep takes, the exposed ends included.
    """
    store = dmas[0][0]
    prologue = sum(cycles for cycles, _ in dmas[1:])
    if steps == 1:
        return prologue + compute + store

    total = sum(
        count * max(owed, compute) for owed, count in _step_classes(dmas, steps)
    )

    # ``recur`` is what a step past the first reloads, ``held`` the same without
    # the store; the test is the period ``_step_classes`` bucketed them under,
    # so the slots it priced and the ones unbooked below cancel exactly.
    recur = sum(c for c, t in dmas if steps // t == 1)
    held = sum(c for c, t in dmas[1:] if steps // t == 1)
    return (
        total
        - max(prologue + store, compute)  # step 0 is the prologue: no compute
        - max(recur, compute)  # step 1 has no prior tile to store
        + prologue  # prologue fetch, overlapping nothing
        + max(held, compute)  # compute tile 0 while tile 1 loads; no store
        + max(store, compute)  # compute the last tile; no load left to issue
        + store  # the last store, once that compute ends
    )


def attention_kv_last(q_block, tq, tkv, sq):
    """The last key block a causal query block attends: the one holding
    the key of its last row.  The rows of query block ``q_block`` sit at
    ``(q_block * tq) % sq`` within their head of ``sq`` rows (the GQA fold
    stacks heads along the rows, a block never straddling one).
    ``q_block`` may be a loop's SymInt."""
    return ((q_block * tq) % sq + tq - 1) // tkv


def attention_tile_latency(node, tiles, grid, config, matrix, bool_mask):
    """Latency and DRAM traffic of a flash-attention node under a tiling.

    Prices the FA3 schedule (``bufferize/attention_v3.py``) a step at a
    time.  The scores product runs first; the context product then runs on
    the matrix unit while the vector unit runs the softmax chain, which had
    to wait for the scores -- so a step costs the scores plus the busier of
    the two, then the rescale that folds the context in.  A one-row query
    tile makes the products matrix-vector, run on the vector unit like any
    (``gemv_compute_cycles``), so its step is the plain sum.  A query block's
    first step is a boundary: the vector unit resets its state, runs the
    softmax, then finalizes the previous block once that block's context has
    landed, so the finalize takes the rescale's place beside the context.
    The products are priced by their interstellar mappings (``matrix``, at
    the head ``attention_head_pad`` pads to); each vector pass at the
    bandwidth-bound rate ``vector_op_utilization`` charges, plus its
    launch.  Query and output move once per query block, while key, value,
    mask and every block scale reload on each step, and the DMAs overlap
    compute the way ``_sweep_cycles`` prices a double-buffered sweep.
    Under ``is_causal`` only the live pairs are steps
    (``attention_kv_last``) and the mask tiles stream from the kernel's
    table, one per step, as int1 codes under ``bool_mask`` and else at the
    scores' width.  A split cache's residual
    (``bufferize/attention_v3.py``) runs serially at each boundary -- its
    two products around a softmax over its R positions, then the rescale
    of the output -- and its operands move once per boundary.

    Args:
        node: The attention node being tiled.
        tiles: Operand FX node -> its SRAM tile shape (``_attention_tiles``).
        grid: The FA3 loop grid, ``(*kv_batch, num_q_blocks, num_kv_blocks)``.
        config (AcceleratorConfig): The hardware description.
        matrix: The matrix unit's cycles per product, keyed as
            ``_attention_products`` names them: ``scores`` and ``context``
            for one step's two, plus ``residual_scores`` and
            ``residual_context`` under a residual.
        bool_mask: Whether the causal table's tiles are int1 codes.

    Returns:
        ``(cycles, DRAM bytes)``; the bytes break a latency tie.
    """
    query, key, value = node.args[0], node.args[1], node.args[2]
    causal = get_arg_value(node, 5, "is_causal", False)
    boundaries = math.prod(grid[:-1])  # one per query block
    tq, head_dim = tiles[query][-2], tiles[query][-1]
    tkv = tiles[key][-1]
    if causal:
        sq = query.value.shape[-2]
        steps = math.prod(grid[:-2]) * sum(
            attention_kv_last(q, tq, tkv, sq) + 1 for q in range(grid[-2])
        )
    else:
        steps = math.prod(grid)

    # The vector unit's passes, at the softmax's width (the output's): one
    # lane group per cycle, bound by the beats its bytes take.
    group_bytes = config.vector_lanes * _node_dtype_bits(node) / 8
    util = min(1.0, 1.0 / math.ceil(group_bytes / config.bytes_per_cycle))

    def passes(count, elems):
        cycles = math.ceil(math.ceil(elems / config.vector_lanes) / util)
        return count * (cycles + KERNEL_LAUNCH_OVERHEAD)

    rows, tile, out = tq, tq * tkv, tq * head_dim
    # The softmax chain: rowmax, exponentials and rowsum over the score tile
    # (and P's quantize under MX); the running max, the rescale factor, its
    # copy and the running sum over a column.  Then the fused
    # rescale-accumulate over the output tile, or at a boundary the reset of
    # the two columns before the softmax and, after it, the accumulate, the
    # finalize and the zeroing of the output tile.
    softmax = passes(4 if node.kwargs.get("block_size") else 3, tile)
    softmax += passes(4, rows)
    rescale = passes(1, out)
    boundary = passes(3, out) + passes(2, rows)
    residual = get_arg_value(node, 7, "key_residual", None)
    if residual is not None:
        length = tiles[residual][-1]
        boundary += matrix["residual_scores"] + matrix["residual_context"]
        boundary += passes(3, tq * length) + passes(4, rows) + rescale
    scores, context = matrix["scores"], matrix["context"]
    if tq == 1:
        # One query row: the products are matrix-vector and run on the
        # vector unit beside the softmax, so nothing overlaps.
        interior = scores + context + softmax + rescale
        boundary_step = scores + context + softmax + boundary
    else:
        interior = scores + max(context, softmax) + rescale
        boundary_step = scores + max(context, softmax + boundary)

    lat = config.access_latency_cycles
    bpc = config.bytes_per_cycle
    # ``_sweep_cycles`` reads the store off the front.
    out_bytes = _operand_bytes(tiles[node], node)
    dmas = [(_transfer_cost(tiles[node], out_bytes, lat, bpc), boundaries)]
    traffic = boundaries * out_bytes
    for operand, transfers in (
        (query, boundaries),
        (key, steps),
        (value, steps),
        (get_arg_value(node, 3, "attn_mask", None), steps),
        (node.kwargs.get("query_scale"), boundaries),
        (node.kwargs.get("key_scale"), steps),
        (node.kwargs.get("value_scale"), steps),
        (get_arg_value(node, 6, "query_residual", None), boundaries),
        (residual, boundaries),
        (get_arg_value(node, 8, "value_residual", None), boundaries),
        (get_arg_value(node, 9, "residual_mask", None), boundaries),
    ):
        if not isinstance(operand, Node):
            continue
        shape = tiles[operand]
        operand_bytes = _operand_bytes(shape, operand)
        dmas.append((_transfer_cost(shape, operand_bytes, lat, bpc), transfers))
        traffic += transfers * operand_bytes
    if causal:
        mask_bits = 1 if bool_mask else _node_dtype_bits(node)
        mask_bytes = math.ceil(tq * tkv * mask_bits / 8)
        dmas.append((_transfer_cost((tq, tkv), mask_bytes, lat, bpc), steps))
        traffic += steps * mask_bytes

    if config.double_buffered_l2:
        latency = _sweep_cycles(dmas, steps, interior)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += steps * interior
    latency += boundaries * (boundary_step - interior)
    return latency, traffic


def gemv_compute_cycles(node, tile_sizes, config):
    """The vector unit's cycles for one ``(X, C, K)`` tile of a
    matrix-vector GEMM: its MACs over the unit's lanes (``gemv_lanes``),
    de-rated by ``vector_op_utilization``."""
    lanes = gemv_lanes(get_anchor_node(node), config)
    ideal = math.ceil(math.prod(tile_sizes) / lanes)
    _, c_tile, k_tile = tile_sizes
    util = vector_op_utilization(node, config, ideal, (k_tile, c_tile))
    return math.ceil(ideal / util)


def gemv_tile_latency(node, tile_sizes, tiled_shapes, tiling, config):
    """Latency and DRAM traffic of a matrix-vector GEMM under a tiling.

    ``tiling`` covers ``(X, C, K)``, but the kernel loops more: the anchor's
    leading batch dims (attention heads) are one grid step each, outside all
    three, and ``build_gemm`` emits them outermost.  Which grid dims dice an
    operand follows from its role (``_GEMV_GRID_DIMS``), and that is what
    ``_block_transfers`` needs to price its reuse -- a GQA KV tile shared by
    four query heads is re-read once per head as soon as a dim inner to the
    head loop splits.

    A tile costs its MACs spread over the width of the unit the GEMV runs
    on (``gemv_lanes``) and de-rated by ``vector_op_utilization``, the
    weight stream's bank switches included -- the same charge
    ``reporting/cost.op_info`` gives the op, so the two models cannot
    disagree.  Double-buffered the
    DRAM engine and the vector unit overlap (``_sweep_cycles``);
    single-buffered they run back to back.

    Args:
        node: The op to cost -- a fully-connected GEMM, or a fused
            ``call_module`` around one.
        tile_sizes: The ``(X, C, K)`` tile; its MAC count sizes the compute.
        tiled_shapes: Operand FX node -> tile shape: ``node`` itself plus each
            allocated activation input.
        tiling: ``(n_x, n_c, n_k)`` -- tile counts over ``(X, C, K)``, ``n_c``
            being the reduction.
        config (AcceleratorConfig): The hardware description.

    Returns:
        ``(cycles, DRAM bytes)``; the bytes break a latency tie in
        ``_search_tiling``.
    """
    n_x, n_c, n_k = tiling
    anchor = get_anchor_node(node) or node
    batch = tuple(anchor.value.shape[:-2])
    nb = len(batch)
    # The grid ``build_gemm`` emits, outermost first.  ``tiling`` is in the
    # search's ``(X, C, K)`` order, so its last two swap.
    grid = batch + (n_x, n_k, n_c)

    lat = config.access_latency_cycles
    bpc = config.bytes_per_cycle
    roles = operand_roles(node)

    # The output is diced by every grid dim but the reduction, which it
    # accumulates over: it is stored on the step that finishes it.  Its dims
    # need no deriving (``build_gemm``'s ``out_index_map``), and a fused
    # ``quantize_mx`` tail makes its shape a *pair*, which ``_operand_spans``
    # could not read anyway.
    out_shape = tiled_shapes[node]
    out_bytes = _operand_bytes(out_shape, node)
    stores = _block_transfers({g: 1 for g in range(nb + 2)}, grid)
    dmas = [(_transfer_cost(out_shape, out_bytes, lat, bpc), stores)]
    traffic = stores * out_bytes

    for n, shape in tiled_shapes.items():
        if n is node or shape is None or not require_allocation(n):
            continue
        # Bytes come off the loaded buffer (its packed dtype); the reuse off
        # the anchor-side operand, which carries the expand.
        role, operand = roles.get(n, (None, n))
        dims = _GEMV_GRID_DIMS.get(role, _OUTPUT_GRID_DIMS)
        transfers = _block_transfers(_operand_spans(operand, dims, nb), grid)
        n_bytes = _operand_bytes(shape, n)
        dmas.append((_transfer_cost(shape, n_bytes, lat, bpc), transfers))
        traffic += transfers * n_bytes

    compute = gemv_compute_cycles(node, tile_sizes, config)

    steps = math.prod(grid)
    if config.double_buffered_l2:
        latency = _sweep_cycles(dmas, steps, compute)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += steps * compute
    return latency, traffic
