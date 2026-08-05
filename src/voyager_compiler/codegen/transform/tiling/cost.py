"""DRAM-aware latency and traffic model for vector-unit L2 tiling.

``vector_tile_latency`` and ``gemv_tile_latency`` score one candidate tiling of
an op that runs on the vector unit -- an elementwise / reduction op, or a
matrix-vector GEMM -- so ``_search_tiling`` can rank the tiles that fit rather
than take the largest.  Both cost the schedule ``PipelinedKernel`` emits, via
``_sweep_cycles``.

Kept independent of the ``reporting`` package on purpose: the tiling pass runs
inside ``transform()``, long before any reporting stage exists.  The dependency
runs the other way -- ``reporting/cost.op_utilization`` imports
``vector_op_utilization`` from here for its vector branch, so this is the single
copy of that formula (and of ``OP_PASSES``).
"""

import math
from typing import Optional

import torch
from torch.fx import Node

import interstellar
from voyager_compiler.codegen.node_info import (
    dtype_byte_size,
    get_anchor_node,
    get_node_to_key_map,
    is_fully_connected,
    is_gemm_op,
    require_allocation,
    weight_transforms,
)

le = interstellar.le

# Passes an op makes over its data; it fetches its operands once per pass.
# Single source of truth: reporting/cost.py imports this via
# ``vector_op_utilization``.
OP_PASSES = {
    torch.ops.aten.layer_norm.default: 4,
    torch.ops.aten.softmax.int: 3,
    torch.ops.quantized_ops.layer_norm.default: 4,
    torch.ops.quantized_ops.softmax.default: 3,
}

# Cycles a tile costs the vector unit whatever it holds: instruction issue and
# pipeline fill/drain.  Fixed per tile, so a tile too small to amortise it is
# dominated by overhead.
KERNEL_LAUNCH_OVERHEAD = 64


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


def vector_op_utilization(
    node, vector_lanes, bytes_per_cycle, ideal_cycles=None
):
    """Fraction of peak a vector ``node`` sustains, bound by SRAM bandwidth.

    Peak is one ``vector_lanes``-wide lane group per cycle, fetched at the
    widest of the op's input / output element widths, once per pass it makes
    over its data (softmax 3, layer_norm 4 -- see ``OP_PASSES``).

    Given ``ideal_cycles`` -- what one tile would cost at 100% -- the fixed
    ``KERNEL_LAUNCH_OVERHEAD`` is folded in, so that ``ideal_cycles / result``
    is the bandwidth-bound cost *plus* the launch.  Utilization then falls as
    the tile shrinks, collapsing to ``ideal_cycles / KERNEL_LAUNCH_OVERHEAD``
    for a tile too small to amortise it; without it the answer depends only on
    dtype widths, lanes and passes, and holds for every tile size.

    Everything not running on the matrix unit is a vector op, and all are
    bandwidth-bound the same way -- only the bytes fetched per lane group
    differ.  A fully-connected (matrix-vector) GEMM streams its weight once per
    output, so it is sized by the weight width; every other vector op is sized
    by the widest of ``node`` and its inputs.  The rules key off the *anchor*
    (``get_anchor_node``), so a fused ``call_module`` -- whose own target is
    just the submodule name -- resolves to the real op inside; a bare vector op
    is its own anchor.  This is the single copy of the formula;
    ``reporting/cost.op_utilization`` calls it for its vector branch.
    """
    anchor = get_anchor_node(node) or node
    if is_fully_connected(anchor):
        weight = anchor.args[1]
        widths = [_node_dtype_bits(weight.meta.get("source_node", weight))]
    else:
        widths = [
            _node_dtype_bits(n)
            for n in [node, *node.all_input_nodes]
            if n is node or require_allocation(n)
        ]
    # A multi-output node contributes one width per output.
    bits = [b for w in widths for b in (w if isinstance(w, list) else [w])]
    total_bytes = vector_lanes * max(bits, default=16) / 8
    num_passes = OP_PASSES.get(anchor.target, 1)
    fetch_cycles = num_passes * math.ceil(total_bytes / bytes_per_cycle)
    util = min(1.0, 1.0 / fetch_cycles)
    if not ideal_cycles:
        return util
    return ideal_cycles / (
        ideal_cycles / util + num_passes * KERNEL_LAUNCH_OVERHEAD
    )


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
    tile_ops = max(_tile_elems(out_shape), in_elems)
    lanes = config.vector_lanes
    bpc = config.bytes_per_cycle
    ideal = math.ceil(tile_ops / lanes)
    util = vector_op_utilization(node, lanes, bpc, ideal)
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
    """Each operand FX node -> the role it plays, for the tables below and for
    grouping into banks.

    A fused ``call_module``'s operands have no roles of their own, so read them
    off the anchor: ``get_node_to_key_map`` traces each placeholder back through
    ``meta['source_node']`` to the outer node the shape map is keyed by.  What
    the tail brought of its own the anchor never reads, so it stays unnamed and
    takes a bank of its own.  A bare node is its own anchor.
    """
    anchor = get_anchor_node(node)
    if not is_gemm_op(anchor):
        return {}
    roles = get_node_to_key_map(anchor)
    # The output is named on the outer node -- what the shape map keys it by --
    # in place of the anchor's own entry, which is inside the submodule.
    del roles[anchor]
    roles[node] = "output"
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


def gemv_tile_latency(node, tile_sizes, tiled_shapes, tiling, config):
    """Latency and DRAM traffic of a matrix-vector GEMM under a tiling.

    ``tiling`` covers ``(X, C, K)``, but the kernel loops more: the anchor's
    leading batch dims (attention heads) are one grid step each, outside all
    three, and ``build_gemm`` emits them outermost.  Which grid dims dice an
    operand follows from its role (``_GEMV_GRID_DIMS``), and that is what
    ``_block_transfers`` needs to price its reuse -- a GQA KV tile shared by
    four query heads is re-read once per head as soon as a dim inner to the
    head loop splits.

    A GEMV runs on the vector unit, so a tile costs its MACs spread over the
    lanes and de-rated by ``vector_op_utilization`` -- the same charge
    ``reporting/cost.op_info`` gives the op, so the two models cannot
    disagree.  Double-buffered the DRAM engine and the vector unit overlap
    (``_sweep_cycles``); single-buffered they run back to back.

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
        dims = _GEMV_GRID_DIMS.get(roles.get(n), _OUTPUT_GRID_DIMS)
        transfers = _block_transfers(_operand_spans(n, dims, nb), grid)
        n_bytes = _operand_bytes(shape, n)
        dmas.append((_transfer_cost(shape, n_bytes, lat, bpc), transfers))
        traffic += transfers * n_bytes

    lanes = config.vector_lanes
    ideal = math.ceil(math.prod(tile_sizes) / lanes)
    util = vector_op_utilization(node, lanes, bpc, ideal)
    compute = math.ceil(ideal / util)

    steps = math.prod(grid)
    if config.double_buffered_l2:
        latency = _sweep_cycles(dmas, steps, compute)
    else:
        latency = sum(count * cycles for cycles, count in dmas)
        latency += steps * compute
    return latency, traffic
