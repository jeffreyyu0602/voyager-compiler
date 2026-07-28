"""DRAM-aware latency model for vector-unit L2 tiling.

``run_vector_op_node_l2_tiling`` used to keep the *largest* tile that fits
on-chip.  Under double-buffering that is the worst choice: the first tile's
DRAM load has no compute to overlap (the prologue), so a larger tile means a
longer prologue and higher total latency.  ``vector_tile_latency`` scores a
candidate tiling with a simple pipeline model so the search can pick the tile
that minimizes latency instead of the one that merely fits.

The model mirrors the interstellar GEMM cost function
(``RuntimeCalculator.calculate_runtime`` in ``codegen/lowering/tiling.py``) and
adds two terms it omits: the unoverlapped prologue load, and a per-transfer
DRAM access latency charged once per read and once per write.  Bringing those
back to the GEMM model is a tracked follow-up.

Kept independent of the ``reporting`` package on purpose: the tiling pass runs
inside ``transform()``, long before any reporting stage exists.  The dependency
runs the other way -- ``reporting/cost.op_utilization`` imports
``vector_op_utilization`` from here for its vector branch, so this is the single
copy of that formula (and of ``OP_PASSES``).
"""

import math

import torch
from torch.fx import Node

from .banking import operand_roles, require_allocation
from ...node_info import get_anchor_node, get_node_bytes, is_fully_connected
from ....pt2e_utils import dtype_byte_size

# Passes an op makes over its data; it fetches its operands once per pass.
# Single source of truth: reporting/cost.py imports this via
# ``vector_op_utilization``.
OP_PASSES = {
    torch.ops.aten.layer_norm.default: 4,
    torch.ops.aten.softmax.int: 3,
}


def _val(node):
    """The node's propagated tensor value; the largest tensor of a multi-output
    node (e.g. ``quantize_mx`` -> ``(scale, quantized)``)."""
    if not isinstance(node, Node):
        return None
    for v in (getattr(node, "value", None), node.meta.get("val")):
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (tuple, list)):
            tensors = [t for t in v if isinstance(t, torch.Tensor)]
            if tensors:
                return max(tensors, key=lambda t: t.numel())
    return None


def _widths(node):
    """Element byte widths of ``node`` -- one per output for a multi-output
    node (its ``meta['dtype']`` is then a list), else a single width."""
    dt = node.meta.get("dtype") if isinstance(node, Node) else None
    if isinstance(dt, (list, tuple)):
        return [dtype_byte_size(d) for d in dt if d is not None]
    if dt is not None:
        return [dtype_byte_size(dt)]
    return [dtype_byte_size(_val(node).dtype)]


def vector_op_utilization(node, vector_lanes, bytes_per_cycle):
    """Fraction of peak a vector ``node`` sustains, bound by SRAM bandwidth.

    Peak is one ``vector_lanes``-wide lane group per cycle, fetched at the
    widest of the op's input / output element widths, once per pass it makes
    over its data (softmax 3, layer_norm 4 -- see ``OP_PASSES``).  Tile-size
    independent: it keys off dtype widths, the lane count, and the pass count
    only, so the caller computes it once and reuses it across every candidate
    tile.

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
        widths = _widths(weight.meta.get("source_node", weight))
    else:
        widths = [
            w
            for n in [node, *node.all_input_nodes]
            if _val(n) is not None
            for w in _widths(n)
        ]
    total_bytes = vector_lanes * max(widths, default=2.0)
    num_passes = OP_PASSES.get(anchor.target, 1)
    fetch_cycles = num_passes * math.ceil(total_bytes / bytes_per_cycle)
    return min(1.0, 1.0 / fetch_cycles)


def _operand_bytes(shape, node):
    """Physical bytes of a tiled operand.  ``shape`` is a single tile shape, or
    a sequence of per-output shapes for a multi-output node (``get_node_bytes``
    then yields one width per output)."""
    width = get_node_bytes(node)
    if isinstance(width, (int, float)):
        return math.ceil(math.prod(shape) * width)
    return sum(math.ceil(math.prod(s) * w) for s, w in zip(shape, width))


def _tile_elems(shape):
    """Element count of a tile shape (largest output of a multi-output node)."""
    if shape and isinstance(shape[0], (tuple, list)):
        return max(math.prod(s) for s in shape)
    return math.prod(shape)


def _num_transfers(shape):
    """DMAs a tiled operand costs -- one per tensor, so a ``quantize_mx`` pair
    is two, each paying its own access latency for a tile of scales that is a
    few hundred bytes."""
    if shape and isinstance(shape[0], (tuple, list)):
        return len(shape)
    return 1


def vector_tile_latency(node, tile_sizes, tiled_shapes, tiling, config):
    """Estimated cycles to run ``node`` under a candidate tiling.

    ``config`` is the ``AcceleratorConfig``.  ``tiled_shapes`` maps operand FX
    node -> tile shape (``node`` itself plus each allocated activation input;
    resident params are absent).  ``tiling`` is the per-dim tile *count*, so
    ``N = prod(tiling)`` is the number of tiles.  Per tile: ``C`` compute
    cycles, ``D_read`` to load the inputs and ``D_write`` to store the output,
    one transfer per *tensor* -- a ``quantize_mx`` output stores its scales as
    a second DMA -- each paying one DRAM access latency.

    Double-buffered: the DRAM engine (all reads + writes, ``D = read + write``
    per tile) and the compute engine run concurrently, tile ``i``'s compute
    overlapping tile ``i+1``'s read and tile ``i-1``'s write.  The makespan is
    whichever engine is the bottleneck:

    * DRAM-bound -- the DRAM stream runs flat out, ``N * D``, plus the one
      stall it cannot avoid: at fill both buffer slots are read before any
      output exists, so the first store waits on the first compute,
      ``max(0, C - read)``.  Compute hides entirely behind the rest.
    * compute-bound -- compute never idles, ``N * C``, preceded by the first
      read and followed by the last write (``+ D``) that cannot overlap it.
      Fill needs no term here: compute is the busy engine, so the DRAM idle
      costs nothing.

    So ``makespan = max(N * D + fill, N * C + D)``, which at ``N == 1`` is
    ``D + C`` (a lone tile cannot overlap), matching the single-buffered cost.
    Single-buffered: load, compute and store run back to back, ``N * (D + C)``.
    """
    num_tiles = math.prod(tiling)

    out_shape = tiled_shapes[node]
    out_bytes = _operand_bytes(out_shape, node)
    in_bytes = sum(
        _operand_bytes(shp, n)
        for n, shp in tiled_shapes.items()
        if n is not node
    )
    num_reads = sum(
        _num_transfers(shp) for n, shp in tiled_shapes.items() if n is not node
    )

    in_elems = max(
        (_tile_elems(s) for n, s in tiled_shapes.items() if n is not node),
        default=0,
    )
    tile_ops = max(_tile_elems(out_shape), in_elems)
    lanes = config.vector_lanes
    bpc = config.bytes_per_cycle
    util = vector_op_utilization(node, lanes, bpc)
    compute = math.ceil(math.ceil(tile_ops / lanes) / util)

    lat = config.access_latency_cycles
    read = num_reads * lat + in_bytes / bpc
    write = _num_transfers(out_shape) * lat + out_bytes / bpc
    dram = read + write

    if config.double_buffered_l2:
        # A memory-bound sweep stalls the DRAM stream at both ends: at fill it
        # holds only reads, so the first store waits on the first compute; at
        # drain only stores, so the last store waits on the last compute.
        # Neither costs anything while compute is the busy engine, and at
        # ``N == 1`` they would be the same wait counted twice.
        stalls = (
            max(0.0, compute - read) + max(0.0, compute - write)
            if dram >= compute
            else 0.0
        )
        return max(num_tiles * dram + stalls, num_tiles * compute + dram)
    return num_tiles * (dram + compute)


# Operand roles whose tile moves with the reduction, so the kernel fetches a
# fresh one every grid step.  Every other role -- a bias, whatever a fused tail
# brings of its own (no role at all) -- is indexed by the output block alone and
# is read once per output tile, on the step that finishes it.
_REDUCTION_ROLES = frozenset(
    {
        "input",
        "input_scale",
        "weight",
        "other",
        "weight_scale",
    }
)


def gemv_tile_latency(node, tile_sizes, tiled_shapes, tiling, config):
    """Estimated cycles to run a matrix-vector GEMM under a candidate tiling.

    ``tiling`` is ``(n_x, n_c, n_k)`` -- the tile counts over ``(X, C,
    K)``, ``n_c`` being the reduction -- so the kernel takes ``n_x * n_c * n_k``
    grid steps to finish ``n_x * n_k`` output tiles.  The reduction is innermost
    (``build_gemm`` accumulates across consecutive steps), which is what splits
    the operands: one diced by it streams a new tile every step, the rest are
    read once per output tile alongside the store.  ``operand_roles`` says which
    is which -- whatever a fused tail brings of its own has no role and is never
    a streamed one.  Each operand is its own DMA and pays its own access
    latency, so a smaller tile buys more of them.

    A GEMV runs on the vector unit, so a tile costs its MACs spread over the
    lanes and de-rated by ``vector_op_utilization`` -- the same charge
    ``reporting/cost.op_info`` gives the op, so the two models cannot disagree.

    Double-buffered, the DRAM engine and the vector unit run concurrently and
    the makespan is whichever is the bottleneck, framed by the prologue read and
    the epilogue write that overlap nothing.  At ``n_c == 1`` with no bias and
    no fused tail this is ``vector_tile_latency``'s ``max(N*D, N*C + D)``.
    """
    n_x, n_c, n_k = tiling
    steps = n_x * n_c * n_k
    out_tiles = n_x * n_k

    lat = config.access_latency_cycles
    bpc = config.bytes_per_cycle

    roles = operand_roles(node)
    streamed, per_output = [], []
    for n, shape in tiled_shapes.items():
        if n is node or shape is None or not require_allocation(n):
            continue
        tile_bytes = _operand_bytes(shape, n)
        if roles.get(n) in _REDUCTION_ROLES:
            streamed.append(tile_bytes)
        else:
            per_output.append(tile_bytes)

    read = len(streamed) * lat + sum(streamed) / bpc
    tail = len(per_output) * lat + sum(per_output) / bpc
    write = lat + _operand_bytes(tiled_shapes[node], node) / bpc

    lanes = config.vector_lanes
    util = vector_op_utilization(node, lanes, bpc)
    compute = math.ceil(math.ceil(math.prod(tile_sizes) / lanes) / util)

    dram = steps * read + out_tiles * (tail + write)
    if config.double_buffered_l2:
        return max(dram, steps * compute + read + write)
    return dram + steps * compute
