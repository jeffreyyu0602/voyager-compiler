"""DRAM-aware latency model for vector-op L2 tiling.

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

from ..mapping import get_anchor_node, get_node_bytes, get_node_to_key_map
from ..mapping_utils import is_fully_connected
from ...pt2e_utils import dtype_byte_size

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


def vector_tile_latency(node, tile_sizes, tiled_shapes, global_tiling, config):
    """Estimated cycles to run ``node`` under a candidate tiling.

    ``config`` is the ``AcceleratorConfig``.  ``tiled_shapes`` maps operand key
    -> tile shape (``"output"`` plus each allocated activation input; resident
    params are absent).  ``global_tiling`` is the per-dim tile *count*, so
    ``N = prod(global_tiling)`` is the number of tiles.  Per tile: ``C`` compute
    cycles, ``D_read`` to load the inputs (a separate transfer per input, each
    paying one DRAM access latency) and ``D_write`` to store the output.

    Double-buffered: the pipeline runs at ``max(D, C)`` per tile, plus the
    unoverlapped prologue load and epilogue store.  Single-buffered: load,
    compute and store run back to back.
    """
    num_tiles = math.prod(global_tiling)

    out_shape = tiled_shapes["output"]
    key_to_node = {k: n for n, k in get_node_to_key_map(node).items()}
    out_bytes = _operand_bytes(out_shape, key_to_node["output"])
    num_inputs = len(tiled_shapes) - 1  # every key but "output"
    in_bytes = sum(
        _operand_bytes(shp, key_to_node[k])
        for k, shp in tiled_shapes.items()
        if k != "output"
    )

    in_elems = max(
        (_tile_elems(s) for k, s in tiled_shapes.items() if k != "output"),
        default=0,
    )
    tile_ops = max(_tile_elems(out_shape), in_elems)
    lanes = config.vector_lanes
    bpc = config.bytes_per_cycle
    util = vector_op_utilization(node, lanes, bpc)
    compute = math.ceil(math.ceil(tile_ops / lanes) / util)

    read = num_inputs * config.access_latency_cycles + in_bytes / bpc
    write = config.access_latency_cycles + out_bytes / bpc
    dram = read + write

    if config.double_buffered_l2:
        return read + num_tiles * max(dram, compute) + write
    return num_tiles * (dram + compute)
