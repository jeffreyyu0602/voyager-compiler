"""Latency (cycles) and DRAM-traffic (bytes) model.

Compute cycles are driven by the **anchor only** — a fused ``call_module``'s
anchor (GEMM/conv) and its pointwise tail run pipelined, so the anchor
dominates.  DRAM bytes apply to ``async_copy`` only (``insert`` is zero-time
DPS bookkeeping).  Byte counts use the physical storage dtype via
``dtype_byte_size`` (sub-byte / quantized aware), so packed formats report
their real footprint.
"""

import math
from typing import Tuple

import torch
from torch.fx import GraphModule, Node

from ...mapping import get_anchor_node
from ...mapping_utils import is_conv2d, is_fully_connected, is_gemm_op
from ....pt2e_utils import dtype_byte_size
from .model import CostParams, OpInfo


def _val(node):
    """The node's propagated tensor value: ``.value`` at the top level, or the
    exported ``meta['val']`` inside loop / cond bodies (mirrors codegen).

    For a multi-output node whose value is a tuple (e.g. ``quantize_mx`` ->
    ``(scale, quantized)``), return its largest tensor — the quantized tile,
    not the small per-block scale — so it is sized like the op it represents."""
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


def _shape(node) -> Tuple[int, ...]:
    """Every node ``_shape`` is asked about is a node of interest (an anchor, a
    tile buffer, a DMA operand) that must carry a shape, so a missing value is
    an error, not a silent ``None`` (``_val`` already reduces a multi-output
    node to its representative tensor)."""
    v = _val(node)
    if v is None:
        raise ValueError(f"{node} has no shape (a sized value is required)")
    return tuple(int(d) for d in v.shape)


def _dtype(node):
    """Logical (quantized) dtype string if present, else the physical dtype."""
    dt = node.meta.get("dtype") if isinstance(node, Node) else None
    if dt is not None:
        return dt
    return _val(node).dtype


# --------------------------------------------------------------------------
# DRAM traffic
# --------------------------------------------------------------------------


def tile_bytes(buffer_node: Node, sizes) -> int:
    """Physical bytes of a ``sizes`` tile of ``buffer_node`` (rounded up; a
    sub-byte dtype gives a fractional bytes-per-element)."""
    n = math.prod(sizes)
    return math.ceil(n * dtype_byte_size(_dtype(buffer_node)))


def dram_cycles(n_bytes: int, cost: CostParams) -> int:
    # GB/s == bytes/ns, so n_bytes / dram_bandwidth is the transfer time in ns;
    # add the access latency (ns) and convert to cycles with frequency (GHz).
    time_ns = cost.dram_access_latency + n_bytes / cost.dram_bandwidth
    return math.ceil(time_ns * cost.frequency)


# --------------------------------------------------------------------------
# Utilization
# --------------------------------------------------------------------------


# Passes an op makes over its data; it fetches its operands once per pass.
OP_PASSES = {
    torch.ops.aten.layer_norm.default: 4,
    torch.ops.aten.softmax.int: 3,
}


def op_utilization(
    node: Node,
    anchor: Node,
    units: Tuple[str, ...],
    ideal_cycles: int,
    cost: CostParams,
) -> float:
    """The utilization to charge ``node``.

    A matrix op the interstellar tiler mapped carries the RuntimeCalculator's
    per-tile compute cycles (stamped on it by the builder), which is what one of
    its tiled executions really costs -- so its utilization is the ideal-to-
    actual ratio.

    A vector op is bound by SRAM bandwidth instead (DRAM is modeled separately,
    as ``async_copy`` events).  The SRAM is provisioned to match DRAM, so its
    bandwidth is ``dram_bandwidth / frequency`` bytes per cycle.  Peak is one
    ``unroll[1]``-wide lane group per cycle, fetched at the widest of the
    anchor's input / output element widths and once per pass the op makes over
    its data, so the op sustains the fraction of peak at which that fetch keeps
    up.

    The rules key off the *anchor*: a fused ``call_module``'s own target is just
    the submodule name, so only the anchor names the real op.
    """
    per_tile = node.meta.get("per_tile_cycles")
    if "mma" in units and per_tile and ideal_cycles > 0:
        # Only ever a slowdown; clamp away model noise (and a zero divisor).
        return min(1.0, ideal_cycles / per_tile)
    if is_fully_connected(anchor):
        return 1.0
    sram_bandwidth = cost.dram_bandwidth / cost.frequency  # bytes / cycle
    # A node whose value never got propagated carries no dtype -- skip it
    # rather than guess at a width.
    widths = [
        dtype_byte_size(_dtype(n))
        for n in [node, *node.all_input_nodes]
        if _val(n) is not None
    ]
    num_bytes = cost.unroll[1] * max(widths, default=2.0)
    fetch_cycles = OP_PASSES.get(anchor.target, 1) * math.ceil(
        num_bytes / sram_bandwidth
    )
    # Bandwidth to spare does not beat peak, it only reaches it.
    return min(1.0, 1.0 / fetch_cycles)


# --------------------------------------------------------------------------
# Compute cycles
# --------------------------------------------------------------------------


def op_info(node: Node, cost: CostParams) -> OpInfo:
    """Classify a compute node and compute its ideal (100%-utilization) cycle
    count.  GEMM/conv use the ``unroll[0]*unroll[1]`` systolic throughput;
    vector ops use the ``unroll[1]`` lane count.

    All shapes come from the *anchor* -- the real GEMM/conv/vector op (inside
    the submodule for a fused ``call_module``, the node itself when bare).  Its
    inner nodes are shape-propagated at fusion, so ``anchor.args[i]`` is
    unambiguously the i-th operand (a fused wrapper's args may interleave
    scales / codes / bias).
    """
    anchor = get_anchor_node(node)
    macs_unroll = max(1, cost.unroll[0] * cost.unroll[1])
    vec_unroll = max(1, cost.unroll[1])
    out = _shape(anchor)

    # A fused submodule drives the vector unit too (its tail runs on the VU,
    # even a bare reshape); a bare matrix op occupies only the matrix unit.
    fused = isinstance(node.meta.get("submodule"), GraphModule)
    matrix_units = ("mma", "vector") if fused else ("mma",)

    if is_conv2d(anchor):
        # MACs = prod(out) * C * kh * kw; the channel / kernel dims sit at
        # different positions per layout (transposed => HWIO, else OIHW).
        w = _shape(anchor.args[1])
        if anchor.meta.get("transposed", False):  # HWIO = [kh, kw, C, K]
            kh, kw, c = w[0], w[1], w[2]
        else:  # OIHW = [K, C, kh, kw]
            c, kh, kw = w[1], w[2], w[3]
        macs = math.prod(out) * c * kh * kw
        ideal = math.ceil(macs / macs_unroll)
        return OpInfo(
            node.name,
            "conv",
            ideal,
            {
                "macs": macs,
                "input": _shape(anchor.args[0]),
                "weight": w,
                "output": out,
            },
            units=matrix_units,
            utilization=op_utilization(node, anchor, matrix_units, ideal, cost),
        )

    if is_gemm_op(anchor):
        # MACs = prod(out) * K; each output element costs K = in[-1] MACs
        # (folds batch dims; works for linear / matmul / bmm).
        inp = _shape(anchor.args[0])
        macs = math.prod(out) * inp[-1]
        # A fully-connected (matrix-vector) GEMM runs on the vector unit, so
        # its throughput is the vector lane count, not the systolic MAC count.
        fc = is_fully_connected(anchor)
        divisor = vec_unroll if fc else macs_unroll
        ideal = math.ceil(macs / divisor)
        units = ("vector",) if fc else matrix_units
        return OpInfo(
            node.name,
            "gemm",
            ideal,
            {
                "macs": macs,
                "input": inp,
                "weight": _shape(anchor.args[1]),
                "output": out,
            },
            units=units,
            utilization=op_utilization(node, anchor, units, ideal, cost),
        )

    # Vector op: work is the larger of input / output element count -- a
    # reduction reads its whole input to make a smaller output.  Use the
    # anchor's primary input (all_input_nodes[0]) to skip qmap / codebook
    # operands.
    in_nodes = anchor.all_input_nodes
    in_shape = _shape(in_nodes[0]) if in_nodes else ()
    ops = max(math.prod(out), math.prod(in_shape))
    ideal = math.ceil(ops / vec_unroll)
    return OpInfo(
        node.name,
        "vector",
        ideal,
        {"ops": ops, "input": in_shape, "output": out},
        units=("vector",),
        utilization=op_utilization(node, anchor, ("vector",), ideal, cost),
    )
