"""Latency (cycles) and DRAM-traffic (bytes) model.

Compute cycles are driven by the **anchor only** — a fused ``call_module``'s
anchor (GEMM/conv) and its pointwise tail run pipelined, so the anchor
dominates.  DRAM bytes apply to ``async_copy`` only (``copy_tile`` is zero-time
DPS bookkeeping).  Byte counts use the physical storage dtype via
``dtype_byte_size`` (sub-byte / quantized aware), so packed formats report
their real footprint.
"""

import math
from typing import Optional, Tuple

import torch
from torch.fx import Node

from ...mapping import get_anchor_node
from ...mapping_utils import is_conv2d, is_gemm_op
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


def _shape(node) -> Optional[Tuple[int, ...]]:
    v = _val(node)
    return tuple(int(d) for d in v.shape) if v is not None else None


def _dtype(node):
    """Logical (quantized) dtype string if present, else the physical dtype."""
    dt = node.meta.get("dtype") if isinstance(node, Node) else None
    if dt is not None:
        return dt
    v = _val(node)
    return v.dtype if v is not None else torch.float32


# --------------------------------------------------------------------------
# DRAM traffic
# --------------------------------------------------------------------------


def tile_bytes(buffer_node: Node, sizes) -> int:
    """Physical bytes of a ``sizes`` tile of ``buffer_node`` (rounded up; a
    sub-byte dtype gives a fractional bytes-per-element)."""
    n = 1
    for s in sizes:
        n *= int(s)
    return math.ceil(n * dtype_byte_size(_dtype(buffer_node)))


def dram_cycles(n_bytes: int, cost: CostParams) -> int:
    return cost.setup_cycles + math.ceil(n_bytes / cost.bytes_per_cycle)


# --------------------------------------------------------------------------
# Compute cycles
# --------------------------------------------------------------------------


def _prod(dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


# A fused ``call_module``'s *output* shape lives on its anchor (the submodule's
# GEMM/conv); its *input* shapes live on the compute node's own args.  For a
# bare op, anchor is the node itself, so both come from the same place.


def _gemm_macs(node: Node, anchor: Node) -> int:
    """MACs of a GEMM tile: ``prod(out[:-1]) * out[-1] * in[-1]`` (folds any
    batch dims into the row count; works for linear / matmul / bmm)."""
    out = _shape(anchor)
    inp = _shape(node.args[0]) if node.args else None
    if not out or not inp:
        return 0
    return _prod(out[:-1]) * out[-1] * inp[-1]


def _conv_macs(node: Node, anchor: Node) -> int:
    """MACs of a conv tile: ``prod(out) * C * kh * kw``.  Channel / kernel dims
    depend on the physical layout (``meta['transposed']`` => NHWC / HWIO)."""
    out = _shape(anchor)
    w = _shape(node.args[1]) if len(node.args) > 1 else None
    if not out or not w:
        return 0
    transposed = anchor.meta.get("transposed", node.meta.get("transposed"))
    if transposed:  # weight HWIO = [kh, kw, C, K]
        kh, kw, c = w[0], w[1], w[2]
    else:  # weight OIHW = [K, C, kh, kw]
        c, kh, kw = w[1], w[2], w[3]
    return _prod(out) * c * kh * kw


def op_info(node: Node, cost: CostParams) -> OpInfo:
    """Classify a compute node and compute its ideal (100%-utilization) cycle
    count.  GEMM/conv use the ``unroll[0]*unroll[1]`` systolic throughput;
    vector ops use the ``unroll[1]`` lane count."""
    anchor = get_anchor_node(node) or node
    macs_unroll = max(1, cost.unroll[0] * cost.unroll[1])
    vec_unroll = max(1, cost.unroll[1])
    if is_conv2d(anchor):
        macs = _conv_macs(node, anchor)
        return OpInfo(
            node.name, "conv", math.ceil(macs / macs_unroll), {"macs": macs}
        )
    if is_gemm_op(anchor):
        macs = _gemm_macs(node, anchor)
        return OpInfo(
            node.name, "gemm", math.ceil(macs / macs_unroll), {"macs": macs}
        )
    out = _shape(anchor) or _shape(node) or ()
    ops = _prod(out)
    return OpInfo(
        node.name, "vector", math.ceil(ops / vec_unroll), {"ops": ops}
    )


def compute_cycles(node: Node, cost: CostParams) -> int:
    return op_info(node, cost).ideal_cycles
