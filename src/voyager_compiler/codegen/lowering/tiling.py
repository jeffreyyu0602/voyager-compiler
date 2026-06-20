"""Interstellar-driven per-node tiling for the bufferization lowering.

The bufferization builders (``build_gemm`` / ``build_conv2d``) call
``interstellar_tile_sizes`` on their anchor op to get the on-chip tile sizes
directly — there is no separate tiling pass that annotates ``node.meta`` and no
op decomposition.  A reduction tile smaller than the full extent is what drives
the ``PipelinedKernel``'s ``num_k`` accumulation loop.

``build_interstellar_tiler`` builds the 4-level interstellar architecture once
(from the raw hardware description) and returns a ``TilerContext`` threaded down
to each builder; per-node element widths are read from the nodes themselves.
"""

import math
from dataclasses import dataclass, field

import torch

from ..mapping_utils import is_bmm, is_conv2d, is_linear, is_matmul
from ..tiler import _node_dtype_bits


@dataclass
class TilerContext:
    """The interstellar architecture + run options, built once and shared by
    every builder so each can map its anchor node on demand."""

    arch: object
    schedule: object
    dram_bandwidth: int  # bytes per cycle
    double_buffered_accum_buffer: bool = False
    double_buffered_l2: bool = False
    # Per-layer mapping cache (interstellar's optimizer is slow); keyed by
    # ``_layer_cache_key`` so identical layers map once.
    cache: dict = field(default_factory=dict)


def build_interstellar_tiler(
    unroll,
    input_buffer_size,
    weight_buffer_size,
    accum_buffer_size,
    scratchpad_size,
    dram_size,
    dram_bandwidth=0,
    frequency=1.0,
    double_buffered_accum_buffer=False,
    double_buffered_l2=False,
    dram_access_cost=1000,
):
    """Build the 4-level (PE / L1 / L2 / DRAM) interstellar architecture and
    schedule and wrap them in a ``TilerContext``.

    ``unroll`` is ``(ic_dim, oc_dim)``.  ``dram_bandwidth`` is GB/s and is
    converted here to **bytes per cycle** (``bytes/cycle = bw_GBs / freq_GHz`` —
    GB/s and GHz share the 1e9 factor); the DRAM transfer accounting is in
    absolute bytes, so it is not normalized by any element width.

    The L0/L1 capacities are slot arrays: one fixed-width slot per element (the
    max dtype in a mixed-precision design; narrower dtypes are padded into a full
    slot), so they are element / slot counts and the fit check is
    dtype-independent.  The flat L2/L3 byte pools let sub-byte operands pack, so
    those stay in bytes.  When ``double_buffered_l2`` is set, two L3 tiles must
    fit in L2 at once (one computing, one loading), so the effective L2 capacity
    is halved.
    """
    import interstellar

    ic_dim, oc_dim = unroll
    # DRAM bandwidth GB/s -> bytes per cycle (GB/s and GHz share the 1e9 factor).
    dram_bw = int(dram_bandwidth / frequency)

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                input_buffer_size * ic_dim,
                accum_buffer_size * oc_dim,
                weight_buffer_size * oc_dim,
            ],
            # TODO: in the future we should directly pass the full SRAM size.
            [scratchpad_size if double_buffered_l2 else scratchpad_size * 2],
            [dram_size],
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
    )

    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": ic_dim},
                "level1": {"order": -1},
                "level2": {"order": -1},
            },
            "OC": {
                "level0": {"order": 0, "partitioning_size": oc_dim},
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
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

    return TilerContext(
        arch=architecture,
        schedule=schedule,
        dram_bandwidth=dram_bw,
        double_buffered_accum_buffer=double_buffered_accum_buffer,
        double_buffered_l2=double_buffered_l2,
    )


def _layer_cache_key(node, in_bits, w_bits, out_bits):
    """A hashable key capturing everything (besides the fixed architecture) that
    determines a node's interstellar mapping: op, operand / output shapes, the
    conv stride/padding/dilation, and the element widths.  Identical layers thus
    share one optimizer run."""
    from ..passes.utils import _pair, get_arg_value

    val = node.value
    out_shape = tuple(val.shape) if isinstance(val, torch.Tensor) else None
    key = [
        node.target,
        tuple(node.args[0].shape),
        tuple(node.args[1].shape),
        out_shape,
        in_bits,
        w_bits,
        out_bits,
    ]
    if is_conv2d(node):
        key += [
            _pair(get_arg_value(node, 3, "stride", 1)),
            _pair(get_arg_value(node, 4, "padding", 0)),
            _pair(get_arg_value(node, 5, "dilation", 1)),
        ]
    return tuple(key)


class RuntimeCalculator:
    """
    Runtime cost model for a 4-level memory hierarchy:
      L0 = PE registers, L1 = scratchpad, L2 = on-chip SRAM, L3 = DRAM.

    Mirrors RuntimeCalculator for L0-L2, then wraps the L2 timing in an outer
    L3 loop and adds DRAM transfer latency (input + weight tile loaded from DRAM
    to L2 before each L3 iteration).

    dram_bandwidth: DRAM bandwidth in bytes per cycle; the DRAM transfer sizes
    below are in absolute bytes, so the two share a unit.

    Input and weight transfers are assumed sequential. If the memory controller
    issues both simultaneously, replace the sum with max(...) in
    calculate_runtime.

    double_buffered_l2: when True, DRAM I/O and compute overlap (ping-pong).
    The L3 loop cost becomes max(dram_loading_time, per_l3_compute_time) instead
    of their sum. The L2 capacity should be halved in
    build_interstellar_tiler when this is enabled, so that two L3 tiles fit
    on-chip simultaneously.
    """

    def __init__(
        self,
        input_dtype_width: int,
        weight_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        dram_bandwidth: int,
        double_buffered_l2: bool = False,
        has_sparse_op: bool = False,
        has_high_precision_vector_input: bool = False,
    ):
        self.input_dtype_width = input_dtype_width
        self.weight_dtype_width = weight_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.dram_bandwidth = dram_bandwidth
        self.double_buffered_l2 = double_buffered_l2
        self.has_sparse_op = has_sparse_op
        self.has_high_precision_vector_input = has_high_precision_vector_input

    def calculate_runtime(self, architecture, layer, mapping):
        import interstellar

        le = interstellar.le

        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing (identical to RuntimeCalculator) ---
        sa_weight_loading_time = partitionings[le.IC][0] + 2

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

        input_buffer_loading_size = 1
        for loop in [le.IC, le.OY, le.OX]:
            input_buffer_loading_size *= blockings[loop][1]
        input_buffer_loading_time = input_buffer_loading_size

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        weight_buffer_loading_size *= partitionings[le.IC][0]
        weight_buffer_loading_time = (
            weight_buffer_loading_size
            * self.weight_dtype_width
            / self.input_dtype_width
        )
        if self.has_sparse_op:
            weight_buffer_loading_time *= 2

        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        vector_unit_time = output_size

        requires_high_precision = (
            self.output_dtype_width > self.input_dtype_width
            or self.has_high_precision_vector_input
        )
        if requires_high_precision:
            vector_unit_time *= 2

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and requires_high_precision
        )

        if not using_double_buffer_accum_buffer:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
            )
        else:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
                vector_unit_time,
            )

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        if requires_high_precision and not self.double_buffered_accum_buffer:
            extra_vector_unit_time = output_size
        else:
            extra_vector_unit_time = 0

        if self.double_buffered_accum_buffer:
            per_l3_compute_time = (
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + vector_unit_time
            )
        else:
            per_l3_compute_time = (
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        # --- L3 (DRAM): outer tile loop + transfer latency ---
        # IC is pinned to blocking_size=1 at L3, so l3_blocks is purely spatial.
        # Each L3 iteration loads inputs+weights from DRAM and writes outputs
        # back.
        l3_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l3_blocks *= blockings[i][3]

        # DRAM transfer size (in absolute bytes) for one L3 block (levels 0-2
        # only; [3] is the L3 iteration count and belongs in l3_blocks, not in
        # the per-iteration tile size).  ``dram_bandwidth`` is bytes/cycle.
        dram_input_size = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
            * self.input_dtype_width
            / 8
        )
        dram_weight_size = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.FY][1]
            * blockings[le.FX][1]
            * self.weight_dtype_width
            / 8
        )
        dram_output_size = (
            partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
            * self.output_dtype_width
            / 8
        )
        dram_loading_time = (
            dram_input_size + dram_weight_size + dram_output_size
        ) / self.dram_bandwidth

        if self.double_buffered_l2:
            # DRAM I/O and compute overlap: bottleneck is the slower of the two.
            total_time = l3_blocks * max(dram_loading_time, per_l3_compute_time)
        else:
            total_time = l3_blocks * (dram_loading_time + per_l3_compute_time)

        return total_time


def _extract_layer_from_node(node):
    """
    Build an interstellar Layer from a node's current (pre-tiling) shapes.

    Reads shapes directly from the FX node, before any tiling has occurred.

    Returns None for layers that should be skipped (depthwise, FC with batch=1,
    3-channel first conv, unsupported weight shapes).
    """
    import interstellar

    from ..mapping_utils import is_depthwise_conv
    from ..passes.tiling import _conv2d_layout
    from ..passes.utils import _pair, get_arg_value

    if is_depthwise_conv(node):
        return None

    weight_shape = node.args[1].shape

    if is_conv2d(node):
        transposed = node.meta.get("transposed", False)
        kH, kW, input_channels, output_channels = _conv2d_layout(
            weight_shape, True, not transposed
        )
        _, height, width, _ = _conv2d_layout(node.shape, False, not transposed)

        if input_channels == 3:
            return None

        stride_h, stride_w = _pair(get_arg_value(node, 3, "stride", 1))
    else:
        if len(weight_shape) < 2:
            return None

        input_shape = node.args[0].shape
        # Per-batch output rows M: a bmm keeps its batch dim(s) separate (the
        # tiler maps one (M, N, K) gemm and leaves the batch whole) while a
        # linear / broadcast-matmul flattens its leading dims into M.  Matches
        # ``interstellar_tile_sizes``' OX (X) dim.
        width = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])
        if width == 1:
            return None  # matrix-vector: nothing to tile along M

        # Weight (other operand) is (.., K, N): reduction K and output N are its
        # last two dims, flipped by ``is_matmul XOR transposed`` (rank-agnostic,
        # so a batched (B, K, N) weight reads K/N, not the batch dim).
        weight_transposed = is_matmul(node) ^ node.meta.get("transposed", False)
        if weight_transposed:
            input_channels, output_channels = weight_shape[-2], weight_shape[-1]
        else:
            output_channels, input_channels = weight_shape[-2], weight_shape[-1]

        kH, kW = 1, 1
        height = 1
        stride_h, stride_w = 1, 1

    input_node = node.args[0] if len(node.args) > 0 else None
    weight_node = node.args[1] if len(node.args) > 1 else None
    return interstellar.Layer(
        nifm=input_channels,
        nofm=output_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
        if_dtype_bits=_node_dtype_bits(input_node),
        fl_dtype_bits=_node_dtype_bits(weight_node),
        of_dtype_bits=_node_dtype_bits(node),
    )


def run_interstellar_tiling(
    node,
    architecture,
    schedule,
    dram_bandwidth: int,
    double_buffered_accum_buffer: bool = False,
    double_buffered_l2: bool = False,
):
    """
    Run interstellar with the 4-level DRAM architecture for a single GEMM/conv
    node.

    Extracts layer dims from the node's current (pre-tiling) shapes, runs the
    optimizer, and logs the resulting L1/L2/L3 tile sizes.

    Returns the best MappingPoint (its ``loop_blockings`` give the per-level
    tile factors), or None if the node is skipped.
    """
    import logging

    import interstellar

    logger = logging.getLogger(__name__)

    layer = _extract_layer_from_node(node)
    if layer is None:
        return None

    logger.info(
        f"[interstellar DRAM] {node.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil}"
    )

    # Use the layer's per-node dtype widths so the timing model and the
    # feasibility check (which reads the same Layer dtypes) stay consistent.
    rc = RuntimeCalculator(
        layer.if_dtype_bits,
        layer.fl_dtype_bits,
        layer.of_dtype_bits,
        double_buffered_accum_buffer,
        dram_bandwidth,
        double_buffered_l2=double_buffered_l2,
    )

    _, runtime, mapping, _ = interstellar.optimizer.opt_optimizer(
        architecture,
        layer,
        schedule,
        rc.calculate_runtime,
        verbose=False,
    )

    le = interstellar.le
    b = mapping.loop_blockings
    logger.info(
        f"[interstellar] {node.name} L1 tiles: "
        f"IC={b[le.IC][1]} OC={b[le.OC][1]} "
        f"OX={b[le.OX][1]} OY={b[le.OY][1]} ON={b[le.ON][1]}"
    )
    logger.info(
        f"[interstellar] {node.name} L2 tiles: "
        f"IC={b[le.IC][2]} OC={b[le.OC][2]} "
        f"OX={b[le.OX][2]} OY={b[le.OY][2]} ON={b[le.ON][2]}"
    )
    logger.info(
        f"[interstellar] {node.name} L3 tiles: "
        f"IC={b[le.IC][3]} OC={b[le.OC][3]} "
        f"OX={b[le.OX][3]} OY={b[le.OY][3]} ON={b[le.ON][3]}"
    )
    logger.info(f"[interstellar] {node.name} estimated runtime: {runtime}")
    logger.info(interstellar.utils.format_tiling(mapping))

    return mapping


def interstellar_tile_sizes(node, ctx):
    """Run interstellar on a GEMM/conv ``node`` and return its on-chip tile
    sizes, or ``None`` if the node is skipped (e.g. the 3-channel first conv).

    ``mapping.loop_blockings[dim][level]`` are per-level blocking factors whose
    product over levels (0..3 = PE / L1 / L2 / DRAM) is the loop extent, so the
    on-chip (SRAM) tile for a dim is ``full_dim // b[dim][3]`` (everything below
    DRAM) and ``b[dim][3]`` is the number of DRAM tiles.

    Returns conv ``(tile_y, tile_x, tile_c, tile_k)`` or gemm
    ``(tile_m, tile_n, tile_k)`` — the reduction dim (``IC``) is the conv
    ``tile_c`` / gemm ``tile_k`` (so gemm is already in output ``M, N`` + reduction
    order, ready to append to the batch dims).
    """
    import interstellar

    le = interstellar.le

    in_bits = _node_dtype_bits(node.args[0])
    w_bits = _node_dtype_bits(node.args[1])
    out_bits = _node_dtype_bits(node)
    key = _layer_cache_key(node, in_bits, w_bits, out_bits)
    if key in ctx.cache:
        mapping = ctx.cache[key]  # interstellar is slow; map each layer once
    else:
        mapping = run_interstellar_tiling(
            node,
            ctx.arch,
            ctx.schedule,
            ctx.dram_bandwidth,
            ctx.double_buffered_accum_buffer,
            double_buffered_l2=ctx.double_buffered_l2,
        )
        ctx.cache[key] = mapping  # cache None too (skipped layers)
    if mapping is None:
        return None
    b = mapping.loop_blockings

    def tile(full, dim):
        count = b[dim][3]  # number of DRAM-level tiles for this loop dim
        return max(1, full // count) if count else full

    if is_conv2d(node):
        from ..passes.tiling import _conv2d_layout

        transposed = node.meta.get("transposed", False)
        _, Y, X, K = _conv2d_layout(node.shape, False, not transposed)
        _, _, C, _ = _conv2d_layout(node.args[1].shape, True, not transposed)
        return (
            tile(Y, le.OY),
            tile(X, le.OX),
            tile(C, le.IC),
            tile(K, le.OC),
        )

    # linear / matmul: IC == reduction (C), OC == N (K), OX == M (rows).  Return
    # in (M, N, reduction) order so it appends straight after the batch dims.
    input_shape = node.args[0].shape
    X = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])
    C = input_shape[-1]
    weight_shape = node.args[1].shape
    # N is the weight's last/second-to-last dim (rank-agnostic, so a batched
    # bmm weight (B, K, N) reads N, not the batch dim).
    weight_transposed = is_matmul(node) ^ node.meta.get("transposed", False)
    K = weight_shape[-1] if weight_transposed else weight_shape[-2]
    return (tile(X, le.OX), tile(K, le.OC), tile(C, le.IC))


def _anchor_of(node, predicate):
    """The GEMM/conv anchor for ``node``: ``node`` itself for a standalone op, or
    the matching op inside a fused ``call_module``'s submodule; ``None`` if there
    is no matching anchor."""
    from ..mapping import get_anchor_node

    if node.op == "call_module":
        submod = node.meta.get("submodule")
        if not isinstance(submod, torch.fx.GraphModule):
            return None
        anchor = get_anchor_node(submod.graph.nodes)
    else:
        anchor = node
    return anchor if anchor is not None and predicate(anchor) else None


def conv2d_tiling(node, tiler=None):
    """Logical on-chip tile ``(tile_y, tile_x, tile_c, tile_k)`` for a conv2d
    ``node`` (standalone or fused ``call_module``), or ``None`` (untiled /
    unsupported).

    Parses ``node.meta['tiled_shapes']`` when present (the per-tensor tiles the
    tiler annotated, keyed by outer nodes); otherwise runs interstellar via
    ``tiler``.  Both paths return the same logical format; ``tile_c`` is the
    reduction (input-channel) tile driving the builder's ``num_k``.
    """
    from .utils import _NHWC, _unproject

    anchor = _anchor_of(node, is_conv2d)
    if anchor is None:
        return None

    shapes = node.meta.get("tiled_shapes") or {}
    if shapes:
        dims = _NHWC if anchor.meta.get("transposed", False) else None
        out_keyed = shapes.get(node)
        if out_keyed is None:
            return None  # untiled (whole tensor)
        if isinstance(node.value, (list, tuple)):
            out_keyed = out_keyed[-1]  # activation output drives the grid
        _, tk, toh, tow = (int(x) for x in _unproject(out_keyed, dims))
        in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
        in_keyed = shapes.get(in_node)
        if in_keyed is None:
            return None
        tc = int(_unproject(in_keyed, dims)[1])
        return (toh, tow, tc, tk)

    return interstellar_tile_sizes(anchor, tiler) if tiler is not None else None


def gemm_tiling(node, tiler=None):
    """Flat tile ``(batch.., tile_m, tile_n, tile_k)`` for a GEMM ``node``
    (standalone or fused), or ``None``.

    The leading dims are the full output tile (batch may be tiled); the trailing
    ``tile_k`` is the reduction (K) tile driving the builder's ``num_k``.  Parses
    ``node.meta['tiled_shapes']`` when present, else runs interstellar.
    """
    anchor = _anchor_of(node, lambda n: is_linear(n) or is_matmul(n))
    if anchor is None:
        return None

    shapes = node.meta.get("tiled_shapes") or {}
    if shapes:
        out_keyed = shapes.get(node)
        if out_keyed is None:
            return None  # untiled (whole tensor)
        if isinstance(node.value, (list, tuple)):
            out_keyed = out_keyed[-1]
        in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
        in_keyed = shapes.get(in_node)
        if in_keyed is None:
            return None
        return tuple(out_keyed) + (int(in_keyed[-1]),)

    if tiler is None:
        return None
    ts = interstellar_tile_sizes(anchor, tiler)  # (tile_m, tile_n, tile_k)
    if ts is None:
        return None
    # interstellar does not tile the batch dims -> keep them whole.
    out = anchor.value
    return tuple(out.shape[: out.ndim - 2]) + ts


def output_tiling(node, tile):
    """Per-output-dim tile *counts* (full // tile) in the output's physical
    layout for a GEMM/conv ``node``, or ``None`` (untiled).

    ``tile`` is the already-computed ``conv2d_tiling`` / ``gemm_tiling`` result
    (so interstellar is not re-run).  The fused pointwise operands / outputs tile
    at the output block, so this is the divisor passed to ``compute_tiled_shape``
    / ``compute_output_tiled_shapes`` (mirroring ``mapping.adjust_tiling``).
    """
    from .utils import _NHWC, _project, _unproject

    if tile is None:
        return None
    anchor = _anchor_of(
        node, lambda n: is_conv2d(n) or is_linear(n) or is_matmul(n)
    )
    if anchor is None:
        return None
    out = node.value
    out = (
        out[-1] if isinstance(out, (list, tuple)) else out
    )  # activation output

    if is_conv2d(anchor):
        toh, tow, _, tk = (
            tile  # conv2d_tiling: (tile_y, tile_x, tile_c, tile_k)
        )
        dims = _NHWC if anchor.meta.get("transposed", False) else None
        N, _, _, _ = _unproject(out.shape, dims)
        out_tile = _project((N, tk, toh, tow), dims)  # physical output tile
    else:
        out_tile = tile[:-1]  # gemm_tiling: (batch.., tile_m, tile_n, tile_k)
    return tuple(f // t for f, t in zip(out.shape, out_tile))
