"""Per-node tiling for the bufferization lowering.

The bufferization builders (``build_gemm`` / ``build_conv2d``) call
``get_tiling`` on their anchor op to get the per-dim tile *factors* — preferring
the anchor's ``l2_tiling`` meta (set by the matrix L2 tiling pass), else running
interstellar directly.  A reduction factor greater than 1 is what drives the
``PipelinedKernel``'s ``num_k`` accumulation loop.

``build_interstellar_tiler`` builds the 4-level interstellar architecture once
(from the raw hardware description) and returns a ``TilerContext`` threaded down
to each builder; per-node element widths are read from the nodes themselves.
"""

import logging
import math
from dataclasses import dataclass, field

import torch

from ..mapping_utils import (
    is_bmm,
    is_conv2d,
    is_linear,
    is_matmul,
    quant_table_arg_nodes,
)
from ..tiler import _node_dtype_bits, get_dtype_width

logger = logging.getLogger(__name__)


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
    num_banks=None,
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

    # Banking applies only at L2 (the on-chip scratchpad). The bank size is the
    # physical cache split into ``num_banks`` -> derive it from the real cache
    # size (``scratchpad_size``), not the L2 capacity below (which is fudged by
    # the double-buffer ``*2`` hack). None = no banking.
    l2_bank_size = None if num_banks is None else scratchpad_size // num_banks

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
        bank_size_list=[None, None, l2_bank_size, None],
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


def _layer_cache_key(node, in_bits, w_bits, out_dtype, fused=()):
    """A hashable key capturing everything (besides the fixed architecture) that
    determines a node's interstellar mapping: op, operand / output shapes, the
    conv stride/padding/dilation, the element widths, and the fused post-op
    operand descriptors.  Identical layers thus share one optimizer run.
    ``out_dtype`` is the outer node's output dtype (a ``(scale, value)`` list for
    a fused mx output); list-ified to a tuple so the key stays hashable."""
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
        tuple(out_dtype) if isinstance(out_dtype, list) else out_dtype,
        tuple(fused),
    ]
    if is_conv2d(node):
        key += [
            _pair(get_arg_value(node, 3, "stride", 1)),
            _pair(get_arg_value(node, 4, "padding", 0)),
            _pair(get_arg_value(node, 5, "dilation", 1)),
        ]
    return tuple(key)


def _output_pos_to_loop_dim(anchor):
    """Map each output-tensor dim position to its interstellar output loop dim
    (``ON`` batch / ``OC`` channels / ``OY`` height / ``OX`` width), in the
    anchor's physical layout — so a fused operand's shape can be broadcast onto
    the output tile.  conv: NCHW or NHWC (``meta['transposed']``); gemm:
    ``(batch.., M, N)`` with ``M -> OX``, ``N -> OC``, batch dims ``-> ON``.
    """
    import interstellar

    le = interstellar.le
    if is_conv2d(anchor):
        if anchor.meta.get("transposed", False):
            return [le.ON, le.OY, le.OX, le.OC]  # NHWC
        return [le.ON, le.OC, le.OY, le.OX]  # NCHW
    ndim = anchor.value.ndim
    return [le.ON] * (ndim - 2) + [le.OX, le.OC]


def _operand_placeholders(root):
    """External placeholder operands feeding ``root``'s subtree (``root``
    inclusive), tracing through pre-processing ops (``dequantize`` / reshape)
    and skipping each op's quantization codebook / qmap args.
    """
    leaves, stack, visited = [], [root], set()
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if n.op == "placeholder":
            leaves.append(n)
            continue
        codebooks = quant_table_arg_nodes(n)
        for inp in n.all_input_nodes:
            if inp not in codebooks:
                stack.append(inp)
    return leaves


def _fused_operand_specs(node, anchor):
    """Per fused post-op operand of a fused ``call_module`` ``node``: a
    ``(dims, dtype_bits)`` pair, where ``dims`` are the interstellar output loop
    dims (a subset of ``ON/OC/OY/OX``) the operand is *tiled* along — broadcast
    (size-1) dims dropped, so its tile size is ``prod(out_tile[d] for d in
    dims)``.  Empty for a bare node, or one whose fused ops add no tiled tensor
    operand (codebooks / scalars don't count).

    A post-op operand is any submodule placeholder that is *not* one of the
    anchor's own operands (act / weight / scales, traced through any input
    dequantize / reshape — those are counted by the interstellar ``Layer``) and
    is not a codebook / qmap or scalar.  Defining it by exclusion catches an
    operand fed through a ``dequantize`` (e.g. the attention mask).  The
    submodule is already ShapeProp'd (placeholders carry ``.value``).
    """
    submod = node.meta.get("submodule")
    if submod is None:
        return []

    pos_to_loop_dim = _output_pos_to_loop_dim(anchor)
    out_ndim = anchor.value.ndim

    anchor_operands = set(_operand_placeholders(anchor))
    codebooks = set()
    for n in submod.graph.nodes:
        codebooks |= quant_table_arg_nodes(n)

    specs = []
    for p in submod.graph.nodes:
        if p.op != "placeholder":
            continue
        if p.value.numel() == 1 or p in anchor_operands or p in codebooks:
            continue
        op_shape = tuple(p.shape)
        offset = out_ndim - len(op_shape)  # right-align (broadcast)
        dims = tuple(
            pos_to_loop_dim[offset + i]
            for i, sz in enumerate(op_shape)
            if sz > 1
        )
        specs.append((dims, _node_dtype_bits(p)))
    return specs


def _make_fused_size_fn(specs):
    """Build the ``Layer.fused_size_fn`` closure from ``_fused_operand_specs``.
    ``fn(out_tile, bank_size) -> bytes``: each operand's tile bytes is
    ``prod(out_tile[d] for d in dims) * dtype_bits / 8``; an operand whose tile
    is >= half a bank gets its own bank(s), the sub-half ones pack into one
    shared bank (an un-banked level just sums, no rounding).  ``None`` when there
    are no fused operands.
    """
    if not specs:
        return None

    def fn(out_tile, bank_size):
        sizes = []
        for dims, dtype_bits in specs:
            count = 1
            for d in dims:
                count *= out_tile[d]
            sizes.append(count * dtype_bits / 8.0)

        if not bank_size:
            return sum(sizes)
        total = 0.0
        small = 0.0
        for s in sizes:
            if s >= bank_size / 2:
                total += math.ceil(s / bank_size) * bank_size
            else:
                small += s
        if small:
            total += math.ceil(small / bank_size) * bank_size
        return total

    return fn


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


def _extract_layer_from_node(node, out_dtype=None, fused_size_fn=None):
    """
    Build an interstellar Layer from a node's current (pre-tiling) shapes.

    Reads shapes directly from the FX node, before any tiling has occurred.
    ``out_dtype`` is the outer (fused) node's output dtype (the true quantized
    output dtype lives there, not on the matmul/conv anchor).  A fused mx output
    is ``(scale, value)`` so ``out_dtype`` is then a 2-list -- value (the output
    dtype) last, scale dtype first; a single dtype (or None) means no output
    scale and the output width falls back to the anchor's own dtype.

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
        # ``get_tiling``'s OX (X) dim.
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
    # Microscaling (mx) ops carry per-block scale operands; their dtype widths +
    # block size let interstellar account for the scale tensors' L2+ footprint.
    input_scale = node.kwargs.get("input_scale")
    weight_scale = node.kwargs.get("weight_scale")
    # A fused mx output is (scale, value): value (the true quantized output
    # dtype) is last, the scale dtype first.  A single / None out_dtype has no
    # output scale; the output width then falls back to the anchor's own dtype.
    if isinstance(out_dtype, (list, tuple)):
        of_dtype, of_scale_dtype = out_dtype[-1], out_dtype[0]
    else:
        of_dtype, of_scale_dtype = out_dtype, None
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
        of_dtype_bits=(
            get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
        ),
        if_scale_bits=_node_dtype_bits(input_scale, 0),
        fl_scale_bits=_node_dtype_bits(weight_scale, 0),
        of_scale_bits=(
            get_dtype_width(of_scale_dtype) if of_scale_dtype else 0
        ),
        block_size=node.kwargs.get("block_size") or 1,
        fused_size_fn=fused_size_fn,
    )


def run_interstellar(
    node,
    architecture,
    schedule,
    dram_bandwidth: int,
    double_buffered_accum_buffer: bool = False,
    double_buffered_l2: bool = False,
    out_dtype=None,
    fused_size_fn=None,
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

    layer = _extract_layer_from_node(node, out_dtype, fused_size_fn)
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


def get_tiling(node, tiler=None):
    """Per-dim tile *counts* for a GEMM/conv ``node`` (standalone or fused
    ``call_module``), or ``None`` (not a matrix op / untiled / skipped).

    conv -> ``(n_y, n_x, n_c, n_k)``; gemm -> ``(batch.., n_m, n_n, n_k)`` — the
    output-spatial / M / N counts plus the reduction count last (``n_c`` for
    conv, ``n_k`` for gemm; the builder's ``num_k``).  The builder derives the
    tile sizes as ``full_dim // count``.

    Prefers the anchor's ``l2_tiling`` (set by the matrix L2 tiling pass —
    output-dim factors; the reduction is kept whole / decomposed away, so its
    factor is 1).  ``l2_tiling`` may carry the reduction factor explicitly — a
    3-tuple gemm ``(n_m, n_n, n_k)`` / a 5-tuple conv ``(n_N, n_k, n_y, n_x,
    n_c)`` — to drive a ``num_k > 1`` reduction sweep.  Otherwise runs
    interstellar via ``tiler`` (caching each layer's mapping; interstellar does
    not tile the batch, so batch factors are 1).
    """
    import interstellar

    from ..mapping import get_anchor_node

    anchor = get_anchor_node(node)
    is_conv = is_conv2d(anchor)
    if not (is_conv or is_linear(anchor) or is_matmul(anchor)):
        return None

    l2_tiling = anchor.meta.get("l2_tiling")
    logger.debug(f"Found {anchor.name} tiling: {l2_tiling}")
    if l2_tiling is not None:
        if is_conv:
            # ``(n_N, n_k, n_y, n_x)`` keeps the reduction whole (``n_c = 1``);
            # an optional 5th element sets the C-reduction factor directly.
            if len(l2_tiling) == 5:
                _, nk, ny, nx, nc = l2_tiling
            else:
                _, nk, ny, nx = l2_tiling
                nc = 1
            return (ny, nx, nc, nk)
        # ``(n_m, n_n)`` keeps the reduction whole (``n_k = 1``); an optional
        # 3rd element sets the K-reduction factor directly.
        if len(l2_tiling) == 3:
            nm, nn, nk = l2_tiling
        else:
            nm, nn = l2_tiling
            nk = 1
        return (1,) * (anchor.value.ndim - 2) + (nm, nn, nk)

    if tiler is None:
        return None

    # Run interstellar (cached per layer; the optimizer is slow).  ``out_dtype``
    # is the outer (fused) node's output dtype; the fused post-op operands add
    # their own L2+ banks (modeled via ``layer.fused_size_fn``).
    out_dtype = node.meta.get("dtype")
    fused_specs = _fused_operand_specs(node, anchor)
    key = _layer_cache_key(
        anchor,
        _node_dtype_bits(anchor.args[0]),
        _node_dtype_bits(anchor.args[1]),
        out_dtype,
        tuple(fused_specs),
    )
    if key in tiler.cache:
        mapping = tiler.cache[key]
    else:
        logger.debug(f"Running interstellar for {anchor.name}")
        mapping = run_interstellar(
            anchor,
            tiler.arch,
            tiler.schedule,
            tiler.dram_bandwidth,
            tiler.double_buffered_accum_buffer,
            double_buffered_l2=tiler.double_buffered_l2,
            out_dtype=out_dtype,
            fused_size_fn=_make_fused_size_fn(fused_specs),
        )
        tiler.cache[key] = mapping  # cache None too (skipped layers)
    if mapping is None:
        return None

    le = interstellar.le
    b = mapping.loop_blockings  # b[dim][3] = number of DRAM tiles for the dim

    if is_conv:
        return (b[le.OY][3], b[le.OX][3], b[le.IC][3], b[le.OC][3])
    return (1,) * (anchor.value.ndim - 2) + (
        b[le.OX][3],
        b[le.OC][3],
        b[le.IC][3],
    )
