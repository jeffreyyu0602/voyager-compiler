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
import time
from dataclasses import dataclass, field

import interstellar
import torch

from ..mapping import get_anchor_node
from ..mapping_utils import (
    is_bmm,
    is_conv2d,
    is_depthwise_conv,
    is_fully_connected,
    is_linear,
    is_matmul,
    quant_param_arg_nodes,
    trailing_mha_perm,
)
from ..passes.utils import _pair, get_arg_value
from .utils import _unproject, _NHWC, _HWIO
from ..tiler import _node_dtype_bits, get_dtype_width

logger = logging.getLogger(__name__)
le = interstellar.le


@dataclass
class TilerContext:
    """The interstellar architecture + run options, built once and shared by
    every builder so each can map its anchor node on demand.  ``arch`` /
    ``schedule`` are built from ``config``; ``cache`` is per-run memoization."""

    arch: object
    schedule: object
    config: object  # AcceleratorConfig
    cache: dict = field(default_factory=dict)


def build_interstellar_tiler(config, dram_access_cost=1000):
    """Build the 4-level (PE / L1 / L2 / DRAM) interstellar architecture and
    schedule and wrap them in a ``TilerContext``.

    ``config.pe_array_size`` is ``(ic_dim, oc_dim)``.  The DRAM transfer
    accounting is in absolute bytes (bandwidth as ``config.bytes_per_cycle`` is
    read at run time), so it is not normalized by any element width.

    The L0/L1 capacities are slot arrays: one fixed-width slot per element (the
    max dtype in a mixed-precision design; narrower dtypes are padded into a
    full slot), so they are element / slot counts and the fit check is
    dtype-independent.  The flat L2/L3 byte pools let sub-byte operands pack, so
    those stay in bytes.  When ``double_buffered_l2`` is set, two L3 tiles must
    fit in L2 at once (one computing, one loading), so the effective L2 capacity
    is halved.

    ``config`` carries physical units (GB); the interstellar model wants bytes,
    so ``dram_size`` is scaled to bytes here (the ``dram_bandwidth`` conversion
    to bytes/cycle lives on ``config.bytes_per_cycle``, read at run time).
    """
    ic_dim, oc_dim = config.pe_array_size
    scratchpad_size = config.scratchpad_size
    num_banks = config.num_banks

    # Banking applies only at L2 (the on-chip scratchpad). The bank size is the
    # physical cache split into ``num_banks`` -> derive it from the real cache
    # size (``scratchpad_size``), not the L2 capacity below (which is fudged by
    # the double-buffer ``*2`` hack). None = no banking.
    bank_size = scratchpad_size // num_banks if num_banks is not None else None

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                config.input_buffer_size * ic_dim,
                config.accum_buffer_size * oc_dim,
                config.weight_buffer_size * oc_dim,
            ],
            [scratchpad_size],
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
        bank_size_list=[None, None, bank_size, None],
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
        config=config,
    )


def _layer_cache_key(node, out_dtype, fused=()):
    """A hashable key capturing everything (besides the fixed architecture) that
    determines a node's interstellar mapping: op, operand / output shapes, the
    conv stride/padding/dilation, the operand + scale element widths, the
    microscaling block size, and the fused post-op operand descriptors.
    Identical layers thus share one optimizer run.  ``out_dtype`` is the outer
    node's output dtype (a ``(scale, value)`` list for a fused mx output);
    list-ified to a tuple so the key stays hashable."""
    val = node.value
    out_shape = tuple(val.shape) if isinstance(val, torch.Tensor) else None
    key = [
        node.target,
        tuple(node.args[0].shape),
        tuple(node.args[1].shape),
        out_shape,
        _node_dtype_bits(node.args[0]),
        _node_dtype_bits(node.args[1]),
        _node_dtype_bits(node.kwargs.get("input_scale"), 0),
        _node_dtype_bits(node.kwargs.get("weight_scale"), 0),
        node.kwargs.get("block_size"),
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
        codebooks = quant_param_arg_nodes(n)
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
        codebooks |= quant_param_arg_nodes(n)

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


# Slots of interstellar's (input, output, weight) byte triple.
_IF, _OF, _FL = 0, 1, 2


def output_is_psum(point, level):
    """Whether the output stored at ``level`` is still a partial sum: it is,
    while the IC reduction is incomplete *above* this level.  A partial sum is
    held at the accumulator's width, and carries no output scale yet."""
    num_levels = len(point.loop_blocking(le.IC))
    ic_above = 1
    for lvl in range(level + 1, num_levels):
        ic_above *= point.loop_blocking(le.IC)[lvl]
        ic_above *= point.loop_partitioning(le.IC)[lvl]
    return ic_above > 1


def make_size_fn(
    node,
    out_dtype=None,
    fused_specs=(),
    extra_sharing=0,
    oc_align=None,
):
    """Build a ``Layer.size_fn``: the bytes a tile occupies at a byte-pool
    level.

    Interstellar hands over element counts and the mapping; everything that
    turns those into bytes is policy and lives here -- element widths,
    microscaling scale tensors (one scale per ``block_size`` values), the bias
    and fused post-op operands interstellar knows nothing about, and how all of
    them are packed into banks.

    Operands are grouped one per bank ideally:

        input | input_scale | weight+weight_scale+bias | output+output_scale
              | each fused operand

    A bank cannot be split between groups, so each group rounds up to a whole
    bank -- which puts a floor of ``len(groups) * bank_size`` on the tile,
    however small it is.  With more groups than banks the two *smallest* are
    merged until they fit (a tiny scale tensor would otherwise waste a whole
    bank).  ``extra_sharing`` forces further merges; ``run_interstellar`` raises
    it when nothing maps even at the minimum.

    ``fused_specs`` are the ``(dims, dtype_bits)`` pairs from
    ``_fused_operand_specs``.  ``bank_size is None`` -> no banking: just sum.
    """
    psum_bits = 32

    if isinstance(out_dtype, (list, tuple)):
        of_scale_dtype, of_dtype = out_dtype[-2], out_dtype[-1]
    else:
        of_scale_dtype, of_dtype = None, out_dtype
    if_bits = _node_dtype_bits(node.args[0])
    fl_bits = _node_dtype_bits(node.args[1])
    of_bits = get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
    bias_bits = _node_dtype_bits(get_arg_value(node, 2, "bias", None), 0)
    if_scale_bits = _node_dtype_bits(node.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(node.kwargs.get("weight_scale"), 0)
    of_scale_bits = get_dtype_width(of_scale_dtype) if of_scale_dtype else 0
    block_size = node.kwargs.get("block_size") or 1

    def _scale_bytes(count, bits):
        return count / block_size * bits / 8.0 if bits else 0.0

    def size_fn(counts, point, level, partitioning_accum, bank_size, num_banks):
        if_count, of_count, fl_count = counts
        is_psum = output_is_psum(point, level)

        def extent(d):
            """Output-dim extent here: one bank's worth, or the whole spatially
            replicated block when a partitioning is given."""
            e = 1
            for b in point.loop_blocking(d)[: level + 1]:
                e *= b
            if partitioning_accum is not None:
                e *= partitioning_accum[d]
            return e

        # Veto an OC tile that splits an attention head — the MHA relayout must
        # store whole heads.
        if oc_align and extent(le.OC) % oc_align != 0:
            return (float("inf"),) * 3

        # The output is a wide partial sum until IC is fully reduced; only the
        # final value carries an output scale.
        out_bits = psum_bits if is_psum else of_bits
        of_scale = 0.0 if is_psum else _scale_bytes(of_count, of_scale_bits)
        bias = extent(le.OC) * bias_bits / 8.0 if bias_bits else 0.0

        groups = [
            (if_count * if_bits / 8.0, _IF),
            (_scale_bytes(if_count, if_scale_bits), _IF),
            (
                fl_count * fl_bits / 8.0
                + _scale_bytes(fl_count, fl_scale_bits)
                + bias,
                _FL,
            ),
            (of_count * out_bits / 8.0 + of_scale, _OF),
        ]
        for dims, bits in fused_specs:
            count = 1
            for d in dims:
                count *= extent(d)
            groups.append((count * bits / 8.0, _OF))

        # An absent operand (no scale, no bias) occupies no bank.
        groups = [g for g in groups if g[0] > 0]

        out = [0.0, 0.0, 0.0]
        if not bank_size:
            for size, slot in groups:
                out[slot] += size
            return tuple(out)

        target = max(1, (num_banks or len(groups)) - extra_sharing)
        for _ in range(max(0, len(groups) - target)):
            groups.sort(key=lambda g: g[0])
            (s0, k0), (s1, k1) = groups[0], groups[1]
            # Charge the shared bank to the larger member's operand.
            groups = [(s0 + s1, k0 if s0 >= s1 else k1)] + groups[2:]

        for size, slot in groups:
            out[slot] += math.ceil(size / bank_size) * bank_size
        return tuple(out)

    return size_fn


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

    dram_access_latency_cycles: fixed DRAM latency (cycles) charged once per
    transfer -- the input load, the weight load and the output store are three
    separate transfers per L3 block -- so a tiling with more, smaller L3 blocks
    pays that latency more often. Mirrors the vector-op tiling model
    (``vector_tile_latency`` in ``codegen/passes/tiling_cost.py``).

    double_buffered_l2: when True, DRAM I/O and compute overlap (ping-pong).
    The L3 loop cost becomes max(dram_loading_time, per_l3_compute_time) instead
    of their sum, plus the unoverlapped prologue read and epilogue write (the
    first block's load and the last block's store have nothing to overlap with).
    The L2 capacity should be halved in build_interstellar_tiler when this is
    enabled, so that two L3 tiles fit on-chip simultaneously.
    """

    def __init__(
        self,
        input_dtype_width: int,
        weight_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        sram_bandwidth: int,
        dram_bandwidth: int,
        dram_access_latency_cycles: float,
        double_buffered_l2: bool = False,
        has_sparse_op: bool = False,
        has_tail_operands: bool = False,
    ):
        self.input_dtype_width = input_dtype_width
        self.weight_dtype_width = weight_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.sram_bandwidth = sram_bandwidth
        self.dram_bandwidth = dram_bandwidth
        self.dram_access_latency_cycles = dram_access_latency_cycles
        self.double_buffered_l2 = double_buffered_l2
        self.has_sparse_op = has_sparse_op
        self.has_tail_operands = has_tail_operands

    def _compute_terms(self, mapping):
        """The L0-L2 compute of one L3 tile, split into ``(fill, steady,
        drain)``: the L1 input/weight buffer prologue, the L2 sweep of
        weight-reuse tiles, and the vector-unit epilogue.

        Kept apart because a double-buffered sweep overlaps the prologue and
        the epilogue with the neighbouring tiles, so they are amortized over
        the sweep rather than paid per tile (see ``per_tile_compute_cycles``).
        """
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
        input_buffer_loading_time = (
            input_buffer_loading_size
            * self.input_dtype_width
            / 8
            / self.sram_bandwidth
        )

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        weight_buffer_loading_size *= partitionings[le.IC][0]
        weight_buffer_loading_time = (
            weight_buffer_loading_size
            * self.weight_dtype_width
            / 8
            / self.sram_bandwidth
        )
        if self.has_sparse_op:
            weight_buffer_loading_time *= 2

        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        vector_unit_time = output_size

        requires_high_precision = (
            self.output_dtype_width > self.input_dtype_width
            or self.has_tail_operands
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

        fill = max(input_buffer_loading_time, weight_buffer_loading_time)
        steady = l2_blocks * l1_time
        if self.double_buffered_accum_buffer:
            drain = vector_unit_time
        else:
            drain = extra_vector_unit_time
        return fill, steady, drain

    @staticmethod
    def _l3_blocks(mapping):
        """Number of L3 (DRAM) tiles.  IC is pinned to blocking_size=1 at L3,
        so this is purely spatial."""
        blockings = mapping.loop_blockings
        l3_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l3_blocks *= blockings[i][3]
        return l3_blocks

    def per_tile_compute_cycles(self, mapping):
        """Compute cycles one tiled operation really costs in the L3 sweep, DRAM
        excluded -- both what ``calculate_runtime`` sweeps and the reporting
        model's utilization denominator.

        With a double-buffered L2 consecutive tiles overlap, so the buffer fill
        and the vector drain are paid *once for the whole sweep*, not once per
        tile -- only the first tile fills and only the last drains.  Charging
        them per tile overstates the cost (and so understates utilization).

        The reporting model excludes DRAM here because the scheduler already
        models it as ``async_copy`` events; folding it in would double-count.
        """
        fill, steady, drain = self._compute_terms(mapping)
        if not self.double_buffered_l2:
            return fill + steady + drain
        l3_blocks = self._l3_blocks(mapping)
        return (fill + l3_blocks * steady + drain) / l3_blocks

    def calculate_runtime(self, architecture, layer, mapping):
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings

        l3_blocks = self._l3_blocks(mapping)
        per_l3_compute_time = self.per_tile_compute_cycles(mapping)

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
        # Input, weight and output are three separate sequential transfers,
        # each paying one DRAM access latency: the read loads input + weight
        # (two transfers), the write stores the output (one).
        lat = self.dram_access_latency_cycles
        read = (
            2 * lat + (dram_input_size + dram_weight_size) / self.dram_bandwidth
        )
        write = lat + dram_output_size / self.dram_bandwidth
        dram_loading_time = read + write

        if self.double_buffered_l2:
            # DRAM I/O and compute overlap: each block runs at the slower of the
            # two, plus the unoverlapped prologue read and epilogue write.
            total_time = (
                read
                + l3_blocks * max(dram_loading_time, per_l3_compute_time)
                + write
            )
        else:
            total_time = l3_blocks * (dram_loading_time + per_l3_compute_time)

        return total_time


def _extract_layer_from_node(node):
    """
    Build an interstellar Layer from a node's current (pre-tiling) shapes.
    Return None for layers that should be skipped (depthwise, FC with batch=1,
    3-channel first conv, unsupported weight shapes).
    """
    if is_depthwise_conv(node) or is_fully_connected(node):
        return None

    weight_shape = node.args[1].shape
    transposed = node.meta.get("transposed", False)

    if is_conv2d(node):
        w_dims = _HWIO if transposed else None
        in_dims = _NHWC if transposed else None
        output_channels, input_channels, kH, kW = _unproject(
            weight_shape, w_dims
        )
        _, _, height, width = _unproject(node.shape, in_dims)

        if input_channels == 3:
            return None

        stride_h, stride_w = _pair(get_arg_value(node, 3, "stride", 1))
    else:
        if len(weight_shape) < 2:
            return None

        input_shape = node.args[0].shape
        width = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])

        # Weight (other operand) is (.., K, N): reduction K and output N are its
        # last two dims, flipped by ``is_matmul XOR transposed`` (rank-agnostic,
        # so a batched (B, K, N) weight reads K/N, not the batch dim).
        if is_matmul(node) ^ transposed:
            input_channels, output_channels = weight_shape[-2], weight_shape[-1]
        else:
            output_channels, input_channels = weight_shape[-2], weight_shape[-1]

        kH, kW = 1, 1
        height = 1
        stride_h, stride_w = 1, 1

    return interstellar.Layer(
        nifm=input_channels,
        nofm=output_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
    )


# The optimizer reports "nothing fits" with a bare assert, so match it narrowly:
# every other AssertionError in interstellar is a real invariant break.
_NO_MAPPING = "No valid mapping point found"


def _try_optimize(tiler, layer, rc):
    """Map ``layer``, or ``None`` when no tiling fits the on-chip budget."""
    try:
        return interstellar.optimizer.opt_optimizer(
            tiler.arch,
            layer,
            tiler.schedule,
            rc.calculate_runtime,
            verbose=False,
        )
    except AssertionError as e:
        if _NO_MAPPING not in str(e):
            raise
        return None


def run_interstellar(
    node,
    tiler,
    out_dtype=None,
    fused_specs=(),
    oc_align=None,
):
    """Run interstellar with the 4-level DRAM architecture for a single
    GEMM/conv node.

    Extracts layer dims from the node's current (pre-tiling) shapes, runs the
    optimizer, and logs the resulting L1/L2/L3 tile sizes.  The L2 -> L1 bus
    carries ``min(unroll)`` input elements per cycle -- the rate the array's
    narrow side consumes them at -- so its width is
    ``min(unroll) * if_bits / 8`` bytes per cycle.

    Each operand ideally gets a bank of its own, but a bank cannot be split, so
    a layer with more operands than banks has no tiling at any size.  Retry with
    progressively more bank sharing and keep the first (least-shared) mapping.

    Args:
        node: The GEMM/conv anchor to map.
        tiler (TilerContext): The shared architecture, schedule and unroll.
        out_dtype: The outer (fused) node's output dtype; a ``(scale, value)``
            list for a fused mx output.
        fused_specs: The fused tail's own tiled operands, from
            ``_fused_operand_specs``; they need banks of their own.  A non-empty
            list also keeps the vector unit at high precision (the tail reads a
            tiled residual / mask).
        oc_align (int, optional): ``head_dim`` for a projection GEMM feeding an
            MHA output relayout — its OC tile is constrained to whole heads.

    Returns:
        ``(mapping, per_tile_cycles, access_list)`` -- the best MappingPoint
        (its ``loop_blockings`` give the per-level tile factors), the compute
        cycles of one L3 tile under it (the reporting model's utilization
        denominator), and the per-level ``(input, output, weight)`` access
        counts the ``Tiling`` proto reports.  All ``None`` if the node is
        skipped.
    """
    layer = _extract_layer_from_node(node)
    if layer is None:
        return None, None, None

    of_dtype = (
        out_dtype[-1] if isinstance(out_dtype, (list, tuple)) else out_dtype
    )
    if_bits = _node_dtype_bits(node.args[0])
    fl_bits = _node_dtype_bits(node.args[1])
    of_bits = get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
    if_scale_bits = _node_dtype_bits(node.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(node.kwargs.get("weight_scale"), 0)

    logger.info(
        f"[interstellar] {node.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil} | "
        f"if={if_bits}b fl={fl_bits}b of={of_bits}b "
        f"if_scale={if_scale_bits}b fl_scale={fl_scale_bits}b "
        f"bs={node.kwargs.get('block_size')}"
    )

    sram_bandwidth = min(tiler.config.pe_array_size) * if_bits / 8

    # Use the node's dtype widths so the timing model and the feasibility check
    # (which sizes the same operands) stay consistent.
    rc = RuntimeCalculator(
        if_bits,
        fl_bits,
        of_bits,
        tiler.config.double_buffered_accum_buffer,
        sram_bandwidth,
        tiler.config.bytes_per_cycle,
        tiler.config.access_latency_cycles,
        double_buffered_l2=tiler.config.double_buffered_l2,
        has_tail_operands=bool(fused_specs),
    )

    result = None
    for extra_sharing in range(4 + len(fused_specs)):
        layer.size_fn = make_size_fn(
            node, out_dtype, fused_specs, extra_sharing, oc_align
        )
        result = _try_optimize(tiler, layer, rc)
        if result is not None:
            if extra_sharing:
                logger.info(
                    f"[interstellar] {node.name}: no tiling fits one bank per "
                    f"operand; sharing {extra_sharing} more"
                )
            break
    if result is None:
        raise RuntimeError(
            f"{node.name}: no tiling fits on chip even with every operand "
            f"sharing one bank"
        )
    _, runtime, mapping, _ = result

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
    per_tile_cycles = rc.per_tile_compute_cycles(mapping)
    logger.info(f"[interstellar] {node.name} estimated runtime: {runtime}")
    logger.info(
        f"[interstellar] {node.name} per-tile compute cycles: "
        f"{per_tile_cycles}"
    )
    logger.info(interstellar.utils.format_tiling(mapping))

    _, _, access_list = interstellar.cost_model.get_cost(
        tiler.arch, mapping, layer
    )

    return mapping, per_tile_cycles, access_list


def get_tiling(node, tiler=None):
    """Per-dim tile *counts* for a GEMM/conv ``node`` (standalone or fused
    ``call_module``), or ``None`` (not a matrix op / untiled / skipped).

    conv -> ``(n_y, n_x, n_k, n_c)``; gemm -> ``(batch.., n_m, n_n, n_k)`` — the
    output-spatial / M / N counts plus the reduction count last (``n_c`` for
    conv, ``n_k`` for gemm; the builder's ``num_k``).  The builder derives the
    tile sizes as ``full_dim // count``.

    Prefers the anchor's ``l2_tiling`` (set by the matrix L2 tiling pass —
    output-dim factors; the reduction is kept whole / decomposed away, so its
    factor is 1).  ``l2_tiling`` may carry the reduction factor explicitly — a
    3-tuple gemm ``(n_m, n_n, n_k)`` / a 5-tuple conv ``(n_N, n_k, n_y, n_x,
    n_c)`` — to drive a ``num_k > 1`` reduction sweep.  Otherwise runs
    interstellar via ``tiler`` (caching each layer's mapping).  Interstellar
    tiles a single (M, N, K) gemm and never tiles the leading batch dims (e.g.
    attention heads); the builder loops them, so their counts are the full
    extent (one tile per batch element).
    """
    anchor = get_anchor_node(node)
    is_conv = is_conv2d(anchor)
    if not (is_conv or is_linear(anchor) or is_matmul(anchor)):
        return None

    # Interstellar tiles a single (M, N, K) gemm and never tiles the leading
    # batch dims (e.g. attention heads); the builder loops them, so emit a
    # full-extent count -- one tile per batch element.
    gemm_batch = tuple(anchor.value.shape[: anchor.value.ndim - 2])

    if (tiling := anchor.meta.get("l2_tiling")) is not None:
        logger.debug(f"Found {anchor.name} tiling: {tiling}")
        if is_conv:
            if len(tiling) not in (4, 5):
                raise ValueError(
                    f"{anchor.name} tiling {tiling} must be 4 or 5 elements"
                )
            _, nk, ny, nx, *nc = tiling
            nc = nc[0] if nc else 1
            return (ny, nx, nk, nc)
        if len(tiling) not in (2, 3):
            raise ValueError(
                f"{anchor.name} tiling {tiling} must be 2 or 3 elements"
            )
        nm, nn, *nk = tiling
        nk = nk[0] if nk else 1
        return gemm_batch + (nm, nn, nk)

    if tiler is None:
        return None

    sub_gm = node.meta.get("submodule")

    # Fused-submodule placeholders lack the quant ``meta['dtype']``; copy it
    # from the outer ``all_input_nodes``
    if sub_gm is not None:
        ph_dtypes = [n.meta.get("dtype") for n in node.all_input_nodes]
        placeholdes = [n for n in sub_gm.graph.nodes if n.op == "placeholder"]
        for i, ph in enumerate(placeholdes):
            ph.meta["dtype"] = ph_dtypes[i]

    out_dtype = node.meta.get("dtype")
    fused_specs = _fused_operand_specs(node, anchor)

    # A projection GEMM feeding an MHA relayout must tile OC on whole heads
    # (else ``_detect_mha_relayout`` can't store the tile).  ``oc_align`` is the
    # ``head_dim`` when the fused tail's permute grows the rank, else ``None``.
    oc_align = None
    if sub_gm is not None and not is_conv:
        nodes = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        perm = trailing_mha_perm(nodes)
        if perm is not None and perm.value.ndim > anchor.value.ndim:
            oc_align = perm.value.shape[-1]

    key = _layer_cache_key(anchor, out_dtype, tuple(fused_specs)) + (oc_align,)
    if key in tiler.cache:
        mapping, per_tile_cycles, access_list = tiler.cache[key]
        logger.debug(
            "[tiling] %s: mapping cache hit (%d entries)",
            anchor.name,
            len(tiler.cache),
        )
    else:
        logger.info("[tiling] %s: running interstellar", anchor.name)
        t0 = time.perf_counter()
        mapping, per_tile_cycles, access_list = run_interstellar(
            anchor,
            tiler,
            out_dtype=out_dtype,
            fused_specs=fused_specs,
            oc_align=oc_align,
        )
        logger.info(
            "[tiling] %s: interstellar took %.2fs",
            anchor.name,
            time.perf_counter() - t0,
        )
        # cache None too (skipped layers)
        tiler.cache[key] = (mapping, per_tile_cycles, access_list)
    if mapping is None:
        return None

    # The builders copy these onto the nest they build (the anchor is erased on
    # splice): ``per_tile_cycles`` so the reporting cost model can turn it into
    # a utilization, the mapping / architecture so the proto emitter can
    # serialize the ``Tiling`` message.
    anchor.meta["tiling"] = {
        "per_tile_cycles": per_tile_cycles,
        "interstellar_tiling": (mapping, access_list),
        "interstellar_architecture": tiler.arch,
    }

    b = mapping.loop_blockings  # b[dim][3] = number of DRAM tiles for the dim

    if is_conv:
        return (b[le.OY][3], b[le.OX][3], b[le.OC][3], b[le.IC][3])
    return gemm_batch + (b[le.OX][3], b[le.OC][3], b[le.IC][3])
