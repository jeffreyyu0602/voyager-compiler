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

import gc
import logging
import math
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import interstellar
import torch

from voyager_compiler.codegen.shape_prop import ShapeProp

from ...node_info import (
    _pair,
    get_anchor_node,
    get_arg_value,
    is_bmm,
    is_conv2d,
    is_depthwise_conv,
    is_fully_connected,
    is_linear,
    is_matmul,
    quant_param_arg_nodes,
    trailing_mha_perm,
    weight_is_ck,
)
from ..tiling.search import gemv_op_tiling
from .utils import _unproject, _NHWC, _HWIO
from ....pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)
le = interstellar.le

# How much longer than the best modeled runtime a mapping may take and still
# be chosen; the least-energy one among those wins.  0.0 = only the fastest.
DEFAULT_RUNTIME_TOLERANCE = 0.01

# Partial-sum width, bits -- shared by the timing model and the tile sizing.
# ``accumulate_fp32`` is never enabled today; wire it through if that changes.
PSUM_BITS = 16

# The non-reduction L3 loops a builder's grid may permute, outermost to
# innermost, in the order it emits when nothing says otherwise.  The reduction
# is always innermost (the kernels accumulate in place) and the gemm batch dims
# are always outermost, so neither appears here.
GEMM_L3_ORDER = ("M", "N")
CONV_L3_ORDER = ("K", "Y", "X")

# The interstellar loop each of those tags maps onto.  A gemm is modelled as a
# 1x1 conv, so M rides on OX and N on OC.
_GEMM_LOOP = {"M": le.OX, "N": le.OC}
_CONV_LOOP = {"K": le.OC, "Y": le.OY, "X": le.OX}


def get_dtype_width(dtype) -> int:
    """Element width in bits, derived from the canonical ``dtype_byte_size``
    so the dtype-name parsing lives in exactly one place."""
    return round(dtype_byte_size(dtype) * 8)


def _node_dtype_bits(node, default: Optional[int] = None) -> int:
    """
    Element width in bits for an FX node's tensor, read from the graph.

    Prefers node.meta["dtype"] (the compiler's tracked storage dtype, e.g. an
    NF4 weight), falling back to the runtime tensor dtype node.value.dtype.  A
    multi-output node (meta dtype / value is a list, e.g. quantize_mx) uses its
    primary (last) output.
    """
    if node is not None:
        dtype = node.meta.get("dtype")
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[-1]
        if dtype is None:
            val = getattr(node, "value", None)
            if isinstance(val, (list, tuple)):
                val = val[-1] if val else None
            dtype = getattr(val, "dtype", None)
        if dtype is not None:
            return round(dtype_byte_size(dtype) * 8)
    if default is None:
        raise ValueError(f"node {node} has no dtype to size the operand")
    return default


@dataclass
class TilerContext:
    """The interstellar architecture + run options, built once and shared by
    every builder so each can map its anchor node on demand.  ``arch`` /
    ``schedule`` are built from ``config``; ``cache`` is per-run
    memoization."""

    arch: object
    schedule: object
    config: object  # AcceleratorConfig
    runtime_tolerance: float = DEFAULT_RUNTIME_TOLERANCE
    cache: dict = field(default_factory=dict)


def build_interstellar_tiler(
    config, dram_access_cost=1000, runtime_tolerance=DEFAULT_RUNTIME_TOLERANCE
):
    """Build the 4-level (PE / L1 / L2 / DRAM) interstellar architecture and
    schedule and wrap them in a ``TilerContext``.

    ``config.pe_array_size`` is ``(ic_dim, oc_dim)``.  The DRAM transfer
    accounting is in absolute bytes (bandwidth as ``config.bytes_per_cycle`` is
    read at run time), so it is not normalized by any element width.

    The L0/L1 capacities are slot arrays: one fixed-width slot per element (the
    max dtype in a mixed-precision design; narrower dtypes are padded into a
    full slot), so they are element / slot counts and the fit check is
    dtype-independent.  The flat L2/L3 byte pools let sub-byte operands pack, so
    those stay in bytes.

    L2 is planned exactly the way ``plan_memory`` allocates it: the capacity is
    the *physical* scratchpad and, under ``double_buffered_l2``, each source is
    charged once per ping-pong copy (``size_fn``) rather than the capacity being
    halved.  Halving cannot express a buffer that is allocated only once -- the
    reduction scratch -- and it is that omission that let the tiler hand back
    tilings the allocator could not place.

    ``config`` carries physical units (GB); the interstellar model wants bytes,
    so ``dram_size`` is scaled to bytes here (the ``dram_bandwidth`` conversion
    to bytes/cycle lives on ``config.bytes_per_cycle``, read at run time).
    """
    ic_dim, oc_dim = config.pe_array_size

    # Banking applies only at L2 (the on-chip scratchpad).  ``scratchpad_size``
    # is the per-copy budget, so the physical pool -- and its bank count -- are
    # doubled when the planner ping-pongs (``plan_memory`` does the same).
    copies = 2 if config.double_buffered_l2 else 1
    bank_size = (
        config.scratchpad_size // config.num_banks
        if config.num_banks is not None
        else None
    )

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                config.input_buffer_size * ic_dim,
                config.accum_buffer_size * oc_dim,
                config.weight_buffer_size * oc_dim,
            ],
            [config.scratchpad_size * copies],
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
                "level2": {"order": 0},
                "level3": {"order": 0},
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
        runtime_tolerance=runtime_tolerance,
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
    has_tail=False,
    copies=1,
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

    Each such source is ping-ponged, so it costs ``copies`` whole banks -- the
    two halves live in *separate* banks, which is what the planner does and
    what lets a load overlap the compute reading the other half.  A split
    reduction with a fused tail also accumulates into a scratch buffer the
    builders allocate exactly once (``_ScratchSpec``); it is charged a single
    bank-aligned region on top.  Leaving it out is what let the tiler return
    tilings ``plan_memory`` could not place.

    A bank cannot be split between groups, so each group rounds up to a whole
    bank -- which puts a floor of ``copies * len(groups) * bank_size`` on the
    tile, however small it is.  With more groups than banks the two *smallest*
    are merged until they fit (a tiny scale tensor would otherwise waste a
    whole bank).  Only whole sources merge, never a source's own two copies --
    each copy must keep its own bank.  ``extra_sharing`` forces further merges;
    ``run_interstellar`` raises it when nothing maps even at the minimum.

    ``fused_specs`` are the ``(dims, dtype_bits)`` pairs from
    ``_fused_operand_specs``.  ``bank_size is None`` -> no banking: just sum.
    """
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
        out_bits = PSUM_BITS if is_psum else of_bits
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

        # A reduction split across L3 steps accumulates into a scratch buffer
        # the builders allocate once for the whole kernel, not per ping-pong
        # half -- so it is charged one region, outside ``copies``.
        scratch = (
            of_count * PSUM_BITS / 8.0
            if has_tail and point.loop_blocking(le.IC)[3] > 1
            else 0.0
        )

        out = [0.0, 0.0, 0.0]
        if not bank_size:
            for size, slot in groups:
                out[slot] += copies * size
            out[_OF] += scratch
            return tuple(out)

        scratch_banks = math.ceil(scratch / bank_size) if scratch else 0
        # Banks left for the ping-ponged sources, in whole sources.
        budget = (num_banks or copies * len(groups)) - scratch_banks
        target = max(1, budget // copies - extra_sharing)
        for _ in range(max(0, len(groups) - target)):
            groups.sort(key=lambda g: g[0])
            (s0, k0), (s1, k1) = groups[0], groups[1]
            # Charge the shared bank to the larger member's operand.
            groups = [(s0 + s1, k0 if s0 >= s1 else k1)] + groups[2:]

        for size, slot in groups:
            out[slot] += copies * math.ceil(size / bank_size) * bank_size
        out[_OF] += scratch_banks * bank_size
        return tuple(out)

    return size_fn


class RuntimeCalculator:
    """Runtime cost model for a 4-level hierarchy (PE / L1 / L2 / DRAM).

    Wraps the L0-L2 compute in the outer L3 grid loop (K included) and adds
    its DRAM transfers.  A double-buffered step costs the slower of its DRAM
    and its compute, framed by an unoverlapped prologue load and epilogue
    store.

    The store is charged on the step that issues it, not averaged over the
    sweep -- a tile can outlast one step of compute, and averaging hides the
    stall.

    Args:
        input_dtype_width: Input element width, bits.
        weight_dtype_width: Weight element width, bits.
        output_dtype_width: Output element width, bits.
        accum_dtype_width: Partial-sum width, bits, while K accumulates.
        double_buffered_accum_buffer: Overlap the accumulator drain.
        sram_bandwidth: L2 -> L1 bandwidth, bits per cycle.
        dram_bandwidth: DRAM bandwidth, bytes per cycle (sizes are bytes).
        dram_access_latency_cycles: Fixed latency, once per transfer.
        double_buffered_l2: Overlap DRAM I/O with compute, so a grid step
            costs ``max(dram, compute)`` and not their sum.
        has_sparse_op: Double the weight load time.
        has_tail: The node has a fused post-op, so a reduction drains through
            the vector unit; a bare GEMM reduces in place and does not.
        tail_specs: The fused tail's own tiled operands as ``(dims, bits)``
            pairs, from ``_fused_operand_specs``; ``tail_fetch_cycles`` turns
            them into a per-vector read cost and ``tail_tile_sizes`` into the
            DRAM they stream per output tile.
        input_scale_width: Input block-scale width, bits (0 = not microscaled).
        weight_scale_width: Weight block-scale width, bits.
        output_scale_width: Output block-scale width, bits -- a fused
            ``quantize_mx`` tail stores the tile's scales next to its values.
        scale_block_size: Elements per block scale.
    """

    # The L3 loops each DRAM operand's tile spans.  A loop outside them is a
    # reuse window: the tile does not change while it turns.
    _IF_DIMS = (le.OX, le.OY, le.IC, le.ON)
    _FL_DIMS = (le.OC, le.IC, le.FX, le.FY)

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
        has_sparse_op: bool = False,
        has_tail: bool = False,
        tail_specs=(),
        input_scale_width: int = 0,
        weight_scale_width: int = 0,
        output_scale_width: int = 0,
        scale_block_size: int = 1,
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
        self.has_sparse_op = has_sparse_op
        self.has_tail = has_tail
        self.tail_specs = tuple(tail_specs)
        self.input_scale_width = input_scale_width
        self.weight_scale_width = weight_scale_width
        self.output_scale_width = output_scale_width
        self.scale_block_size = scale_block_size

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

    def tail_fetch_cycles(self, mapping):
        """SRAM cycles to fetch one vector of the fused tail's operands -- the
        read-side counterpart of ``store_cycles``.  Each operand has a bank of
        its own, so they are read at once and the slowest sets the rate; one
        broadcast along OC feeds a whole vector from a single element."""
        oc_dim = mapping.loop_partitionings[le.OC][0]
        return max(
            (
                math.ceil(
                    bits
                    * (oc_dim if le.OC in dims else 1)
                    / self.sram_bandwidth
                )
                for dims, bits in self.tail_specs
            ),
            default=0,
        )

    def matrix_unit_cycles(self, mapping):
        """Matrix-unit cycles of one L3 grid step: the L2 sweep of
        weight-reuse tiles, plus the once-per-sweep overhead (buffer fill,
        systolic skew, accumulator drain) spread over the steps a
        double-buffered L2 overlaps it with.  Also the reporting model's
        per-tile utilization denominator.
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
            / self.sram_bandwidth
        )

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        weight_buffer_loading_size *= partitionings[le.IC][0]
        weight_buffer_loading_time = (
            weight_buffer_loading_size
            * self.weight_dtype_width
            / self.sram_bandwidth
        )
        if self.has_sparse_op:
            weight_buffer_loading_time *= 2

        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        oc_dim = partitionings[le.OC][0]
        num_k = blockings[le.IC][3]
        output_width = (
            self.accum_dtype_width if num_k > 1 else self.output_dtype_width
        )
        store_cycles = math.ceil(output_width * oc_dim / self.sram_bandwidth)
        # The tail runs only on the last K step: with num_k > 1 that step is
        # charged by ``vector_unit_cycles``, and the rest only accumulate --
        # adding the output into a psum-width partial sum, reading no operand.
        if num_k == 1:
            store_cycles = max(store_cycles, self.tail_fetch_cycles(mapping))
        vector_unit_time = output_size * store_cycles

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and store_cycles > 1
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

        buffer_fill = max(input_buffer_loading_time, weight_buffer_loading_time)
        skew = partitionings[le.IC][0] + partitionings[le.OC][0] - 2
        steady = l2_blocks * l1_time
        if self.double_buffered_accum_buffer:
            drain = vector_unit_time
        else:
            drain = output_size * (store_cycles - 1)
        overhead = buffer_fill + skew + drain

        if not self.double_buffered_l2:
            return steady + overhead

        normalize_factor = self._l3_blocks(mapping) if num_k == 1 else num_k
        return steady + overhead / normalize_factor

    def vector_unit_cycles(self, mapping):
        """Vector-unit cycles to finish one L3 output tile: the tail's reads
        and the write-out, at the node's output width rather than the partial
        sum's.  Charged on the grid step that ends a K sweep, after that
        step's accumulation.
        """
        blockings = mapping.loop_blockings
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1] * blockings[loop][2]
        oc_dim = mapping.loop_partitionings[le.OC][0]
        store_cycles = math.ceil(
            self.output_dtype_width * oc_dim / self.sram_bandwidth
        )
        fetch_cycles = self.tail_fetch_cycles(mapping)
        vector_unit_time = output_size * max(store_cycles, fetch_cycles)
        return vector_unit_time

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

    def calculate_runtime(self, architecture, layer, mapping):
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings

        # Elements of one L3 tile: levels 0-2 only, since [3] is the grid trip
        # count, not part of the tile.
        input_elems = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
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

        input_load = transfer(
            input_elems * self.input_dtype_width / 8,
            input_elems / self.scale_block_size * self.input_scale_width / 8,
        )
        weight_load = transfer(
            weight_elems * self.weight_dtype_width / 8,
            weight_elems / self.scale_block_size * self.weight_scale_width / 8,
        )
        store = transfer(
            output_elems * self.output_dtype_width / 8,
            output_elems / self.scale_block_size * self.output_scale_width / 8,
        )
        # The fused tail streams its own operands (a residual, a mask) once per
        # output tile, on the step that runs it.
        tail_load = transfer(*self.tail_tile_sizes(mapping))

        l3_blocks = self._l3_blocks(mapping)
        num_k = blockings[le.IC][3]
        output_tiles = l3_blocks // num_k
        matrix_unit_cycles = self.matrix_unit_cycles(mapping)
        vector_unit_cycles = (
            self.vector_unit_cycles(mapping) if self.has_tail else 0
        )

        # An operand is re-fetched only when its block index moves, which the
        # mapping's L3 order decides -- so read it off the mapping rather than
        # assuming a nest.  With the reduction split it is innermost and both
        # operands change every step; otherwise the operand that does not span
        # the innermost loop is held across that loop's whole run.  Which one
        # that is depends on the op: with N innermost the input is held (it
        # does not span N), with a conv's X innermost it is the weight.
        input_steps = self._l3_loads(mapping, self._IF_DIMS)
        weight_steps = self._l3_loads(mapping, self._FL_DIMS)

        if self.double_buffered_l2:
            if num_k == 1:
                # Every step finishes an output tile: it stores, runs the tail
                # and reloads the streaming operand; only the steps where the
                # held one moves also pay for it.  Every L3 loop belongs to one
                # operand or the other, so exactly one of them streams.
                held_steps = min(input_steps, weight_steps)
                held_load, streamed_load = (
                    (input_load, weight_load)
                    if input_steps < weight_steps
                    else (weight_load, input_load)
                )
                step = streamed_load + tail_load + store
                total_time = (
                    input_load
                    + weight_load
                    + held_steps * max(step + held_load, matrix_unit_cycles)
                    + (l3_blocks - held_steps) * max(step, matrix_unit_cycles)
                    + store
                )
            else:
                load = input_load + weight_load
                store_steps = output_tiles - 1
                accum_steps = l3_blocks - 2 * store_steps

                prefetch = load + tail_load
                compute = max(matrix_unit_cycles, prefetch) + matrix_unit_cycles
                dma = (
                    max(matrix_unit_cycles + vector_unit_cycles, prefetch)
                    + store
                    + load
                )
                total_time = (
                    load
                    + accum_steps * max(load, matrix_unit_cycles)
                    + store_steps * max(compute, dma)
                    + vector_unit_cycles
                    + store
                )
        else:
            total_time = (
                l3_blocks * matrix_unit_cycles
                + weight_steps * weight_load
                + input_steps * input_load
                + output_tiles * (tail_load + store)
            )
            if num_k > 1:
                total_time += output_tiles * vector_unit_cycles

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
        # last two dims, flipped when it is stored CK (rank-agnostic, so a
        # batched (B, K, N) weight reads K/N, not the batch dim).
        if weight_is_ck(node):
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
            runtime_tolerance=tiler.runtime_tolerance,
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
    has_tail=False,
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
        has_tail: The anchor carries a fused post-op, so its reduction drains
            through the vector unit (see ``RuntimeCalculator``).

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

    mx_out = isinstance(out_dtype, (list, tuple))
    of_dtype = out_dtype[-1] if mx_out else out_dtype
    if_bits = _node_dtype_bits(node.args[0])
    fl_bits = _node_dtype_bits(node.args[1])
    of_bits = get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
    if_scale_bits = _node_dtype_bits(node.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(node.kwargs.get("weight_scale"), 0)
    of_scale_bits = get_dtype_width(out_dtype[-2]) if mx_out else 0

    logger.info(
        f"[interstellar] {node.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil} | "
        f"if={if_bits}b fl={fl_bits}b of={of_bits}b "
        f"if_scale={if_scale_bits}b fl_scale={fl_scale_bits}b "
        f"bs={node.kwargs.get('block_size')}"
    )

    sram_bandwidth = min(tiler.config.pe_array_size) * if_bits

    # The node's own widths, so the timing model and ``make_size_fn`` size the
    # same operands.
    rc = RuntimeCalculator(
        if_bits,
        fl_bits,
        of_bits,
        PSUM_BITS,
        tiler.config.double_buffered_accum_buffer,
        sram_bandwidth,
        tiler.config.bytes_per_cycle,
        tiler.config.access_latency_cycles,
        double_buffered_l2=tiler.config.double_buffered_l2,
        has_tail=has_tail,
        tail_specs=fused_specs,
        input_scale_width=if_scale_bits,
        weight_scale_width=fl_scale_bits,
        output_scale_width=of_scale_bits,
        scale_block_size=node.kwargs.get("block_size") or 1,
    )

    result = None
    for extra_sharing in range(4 + len(fused_specs)):
        layer.size_fn = make_size_fn(
            node,
            out_dtype,
            fused_specs,
            extra_sharing,
            oc_align,
            has_tail=has_tail,
            copies=2 if tiler.config.double_buffered_l2 else 1,
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
    per_tile_cycles = rc.matrix_unit_cycles(mapping)
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


def tiling_request(node, tiler):
    """``(key, anchor, out_dtype, fused_specs, oc_align, has_tail)`` — all
    of ``get_tiling``'s preparation, stopping short of the search itself.

    ``None`` when no interstellar run is needed: not a matrix op, a GEMV (which
    ``gemv_op_tiling`` searches instead), or an anchor that already carries an
    ``l2_tiling``.  ``get_tiling`` and the prefetch both go through this, so the
    cache key they compute cannot drift apart.
    """
    anchor = get_anchor_node(node)
    if not (is_conv2d(anchor) or is_linear(anchor) or is_matmul(anchor)):
        return None
    if is_fully_connected(anchor) or anchor.meta.get("l2_tiling") is not None:
        return None

    sub_gm = node.meta.get("submodule")
    if sub_gm is not None:
        # ``_fused_operand_specs`` reads the placeholders' shapes, and the
        # builder only shape-props the submodule just before it calls in.
        ShapeProp(sub_gm).propagate(
            *(n.value.clone() for n in node.all_input_nodes)
        )
        # Fused-submodule placeholders lack the quant ``meta['dtype']``; copy it
        # from the outer ``all_input_nodes``
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
    if sub_gm is not None and not is_conv2d(anchor):
        nodes = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        perm = trailing_mha_perm(nodes)
        if perm is not None and perm.value.ndim > anchor.value.ndim:
            oc_align = perm.value.shape[-1]

    # A bare anchor and a fused one map differently (the fused reduction
    # drains through the vector unit), so ``has_tail`` is part of the key.
    has_tail = sub_gm is not None
    key = _layer_cache_key(anchor, out_dtype, tuple(fused_specs)) + (
        oc_align,
        has_tail,
    )
    return key, anchor, out_dtype, fused_specs, oc_align, has_tail


# Populated in the parent before forking; a worker reads its job by index so
# only the index crosses the process boundary (the FX nodes are inherited).
_PREFETCH_JOBS = []


def _run_prefetch_job(index):
    """Run one prefetched search in a forked worker.  Returns ``(ok, result)``
    so a failure leaves the entry uncached and is re-raised by the serial path,
    where it carries its normal traceback."""
    _, anchor, out_dtype, fused_specs, oc_align, has_tail, tiler = (
        _PREFETCH_JOBS[index]
    )
    try:
        return True, run_interstellar(
            anchor,
            tiler,
            out_dtype=out_dtype,
            fused_specs=fused_specs,
            oc_align=oc_align,
            has_tail=has_tail,
        )
    except Exception:
        return False, None


def prefetch_tilings(nodes, tiler):
    """Map every node's layer up front, concurrently, into ``tiler.cache``.

    Each search is independent and single-threaded Python, so they scale across
    processes; ``fork`` lets a worker read the FX node it inherited instead of
    marshalling it, which matters because ``Layer.size_fn`` is a closure and
    cannot be pickled.  A key that does not match the one ``get_tiling``
    recomputes during the build simply misses and is redone serially — a stale
    key costs time, never correctness.
    """
    global _PREFETCH_JOBS

    jobs = {}
    for node in nodes:
        request = tiling_request(node, tiler)
        if request is None or request[0] in jobs or request[0] in tiler.cache:
            continue
        jobs[request[0]] = request + (tiler,)

    if len(jobs) < 2:
        return

    _PREFETCH_JOBS = list(jobs.values())
    start = time.perf_counter()
    # Small cap: the runner already forks per design point, so an uncapped
    # pool multiplies to jobs x cpu_count.  VOYAGER_TILING_JOBS overrides.
    workers = min(
        len(_PREFETCH_JOBS),
        int(os.environ.get("VOYAGER_TILING_JOBS", "4")),
        os.cpu_count() or 1,
    )
    # Keep the parent heap out of the workers' GC so fork stays copy-on-write.
    gc.freeze()
    try:
        context = multiprocessing.get_context("fork")
        with context.Pool(workers) as pool:
            results = pool.map(_run_prefetch_job, range(len(_PREFETCH_JOBS)))
    except Exception as e:
        logger.warning(
            "[tiling] parallel prefetch failed (%s); going serial", e
        )
        _PREFETCH_JOBS = []
        return
    finally:
        gc.unfreeze()

    cached = 0
    for job, (ok, result) in zip(_PREFETCH_JOBS, results):
        if ok:
            tiler.cache[job[0]] = result
            cached += 1
    _PREFETCH_JOBS = []
    logger.info(
        "[tiling] prefetched %d/%d mappings in %.2fs",
        cached,
        len(results),
        time.perf_counter() - start,
    )


def _l3_order_from_mapping(mapping, canonical, loop_of):
    """``canonical`` resorted into the L3 loop order ``mapping`` chose.

    ``loop_orders[d][3]`` counts from the innermost (0), so sorting descending
    puts the outermost first, which is how the builders read a grid.  A loop
    that is empty at L3 carries interstellar's ``le.NUM - 1`` sentinel rather
    than a real rank, so ordering it would be meaningless -- those tags keep
    their canonical slot and only the genuinely tiled ones are permuted.  A
    size-1 grid dim is inert anyway: the scheduler skips it when it looks for
    the innermost tiled dim.
    """
    orders, blockings = mapping.loop_orders, mapping.loop_blockings
    if blockings[le.IC][3] > 1 and orders[le.IC][3] != 0:
        # The reduction kernels accumulate across *consecutive* steps, so a
        # split reduction has to be innermost.  The schedule hint pins it
        # (``IC`` level3 order 0); shout rather than emit a wrong nest.
        raise ValueError(
            "interstellar returned a split L3 reduction that is not "
            f"innermost (IC order {orders[le.IC][3]}); the builders cannot "
            "emit that nest"
        )
    tiled = [t for t in canonical if blockings[loop_of[t]][3] > 1]
    ranked = iter(sorted(tiled, key=lambda t: -orders[loop_of[t]][3]))
    return tuple(next(ranked) if t in tiled else t for t in canonical)


def get_tiling(node, tiler=None):
    """``(counts, l3_order)`` for a GEMM/conv ``node`` (standalone or fused
    ``call_module``); ``counts`` is ``None`` for a node that is not a matrix op
    / is untiled / was skipped.

    ``counts`` holds the per-dim tile counts: conv -> ``(n_y, n_x, n_k, n_c)``;
    gemm -> ``(batch.., n_m, n_n, n_k)`` — the output-spatial / M / N counts
    plus the reduction count last (``n_c`` for conv, ``n_k`` for gemm; the
    builder's ``num_k``).  The builder derives the tile sizes as
    ``full_dim // count``.

    ``l3_order`` permutes the builder's non-reduction grid dims, outermost to
    innermost (a permutation of ``CONV_L3_ORDER`` / ``GEMM_L3_ORDER``);
    ``None`` asks for the canonical order.

    Prefers the anchor's ``l2_tiling`` (the attention builders' explicit
    output-dim factors; the reduction is kept whole / decomposed away, so its
    factor is 1).  ``l2_tiling`` may carry the reduction factor explicitly — a
    3-tuple gemm ``(n_m, n_n, n_k)`` / a 5-tuple conv ``(n_N, n_k, n_y, n_x,
    n_c)`` — to drive a ``num_k > 1`` reduction sweep.  Otherwise searches: a
    GEMV through ``gemv_op_tiling``, everything else through interstellar via
    ``tiler`` (caching each layer's mapping).  Neither tiles the leading batch
    dims (e.g. attention heads); the builder loops them, so their counts are the
    full extent (one tile per batch element).
    """
    anchor = get_anchor_node(node)
    is_conv = is_conv2d(anchor)
    if not (is_conv or is_linear(anchor) or is_matmul(anchor)):
        return None, None

    # Neither search tiles the leading batch dims (e.g. attention heads); the
    # builder loops them, so emit a full-extent count -- one tile per batch
    # element.
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
            return (ny, nx, nk, nc), None
        if len(tiling) not in (2, 3):
            raise ValueError(
                f"{anchor.name} tiling {tiling} must be 2 or 3 elements"
            )
        nm, nn, *nk = tiling
        nk = nk[0] if nk else 1
        return gemm_batch + (nm, nn, nk), None

    if tiler is None:
        return None, None

    # Interstellar maps a systolic array and skips a batch-1 GEMM; that one runs
    # on the vector unit and has a search of its own.
    if is_fully_connected(anchor):
        return gemm_batch + gemv_op_tiling(node, tiler.config), None

    key, anchor, out_dtype, fused_specs, oc_align, has_tail = tiling_request(
        node, tiler
    )
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
            has_tail=has_tail,
        )
        logger.info(
            "[tiling] %s: interstellar took %.2fs",
            anchor.name,
            time.perf_counter() - t0,
        )
        # cache None too (skipped layers)
        tiler.cache[key] = (mapping, per_tile_cycles, access_list)
    if mapping is None:
        return None, None

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
        order = _l3_order_from_mapping(mapping, CONV_L3_ORDER, _CONV_LOOP)
        return (b[le.OY][3], b[le.OX][3], b[le.OC][3], b[le.IC][3]), order
    order = _l3_order_from_mapping(mapping, GEMM_L3_ORDER, _GEMM_LOOP)
    return gemm_batch + (b[le.OX][3], b[le.OC][3], b[le.IC][3]), order
