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
import itertools
import logging
import math
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    _pair,
    get_anchor_node,
    get_arg_value,
    is_bmm,
    is_conv2d,
    is_depthwise_conv,
    is_fully_connected,
    is_gemm_op,
    quant_param_arg_nodes,
    stream_breaking_quantize,
    trailing_mha_perm,
    weight_is_ck,
    weight_transforms,
)
from voyager_compiler.codegen.transform.tiling.cost import (
    BANK_SWITCH_CYCLES,
    _node_dtype_bits,
    _step_classes,
    _sweep_cycles,
    attention_tile_latency,
    gemv_compute_cycles,
    get_dtype_width,
    strided_bank_walk,
)
from voyager_compiler.codegen.transform.tiling.search import (
    DEFAULT_RUNTIME_TOLERANCE,
    _attention_sram_bytes,
    _attention_tiles,
    _divisors_descending,
    _operand_placeholders,
    attention_head_pad,
    gemv_op_tiling,
)
from voyager_compiler.ops.layout import NCHW_TO_NHWC, OIHW_TO_HWIO, unproject
from voyager_compiler.shape_prop import ShapeProp, set_node_value

logger = logging.getLogger(__name__)
le = interstellar.le

# Finished output vectors the matrix -> vector path holds before a
# single-buffered accumulator's array feels the tail's drain rate: the
# matrix processor's output FIFO (8), the vector pipeline's input FIFO (9)
# and the stages between.  Measured on the E4M3x16x16 SoC as 20 (MobileBERT
# GEMMs) to 40 (ResNet18) vectors; either way the error is under ~60 cycles
# per L3 step.
OUTPUT_SLACK = 24

# Cycles the SpMM unit spends per row of each PE-array-wide pass on top of
# the row's outliers: it drains and restarts its accumulator ring between
# rows.  Measured on the Sphinx SoC: 7.3-7.5.
SPMM_ROW_CYCLES = 8

# Block-scale rows the SpMM unit's weight-scale buffer holds
# (``DoubleBuffer<32>`` in ``SpMMUnit.h``, fixed in the Sphinx silicon).  A
# streaming weight tile takes one row per K block per inner OC pass, and the
# hardware wraps the address rather than checking it.
SPMM_SCALE_ROWS = 32


def spmm_scale_rows(mapping):
    """Weight-scale buffer rows ``mapping``'s tile takes in the SpMM unit:
    its K blocks (the L1 and L2 IC blockings; the PE level is the array)
    times its L1 OC passes -- ``C * K0`` in the toolchain's ``SpMM.h``."""
    ic, oc = mapping.loop_blockings[le.IC], mapping.loop_blockings[le.OC]
    return ic[1] * ic[2] * oc[1]


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


@dataclass(frozen=True)
class TileConstraint:
    """Loop extents a search must land on at the scratchpad level, keyed by
    interstellar loop (``le.IC``, ``le.OC``, ...).  ``exact`` pins a loop's
    tile to one extent, ``multiple`` requires it to be a whole number of the
    value; both are ``((loop, value), ...)`` so a constraint can key the
    tiling cache.  A GEMM feeding an MHA relayout takes a ``le.OC``
    multiple of the head; a CSR consumer an exact ``le.IC`` (the slice
    width) and a CSR producer a ``le.OC`` multiple of it (its column tile
    is cut into slices), which is how ``plan_csr_slices`` searches the two
    under one width.
    """

    exact: Tuple[Tuple[int, int], ...] = ()
    multiple: Tuple[Tuple[int, int], ...] = ()

    def allows(self, extent) -> bool:
        """Whether tile extents ``extent(loop)`` satisfy every entry."""
        return all(extent(loop) == v for loop, v in self.exact) and all(
            extent(loop) % v == 0 for loop, v in self.multiple
        )

    def merged(self, other):
        """This constraint and ``other`` (``None`` = nothing) together."""
        if other is None:
            return self
        return TileConstraint(
            tuple(sorted(self.exact + other.exact)),
            tuple(sorted(self.multiple + other.multiple)),
        )


@dataclass
class TilerContext:
    """The interstellar architecture + run options, built once and shared by
    every builder so each can map its anchor node on demand.  ``arch`` /
    ``schedule`` are built from ``config``; ``cache`` is per-run
    memoization, and ``constraints`` the :class:`TileConstraint` a joint
    optimization settled on for an anchor, which its search then runs
    under."""

    arch: object
    schedule: object
    config: object  # AcceleratorConfig
    runtime_tolerance: float = DEFAULT_RUNTIME_TOLERANCE
    cache: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)


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
    ``usable_scratchpad_size``, the scratchpad above whatever
    ``scratchpad_offset`` reserves, and a ping-ponged source is charged one
    bank per slot (``size_fn``) rather than the capacity being halved.
    Halving cannot express a buffer that is allocated only once -- the
    reduction scratch -- and it is that omission that let the tiler hand back
    tilings the allocator could not place.

    ``config`` carries physical units (GB); the interstellar model wants bytes,
    so ``dram_size`` is scaled to bytes here (the ``dram_bandwidth`` conversion
    to bytes/cycle lives on ``config.bytes_per_cycle``, read at run time).
    """
    ic_dim, oc_dim = config.pe_array_size

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                config.input_buffer_size * ic_dim,
                config.accum_buffer_size * oc_dim,
                config.weight_buffer_size * oc_dim,
            ],
            [config.usable_scratchpad_size],
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
        bank_size_list=[None, None, config.bank_size, None],
    )

    # L1 IC is outermost; the inner order is pinned to FY > FX > OY > OX.
    # OX/OY innermost is never slower -- the L1 sweep costs
    # ``max(loading, reused_tile) * remaining``, monotone in the reused
    # tile -- and the arrangement of the loops above them ties on both
    # runtime and energy, so FX/FY are fixed to the representative the
    # search's first-seen tie-break picked anyway.  FX=2/FY=3 assumes a
    # square kernel (equal FX/FY blocking); a non-square model may prefer
    # them swapped.
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
            "OX": {
                "level1": {"order": 0},
            },
            "OY": {
                "level1": {"order": 1},
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 2},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level1": {"order": 3},
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


def _layer_cache_key(node):
    """A hashable key for what the *node alone* says about its interstellar
    mapping: op, operand / output shapes, the conv stride/padding/dilation, the
    operand + scale + outlier-CSR element widths and the microscaling block
    size.  Identical layers thus share one optimizer run.

    The architecture is fixed for a run, so it is left out; so is anything the
    caller knows and the node does not -- the output dtype, the fused post-op
    operands -- which the caller appends to what it gets back."""
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
        _node_dtype_bits(node.kwargs.get("A_indptr"), 0),
        _node_dtype_bits(node.kwargs.get("A_data"), 0),
        _node_dtype_bits(node.kwargs.get("A_indices"), 0),
        node.kwargs.get("block_size"),
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


def _fused_operand_nodes(node, anchor):
    """The fused post-op operand placeholders of ``node``'s submodule, in the
    order ``_fused_operand_specs`` sizes them — so a ``("fused", i)`` role in
    a bank partition names ``_fused_operand_nodes(node, anchor)[i]``.

    A post-op operand is any submodule placeholder that is *not* one of the
    anchor's own operands (act / weight / scales, traced through any input
    dequantize / reshape — those are counted by the interstellar ``Layer``) and
    is not a codebook / qmap or scalar.  Defining it by exclusion catches an
    operand fed through a ``dequantize`` (e.g. the attention mask).  The
    submodule is already ShapeProp'd (placeholders carry ``.value``).  Empty
    for a bare node.
    """
    submod = node.meta.get("submodule")
    if submod is None:
        return []

    anchor_operands = set(_operand_placeholders(anchor))
    codebooks = set()
    for n in submod.graph.nodes:
        codebooks |= quant_param_arg_nodes(n)

    return [
        p
        for p in submod.graph.nodes
        if p.op == "placeholder"
        and p.value.numel() != 1
        and p not in anchor_operands
        and p not in codebooks
    ]


def _fused_operand_specs(node, anchor):
    """Per fused post-op operand of a fused ``call_module`` ``node``: a
    ``(dims, dtype_bits)`` pair, where ``dims`` are the interstellar output loop
    dims (a subset of ``ON/OC/OY/OX``) the operand is *tiled* along — broadcast
    (size-1) dims dropped, so its tile size is ``prod(out_tile[d] for d in
    dims)``.  Empty for a bare node, or one whose fused ops add no tiled tensor
    operand (codebooks / scalars don't count).  The operands themselves come
    from ``_fused_operand_nodes``, in the same order.
    """
    operands = _fused_operand_nodes(node, anchor)
    if not operands:
        return []

    pos_to_loop_dim = _output_pos_to_loop_dim(anchor)
    out_ndim = anchor.value.ndim

    specs = []
    for p in operands:
        op_shape = tuple(p.shape)
        offset = out_ndim - len(op_shape)  # right-align (broadcast)
        dims = tuple(
            pos_to_loop_dim[offset + i]
            for i, sz in enumerate(op_shape)
            if sz > 1
        )
        specs.append((dims, _node_dtype_bits(p)))
    return specs


# Positions of interstellar's (input, output, weight) byte triple.
_IF, _OF, _FL = 0, 1, 2

# The L3 loops each operand's tile spans.  A tile advances only when one of
# them turns, so an operand whose every entry is a single L3 step reads one
# tile for the whole sweep.
_IF_DIMS = (le.OX, le.OY, le.IC, le.ON)
_FL_DIMS = (le.OC, le.IC, le.FX, le.FY)
_OF_DIMS = (le.OC, le.OY, le.OX, le.ON)

_QUANTIZE_MX_OUTLIER = torch.ops.quantized_ops.quantize_mx_outlier.default


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
    constraint=None,
    has_tail=False,
    single_k_tail_extra_pass=False,
    tail_keeps_shape=False,
    scratch_regions=1,
    num_slots=1,
    batch=1,
    weight_batch=1,
    outlier_pct=0.0,
    out_outlier_pct=0.0,
    bank_width=None,
    vector_lanes=None,
):
    """Build a ``Layer.size_fn``: the bytes a tile occupies at a byte-pool
    level.

    Interstellar hands over element counts and the mapping; everything that
    turns those into bytes is policy and lives here -- element widths,
    microscaling scale tensors (one scale per ``block_size`` values), the bias
    and fused post-op operands interstellar knows nothing about, and how all of
    them are packed into banks.

    Operands are grouped one per bank ideally:

        input+csr | input_scale | weight+weight_scale+bias
                  | output+output_scale+csr staging | each fused operand

    An outlier CSR the GEMM consumes rides the input's bank -- both are
    DMA-written and compute-read at the same depth -- and the CSR a fused
    tail emits stages in the output's, beside the dense pair it stores by
    the same route.

    Each such source is ping-ponged, so it costs ``num_slots`` whole banks --
    the two halves live in *separate* banks, which is what the planner does and
    what lets a load overlap the compute reading the other half.  A split
    reduction with a fused tail also accumulates into a scratch buffer the
    builders allocate exactly once (``_ScratchSpec``); it is charged a single
    bank-aligned region on top.  So is the finished tile a tail that needs
    a pass of its own (``single_k_tail_extra_pass``) parks even for a
    single round, unless it keeps the tile's shape (``tail_keeps_shape``)
    and runs that pass in place on the output slot.  Leaving either out is
    what let the tiler return tilings ``plan_memory`` could not place.  A
    split reduction whose tail reads that tile back from scratch may take
    ``scratch_regions`` of them: a second region lets the read-back pass
    ride the finalize commit instead of running bare, so it is charged
    whenever it fits beside the sources, and the tile falls back to one
    region -- priced as the bare pass -- when it does not.

    A bank cannot be split between groups, so each group rounds up to a whole
    bank -- which puts a floor of ``num_slots * len(groups) * bank_size`` on the
    tile, however small it is.  While the groups' banks exceed the budget the
    two *smallest* are merged (a tiny scale tensor would otherwise waste a
    whole bank), so a tile is admitted with the least sharing that fits it
    and the cost model prices that sharing.  Only whole sources merge, never
    a source's own two slots -- each slot must keep its own bank.

    Every operand is sized the way ``tensor_alloc_bytes`` sizes it for the
    planner -- payload, the slack a final store beat overshoots by, aligned to
    ``bank_width`` -- and each operand of a shared bank is padded before they
    are summed, because the planner lays them out one after another.  Sizing
    a tile at its raw payload instead lets a group whose payload lands on a
    bank boundary claim one more bank in the plan than the search charged.

    ``fused_specs`` are the ``(dims, dtype_bits)`` pairs from
    ``_fused_operand_specs``.  ``bank_size is None`` -> no banking: just sum.

    The returned ``size_fn`` exposes the partition itself as
    ``size_fn.compute_groups``: ``_finish_search`` replays it for the winning
    mapping and stamps the surviving role partition for the memory planner,
    so the plan realizes exactly the banking the fit check priced.
    """
    if isinstance(out_dtype, (list, tuple)):
        of_scale_dtype, of_dtype = out_dtype[-2], out_dtype[-1]
    else:
        of_scale_dtype, of_dtype = None, out_dtype
    # A CSR-producing tail returns (data, indices, indptr, scale, inliers);
    # a staged entry is an outlier value and its column index, each its own
    # buffer.
    out_csr_data_bits, out_csr_index_bits = (
        (get_dtype_width(out_dtype[0]), get_dtype_width(out_dtype[1]))
        if isinstance(out_dtype, (list, tuple)) and len(out_dtype) == 5
        else (0, 0)
    )
    if_bits = _node_dtype_bits(node.args[0])
    fl_bits = _node_dtype_bits(node.args[1])
    of_bits = get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
    bias_bits = _node_dtype_bits(get_arg_value(node, 2, "bias", None), 0)
    if_scale_bits = _node_dtype_bits(node.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(node.kwargs.get("weight_scale"), 0)
    of_scale_bits = get_dtype_width(of_scale_dtype) if of_scale_dtype else 0
    stage_bits = get_dtype_width(node.value.dtype)
    block_size = node.kwargs.get("block_size") or 1
    # A gathered-CSR entry: the outlier value and its column index, each
    # staged in its own buffer.
    csr_data_bits = _node_dtype_bits(node.kwargs.get("A_data"), 0)
    csr_index_bits = _node_dtype_bits(node.kwargs.get("A_indices"), 0)
    # A CSR consumer streams its weight tile through the SpMM unit.
    is_spmm = node.kwargs.get("A_data") is not None

    def _align(size):
        """Round ``size`` up to a whole ``bank_width`` store word."""
        if not bank_width:
            return size
        return math.ceil(size / bank_width) * bank_width

    def _alloc_bytes(count, bits):
        """Bytes an on-chip buffer of ``count`` ``bits``-wide elements takes.

        Mirrors ``tensor_alloc_bytes``, which is what the memory planner
        allocates through: the payload, plus the slack a final store beat of
        ``vector_lanes`` values overshoots the payload by, aligned to a whole
        store word.
        """
        if not bits or count <= 0:
            return 0.0
        size = math.ceil(count * bits / 8.0)
        if bank_width and vector_lanes:
            beat = math.ceil(vector_lanes * bits / 8.0)
            size += _align(beat) - beat
        return float(_align(size))

    def _scale_bytes(count, bits):
        """Bytes the scales of ``count`` values take: one per block."""
        return _alloc_bytes(count / block_size, bits)

    def compute_groups(
        counts, point, level, partitioning_accum, bank_size, num_banks
    ):
        """The bank partition of one candidate tile: the merged ``(bytes,
        kind, slots, roles)`` groups, the scratch region's bytes and the
        regions it takes, or ``(None, 0.0, 1)`` for a vetoed tile.
        ``roles`` is the set of operand
        roles sharing the bank (``"input"``/``"csr"``, ``"input_scale"``,
        ``"weight"``/``"weight_scale"``/``"bias"``, ``"output"``,
        ``("fused", i)``); a merge unions the two sets.  ``size_fn`` sums
        exactly these groups, so the stamped partition and the fit check can
        never disagree."""
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

        # Veto a tile off the pinned loop extents: an OC tile that splits an
        # attention head (the MHA relayout must store whole heads), a CSR
        # slice width the coupled ops agreed on.
        if constraint is not None and not constraint.allows(extent):
            return None, 0.0, 1
        # TEMPORARY WORKAROUND: veto a tile whose block scales overflow the
        # SpMM unit's fixed 32-row buffer (the RTL wraps and applies the
        # wrong scales).  Drop once the depth is an accelerator parameter.
        if is_spmm and spmm_scale_rows(point) > SPMM_SCALE_ROWS:
            return None, 0.0, 1

        # The output is a partial sum in the anchor's dtype until IC is fully
        # reduced; only the final value carries an output scale.
        out_bits = stage_bits if is_psum else of_bits
        of_scale = 0.0 if is_psum else _scale_bytes(of_count, of_scale_bits)
        bias = _alloc_bytes(extent(le.OC), bias_bits)
        # A CSR the tail emits stages in the output's bank before its packed
        # stores: a block's (value, index) pairs and its row pointers.  The
        # quantize sees the output tile, so the budget follows it.
        staged_csr = 0.0
        if out_csr_data_bits:
            rows = of_count / max(1.0, extent(le.OC))
            staged = of_count * out_outlier_pct
            staged_csr = (
                _alloc_bytes(staged, out_csr_data_bits)
                + _alloc_bytes(staged, out_csr_index_bits)
                + _alloc_bytes(rows + 1, 32)
            )

        def slots(dims, distinct):
            """Banks an operand spanning ``dims`` needs: a second holds the
            next tile, so an operand with one tile over the whole sweep needs
            one.  The mapping covers a single batch element -- the builder
            loops the rest -- so the sweep's tile count is the L3 trips times
            ``distinct``, the operand's distinct tiles over that batch loop
            (``_batch_loads`` counts fetches the same way).  The builders'
            ``PipelinedKernel._num_slots`` resolves the depth from the full
            grid, so this never charges less than is allocated."""
            tiles = distinct
            for d in dims:
                tiles *= point.loop_blocking(d)[3]
            return num_slots if tiles > 1 else 1

        # The gathered CSR shares the input's bank: a block's (value, index)
        # pairs, at the fraction of the activation the stream is sized for.
        # Its staging ping-pongs exactly when the input does -- a sweep that
        # gathers once keeps one slot (``_SparseGemm.forward``).
        gathered = if_count * outlier_pct
        groups = [
            (
                _alloc_bytes(if_count, if_bits)
                + _alloc_bytes(gathered, csr_data_bits)
                + _alloc_bytes(gathered, csr_index_bits),
                _IF,
                slots(_IF_DIMS, batch),
                {"input", "csr"},
            ),
            (
                _scale_bytes(if_count, if_scale_bits),
                _IF,
                slots(_IF_DIMS, batch),
                {"input_scale"},
            ),
            (
                _alloc_bytes(fl_count, fl_bits)
                + _scale_bytes(fl_count, fl_scale_bits)
                + bias,
                _FL,
                slots(_FL_DIMS, weight_batch),
                {"weight", "weight_scale", "bias"},
            ),
            (
                _alloc_bytes(of_count, out_bits) + of_scale + staged_csr,
                _OF,
                slots(_OF_DIMS, batch),
                {"output"},
            ),
        ]
        for i, (dims, bits) in enumerate(fused_specs):
            count = 1
            for d in dims:
                count *= extent(d)
            distinct = batch if le.ON in dims else 1
            groups.append(
                (
                    _alloc_bytes(count, bits),
                    _OF,
                    slots(dims, distinct),
                    {("fused", i)},
                )
            )

        # An absent operand (no scale, no bias) occupies no bank.
        groups = [g for g in groups if g[0] > 0]

        # A reduction split across L3 steps accumulates into a scratch buffer
        # the builders allocate once for the whole kernel, not per ping-pong
        # half -- so it is charged one region, outside ``num_slots``.  A
        # shape-changing extra pass parks even a single round's tile.
        if has_tail and (
            point.loop_blocking(le.IC)[3] > 1
            or (single_k_tail_extra_pass and not tail_keeps_shape)
        ):
            scratch = _alloc_bytes(of_count, stage_bits)
        else:
            scratch = 0.0
        # Only a split reduction's read-back tail has a second region to
        # gain; a single staged round keeps one.
        regions = 1
        if scratch and point.loop_blocking(le.IC)[3] > 1:
            regions = scratch_regions

        if not bank_size:
            return groups, scratch, regions

        scratch_banks = math.ceil(scratch / bank_size) if scratch else 0

        def banks(groups):
            return sum(
                n * math.ceil(size / bank_size) for size, _, n, _ in groups
            )

        def merge_to_fit(budget):
            """The groups with their two smallest merged until their banks
            fit ``budget``, and whether they do -- one group may still not."""
            merged = list(groups)
            while len(merged) > 1 and banks(merged) > budget:
                merged.sort(key=lambda g: g[0])
                (s0, k0, n0, r0), (s1, k1, n1, r1) = merged[0], merged[1]
                # Charge the shared bank to the larger member's operand, and
                # to the deeper pipeline: one buffer holding both has to
                # ping-pong if either of them does.
                merged = [
                    (s0 + s1, k0 if s0 >= s1 else k1, max(n0, n1), r0 | r1)
                ] + merged[2:]
            return merged, banks(merged) <= budget

        # Banks left for the ping-ponged sources, after the scratch regions.
        # Without a bank count the partition is one source per bank, so
        # nothing merges.  A second scratch region is kept only when the
        # sources still fit beside it.
        while True:
            budget = (num_banks or math.inf) - regions * scratch_banks
            merged, fits = merge_to_fit(budget)
            if fits or regions == 1:
                return merged, scratch, regions
            regions -= 1

    def size_fn(counts, point, level, partitioning_accum, bank_size, num_banks):
        groups, scratch, regions = compute_groups(
            counts, point, level, partitioning_accum, bank_size, num_banks
        )
        if groups is None:
            return (float("inf"),) * 3

        out = [0.0, 0.0, 0.0]
        if not bank_size:
            for size, kind, n, _ in groups:
                out[kind] += n * size
            out[_OF] += regions * scratch
            return tuple(out)

        for size, kind, n, _ in groups:
            out[kind] += n * math.ceil(size / bank_size) * bank_size
        if scratch:
            out[_OF] += regions * math.ceil(scratch / bank_size) * bank_size
        return tuple(out)

    size_fn.compute_groups = compute_groups
    return size_fn


class RuntimeCalculator:
    """Runtime cost model for a 4-level hierarchy (PE / L1 / L2 / DRAM).

    A mapping is priced in cycles as its L3 grid sweep.  With a
    double-buffered L2 each step costs the slower of its DRAM transfers and
    its compute, otherwise their sum, and the sweep is framed by the
    un-overlapped first load and last store; a split reduction adds its
    accumulate steps and the tail pass that finishes each output tile.  A
    DRAM transfer costs its bytes at ``dram_bandwidth`` plus one access
    latency, block scales being a transfer of their own, and the batch loop
    outside the mapping shares a weight tile among the elements of one
    group.  Energy is interstellar's cost model, not this one.

    The compute of an L2 block is the busier of the matrix unit and the
    scratchpad bus.  The matrix unit charges the systolic passes, each weight
    tile costing the longer of its loading and the rows streamed through it,
    plus the back-pressure of a single-buffered accumulator against
    ``OUTPUT_SLACK`` of buffering.  The bus charges the words every operand
    role moves, summed per bank group of the planner's partition with the
    busiest bank setting the pace: input and weight rows in whole beats,
    packed as the toolchain packs them, with the read interface's lost cycle
    whenever an aligned request follows an unaligned one; block scales one
    word each; the output, the fused tail operands, the bias and the
    reduction scratch; and the round trip a stream pays when it changes bank
    on its bank-aligned tile buffer.  A weight tile held across the spatial
    loops is fetched once.  The SpMM unit adds a turnaround per row visit and
    the outlier rows it gathers at the block's K depth, most of them a bank
    switch when the weight tile spans several banks.  A tail pass of its own
    costs the vector unit's lane rate or its bank words, whichever is
    slower.  The ramps of a sweep, the first buffer fill, the systolic skew
    and the last drain, are spread over the ops the sweep dispatches.

    Not modeled are the per-op costs (parameter load, deserialisation, the
    start/done handshake, the drain between uncommitted ops, the host's
    dispatch time), so an op that runs alone pays its ramps in full where
    the spreading charges a share; the cycle-exact datapath, the systolic
    skew being a constant of the array dims; a stream-breaking
    ``quantize_mx`` tail, charged as a staged region rather than a drained
    pass; a conv input tile's halo; more than one port width, every operand
    moving at ``sram_bandwidth`` over one bus per bank; the block scales'
    bank switches; and the outlier density of an individual tile, priced at
    the layer's average.
    """

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
        outlier_rate: float = 0.0,
        batch: int = 1,
        weight_batch: Optional[int] = None,
        has_tail: bool = False,
        single_k_tail_extra_pass: bool = False,
        split_k_tail_extra_pass: bool = False,
        tail_keeps_shape: bool = False,
        tail_specs=(),
        input_scale_width: int = 0,
        weight_scale_width: int = 0,
        output_scale_width: int = 0,
        scale_block_size: int = 1,
        bias_width: int = 0,
        stride: Tuple[int, int] = (1, 1),
        bank_size: Optional[int] = None,
        weight_transposed: bool = False,
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
        self.outlier_rate = outlier_rate
        self.batch = batch
        self.weight_batch = batch if weight_batch is None else weight_batch
        self.has_tail = has_tail
        self.single_k_tail_extra_pass = single_k_tail_extra_pass
        self.split_k_tail_extra_pass = split_k_tail_extra_pass
        self.tail_keeps_shape = tail_keeps_shape
        self.tail_specs = tuple(tail_specs)
        self.input_scale_width = input_scale_width
        self.weight_scale_width = weight_scale_width
        self.output_scale_width = output_scale_width
        self.scale_block_size = scale_block_size
        self.bias_width = bias_width
        self.stride = stride
        self.bank_size = bank_size
        self.weight_transposed = weight_transposed
        self.dram_bytes = {}

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

    def _bank_cycles(self, words, bank_groups):
        """Cycles the busiest scratchpad bank spends moving ``words`` -- bus
        words per operand role -- when the roles sharing a bank
        (``bank_groups``: the search's partition as role sets, ``None`` =
        nothing shares) queue on its single port.  A role the partition does
        not name keeps a port of its own.
        """
        placed = set()
        busiest = 0
        for roles in bank_groups or ():
            busiest = max(busiest, sum(words.get(role, 0) for role in roles))
            placed.update(roles)
        loose = [count for role, count in words.items() if role not in placed]
        return max([busiest, *loose])

    def _request_words(self, requests, row_elems, bits, loop_bound, pitch):
        """Bus words ``requests`` fetches of one ``row_elems``-element row
        take.  A request is served in whole beats, so a row that is not a
        multiple of the port costs more than its bytes.  The controller packs
        the rows that fill whole beats into one request when the L1 loop
        that walks them, of ``loop_bound`` steps, divides into that many
        (the toolchain's ``get_packing_factor``); a ``loop_bound`` of 0 never
        packs.  Consecutive requests are ``pitch`` bytes apart, the row
        pitch of the tile buffer.  The SoC's read interface flushes the tail
        of a request that started inside a beat in a cycle of its own and
        holds the next request's first beat when that one starts on a beat
        boundary, so a pitch that is not a multiple of the beat cycles the
        requests through the beat offsets and costs one cycle per aligned
        request that follows an unaligned one."""
        if not bits or requests <= 0:
            return 0
        row_bits = row_elems * bits
        pf = math.lcm(row_bits, self.sram_bandwidth) // row_bits
        rows_per_request = pf if loop_bound and loop_bound % pf == 0 else 1
        request_words = math.ceil(
            rows_per_request * row_bits / self.sram_bandwidth
        )
        count = math.ceil(requests / rows_per_request)
        beat = self.sram_bandwidth // 8
        misalignment = rows_per_request * pitch % beat
        flushes = 0
        if misalignment:
            flushes = count * math.gcd(misalignment, beat) / beat
        return count * request_words + flushes

    def _bus_words(self, count, bits, bandwidth=None):
        """Bus words ``count`` elements of ``bits`` occupy at the bank's full
        width, or at ``bandwidth`` bytes per cycle."""
        if not bits or count <= 0:
            return 0
        return math.ceil(
            count * bits / 8 / (bandwidth or self.sram_bandwidth / 8)
        )

    @staticmethod
    def _extent(mapping, loop, level):
        """Elements along ``loop`` in one tile at ``level``: the blockings
        through that level times the PE-array partition."""
        extent = mapping.loop_partitionings[loop][0]
        for blocking in mapping.loop_blockings[loop][1 : level + 1]:
            extent *= blocking
        return extent

    def _load_words(self, mapping):
        """Bus words one L1 tile's every-round operands take, per role: the
        input and its block scales, the weight and its scales.  The input
        and the weight arrive one PE-array row per request
        (``_request_words``), packed several rows per request when the L1
        loop that walks them allows it -- never for a transposed weight;
        the weight scales one PE-array row of scales per request, never
        packed.  An input scale is delivered one
        per bus word however narrow it is; every outlier in the input tile
        gathers one more weight row on top of the dense weight tile."""
        ext = lambda loop: self._extent(mapping, loop, 1)
        rows = ext(le.OX) * ext(le.OY) * ext(le.ON)
        depth = ext(le.IC)
        taps = ext(le.FX) * ext(le.FY)
        gathered_rows = rows * depth * self.outlier_rate
        blockings = mapping.loop_blockings
        ic_unroll = mapping.loop_partitionings[le.IC][0]
        oc_unroll = mapping.loop_partitionings[le.OC][0]
        weight_loop = 0 if self.weight_transposed else blockings[le.OC][1]
        # The tile buffers are [rows, IC] for the input and [IC, OC] for the
        # weight and its scales: a request walks one row of each.
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        input_pitch = ic3 * self.input_dtype_width // 8
        weight_pitch = oc3 * self.weight_dtype_width // 8
        scale_pitch = oc3 * self.weight_scale_width // 8
        words = {
            "input": self._request_words(
                rows * depth / ic_unroll,
                ic_unroll,
                self.input_dtype_width,
                blockings[le.IC][1],
                input_pitch,
            ),
            "weight": self._request_words(
                (depth * taps + gathered_rows) * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_dtype_width,
                weight_loop,
                weight_pitch,
            ),
            "weight_scale": self._request_words(
                depth * taps / self.scale_block_size * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_scale_width,
                0,
                scale_pitch,
            ),
        }
        if self.input_scale_width:
            words["input_scale"] = math.ceil(
                rows * depth / self.scale_block_size
            )
        return words

    def _tail_words(self, mapping, level):
        """Bus words the tail's operands take for one tile at ``level`` (1 =
        an L1 output tile, 2 = the whole L3 output tile), per role: the
        finished output with its block scales -- each scale leaves in a bus
        word of its own, however narrow, as the input scales arrive -- and
        each fused operand over the output dims it is tiled along.  The tail
        rides on the array's output one vector (a row of ``oc_dim`` values)
        at a time and fetches a fused operand in one unpacked request per
        vector, so an operand narrower than a bus word still costs a word
        per vector."""
        ext = lambda loop: self._extent(mapping, loop, level)
        out = ext(le.OC) * ext(le.OY) * ext(le.OX) * ext(le.ON)
        words = {"output": self._bus_words(out, self.output_dtype_width)}
        if self.output_scale_width:
            words["output"] += math.ceil(out / self.scale_block_size)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        oc2 = self._extent(mapping, le.OC, 2)
        for i, (dims, bits) in enumerate(self.tail_specs):
            if le.OC in dims:
                rows = math.prod(ext(dim) for dim in dims if dim != le.OC)
                vectors = rows * math.ceil(ext(le.OC) / oc_dim)
                words[("fused", i)] = self._request_words(
                    vectors, oc_dim, bits, 0, math.ceil(oc2 * bits / 8)
                )
            else:
                words[("fused", i)] = self._bus_words(
                    math.prod(ext(dim) for dim in dims), bits
                )
        return words

    def _stream_switches(self, mapping, key_loops, walk_of, held_loops=()):
        """Compose L1 bank walks in the emitted L2 loop order.

        ``walk_of(idx)`` returns ``(switches, first_bank, last_bank)`` for
        one L1 request nest. Key loops change its addresses; other loops
        repeat it, unless the controller holds the operand across them.
        Fold repeats at their actual nesting depth, including loops between
        two key loops, so both rewinds and inter-block transitions survive.
        """
        blockings, orders = mapping.loop_blockings, mapping.loop_orders
        nest = sorted(
            (
                i
                for i in range(le.NUM)
                if blockings[i][2] > 1 and i not in held_loops
            ),
            key=lambda i: -orders[i][2],
        )

        def walk(depth, idx):
            if depth == len(nest):
                return walk_of(idx)
            loop = nest[depth]
            count = blockings[loop][2]
            if loop not in key_loops:
                inner, start, end = walk(depth + 1, idx)
                return count * inner + (count - 1) * (start != end), start, end
            total = 0
            first = last = None
            for index in range(count):
                idx[loop] = index
                inner, start, end = walk(depth + 1, idx)
                total += inner + (last is not None and start != last)
                if first is None:
                    first = start
                last = end
            return total, first, last

        return walk(0, {})[0]

    def _bank_switch_cycles(self, mapping):
        """Cycles one L3 step's input and weight streams lose to bank
        switches, per role (``BANK_SWITCH_CYCLES`` each).

        Follow the mapping's L1 input order and the weight FY/FX/IC/OC scan.
        Packing combines adjacent channel groups into a request exactly
        when the toolchain permits it, matching _request_words. Buffers
        start on banks; input scales are not included.
        A weight tile held across spatial loops is not refetched by them.
        """
        if not self.bank_size:
            return {}
        b = mapping.loop_blockings
        orders = mapping.loop_orders
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        ic1 = self._extent(mapping, le.IC, 1)
        oc1 = self._extent(mapping, le.OC, 1)
        fy, fx = b[le.FY][1], b[le.FX][1]
        oy1, ox1 = b[le.OY][1], b[le.OX][1]
        hs, ws = self.stride
        y_in = (oy1 * b[le.OY][2] - 1) * hs + fy
        x_in = (ox1 * b[le.OX][2] - 1) * ws + fx
        pitch_in = ic3 * self.input_dtype_width / 8
        ic_dim = mapping.loop_partitionings[le.IC][0]
        oc_dim = mapping.loop_partitionings[le.OC][0]

        def packed_width(lanes, bits, count):
            row_bits = lanes * bits
            factor = math.lcm(row_bits, self.sram_bandwidth) // row_bits
            return lanes * (factor if count and count % factor == 0 else 1)

        input_chunk = packed_width(ic_dim, self.input_dtype_width, b[le.IC][1])
        input_order = sorted((le.IC, le.OY, le.OX), key=lambda i: -orders[i][1])

        def input_walk(idx):
            y0 = idx.get(le.OY, 0) * oy1 * hs
            x0 = idx.get(le.OX, 0) * ox1 * ws
            # Filter taps expand the spatial fetch; the controller disables
            # its separate L1 FX/FY/OC loops. Clip the last halo to the tile.
            sy, sx = (hs if fy == 1 else 1), (ws if fx == 1 else 1)
            ny = min(
                oy1 if fy == 1 else oy1 * hs + fy - 1, (y_in - 1 - y0) // sy + 1
            )
            nx = min(
                ox1 if fx == 1 else ox1 * ws + fx - 1, (x_in - 1 - x0) // sx + 1
            )
            width = input_chunk * self.input_dtype_width / 8
            scans = {
                le.IC: (ic1 // input_chunk, width),
                le.OY: (ny, sy * x_in * pitch_in),
                le.OX: (nx, sx * pitch_in),
            }
            loops = tuple(scans[i] for i in input_order)
            offset = (y0 * x_in + x0) * pitch_in
            offset += idx.get(le.IC, 0) * ic1 * self.input_dtype_width / 8
            return strided_bank_walk(loops, width, self.bank_size, offset)

        pitch_w = oc3 * self.weight_dtype_width / 8
        beat_w = oc1 * self.weight_dtype_width / 8
        weight_chunk = packed_width(
            oc_dim,
            self.weight_dtype_width,
            0 if self.weight_transposed else b[le.OC][1],
        )

        def weight_walk(idx):
            c0 = idx.get(le.IC, 0) * ic1
            k_off = idx.get(le.OC, 0) * beat_w
            width = weight_chunk * self.weight_dtype_width / 8
            loops = (
                (fy * fx, ic3 * pitch_w),
                (ic1, pitch_w),
                (oc1 // weight_chunk, width),
            )
            return strided_bank_walk(
                loops, width, self.bank_size, c0 * pitch_w + k_off
            )

        held = ()
        if b[le.IC][2] == 1:
            held = tuple(
                loop
                for loop in (le.OX, le.OY)
                if orders[loop][2] < orders[le.OC][2]
            )
        return {
            "input": BANK_SWITCH_CYCLES
            * self._stream_switches(mapping, (le.OX, le.OY, le.IC), input_walk),
            "weight": BANK_SWITCH_CYCLES
            * self._stream_switches(mapping, (le.OC, le.IC), weight_walk, held),
        }

    def _tail_bank_switch_cycles(self, mapping, tiled):
        """Fused-operand read latency, charged only on a finishing pass.

        A riding tail uses MatrixOps' filtered L2/L1 output loops. A
        separate vector pass scans the finished output tile in storage
        order. Broadcast dimensions have zero address stride. Multi-beat
        requests overlap part of the bank-switch drain (two beats lose
        seven cycles, versus eight for one), as in pool_bank_switch_cycles.
        """
        if not self.bank_size:
            return {}
        b, orders = mapping.loop_blockings, mapping.loop_orders
        dims_out = (le.ON, le.OY, le.OX, le.OC)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        l1_order = sorted(dims_out, key=lambda i: -orders[i][1])
        result = {}
        for i, (dims, bits) in enumerate(self.tail_specs):
            strides = {}
            pitch = bits / 8
            for dim in reversed(dims_out):
                strides[dim] = pitch if dim in dims else 0
                if dim in dims:
                    pitch *= self._extent(mapping, dim, 2)
            lanes = oc_dim if le.OC in dims else 1
            width = math.ceil(lanes * bits / 8)

            def tile_walk(idx):
                offset = sum(
                    idx.get(d, 0) * self._extent(mapping, d, 1) * strides[d]
                    for d in dims_out
                )
                loops = tuple(
                    (b[d][1], strides[d] * mapping.loop_partitionings[d][0])
                    for d in l1_order
                )
                return strided_bank_walk(loops, width, self.bank_size, offset)

            if tiled:
                switches = self._stream_switches(
                    mapping, dims, tile_walk, (le.IC, le.FX, le.FY)
                )
            else:
                loops = tuple(
                    (
                        b[d][1] * b[d][2],
                        strides[d] * mapping.loop_partitionings[d][0],
                    )
                    for d in dims_out
                )
                switches = strided_bank_walk(loops, width, self.bank_size)[0]
            beats = math.ceil(width * 8 / self.sram_bandwidth)
            result[("fused", i)] = switches * max(
                1, BANK_SWITCH_CYCLES + 1 - beats
            )
        return result

    def _tail_stall(self, mapping, bank_groups, words, compute, bank):
        """Cycles the array loses to the tail's burst on a bank that feeds
        one of its every-round buffers.

        Those buffers ping-pong per L1 sweep (``blockings[IC][2]`` sweeps
        to a block), so each sweep's fetch must land inside the sweep before
        it.  The tail's words on such a bank do not spread over the block:
        they arrive together, and the bank's round-robin port lets them
        through at the tail's beats per request round for every grant of a
        stream that is always pending, or in the sweep's free cycles when
        the stream idles between its requests.  The sweeps the burst lands
        on are priced one by one, and what they cost beyond the block's
        price is the stall.  A tail buffer meets the stream's bank only on
        the steps whose ping-pong slots agree: every step when the stream
        is refetched with it, else every other.
        """
        if not bank_groups:
            return 0
        sweeps = mapping.loop_blockings[le.IC][2]
        steps = self._l3_blocks(mapping)
        if sweeps == 1 or steps == 1:
            return 0
        sweep_compute = compute / sweeps
        vectors = 1
        for loop in [le.OC, le.OY, le.OX]:
            vectors *= mapping.loop_blockings[loop][1]
        stream_dims = {
            "input": _IF_DIMS,
            "input_scale": _IF_DIMS,
            "weight": _FL_DIMS,
            "weight_scale": _FL_DIMS,
        }
        stall = 0.0
        for roles in bank_groups:
            streams = [role for role in roles if role in stream_dims]
            tails = []
            for role in roles:
                is_fused = isinstance(role, tuple) and role[0] == "fused"
                if (role == "output" or is_fused) and words.get(role, 0):
                    tails.append(role)
            if not streams or not tails:
                continue
            others = [
                role
                for role in roles
                if role not in streams and role not in tails
            ]
            stream_words = [words.get(role, 0) / sweeps for role in streams]
            busiest = max(stream_words)
            pending = sum(stream_words)
            spread = sum(words.get(role, 0) for role in others) / sweeps
            tail_words = sum(words[role] for role in tails)
            # The output's beats and its scales leave on two requesters,
            # every beat and every scale a request of its own; a fused
            # operand is fetched once per output vector.
            oc_dim = mapping.loop_partitionings[le.OC][0]
            requests = []
            for role in tails:
                if role == "output":
                    out = vectors * oc_dim
                    stores = self._bus_words(out, self.output_dtype_width)
                    requests.append(stores)
                    if self.output_scale_width:
                        requests.append(math.ceil(out / self.scale_block_size))
                elif le.OC in self.tail_specs[role[1]][0]:
                    requests.append(vectors)
                else:
                    requests.append(words[role])
            beats_per_grant = tail_words / max(requests)
            remaining = tail_words
            priced = 0.0
            for sweep in range(sweeps):
                free = max(0.0, sweep_compute - pending - spread)
                landed = min(remaining, max(beats_per_grant * busiest, free))
                if sweep == sweeps - 1:
                    landed = remaining
                remaining -= landed
                priced += max(sweep_compute, pending + spread + landed)
            refetched = [
                self._l3_loads(mapping, stream_dims[role]) == steps
                for role in streams
            ]
            fraction = 1.0 if any(refetched) else 0.5
            excess = max(0.0, priced - max(compute, bank))
            stall = max(stall, fraction * excess)
        return stall

    def matrix_cycles(self, mapping, bank_groups):
        """Cycles of one L3 grid step: the L2 sweep of weight-reuse tiles,
        each costing the busier of the matrix unit -- its systolic passes,
        plus the back-pressure a single-buffered accumulator takes while the
        tail drains each finished tile -- and the scratchpad bank with the
        most to move for it -- the every-round operands of its L1 sub-tiles,
        the accumulator read back and rewritten while the reduction is
        split, the finished tile and the tail's operands when it is not,
        the round trips a stream idles for when it changes bank, each
        summed with whatever shares its bank -- plus the stall the tail's
        burst inflicts where it shares a bank with one of the array's
        every-round buffers (``_tail_stall``) -- and, for an outlier GEMM,
        the SpMM unit, which must deliver the block's sparse correction
        before the vector pipeline releases any of its rows: per 64-column
        pass it walks every row of the block, paying ``SPMM_ROW_CYCLES`` of
        turnaround plus that row's outliers, each gather a bank switch when
        the weight tile spans several banks -- plus the once-per-sweep
        overhead (buffer fill, systolic skew, the last parked tile's drain)
        spread over the ops a double-buffered L2 overlaps it with.  Also
        the reporting model's per-tile utilization denominator.

        Args:
            mapping: The interstellar mapping to price.
            bank_groups: Its bank partition as role sets (``bank_partition``),
                or ``None`` when nothing shares a bank.
        """
        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing ---
        sa_weight_loading_time = partitionings[le.IC][0]

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

        # --- the finished L1 output tile: its vectors, and the bus beats the
        # tail spends on each -- storing it, and reading a fused operand
        # alongside when one is wider than the port ---
        num_k = blockings[le.IC][3]
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        oc_dim = partitionings[le.OC][0]
        output_width = (
            self.accum_dtype_width if num_k > 1 else self.output_dtype_width
        )
        store_cycles = math.ceil(output_width * oc_dim / self.sram_bandwidth)
        if num_k == 1 and self.output_scale_width:
            # The block scales leave on a requester of their own, one beat
            # per vector, on the output's bank.
            store_cycles += 1
        vector_beats = store_cycles
        if num_k == 1 and not self.single_k_tail_extra_pass:
            for dims, bits in self.tail_specs:
                vector_beats = max(
                    vector_beats,
                    math.ceil(
                        bits
                        * (oc_dim if le.OC in dims else 1)
                        / self.sram_bandwidth
                    ),
                )

        # Without a bank to park the finished tile in, its vectors leave the
        # array one per step during the last reduction pass of each weight
        # tile -- as one burst of the whole tile when the OC passes are
        # adjacent (no filter loops at L1), else as OC1 bursts of one spatial
        # tile -- and the tail drains them at ``vector_beats`` apiece.  The
        # path absorbs ``OUTPUT_SLACK`` of them; past that the array runs at
        # the tail's pace for the rest of the burst.  A double-buffered
        # accumulator parks a tile whose tail moves more than a bus word per
        # vector on some port (the toolchain's ``should_use_direct_path``);
        # a narrower tail rides the pass and pays like a single-buffered one.
        parked = self.double_buffered_accum_buffer
        if parked:
            widths = [output_width * oc_dim]
            for dims, bits in self.tail_specs:
                widths.append(bits * (oc_dim if le.OC in dims else 1))
            parked = max(widths) > self.sram_bandwidth
        if not parked:
            burst_vectors = weight_reuse_tile_size
            burst_cycles = weight_reuse_tile_time
            bursts = blockings[le.OC][1]
            if blockings[le.FX][1] * blockings[le.FY][1] == 1:
                burst_vectors *= bursts
                burst_cycles *= bursts
                bursts = 1
            computation_l1_time += bursts * max(
                0, burst_vectors * vector_beats - burst_cycles - OUTPUT_SLACK
            )

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        # --- bus traffic of one L2 output block: the loads of its L1
        # sub-tiles, then what the vector unit moves for the block itself ---
        loads = self._load_words(mapping)
        words = {
            role: count * blockings[le.IC][2] for role, count in loads.items()
        }
        # With the whole reduction inside the block, a weight tile whose L2
        # loop is outside the spatial ones is fetched once and held across
        # them (the input is refetched every block), so its words are spread
        # over the blocks that reuse it.
        if blockings[le.IC][2] == 1:
            held = 1
            for loop in [le.OX, le.OY]:
                if orders[loop][2] < orders[le.OC][2]:
                    held *= blockings[loop][2]
            words["weight"] /= held
            words["weight_scale"] /= held
        # The bias is read once per output tile: spread over its rounds.
        words["bias"] = (
            self._bus_words(self._extent(mapping, le.OC, 1), self.bias_width)
            / num_k
        )
        output_elems = 1
        for loop in [le.OC, le.OY, le.OX, le.ON]:
            output_elems *= self._extent(mapping, loop, 1)
        if num_k > 1:
            # A split reduction reads the running partial back through the
            # same single-ported bank it writes the new one to.
            words["scratch"] = 2 * self._bus_words(
                output_elems, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass and not self.tail_keeps_shape:
            # A staged single round parks the finished tile in scratch for
            # the tail's own pass (``vector_cycles``) to read back.
            words["scratch"] = self._bus_words(
                output_elems, self.output_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # An in-place pass reads the tile back from its output slot and
            # rewrites it: two bank visits beyond a riding tail's.
            words.update(self._tail_words(mapping, 1))
            words["output"] += 2 * self._bus_words(
                output_elems, self.output_dtype_width
            )
        else:
            words.update(self._tail_words(mapping, 1))
        # A stream that changes bank idles its port for a round trip each
        # time; spread the step's switches over its output blocks.
        switches = self._bank_switch_cycles(mapping)
        if num_k == 1 and (
            not self.single_k_tail_extra_pass or self.tail_keeps_shape
        ):
            switches.update(
                self._tail_bank_switch_cycles(
                    mapping, tiled=not self.single_k_tail_extra_pass
                )
            )
        for role, cycles in switches.items():
            words[role] += cycles / l2_blocks
        # The matrix unit and the vector unit are pipelined -- one drains a
        # tile while the other computes the next -- so a block costs the
        # busier of the two.
        bank = self._bank_cycles(words, bank_groups)
        block_time = max(computation_l1_time, bank)
        # The tail's words do not spread over the block: they land as a
        # burst on a few of its sweeps, and on a bank that feeds one of the
        # array's ping-pong buffers they can outrun the sweeps they land on.
        block_time += self._tail_stall(
            mapping, bank_groups, words, computation_l1_time, bank
        )

        # The SpMM unit runs the block alongside and the vector pipeline
        # waits for its correction on every output vector, so a block also
        # costs its pace: per PE-array-wide pass, every row's turnaround
        # plus its gathered weight rows.
        if self.outlier_rate:
            rows = 1
            for loop in [le.OX, le.OY, le.ON]:
                rows *= self._extent(mapping, loop, 1)
            k_block = self._extent(mapping, le.IC, 2)
            passes = blockings[le.OC][1]
            visits = passes * rows
            gathers = visits * k_block * self.outlier_rate
            # Gathers hit random K rows of the weight tile; when it spans
            # several banks most consecutive gathers change bank and pay the
            # read path's round trip, as the streams above do.
            switch = 0.0
            if self.bank_size:
                weight_tile_bytes = (
                    self._extent(mapping, le.OC, 2)
                    * k_block
                    * blockings[le.FX][1]
                    * blockings[le.FY][1]
                    * self.weight_dtype_width
                    / 8
                )
                banks = max(1, math.ceil(weight_tile_bytes / self.bank_size))
                switch = BANK_SWITCH_CYCLES * (1 - 1 / banks)
            spmm_block_time = visits * SPMM_ROW_CYCLES + gathers * (1 + switch)
            block_time = max(block_time, spmm_block_time)

        # The first tile's loads overlap nothing; the last parked tile's drain
        # is a whole vector pass, while a single-buffered accumulator's drain
        # is already in its blocks' own time.
        buffer_fill = self._bank_cycles(loads, bank_groups)
        skew = partitionings[le.IC][0] + partitionings[le.OC][0] - 2
        drain = output_size * vector_beats if parked else 0
        overhead = buffer_fill + skew + drain
        steady = l2_blocks * block_time

        if not self.double_buffered_l2:
            return steady + overhead

        # Every op the sweep dispatches -- the L3 steps and the batch elements
        # looped outside the mapping -- overlaps the ramps of its neighbours.
        steps = self._l3_blocks(mapping) if num_k == 1 else num_k
        return steady + overhead / (steps * self.batch)

    def vector_cycles(self, mapping, bank_groups):
        """Vector-unit cycles to finish one L3 output tile.  Charged on the grid
        step that ends a K sweep, after that step's accumulation.

        The busier of the unit's own rate -- one lane group per cycle, sized
        by the widest element the tail touches (the partial sum it reads, not
        the narrower value a ``quantize_mx`` writes) at ``dram_bandwidth``,
        which is what ``vector_op_utilization`` charges for the same tail in
        the reporting model -- and the busiest bank: the accumulator read
        back, the finished tile and its scales written, each tail operand
        read, summed wherever the partition puts them together.
        """
        blockings = mapping.loop_blockings
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1] * blockings[loop][2]
        oc_dim = mapping.loop_partitionings[le.OC][0]
        widths = [self.output_dtype_width, self.accum_dtype_width]
        widths += [bits for _, bits in self.tail_specs]
        lane_bytes = max(widths) / 8 * oc_dim
        lanes = output_size * math.ceil(lane_bytes / self.dram_bandwidth)
        words = self._tail_words(mapping, 2)
        for role, cycles in self._tail_bank_switch_cycles(
            mapping, tiled=False
        ).items():
            words[role] += cycles
        if blockings[le.IC][3] > 1 or (
            self.single_k_tail_extra_pass and not self.tail_keeps_shape
        ):
            words["scratch"] = self._bus_words(
                output_size * oc_dim, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # The in-place pass reads the finished tile from its output slot.
            words["output"] += self._bus_words(
                output_size * oc_dim, self.output_dtype_width
            )
        return max(lanes, self._bank_cycles(words, bank_groups))

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

    def _batch_loads(self, mapping, dims, distinct):
        """Fetches of an operand spanning ``dims`` over the whole sweep, the
        batch loop the builder wraps the mapping in included.

        Diced inside a batch step, the operand re-reads in full on every one:
        the next step restarts the sequence, so its first block differs from
        the last one loaded and the guard never fires.  Held whole, the tile
        survives into the next step and only a change of block costs --
        ``distinct`` of them over the batch, which is fewer than ``batch``
        for an operand a group shares and 1 for one they all share.
        """
        per_step = self._l3_loads(mapping, dims) if dims else 1
        return per_step * (self.batch if per_step > 1 else distinct)

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

        input_sizes = (
            input_elems * self.input_dtype_width / 8,
            input_elems / self.scale_block_size * self.input_scale_width / 8,
        )
        weight_sizes = (
            weight_elems * self.weight_dtype_width / 8,
            weight_elems / self.scale_block_size * self.weight_scale_width / 8,
        )
        output_sizes = (
            output_elems * self.output_dtype_width / 8,
            output_elems / self.scale_block_size * self.output_scale_width / 8,
        )
        input_load = transfer(*input_sizes)
        weight_load = transfer(*weight_sizes)
        store = transfer(*output_sizes)

        # A tail operand spans output dims alone, so its count already runs
        # over the output steps -- the only ones that read it.
        tail_sizes = self.tail_tile_sizes(mapping)
        tail_dmas = [
            (
                transfer(size),
                self._batch_loads(
                    mapping, dims, self.batch if le.ON in dims else 1
                ),
            )
            for (dims, _), size in zip(self.tail_specs, tail_sizes)
        ]

        bank_groups, scratch_slots = bank_partition(
            architecture, layer.size_fn, layer, mapping
        )
        matrix_cycles = self.matrix_cycles(mapping, bank_groups)
        vector_cycles = (
            self.vector_cycles(mapping, bank_groups) if self.has_tail else 0
        )

        input_steps = self._batch_loads(mapping, _IF_DIMS, self.batch)
        weight_steps = self._batch_loads(mapping, _FL_DIMS, self.weight_batch)

        # The mapping covers one batch element; the builder loops the rest.
        l3_blocks = self._l3_blocks(mapping) * self.batch
        num_k = blockings[le.IC][3]
        output_tiles = l3_blocks // num_k

        # Traffic the sweep moves, for a caller ranking by DRAM rather than by
        # time, and to check the reuse counts against a profile.  Every
        # candidate mapping is priced through here, so it describes the last
        # one scored -- read it straight after the call that priced the mapping
        # in question.
        self.dram_bytes = {
            "input": input_steps * sum(input_sizes),
            "weight": weight_steps * sum(weight_sizes),
            "output": output_tiles * sum(output_sizes),
            "tail": sum(t * s for (_, t), s in zip(tail_dmas, tail_sizes)),
        }

        dmas = [
            (store, output_tiles),
            (input_load, input_steps),
            (weight_load, weight_steps),
            *tail_dmas,
        ]

        if not self.double_buffered_l2:
            total_time = l3_blocks * matrix_cycles + sum(t * c for c, t in dmas)
            if num_k > 1 or self.single_k_tail_extra_pass:
                total_time += output_tiles * vector_cycles
            return total_time

        if num_k == 1:
            # Every step finishes a tile: one schedule covers the sweep.  A
            # riding tail drains inside the matrix pass, and an in-place one
            # overlaps the next tile's (its bank words are in the block);
            # a staged one is a pass of its own, serial on the single
            # scratch region.
            step = matrix_cycles
            if self.single_k_tail_extra_pass and not self.tail_keeps_shape:
                step += vector_cycles
            return _sweep_cycles(dmas, l3_blocks, step)

        load = input_load + weight_load
        # The sweep's last step has no tile after it to prefetch, so it costs
        # compute alone: hold it out of the count and let the epilogue charge
        # it, with the tail and store that drain behind it.
        accum_steps = l3_blocks - 2 * (output_tiles - 1) - 1
        classes = _step_classes(tail_dmas, output_tiles)
        # Hold out the first output step in the same way -- the prologue fetches
        # its tail, with nothing running yet to hide it behind.  Taking what
        # that step owed leaves every tail fetch counted exactly once.
        first_tail = classes[-1][0]
        classes[-1] = (classes[-1][0], classes[-1][1] - 1)

        total_time = load + first_tail + accum_steps * max(load, matrix_cycles)
        # One window per remaining tile, spanning two grid steps: the matrix
        # unit finishes this tile and starts the next, while DRAM fits that
        # tile's tail read, its store and the next prefetch into the same span.
        # The busier side sets the price, and only the tail differs from one
        # window to the next -- hence one price per class.
        for tail, count in classes:
            prefetch = load + tail
            if self.split_k_tail_extra_pass and scratch_slots == 1:
                # The bare pass holds the control stream through both the
                # matrix and the vector pass, so the window's loads are
                # issued only then and nothing hides them.
                total_time += count * (
                    matrix_cycles + vector_cycles + prefetch + matrix_cycles
                )
                continue
            compute = max(matrix_cycles, prefetch) + matrix_cycles
            dma = max(matrix_cycles + vector_cycles, prefetch) + store + load
            total_time += count * max(compute, dma)
        total_time += matrix_cycles + vector_cycles + store
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
        w_dims = OIHW_TO_HWIO if transposed else None
        in_dims = NCHW_TO_NHWC if transposed else None
        out_channels, in_channels, kH, kW = unproject(weight_shape, w_dims)
        _, _, height, width = unproject(node.shape, in_dims)

        if in_channels == 3:
            return None

        stride_h, stride_w = _pair(get_arg_value(node, 3, "stride", 1))
    else:
        width = node.shape[-2] if is_bmm(node) else math.prod(node.shape[:-1])

        if weight_is_ck(node):
            in_channels, out_channels = weight_shape[-2:]
        else:
            out_channels, in_channels = weight_shape[-2:]

        kH = kW = height = stride_h = stride_w = 1

    return interstellar.Layer(
        nifm=in_channels,
        nofm=out_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
    )


@dataclass
class _Search:
    """One node's mapping search, prepared and ready to run.

    Every step that reads the FX node -- shapes, dtypes, the GQA weight repeat
    -- happens while this is built, so ``_run_search`` needs only
    ``interstellar``.  It must stay that way: ``prefetch_tilings`` runs it in a
    forked worker, where dispatching a torch op deadlocks.

    Attributes:
        name: The anchor node's name, for logging.
        tiler: The shared ``TilerContext``.
        layer: The interstellar ``Layer`` to map.
        rc: The ``RuntimeCalculator`` scoring each candidate mapping.
        size_fn: The ``Layer.size_fn`` the fit check runs.
    """

    name: str
    tiler: TilerContext
    layer: object
    rc: RuntimeCalculator
    size_fn: object


def _prepare_search(node, tiler, constraint=None):
    """Everything a GEMM/conv ``node`` needs before its mapping search.

    Reads the node -- shapes, dtypes, the fused tail's operands, the GQA weight
    repeat -- to build the interstellar layer, the timing model and the size
    functions, plus the cache key naming the result.  Runs in the parent, so
    what it hands back needs only ``interstellar`` (see :class:`_Search`).

    The layer dims come from the node's current (pre-tiling) shapes, and the
    timing model and the size functions are built from its own widths so the
    two size the same operands.  The L2 -> L1 bus carries ``min(unroll)`` input
    elements per cycle -- the rate the array's narrow side consumes them at --
    so its width is ``min(unroll) * if_bits / 8`` bytes per cycle.

    ``constraint`` (a :class:`TileConstraint`) pins loop extents every size
    function vetoes against -- the caller's, merged with the head alignment
    an MHA relayout demands -- and keys the result apart from the free
    search's.

    Returns:
        ``(key, search)``, or ``None`` when no interstellar run is needed: not
        a matrix op, a GEMV (``gemv_op_tiling`` searches those instead), or an
        anchor that already carries an ``l2_tiling``.  ``search`` is ``None``
        when interstellar skips the layer.
    """
    anchor = get_anchor_node(node)
    if (
        not is_gemm_op(anchor)
        or is_fully_connected(anchor)
        or anchor.meta.get("l2_tiling") is not None
    ):
        return None

    sub_gm = node.meta.get("submodule")
    if sub_gm is not None:
        ShapeProp(sub_gm).propagate(
            *(n.value.clone() for n in node.all_input_nodes)
        )
        dtypes = [n.meta.get("dtype") for n in node.all_input_nodes]
        phs = [n for n in sub_gm.graph.nodes if n.op == "placeholder"]
        for i, ph in enumerate(phs):
            ph.meta["dtype"] = dtypes[i]

    out_dtype = node.meta.get("dtype")
    fused_specs = _fused_operand_specs(node, anchor)
    has_tail = sub_gm is not None
    # The tail needs a pass of its own when its quantize breaks the stream
    # or the pipeline has no stage left for it: after the drain on a single
    # round, after the accumulate on a split one, which takes one more.
    breaks_stream = stream_breaking_quantize(sub_gm) is not None
    single_k_tail_extra_pass = breaks_stream or not node.meta.get(
        "single_k_tail_fusible", True
    )
    # The tail's own ``quantize_mx_outlier``, if it has one: this group is
    # then a CSR producer as well as (possibly) a consumer.
    out_quant = None
    if sub_gm is not None:
        out_quant = next(
            (n for n in sub_gm.graph.nodes if n.target is _QUANTIZE_MX_OUTLIER),
            None,
        )
    if not single_k_tail_extra_pass and out_quant is not None:
        # A CSR-producing epilogue (``_EpilogueTail``) re-dices its tile at
        # the consumers' slice width: ops between the anchor and the
        # quantize run once on the whole tile, and their result is staged in
        # the scratch whenever a slice is finer than the column tile.  The
        # slice count is a consumer property the search cannot see, so every
        # prefixed producer tail charges the staged region.
        single_k_tail_extra_pass = (
            isinstance(out_quant.args[0], torch.fx.Node)
            and out_quant.args[0].target is not anchor.target
        )
    # A single round's extra pass runs in place when the tail keeps the
    # tile's shape, else through scratch (``_gemm_scratch_and_kernel``).
    tail_keeps_shape = not isinstance(node.value, (tuple, list)) and tuple(
        node.value.shape
    ) == tuple(anchor.value.shape)
    # A split reduction's extra pass reads the tile back from scratch; a
    # second region lets it ride the finalize commit, except in a
    # CSR-producing nest, whose bare stores pin one (``_SparseGemm``).
    split_k_tail_extra_pass = has_tail and (
        breaks_stream or not node.meta.get("split_k_tail_fusible", False)
    )
    scratch_regions = 2 if split_k_tail_extra_pass and out_quant is None else 1

    # A projection GEMM feeding an MHA relayout must tile OC on whole heads
    # else _detect_mha_relayout can't store the tile: that joins whatever
    # constraint the caller registered.
    if sub_gm is not None and not is_conv2d(anchor):
        nodes = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        perm = trailing_mha_perm(nodes)
        if perm is not None and perm.value.ndim > anchor.value.ndim:
            head = TileConstraint(multiple=((le.OC, perm.value.shape[-1]),))
            constraint = head.merged(constraint)

    outlier_pct = 0.0
    outlier_rate = 0.0
    a_data = anchor.kwargs.get("A_data")
    if a_data is not None and getattr(a_data, "value", None) is not None:
        act = anchor.args[0].value
        elements = act.shape[-2] * act.shape[-1]
        outlier_pct = a_data.value.shape[-1] / elements
        outlier_rate = anchor.meta["outlier_rate"]

    # A CSR producer's stream is unquantized, so ``meta["dtype"]`` leaves its
    # entries empty; resolve them against the traced values (the rule
    # ``_node_dtype_bits`` documents) to price the store staging.
    out_outlier_pct = 0.0
    if out_quant is not None:
        out_outlier_pct = get_arg_value(out_quant, 9, "max_pct", 0.01)
        vals = getattr(node, "value", None)
        if isinstance(vals, (list, tuple)):
            tracked = (
                out_dtype
                if isinstance(out_dtype, (list, tuple))
                else [None] * len(vals)
            )
            out_dtype = [
                d if d is not None else v.dtype for d, v in zip(tracked, vals)
            ]

    key = _layer_cache_key(anchor) + (
        tuple(out_dtype) if isinstance(out_dtype, list) else out_dtype,
        tuple(fused_specs),
        has_tail,
        single_k_tail_extra_pass,
        split_k_tail_extra_pass,
        tail_keeps_shape,
        scratch_regions,
        constraint,
        outlier_pct,
        out_outlier_pct,
        outlier_rate,
    )

    layer = _extract_layer_from_node(anchor)
    if layer is None:
        return key, None

    mx_out = isinstance(out_dtype, (list, tuple))
    of_dtype = out_dtype[-1] if mx_out else out_dtype
    if_bits = _node_dtype_bits(anchor.args[0])
    fl_bits = _node_dtype_bits(anchor.args[1])
    of_bits = (
        get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(anchor)
    )
    if_scale_bits = _node_dtype_bits(anchor.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(anchor.kwargs.get("weight_scale"), 0)
    of_scale_bits = get_dtype_width(out_dtype[-2]) if mx_out else 0

    logger.info(
        f"[interstellar] {anchor.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil} | "
        f"if={if_bits}b fl={fl_bits}b of={of_bits}b "
        f"if_scale={if_scale_bits}b fl_scale={fl_scale_bits}b "
        f"bs={anchor.kwargs.get('block_size')}"
    )

    # A scratchpad bank moves one store word per cycle, ``bank_width`` bytes:
    # a port the hardware fixes, not one that widens with the element.  So a
    # row of elements wider than the port's lanes (int6 attention operands on
    # the 4-bit NF4 port) takes more than one beat.  Without a bank width the
    # port is taken as one input row per cycle.
    sram_bandwidth = (
        tiler.config.bank_width * 8
        if tiler.config.bank_width
        else min(tiler.config.pe_array_size) * if_bits
    )

    batch = math.prod(anchor.value.shape[:-2]) if is_bmm(anchor) else 1

    weight = anchor.args[1]
    transposed, repeat = weight_transforms(weight)[1:3]
    weight_repeat = (
        math.prod(repeat[: max(0, len(weight.shape) - 2)]) if repeat else 1
    )

    rc = RuntimeCalculator(
        if_bits,
        fl_bits,
        of_bits,
        get_dtype_width(anchor.value.dtype),
        tiler.config.double_buffered_accum_buffer,
        sram_bandwidth,
        tiler.config.bytes_per_cycle,
        tiler.config.access_latency_cycles,
        double_buffered_l2=tiler.config.double_buffered_l2,
        batch=batch,
        weight_batch=batch // weight_repeat,
        has_tail=has_tail,
        single_k_tail_extra_pass=single_k_tail_extra_pass,
        split_k_tail_extra_pass=split_k_tail_extra_pass,
        tail_keeps_shape=tail_keeps_shape,
        tail_specs=fused_specs,
        input_scale_width=if_scale_bits,
        weight_scale_width=fl_scale_bits,
        output_scale_width=of_scale_bits,
        scale_block_size=anchor.kwargs.get("block_size") or 1,
        outlier_rate=outlier_rate,
        bias_width=_node_dtype_bits(get_arg_value(anchor, 2, "bias", None), 0),
        stride=(layer.hstd, layer.wstd),
        bank_size=tiler.config.bank_size,
        weight_transposed=transposed,
    )

    # Built up front rather than per attempt: each one reads the node, which
    # only the parent may do.  They close over it, so they cannot be pickled --
    # hence a forked worker rather than a spawned one.
    size_fn = make_size_fn(
        anchor,
        out_dtype,
        fused_specs,
        constraint=constraint,
        has_tail=has_tail,
        single_k_tail_extra_pass=single_k_tail_extra_pass,
        tail_keeps_shape=tail_keeps_shape,
        scratch_regions=scratch_regions,
        num_slots=tiler.config.num_slots,
        batch=batch,
        weight_batch=batch // weight_repeat,
        outlier_pct=outlier_pct,
        out_outlier_pct=out_outlier_pct,
        bank_width=tiler.config.bank_width,
        vector_lanes=tiler.config.vector_lanes,
    )
    return key, _Search(anchor.name, tiler, layer, rc, size_fn)


def _run_search(search):
    """Map ``search``'s layer.

    Dispatches no torch op, so it is safe in a forked worker (:class:`_Search`).

    Returns:
        ``(runtime, mapping)`` -- the estimated runtime and the
        ``MappingPoint`` itself.

    Raises:
        RuntimeError: No tiling fits on chip.
    """
    search.layer.size_fn = search.size_fn
    try:
        result = interstellar.optimizer.opt_optimizer(
            search.tiler.arch,
            search.layer,
            search.tiler.schedule,
            search.rc.calculate_runtime,
            verbose=False,
            runtime_tolerance=search.tiler.runtime_tolerance,
        )
    except AssertionError as e:
        # The optimizer reports "nothing fits" with a bare assert, so match
        # it narrowly: every other AssertionError in interstellar is a real
        # invariant break.
        if "No valid mapping point found" not in str(e):
            raise
        raise RuntimeError(f"{search.name}: no tiling fits on chip") from e
    _, runtime, mapping, _ = result
    return runtime, mapping


def bank_partition(architecture, size_fn, layer, mapping):
    """The role partition ``mapping``'s L2 fit is checked with.

    Rebuilds interstellar's scratchpad-level ``size_fn`` invocation
    (``cost_model.get_block_size``): the element counts from the blocking /
    partitioning products through L2, the bank geometry from the
    architecture -- and replays ``size_fn``'s own group construction, so the
    partition is exactly the one the fit check priced.  The runtime model
    prices every candidate mapping through it, and the winner's partition is
    stamped for the memory planner and the reporting model.

    Returns:
        ``(partition, scratch_slots)``: a list of role sets, one per bank
        group -- plus a ``{"scratch"}`` entry when the search charged the
        reduction scratch its own regions -- and how many regions it
        charged, the slot count the scratch is allocated with.  The
        partition is ``None`` when the architecture has no banked level or
        there is no ``size_fn`` (nothing checked the fit, so nothing shares
        a bank).
    """
    if size_fn is None:
        return None, 1
    level = 2
    buf = architecture.buffer(level)
    bank_size = buf.bank_size
    if not bank_size:
        return None, 1
    capacity = buf.capacity
    if isinstance(capacity, list):
        capacity = capacity[0]
    num_banks = capacity // bank_size

    blocking_accum = []
    partitioning_accum = []
    for i in range(le.NUM):
        blocking_accum.append(math.prod(mapping.loop_blocking(i)[: level + 1]))
        partitioning_accum.append(
            math.prod(mapping.loop_partitioning(i)[: level + 1])
        )
    partitioning = list(zip(*mapping.loop_partitionings))[level]

    cost_model = interstellar.cost_model
    counts = (
        cost_model.get_if_size(
            blocking_accum, partitioning_accum, partitioning, layer
        ),
        cost_model.get_of_size(
            blocking_accum, partitioning_accum, partitioning
        ),
        cost_model.get_fl_size(
            blocking_accum, partitioning_accum, partitioning
        ),
    )
    groups, scratch, regions = size_fn.compute_groups(
        counts, mapping, level, partitioning_accum, bank_size, num_banks
    )
    if groups is None:
        return None, 1
    partition = [roles for _, _, _, roles in groups]
    if scratch:
        partition.append({"scratch"})
    return partition, regions


def _finish_search(search, found):
    """Log ``found``'s tile sizes and price it, in the parent.

    ``get_cost`` sizes tiles through ``layer.size_fn``, which is set on the
    layer first -- a forked search set it only on the worker's copy.

    Returns:
        ``(mapping, access_list, bank_groups, scratch_slots)`` -- the best
        MappingPoint (its ``loop_blockings`` give the per-level tile
        factors), the per-level ``(input, output, weight)`` access counts
        the ``Tiling`` proto reports, and the winning bank partition with
        its scratch slot count (``bank_partition``).
    """
    runtime, mapping = found
    search.layer.size_fn = search.size_fn

    b = mapping.loop_blockings
    logger.info(
        f"[interstellar] {search.name} L1 tiles: "
        f"IC={b[le.IC][1]} OC={b[le.OC][1]} "
        f"OX={b[le.OX][1]} OY={b[le.OY][1]} ON={b[le.ON][1]}"
    )
    logger.info(
        f"[interstellar] {search.name} L2 tiles: "
        f"IC={b[le.IC][2]} OC={b[le.OC][2]} "
        f"OX={b[le.OX][2]} OY={b[le.OY][2]} ON={b[le.ON][2]}"
    )
    logger.info(
        f"[interstellar] {search.name} L3 tiles: "
        f"IC={b[le.IC][3]} OC={b[le.OC][3]} "
        f"OX={b[le.OX][3]} OY={b[le.OY][3]} ON={b[le.ON][3]}"
    )
    logger.info(f"[interstellar] {search.name} estimated runtime: {runtime}")
    logger.info(interstellar.utils.format_tiling(mapping))

    _, _, access_list = interstellar.cost_model.get_cost(
        search.tiler.arch, mapping, search.layer
    )
    bank_groups, scratch_slots = bank_partition(
        search.tiler.arch, search.size_fn, search.layer, mapping
    )
    return mapping, access_list, bank_groups, scratch_slots


# Prepared in the parent before forking; a worker reads its job by index so
# only the index crosses the process boundary (the searches are inherited).
_PREFETCH_JOBS = []

# How long the pool may take before the serial path takes the work back; the
# searches carry no deadline of their own.
PREFETCH_TIMEOUT_S = 300.0


def _run_prefetch_job(index):
    """Run one prefetched search in a forked worker.  Returns ``(ok, found)``
    so a failure leaves the entry uncached and is re-raised by the serial path,
    where it carries its normal traceback."""
    try:
        return True, _run_search(_PREFETCH_JOBS[index])
    except Exception:
        return False, None


# ``_search_in_pool``'s result for a search still running at the deadline,
# as opposed to ``None`` for one that raised.
SEARCH_TIMED_OUT = object()


def _search_in_pool(searches):
    """Run ``searches`` concurrently in forked workers.

    Only the search is forked out; each :class:`_Search` was prepared in the
    parent.  ``fork`` lets a worker use the ``size_fn`` closures it inherited
    instead of marshalling them, which matters because a closure cannot be
    pickled.  The deadline is on the pool, not on any one search: whatever
    has finished when it expires is kept.

    Returns:
        One ``_run_search`` result per search, in order -- ``None`` for one
        that raised (nothing fits, or a worker died), ``SEARCH_TIMED_OUT``
        for one that missed the deadline.
    """
    global _PREFETCH_JOBS

    if not searches:
        return []
    _PREFETCH_JOBS = searches
    start = time.perf_counter()
    # Small cap: the runner already forks per design point, so an uncapped
    # pool multiplies to jobs x cpu_count.  VOYAGER_TILING_JOBS overrides.
    workers = min(
        len(searches),
        int(os.environ.get("VOYAGER_TILING_JOBS", "4")),
        os.cpu_count() or 1,
    )
    # Keep the parent heap out of the workers' GC so fork stays copy-on-write.
    gc.freeze()
    pool = None
    # One slot per search; a slot still empty when the deadline expires is
    # left for the serial path.
    results = [SEARCH_TIMED_OUT] * len(searches)
    try:
        context = multiprocessing.get_context("fork")
        pool = context.Pool(workers)
        pending = [
            pool.apply_async(_run_prefetch_job, (index,))
            for index in range(len(searches))
        ]
        deadline = start + PREFETCH_TIMEOUT_S
        for index, job in enumerate(pending):
            # Collected one at a time rather than as one map: past the
            # deadline a zero timeout still hands back every job already
            # finished, so a search that overruns costs only itself.
            try:
                results[index] = job.get(
                    max(0.0, deadline - time.perf_counter())
                )
            except multiprocessing.TimeoutError:
                continue
            except Exception as e:  # a worker died outright
                results[index] = None
                logger.warning("[tiling] prefetched search failed (%s)", e)
    except Exception as e:
        logger.warning(
            "[tiling] parallel prefetch failed (%s); going serial", e
        )
    finally:
        if pool is not None:
            # Terminate, not close: a worker still running on timeout has to
            # go with the pool.
            pool.terminate()
            pool.join()
        gc.unfreeze()
        _PREFETCH_JOBS = []
    return [
        (
            r
            if r is SEARCH_TIMED_OUT
            else (r[1] if r is not None and r[0] else None)
        )
        for r in results
    ]


def prefetch_tilings(nodes, tiler):
    """Map every node's layer up front, concurrently, into ``tiler.cache``.

    The nodes are read here, in the parent (see :class:`_Search`); only the
    searches go to the pool (``_search_in_pool``), each under the constraint
    ``plan_csr_slices`` may have registered for its anchor.  A key that does
    not match the one ``get_tiling`` recomputes during the build simply
    misses and is redone serially — a stale key costs time, never
    correctness, and the same holds for a search the deadline passes over.
    """
    jobs = {}
    for node in nodes:
        prepared = _prepare_search(
            node, tiler, tiler.constraints.get(get_anchor_node(node))
        )
        if prepared is None:
            continue
        key, search = prepared
        if key in tiler.cache:
            continue
        if search is None:
            tiler.cache[key] = (None, None, None, 1)  # interstellar skips it
            continue
        jobs[key] = search

    if len(jobs) < 2:
        return

    keys = list(jobs)
    searches = [jobs[k] for k in keys]
    start = time.perf_counter()
    results = _search_in_pool(searches)

    cached = 0
    for key, search, found in zip(keys, searches, results):
        if found is None or found is SEARCH_TIMED_OUT:
            continue
        try:
            tiler.cache[key] = _finish_search(search, found)
        except Exception:  # redone, and re-raised, by the serial path
            continue
        cached += 1
    logger.info(
        "[tiling] prefetched %d/%d mappings in %.2fs",
        cached,
        len(searches),
        time.perf_counter() - start,
    )
    if cached < len(searches):
        logger.warning(
            "[tiling] %d mapping(s) did not finish in %.0fs; going serial",
            len(searches) - cached,
            PREFETCH_TIMEOUT_S,
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
    if not is_gemm_op(anchor):
        return None, None
    is_conv = is_conv2d(anchor)

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
    constraint = tiler.constraints.get(anchor)
    if is_fully_connected(anchor):
        counts = gemv_op_tiling(node, tiler.config, constraint)
        return gemm_batch + counts, None

    key, search = _prepare_search(node, tiler, constraint)
    if key in tiler.cache:
        mapping, access_list, bank_groups, scratch_slots = tiler.cache[key]
        logger.debug(
            "[tiling] %s: mapping cache hit (%d entries)",
            anchor.name,
            len(tiler.cache),
        )
    else:
        logger.info("[tiling] %s: running interstellar", anchor.name)
        t0 = time.perf_counter()
        if search is None:
            mapping = access_list = bank_groups = None
            scratch_slots = 1
        else:
            mapping, access_list, bank_groups, scratch_slots = _finish_search(
                search, _run_search(search)
            )
        logger.info(
            "[tiling] %s: interstellar took %.2fs",
            anchor.name,
            time.perf_counter() - t0,
        )
        tiler.cache[key] = (mapping, access_list, bank_groups, scratch_slots)

    if mapping is None:
        return None, None

    # The builders copy these onto the nest they build (the anchor is erased
    # on splice): the mapping / architecture for the proto's ``Tiling``,
    # ``bank_groups`` for the memory planner, ``scratch_slots`` for the
    # reduction kernel, and the calculator / layer so the reporting model
    # prices the mapping actually emitted.
    anchor.meta["tiling"] = {
        "interstellar_tiling": (mapping, access_list),
        "interstellar_architecture": tiler.arch,
        "bank_groups": bank_groups,
        "scratch_slots": scratch_slots,
        "runtime_calculator": search.rc,
        "layer": search.layer,
    }

    b = mapping.loop_blockings  # b[dim][3] = number of DRAM tiles for the dim

    if is_conv:
        order = _l3_order_from_mapping(mapping, CONV_L3_ORDER, _CONV_LOOP)
        return (b[le.OY][3], b[le.OX][3], b[le.OC][3], b[le.IC][3]), order
    order = _l3_order_from_mapping(mapping, GEMM_L3_ORDER, _GEMM_LOOP)
    return gemm_batch + (b[le.OX][3], b[le.OC][3], b[le.IC][3]), order


def _product_node(name, out_dtype, left, right, block_size, scales, codes):
    """A bare 2-D product ``left @ right`` as a node of its own graph, for
    the mapping search: placeholders carrying the operands' tile shapes and
    the logical dtypes of the nodes the tiles are cut from.

    Args:
        name: The node's name, for the search's logging.
        out_dtype: The product's torch dtype.
        left: ``(shape, source)`` -- the tile shape and the FX node it is
            cut from, whose dtype it takes.  Likewise ``right``.
        block_size: The MX block, or ``None`` for an unquantized product.
        scales: Under MX, the ``(shape, source)`` of each operand's block
            scales, left then right.
        codes: Under MX, the two codebook nodes, left then right, each
            ``None`` for a format without one (``fp4``).
    """
    graph = torch.fx.Graph()

    def placeholder(label, source, shape=None):
        ph = graph.placeholder(f"{name}_{label}")
        shape = tuple(source.value.shape) if shape is None else shape
        set_node_value(ph, torch.empty(shape, dtype=source.value.dtype))
        ph.meta["dtype"] = source.meta.get("dtype")
        return ph

    a = placeholder("a", left[1], left[0])
    b = placeholder("b", right[1], right[0])
    if block_size is None:
        node = graph.call_function(
            torch.ops.aten.matmul.default, (a, b), name=name
        )
    else:
        (a_scale, b_scale), (a_code, b_code) = scales, codes
        kwargs = {
            "input_scale": placeholder("a_scale", a_scale[1], a_scale[0]),
            "weight_scale": placeholder("b_scale", b_scale[1], b_scale[0]),
            "block_size": block_size,
        }
        for label, code in (("input_code", a_code), ("weight_code", b_code)):
            if code is not None:
                kwargs[label] = placeholder(label, code)
        node = graph.call_function(
            torch.ops.quantized_ops.matmul_mx.default, (a, b), kwargs, name=name
        )
    set_node_value(
        node, torch.empty((left[0][0], right[0][1]), dtype=out_dtype)
    )
    return node


def _attention_products(node, tq, tkv, head_pad):
    """The two products a ``(tq, tkv)`` attention tile runs, as bare 2-D
    GEMM nodes of the tile shapes (``_product_node``): ``"scores"``, the
    query tile times the transposed key tile, and ``"context"``, the
    probabilities times the value tile.  The probabilities take the query's
    dtype and scales, as the kernel quantizes them so.  The products hold
    the head at ``head_pad`` (``attention_head_pad``), the kernel's tile
    width."""
    query, key, value = node.args[0], node.args[1], node.args[2]
    block_size = node.kwargs.get("block_size")
    out_dtype = node.value.dtype
    kw = node.kwargs
    if block_size is None:
        codes = scores_scales = context_scales = ()
    else:
        codes = (kw.get("input_code"), kw.get("weight_code"))
        # A block wider than the head is one truncated block.
        head_blocks = math.ceil(head_pad / block_size)
        scores_scales = (
            ((tq, head_blocks), kw["query_scale"]),
            ((head_blocks, tkv), kw["key_scale"]),
        )
        context_scales = (
            ((tq, tkv // block_size), kw["query_scale"]),
            ((tkv // block_size, head_pad), kw["value_scale"]),
        )
    scores = _product_node(
        f"attention_scores_{tq}x{tkv}",
        out_dtype,
        ((tq, head_pad), query),
        ((head_pad, tkv), key),
        block_size,
        scores_scales,
        codes,
    )
    context = _product_node(
        f"attention_context_{tq}x{tkv}",
        out_dtype,
        ((tq, tkv), query),
        ((tkv, head_pad), value),
        block_size,
        context_scales,
        codes,
    )
    products = {"scores": scores, "context": context}
    residual = get_arg_value(node, 7, "key_residual", None)
    if residual is not None:
        # The split cache's residual: plain products over its R positions,
        # the probabilities at the residual query's dtype.
        length = residual.value.shape[-2]
        query_residual = get_arg_value(node, 6, "query_residual")
        products["residual_scores"] = _product_node(
            f"attention_residual_scores_{tq}x{length}",
            out_dtype,
            ((tq, head_pad), query_residual),
            ((head_pad, length), residual),
            None,
            (),
            (),
        )
        products["residual_context"] = _product_node(
            f"attention_residual_context_{tq}x{length}",
            out_dtype,
            ((tq, length), query_residual),
            ((length, head_pad), get_arg_value(node, 8, "value_residual")),
            None,
            (),
            (),
        )
    return products


def _product_cycles(node, tiler):
    """The cycles for the whole product ``node`` -- mapped as one on-chip
    tile (``attention_op_tiling`` pins it so) -- and the mapping metadata
    the kernel running it is stamped with (what ``get_tiling`` leaves on a
    GEMM), or ``None`` when interstellar maps nothing.  A one-row product
    is matrix-vector: ``get_tiling`` sizes it with the GEMV search, which
    leaves no mapping, and its compute is priced the way that search
    prices it."""
    counts, _ = get_tiling(node, tiler)
    if counts is None or math.prod(counts) != 1:
        return None
    if is_fully_connected(node):
        m, k = node.args[0].value.shape
        n = node.args[1].value.shape[-1]
        return gemv_compute_cycles(node, (m, k, n), tiler.config), {}
    tiling = node.meta["tiling"]
    mapping, _ = tiling["interstellar_tiling"]
    cycles = tiling["runtime_calculator"].matrix_cycles(
        mapping, tiling["bank_groups"]
    )
    return cycles, tiling


def attention_op_tiling(
    node, tiler, *, sq_eff, kv_batch, acc_dtype, bool_mask=True
):
    """Block counts for a flash-attention node, ``(num_q_blocks,
    num_kv_blocks)``.

    Enumerates the query / key tile lengths that divide the (GQA-folded)
    query rows and the key rows in whole PE-array widths, keeps those whose
    FA3 SRAM footprint fits the scratchpad, maps each one's two products
    (``_attention_products``, the head padded to the array by
    ``attention_head_pad``) through interstellar -- ``get_tiling``, so
    identical shapes share one search, run concurrently by
    ``prefetch_tilings`` -- and ranks the candidates by
    ``attention_tile_latency`` the way ``_search_tiling`` ranks a vector
    op: the least DRAM traffic among the tilings within the tiler's
    ``runtime_tolerance`` of the fastest.  The winner's product
    mappings are left as ``node.meta["product_tilings"]`` -- ``"scores"``
    and ``"context"``, each what ``get_tiling`` stamps on a GEMM -- for the
    builder to copy onto the kernels that run them.

    Args:
        node: The attention node to tile.
        tiler: The ``TilerContext``.
        sq_eff: Query rows after the GQA fold (``plan_gqa_fold``).
        kv_batch: The key / value batch dims, which lead the loop grid.
        acc_dtype: The accumulation dtype of the softmax state.

    Returns:
        ``(num_q_blocks, num_kv_blocks)``.

    Raises:
        RuntimeError: when no tiling of the operands fits the scratchpad.
    """
    config = tiler.config
    query, key = node.args[0], node.args[1]
    head_dim = query.value.shape[-1]
    skv = key.value.shape[-2]
    # Under MX the query and key block along head_dim, the value and the
    # probabilities along the keys, so a key tile holds whole blocks.
    block_size = node.kwargs.get("block_size")
    logger.info(f"Running L2 tiling for attention: {node}")

    unit = max(config.pe_array_size)
    budget = config.usable_scratchpad_size
    # A causal query tile must not straddle a head under the GQA fold
    # (``attention_kv_last``).
    causal = get_arg_value(node, 5, "is_causal", False)
    sq = query.value.shape[-2]
    candidates = []  # (tq, tkv, tiles)
    for tq in _divisors_descending(sq_eff):
        if tq % unit and tq != sq_eff:
            continue
        if causal and sq % tq:
            continue
        for tkv in _divisors_descending(skv):
            if tkv % unit and tkv != skv:
                continue
            if block_size is not None and tkv % block_size:
                continue
            tiles = _attention_tiles(node, tq, tkv)
            if (
                _attention_sram_bytes(node, tiles, acc_dtype, config, bool_mask)
                > budget
            ):
                continue
            candidates.append((tq, tkv, tiles))
    if not candidates:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad"
        )

    head_pad = attention_head_pad(query.value.shape[-1], config)
    products = {
        (tq, tkv): _attention_products(node, tq, tkv, head_pad)
        for tq, tkv, _ in candidates
    }
    # The kernel holds each product's operands whole on chip, so its L2 tile
    # is pinned to the product and the search maps only the L1 / PE levels.
    for pair in products.values():
        for product in pair.values():
            m, k = product.args[0].value.shape
            n = product.args[1].value.shape[-1]
            tiler.constraints[product] = TileConstraint(
                exact=((le.IC, k), (le.OC, n), (le.OX, m))
            )
    prefetch_tilings(
        [p for pair in products.values() for p in pair.values()], tiler
    )

    scored = []  # (latency, dram_bytes, blocks, product tilings)
    for tq, tkv, tiles in candidates:
        try:
            priced = {
                name: _product_cycles(p, tiler)
                for name, p in products[(tq, tkv)].items()
            }
        except RuntimeError:  # a product no mapping fits on chip
            continue
        if any(v is None for v in priced.values()):
            continue
        blocks = (sq_eff // tq, skv // tkv)
        latency, traffic = attention_tile_latency(
            node,
            tiles,
            tuple(kv_batch) + blocks,
            config,
            {name: cycles for name, (cycles, _) in priced.items()},
            bool_mask,
        )
        scored.append(
            (latency, traffic, blocks, {k: v[1] for k, v in priced.items()})
        )
        logger.info(
            "[tiling] %s: %d x %d -> %.0f cycles, %.1f MB",
            node.name,
            tq,
            tkv,
            latency,
            traffic / 1e6,
        )
    if not scored:
        raise RuntimeError(f"{node}: no tiling of its products maps on chip")

    fastest = min(s[0] for s in scored) * (1.0 + tiler.runtime_tolerance)
    best = min(
        (s for s in scored if s[0] <= fastest), key=lambda s: (s[1], s[0])
    )
    node.meta["product_tilings"] = best[3]
    return best[2]
