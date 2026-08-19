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
    _node_dtype_bits,
    _step_classes,
    _sweep_cycles,
    get_dtype_width,
)
from voyager_compiler.codegen.transform.tiling.search import (
    DEFAULT_RUNTIME_TOLERANCE,
    _operand_placeholders,
    gemv_op_tiling,
)
from voyager_compiler.ops.layout import NCHW_TO_NHWC, OIHW_TO_HWIO, unproject
from voyager_compiler.shape_prop import ShapeProp

logger = logging.getLogger(__name__)
le = interstellar.le

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
    ``scratchpad_size``, the whole physical scratchpad, and a ping-ponged source
    is charged one bank per slot (``size_fn``) rather than the capacity being
    halved.  Halving cannot express a buffer that is allocated only once -- the
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
            [config.scratchpad_size],
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

    # The L1 order is pinned to ... > FY > FX > OY > OX (IC/OC free above).
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
    staged_tail=False,
    num_slots=1,
    batch=1,
    weight_batch=1,
    outlier_pct=0.0,
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

    Each such source is ping-ponged, so it costs ``num_slots`` whole banks --
    the two halves live in *separate* banks, which is what the planner does and
    what lets a load overlap the compute reading the other half.  A split
    reduction with a fused tail also accumulates into a scratch buffer the
    builders allocate exactly once (``_ScratchSpec``); it is charged a single
    bank-aligned region on top.  So is the finished tile a stream-breaking
    tail (``staged_tail``) stages even for a single round.  Leaving either
    out is what let the tiler return tilings ``plan_memory`` could not
    place.

    A bank cannot be split between groups, so each group rounds up to a whole
    bank -- which puts a floor of ``num_slots * len(groups) * bank_size`` on the
    tile, however small it is.  With more groups than banks the two *smallest*
    are merged until they fit (a tiny scale tensor would otherwise waste a
    whole bank).  Only whole sources merge, never a source's own two slots --
    each slot must keep its own bank.  ``extra_sharing`` forces further merges;
    ``_run_search`` raises it when nothing maps even at the minimum.

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
    if_bits = _node_dtype_bits(node.args[0])
    fl_bits = _node_dtype_bits(node.args[1])
    of_bits = get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
    bias_bits = _node_dtype_bits(get_arg_value(node, 2, "bias", None), 0)
    if_scale_bits = _node_dtype_bits(node.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(node.kwargs.get("weight_scale"), 0)
    of_scale_bits = get_dtype_width(of_scale_dtype) if of_scale_dtype else 0
    # A staged single round holds the anchor's finished tile in its own
    # physical dtype (``_gemm_scratch_and_kernel``'s num_k == 1 staging).
    stage_bits = get_dtype_width(node.value.dtype)
    block_size = node.kwargs.get("block_size") or 1
    # One gathered-CSR entry: the outlier value plus its column index.
    csr_data_bits = _node_dtype_bits(node.kwargs.get("A_data"), 0)
    csr_index_bits = _node_dtype_bits(node.kwargs.get("A_indices"), 0)
    csr_bits = csr_data_bits + csr_index_bits

    def _scale_bytes(count, bits):
        return count / block_size * bits / 8.0 if bits else 0.0

    def compute_groups(
        counts, point, level, partitioning_accum, bank_size, num_banks
    ):
        """The bank partition of one candidate tile: the merged ``(bytes,
        kind, slots, roles)`` groups plus the scratch region's bytes, or
        ``(None, 0.0)`` for a vetoed tile.  ``roles`` is the set of operand
        roles sharing the bank (``"input"``, ``"input_scale"``, ``"csr"``,
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

        # Veto an OC tile that splits an attention head — the MHA relayout must
        # store whole heads.
        if oc_align and extent(le.OC) % oc_align != 0:
            return None, 0.0

        # The output is a wide partial sum until IC is fully reduced; only the
        # final value carries an output scale.
        out_bits = PSUM_BITS if is_psum else of_bits
        of_scale = 0.0 if is_psum else _scale_bytes(of_count, of_scale_bits)
        bias = extent(le.OC) * bias_bits / 8.0 if bias_bits else 0.0

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

        groups = [
            (
                if_count * if_bits / 8.0,
                _IF,
                slots(_IF_DIMS, batch),
                {"input"},
            ),
            (
                _scale_bytes(if_count, if_scale_bits),
                _IF,
                slots(_IF_DIMS, batch),
                {"input_scale"},
            ),
            (
                if_count * outlier_pct * csr_bits / 8.0,
                _IF,
                slots(_IF_DIMS, batch),
                {"csr"},
            ),
            (
                fl_count * fl_bits / 8.0
                + _scale_bytes(fl_count, fl_scale_bits)
                + bias,
                _FL,
                slots(_FL_DIMS, weight_batch),
                {"weight", "weight_scale", "bias"},
            ),
            (
                of_count * out_bits / 8.0 + of_scale,
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
                    count * bits / 8.0,
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
        # stream-breaking tail (``staged_tail``) stages even a single
        # round's finished tile, in the anchor's own dtype.
        if has_tail and point.loop_blocking(le.IC)[3] > 1:
            scratch = of_count * PSUM_BITS / 8.0
        elif has_tail and staged_tail:
            scratch = of_count * stage_bits / 8.0
        else:
            scratch = 0.0

        if not bank_size:
            return groups, scratch

        scratch_banks = math.ceil(scratch / bank_size) if scratch else 0
        # Banks left for the ping-ponged sources, in whole sources.
        budget = (num_banks or num_slots * len(groups)) - scratch_banks
        target = max(1, budget // num_slots - extra_sharing)
        for _ in range(max(0, len(groups) - target)):
            groups.sort(key=lambda g: g[0])
            (s0, k0, n0, r0), (s1, k1, n1, r1) = groups[0], groups[1]
            # Charge the shared bank to the larger member's operand, and to the
            # deeper pipeline: one buffer holding both has to ping-pong if
            # either of them does.
            merged = (s0 + s1, k0 if s0 >= s1 else k1, max(n0, n1), r0 | r1)
            groups = [merged] + groups[2:]
        return groups, scratch

    def size_fn(counts, point, level, partitioning_accum, bank_size, num_banks):
        groups, scratch = compute_groups(
            counts, point, level, partitioning_accum, bank_size, num_banks
        )
        if groups is None:
            return (float("inf"),) * 3

        out = [0.0, 0.0, 0.0]
        if not bank_size:
            for size, kind, n, _ in groups:
                out[kind] += n * size
            out[_OF] += scratch
            return tuple(out)

        for size, kind, n, _ in groups:
            out[kind] += n * math.ceil(size / bank_size) * bank_size
        if scratch:
            out[_OF] += math.ceil(scratch / bank_size) * bank_size
        return tuple(out)

    size_fn.compute_groups = compute_groups
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
        batch: Grid steps the builder loops *outside* the mapping -- a bmm's
            leading dims, which share no operand.
        weight_batch: Distinct weight tiles over those ``batch`` steps, fewer
            when a group shares one -- eight KV heads reaching an attention
            matmul as thirty-two.  ``None`` = one apiece.
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
        batch: int = 1,
        weight_batch: Optional[int] = None,
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
        self.batch = batch
        self.weight_batch = batch if weight_batch is None else weight_batch
        self.has_tail = has_tail
        self.tail_specs = tuple(tail_specs)
        self.input_scale_width = input_scale_width
        self.weight_scale_width = weight_scale_width
        self.output_scale_width = output_scale_width
        self.scale_block_size = scale_block_size
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

    def matrix_cycles(self, mapping):
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
        # charged by ``vector_cycles``, and the rest only accumulate --
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

    def vector_cycles(self, mapping):
        """Vector-unit cycles to finish one L3 output tile.  Charged on the grid
        step that ends a K sweep, after that step's accumulation.

        One lane group per cycle, sized by the widest element the tail touches
        -- the partial sum it reads, not the narrower value a ``quantize_mx``
        writes.  Divides by ``dram_bandwidth``, not ``sram_bandwidth``: that is
        what ``vector_op_utilization`` charges, and it prices this same tail in
        the reporting model.
        """
        blockings = mapping.loop_blockings
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1] * blockings[loop][2]
        oc_dim = mapping.loop_partitionings[le.OC][0]
        widths = [self.output_dtype_width, self.accum_dtype_width]
        widths += [bits for _, bits in self.tail_specs]
        lane_bytes = max(widths) / 8 * oc_dim
        return output_size * math.ceil(lane_bytes / self.dram_bandwidth)

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

        matrix_cycles = self.matrix_cycles(mapping)
        vector_cycles = self.vector_cycles(mapping) if self.has_tail else 0

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
            if num_k > 1:
                total_time += output_tiles * vector_cycles
            return total_time

        if num_k == 1:
            # Every step finishes a tile: one schedule covers the sweep.
            return _sweep_cycles(dmas, l3_blocks, matrix_cycles)

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
        size_fns: Successive bank-sharing relaxations, least-shared first;
            the first that maps wins.
    """

    name: str
    tiler: TilerContext
    layer: object
    rc: RuntimeCalculator
    size_fns: list


def _prepare_search(node, tiler):
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

    Returns:
        ``(key, search)``, or ``None`` when no interstellar run is needed: not
        a matrix op, a GEMV (``gemv_op_tiling`` searches those instead), or an
        anchor that already carries an ``l2_tiling``.  ``search`` is ``None``
        when there is nothing left to run -- ``key`` is already in
        ``tiler.cache``, or interstellar skips the layer -- so a caller tells
        the two apart by looking the key up; either way the key names the
        entry.
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
    # A tail whose quantize breaks the compute stream stages the finished
    # tile even for a single-round reduction (``_gemm_scratch_and_kernel``);
    # the footprint model must charge that region for unsplit mappings too.
    staged_tail = stream_breaking_quantize(sub_gm, in_place=False) is not None
    if not staged_tail and sub_gm is not None:
        # A CSR-producing epilogue (``_EpilogueTail``) re-dices its tile at
        # the consumers' slice width: ops between the anchor and its
        # ``quantize_mx_outlier`` run once on the whole tile, and their
        # result is staged in the scratch whenever a slice is finer than
        # the column tile.  The slice count is a consumer property the
        # search cannot see, so every prefixed producer tail charges the
        # staged region.
        staged_tail = any(
            n.op == "call_function"
            and n.target is torch.ops.quantized_ops.quantize_mx_outlier.default
            and isinstance(n.args[0], torch.fx.Node)
            and n.args[0].target is not anchor.target
            for n in sub_gm.graph.nodes
        )

    # A projection GEMM feeding an MHA relayout must tile OC on whole heads
    # else _detect_mha_relayout can't store the tile.
    oc_align = None
    if sub_gm is not None and not is_conv2d(anchor):
        nodes = [n for n in sub_gm.graph.nodes if n.op == "call_function"]
        perm = trailing_mha_perm(nodes)
        if perm is not None and perm.value.ndim > anchor.value.ndim:
            oc_align = perm.value.shape[-1]

    # An outlier GEMM's CSR: what fraction of the activation the packed stream
    # is sized to hold.  Read off the operand rather than the producer's
    # geometry, so tiling stays independent of the bufferize builders.
    outlier_pct = 0.0
    a_data = anchor.kwargs.get("A_data")
    if a_data is not None and getattr(a_data, "value", None) is not None:
        rows, cols = anchor.args[0].value.shape[-2:]
        outlier_pct = a_data.value.shape[-1] / (rows * cols)

    key = _layer_cache_key(anchor) + (
        tuple(out_dtype) if isinstance(out_dtype, list) else out_dtype,
        tuple(fused_specs),
        has_tail,
        staged_tail,
        oc_align,
        outlier_pct,
    )

    if key in tiler.cache:
        return key, None

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

    sram_bandwidth = min(tiler.config.pe_array_size) * if_bits

    batch = math.prod(anchor.value.shape[:-2]) if is_bmm(anchor) else 1

    weight = anchor.args[1]
    repeat = weight_transforms(weight)[2]
    weight_repeat = (
        math.prod(repeat[: max(0, len(weight.shape) - 2)]) if repeat else 1
    )

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
        batch=batch,
        weight_batch=batch // weight_repeat,
        has_tail=has_tail,
        tail_specs=fused_specs,
        input_scale_width=if_scale_bits,
        weight_scale_width=fl_scale_bits,
        output_scale_width=of_scale_bits,
        scale_block_size=anchor.kwargs.get("block_size") or 1,
        has_sparse_op=outlier_pct > 0,
    )

    # Built up front rather than per attempt: each one reads the node, which
    # only the parent may do.  They close over it, so they cannot be pickled --
    # hence a forked worker rather than a spawned one.
    size_fns = [
        make_size_fn(
            anchor,
            out_dtype,
            fused_specs,
            extra_sharing,
            oc_align,
            has_tail=has_tail,
            staged_tail=staged_tail,
            num_slots=tiler.config.num_slots,
            batch=batch,
            weight_batch=batch // weight_repeat,
            outlier_pct=outlier_pct,
        )
        for extra_sharing in range(4 + len(fused_specs))
    ]
    return key, _Search(anchor.name, tiler, layer, rc, size_fns)


def _run_search(search):
    """Map ``search``'s layer, relaxing bank sharing until one fits.

    Each operand ideally gets a bank of its own, but a bank cannot be split, so
    a layer with more operands than banks has no tiling at any size.  Retry
    with progressively more bank sharing and keep the first (least-shared)
    mapping.

    Dispatches no torch op, so it is safe in a forked worker (:class:`_Search`).

    Returns:
        ``(index, runtime, mapping)`` -- which ``size_fns`` entry mapped, its
        estimated runtime, and the ``MappingPoint`` itself.

    Raises:
        RuntimeError: No tiling fits even at maximum sharing.
    """
    for index, size_fn in enumerate(search.size_fns):
        search.layer.size_fn = size_fn
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
            continue
        if index:
            logger.info(
                f"[interstellar] {search.name}: no tiling fits one bank "
                f"per operand; sharing {index} more"
            )
        _, runtime, mapping, _ = result
        return index, runtime, mapping
    raise RuntimeError(
        f"{search.name}: no tiling fits on chip even with every operand "
        f"sharing one bank"
    )


def _winning_bank_groups(search, index, mapping):
    """The role partition the winning mapping's L2 fit was checked with.

    Rebuilds interstellar's scratchpad-level ``size_fn`` invocation
    (``cost_model.get_block_size``): the element counts from the blocking /
    partitioning products through L2, the bank geometry from the
    architecture — and replays the winning ``size_fn``'s own group
    construction, so the partition is exactly the one the fit check priced.

    Returns:
        A list of role sets, one per bank group — plus a ``{"scratch"}``
        entry when the search charged the reduction scratch its own region —
        or ``None`` when the architecture has no banked level.
    """
    level = 2
    buf = search.tiler.arch.buffer(level)
    bank_size = buf.bank_size
    if not bank_size:
        return None
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
            blocking_accum, partitioning_accum, partitioning, search.layer
        ),
        cost_model.get_of_size(
            blocking_accum, partitioning_accum, partitioning
        ),
        cost_model.get_fl_size(
            blocking_accum, partitioning_accum, partitioning
        ),
    )
    groups, scratch = search.size_fns[index].compute_groups(
        counts, mapping, level, partitioning_accum, bank_size, num_banks
    )
    partition = [roles for _, _, _, roles in groups]
    if scratch:
        partition.append({"scratch"})
    return partition


def _finish_search(search, found):
    """Log ``found``'s tile sizes and price it, in the parent.

    ``get_cost`` sizes tiles through ``layer.size_fn``, so the winning
    relaxation is set on the layer first -- a forked search set it only on the
    worker's copy.

    Returns:
        ``(mapping, per_tile_cycles, access_list, bank_groups)`` -- the best
        MappingPoint (its ``loop_blockings`` give the per-level tile factors),
        the compute cycles of one L3 tile under it (the reporting model's
        utilization denominator), the per-level ``(input, output, weight)``
        access counts the ``Tiling`` proto reports, and the winning bank
        partition (``_winning_bank_groups``).
    """
    index, runtime, mapping = found
    search.layer.size_fn = search.size_fns[index]

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
    per_tile_cycles = search.rc.matrix_cycles(mapping)
    logger.info(f"[interstellar] {search.name} estimated runtime: {runtime}")
    logger.info(
        f"[interstellar] {search.name} per-tile compute cycles: "
        f"{per_tile_cycles}"
    )
    logger.info(interstellar.utils.format_tiling(mapping))

    _, _, access_list = interstellar.cost_model.get_cost(
        search.tiler.arch, mapping, search.layer
    )
    bank_groups = _winning_bank_groups(search, index, mapping)
    return mapping, per_tile_cycles, access_list, bank_groups


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


def prefetch_tilings(nodes, tiler):
    """Map every node's layer up front, concurrently, into ``tiler.cache``.

    Only the search is forked out; the nodes are read here, in the parent (see
    :class:`_Search`).  ``fork`` lets a worker use the ``size_fn`` closures it
    inherited instead of marshalling them, which matters because a closure
    cannot be pickled.  A key that does not match the one ``get_tiling``
    recomputes during the build simply misses and is redone serially — a stale
    key costs time, never correctness, and the same holds for a search the
    deadline passes over.  The deadline is on the pool, not on any one search:
    whatever has finished when it expires is kept, and only the rest go serial.
    """
    global _PREFETCH_JOBS

    jobs = {}
    for node in nodes:
        prepared = _prepare_search(node, tiler)
        if prepared is None:
            continue
        key, search = prepared
        if key in tiler.cache:
            continue
        if search is None:
            tiler.cache[key] = (None,) * 4  # interstellar skips it
            continue
        jobs[key] = search

    if len(jobs) < 2:
        return

    keys = list(jobs)
    searches = [jobs[k] for k in keys]
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
    results = [None] * len(searches)
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

    cached = 0
    for key, search, result in zip(keys, searches, results):
        if result is None or not result[0]:
            continue
        try:
            tiler.cache[key] = _finish_search(search, result[1])
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
    if is_fully_connected(anchor):
        return gemm_batch + gemv_op_tiling(node, tiler.config), None

    key, search = _prepare_search(node, tiler)
    if key in tiler.cache:
        mapping, per_tile_cycles, access_list, bank_groups = tiler.cache[key]
        logger.debug(
            "[tiling] %s: mapping cache hit (%d entries)",
            anchor.name,
            len(tiler.cache),
        )
    else:
        logger.info("[tiling] %s: running interstellar", anchor.name)
        t0 = time.perf_counter()
        if search is None:
            mapping = per_tile_cycles = access_list = bank_groups = None
        else:
            mapping, per_tile_cycles, access_list, bank_groups = _finish_search(
                search, _run_search(search)
            )
        logger.info(
            "[tiling] %s: interstellar took %.2fs",
            anchor.name,
            time.perf_counter() - t0,
        )
        tiler.cache[key] = (mapping, per_tile_cycles, access_list, bank_groups)

    if mapping is None:
        return None, None

    # The builders copy these onto the nest they build (the anchor is erased on
    # splice): ``per_tile_cycles`` so the reporting cost model can turn it into
    # a utilization, the mapping / architecture so the proto emitter can
    # serialize the ``Tiling`` message, and ``bank_groups`` (the winning role
    # partition) so the builders can stamp each SRAM alloc with its bank group
    # for the memory planner.
    anchor.meta["tiling"] = {
        "per_tile_cycles": per_tile_cycles,
        "interstellar_tiling": (mapping, access_list),
        "interstellar_architecture": tiler.arch,
        "bank_groups": bank_groups,
    }

    b = mapping.loop_blockings  # b[dim][3] = number of DRAM tiles for the dim

    if is_conv:
        order = _l3_order_from_mapping(mapping, CONV_L3_ORDER, _CONV_LOOP)
        return (b[le.OY][3], b[le.OX][3], b[le.OC][3], b[le.IC][3]), order
    order = _l3_order_from_mapping(mapping, GEMM_L3_ORDER, _GEMM_LOOP)
    return gemm_batch + (b[le.OX][3], b[le.OC][3], b[le.IC][3]), order
