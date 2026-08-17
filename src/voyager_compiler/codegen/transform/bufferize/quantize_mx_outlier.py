"""Bufferized ``quantize_mx_outlier``: a row sweep with a packed CSR store.

The op splits an activation into quantized inliers and a CSR of outliers, and
the SpMM engine that consumes that CSR cannot offset column indices — a block's
columns must already lie inside the weight tile that multiplies them. So the
quantize has to run per ``(row block, K slice)``, which is a tiling the op's
own anchor cannot give it: when the quantize is fused onto a ``layer_norm``,
``_vector_op_tiling_limits`` pins the tile to whole rows, and a full-K tile
emits columns spanning ``0..K-1``.

Hence two loops sharing one row sweep. The outer loop is an ordinary
``PipelinedKernel`` over row blocks, so whatever was fused ahead of the
quantize keeps its natural ``[rows, K]`` tile and its double-buffered input
fetch; its result lands in a named scratchpad intermediate. The inner loop
re-dices that intermediate along K and runs the op once per slice, so the
activation is read from DRAM exactly once.

Three DRAM outputs are ordinary tiled stores (``inliers``, ``scale``, and the
row pointers). ``data`` / ``indices`` are not: outliers concentrate in a few
channels, so blocks differ wildly in length and a fixed per-block slot would
overflow the hot ones while the cold ones sat empty. They are **packed** —
block ``b`` starts where ``b-1`` ended — so a block's store address is only
known at run time. Two pieces of loop state carry that:

``base``
    the position in the packed stream, and what the consumer needs in order to
    find a block. Written into a scratchpad table the consumer reads directly.
``run``
    the running nonzero count per K slice, handed to the op as
    ``indptr_offset`` so each block emits pointers already in slice
    coordinates. The per-slice stores then assemble one continuous CSR pointer
    array per slice in DRAM, overlapping by one entry at each block boundary —
    idempotently, since block ``m``'s last pointer is block ``m+1``'s first.

Batch dims simply prepend: ``filter_outlier`` reads ``shape[-2:]`` as the
matrix, so a 4-D attention activation is the 2-D case with leading grid dims of
tile 1. Each batch element owns its own stream, ``run`` and ``base``, because
its CSR is independent and its pointers restart at zero.
"""

import logging
import operator
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from torch.fx import Node

from voyager_compiler.codegen.node_info import (
    ancestors,
    get_anchor_node,
    get_arg_value,
    is_gemm_op,
    is_nop,
    quant_param_arg_nodes,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    MemoryLevel,
    oracle_disabled,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    PipelinedKernel,
    _DEFAULT_NUM_SLOTS,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _compute_input_spec,
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    outline_dps_ops,
    voyager,
)
from voyager_compiler.codegen.transform.tiling.search import vector_op_tiling
from voyager_compiler.codegen.transform.tiling.tiler import get_tiling
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import ShapeProp

logger = logging.getLogger(__name__)

_SRAM = int(MemoryLevel.SRAM)
_QUANTIZE_MX_OUTLIER = torch.ops.quantized_ops.quantize_mx_outlier.default

# Meta key naming the producer whose base table an alloc belongs to. The
# producer and each of its consumers allocate the table independently; a pass
# in ``bufferize_graph`` merges the group after every splice.
BASE_TABLE_TAG = "outlier_base_table"

# Meta key carrying a producer's ``_Geometry`` to its consumers: how the packed
# stream was diced, which the CSR's shapes alone do not say.
GEOMETRY_META = "outlier_geometry"

# Meta key naming the producer a CSR came from, so a consumer can tag its own
# base-table alloc into the same merge group.
PRODUCER_META = "outlier_producer"


def _scalar(t: torch.Tensor):
    """A single-element tensor as a SymInt.

    ``Tensor.item()`` decomposes through ``_refs.item``, whose
    ``number_type(...)`` cast rejects a FakeTensor once a ``while_loop`` body
    runs under fake tracing; the aten op yields the unbacked symbol directly.
    """
    if t.numel() == 1:
        # Already one element: indexing it would window a window, which a
        # TensorBoxRef cannot carry.
        return torch.ops.aten._local_scalar_dense.default(t)
    return torch.ops.aten._local_scalar_dense.default(t.reshape(-1)[0])


def _entry(t, i, nb, ones):
    """One element of a pointer tile, as a window rather than an index.

    ``t[i]`` is an ``aten.select`` — compute, which the memory model then wants
    stored somewhere. A window is addressing, which is what reading a pointer
    out of a buffer actually is.
    """
    return voyager.subview(t, [0] * nb + [i], ones + (1,), (1,) * (nb + 1), [])


@dataclass(frozen=True)
class _Geometry:
    """The producer's tile geometry.

    Attributes:
        batch: Leading batch dims of the activation, ``()`` when it is 2-D.
        rows: Rows the outer loop handles per step.
        tk: Columns one K slice spans — the consuming GEMM's own k tile, so
            each block's columns land inside the weight tile.
        block_size: Microscaling block size, which ``tk`` must be a multiple of.
        budget: Nonzeros one block may hold, the cap ``_pad_csr`` enforces.
        max_pct: Fraction of a tile the outliers may occupy, which sets both
            ``budget`` and the declared buffer length.
        dropped: Leading batch dims a reshape removed between the producer and
            this consumer; 0 on a producer's own geometry.
    """

    batch: Tuple[int, ...]
    M: int
    K: int
    rows: int
    tk: int
    block_size: int
    budget: int
    max_pct: float
    # Leading batch dims a reshape dropped between the producer and this
    # consumer. They are all extent 1, so the table keeps the producer's rank
    # and the consumer pins them to 0; the packed stream is addressed at the
    # consumer's own rank.
    dropped: int = 0

    def __post_init__(self):
        if self.packed > self.stream:
            raise ValueError(
                f"packed CSR stream {self.packed} exceeds the declared "
                f"{self.stream} entries at rows={self.rows} tk={self.tk}"
            )

    @property
    def n_rb(self) -> int:
        return self.M // self.rows

    @property
    def n_k(self) -> int:
        return self.K // self.tk

    @property
    def packed(self) -> int:
        """Entries one batch element's packed blocks actually occupy."""
        return self.n_rb * self.n_k * self.budget

    @property
    def stream(self) -> int:
        """Declared length of one batch element's ``data`` / ``indices``.

        The op's own bound, which the rest of the graph is already built
        against -- a view flattening the batch dim, and a consumer's operand
        buffers, both carry it as a literal.  The packed blocks occupy a prefix
        of it: each block's budget truncates, so ``packed`` is never larger.
        """
        return int(self.M * self.K * self.max_pct)

    @property
    def nb(self) -> int:
        return len(self.batch)

    @property
    def ones(self) -> Tuple[int, ...]:
        """A unit extent per batch dim — every tile spans one batch element."""
        return (1,) * self.nb


def _check_block_budget(act, threshold, geom) -> None:
    """Raise unless the worst block of ``act`` fits ``geom.budget``.

    Outliers concentrate by channel, so a K slice covering outlier-heavy
    channels holds far more than the whole tensor's average rate -- the
    ``max_pct`` a block's capacity is derived from. ``_pad_csr`` truncates such
    a block rather than raising, because shape propagation runs a nest over
    buffers no step has written yet and would read uninitialized memory as
    nearly all-outlier; the activation here is the live one, so this is where
    an undersized budget is caught.

    Args:
        act: The tensor the quantize sees, ``[*batch, M, K]``; ``None`` when
            no value reached this node, which leaves nothing to check.
        threshold: Magnitude above which an element is an outlier; ``None``
            disables the check, there being no CSR.
        geom: The producer's geometry, giving the block shape and its budget.
    """
    if act is None or threshold is None or act.is_meta:
        return
    counts = (
        act.abs()
        .gt(threshold)
        .reshape(*act.shape[:-2], geom.n_rb, geom.rows, geom.n_k, geom.tk)
        .sum(dim=(-3, -1))
    )
    worst = int(counts.max())
    if worst > geom.budget:
        raise ValueError(
            f"a {geom.rows}x{geom.tk} block holds up to {worst} outliers but "
            f"its budget is {geom.budget} ({geom.max_pct:.0%} of the block); "
            f"widen the k tile to pool more channels, or raise max_pct"
        )


class _Bufs:
    """The nest's buffers, held off the module.

    Every one is allocated inside ``forward`` and closed over rather than
    scheduler-managed, since the packed stores are addressed by loop state
    rather than by the grid. Assigning them straight onto the module would make
    export demand they be registered buffers.
    """


class _OutlierProducer(torch.nn.Module):
    """Row-block sweep over a fused prefix, each row block re-diced along K.

    The outer loop is a ``PipelinedKernel`` with no scheduler-managed outputs:
    every buffer is allocated here and closed over, because the packed stores
    are addressed by loop state rather than by the grid. ``base``, ``run`` and
    the shared table use ``voyager.alloc`` in Scratchpad rather than a
    ``_ScratchSpec``, which would arrive as a trailing kernel argument on a
    signature every dense GEMM and conv shares.
    """

    def __init__(
        self,
        *,
        geom: _Geometry,
        in_specs,
        op_args,
        out_dtypes,
        mid_dtype,
        prefix_gm: Optional[torch.nn.Module],
        act_idx: int,
        kw_idx: dict,
        num_slots: int = _DEFAULT_NUM_SLOTS,
    ):
        super().__init__()
        self.geom = geom
        self.op_args = op_args  # (axes, bs, qmax, force_po2, thr, max_pct)
        self.data_dtype, self.scale_dtype, self.inlier_dtype = out_dtypes
        self.mid_dtype = mid_dtype
        self.prefix_gm = prefix_gm
        self.act_idx = act_idx
        self.kw_idx = kw_idx
        self.inner = PipelinedKernel(
            self._kernel,
            grid=geom.batch + (geom.n_rb,),
            in_specs=in_specs,
            out_specs=[],
            num_slots=num_slots,
        )

    @property
    def num_steps(self) -> int:
        return self.inner.num_steps

    # --- the op on one slice ------------------------------------------------

    def _quantize(self, tile, slots, indptr_offset):
        axes, bs, qmax, force_po2, thr, max_pct = self.op_args
        return _QUANTIZE_MX_OUTLIER(
            tile,
            slots[self.kw_idx["qmap"]],
            axes,
            bs,
            qmax,
            force_po2,
            self._maybe(slots, "scale_qmap"),
            self._maybe(slots, "output_code"),
            thr,
            max_pct,
            indptr_offset,
        )

    def _maybe(self, slots, name):
        i = self.kw_idx.get(name)
        return None if i is None else slots[i]

    # --- the two loops ------------------------------------------------------

    def _kernel(self, idx, *slots):
        """One row block: run whatever was fused, then sweep its K slices."""
        mid = self._bufs.row_block
        if self.prefix_gm is None:
            # The loaded tile is a slot window already, and the K slice below
            # would window a window (see ``_scalar``); stage it into a buffer
            # of its own instead.
            voyager.insert(slots[self.act_idx].clone(), mid)
        else:
            voyager.insert(self.prefix_gm(*slots), mid)
        self._slice_loop(idx, mid, slots)

    def _slice_loop(self, idx, mid, slots):
        g, bufs = self.geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[i] for i in range(nb)]
        m = idx[nb]

        def cond_fn(k):
            return k < g.n_k

        def body_fn(k):
            # A K slice of the resident intermediate is a window, not a
            # transfer: it is already on chip, which is the whole point of
            # splitting here rather than round-tripping through DRAM.
            tile = voyager.subview(
                mid,
                [0] * nb + [0, k * g.tk],
                ones + (g.rows, g.tk),
                (1,) * (nb + 2),
                [],
            )

            # This slice's running count, so the block emits pointers already
            # in slice coordinates rather than starting at 0.
            base_ref = voyager.subview(
                bufs.stream_pos, bidx + [0], ones + (1,), (1,) * (nb + 1), []
            )
            run_ref = voyager.subview(
                bufs.slice_nnz, bidx + [k], ones + (1,), (1,) * (nb + 1), []
            )
            offset = _scalar(run_ref)
            torch._check(offset >= 0)

            d, i, p, s, q = self._quantize(tile, slots, offset)

            # A compute result reaches DRAM through a scratchpad tile: the
            # memory model has the datapath write on chip (``insert``) and the
            # DMA move it out, never straight from the op.
            voyager.insert(d, bufs.store_data_tile)
            voyager.insert(i, bufs.store_index_tile)
            voyager.insert(p, bufs.store_indptr_tile)
            voyager.insert(s, bufs.store_scale_tile)
            voyager.insert(q, bufs.store_inlier_tile)
            d, i, p, s, q = (
                bufs.store_data_tile,
                bufs.store_index_tile,
                bufs.store_indptr_tile,
                bufs.store_scale_tile,
                bufs.store_inlier_tile,
            )

            # Grid-addressed stores.
            voyager.async_copy(
                q,
                bufs.inliers,
                bidx + [m, k],
                ones + (g.rows, g.tk),
                bufs.store_sem,
            )
            voyager.async_wait(bufs.store_sem)
            voyager.async_copy(
                s,
                bufs.scale,
                bidx + [m, k],
                ones + (g.rows, g.tk // g.block_size),
                bufs.store_sem,
            )
            voyager.async_wait(bufs.store_sem)
            # Slice k, rows [m*rows, m*rows+rows] inclusive: rows+1 entries at
            # stride rows, so consecutive blocks share a boundary element. Both
            # writes carry the same value there, so the overlap is idempotent
            # and the array's last entry falls out of the last block's write.
            voyager.async_copy(
                p.reshape(ones + (1, g.rows + 1)),
                bufs.csr_indptr,
                bidx + [k, m],
                ones + (1, g.rows + 1),
                bufs.store_sem,
                None,
                [1] * nb + [1, g.rows],
            )
            voyager.async_wait(bufs.store_sem)

            # The block's real length is the span its pointers cover, not the
            # last one: they are offset by ``run``.
            last = _entry(p, g.rows, nb, ones)
            nnz = _scalar(last) - _scalar(_entry(p, 0, nb, ones))
            torch._check(nnz >= 0)
            at = _scalar(base_ref)
            torch._check(at >= 0)

            # Packed stores. ``strides`` of 1 collapses the block-index
            # multiply so ``base`` is a raw element offset; ``sizes`` stays the
            # whole budget because it also picks the store branch, and
            # ``count`` moves only the real entries. ``dims`` must be None —
            # an empty one would drop the offset and land every block at 0.
            voyager.async_copy(
                d,
                bufs.csr_data,
                bidx + [at],
                ones + (g.budget,),
                bufs.store_sem,
                None,
                [1] * (nb + 1),
                count=[1] * nb + [nnz],
            )
            voyager.async_wait(bufs.store_sem)
            voyager.async_copy(
                i,
                bufs.csr_indices,
                bidx + [at],
                ones + (g.budget,),
                bufs.store_sem,
                None,
                [1] * (nb + 1),
                count=[1] * nb + [nnz],
            )
            voyager.async_wait(bufs.store_sem)

            # Bookkeeping, both scratchpad writes: where this block sits in the
            # stream (for the consumer), and where the slice now stands.
            voyager.insert(
                base_ref.reshape(ones + (1, 1)).clone(),
                voyager.subview(
                    bufs.base_table,
                    bidx + [k, m],
                    ones + (1, 1),
                    (1,) * (nb + 2),
                    [],
                ),
            )
            voyager.insert(last.clone(), run_ref)
            # Advance the packed-stream position for the next block: an op
            # result, so it has a destination of its own.
            voyager.insert(base_ref + nnz, base_ref)
            return (k + 1,)

        # The loop's result must be consumed: an unused ``while_loop`` is
        # pruned, taking the body's buffer stores with it.
        (k_end,) = while_loop(cond_fn, body_fn, (0,))
        torch._check(k_end >= 0)

    # --- entry --------------------------------------------------------------

    def forward(self, *inputs):
        g = self.geom
        bufs = _Bufs()
        bufs.csr_data = voyager.alloc(
            list(g.batch) + [g.stream], self.data_dtype
        )
        bufs.csr_indices = voyager.alloc(
            list(g.batch) + [g.stream], torch.int32
        )
        bufs.csr_indptr = voyager.alloc(
            list(g.batch) + [g.n_k, g.M + 1], torch.int32
        )
        bufs.scale = voyager.alloc(
            list(g.batch) + [g.M, g.K // g.block_size], self.scale_dtype
        )
        bufs.inliers = voyager.alloc(
            list(g.batch) + [g.M, g.K], self.inlier_dtype
        )

        bufs.row_block = voyager.alloc(
            list(g.ones) + [g.rows, g.K], self.mid_dtype, _SRAM
        )
        # ``alloc`` rather than ``zeros``: a ``zeros`` is read as a semaphore
        # buffer and carries no memory space, so an ``insert`` into one is a
        # tile-store violation. These are tensor buffers the loop writes.
        bufs.slice_nnz = voyager.alloc(
            list(g.batch) + [g.n_k], torch.int32, _SRAM
        )
        bufs.stream_pos = voyager.alloc(list(g.batch) + [1], torch.int32, _SRAM)
        bufs.base_table = voyager.alloc(
            list(g.batch) + [g.n_k, g.n_rb], torch.int32, _SRAM
        )
        bufs.store_sem = voyager.zeros([], torch.int64)
        # Per-block staging tiles for ``_slice_loop``'s stores.
        bufs.store_data_tile = voyager.alloc(
            list(g.ones) + [g.budget], self.data_dtype, _SRAM
        )
        bufs.store_index_tile = voyager.alloc(
            list(g.ones) + [g.budget], torch.int32, _SRAM
        )
        bufs.store_indptr_tile = voyager.alloc(
            list(g.ones) + [g.rows + 1], torch.int32, _SRAM
        )
        bufs.store_scale_tile = voyager.alloc(
            list(g.ones) + [g.rows, g.tk // g.block_size],
            self.scale_dtype,
            _SRAM,
        )
        bufs.store_inlier_tile = voyager.alloc(
            list(g.ones) + [g.rows, g.tk], self.inlier_dtype, _SRAM
        )

        self._bufs = bufs
        self.inner(*inputs)
        return (
            bufs.csr_data,
            bufs.csr_indices,
            bufs.csr_indptr,
            bufs.scale,
            bufs.inliers,
        )


def base_table_shape(geom: _Geometry) -> list:
    """Shape of the base table both the producer and its consumers allocate.

    Always the producer's rank: a consumer reached through a reshape sees
    fewer batch dims, but the table is one buffer and both must agree.
    """
    return [1] * geom.dropped + list(geom.batch) + [geom.n_k, geom.n_rb]


def tag_base_table(gm, producer_name: str, shape: list) -> None:
    """Mark this nest's base-table alloc as belonging to ``producer_name``.

    The producer and each consumer allocate the table independently, because
    neither can see the other's graph; a pass in ``bufferize_graph`` merges the
    tagged group once every nest has been spliced. Matching on the shape picks
    it out from the nest's other integer buffers. A nest that both
    consumes and produces a CSR holds two tables, so each call claims the
    first untagged match and stops.
    """
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target is not voyager.alloc.default:
            continue
        if n.meta.get(BASE_TABLE_TAG) is not None:
            continue
        if list(n.args[0]) == list(shape) and n.args[1] == torch.int32:
            n.meta[BASE_TABLE_TAG] = producer_name
            return


def merge_base_tables(model) -> None:
    """Collapse each producer's base-table allocs into one buffer.

    A producer writes every block's position in its packed CSR stream into a
    scratchpad table its consumers read; neither can see the other's graph, so
    each nest allocates the table itself and tags it with the producer's name
    (:func:`tag_base_table`). Once every nest is spliced, the group is one
    buffer: keep the first alloc, repoint the rest at it and erase them.
    """
    groups = {}
    for node in model.graph.nodes:
        tag = node.meta.get(BASE_TABLE_TAG)
        if tag is not None:
            groups.setdefault(tag, []).append(node)

    for tag, allocs in groups.items():
        keep, rest = allocs[0], allocs[1:]
        for other in rest:
            if list(other.args[0]) != list(keep.args[0]):
                raise ValueError(
                    f"base table '{tag}': allocs disagree on shape "
                    f"({list(keep.args[0])} vs {list(other.args[0])})"
                )
            other.replace_all_uses_with(keep)
            model.graph.erase_node(other)
        if rest:
            logger.debug(
                "[bufferize] merged %d base-table allocs for %s",
                len(allocs),
                tag,
            )


def _quantize_node(node):
    """The ``quantize_mx_outlier`` this node lowers, or ``None``.

    A bare node is its own; a fused ``call_module`` carries it as the terminal
    op of its submodule.
    """
    if node.op == "call_function" and node.target is _QUANTIZE_MX_OUTLIER:
        return node
    if node.op != "call_module":
        return None
    submod = node.meta.get("submodule")
    if not isinstance(submod, torch.fx.GraphModule):
        return None
    for n in submod.graph.nodes:
        if n.op == "call_function" and n.target is _QUANTIZE_MX_OUTLIER:
            return n
    return None


def _split_prefix(submod, qnode):
    """Everything the fused group computes ahead of the quantize, as a gm.

    Keeps every submodule placeholder in its original order so the result can
    be called with the kernel's full slot list; the quantize's own codebook
    operands come along as unused placeholders.

    Args:
        submod: The fused submodule.
        qnode: Its ``quantize_mx_outlier`` node.

    Returns:
        A ``GraphModule`` mapping the submodule's inputs to the quantize's
        activation operand, or ``None`` when the quantize consumes a
        placeholder directly (nothing is fused ahead of it).
    """
    source = qnode.args[0]
    if source.op == "placeholder":
        return None

    g = torch.fx.Graph()
    remap = {}
    for n in submod.graph.nodes:
        if n.op == "placeholder":
            remap[n] = g.placeholder(n.name)
    keep = ancestors(source) | {source}
    for n in submod.graph.nodes:
        if n.op in ("placeholder", "output") or n not in keep:
            continue
        remap[n] = g.node_copy(n, lambda x: remap[x])
    g.output(remap[source])
    g.lint()
    return torch.fx.GraphModule(submod, g)


def consumer_k_tile(node, tiler) -> Tuple[Optional[int], Optional[int]]:
    """The ``tk`` every sparse consumer of ``node`` agrees on.

    Each consumer reads column indices local to the weight tile it multiplies
    and the engine cannot rebase them, so a producer feeding several GEMMs must
    slice K the way all of them do. ``bufferize_graph`` runs
    ``prefetch_tilings`` over every GEMM before the build loop, so their
    tilings are already decided by the time this is asked.

    Args:
        node: The producer being lowered.
        tiler: The interstellar ``TilerContext``.

    Returns:
        ``(tk, tm)`` — the K slice every consumer will read, and the smallest
        row tile any of them uses. ``(None, None)`` when no sparse consumer
        names one: a producer whose CSR feeds the model output, for instance,
        in which case the caller keeps K whole and sizes rows itself.
    """
    found, rows = {}, []
    for use in node.users:
        if use.target is not operator.getitem:
            continue
        # A nop view can sit between the CSR and its consumer (the reshape
        # between a decoder layer and ``lm_head``); walk through it.
        stack = list(use.users)
        while stack:
            user = stack.pop()
            if is_nop(user):
                stack.extend(user.users)
                continue
            anchor = get_anchor_node(user)
            if anchor is None or not is_gemm_op(anchor):
                continue
            if anchor.kwargs.get("A_indptr") is None:
                continue
            tiling = get_tiling(user, tiler)[0]
            if tiling is None:
                continue
            shape = anchor.args[0].value.shape
            found[user.name] = shape[-1] // tiling[-1]
            nb = anchor.value.ndim - 2
            rows.append(shape[-2] // tiling[nb])
    if not found:
        return None, None
    # Consumers of one producer often want different k tiles (q/k/v differ from
    # each other in N, and gate_proj from up_proj), but the CSR is produced
    # already partitioned, so they must share one. Take the finest: a consumer
    # forced to a *smaller* k tile only ever shrinks its operand tiles, which
    # cannot stop fitting the scratchpad, while a larger one could.
    tk = min(found.values())
    if len(set(found.values())) > 1:
        logger.info(
            "%s: sparse consumers want k tiles %s; using %d for all",
            node.name,
            found,
            tk,
        )
    return tk, min(rows)


# What one row of the nest costs, in units of the ``[rows, K]`` intermediate.
# The tile search cannot size this nest: the intermediate is allocated inside
# it and is invisible to the search, which sees only the operands crossing the
# boundary.  Three buffers scale with the row block -- the intermediate, the
# input tile it is produced from (the same rows at the same width), and the
# per-slice staging tiles, which together span one K slice and so cannot
# exceed the intermediate either.
_NEST_ROW_BLOCK_COPIES = 3


def _row_block(node, tiler, geom_shape, mid_bytes_per_row, cap=None) -> int:
    """Rows the outer loop handles per step.

    Starts from the vector-op tile search (which sizes the operands the fused
    group loads) and shrinks until everything in the nest that scales with the
    row block fits the scratchpad -- not the intermediate alone, which is only
    a third of it (``_NEST_ROW_BLOCK_COPIES``).

    ``cap`` is the smallest row tile any sparse consumer uses. A consumer
    gathers a whole number of blocks, so the block must divide its row tile --
    which makes a block no larger than one.
    """
    M = geom_shape
    tiling = vector_op_tiling(node, tiler.config)
    rows = M if tiling is None else max(1, M // max(1, tiling[-2]))
    if cap is not None:
        rows = min(rows, cap)
    per_row = _NEST_ROW_BLOCK_COPIES * mid_bytes_per_row
    slot_size = tiler.config.scratchpad_size // tiler.config.num_slots
    while rows > 1 and rows * per_row > slot_size:
        rows //= 2
    while rows > 1 and (M % rows or (cap is not None and cap % rows)):
        rows -= 1
    return rows


def build_quantize_mx_outlier(
    node, *, tiler, num_slots: int = _DEFAULT_NUM_SLOTS
) -> Optional[torch.fx.GraphModule]:
    """Bufferize a row-swept ``quantize_mx_outlier`` — bare or fused, 2-D or
    batched — as a row-block loop whose body re-dices each block along K.

    Args:
        node: The producer: a bare ``quantize_mx_outlier`` call, or a fused
            ``call_module`` whose submodule ends in one.
        tiler: The interstellar ``TilerContext``; supplies both the consumer's
            k tile and the row-block search.
        num_slots: Software-pipeline depth for the outer loop's inputs.

    Returns:
        The bufferized gm, or ``None`` when ``node`` is not one this builds.
    """
    qnode = _quantize_node(node)
    if qnode is None or tiler is None:
        return None

    act = qnode.args[0].value
    if act.ndim < 2:
        raise ValueError(
            f"{node.name}: a {act.ndim}-D activation has no rows to sweep"
        )
    batch, M, K = tuple(act.shape[:-2]), act.shape[-2], act.shape[-1]

    bs = qnode.args[3]
    tk, tm_consumer = consumer_k_tile(node, tiler)
    tk = tk or K
    if K % tk or tk % bs:
        raise ValueError(
            f"{node.name}: k tile {tk} must divide K={K} and hold whole "
            f"microscaling blocks of {bs}"
        )

    max_pct = get_arg_value(qnode, 9, "max_pct", 0.01)
    submod = node.meta.get("submodule")
    prefix_gm = (
        _split_prefix(submod, qnode)
        if isinstance(submod, torch.fx.GraphModule)
        else None
    )

    rows = _row_block(node, tiler, M, K * act.element_size(), cap=tm_consumer)
    if M % rows:
        raise ValueError(f"{node.name}: row block {rows} does not divide M={M}")
    geom = _Geometry(
        batch=batch,
        M=M,
        K=K,
        rows=rows,
        tk=tk,
        block_size=bs,
        budget=int(rows * tk * max_pct),
        max_pct=max_pct,
    )
    threshold = get_arg_value(qnode, 8, "threshold")
    _check_block_budget(act, threshold, geom)

    # Operand roles. The activation tiles along the row-block grid dim and
    # keeps K whole (the inner loop dices it); codebooks load whole.
    in_nodes = list(node.all_input_nodes)
    order = {n: i for i, n in enumerate(in_nodes)}
    codebooks = quant_param_arg_nodes(qnode)
    src = lambda n: n.meta.get("source_node", n)

    # A fused submodule's placeholders sit in ``all_input_nodes`` order, so an
    # operand the quantize names inside the submodule is found by position.
    if isinstance(submod, torch.fx.GraphModule):
        phs = [n for n in submod.graph.nodes if n.op == "placeholder"]
        slot = {p: i for i, p in enumerate(phs)}
        outer = lambda v: slot.get(v)
        codebooks = {in_nodes[slot[c]] for c in codebooks if c in slot}
    else:
        outer = lambda v: order.get(src(v)) if isinstance(v, Node) else None
        codebooks = {src(c) for c in codebooks}

    # Tile *counts*: every batch dim is diced to the element (one grid step per
    # batch element, since each carries its own CSR), rows to the row block,
    # and K left whole for the inner loop to slice.
    out_tiling = tuple(batch) + (geom.n_rb, 1)
    grid_map = tuple(range(len(batch))) + (len(batch), None)
    in_specs = []
    for n in in_nodes:
        if n in codebooks or n.value.numel() == 1:
            in_specs.append(None)
        else:
            in_specs.append(
                _compute_input_spec(out_tiling, tuple(n.shape), grid_map)
            )

    # The quantize's own tensor operands, by slot: bound here because the
    # kernel body is traced, where dynamo rejects an FX-node lookup.
    names = ("qmap", "scale_qmap", "output_code")
    kw_idx = {}
    for name, i in zip(names, (1, 6, 7)):
        v = qnode.args[i] if len(qnode.args) > i else None
        if isinstance(v, Node) and (j := outer(v)) is not None:
            kw_idx[name] = j
    if "qmap" not in kw_idx:
        raise ValueError(
            f"{node.name}: its quantize's qmap is not one of the group's "
            f"operands, so the kernel cannot reach it"
        )

    act_idx = outer(qnode.args[0])
    if act_idx is None and prefix_gm is None:
        raise ValueError(
            f"{node.name}: nothing is fused ahead of the quantize and its "
            f"activation is not one of the group's operands"
        )
    vals = node.value
    producer = _OutlierProducer(
        geom=geom,
        in_specs=in_specs,
        # ``axes`` as a plain list: an FX ``immutable_list`` has no
        # ``as_proxy()``, so dynamo cannot capture the op call inside the loop.
        op_args=(
            list(qnode.args[2]),
            bs,
            get_arg_value(qnode, 4, "quant_max"),
            get_arg_value(qnode, 5, "force_scale_power_of_two", False),
            threshold,
            max_pct,
        ),
        out_dtypes=(vals[0].dtype, vals[3].dtype, vals[4].dtype),
        mid_dtype=act.dtype,
        prefix_gm=prefix_gm,
        act_idx=act_idx,
        kw_idx=kw_idx,
        num_slots=num_slots,
    )

    inputs = tuple(n.value.clone() for n in in_nodes)
    with _lenient_verifier():
        gm = export_model(producer, inputs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(0, producer.num_steps, 1)], [(0, geom.n_k, 1)]])
    tag_base_table(gm, node.name, base_table_shape(geom))
    # The consumer has to know how the stream was diced -- how many blocks span
    # one of its row tiles, and how big each may be -- and cannot derive it
    # from the CSR's shapes alone. ``bufferize_graph`` copies this onto the
    # spliced results, where the consumer finds it through its A_indptr kwarg.
    gm.meta[GEOMETRY_META] = geom
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*inputs)
    # Whatever was fused ahead of the quantize inlines as loose nodes; the
    # memory model wants one destination-passing op per store, so outline
    # them.  After ShapeProp, which the outlining reads values from.
    if prefix_gm is not None:
        outline_dps_ops(gm)
    return gm
