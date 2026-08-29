"""A GEMM whose activation carries an outlier CSR.

``linear_mx`` / ``matmul_mx`` fuse a dense low-precision matmul with a sparse
high-precision correction, and the engine takes **one** CSR per call covering
exactly the dense tile's rows. The producer, though, emits its CSR per
``(row block, K slice)`` and packs the blocks end to end (see
``quantize_mx_outlier``), so a row tile spanning ``R`` row blocks finds its
correction in ``R`` runs scattered through that stream. The consumer therefore
concatenates them on chip before the call.

That concatenation is a valid CSR without re-indexing anything. Column indices
are already local to ``tk``, so a block's columns lie in ``[0, tk)`` whichever
rows they came from, and stacking blocks that share a ``k`` just extends the
row list. The row pointers need no work either: the producer wrote them in
slice coordinates, so the tile's pointer range is a contiguous window of one
array, and the engine subtracts ``indptr[0]`` itself. Placing block ``j``'s
data at ``p[j*rows] - p[0]`` is exactly what makes those pointers address it.

Everything else is the dense GEMM: ``_gemm_plan`` derives the grid, the operand
specs and the per-step call, and this module only adds the fetches and swaps
the CSR kwargs for its own scratch.
"""

import contextlib
import math
from dataclasses import replace
from typing import Optional

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler.codegen.node_info import (
    get_anchor_node,
    is_gemm_op,
    is_nop,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    MemoryLevel,
    oracle_disabled,
    UNPIPELINED,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    _bank_group_list,
    _DEFAULT_NUM_SLOTS,
    _DONE_DEPTH,
    _gemm_plan,
    _gemm_scratch_and_kernel,
    _split_stream_break,
    _stamp_anchor_meta,
    _stamp_bank_groups,
    AsyncPipelinedKernel,
    get_slot,
    PipelinedKernel,
)
from voyager_compiler.codegen.transform.bufferize.quantize_mx_outlier import (
    _Bufs,
    _entry,
    _fit_block_budget,
    _Geometry,
    _QUANTIZE_MX_OUTLIER,
    _scalar,
    _split_prefix,
    base_table_shape,
    consumer_k_tile,
    copy_base_table,
    GEOMETRY_META,
    PRODUCER_META,
    tag_base_table,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    effect_cond,
    outline_dps_ops,
    voyager,
)
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import set_node_value, ShapeProp

_SRAM = int(MemoryLevel.SRAM)

# Stores ``_store_staged_csr`` issues per staged sub-slice (the CSR's
# indptr, packed data and packed indices, then the dense scale and
# inliers), all posting ``store_sem`` — the posts one drain must consume.
_CSR_STORES = 5

# Steps between a chained finalize and the store that empties its staging
# tiles, in the async nest.  The scheduler's lagged retire waits a commit's
# post one step after dispatching it, so a step's own results are readable
# only from two steps on — the earliest a store can run without a wait of
# its own.
_STORE_LAG = 2


class _EpilogueTail(torch.nn.Module):
    """The fused tail of a CSR-producing GEMM, plus its CSR stores.

    Runs the tail's prefix once on the whole accumulator tile, then the
    quantize once per ``tk``-wide sub-slice of it, so the CSR comes out at
    the *consumers'* slice width while this GEMM's own tiling stays whatever
    its search picked.  Wrapping rather than threading extra arguments keeps
    the tail's arity: the reduction kernel applies it on the last reduction
    step and calls it with the group's own operands.  The offset, the grid
    index and the output slots it needs are captured here instead, which is
    also what puts the stores on the step that actually completes an output
    tile.  The stores below own their subgraph — the rolled sub-slice loop
    — rather than relying on the reduction's last-step ``torch.cond``, which
    a single-round reduction does not emit.

    With ``chain_epilogue`` the tail is pure compute, chained on the live
    total inside the anchor's own pass: all five results land in the staging
    tiles as destinations and the stores run bare afterwards — on the same
    round in the sync nest (``_SparseGemm._kernel``), two steps later in
    the async one (``_SparseGemm._async_kernel``).  Otherwise the stores
    run here, interleaved.
    """

    def __init__(self, owner, idx, scratch=None):
        super().__init__()
        self.owner = owner
        self.idx = idx
        self.scratch = scratch

    def _sub_slice(self, tile, j, offset, bound, tiles):
        """Quantize one sub-slice into the staging tiles, and store it.

        Args:
            tile: The sub-slice of the output tile to quantize.
            j: Its index within the column tile, naming its K slice.
            offset: The ``indptr_offset`` an async chained tail receives as
                an operand, or ``None`` to read the running count here.
            bound: The tail's operand list, which ``epi_q_args`` indexes.
            tiles: The five staging tiles the results land in.
        """
        o = self.owner

        def val(t):
            kind, v = t
            return bound[v] if kind == "arg" else v

        d, i, p, s, q = _QUANTIZE_MX_OUTLIER(
            tile,
            *[val(t) for t in o.epi_q_args],
            **{k: val(t) for k, t in o.epi_q_kwargs.items()},
            indptr_offset=(
                offset if offset is not None else o._offset_at(self.idx, j)
            ),
        )
        for res, tile in zip((d, i, p, s, q), tiles):
            voyager.insert(res, tile)
        if not o.chain_epilogue:
            o._store_staged_csr(self.idx, j)
            o._drain_csr_stores()

    def forward(self, acc, *operands):
        o = self.owner
        og = o.out_geom
        bufs = o._bufs
        nb, ones = og.nb, og.ones
        # The async chained tail runs inside a commit, whose traced body
        # must not capture free values: its staging tiles and precomputed
        # ``indptr_offset`` ride the commit's operand list and arrive as
        # trailing operands here.  The sync chained tail runs inline and
        # reaches both directly.
        offset = None
        if o.chain_epilogue and o.async_pipeline:
            tiles = operands[-6:-1]
            offset = operands[-1]
            operands = operands[:-6]
        else:
            tiles = (
                bufs.store_data_tile,
                bufs.store_index_tile,
                bufs.store_indptr_tile,
                bufs.store_scale_tile,
                bufs.store_inlier_tile,
            )
        bound = (acc,) + operands

        x = acc if o.epi_prefix is None else o.epi_prefix(*bound)
        if o.n_sub == 1:
            self._sub_slice(x, 0, offset, bound, tiles)
            return None

        # TODO: chain the prefix onto the reduction instead.  It is pure
        # compute, so it could run on the live total of the last K step --
        # the ``split`` head path -- and stage only its result, saving the
        # full read and write of the tile below.  It does not today because
        # ``stream_breaking_quantize`` recognizes only ``quantize_mx``, so
        # an ``_EpilogueTail`` is never split and prefix, quantize and
        # stores stay one unchainable unit.
        if o.epi_prefix is not None:
            if x.dtype != self.scratch.dtype:
                x = x.to(self.scratch.dtype)
            voyager.insert(x, self.scratch)

        def cond_fn(j):
            return j < o.n_sub

        def body_fn(j):
            tile = voyager.subview(
                self.scratch,
                [0] * nb + [0, j * og.tk],
                ones + (og.rows, og.tk),
                (1,) * (nb + 2),
                [],
            )
            if tile.dtype != o.plan.out_dtype:
                tile = tile.to(o.plan.out_dtype)
            self._sub_slice(tile, j, offset, bound, tiles)
            return (j + 1,)

        (j_end,) = while_loop(cond_fn, body_fn, (0,))
        torch._check(j_end >= 0)
        return None


class _SparseGemm(torch.nn.Module):
    """The dense GEMM nest plus a per-step CSR gather.

    The gather cannot live inside the plan's ``gemm_kernel``, which is
    index-blind by design: a block's source address comes from the shared base
    table, indexed by ``(K slice, row block)``. So it runs here, where the grid
    index is in hand, and the kernel below only substitutes the filled scratch
    for the raw operands.

    Two scheduling modes.  The synchronous nest waits every operand inline.
    The async nest (``async_pipeline``) dispatches each step's compute as a
    ``voyager.commit`` whose dependencies are the dense load semaphores plus
    this step's gather copies, so the whole fetch chain overlaps the previous
    tile's compute; the one wait the control stream keeps is the pointer
    tile's, whose values drive the gather DMAs' addresses at issue time.
    """

    def __init__(
        self,
        plan,
        geom,
        data_dtype,
        *,
        out_geom=None,
        accumulate_fusible: bool,
        num_slots: int = _DEFAULT_NUM_SLOTS,
        accumulate_fp32: bool = False,
        async_pipeline: bool = False,
    ):
        super().__init__()
        self.plan = plan
        self.accumulate_fp32 = accumulate_fp32
        self.data_dtype = data_dtype
        self.geom = geom
        self.out_geom = out_geom
        self.R = plan.tile_m // geom.rows if geom is not None else 1
        self.span = self.R * geom.budget if geom is not None else 0
        self.grid_m, self.grid_n, self.grid_k = plan.grid_dims
        if out_geom is not None:
            # Every output is stored by hand, the dense pair by the same
            # route as the CSR, so the output grid dices none of them.
            plan.out_specs = list(plan.out_specs)[5:]
            # The quantize runs per ``tk``-wide sub-slice of the accumulator
            # tile (see ``_EpilogueTail``): split the tail at the quantize and
            # bind the op's own arguments to their positions in the tail's
            # operand list, so the per-slice calls can be issued directly.
            self.n_sub = plan.tile_n // out_geom.tk
            qnode = next(
                n
                for n in plan.fused_gm.graph.nodes
                if n.op == "call_function" and n.target is _QUANTIZE_MX_OUTLIER
            )
            self.epi_prefix = _split_prefix(plan.fused_gm, qnode)
            phs = [
                n for n in plan.fused_gm.graph.nodes if n.op == "placeholder"
            ]
            slot = {p: i for i, p in enumerate(phs)}

            def resolve(a):
                if isinstance(a, torch.fx.Node):
                    return ("arg", slot[a])
                if isinstance(a, (list, tuple)):
                    return ("lit", list(a))
                return ("lit", a)

            self.epi_q_args = [resolve(a) for a in qnode.args[1:]]
            self.epi_q_kwargs = {k: resolve(v) for k, v in qnode.kwargs.items()}
            # A fitted budget reaches the per-sub-slice calls through their
            # ``max_pct``; an unfitted geometry leaves the op's own
            # arguments untouched.
            if out_geom.op_max_pct != out_geom.max_pct:
                if len(qnode.args) > 9:
                    self.epi_q_args = self.epi_q_args[:8]
                self.epi_q_kwargs["max_pct"] = ("lit", out_geom.op_max_pct)
        # The CSR-store epilogue runs bare in the loop body after the
        # commit — its DMAs, waits and stream bookkeeping cannot live in a
        # commit subgraph — which requires a single-slot scratch
        # (``_reduction_fused_kernel``'s bare-tail path).
        if out_geom is not None and async_pipeline:
            plan.anchor.meta.setdefault("tiling", {})["scratch_slots"] = 1
        self.async_pipeline = async_pipeline
        self.chain_epilogue = out_geom is not None and self.n_sub == 1
        self.chain_fused_tail = accumulate_fusible and out_geom is None
        if geom is not None:
            # ``in_sems`` carries one semaphore per *tiled* operand — a
            # ``None``-spec operand is passed through without one — so the
            # pointer semaphore is indexed by rank among the real specs.
            self.ptr_sem_index = sum(
                1
                for s in plan.in_specs[: plan.kw_idx["A_indptr"]]
                if s is not None
            )
        # Classified once: the traced per-step kernel rebuild cannot walk
        # a graph.
        self.tail_split = _split_stream_break(plan.fused_gm, plan.acc_shape)
        # A multi-sub-slice epilogue reads windows of its tile at an index
        # the rolled loop supplies at runtime, and only a buffer can be
        # addressed that way.  ``staged`` is what puts the tile in one: its
        # sole effect is to disqualify ``_map_kernel``, which allocates no
        # scratch, leaving the reduction scratch to read windows out of.
        self.stage_tile = out_geom is not None and self.n_sub > 1
        # A finalize writes the five staging tiles, and the store that
        # empties them runs ``_STORE_LAG`` steps later; in between, no other
        # finalize may write them.  With ``num_k > 1`` finalizes are
        # ``num_k`` steps apart and none lands in the gap, so one copy of
        # each tile does.  With ``num_k == 1`` every step finalizes, so the
        # next one would: the tiles rotate instead, and consecutive output
        # tiles alternate between the copies.  A finalize on the store's own
        # step is safe either way -- the drain below runs first.
        self.store_slots = (
            _STORE_LAG
            if self.chain_epilogue and async_pipeline and plan.num_k == 1
            else 1
        )
        # Output tiles the sweep completes — the reduction dim is last.
        self.tile_count = math.prod(plan.grid[: self.grid_k])
        # The K accumulator, when the reduction needs one.  The kernel built
        # alongside it is rebuilt per step (with the grid index captured) and
        # discarded here; only the specs are needed before the loop exists.
        self.scratch_specs = self._scratch_and_kernel(plan.fused_gm)[0]
        if async_pipeline:
            kernel, kernel_cls = self._async_kernel, AsyncPipelinedKernel
        else:
            kernel, kernel_cls = self._kernel, PipelinedKernel
        self.inner = kernel_cls(
            kernel,
            grid=plan.grid,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            scratch_specs=self.scratch_specs,
            num_slots=num_slots,
        )

    # --- kernel assembly ----------------------------------------------------

    def _scratch_and_kernel(self, tail):
        """The dense path's scratch and kernel, over a ``gemm_kernel`` that
        reads the gathered CSR instead of the raw operands.

        The gathered blocks reach the op differently per mode: the async
        kernel appends this step's gather-slot views to the commit's operands
        (a traced commit subgraph must not capture free tensors), so they
        arrive as extra trailing entries of ``in_tiles``; the sync kernel
        reads the single-slot staging buffers directly.

        Args:
            tail: The fused tail to apply — the plan's own, or its
                ``_EpilogueTail`` wrapper when a step's tail also stores a
                CSR. Rebuilt per step because a HOP body may not mutate an
                outer Python object, so the stores have to run where the
                results exist.

        Returns:
            ``(scratch_specs, kernel)``, the kernel matching the arity of
            the dense path's.
        """
        plan = self.plan
        num_operands = len(plan.in_specs)

        indptr_idx = plan.kw_idx["A_indptr"]
        indices_idx = plan.kw_idx["A_indices"]
        data_idx = plan.kw_idx["A_data"]

        def gemm_kernel(in_tiles, first):
            tiles = list(in_tiles[:num_operands])
            if self.geom is not None:
                if self.async_pipeline:
                    tiles[data_idx] = in_tiles[num_operands]
                    tiles[indices_idx] = in_tiles[num_operands + 1]
                else:
                    tiles[data_idx] = self._bufs.gather_data
                    tiles[indices_idx] = self._bufs.gather_indices
                # One call consumes one K slice, so the slice axis is
                # addressing, not data: the op wants a plain [*batch, tm+1]
                # pointer array.
                ptr = in_tiles[indptr_idx]
                tiles[indptr_idx] = ptr.reshape(
                    tuple(ptr.shape[:-2]) + (ptr.shape[-1],)
                )
            return plan.gemm_kernel(tuple(tiles), first)

        fused_idx = list(plan.fused_idx)
        if self.chain_epilogue and self.async_pipeline:
            # The staging tiles and the offset ride as trailing kernel
            # operands (after the gather views); ``_finalize`` hands them
            # to the tail with the group's own operands.
            base = num_operands + (2 if self.geom is not None else 0)
            fused_idx += [base + n for n in range(6)]
        return _gemm_scratch_and_kernel(
            gemm_kernel,
            tail,
            reduction_dim=self.grid_k,
            num_k=plan.num_k,
            acc_shape=plan.acc_shape,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            fused_idx=fused_idx,
            anchor=plan.anchor,
            accumulate_fp32=self.accumulate_fp32,
            # A multi-slice epilogue re-reads the staged accumulator — its
            # interleaved stores cannot ride the pass; a single-slice tail
            # is pure compute and chains, its stores running bare in
            # ``_kernel`` / ``_async_kernel``.  A consumer's fusible tail
            # chains like a dense GEMM's.
            chain_tail=self.chain_epilogue or self.chain_fused_tail,
            async_pipeline=self.async_pipeline,
            # An ``_EpilogueTail`` stages its own stores and is never split.
            split=self.tail_split if tail is plan.fused_gm else None,
            staged=self.stage_tile,
        )

    def _gather_rolled(self, idx, slots, parity=None):
        """``_gather``, rolled into a ``while_loop`` over the ``R`` blocks.

        The same copies in the same order; only the ``R`` repetitions the
        graph spells out collapse.  Two things are picked before the loop:
        the pointer tile's first entry, which every block subtracts, and
        this parity's row of the semaphore array, so a copy inside names the
        slot it signals as ``2 * j + b`` -- the same expression the caller
        applies to the same row to rebuild those views for the compute
        commit's dependency list, which a ``while_loop`` body cannot hand
        back.  A trip reads both of its own block's bounds rather than
        carrying the boundary it shares with the next one, so the loop
        carries nothing but its index, at one extra scalar read per block.

        Args:
            idx: The grid coordinate.
            slots: The operand tiles, positionally.
            parity: The gather slot this step writes (async mode), or ``None``
                for the synchronous mode -- single-slot buffers, every copy
                waited inline.

        Returns:
            The semaphore-slot views the copies signal (async), else ``[]``.
        """
        plan, g, bufs = self.plan, self.geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[i] for i in range(nb)]
        m_c, k = idx[self.grid_m], idx[self.grid_k]

        p = slots[plan.kw_idx["A_indptr"]].reshape(ones + (plan.tile_m + 1,))
        lo = _scalar(_entry(p, 0, nb, ones))
        pairs = (
            (slots[plan.kw_idx["A_data"]], bufs.gather_data),
            (slots[plan.kw_idx["A_indices"]], bufs.gather_indices),
        )
        row = None if parity is None else get_slot(bufs.gather_sem, parity)

        def cond_fn(j):
            return j < self.R

        def body_fn(j):
            slot = voyager.subview(
                bufs.base_table,
                [0] * g.dropped + bidx + [k, m_c * self.R + j],
                (1,) * g.dropped + ones + (1, 1),
                (1,) * (g.dropped + nb + 2),
                [],
            )
            at = _scalar(slot)
            torch._check(at >= 0)
            at_j = _scalar(_entry(p, j * g.rows, nb, ones))
            start = at_j - lo
            torch._check(start >= 0)
            count = _scalar(_entry(p, (j + 1) * g.rows, nb, ones)) - at_j
            torch._check(count >= 0)

            for b, (src, dst) in enumerate(pairs):
                if parity is None:
                    window = voyager.subview(
                        dst,
                        [0] * nb + [start],
                        ones + (g.budget,),
                        (1,) * (nb + 1),
                        [],
                    )
                    sem = bufs.gather_sem
                else:
                    window = voyager.subview(
                        dst,
                        [parity] + [0] * nb + [start],
                        [1] + list(ones) + [g.budget],
                        (1,) * (nb + 2),
                        [0],
                    )
                    sem = get_slot(row, 2 * j + b)
                voyager.async_copy(
                    src,
                    window,
                    bidx + [at],
                    ones + (g.budget,),
                    sem,
                    None,
                    [1] * (nb + 1),
                    count=[1] * nb + [count],
                )
                if parity is None:
                    voyager.async_wait(sem)
            return (j + 1,)

        # An unused ``while_loop`` is pruned, taking the body's copies with
        # it, so the trip count has to be consumed.
        (j_end,) = while_loop(cond_fn, body_fn, (0,))
        torch._check(j_end >= 0)
        if parity is None:
            return []
        # The body cannot return these; rebuild them off the same row, in
        # the order the copies inside signal them.  Their offsets are
        # literal, so they cost no arithmetic of their own.
        return [
            get_slot(row, 2 * j + b) for j in range(self.R) for b in range(2)
        ]

    def _offset_at(self, idx, j):
        """Sub-slice ``j``'s running nonzero count, the op's
        ``indptr_offset``."""
        g, bufs = self.out_geom, self._bufs
        nb = g.nb
        off = _scalar(
            voyager.subview(
                bufs.slice_nnz,
                [idx[x] for x in range(nb)]
                + [idx[self.grid_n] * self.n_sub + j],
                g.ones + (1,),
                (1,) * (nb + 1),
                [],
            )
        )
        torch._check(off >= 0)
        return off

    def _drain_csr_stores(self):
        """Consume one staged-store round's posts from ``store_sem``."""
        for _ in range(_CSR_STORES):
            voyager.async_wait(self._bufs.store_sem)

    def _tile_ordinal(self, idx):
        """The output tile's position in the sweep — the flat index over
        every grid dim but the reduction's, which is the last."""
        ordinal = 0
        for d in range(self.grid_k):
            ordinal = ordinal * self.plan.grid[d] + idx[d]
        return ordinal

    def _tile_coord(self, ordinal):
        """Grid coordinate of output tile ``ordinal``'s finalize step, as
        plain integers — the after-loop stores address literal tiles."""
        coord, rest = [], ordinal
        for extent in reversed(self.plan.grid[: self.grid_k]):
            coord.append(rest % extent)
            rest //= extent
        coord.reverse()
        return coord + [self.plan.num_k - 1]

    def _store_tiles(self, ordinal):
        """The five staging tiles output tile ``ordinal`` writes — its slot
        of each, where they rotate."""
        bufs = self._bufs
        tiles = (
            bufs.store_data_tile,
            bufs.store_index_tile,
            bufs.store_indptr_tile,
            bufs.store_scale_tile,
            bufs.store_inlier_tile,
        )
        if self.store_slots == 1:
            return tiles
        return tuple(get_slot(t, ordinal % self.store_slots) for t in tiles)

    def _store_staged_csr(self, idx, j):
        """Store one staged sub-slice: its CSR block and its dense pair.

        The same layout the row-swept producer writes, one block at a time:
        the row pointers land in their slice's continuous array, the data and
        indices at the stream position ``base`` names, the scale and inliers
        at the sub-slice's own place in the grid, and the block's position
        goes into the shared table for its consumer.  The ``_CSR_STORES``
        copies post ``store_sem`` unwaited; the caller owns the drain — the
        chained modes lag it to a later finalize (``_kernel`` /
        ``_async_kernel``, the rounds left over in ``forward``), the
        interleaved mode drains inline (``_EpilogueTail._sub_slice``).
        """
        g, bufs = self.out_geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[x] for x in range(nb)]
        m = idx[self.grid_m]
        # The CSR's K slice is sub-slice ``j`` of this GEMM's column tile.
        k = idx[self.grid_n] * self.n_sub + j

        d, i, p, s, q = self._store_tiles(self._tile_ordinal(idx))

        voyager.async_copy(
            p.reshape(ones + (1, g.rows + 1)),
            bufs.csr_indptr,
            bidx + [k, m],
            ones + (1, g.rows + 1),
            bufs.store_sem,
            None,
            [1] * nb + [1, g.rows],
        )

        last = _entry(p, g.rows, nb, ones)
        nnz = _scalar(last) - _scalar(_entry(p, 0, nb, ones))
        torch._check(nnz >= 0)
        base_ref = voyager.subview(
            bufs.stream_pos, bidx + [0], ones + (1,), (1,) * (nb + 1), []
        )
        at = _scalar(base_ref)
        torch._check(at >= 0)

        for src, dst in ((d, bufs.csr_data), (i, bufs.csr_indices)):
            voyager.async_copy(
                src,
                dst,
                bidx + [at],
                ones + (g.budget,),
                bufs.store_sem,
                None,
                [1] * (nb + 1),
                count=[1] * nb + [nnz],
            )

        for tile, dst, width in (
            (s, bufs.scale, g.tk // g.block_size),
            (q, bufs.inliers, g.tk),
        ):
            voyager.async_copy(
                tile,
                dst,
                bidx + [m, k],
                ones + (g.rows, width),
                bufs.store_sem,
                None,
                [1] * nb + [g.rows, width],
            )

        voyager.insert(
            base_ref.reshape(ones + (1, 1)).clone(),
            voyager.subview(
                bufs.out_base_table,
                bidx + [k, m],
                ones + (1, 1),
                (1,) * (nb + 2),
                [],
            ),
        )
        voyager.insert(base_ref + nnz, base_ref)
        voyager.insert(
            last.clone(),
            voyager.subview(
                bufs.slice_nnz, bidx + [k], ones + (1,), (1,) * (nb + 1), []
            ),
        )

    def _kernel(self, idx, *slots):
        if self.geom is not None:
            self._gather_rolled(idx, slots)
        tail = self.plan.fused_gm
        if self.out_geom is not None:
            # The scheduler passes ``*in_slots, *out_slots, *scratch``, and
            # this nest declares no scheduler-managed output.
            scratch = slots[len(self.plan.in_specs) :]
            tail = _EpilogueTail(self, idx, scratch[0] if scratch else None)
        if self.chain_epilogue:
            # The previous finalize's store DMAs drained under the rounds
            # between; consume their posts before this finalize's pass
            # reuses the staging tiles.  The first finalize has none in
            # flight; the last round's stores drain in ``forward``.
            effect_cond(
                (idx[self.grid_k] == self.plan.num_k - 1)
                & (self._tile_ordinal(idx) >= self.store_slots),
                self._drain_csr_stores,
            )
        self._scratch_and_kernel(tail)[1](idx, *slots)
        if self.chain_epilogue:
            # The chained pass staged this tile's CSR; its DMAs and
            # stream bookkeeping run bare, on the round that completed
            # the tile.
            effect_cond(
                idx[self.grid_k] == self.plan.num_k - 1,
                lambda: self._store_staged_csr(idx, 0),
            )

    def _async_kernel(
        self, idx, in_slots, out_slots, scratch, in_sems, out_sems, post
    ):
        """The async step: gather, then dispatch the dense kernel's commit.

        The pointer tile is consumed on the control stream — its values drive
        the gather DMAs' addresses at issue time — so its load semaphore is
        waited here and withheld from the commit, which instead depends on
        the dense loads plus this step's gather copies.  The gather-slot
        views ride the dependency-carrying operand list into the commit (see
        ``_scratch_and_kernel``).
        """
        # The scheduler hands the delinearized grid coordinate; the gather
        # slot ping-pongs with the flat step, like the compute-done slot.
        step = idx[0]
        for extent, coord in zip(self.plan.grid[1:], idx[1:]):
            step = step * extent + coord
        parity = step % _DONE_DEPTH
        if self.chain_epilogue:
            # Store the CSR a finalize commit staged two steps back: the
            # scheduler's lagged retire waited that commit's post last
            # step, so its results are readable without a wait of our
            # own.  Runs before this step's ``_offset_at`` read, which
            # depends on the bookkeeping.  The loop's final tile has no
            # step-two-later and is stored in ``forward``.
            prev2 = voyager.delinearize_index(step - 2, list(self.plan.grid))
            effect_cond(
                (step >= 2) & (prev2[self.grid_k] == self.plan.num_k - 1),
                lambda: self._store_staged_csr(prev2, 0),
            )
        deps, tiles = list(in_sems), list(in_slots)
        if self.geom is not None:
            voyager.async_wait(deps.pop(self.ptr_sem_index))
            deps += self._gather_rolled(idx, in_slots, parity)
            tiles.append(get_slot(self._bufs.gather_data, parity))
            tiles.append(get_slot(self._bufs.gather_indices, parity))
        tail = self.plan.fused_gm
        if self.out_geom is not None:
            tail = _EpilogueTail(self, idx, scratch[0] if scratch else None)
        if self.chain_epilogue:
            bufs = self._bufs
            tiles += list(self._store_tiles(self._tile_ordinal(idx))) + [
                self._offset_at(idx, 0)
            ]
            # Consume the previous finalize's store posts before this
            # step's finalize commit is enqueued: the pass reuses the
            # staging tiles as its destinations, and the commit runs
            # only after its dispatch, so enqueue-after-drain orders the
            # overwrite behind the DMAs.  By now those DMAs have had the
            # reduction rounds between to drain, so the waits are free.
            effect_cond(
                (idx[self.grid_k] == self.plan.num_k - 1)
                & (self._tile_ordinal(idx) >= self.store_slots),
                self._drain_csr_stores,
            )
        self._scratch_and_kernel(tail)[1](
            idx, tiles, out_slots, scratch, deps, out_sems, post
        )

    # --- entry --------------------------------------------------------------

    def forward(self, *inputs):
        """Allocate every buffer this nest owns, then run its loop.

        The SRAM allocations below are ORDER-SENSITIVE. ``build_sparse_gemm``
        stamps their bank groups positionally, by zipping them against its
        own ``hand_groups`` list, so adding, removing or reordering one
        here silently misassigns banks unless that list is changed to
        match. Semaphores and DRAM allocations are skipped and may move
        freely.
        """
        bufs = _Bufs()
        g = self.geom
        if g is not None:
            num_slots = _DONE_DEPTH if self.async_pipeline else UNPIPELINED
            # Names the producer's table rather than adding one:
            # ``merge_base_tables`` collapses the group onto its alloc.
            bufs.base_table_dram = voyager.alloc(
                base_table_shape(g), torch.int32
            )
            bufs.gather_data = voyager.alloc(
                list(g.ones) + [self.span],
                self.data_dtype,
                _SRAM,
                num_slots,
            )
            bufs.gather_indices = voyager.alloc(
                list(g.ones) + [self.span], torch.int32, _SRAM, num_slots
            )
            bufs.base_table = voyager.alloc(
                base_table_shape(g), torch.int32, _SRAM
            )
            bufs.gather_sem = voyager.zeros(
                [2 * self.R] if self.async_pipeline else [],
                torch.int64,
                num_slots,
            )
            bufs.table_sem = voyager.zeros([], torch.int64)

        og = self.out_geom
        if og is not None:
            vals = self.out_values
            bufs.csr_data = voyager.alloc(
                list(og.batch) + [og.stream], vals[0].dtype
            )
            bufs.csr_indices = voyager.alloc(
                list(og.batch) + [og.stream], torch.int32
            )
            bufs.csr_indptr = voyager.alloc(
                list(og.batch) + [og.n_k, og.M + 1], torch.int32
            )
            bufs.scale = voyager.alloc(
                list(og.batch) + [og.M, og.K // og.block_size], vals[3].dtype
            )
            bufs.inliers = voyager.alloc(
                list(og.batch) + [og.M, og.K], vals[4].dtype
            )
            bufs.out_base_table_dram = voyager.alloc(
                base_table_shape(og), torch.int32
            )
            bufs.slice_nnz = voyager.alloc(
                list(og.batch) + [og.n_k], torch.int32, _SRAM
            )
            bufs.stream_pos = voyager.alloc(
                list(og.batch) + [1], torch.int32, _SRAM
            )
            bufs.out_base_table = voyager.alloc(
                base_table_shape(og), torch.int32, _SRAM
            )
            bufs.store_sem = voyager.zeros([], torch.int64)
            bufs.out_table_sem = voyager.zeros([], torch.int64)
            # Per-sub-slice staging: a compute result reaches DRAM through
            # a scratchpad tile, never straight from the op.
            slots = self.store_slots if self.store_slots > 1 else UNPIPELINED
            bufs.store_data_tile = voyager.alloc(
                list(og.ones) + [og.budget], vals[0].dtype, _SRAM, slots
            )
            bufs.store_index_tile = voyager.alloc(
                list(og.ones) + [og.budget], torch.int32, _SRAM, slots
            )
            bufs.store_indptr_tile = voyager.alloc(
                list(og.ones) + [og.rows + 1], torch.int32, _SRAM, slots
            )
            bufs.store_scale_tile = voyager.alloc(
                list(og.ones) + [og.rows, og.tk // og.block_size],
                vals[3].dtype,
                _SRAM,
                slots,
            )
            bufs.store_inlier_tile = voyager.alloc(
                list(og.ones) + [og.rows, og.tk], vals[4].dtype, _SRAM, slots
            )
        self._bufs = bufs
        if g is not None:
            # The gather reads the table's scalars to address its copies, so
            # the whole table lands on chip before the loop runs.
            copy_base_table(
                bufs.base_table_dram,
                bufs.base_table,
                base_table_shape(g),
                bufs.table_sem,
            )
        dense = self.inner(*inputs)
        if og is None:
            return dense
        if self.chain_epilogue and self.async_pipeline:
            # A tile finalizing within ``_STORE_LAG`` steps of the end has
            # no step-two-later to store it, so it stores here.  That is
            # the last tile alone while finalizes are ``num_k`` steps
            # apart, and ``store_slots`` of them when every step finalizes
            # -- the same count the rotation holds, and never more tiles
            # than the sweep has.  The scheduler's own epilogue has
            # already waited the final commit's post.
            owed = min(self.tile_count, self.store_slots)
            for tile in range(self.tile_count - owed, self.tile_count):
                self._store_staged_csr(self._tile_coord(tile), 0)
        if self.chain_epilogue:
            # Every store the loop did not drain is still in flight, and
            # the consumer's nest reads the CSR next.  The loop drains one
            # round per finalize past the first ``store_slots``, leaving
            # exactly the rounds counted above.
            for _ in range(min(self.tile_count, self.store_slots)):
                self._drain_csr_stores()
        copy_base_table(
            bufs.out_base_table,
            bufs.out_base_table_dram,
            base_table_shape(og),
            bufs.out_table_sem,
        )
        return (
            bufs.csr_data,
            bufs.csr_indices,
            bufs.csr_indptr,
            bufs.scale,
            bufs.inliers,
        )


def gemm_produces_csr(node) -> bool:
    """Whether this GEMM's fused tail ends in a ``quantize_mx_outlier``.

    Such a group is a producer as well as (possibly) a consumer: the quantize
    runs on the accumulator tile, so its CSR is emitted per output tile and has
    to be stored by hand rather than diced by the output grid. A group whose
    *anchor* is the quantize is not one -- that is the row-swept case.
    """
    submod = node.meta.get("submodule")
    if not isinstance(submod, torch.fx.GraphModule):
        return False
    anchor = get_anchor_node(node)
    if anchor is None or not is_gemm_op(anchor):
        return False
    if not any(
        n.op == "call_function" and n.target is _QUANTIZE_MX_OUTLIER
        for n in submod.graph.nodes
    ):
        return False
    return isinstance(node.value, (list, tuple)) and len(node.value) == 5


def _epilogue_geometry(node, plan, tiler) -> _Geometry:
    """The CSR geometry a GEMM epilogue produces.

    The quantize sees the GEMM's output tile, so the row block *is* the row
    tile.  The K slice, though, is the *consumers'* k tile: the tail dices
    each column tile into ``tk``-wide sub-slices (``_EpilogueTail``), so the
    slice width follows the consumers — never coarser than a consumer's own
    search validated — while this GEMM's tiling stays its own.  The gcd with
    ``tile_n`` keeps a slice inside one column tile; forcing *finer* than a
    consumer asked only shrinks its tiles, which is always safe.
    """
    vals = node.value
    inliers = vals[4]
    batch = tuple(inliers.shape[:-2])
    M, K = inliers.shape[-2], inliers.shape[-1]
    submod = node.meta["submodule"]
    qnode = next(
        n
        for n in submod.graph.nodes
        if n.op == "call_function" and n.target is _QUANTIZE_MX_OUTLIER
    )
    bs = qnode.args[3]
    max_pct = qnode.args[9] if len(qnode.args) > 9 else 0.01
    tk_pref, _ = consumer_k_tile(node, tiler)
    tk = plan.tile_n if tk_pref is None else math.gcd(tk_pref, plan.tile_n)
    if tk % bs:
        # A slice must hold whole microscaling blocks; fall back to the
        # column tile (the consumer's own guard reports the mismatch).
        tk = plan.tile_n
    geom = _Geometry(
        batch=batch,
        M=M,
        K=K,
        rows=plan.tile_m,
        tk=tk,
        block_size=bs,
        budget=int(plan.tile_m * tk * max_pct),
        max_pct=max_pct,
    )
    # The tile search has already run, so the submodule carries the GEMM's
    # real output -- what the quantize will see, and the only chance to size
    # a block that does not fit before ``_pad_csr`` silently drops its tail.
    return _fit_block_budget(
        getattr(qnode.args[0], "value", None),
        qnode.args[8] if len(qnode.args) > 8 else None,
        geom,
    )


def _retarget_csr_views(node) -> None:
    """Let a CSR reach a lower-rank consumer without losing its slicing.

    A reshape can sit between producer and consumer — the boundary between a
    decoder layer and ``lm_head`` drops the leading batch dim — and it carries
    no meta of its own. Dropping the batch dim is what the consumer wants;
    flattening the K-slice axis away with it is not, since the pointer array
    holds one array per slice. So the pointer view is retargeted to keep that
    axis, and every CSR view inherits the producer's geometry.
    """
    for arg in list(node.args) + list(node.kwargs.values()):
        if not isinstance(arg, torch.fx.Node) or not is_nop(arg):
            continue
        src = arg
        while is_nop(src) and src.all_input_nodes:
            src = src.all_input_nodes[0]
        geom = src.meta.get(GEOMETRY_META)
        if geom is None:
            continue
        arg.meta[GEOMETRY_META] = geom
        arg.meta[PRODUCER_META] = src.meta.get(PRODUCER_META)
        value = getattr(arg, "value", None)
        if (
            value is None
            or value.dtype != torch.int32
            or value.shape[-1] != geom.M + 1
            or value.ndim >= src.value.ndim
        ):
            continue
        shape = list(value.shape[:-1]) + [geom.n_k, value.shape[-1]]
        arg.args = (arg.args[0], shape)
        set_node_value(arg, src.value.reshape(shape))


def _csr_geometry(node):
    """This GEMM's ``A_indptr`` operand and the geometry its producer stamped.

    Identified by shape rather than by the anchor's kwargs, which on a fused
    group name submodule placeholders that carry no meta.

    Returns:
        ``(operand, geometry)``, or ``(None, None)``.
    """
    for operand in node.all_input_nodes:
        geom = operand.meta.get(GEOMETRY_META)
        value = getattr(operand, "value", None)
        if geom is None or not isinstance(value, torch.Tensor):
            continue
        if value.dtype == torch.int32 and value.shape[-1] == geom.M + 1:
            # A reshape on the way in may have dropped batch dims the producer
            # had; the consumer indexes only the ones it still sees.
            batch = tuple(value.shape[:-2])
            if batch != geom.batch:
                geom = replace(
                    geom,
                    batch=batch,
                    dropped=len(geom.batch) - len(batch),
                )
            return operand, geom
    return None, None


@contextlib.contextmanager
def _slice_axis_hidden(node):
    """Hide the K-slice axis of this GEMM's ``A_indptr`` operand.

    The producer's buffer is ``[*batch, n_k, M+1]`` -- one cumulative pointer
    array per K slice -- but a single ``linear_mx`` consumes exactly one slice,
    so what the op is handed is ``[*batch, M+1]``. Shape derivation runs the
    submodule on the operand *values*, and ``spmm_csr`` reads every leading dim
    as batch, so an extra one there gives the GEMM's output a rank it does not
    have.

    Only derivation sees the shorter value: the fetch still addresses the whole
    buffer, picking its slice off the reduction grid dim, so the axis is put
    back on the way out.

    Yields:
        The ``(node, full_value)`` pair that was hidden, or ``None``.
    """
    hidden = None
    for operand in node.all_input_nodes:
        geom = operand.meta.get(GEOMETRY_META)
        value = getattr(operand, "value", None)
        if geom is None or not isinstance(value, torch.Tensor):
            continue
        if (
            value.dtype == torch.int32
            and value.ndim == geom.nb + 2
            and value.shape[-1] == geom.M + 1
        ):
            hidden = (operand, value)
            operand.value = value[..., 0, :]
            operand.shape = tuple(operand.value.shape)
            break
    try:
        yield hidden
    finally:
        if hidden is not None:
            operand, value = hidden
            operand.value = value
            operand.shape = tuple(value.shape)


def build_sparse_gemm(
    node,
    *,
    num_slots: int = _DEFAULT_NUM_SLOTS,
    accumulate_fp32: bool = False,
    tiler=None,
    async_pipeline: bool = False,
) -> Optional[torch.fx.GraphModule]:
    """Bufferize a GEMM whose activation carries an outlier CSR.

    Args:
        node: The sparse ``linear_mx`` / ``matmul_mx``, bare or fused.
        num_slots: Software-pipeline depth for the dense operands.
        accumulate_fp32: Accumulate the K reduction in fp32.
        tiler: The interstellar ``TilerContext``.
        async_pipeline: Dispatch each step's compute as a ``voyager.commit``
            (the :class:`AsyncPipelinedKernel` nest), overlapping the CSR
            gather and the dense loads with the previous tile's compute;
            the synchronous nest otherwise.

    Returns:
        The bufferized gm, or ``None`` when the node has no CSR or the
        producer's geometry never reached it.
    """
    produces = gemm_produces_csr(node)
    # The CSR's geometry is fixed by its producer, and the GEMM has to split K
    # the same way, so read it before deriving anything and hand the derivation
    # that slice count instead of the tile search's.
    _retarget_csr_views(node)
    ptr, geom = _csr_geometry(node)
    if geom is None and not produces:
        return None

    with _slice_axis_hidden(node) as hidden:
        plan = _gemm_plan(
            node,
            tiler,
            k_tiles=geom.n_k if geom is not None else None,
        )
    if plan is None:
        return None
    operands = list(node.all_input_nodes)
    if hidden is not None:
        # Derivation ran against one slice; the nest fetches from the whole
        # buffer, so hand the loop the operand it will actually address.
        inputs = list(plan.inputs)
        inputs[plan.kw_idx["A_indptr"]] = hidden[1].clone()
        plan.inputs = tuple(inputs)

    data_dtype = None
    if geom is not None:
        if plan.tile_k != geom.tk:
            raise ValueError(
                f"{node.name}: k tile {plan.tile_k} does not match the CSR's "
                f"slice width {geom.tk}; column indices are local to a slice "
                f"and the engine cannot rebase them"
            )
        if plan.tile_m % geom.rows:
            raise ValueError(
                f"{node.name}: row tile {plan.tile_m} is not a whole number "
                f"of the producer's {geom.rows}-row blocks"
            )
        data_dtype = operands[plan.kw_idx["A_data"]].value.dtype

    out_geom = _epilogue_geometry(node, plan, tiler) if produces else None
    pattern = _SparseGemm(
        plan,
        geom,
        data_dtype,
        out_geom=out_geom,
        accumulate_fusible=node.meta.get("accumulate_fusible", False),
        num_slots=num_slots,
        accumulate_fp32=accumulate_fp32,
        async_pipeline=async_pipeline,
    )
    if out_geom is not None:
        pattern.out_values = list(node.value)
    with _lenient_verifier():
        gm = export_model(pattern, plan.inputs)
    gm = _finalize_exported_gm(gm)
    # The scheduler's allocs take the same bank groups a dense GEMM's would;
    # ``forward``'s hand allocs precede them in allocation order.  Every
    # buffer that addresses the CSR joins the ``"csr"`` bank the search
    # charged: the gather staging pair, the pointer tile (a scheduler slot,
    # grouped by role) and the base-table tile whose scalars place the
    # gather's copies -- all of them this nest's own, and all read on the
    # same serialized fetch chain.  The CSR this nest *emits* gets a bank of
    # its own (``"csr_out"``): its staging is compute-written and DMA-read
    # while the gather's is the reverse, and a nest that both consumes and
    # produces has both live in one step.
    roles = plan.anchor.meta.get("tiling", {}).get("bank_groups")
    if roles is not None:

        def group_of(role):
            return next(
                (i for i, role_set in enumerate(roles) if role in role_set),
                None,
            )

        # One entry per SRAM alloc ``_SparseGemm.forward`` makes, in its
        # order -- keep the two in step.
        hand_groups = []
        if geom is not None:
            # gather_data, gather_indices, base_table
            hand_groups += [group_of("csr")] * 3
        if out_geom is not None:
            # slice_nnz, stream_pos, out_base_table and the five
            # store-staging tiles
            hand_groups += [group_of("csr_out")] * 8
        _stamp_bank_groups(
            gm,
            hand_groups
            + _bank_group_list(
                node, plan.in_specs, plan.out_specs, pattern.scratch_specs
            ),
        )
    # The rolled gather nests a second ``while_loop`` inside the step,
    # over the ``R`` row blocks it concatenates.
    extents = [[(0, pattern.inner.num_steps, 1)]]
    if geom is not None:
        extents.append([(0, pattern.R, 1)])
    _tag_loop_extents(gm, extents)
    if geom is not None:
        tag_base_table(gm, ptr.meta[PRODUCER_META], base_table_shape(geom))
    if out_geom is not None:
        tag_base_table(gm, node.name, base_table_shape(out_geom))
        gm.meta[GEOMETRY_META] = out_geom
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*plan.inputs)
    if plan.outline_dps:
        outline_dps_ops(gm)
    _stamp_anchor_meta(gm, plan.anchor)
    return gm
