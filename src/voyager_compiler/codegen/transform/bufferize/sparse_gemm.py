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
    _DEFAULT_NUM_SLOTS,
    _DONE_DEPTH,
    _gemm_plan,
    _gemm_scratch_and_kernel,
    _stamp_anchor_meta,
    AsyncPipelinedKernel,
    get_slot,
    PipelinedKernel,
)
from voyager_compiler.codegen.transform.bufferize.quantize_mx_outlier import (
    _Bufs,
    _check_block_budget,
    _entry,
    _Geometry,
    _QUANTIZE_MX_OUTLIER,
    _scalar,
    _split_prefix,
    base_table_shape,
    consumer_k_tile,
    GEOMETRY_META,
    PRODUCER_META,
    tag_base_table,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    outline_dps_ops,
    voyager,
)
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import set_node_value, ShapeProp

_SRAM = int(MemoryLevel.SRAM)


class _EpilogueTail(torch.nn.Module):
    """The fused tail of a CSR-producing GEMM, plus its CSR stores.

    Runs the tail's prefix once on the whole accumulator tile, then the
    quantize once per ``tk``-wide sub-slice of it, so the CSR comes out at
    the *consumers'* slice width while this GEMM's own tiling stays whatever
    its search picked.  Wrapping rather than threading extra arguments keeps
    the tail's arity: the reduction kernel applies it inside a ``torch.cond``
    on the last reduction step and calls it with the group's own operands.
    The offset and the grid index it needs are captured here instead, which
    is also what puts the stores on the step that actually completes an
    output tile.
    """

    def __init__(self, owner, idx):
        super().__init__()
        self.owner = owner
        self.idx = idx

    def forward(self, acc, *operands):
        o = self.owner
        og = o.out_geom
        bufs = o._bufs
        nb, ones = og.nb, og.ones
        bound = (acc,) + operands

        def val(t):
            kind, v = t
            return bound[v] if kind == "arg" else v

        # A value shared by several sub-slices must cross a *buffer*: DPS
        # outlining grows one op per store, and ops sharing an SSA edge
        # cannot come out disjoint.  So with several sub-slices the prefix
        # result lands in a staging buffer the slices window, and each
        # slice's scale / inlier tiles land in a full-tile staging buffer
        # the kernel then moves into the output slot.
        if o.epi_prefix is None:
            x = acc
        elif o.n_sub == 1:
            x = o.epi_prefix(*bound)
        else:
            voyager.insert(o.epi_prefix(*bound), bufs.epi_mid)
            x = None

        for j in range(o.n_sub):
            if x is None:
                tile = voyager.subview(
                    bufs.epi_mid,
                    [0] * nb + [0, j * og.tk],
                    ones + (og.rows, og.tk),
                    (1,) * (nb + 2),
                    [],
                )
            elif o.n_sub > 1:
                tile = x[..., j * og.tk : (j + 1) * og.tk]
            else:
                tile = x
            d, i, p, s, q = _QUANTIZE_MX_OUTLIER(
                tile,
                *[val(t) for t in o.epi_q_args],
                **{k: val(t) for k, t in o.epi_q_kwargs.items()},
                indptr_offset=o._offset_at(self.idx, j),
            )
            o._store_csr(self.idx, j, d, i, p)
            if o.n_sub == 1:
                return s, q
            sb = og.tk // og.block_size
            voyager.insert(
                s,
                voyager.subview(
                    bufs.epi_scale_tile,
                    [0] * nb + [0, j * sb],
                    ones + (og.rows, sb),
                    (1,) * (nb + 2),
                    [],
                ),
            )
            voyager.insert(
                q,
                voyager.subview(
                    bufs.epi_out_tile,
                    [0] * nb + [0, j * og.tk],
                    ones + (og.rows, og.tk),
                    (1,) * (nb + 2),
                    [],
                ),
            )
        return bufs.epi_scale_tile, bufs.epi_out_tile


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
            # The CSR outputs are stored by hand; only the dense pair is diced
            # by the output grid.
            plan.out_specs = list(plan.out_specs)[3:]
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
        # The CSR-store epilogue runs bare in the loop body after the
        # commit — its DMAs, waits and stream bookkeeping cannot live in a
        # commit subgraph — which requires a single-slot scratch
        # (``_reduction_fused_kernel``'s bare-tail path) and a reduction to
        # attach it to: a ``num_k == 1`` producer keeps the synchronous
        # nest.
        if out_geom is not None and async_pipeline:
            if plan.num_k == 1:
                async_pipeline = False
            else:
                plan.anchor.meta.setdefault("tiling", {})["scratch_slots"] = 1
        self.async_pipeline = async_pipeline
        if geom is not None:
            # ``in_sems`` carries one semaphore per *tiled* operand — a
            # ``None``-spec operand is passed through without one — so the
            # pointer semaphore is indexed by rank among the real specs.
            self.ptr_sem_index = sum(
                1
                for s in plan.in_specs[: plan.kw_idx["A_indptr"]]
                if s is not None
            )
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

        return _gemm_scratch_and_kernel(
            gemm_kernel,
            tail,
            reduction_dim=self.grid_k,
            num_k=plan.num_k,
            acc_shape=plan.acc_shape,
            in_specs=plan.in_specs,
            out_specs=plan.out_specs,
            fused_idx=plan.fused_idx,
            anchor=plan.anchor,
            accumulate_fp32=self.accumulate_fp32,
            # The CSR epilogue always re-reads the staged accumulator; a
            # chained tail would trace its stores into the commit.
            chain_tail=False,
            async_pipeline=self.async_pipeline,
        )

    # --- the gather ---------------------------------------------------------

    def _gather(self, idx, slots, parity=None):
        """Concatenate this row tile's ``R`` blocks into one CSR.

        Source addresses come from the resident base table (a plain read, not a
        fetch); destinations and lengths come from the pointer tile that was
        loaded with the dense operands, so no arithmetic beyond reading scalars.

        Args:
            idx: The grid coordinate.
            slots: The operand tiles, positionally.
            parity: The gather slot this step writes (async mode): the copies
                land in slot ``parity`` of the two-slot staging buffers and
                each signals its own semaphore slot, returned for the compute
                commit's dependency list.  ``None`` is the synchronous mode —
                single-slot buffers, every copy waited inline.

        Returns:
            The semaphore-slot views the copies signal (async), else ``[]``.
        """
        plan, g, bufs = self.plan, self.geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[i] for i in range(nb)]
        m_c, k = idx[self.grid_m], idx[self.grid_k]

        # The loaded pointer tile is a slot window, so an entry read off it
        # would window a window (see ``_scalar``); stage it first.
        ptr = slots[plan.kw_idx["A_indptr"]]
        voyager.insert(
            ptr.reshape(ones + (plan.tile_m + 1,)).clone(), bufs.gather_ptr_tile
        )
        p = bufs.gather_ptr_tile
        lo = _scalar(_entry(p, 0, nb, ones))

        sems = []
        for j in range(self.R):
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

            for b, (src, dst) in enumerate(
                (
                    (slots[plan.kw_idx["A_data"]], bufs.gather_data),
                    (slots[plan.kw_idx["A_indices"]], bufs.gather_indices),
                )
            ):
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
                    # The slot offset composes into one subview: a window of
                    # a ``get_slot`` window could not fold into a single
                    # ``TensorBoxRef``.
                    window = voyager.subview(
                        dst,
                        [parity] + [0] * nb + [start],
                        [1] + list(ones) + [g.budget],
                        (1,) * (nb + 2),
                        [0],
                    )
                    sem = get_slot(
                        bufs.gather_sem, parity * 2 * self.R + 2 * j + b
                    )
                    sems.append(sem)
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
        return sems

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

    def _store_csr(self, idx, j, d, i, p):
        """Append one sub-slice's CSR block to the packed stream.

        The same layout the row-swept producer writes, one block at a time:
        the row pointers land in their slice's continuous array, the data and
        indices at the stream position ``base`` names, and the block's position
        goes into the shared table for its consumer.
        """
        g, bufs = self.out_geom, self._bufs
        nb, ones = g.nb, g.ones
        bidx = [idx[x] for x in range(nb)]
        m = idx[self.grid_m]
        # The CSR's K slice is sub-slice ``j`` of this GEMM's column tile.
        k = idx[self.grid_n] * self.n_sub + j

        # A compute result reaches DRAM through a scratchpad tile (see
        # ``_OutlierProducer._slice_loop``); stage the op's results first.
        voyager.insert(d, bufs.store_data_tile)
        voyager.insert(i, bufs.store_index_tile)
        voyager.insert(p, bufs.store_indptr_tile)
        d, i, p = (
            bufs.store_data_tile,
            bufs.store_index_tile,
            bufs.store_indptr_tile,
        )

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
            voyager.async_wait(bufs.store_sem)

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
            self._gather(idx, slots)
        tail = self.plan.fused_gm
        if self.out_geom is not None:
            tail = _EpilogueTail(self, idx)
        self._scratch_and_kernel(tail)[1](idx, *slots)

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
        deps, tiles = list(in_sems), list(in_slots)
        if self.geom is not None:
            voyager.async_wait(deps.pop(self.ptr_sem_index))
            deps += self._gather(idx, in_slots, parity)
            tiles.append(get_slot(self._bufs.gather_data, parity))
            tiles.append(get_slot(self._bufs.gather_indices, parity))
        tail = self.plan.fused_gm
        if self.out_geom is not None:
            tail = _EpilogueTail(self, idx)
        self._scratch_and_kernel(tail)[1](
            idx, tiles, out_slots, scratch, deps, out_sems, post
        )

    # --- entry --------------------------------------------------------------

    def forward(self, *inputs):
        bufs = _Bufs()
        g = self.geom
        if g is not None:
            num_slots = _DONE_DEPTH if self.async_pipeline else UNPIPELINED
            sem_slots = (
                _DONE_DEPTH * 2 * self.R if self.async_pipeline else UNPIPELINED
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
            bufs.gather_ptr_tile = voyager.alloc(
                list(g.ones) + [self.plan.tile_m + 1], torch.int32, _SRAM
            )
            bufs.gather_sem = voyager.zeros([], torch.int64, sem_slots)

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
            # Per-tile staging for ``_store_csr``'s stores.
            bufs.store_data_tile = voyager.alloc(
                list(og.ones) + [og.budget], vals[0].dtype, _SRAM
            )
            bufs.store_index_tile = voyager.alloc(
                list(og.ones) + [og.budget], torch.int32, _SRAM
            )
            bufs.store_indptr_tile = voyager.alloc(
                list(og.ones) + [og.rows + 1], torch.int32, _SRAM
            )
            # Sub-slice staging (see ``_EpilogueTail``): the shared prefix
            # result, and the reassembled scale / inlier column tiles.
            if self.epi_prefix is not None and self.n_sub > 1:
                bufs.epi_mid = voyager.alloc(
                    list(og.ones) + [og.rows, self.plan.tile_n],
                    self.plan.out_dtype,
                    _SRAM,
                )
            if self.n_sub > 1:
                bufs.epi_scale_tile = voyager.alloc(
                    list(og.ones)
                    + [og.rows, self.plan.tile_n // og.block_size],
                    vals[3].dtype,
                    _SRAM,
                )
                bufs.epi_out_tile = voyager.alloc(
                    list(og.ones) + [og.rows, self.plan.tile_n],
                    vals[4].dtype,
                    _SRAM,
                )
        self._bufs = bufs
        dense = self.inner(*inputs)
        if og is None:
            return dense
        scale, inliers = (dense if isinstance(dense, tuple) else (dense,))[:2]
        return (
            bufs.csr_data,
            bufs.csr_indices,
            bufs.csr_indptr,
            scale,
            inliers,
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
    # The tile search has already run, so the submodule carries the GEMM's real
    # output -- what the quantize will see, and the only chance to catch a
    # block that does not fit before ``_pad_csr`` silently drops its tail.
    _check_block_budget(
        getattr(qnode.args[0], "value", None),
        qnode.args[8] if len(qnode.args) > 8 else None,
        geom,
    )
    return geom


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
        num_slots=num_slots,
        accumulate_fp32=accumulate_fp32,
        async_pipeline=async_pipeline,
    )
    if out_geom is not None:
        pattern.out_values = list(node.value)
    with _lenient_verifier():
        gm = export_model(pattern, plan.inputs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(0, pattern.inner.num_steps, 1)]])
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
