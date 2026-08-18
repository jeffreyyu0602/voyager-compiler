"""Unified Pallas-``pallas_call``-style kernel scheduler.

One scheduler subsuming the pointwise / pooling / GEMM bufferization builders:
given a ``kernel``, a ``grid``, and per-operand ``_InputSpec`` / ``_OutputSpec``
block specs, it emits a single rolled ``while_loop`` over the flattened grid.

Spec-driven for tile addressing, mutate-style for compute (Pallas ``out_ref``
semantics): each grid step loads every tiled input's current block into its SRAM
slot, calls ``kernel(grid_index, *in_slots, *out_slots)`` which writes each
output SRAM slot, then stores each out slot to DRAM.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler.codegen.node_info import (
    _pair,
    ancestors,
    compute_output_tiled_shapes,
    get_anchor_node,
    get_arg_value,
    is_bmm,
    is_conv2d,
    is_nop,
    is_reshape_op,
    quant_param_arg_nodes,
    reduction_op,
    reduction_scratch,
    trailing_mha_perm,
    weight_is_ck,
    weight_transforms,
)
from voyager_compiler.codegen.subgraph import copy_graph_module
from voyager_compiler.codegen.transform.bufferize.ops import (
    MemoryLevel,
    commit,
    oracle_disabled,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _build_fused_gm,
    _compute_input_spec,
    _finalize_exported_gm,
    _InputSpec,
    _lenient_verifier,
    _OutputSpec,
    _ScratchSpec,
    _tag_loop_extents,
    effect_cond,
    outline_dps_ops,
    voyager,
)
from voyager_compiler.codegen.transform.tiling.search import (
    pool_op_tiling,
    vector_op_tiling,
)
from voyager_compiler.codegen.transform.tiling.tiler import (
    CONV_L3_ORDER,
    GEMM_L3_ORDER,
    get_tiling,
)
from voyager_compiler.export_utils import export_model
from voyager_compiler.ops.layout import (
    NCHW_TO_NHWC,
    OIHW_TO_HWIO,
    project,
    unproject,
)
from voyager_compiler.shape_prop import ShapeProp

_SRAM = int(MemoryLevel.SRAM)

# A microscaling quantize fuses onto the *end* of an MHA output relayout, so it
# is what the body returns and the permute sits one step above it.
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default

# Default software-pipeline depth (2 = double buffering).  Single source of
# truth for the ``num_slots`` default across the scheduler and op builders; a
# spec may override it per operand (``_InputSpec`` / ``_OutputSpec.num_slots``).
_DEFAULT_NUM_SLOTS = 2


def spec_tiled_dims(spec, grid):
    """``(d, g, r)`` for each operand dim of ``spec`` that is dynamically
    indexed: it maps to a tiled grid dim ``g`` (``grid > 1``) and is not
    broadcast.  Whole / broadcast / ``None``-mapped dims stay at block 0 and
    are left out.  ``r`` is how many consecutive grid steps read the *same*
    block — 1 unless the operand repeats over ``g`` (a GQA head, say).
    """
    bcast = getattr(spec, "is_broadcast", None)
    rep = getattr(spec, "repeat", None)
    for d, g in enumerate(spec.index_map):
        if (
            g is not None
            and grid[g] > 1
            and not (bcast is not None and bcast[d])
        ):
            yield d, g, (rep[d] if rep is not None else 1)


@dataclass
class _Window:
    """The ``num_slots``-dependent slice of a grid step's context: the current
    read slot and the depth-``D`` prefetch window (``D = num_slots - 1``).  One
    per distinct buffer count in use, so operands sharing a depth share these
    nodes (the uniform case emits exactly one window — an unchanged graph).
    """

    cur_slot: object
    fetch_idx: object
    prev_edge: object
    has_fetch: object
    first: object


@dataclass
class _StepCtx:
    """Per-grid-step values computed once in ``body_fn`` and shared by every
    operand's scheduler.  The count-independent indices (``cur`` / ``next`` /
    ``prev`` / ``last``) are shared directly; the count-dependent slot and
    prefetch window live in ``windows`` keyed by buffer count, so a reader takes
    ``windows[self.num_slots]`` for its own depth.  Nothing is recomputed inside
    a scheduler — that would duplicate the traced ``delinearize_index`` nodes.
    """

    step: object
    cur: object
    next: object
    prev: object
    last: object
    windows: dict  # num_slots -> _Window


def get_slot(buf, slot):
    """One slot of a pipelined buffer (``[num_slots, *tile]``), as an explicit
    ``voyager.subview``: offset ``slot`` along the slot dim, the whole tile
    along the rest, and the slot dim dropped — it is not a tensor dim.  ``slot``
    may be a runtime value (``step % num_slots``).

    Said with ``buf[slot]`` this would be an ``aten.select``, indistinguishable
    from a model slicing a tensor — and the two mean opposite things: a slot
    pick renames storage (it folds into the operand's ``TensorBoxRef``, as the
    window that reference makes), while a slice reads bytes of its own.
    """
    shape = list(buf.shape)
    offsets = [slot] + [0] * (len(shape) - 1)
    sizes = [1] + shape[1:]
    strides = [1] * len(shape)
    return voyager.subview(buf, offsets, sizes, strides, squeeze_dim=[0])


def _guarded_wait(sem, pred=None):
    """``async_wait(sem)`` guarded by ``pred``, so each slot's semaphore is
    waited exactly once per signaling copy (a counting semaphore underflows on a
    stray wait).  ``pred=None`` waits unconditionally — an operand whose block
    changes every step is already once-per-block.
    """
    if pred is None:
        voyager.async_wait(sem)
        return
    effect_cond(pred, lambda: voyager.async_wait(sem))


class _BufferedRef:
    """Per-operand software-pipeline scheduler (Pallas ``BufferedRef``).

    Owns one tiled input's or one output's *window* — its ``num_slots``-deep
    SRAM ``slots`` and per-slot DMA ``sem`` (Pallas's ``window_ref`` +
    ``sem_recvs`` / ``sem_sends``) — plus the state machine that drives it: slot
    selection, copy / wait predicates, prologue priming, and producer /
    consumer / store cursor advancement.

    It holds **no mutable runtime state**: each method takes the loop-carried
    counter as a plain argument and returns the updated SymInt (the live cursors
    live in the ``while_loop`` operands), so the scheduler stays exportable.
    """

    _IN = "in"
    _OUT = "out"

    def __init__(
        self, kind, spec, grid, num_slots, slots, async_pipeline=False
    ):
        self.kind = kind
        self.spec = spec
        self.grid = grid
        self.ndim = len(grid)
        self.num_steps = math.prod(grid)
        self.num_slots = num_slots
        # Prefetch distance (blocks ahead).
        self.D = max(0, num_slots - 2) if async_pipeline else num_slots - 1
        self.slots = slots  # SRAM window: [num_slots, *tile_sizes]
        # Per-slot async-DMA semaphore ([num_slots] int64).  An async
        # output store-sem starts with one credit per slot (``fill(1)``) so each
        # slot's first use has a token to consume — the commit always waits the
        # slot free, no warm-up guard needed.
        if kind == self._OUT and async_pipeline:
            self.sem = voyager.fill([], torch.int64, 1, num_slots=num_slots)
        else:
            self.sem = voyager.zeros([], torch.int64, num_slots=num_slots)

    # --- addressing (shared by both kinds) ----------------------------------

    def _block(self, coord, r):
        """The block index a grid coord addresses: every ``r``-th step advances
        it, so a repeated operand re-reads one tile ``r`` times."""
        return coord if r == 1 else coord // r

    def _block_address(self, grid_idx):
        """The ``(dims, indices)`` addressing this operand's tile for
        ``async_copy`` at ``grid_idx``.  ``dims`` is ``None`` when every dim is
        dynamic (``async_copy``'s "all dims" shorthand).
        """
        dims, indices = [], []
        for d, g, r in spec_tiled_dims(self.spec, self.grid):
            dims.append(d)
            indices.append(self._block(grid_idx[g], r))
        if len(dims) < len(self.spec.index_map):
            return dims, indices
        return None, indices

    def _indices_differ(self, cur, next):
        """Whether this operand's tile block changes between grid points ``cur``
        and ``next`` — the load / store change predicate.

        A chained ``|`` of ``SymBool``s over the tiled (non-broadcast) dims — no
        Python ``any`` short-circuit (would data-dependent-guard inside the
        traced loop), no mixed-radix arithmetic.  Seeded with the first term
        (not ``False``) to avoid a redundant ``False | ...`` node.
        """
        differ = None
        for _, g, r in spec_tiled_dims(self.spec, self.grid):
            term = self._block(cur[g], r) != self._block(next[g], r)
            differ = term if differ is None else (differ | term)
        return False if differ is None else differ

    def _innermost_tiled_dim(self):
        """The fastest-varying tiled grid dim (last dim with extent > 1, whose
        coord advances every step in row-major order); ``None`` if none.
        """
        tiled = [g for g in range(self.ndim) if self.grid[g] > 1]
        return tiled[-1] if tiled else None

    def _single_block(self):
        """Whether the whole sweep reads one block of this operand — every
        tiled dim's block count (``ceil(grid[g] / r)``) collapses to one.
        The same test ``PipelinedKernel._num_slots`` uses to give the buffer
        one slot, so the predicate and the allocation cannot disagree.
        """
        return all(
            self.grid[g] <= r
            for _, g, r in spec_tiled_dims(self.spec, self.grid)
        )

    def _advances_every_step(self):
        """Whether this operand's tile block changes on every grid step (it
        spans the innermost tiled, non-broadcast dim).  Known at build time:
        when True the DMA is emitted unconditionally — no ``torch.cond`` guard,
        no counter ``sym_ite``.  A dim it *repeats* over advances only every
        ``r``-th step, so it does not qualify.
        """
        inner = self._innermost_tiled_dim()
        if inner is None:
            return False
        return any(
            g == inner and r == 1
            for _, g, r in spec_tiled_dims(self.spec, self.grid)
        )

    def _unravel(self, flat):
        """The row-major grid coords of flat index ``flat`` as plain Python
        ints — the build-time counterpart of ``voyager.delinearize_index``, used
        for the static prologue positions so their block dedup is a Python
        ``if`` (no ``torch.cond``).
        """
        out = [0] * self.ndim
        for d in range(self.ndim - 1, -1, -1):
            flat, out[d] = divmod(flat, self.grid[d])
        return tuple(out)

    # --- DMA (input loads / output stores) ----------------------------------

    def _load_tile(self, src, dst, grid_idx, sem, post_count=1):
        """Async-DMA ``src``'s tile at ``grid_idx`` into SRAM ``dst``, carrying
        the input halo (``strides`` / ``pad`` / ``pad_value``) and signaling the
        load semaphore ``sem`` ``post_count`` times (a reused block posts once
        per per-step consumer — see ``AsyncPipelinedKernel``).
        """
        spec = self.spec
        dims, indices = self._block_address(grid_idx)
        sizes, strides = spec.tile_sizes, spec.strides
        if spec.transposed:
            # The spec is in matmul (Kᵀ) order but the DRAM buffer is its
            # (N, K) transpose; swap the fetch's last two dims so the DMA
            # slices it in its own order (``async_copy`` ``.mT``s it back).
            def _swap(seq):
                s = list(seq)
                s[-2], s[-1] = s[-1], s[-2]
                return s

            sizes = _swap(sizes)
            if strides is not None:
                strides = _swap(strides)
            if dims is None:
                indices = _swap(indices)
            else:
                a, b = len(spec.index_map) - 2, len(spec.index_map) - 1
                dims = [b if d == a else a if d == b else d for d in dims]
        voyager.async_copy(
            src,
            dst,
            indices,
            sizes,
            sem,
            dims=dims,
            strides=strides,
            transposed=spec.transposed,
            pad=spec.pad,
            pad_value=spec.pad_value,
            post_count=post_count,
        )

    def _store_tile(self, src, dst, grid_idx, sem):
        """Async-DMA SRAM tile ``src`` -> ``dst``'s block at ``grid_idx``,
        signaling ``sem``.
        """
        dims, indices = self._block_address(grid_idx)
        voyager.async_copy(
            src,
            dst,
            indices,
            self.spec.tile_sizes,
            sem,
            dims=dims,
        )

    # --- input phases (kind == _IN) -----------------------------------------

    def reuse_count(self):
        """Steps that consecutively read the same block: the innermost tiled grid
        dim's own repeat (``r`` consecutive steps re-read one block when the
        operand repeats over it, e.g. a GQA head) times the grid extents inner to
        it (1 when it advances every step).  A reused block's load posts this many
        times so a per-step ``commit`` consume balances a once-per-block load.
        """
        tiled = list(spec_tiled_dims(self.spec, self.grid))
        if not tiled:
            return self.num_steps
        inner = max(g for _, g, _ in tiled)
        count = next(r for _, g, r in tiled if g == inner)
        for g in range(inner + 1, self.ndim):
            count *= self.grid[g]
        return count

    def prime_prologue(self, src, post_count=1):
        """Prime the first ``D`` logical positions from DRAM ``src``,
        deduplicating reused blocks (positions are static concrete coords, so
        the dedup is a Python ``if``).  A single-block operand primes its one
        block even at ``D == 0`` — the slot is never overwritten, and priming
        here keeps the load off the loop body's critical path.  Each block's
        load signals ``post_count`` times.  Return the seed producer count
        for a guarded input, or ``None`` for an always-advance input (no
        cursor).
        """
        num_copies = 0
        prev_idx = None
        for p in range(min(self.D, self.num_steps) or self._single_block()):
            idx = self._unravel(p)
            if p == 0 or self._indices_differ(prev_idx, idx):
                slot = num_copies % self.num_slots
                self._load_tile(
                    src,
                    get_slot(self.slots, slot),
                    idx,
                    get_slot(self.sem, slot),
                    post_count,
                )
                num_copies += 1
            prev_idx = idx
        if self._advances_every_step():
            return None
        return num_copies

    def copy_in(self, ctx, src, load_count, post_count=1):
        """Phase 1 — prefetch the block ``D`` steps ahead of DRAM ``src`` into
        ``copy_slot``, signaling its load semaphore ``post_count`` times.  An
        always-advance input copies unconditionally (gated only by
        ``has_fetch``) and carries no cursor; a guarded one copies only when a
        new block enters the window edge and advances its producer cursor.
        Returns the advanced producer count (guarded) or ``None``
        (always-advance).
        """
        if self.D == 0 and self._single_block():
            # The prologue primed the operand's only block.
            return load_count
        nb = self.num_slots
        w = ctx.windows[nb]
        has_fetch = w.has_fetch if self.D > 0 else True
        if self._advances_every_step():
            copy_slot = (ctx.step + self.D) % nb
            should_copy = has_fetch
            next_count = None
        else:
            copy_slot = load_count % nb
            differ = self._indices_differ(w.prev_edge, w.fetch_idx)
            # No prefetch-ahead (``D == 0``: base single-buffer, or the async
            # kernel's num_slots == 2): copy on the first step or a block change
            # (no prologue primes the first block).  Otherwise prefetch-gated.
            should_copy = (
                (w.first | differ) if self.D == 0 else (has_fetch & differ)
            )
            next_count = torch.sym_ite(should_copy, load_count + 1, load_count)
        # ``_check`` against the slot's own size lets the select bound
        # resolve on the unbacked step (needed for num_slots >= 3).
        torch._check(copy_slot < self.slots.size(0))
        effect_cond(
            should_copy,
            lambda: self._load_tile(
                src,
                get_slot(self.slots, copy_slot),
                w.fetch_idx,
                get_slot(self.sem, copy_slot),
                post_count,
            ),
        )
        return next_count

    def wait_in(self, ctx, wait_count):
        """Phase 2 — wait on the read-slot load semaphore once per consumed
        block, then return the tile.  An always-advance input waits
        unconditionally (changes block every step); a reused input waits on
        entering a new block, or — with ``spec.first_use_at_exit`` — on
        completing one (so a single-buffered late-consumed operand's load
        overlaps the whole sweep).
        """
        if self._advances_every_step():
            rs = ctx.windows[self.num_slots].cur_slot
            pred = None
        else:
            rs = wait_count % self.num_slots
            if self.spec.first_use_at_exit:
                # First read is the sweep's last step: defer the wait to
                # block-exit (the ``finished`` predicate).
                pred = ctx.last | self._indices_differ(ctx.cur, ctx.next)
            else:
                pred = (ctx.step == 0) | self._indices_differ(ctx.prev, ctx.cur)
        torch._check(rs < self.slots.size(0))
        _guarded_wait(get_slot(self.sem, rs), pred)
        return get_slot(self.slots, rs)

    def advance_consumer(self, ctx, wait_count):
        """Phase 6 (guarded inputs) — the current block is done when it changes
        next step or this is the last step; advance the consumer cursor.
        """
        finished = ctx.last | self._indices_differ(ctx.cur, ctx.next)
        return torch.sym_ite(finished, wait_count + 1, wait_count)

    # --- output phases (kind == _OUT) ---------------------------------------

    def wait_out(self, ctx, store_count):
        """Phase 3 — before reusing an output slot, wait on its previous store
        (once per tile, and only when the slot already holds a prior store —
        ``store_count >= nb``; the first ``nb`` uses have nothing to drain).
        With ``spec.first_use_at_exit`` the drain is deferred to block-exit (the
        write step) instead of block-entry, so a single-buffered output's store
        overlaps the next tile's sweep.  Returns ``(out_slot, slot_index)``.
        """
        if self._advances_every_step():
            slot = ctx.windows[self.num_slots].cur_slot
        else:
            slot = store_count % self.num_slots
        torch._check(slot < self.slots.size(0))
        if self.spec.first_use_at_exit:
            # Drain at block-exit (the write step): the ``finished`` predicate,
            # whose ``last`` term catches the final tile (``next`` is OOB).
            changed = ctx.last | self._indices_differ(ctx.cur, ctx.next)
        else:
            changed = self._indices_differ(ctx.prev, ctx.cur)
        pred = changed & (store_count >= self.num_slots)
        _guarded_wait(get_slot(self.sem, slot), pred)
        return get_slot(self.slots, slot), slot

    def copy_out(self, ctx, dst, store_count, out_slot, slot_idx):
        """Phase 5 — store the completed output tile ``out_slot`` to DRAM
        ``dst``, signaling its store semaphore, and return the advanced store
        counter.  Unconditional when the output advances every step; otherwise
        store-in-cond so a reduction writes once per output tile (when its block
        completes or on the last step).
        """
        sem = get_slot(self.sem, slot_idx)
        if self._advances_every_step():
            self._store_tile(out_slot, dst, ctx.cur, sem)
            return store_count + 1
        should_store = self._indices_differ(ctx.cur, ctx.next) | ctx.last
        effect_cond(
            should_store,
            lambda: self._store_tile(out_slot, dst, ctx.cur, sem),
        )
        return torch.sym_ite(should_store, store_count + 1, store_count)

    def drain(self, final_store_count):
        """Finalize — drain each slot's last (un-reused) store so the DRAM
        result is complete.  Slot ``j`` holds a pending store iff ``j <
        final_store_count`` (a small grid leaves the rest un-signaled).
        """
        for j in range(self.num_slots):
            _guarded_wait(get_slot(self.sem, j), j < final_store_count)

    # --- async-kernel input protocol (AsyncPipelinedKernel) ------------------
    #
    # Reuses the base ``copy_in`` / ``prime_prologue`` / ``advance_consumer``
    # prefetch, but emits no input ``async_wait``: a tile's load semaphore
    # feeds ``commit.dependencies`` instead (``read_advancing`` /
    # ``read_reused`` are ``wait_in`` without the blocking wait).  Prefetch
    # distance is one less than the base (``D = num_slots - 2``): a tile is
    # consumed an iteration later, so a slot a prefetch overwrites was already
    # retired — reuse-WAR-safe.  An advancing input posts ``+1`` per load; a
    # reused one (held across an inner sweep) loads once per block and posts
    # ``+R`` (``reuse_count``), balancing a per-step consume.

    def read_advancing(self, ctx):
        """The current read slot's tile and its load semaphore (no wait) — the
        semaphore is handed to ``commit.dependencies``."""
        slot = ctx.windows[self.num_slots].cur_slot
        torch._check(slot < self.slots.size(0))
        return get_slot(self.slots, slot), get_slot(self.sem, slot)

    def read_reused(self, ctx, wait_count):
        """A reused input's current read slot (consumer cursor ``wait_count``)
        and its load semaphore, no wait — like ``wait_in`` without the blocking
        ``async_wait`` (the wait moves into ``commit.dependencies``)."""
        slot = wait_count % self.num_slots
        torch._check(slot < self.slots.size(0))
        return get_slot(self.slots, slot), get_slot(self.sem, slot)


class PipelinedKernel(torch.nn.Module):
    """Spec-driven, mutate-style kernel scheduler (see module docstring).

    ``kernel(grid_index, *in_slots, *out_slots)`` is the per-tile compute; it
    writes each output SRAM slot (via ``voyager.insert``) rather than
    returning a value.  A ``None`` input spec is a whole / scalar / codebook
    operand, passed through un-tiled.  ``num_slots`` is the software-pipeline
    depth (2 = double buffering).

    This class owns orchestration only — buffer / semaphore allocation, the FX
    ``while_loop`` construction, the compute-kernel invocation, and global grid
    traversal (the per-step delinearized indices).  Each operand's per-reference
    state machine lives in a ``_BufferedRef``; ``forward`` drives them in phase
    order, threading each operand's DRAM buffer in per call (Pallas style).
    """

    # The synchronous scheduler; :class:`AsyncPipelinedKernel` sets this True.
    # It selects the prefetch distance (``count - 1`` vs ``count - 2``) in
    # ``_step_ctx`` and the ``_BufferedRef`` ``async_pipeline`` flag, so the
    # two stay consistent from one source.
    _async_pipeline = False

    def __init__(
        self,
        kernel: Callable,
        grid: Tuple[int, ...],
        in_specs: List[Optional[_InputSpec]],
        out_specs: List[_OutputSpec],
        scratch_specs: Sequence[_ScratchSpec] = (),
        num_slots: int = _DEFAULT_NUM_SLOTS,
    ):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.kernel = kernel
        self.grid = grid
        self.in_specs = in_specs
        self.out_specs = out_specs
        # Persistent, unbuffered, non-DMA SRAM refs (e.g. a reduction
        # accumulator); appended after the input/output refs in the kernel call.
        self.scratch_specs = tuple(scratch_specs)
        self.num_slots = num_slots
        self.ndim = len(self.grid)
        self.num_steps = math.prod(self.grid)

    def _num_slots(self, spec):
        """Resolve an operand's software-pipeline depth: its per-spec
        ``num_slots`` override, else the scheduler default -- but one slot for
        an operand the sweep reads a single tile of.  A pipeline slot exists to
        hold the *next* tile while this one is in use, so an operand that never
        advances has nothing to put in a second one; the extra slot would be
        allocated and never written.  A dim indexed by grid dim ``g`` advances
        every ``r`` steps, so it takes ``ceil(grid[g] / r)`` block values, and
        their product is how many tiles the operand has.
        """
        n = self.num_slots if spec.num_slots is None else spec.num_slots
        if n < 1:
            raise ValueError("num_slots must be >= 1")
        blocks = 1
        for _, g, r in spec_tiled_dims(spec, self.grid):
            blocks *= -(-self.grid[g] // r)
        return 1 if blocks == 1 else n

    def _step_ctx(self, step, distinct_counts):
        """The per-step index context: delinearized ``cur`` / ``next`` /
        ``prev`` / ``last`` plus one prefetch window per distinct buffer count.
        An operand of depth ``count`` reads slot ``step % count`` and prefetches
        ``d`` blocks ahead (``prev_edge`` is the block one before the window
        edge).  ``d`` is ``count - 1`` for the synchronous kernel and
        ``count - 2`` for the async one (``_async_pipeline``) — one less,
        because the async commit consumes a tile an iteration later, so it
        prefetches one fewer block to keep slot reuse WAR-safe.  ``d``
        therefore matches each input ref's ``self.D``.
        """
        cur = voyager.delinearize_index(step, self.grid)
        nxt = voyager.delinearize_index(step + 1, self.grid)
        prev = voyager.delinearize_index(step - 1, self.grid)
        last = step + 1 >= self.num_steps
        windows = {}
        for count in distinct_counts:
            d = max(0, count - 2) if self._async_pipeline else count - 1
            cur_slot = step % count
            fetch_step = step + d
            has_fetch = fetch_step < self.num_steps
            first = None
            if d == 0:
                first = step == 0
                fetch_idx = cur
                prev_edge = prev
            elif d == 1:
                fetch_idx, prev_edge = nxt, cur
            else:
                fetch_idx = voyager.delinearize_index(fetch_step, self.grid)
                prev_edge = voyager.delinearize_index(fetch_step - 1, self.grid)
            windows[count] = _Window(
                cur_slot, fetch_idx, prev_edge, has_fetch, first
            )
        return _StepCtx(step, cur, nxt, prev, last, windows)

    def _allocate(self, inputs):
        """Build the shared per-``forward`` allocation: DRAM output buffers, the
        tiled ``(input, spec)`` pairs, each input/output operand's buffer count,
        and one pipelined ``_BufferedRef`` scheduler per operand, plus scratch
        SRAM.  Returns them as a tuple ``(out_bufs, tiled, in_counts,
        out_counts, in_refs, out_refs, scratch_bufs)``.  Each input ref's
        prefetch distance
        follows ``self._async_pipeline`` (``num_slots - 2`` when async,
        ``num_slots - 1`` otherwise)."""
        out_bufs = [voyager.alloc(s.shape, s.dtype) for s in self.out_specs]
        tiled = [
            (inp, s) for inp, s in zip(inputs, self.in_specs) if s is not None
        ]
        # Per-operand software-pipeline depth: each operand's own ``num_slots``
        # (spec override, else the scheduler default), so a reused / low-reuse
        # operand can run a shallower or deeper pipeline than its peers.
        in_counts = [self._num_slots(s) for _, s in tiled]
        out_counts = [self._num_slots(s) for s in self.out_specs]
        # One SRAM slot per operand; separate pass so all slot ``alloc``s
        # precede the refs' semaphore ``zeros`` in graph order.
        in_slots = [
            voyager.alloc(s.tile_sizes, inp.dtype, _SRAM, num_slots=c)
            for (inp, s), c in zip(tiled, in_counts)
        ]
        out_slots = [
            voyager.alloc(s.tile_sizes, s.dtype, _SRAM, num_slots=c)
            for s, c in zip(self.out_specs, out_counts)
        ]
        # Per-operand schedulers: each owns its SRAM window (slot) and
        # allocates its own per-slot semaphore.
        in_refs = [
            _BufferedRef(
                _BufferedRef._IN,
                s,
                self.grid,
                c,
                slots,
                async_pipeline=self._async_pipeline,
            )
            for (inp, s), c, slots in zip(tiled, in_counts, in_slots)
        ]
        out_refs = [
            _BufferedRef(
                _BufferedRef._OUT,
                s,
                self.grid,
                c,
                slots,
                async_pipeline=self._async_pipeline,
            )
            for s, c, slots in zip(self.out_specs, out_counts, out_slots)
        ]
        # Scratch refs, captured like ``out_slots``.  A double-buffered
        # scratch (``_ScratchSpec.num_slots``, async only) allocates slotted
        # like any pipelined buffer; the plain accumulator keeps the
        # unslotted alloc, reused immediately for the next tile's reduction.
        scratch_bufs = tuple(
            (
                voyager.alloc(s.shape, s.dtype, _SRAM, num_slots=s.num_slots)
                if self._async_pipeline and s.num_slots > 1
                else voyager.alloc(s.shape, s.dtype, _SRAM)
            )
            for s in self.scratch_specs
        )
        return (
            out_bufs,
            tiled,
            in_counts,
            out_counts,
            in_refs,
            out_refs,
            scratch_bufs,
        )

    def forward(self, *inputs):
        (
            out_bufs,
            tiled,
            in_counts,
            out_counts,
            in_refs,
            out_refs,
            scratch_bufs,
        ) = self._allocate(inputs)
        num_outputs = len(out_refs)
        distinct_counts = list(dict.fromkeys(in_counts + out_counts))

        # Prologue: prime the first ``D`` logical positions per input,
        # deduplicating reused blocks.  A guarded input's prologue copy count
        # seeds its producer cursor (always-advance inputs return ``None``).
        init_copy_in = []
        for i, (inp, spec) in enumerate(tiled):
            c = in_refs[i].prime_prologue(inp)
            if c is not None:
                init_copy_in.append(c)

        def cond_fn(step, load_counts, wait_counts, store_counts):
            return step < self.num_steps

        def body_fn(step, load_counts, wait_counts, store_counts):
            ctx = self._step_ctx(step, distinct_counts)

            # 1. COPY-IN: prefetch each input; a guarded input advances its
            #    producer cursor (appended in ``tiled`` order), an
            #    always-advance one carries none.
            next_load_counts = []
            g = 0
            for i, (inp, _) in enumerate(tiled):
                ref = in_refs[i]
                if ref._advances_every_step():
                    ref.copy_in(ctx, inp, None)
                else:
                    next_load_counts.append(
                        ref.copy_in(ctx, inp, load_counts[g])
                    )
                    g += 1

            # 2. WAIT-IN: wait on each input's read-slot load semaphore, then
            #    read it.  ``None``-spec operands pass through in kernel order.
            in_slots, i, g = [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_slots.append(inp)
                    continue
                ref = in_refs[i]
                if ref._advances_every_step():
                    in_slots.append(ref.wait_in(ctx, None))
                else:
                    in_slots.append(ref.wait_in(ctx, wait_counts[g]))
                    g += 1
                i += 1

            # 3. WAIT-OUT (reuse): drain each output slot's prior store before
            #    the kernel overwrites it.
            out_slots, out_slot_idxs = [], []
            for i in range(num_outputs):
                slot_ref, slot = out_refs[i].wait_out(ctx, store_counts[i])
                out_slots.append(slot_ref)
                out_slot_idxs.append(slot)

            # 4. KERNEL (mutate-style: writes the output slots).  Scratch
            #    refs follow the input/output args (Pallas's *index, *inputs,
            #    *outputs, *scratch convention).
            self.kernel(ctx.cur, *in_slots, *out_slots, *scratch_bufs)

            # 5. COPY-OUT: store each completed output tile (guarded), signaling
            #    its store semaphore; advance the store counter.
            next_store_counts = []
            for i in range(num_outputs):
                next_store_counts.append(
                    out_refs[i].copy_out(
                        ctx,
                        out_bufs[i],
                        store_counts[i],
                        out_slots[i],
                        out_slot_idxs[i],
                    )
                )

            # 6. Consumer advance (guarded inputs), in ``tiled`` order.
            next_wait_counts, g = [], 0
            for in_ref in in_refs:
                if in_ref._advances_every_step():
                    continue
                next_wait_counts.append(
                    in_ref.advance_consumer(ctx, wait_counts[g])
                )
                g += 1

            return (
                step + 1,
                tuple(next_load_counts),
                tuple(next_wait_counts),
                tuple(next_store_counts),
            )

        init = (
            0,
            # producer cursors, one per guarded input (prologue copy count)
            tuple(init_copy_in),
            # consumer cursors, one per guarded input (start at block 0)
            (0,) * len(init_copy_in),
            # store counters
            (0,) * num_outputs,
        )
        final = while_loop(cond_fn, body_fn, init)

        # Finalize: drain each output slot's last (un-reused) store so the DRAM
        # result is complete.
        final_store_counts = final[3]
        for i in range(num_outputs):
            out_refs[i].drain(final_store_counts[i])

        return out_bufs[0] if len(out_bufs) == 1 else tuple(out_bufs)


# Compute-done semaphore depth: two slots so tile ``s`` and ``s+1`` post to
# different slots (a lagged retire waits ``s`` while the body commits ``s+1``).
_DONE_DEPTH = 2


class AsyncPipelinedKernel(PipelinedKernel):
    """:class:`PipelinedKernel` that overlaps consecutive tiles via ``commit``.

    Each step *dispatches* its compute with ``voyager.commit`` (send the params,
    don't wait), so the next tile's array ramp-up overlaps the current tile's
    ramp-down.  The retire (wait compute-done, then store) lags one tile: the
    body commits tile ``s`` and retires ``s - 1`` (``s >= 1``); an epilogue
    retires the last.

    ``self.kernel`` (the ``async_pipeline`` kernel template) issues the
    ``commit`` itself, waiting the input load semaphores and — on the round it
    writes the output — that slot's store semaphore (seeded with a credit, so
    a slot's first use never underflows).  Inputs are never ``async_wait``-ed
    in the loop; the only loop-level wait is the lagged retire on compute-done.
    Inputs prefetch ``num_slots - 2`` ahead (one less than the base), so a slot
    a load overwrites was already retired — reuse-WAR-safe.
    """

    _async_pipeline = True

    def forward(self, *inputs):
        (
            out_bufs,
            tiled,
            in_counts,
            out_counts,
            in_refs,
            out_refs,
            scratch_bufs,
        ) = self._allocate(inputs)
        num_outputs = len(out_refs)
        distinct_counts = list(dict.fromkeys(in_counts + out_counts))

        # Compute-done semaphore slot: commit(s) posts slot s % 2, the lagged
        # retire of tile s waits it.
        done_sem = voyager.zeros([], torch.int64, num_slots=_DONE_DEPTH)

        # Per-operand reuse count: 1 = advancing (posts +1/step), R = reused
        # (loaded once per block, posts +R so a per-step consume balances it).
        reuse = [ref.reuse_count() for ref in in_refs]

        # Prologue: prime the first ``D = num_slots - 2`` prefetch positions per
        # input (deduped by block, posting the reuse count); a reused input
        # seeds its producer cursor, an advancing one carries none.
        init_load = []
        for i, (inp, _) in enumerate(tiled):
            c = in_refs[i].prime_prologue(inp, reuse[i])
            if c is not None:
                init_load.append(c)

        def _store_out(i, coord, store_count):
            """Store output ``i``'s tile to DRAM at grid ``coord`` from the slot
            of the tile that just finished — ``(store_count - 1) % num_slots``,
            one behind the tile the commit is on."""
            ref = out_refs[i]
            slot = (store_count - 1) % ref.num_slots
            ref._store_tile(
                get_slot(ref.slots, slot),
                out_bufs[i],
                coord,
                get_slot(ref.sem, slot),
            )

        def cond_fn(step, load_counts, wait_counts, store_counts):
            return step < self.num_steps

        def body_fn(step, load_counts, wait_counts, store_counts):
            ctx = self._step_ctx(step, distinct_counts)

            # 1. Prefetch each input ``D`` blocks ahead (base ``copy_in``): an
            #    advancing input copies every step (+1, no cursor); a reused one
            #    copies once per block (+R) and advances its producer cursor.
            next_load, g = [], 0
            for i, (inp, _) in enumerate(tiled):
                ref = in_refs[i]
                if ref._advances_every_step():
                    ref.copy_in(ctx, inp, None, reuse[i])
                else:
                    next_load.append(
                        ref.copy_in(ctx, inp, load_counts[g], reuse[i])
                    )
                    g += 1

            # 2. Read each input's current tile + its load semaphore — no wait,
            #    the semaphore feeds commit.dependencies (the array ramp-up
            #    waits it, not the loop).
            in_slots, in_sems, i, g = [], [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_slots.append(inp)
                    continue
                ref = in_refs[i]
                if ref._advances_every_step():
                    tile, sem = ref.read_advancing(ctx)
                else:
                    tile, sem = ref.read_reused(ctx, wait_counts[g])
                    g += 1
                in_slots.append(tile)
                in_sems.append(sem)
                i += 1

            # The output tile's ordinal is ``store_counts`` (tiles finished so
            # far); all its reduction rounds share slot ``ordinal % num_slots``
            # (they accumulate into it), rotating per tile.
            out_slots, out_sems = [], []
            for i in range(num_outputs):
                slot = store_counts[i] % out_refs[i].num_slots
                out_slots.append(get_slot(out_refs[i].slots, slot))
                out_sems.append(get_slot(out_refs[i].sem, slot))

            # A double-buffered scratch rotates with the output tile's
            # ordinal, like the out slots; the lagged retire alone orders a
            # slot's reuse behind its last reader (see ``_ScratchSpec``).
            scratch_slots = [
                (
                    get_slot(buf, store_counts[0] % spec.num_slots)
                    if spec.num_slots > 1
                    else buf
                )
                for buf, spec in zip(scratch_bufs, self.scratch_specs)
            ]

            post = get_slot(done_sem, step % _DONE_DEPTH)

            # 3. Dispatch the tile's compute: the async kernel issues the
            #    ``commit`` itself — waiting the input load semaphores and, on
            #    the round it writes the output, that slot's store semaphore
            #    free (seeded with a credit, so first uses never underflow).
            self.kernel(
                ctx.cur,
                in_slots,
                out_slots,
                scratch_slots,
                in_sems,
                out_sems,
                post,
            )

            # 4. Lagged retire of tile step-1 (nothing at step 0): wait its
            #    compute-done every step, and store each output once its *own*
            #    block has finished (slot ordinal one behind the tile committed
            #    now).  Flat conds — a ``SymBool`` predicate cannot be captured
            #    into an outer cond's operands, so ``step >= 1`` folds into the
            #    store predicate.
            effect_cond(
                step >= 1,
                lambda: voyager.async_wait(
                    get_slot(done_sem, (step - 1) % _DONE_DEPTH)
                ),
            )

            # 5. Retire (store) + advance the finished-tile count, per output.
            #    An output that advances every step stores every step (guard
            #    just ``step >= 1``) and its count ticks unconditionally — no
            #    ``_indices_differ`` (always true), matching ``copy_out``'s fast
            #    path.  A held output stores / ticks only on its own block
            #    boundary.
            next_store = []
            for i in range(num_outputs):
                ref = out_refs[i]
                if ref._advances_every_step():
                    should_store = step >= 1
                    nxt = store_counts[i] + 1
                else:
                    should_store = (step >= 1) & ref._indices_differ(
                        ctx.prev, ctx.cur
                    )
                    nxt = torch.sym_ite(
                        ref._indices_differ(ctx.cur, ctx.next) | ctx.last,
                        store_counts[i] + 1,
                        store_counts[i],
                    )
                effect_cond(
                    should_store,
                    lambda i=i: _store_out(i, ctx.prev, store_counts[i]),
                )
                next_store.append(nxt)

            # 6. Advance each reused input's consumer cursor (block finished).
            next_wait, g = [], 0
            for in_ref in in_refs:
                if in_ref._advances_every_step():
                    continue
                next_wait.append(in_ref.advance_consumer(ctx, wait_counts[g]))
                g += 1

            return (
                step + 1,
                tuple(next_load),
                tuple(next_wait),
                tuple(next_store),
            )

        init = (
            0,
            tuple(init_load),
            (0,) * len(init_load),
            (0,) * num_outputs,
        )
        final = while_loop(cond_fn, body_fn, init)

        # Epilogue: retire the final step (its commit is never lagged-retired;
        # the last step always completes a tile, so it stores), then drain every
        # slot's last store (one per output tile).
        final_store = final[3]
        if self.num_steps >= 1:
            last = self.num_steps - 1
            coord = out_refs[0]._unravel(last)
            voyager.async_wait(get_slot(done_sem, last % _DONE_DEPTH))
            for i in range(num_outputs):
                _store_out(i, coord, final_store[i])
        for i in range(num_outputs):
            out_refs[i].drain(final_store[i])

        return out_bufs[0] if len(out_bufs) == 1 else tuple(out_bufs)


def build_pipelined_buffers(
    kernel: Callable,
    grid: Tuple[int, ...],
    in_specs: List[Optional[_InputSpec]],
    out_specs: List[_OutputSpec],
    inputs: Tuple[torch.Tensor, ...],
    *,
    scratch_specs: Sequence[_ScratchSpec] = (),
    num_slots: int = _DEFAULT_NUM_SLOTS,
    async_pipeline: bool = False,
    kwargs: Optional[dict] = None,
    wrapper: Optional[Callable] = None,
) -> torch.fx.GraphModule:
    """Build the bufferized FX graph (a single rolled ``while_loop`` over
    ``voyager.*`` primitives) for ``kernel`` over ``grid``.  Mirrors
    ``build_pointwise_buffers``'s export / finalize / extent-tag flow.

    ``async_pipeline`` selects :class:`AsyncPipelinedKernel` (cross-tile
    ramp-up/ramp-down overlap via ``voyager.commit``) instead of the
    synchronous :class:`PipelinedKernel`.  ``kernel`` must then be an async
    kernel template (``_map_kernel(async_pipeline=True)``) that issues the
    ``commit`` itself.

    ``wrapper`` optionally wraps the pattern module before export (``pattern =
    wrapper(pattern)``) -- e.g. the GQA fold that reshapes operands in and the
    output back out; ``inputs`` are then the wrapper's (unfolded) operands.
    """
    cls = AsyncPipelinedKernel if async_pipeline else PipelinedKernel
    pattern = cls(
        kernel,
        grid,
        in_specs,
        out_specs,
        scratch_specs=scratch_specs,
        num_slots=num_slots,
    )
    num_steps = pattern.num_steps
    if wrapper is not None:
        pattern = wrapper(pattern)
    with _lenient_verifier():
        gm = export_model(pattern, inputs, kwargs=kwargs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(0, num_steps, 1)]])
    # Stamp a concrete-offset ``.value`` on every node (incl. loop / cond
    # bodies) so the tail re-fusion's ShapeProp never sees export's symbolic
    # ``step % num_slots`` tile offset.
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*inputs)
    return gm


# ---------------------------------------------------------------------------
# Op-family builders
#
# Each takes the FX ``node`` being lowered and returns a bufferized
# ``GraphModule`` (a rolled ``while_loop`` of ``voyager.*`` primitives) that
# substitutes for the node, or ``None`` when uncovered.  They mirror
# ``bufferization._build_for_*`` but target the pipelined scheduler: a
# return-style per-tile ``compute`` is wrapped mutate-style by ``_map_kernel``
# (each result written into its output slot), accumulating across the reduction
# grid dim for a GEMM / conv and overwriting for a map.
# ---------------------------------------------------------------------------


@dataclass
class _FusedInfo:
    """Parsed pieces of a fused ``call_module`` (GEMM/conv + post-op pointwise
    ops), for the GEMM / conv pipeline builders.

    Attributes:
        anchor: The GEMM/conv reference op -- inside the submodule, so its
            ``args`` are submodule placeholders whose ``meta['source_node']``
            points back to the outer graph.
        fused_gm: Runs the post-op ops as ``[acc, *fused] -> output(s)`` on
            the anchor's result tile; ``None`` when there is no tail to run
            (a submodule can hold the anchor and nothing but its prelude, a
            GQA ``expand``, and still be a ``_fused`` node).
        tiling: The anchor's per-dim tile factors, or ``None`` when untiled.
        l3_order: The L3 loop order the tiler chose, or ``None`` for the
            canonical one.
        in_nodes: Each fused input's outer graph node -- the tail's operands
            only, which the plan merges with the anchor's own and orders
            canonically.
        in_specs: Each input's tile ``_InputSpec``, or ``None`` for a whole
            input.
        out_specs: One fully resolved ``_OutputSpec`` per fused output --
            several when the fused op returns a tuple (``quantize_mx``).
    """

    anchor: torch.fx.Node
    fused_gm: Optional[torch.fx.GraphModule]
    tiling: Optional[Tuple[int, ...]]
    l3_order: Optional[Tuple[str, ...]]
    in_nodes: List[torch.fx.Node]
    in_specs: List[Optional[_InputSpec]]
    out_specs: List[_OutputSpec]


def _retile_mha_view(fused_gm, nb, tm) -> None:
    """Rewrite the MHA relayout ``view`` in the fused body to tile dims.

    The original view's ``M`` dim is the full (untiled) M, so on a tile it
    scrambles the data — replace it with the tile's ``tm`` and let the split
    outer (heads) auto-size via ``-1``.  View dims are ``[*batch, M, H,
    head_dim]`` (M at index ``nb``).  The body isn't ShapeProp'd, so navigate
    output -> [quantize] -> perm -> view by structure, not shape.
    """
    out = next(n for n in fused_gm.graph.nodes if n.op == "output")
    perm = out.args[0]
    if isinstance(perm, (list, tuple)):
        perm = perm[0]
    if perm.target is _QUANTIZE_MX:
        perm = perm.args[0]
    view = perm.args[0]
    dims = list(view.args[1])
    dims[nb] = tm  # M -> tile M
    dims[nb + 1] = -1  # split outer (heads) auto-sizes
    view.update_arg(1, dims)
    fused_gm.graph.lint()
    fused_gm.recompile()


def _detect_mha_relayout(fused_ops, anchor, tiling, gm, grid_m, grid_n):
    """If the fused tail ends with an MHA output relayout
    (``is_mha_qkv_permute`` — a ``transpose(1,2)`` / ``permute([0,2,1,3])`` on a
    4-D tensor), return the relaid-out output ``(tile_sizes, index_map,
    shape)`` (the tile stored to the permuted block, tiling the output on its
    own axes) and, for the projection case, retile the body's ``view`` in place;
    else ``None`` (the caller keeps the default, un-permuted output spec).

    A microscaling quantize may sit on top of the permute, quantizing the tile
    on its way out; the relayout is the permute below it, and the returned
    ``shape`` is the relaid-out *data*, which the caller dices its scale
    against.

    Two kinds, differing in where the head comes from:

      * projection: a ``view`` / ``reshape`` splits the gemm's ``N`` into
        ``(H, head_dim)`` and the permute makes the heads outer ->
        ``[B, H, S, head_dim]``.  Store the gemm tile transposed -- M -> S,
        N -> head (outer = heads, tiled by the N grid; inner = head_dim, whole).
      * ``P @ V`` context matmul: the output is *already* 4-D
        ``[B, H, S, head_dim]`` with the head a *looped* batch dim, and a bare
        ``transpose(1,2)`` moves it after M -> ``[B, S, H, head_dim]``.  Tile
        the output on its permuted axes -- S <- M grid, head <- its (looped)
        batch grid dim (H_t = 1), head_dim <- N grid. The body already emits the
        transposed tile, so there is no view to retile.

    ``fused_ops`` must be non-empty and ``tiling`` must not be ``None``.
    """
    perm = trailing_mha_perm(fused_ops)
    if perm is None:
        return None
    head_dim = perm.value.shape[-1]  # head_dim (unchanged by the perm)
    g_out = anchor.value  # gemm output [*batch, M, N]
    nb = g_out.ndim - 2
    M, N = g_out.shape[-2], g_out.shape[-1]
    nm, nn = tiling[nb], tiling[nb + 1]
    tm, tn = M // nm, N // nn
    if perm.value.ndim > g_out.ndim:
        if tn % head_dim != 0:
            raise NotImplementedError(
                f"MHA output relayout: N tile {tn} is not a multiple of "
                f"head_dim {head_dim} (would split a head across tiles)"
            )
        tb = tuple(g_out.shape[:nb])
        out_tile = tb + (tn // head_dim, tm, head_dim)  # [*b, H_t, S_t, hd]
        out_imap = tuple(range(nb)) + (grid_n, grid_m, None)  # H<-N, S<-M, hd
        _retile_mha_view(gm, nb, tm)
    else:
        outer = tuple(g_out.shape[: nb - 1])
        out_tile = outer + (tm, 1, tn)  # [*outer, S_t, H_t=1, head_dim]
        out_imap = tuple(range(nb - 1)) + (grid_m, nb - 1, grid_n)
    return out_tile, out_imap, tuple(perm.value.shape)


def parse_fused_submodule(node, tiler=None) -> Optional["_FusedInfo"]:
    """Parse a fused ``call_module`` ``node`` into a ``_FusedInfo``, or ``None``
    if ``node`` is not a fused submodule (a bare op the builder reads directly).

    The submodule (``node.meta['submodule']``) holds a GEMM/conv anchor followed
    by post-op pointwise ops.  The fused operands / outputs tile at the output
    block, diced from the anchor's per-dim tile factors (``get_tiling``,
    projected to the output's physical layout).  The factors and the L3 loop
    order are stashed on ``_FusedInfo`` so the builder reuses them -- no second
    tiler run, and one order behind both the ``index_map``s built here and the
    grid the builder emits.
    """
    if node.op != "call_module":
        return None
    submod = node.meta.get("submodule")
    anchor = get_anchor_node(node)
    is_conv = is_conv2d(anchor)

    ShapeProp(submod).propagate(
        *(n.value.clone() for n in node.all_input_nodes)
    )

    tiling, l3_order = get_tiling(node, tiler)
    # ``out_tiling`` is per output *dim* and so is order-independent; only
    # ``out_index_map``, which names a grid dim per output dim, moves with it.
    if is_conv:
        odims = NCHW_TO_NHWC if anchor.meta.get("transposed", False) else None
        order = l3_order or CONV_L3_ORDER
        gK, gY, gX = (1 + order.index(d) for d in CONV_L3_ORDER)
        out_index_map = project((0, gK, gY, gX), odims)
        if tiling is None:
            out_tiling = None
        else:
            ny, nx, nk, _ = tiling  # logical (Y, X, K, C) counts
            out_tiling = project((1, nk, ny, nx), odims)  # physical counts
    else:
        nb = anchor.value.ndim - 2
        order = l3_order or GEMM_L3_ORDER
        grid_m, grid_n = (nb + order.index(d) for d in GEMM_L3_ORDER)
        out_index_map = tuple(range(nb)) + (grid_m, grid_n)
        # gemm (batch.., n_m, n_n, n_k) -> drop K
        out_tiling = tiling[:-1] if tiling is not None else None

    anchor_prelude = ancestors(anchor)
    fused_ops = []
    phs, in_specs = [], []
    for sn in submod.graph.nodes:
        if sn is anchor or sn.op != "call_function" or sn in anchor_prelude:
            continue
        fused_ops.append(sn)
        codebooks = quant_param_arg_nodes(sn)
        for inp in sn.all_input_nodes:
            if inp.op != "placeholder" or inp in phs or inp in anchor_prelude:
                continue
            phs.append(inp)
            if inp in codebooks or inp.value.numel() == 1 or out_tiling is None:
                in_specs.append(None)  # whole operand
            else:
                in_specs.append(
                    _compute_input_spec(
                        out_tiling, tuple(inp.shape), out_index_map
                    )
                )

    fused_gm = (
        _build_fused_gm(submod, anchor, fused_ops, phs) if fused_ops else None
    )
    in_nodes = [n.meta.get("source_node", n) for n in phs]

    multi_outputs = isinstance(node.value, (list, tuple))
    vals = list(node.value) if multi_outputs else [node.value]
    full_shapes = [tuple(v.shape) for v in vals]
    if out_tiling is None and is_bmm(anchor):
        tiled_shape = [(1,) * (len(s) - 2) + tuple(s[-2:]) for s in full_shapes]
    elif out_tiling is None:
        tiled_shape = full_shapes  # untiled -> tile == full tensor (trip-1)
    else:
        tiled_shape = compute_output_tiled_shapes(node, out_tiling)
        tiled_shape = list(tiled_shape) if multi_outputs else [tiled_shape]
    # The default M/N mapping, except for the MHA relayout handled below.
    out_specs = [
        _OutputSpec(s, tuple(t), out_index_map, v.dtype)
        for s, t, v in zip(full_shapes, tiled_shape, vals)
    ]

    if not is_conv and fused_ops and tiling is not None:
        relayout = _detect_mha_relayout(
            fused_ops, anchor, tiling, fused_gm, grid_m, grid_n
        )
        if relayout is not None:
            out_tile, out_imap, data_shape = relayout
            out_specs = [
                _OutputSpec(
                    s,
                    tuple(
                        t * d // full
                        for t, d, full in zip(out_tile, s, data_shape)
                    ),
                    out_imap,
                    v.dtype,
                )
                for s, v in zip(full_shapes, vals)
            ]

    return _FusedInfo(
        anchor, fused_gm, tiling, l3_order, in_nodes, in_specs, out_specs
    )


def _map_kernel(
    compute: Callable,
    num_outputs: int,
    num_scratch: int = 0,
    async_pipeline: bool = False,
):
    """Map kernel (no cross-tile reduction): adapt a return-style
    ``compute(*in_slots) -> Tensor | tuple`` into the scheduler's mutate-style
    kernel, writing each result straight into its output slot.  Every num_k == 1
    op uses this; the reduction case uses ``_reduction_fused_kernel``.

    ``num_scratch`` trailing refs are handed to ``compute`` after the input
    tiles: a map computes nothing across tiles, so its scratch is space the
    backend's own passes write, which the op names but never reads.

    ``async_pipeline=False`` returns the synchronous :class:`PipelinedKernel`
    kernel (``kernel(grid_index, *in_slots, *out_slots)``, insert inline).
    ``async_pipeline=True`` returns the :class:`AsyncPipelinedKernel`
    counterpart: it *dispatches* the same compute with ``voyager.commit`` so
    the next tile's systolic-array ramp-up overlaps this tile's ramp-down.  The
    commit waits the input load semaphores and — every step, since a map writes
    the output each step — that slot's store semaphore free ("wait on whatever
    we write"; the store-sem is seeded with a credit so a slot's first use
    never underflows).
    """

    def inner(grid_index, *args):
        end = len(args) - num_scratch
        in_slots = args[: end - num_outputs]
        out_slots = args[end - num_outputs : end]
        results = compute(*in_slots, *args[end:])
        if results is None:
            # The compute wrote the output slots itself (a CSR-producing
            # epilogue): nothing left to store.
            return
        if not isinstance(results, (tuple, list)):
            results = (results,)
        for slot, value in zip(out_slots, results):
            voyager.insert(value, slot)

    if not async_pipeline:
        return inner

    def kernel(
        grid_index, in_slots, out_slots, scratch, in_sems, out_sems, post
    ):
        operands = [grid_index, *in_slots, *out_slots, *scratch]
        commit(inner, operands, dependencies=[*in_sems, *out_sems], post=post)

    return kernel


def _reduction_inplace_kernel(
    compute: Callable, reduction_dim: int, async_pipeline: bool = False
):
    """Kernel for a reduction with nothing left to do once it completes — no
    cast (the accumulator's dtype is the output's) and no fused tail.  It
    accumulates straight into the output slot, so there is no scratch ref and no
    finalize step: the completed tile is already in the slot the store reads.

    ``_reduction_fused_kernel`` is the general case, where a cast or a tail must
    map the accumulator into the slot and so needs one of its own.

    ``async_pipeline=True`` returns the :class:`AsyncPipelinedKernel` variant:
    the accumulator *is* the output slot, first written on the ``0`` reduction
    coord, so the commit dispatch waits the slot free there.
    """

    def inner(grid_index, *args):
        *in_slots, out_slot = args

        def init():
            return compute(in_slots, True)

        def accumulate(prev=out_slot):
            return compute(in_slots, False) + prev

        voyager.insert(
            torch.cond(grid_index[reduction_dim] == 0, init, accumulate),
            out_slot,
        )

    if not async_pipeline:
        return inner

    # ``init`` writes the output-slot accumulator fresh (waits the slot free),
    # ``accumulate`` RMWs it in place (no output dep).  Split them across the
    # round-0 predicate so neither commit body carries the now-redundant
    # ``K == 0`` inner cond.
    def init_body(grid_index, *args):
        *in_slots, out_slot = args
        voyager.insert(compute(in_slots, True), out_slot)

    def accumulate_body(grid_index, *args):
        *in_slots, out_slot = args
        voyager.insert(compute(in_slots, False) + out_slot, out_slot)

    def kernel(
        grid_index, in_slots, out_slots, scratch, in_sems, out_sems, post
    ):
        operands = [grid_index, *in_slots, *out_slots, *scratch]

        def on_first():
            commit(
                init_body,
                operands,
                dependencies=[*in_sems, *out_sems],
                post=post,
            )
            return 0

        def not_first():
            commit(
                accumulate_body, operands, dependencies=[*in_sems], post=post
            )
            return 0

        torch.cond(grid_index[reduction_dim] == 0, on_first, not_first)

    return kernel


def _split_stream_break(fused_gm):
    """Split a fused tail whose ``quantize_mx`` cannot ride the compute pass.

    A ``quantize_mx`` along a non-last axis takes two vector-op passes, so
    its input must be materialized first; behind a relayout it cannot ride
    either — the hardware does not quantize a permuted stream.  Either way
    the quantize runs as a pass of its own on the staged tile.  The cut
    lands *before* the relayout chain feeding the quantize: the staged tile
    keeps the accumulator's layout and the relayout rides the quantize pass
    as addressing.  (Staging the permuted head in place would be a data
    hazard — a store order that no longer matches the accumulator-read
    order overwrites elements the pass has not read yet.)  Assumes the
    tail's ``quantize_mx``, when present, is the last op, with the relayout
    (if any) immediately before it.

    Args:
        fused_gm: The fused tail ``[acc, *operands] -> output(s)``.

    Returns:
        ``None`` when the tail has no such quantize (run ``fused_gm`` whole),
        else ``(head_gm, quant_gm)``: the tail up to the relayout chain, and
        the relayout + quantize applied to the staged tile, each taking the
        same operand list as ``fused_gm``.
    """
    graph = fused_gm.graph
    phs = [n for n in graph.nodes if n.op == "placeholder"]
    ops = [n for n in graph.nodes if n.op == "call_function"]
    quant = ops[-1] if ops else None
    if (
        quant is None
        or quant.target is not torch.ops.quantized_ops.quantize_mx.default
    ):
        return None

    # The relayout chain feeding the quantize: contiguous transpose /
    # permute / view members on the spine, walked input-ward.
    relayout = []
    spine = quant.args[0]
    while (
        spine.op == "call_function"
        and (is_reshape_op(spine) or is_nop(spine))
        and len(spine.users) == 1
    ):
        relayout.append(spine)
        spine = spine.args[0]
    permuted = any(is_reshape_op(n) for n in relayout)
    if not permuted and all(a == -1 for a in get_arg_value(quant, 2, "axes")):
        return None

    def build(part_ops, root, name, out):
        pg = torch.fx.Graph()
        remap = {root: pg.placeholder(name)}
        for p in phs[1:]:
            remap[p] = pg.placeholder(p.name)
        for n in part_ops:
            remap[n] = pg.node_copy(n, lambda x: remap[x])
        pg.output(remap[out])
        pg.lint()
        return torch.fx.GraphModule(torch.nn.Module(), pg)

    relayout.reverse()
    head_ops = [n for n in ops[:-1] if n not in relayout]
    head_gm = build(head_ops, phs[0], "acc", spine)
    quant_gm = build(relayout + [quant], spine, "staged", quant)
    return head_gm, quant_gm


def _reduction_fused_kernel(
    compute: Callable,
    reduction_dim: int,
    last_idx: int,
    op_dtype: Optional[torch.dtype],
    num_outputs: int,
    fused_gm: Optional[Callable],
    chain_tail: bool,
    fused_operand_indices: List[int] = (),
    scratch_slots: int = 1,
    async_pipeline: bool = False,
):
    """Kernel for an op whose reduction needs > 1 tile (num_k > 1 GEMM / conv;
    the num_k == 1 map case uses ``_map_kernel``).

    The partial accumulates into a scratch ref; the last step casts it to
    ``op_dtype`` and maps it through the fused tail (if any) into the out
    slot(s).  The bias rides only the first step — the same step that
    initializes the accumulator — so bias gate and reduction init share the
    single reduction ``torch.cond``.

    The async variant dispatches each round as a ``commit``.  Its finalize
    runs the tail chained on the live value (``chain_tail``), or reads the
    completed accumulator back from scratch: an unchained tail, or a chained
    one whose stream-breaking ``quantize_mx`` (``_split_stream_break``)
    reads the staged tile in the accumulator's own layout, the relayout (if
    any) folded into that pass.  A
    scratch re-read races the next sweep's first accumulate, and
    ``scratch_slots`` picks how the race is closed: with 1 the reading tail
    runs bare after the ``commit`` call, so program order holds the next
    round-0 commit until it is done — an array bubble per sweep boundary;
    with 2 consecutive tiles alternate scratch slots and the tail stays
    inside the commit — no bubble, one extra accumulator tile of SRAM.

    Args:
        compute: ``compute(in_slots, first)`` — the bare op on the current
            tiles; ``first`` folds the bias straight into the op.
        reduction_dim: Grid dim of the cross-tile reduction (K / C).
        last_idx: Final coordinate along ``reduction_dim`` (``num_k - 1``).
        op_dtype: Output dtype the finished accumulator is cast to, or
            ``None`` when it already accumulates in the output dtype.
        num_outputs: Number of output tiles the kernel writes.
        fused_gm: The fused tail ``[acc, *operands] -> output(s)``, or
            ``None``.
        chain_tail: Finalize the fused tail on the live accumulated value
            instead of materializing it into scratch first
            (``meta['accumulate_fusible']``); ignored by the sync kernel.
        fused_operand_indices: Positions of the tail's extra operands in the
            kernel's input-slot list.
        scratch_slots: Slot count of the scratch accumulator (see above);
            only 1 and 2 are meaningful, and only for the async variant.
        async_pipeline: Return the :class:`AsyncPipelinedKernel` variant
            instead of the synchronous body.

    Returns:
        The kernel callable for ``build_pipelined_buffers``.
    """

    def _split(args):
        # args = [*in_slots, *out_slots, scratch]; one scratch accumulator.
        n_in = len(args) - num_outputs - 1
        return args[:n_in], args[n_in : n_in + num_outputs], args[-1]

    def _to_acc(result, scratch):
        # Upcast a partial to the (possibly wider, e.g. fp32) accumulator dtype.
        return result if op_dtype is None else result.to(scratch.dtype)

    def _accumulate(grid_index, in_slots, scratch):
        """Fold this round's partial into the scratch accumulator: the op with
        bias on the first coord (initializing it), the bare op plus the running
        accumulator after."""

        def init():
            return _to_acc(compute(in_slots, True), scratch)

        def accumulate(prev=scratch):
            return _to_acc(compute(in_slots, False), scratch) + prev

        voyager.insert(
            torch.cond(grid_index[reduction_dim] == 0, init, accumulate),
            scratch,
        )

    def _finalize(in_slots, out_slots, scratch):
        """Cast the completed accumulator, apply the fused tail once, and store
        each output.  A tail that writes the output slots itself returns
        ``None``, leaving nothing to store."""
        outs = scratch if op_dtype is None else scratch.to(op_dtype)
        if fused_gm is not None:
            fused = [in_slots[i] for i in fused_operand_indices]
            outs = fused_gm(outs, *fused)
            if outs is None:
                return
        if not isinstance(outs, (tuple, list)):
            outs = (outs,)
        for slot, out in zip(out_slots, outs):
            voyager.insert(out, slot)

    def inner(grid_index, *args):
        in_slots, out_slots, scratch = _split(args)
        _accumulate(grid_index, in_slots, scratch)
        effect_cond(
            grid_index[reduction_dim] == last_idx,
            lambda: _finalize(in_slots, out_slots, scratch),
        )

    if not async_pipeline:
        return inner

    split = (
        _split_stream_break(fused_gm)
        if chain_tail and fused_gm is not None
        else None
    )

    def accumulate_body(grid_index, *args):
        in_slots, _out_slots, scratch = _split(args)
        _accumulate(grid_index, in_slots, scratch)

    def _staged_quantize(in_slots, out_slots, scratch):
        """The split-off pass: relayout + quantize the staged tile straight
        out of scratch and write the out slots.  The staged tile keeps the
        accumulator's own shape (the head is layout-preserving)."""
        fused = [in_slots[i] for i in fused_operand_indices]
        outs = split[1](scratch, *fused)
        for slot, out in zip(out_slots, outs):
            voyager.insert(out, slot)

    def finalize_body(grid_index, *args):
        in_slots, out_slots, scratch = _split(args)
        total = _to_acc(compute(in_slots, False), scratch) + scratch
        if split is None and chain_tail:
            _finalize(in_slots, out_slots, total)
        elif not chain_tail:
            voyager.insert(total, scratch)
            if scratch_slots > 1:
                _finalize(in_slots, out_slots, scratch)
        else:
            # Layout-preserving chained head; the split-off relayout +
            # quantize pass reads the staged tile.
            fused = [in_slots[i] for i in fused_operand_indices]
            head = split[0](total, *fused)
            voyager.insert(head.reshape(scratch.shape), scratch)
            if scratch_slots > 1:
                _staged_quantize(in_slots, out_slots, scratch)

    def kernel(
        grid_index, in_slots, out_slots, scratch, in_sems, out_sems, post
    ):
        operands = [grid_index, *in_slots, *out_slots, *scratch]

        def on_last():
            commit(
                finalize_body,
                operands,
                dependencies=[*in_sems, *out_sems],
                post=post,
            )
            # The scratch-reading tail, bare outside the commit (see header);
            # a double-buffered scratch keeps it in the commit instead.
            if scratch_slots == 1:
                if not chain_tail:
                    _finalize(in_slots, out_slots, scratch[0])
                elif split is not None:
                    _staged_quantize(in_slots, out_slots, scratch[0])
            return 0

        def not_last():
            commit(
                accumulate_body, operands, dependencies=[*in_sems], post=post
            )
            return 0

        torch.cond(grid_index[reduction_dim] == last_idx, on_last, not_last)

    return kernel


def _single_buffer_reduction_operands(in_specs, out_specs, fused_idx):
    """A >1-tile reduction writes / consumes these operands only on the last K
    step — the output (``post_process``) and the fused post-op operands — so
    they gain nothing from double buffering.  Single-buffer them (halving their
    SRAM) and defer their wait to block-exit (``first_use_at_exit``) so the
    load / store still overlaps the reduction.  Bias is excluded — it is folded
    on the *first* K step, so it stays double-buffered (prefetched, no stall).
    Mutates the passed specs in place; call only when ``num_k > 1``.
    """
    for s in out_specs:
        s.num_slots = 1
        s.first_use_at_exit = True
    for i in fused_idx:
        if in_specs[i] is not None:
            in_specs[i].num_slots = 1
            in_specs[i].first_use_at_exit = True


def _gemm_scratch_and_kernel(
    gemm_kernel,
    fused_gm,
    *,
    reduction_dim,
    num_k,
    acc_shape,
    in_specs,
    out_specs,
    fused_idx,
    anchor,
    accumulate_fp32,
    chain_tail,
    async_pipeline,
    single_buffer_tail=False,
):
    """The scheduler kernel and scratch specs a tiled reduction op needs —
    shared by the dense / sparse GEMM and conv builders.

    Args:
        gemm_kernel: The per-step op call, ``(in_tiles, first) -> tile``.
        fused_gm: The tail to apply, or ``None``.
        reduction_dim: Grid dim the cross-tile reduction runs along.
        num_k: Reduction steps per output tile.
        acc_shape: The accumulator tile's shape.
        in_specs: One block spec per operand; ``single_buffer_tail``
            mutates these in place.
        out_specs: The group's outputs.
        fused_idx: Operand indices the fused tail consumes after the op.
        anchor: The op node, carrying the tiler's ``scratch_slots`` meta.
        accumulate_fp32: Accumulate the reduction in fp32 rather than the
            output dtype.
        chain_tail: Whether the tail chains into the async finalize pass.
        async_pipeline: The async kernel variants, or the sync ones.
        single_buffer_tail: Collapse a sync reduction's operand banks to one.

    Returns:
        ``(scratch_specs, kernel)``.
    """

    def compute(*in_tiles):
        # num_k == 1 map: the op in one step, then the fused tail (if any).
        result = gemm_kernel(in_tiles, True)
        if fused_gm is not None:
            return fused_gm(result, *[in_tiles[i] for i in fused_idx])
        return result

    out_dtype = anchor.value.dtype
    acc_dtype = torch.float32 if accumulate_fp32 else out_dtype
    if num_k == 1:
        scratch_specs = []
        kernel = _map_kernel(
            compute, len(out_specs), async_pipeline=async_pipeline
        )
    elif fused_gm is None and acc_dtype == out_dtype:
        scratch_specs = []
        kernel = _reduction_inplace_kernel(
            gemm_kernel,
            reduction_dim=reduction_dim,
            async_pipeline=async_pipeline,
        )
    else:
        if single_buffer_tail and not async_pipeline:
            _single_buffer_reduction_operands(in_specs, out_specs, fused_idx)
        # Only a tail that reads scratch after the streamed pass gains a
        # second slot; a chained-whole tail never races and keeps one.  The
        # count is the tiler's per-node call (the scratch ladder winner,
        # stamped with the tiling).
        sync = not chain_tail or (
            fused_gm is not None and _split_stream_break(fused_gm) is not None
        )
        scratch_slots = anchor.meta.get("tiling", {}).get("scratch_slots", 1)
        slots = scratch_slots if sync else 1
        scratch_specs = [_ScratchSpec(acc_shape, acc_dtype, num_slots=slots)]
        kernel = _reduction_fused_kernel(
            gemm_kernel,
            reduction_dim=reduction_dim,
            last_idx=num_k - 1,
            num_outputs=len(out_specs),
            op_dtype=(out_dtype if acc_dtype != out_dtype else None),
            fused_gm=fused_gm,
            chain_tail=chain_tail,
            fused_operand_indices=fused_idx,
            scratch_slots=slots,
            async_pipeline=async_pipeline,
        )
    return scratch_specs, kernel


def _stamp_anchor_meta(gm, anchor) -> None:
    """Copy the anchor's interstellar results -- the per-tile compute cycles the
    reporting model turns into a utilization, and the mapping / architecture the
    proto emitter turns into a ``Tiling`` -- onto the nest just built, at every
    nesting level (loop body, cond branch); the anchor itself is erased on
    splice.
    """
    if (tiling := anchor.meta.get("tiling")) is None:
        return
    for m in gm.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        named = dict(m.named_modules())
        for n in m.graph.nodes:
            sub = named.get(n.target) if n.op == "call_module" else None
            if n.target is anchor.target or (
                isinstance(sub, torch.fx.GraphModule)
                and any(x.target is anchor.target for x in sub.graph.nodes)
            ):
                n.meta.update(tiling)


def build_conv2d(
    node,
    *,
    num_slots: int = _DEFAULT_NUM_SLOTS,
    accumulate_fp32: bool = False,
    single_buffer_tail: bool = False,
    async_pipeline: bool = True,
    tiler=None,
):
    """Pipeline builder for a conv2d (groups=1) node — incl. the microscaling /
    codebook (``conv2d_mx``) variant, a fused bias, and the systolic NHWC layout
    — over the input-channel (C) cross-tile reduction.  A map over the (N, K,
    oH, oW) output grid plus a C reduction dim: the input is a strided
    receptive-field halo (pad-on-load), the weight is tiled on (K, C), and the
    kernel convolves each C-block and accumulates.  Grid ``(N, K, oH, oW, C)``;
    for ``num_k == 1`` the C dim is extent 1.  Specs are logical NCHW/OIHW and
    projected onto each operand's physical order (``meta["transposed"]`` selects
    NHWC + HWIO).  Returns the gm or ``None``.
    """
    info = parse_fused_submodule(node, tiler)
    if info is not None:
        tiling, l3_order = info.tiling, info.l3_order
    else:
        tiling, l3_order = get_tiling(node, tiler)
    if info is None and tiling is None:
        return None
    anchor = info.anchor if info is not None else node

    inp = anchor.args[0].value.clone()
    w = anchor.args[1].value.clone()
    out = anchor.value  # the conv output (drives the N/K/oH/oW grid)
    if inp.ndim != 4 or w.ndim != 4:
        return None
    groups = get_arg_value(anchor, 6, "groups", 1)

    nhwc = anchor.meta.get("transposed", False)
    in_dims = NCHW_TO_NHWC if nhwc else None
    # The layout pass never permutes a grouped conv's weight, so it stays
    # OIHW even on a transposed node.
    w_dims = OIHW_TO_HWIO if nhwc and groups == 1 else None
    out_dims = NCHW_TO_NHWC if nhwc else None

    N, C, H, W = unproject(inp.shape, in_dims)
    K, wC, kH, kW = unproject(w.shape, w_dims)
    oH, oW = unproject(out.shape, out_dims)[2:]

    # A grouped conv's C is not a dense reduction, so it cannot be diced --
    # build it whole-tensor (a trip-1 nest).
    if groups != 1 or tiling is None:
        tiling = (1, 1, 1, 1)
    ny, nx, nk, nc = tiling
    tn, toh, tow, tc, tk = N, oH // ny, oW // nx, C // nc, K // nk
    num_k = nc

    sh, sw = _pair(get_arg_value(anchor, 3, "stride", 1))
    ph, pw = _pair(get_arg_value(anchor, 4, "padding", 0))
    dh, dw = _pair(get_arg_value(anchor, 5, "dilation", 1))
    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1

    l3_order = l3_order or CONV_L3_ORDER
    gN, gC = 0, 4
    gK, gY, gX = (1 + l3_order.index(d) for d in CONV_L3_ORDER)
    counts = {"K": nk, "Y": ny, "X": nx}
    grid = (1,) + tuple(counts[d] for d in l3_order) + (nc,)
    in_code = anchor.kwargs.get("input_code")
    pad_value = (
        float(in_code.value.abs().argmin())
        if isinstance(in_code, torch.fx.Node)
        else 0.0
    )
    in_spec = _InputSpec(
        project((tn, tc, ih, iw), in_dims),
        project((gN, gC, gY, gX), in_dims),  # logical N, C, H, W
        (False,) * 4,
        strides=project((tn, tc, toh * sh, tow * sw), in_dims),
        pad=project((0, 0, ph, pw), in_dims),
        pad_value=pad_value,
    )
    w_spec = _InputSpec(
        project((tk, wC // nc, kH, kW), w_dims),
        # kH/kW->None (loaded whole, mapped to no grid dim)
        project((gK, gC, None, None), w_dims),
        (False,) * 4,
    )
    bias_spec = _InputSpec((tk,), (gK,), (False,))
    # The output(s) tile onto the (N, K, oH, oW) grid dims (C reduction
    # dropped); a fused op may produce several (``quantize_mx``).
    out_index_map = project((gN, gK, gY, gX), out_dims)
    if info is None:
        out_specs = [
            _OutputSpec(
                project((N, K, oH, oW), out_dims),
                project((tn, tk, toh, tow), out_dims),
                out_index_map,
                inp.dtype,
            )
        ]
    else:
        out_specs = list(info.out_specs)

    src = lambda n: n.meta.get("source_node", n)
    node_to_spec = {
        src(anchor.args[0]): (inp, in_spec),
        src(anchor.args[1]): (w, w_spec),
    }

    bias_n = get_arg_value(anchor, 2, "bias")
    if bias_n is not None:
        node_to_spec[src(bias_n)] = (bias_n.value.clone(), bias_spec)

    target = anchor.target
    bs = anchor.kwargs.get("block_size")
    scalar_kwargs = {
        k: v
        for k, v in anchor.kwargs.items()
        if not isinstance(v, torch.fx.Node)
    }
    kw_nodes = {}

    def add_kw_input(name: str, spec: _InputSpec | None) -> None:
        v = anchor.kwargs.get(name)
        if not isinstance(v, torch.fx.Node):
            return
        if not hasattr(v, "value"):
            raise ValueError(
                f"Expected materialized value for FX node kwarg {name!r}"
            )
        kw_nodes[name] = src(v)
        node_to_spec[src(v)] = (v.value.clone(), spec)

    if target == torch.ops.quantized_ops.conv2d_mx.default:
        in_scale_qspec = _InputSpec(
            project((tn, tc // bs, ih, iw), in_dims),
            project((gN, gC, gY, gX), in_dims),
            (False,) * 4,
            strides=project((tn, tc // bs, toh * sh, tow * sw), in_dims),
            pad=project((0, 0, ph, pw), in_dims),
            pad_value=1.0,
        )
        # A grouped weight has a single (or partial) in-channel, so its
        # scale's own shape names the channel / kernel extents the dense
        # arithmetic would misderive from the input's.
        ws_tile = (tk, tc // bs, kH, kW)
        ws = anchor.kwargs.get("weight_scale")
        if groups != 1 and isinstance(ws, torch.fx.Node):
            s0, s1, s2, s3 = unproject(ws.value.shape, w_dims)
            ws_tile = (s0 // nk, s1, s2, s3)
        wt_scale_qspec = _InputSpec(
            project(ws_tile, w_dims),
            project((gK, gC, None, None), w_dims),  # kH/kW whole -> None
            (False,) * 4,
        )
        add_kw_input("input_scale", in_scale_qspec)
        add_kw_input("weight_scale", wt_scale_qspec)
        add_kw_input("input_code", None)
        add_kw_input("weight_code", None)

    # Fused post-op operands (a residual, …), keyed by their outer node.
    if info is not None:
        for s, spec in zip(info.in_nodes, info.in_specs):
            node_to_spec[s] = (s.value.clone(), spec)

    order = {n: i for i, n in enumerate(node.all_input_nodes)}
    assert len(node_to_spec) == len(order), "conv operand shared across roles"
    inputs = [node_to_spec[n][0] for n in node.all_input_nodes]
    in_specs = [node_to_spec[n][1] for n in node.all_input_nodes]

    in_idx = order[src(anchor.args[0])]
    w_idx = order[src(anchor.args[1])]
    bias_idx = order[src(bias_n)] if bias_n is not None else None
    kw_idx = {name: order[n] for name, n in kw_nodes.items()}
    fused_idx = [order[s] for s in info.in_nodes] if info is not None else []
    fused_gm = info.fused_gm if info is not None else None

    def _conv(in_tile, w_tile, bias, kw):
        return target(
            in_tile, w_tile, bias, [sh, sw], [0, 0], [dh, dw], groups, **kw
        )

    def _fix_stride(t):
        # ``torch.cond`` rejects a branch output whose stride it can't prove is
        # a product of sizes; a strided conv tile's spatial extent is symbolic
        # with a ``Max(1, .)`` clamp (kept by ``.to`` / ``.contiguous`` / view).
        # Re-view with the concrete output tile shape (``tn, tk, toh, tow``) and
        # its dense stride via ``as_strided`` (a metadata-only NOP, in
        # ``is_nop``).  Needed independently of the accumulator cast.
        d0, d1, d2, d3 = project((tn, tk, toh, tow), out_dims)
        return torch.as_strided(
            t, size=(d0, d1, d2, d3), stride=(d1 * d2 * d3, d2 * d3, d3, 1)
        )

    def conv2d_kernel(in_tiles, first):
        """The bare conv op on the current tiles, dense-strided
        (``_fix_stride``) so it can feed a ``torch.cond`` branch.  On the
        ``first`` step the [K] bias folds straight into the op (conv + bias in
        one hardware pass); later steps only accumulate partials, which must not
        re-add it.
        """
        in_tile = in_tiles[in_idx]
        w_tile = in_tiles[w_idx]
        kw = {name: in_tiles[i] for name, i in kw_idx.items()}
        kw.update(scalar_kwargs)  # block_size / weight_layout / ...
        bias = in_tiles[bias_idx] if (bias_idx is not None and first) else None
        return _fix_stride(_conv(in_tile, w_tile, bias, kw))

    scratch_specs, kernel = _gemm_scratch_and_kernel(
        conv2d_kernel,
        fused_gm,
        reduction_dim=gC,
        num_k=num_k,
        acc_shape=project((tn, tk, toh, tow), out_dims),
        in_specs=in_specs,
        out_specs=out_specs,
        fused_idx=fused_idx,
        anchor=anchor,
        accumulate_fp32=accumulate_fp32,
        chain_tail=node.meta.get("accumulate_fusible", False),
        async_pipeline=async_pipeline,
        single_buffer_tail=single_buffer_tail,
    )
    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs,
        tuple(inputs),
        scratch_specs=scratch_specs,
        num_slots=num_slots,
        async_pipeline=async_pipeline,
    )
    if num_k > 1 or info is not None:
        outline_dps_ops(gm)
    _stamp_anchor_meta(gm, anchor)
    return gm


@dataclass
class _GemmPlan:
    """What a GEMM node determines before any kernel exists: the grid it sweeps,
    its operands and their block specs in splice order, and the bare GEMM call
    on one step's tiles.

    Attributes:
        anchor: The GEMM node itself, or the anchor of the fused group.
        grid: Tile counts, batch dims then M/N in ``l3_order`` then K.
        grid_dims: Grid dims the row / column / reduction tiles run along.
        tile_k: Columns one reduction step spans.
        acc_shape: The accumulator tile, batch dims then ``(tile_m, tile_n)``.
        inputs: Operand values in ``node.all_input_nodes`` order.
        in_specs: One block spec per operand, ``None`` for a whole-tensor one.
        out_specs: The group's outputs.
        kw_idx: Operand index of each tensor kwarg the op takes.
        gemm_kernel: ``(in_tiles, first) -> tile``, the GEMM (plus any weight
            decode) on one step; ``first`` says whether the bias folds in.
        fused_gm: The fused tail, or ``None``.
        fused_idx: Operand indices the fused tail consumes after the GEMM.
        outline_dps: Whether the built gm needs ``outline_dps_ops``.
    """

    anchor: torch.fx.Node
    grid: Tuple[int, ...]
    grid_dims: Tuple[int, int, int]
    tile_k: int
    acc_shape: Tuple[int, ...]
    inputs: Tuple[torch.Tensor, ...]
    in_specs: List[Optional[_InputSpec]]
    out_specs: List[_OutputSpec]
    kw_idx: dict
    gemm_kernel: Callable
    fused_gm: Optional[torch.nn.Module]
    fused_idx: List[int]
    outline_dps: bool

    @property
    def tile_m(self) -> int:
        """Rows one output tile spans."""
        return self.acc_shape[-2]

    @property
    def tile_n(self) -> int:
        """Columns one output tile spans."""
        return self.acc_shape[-1]

    @property
    def num_k(self) -> int:
        """Reduction steps per output tile."""
        return self.grid[self.grid_dims[2]]

    @property
    def out_dtype(self) -> torch.dtype:
        """Element type of the GEMM's own output -- the accumulator's."""
        return self.anchor.value.dtype


def _gemm_plan(node, tiler=None, k_tiles=None) -> Optional[_GemmPlan]:
    """Derive a GEMM node's grid, operands and per-step call — everything a
    builder needs before it picks a kernel.  Returns ``None`` for a node that
    has no tiling and is not a BMM.

    Operands are assembled in the fused node's ``all_input_nodes`` order so the
    positional splice in ``replace_node_with_graph_module`` binds each
    placeholder correctly even when a fused-tail operand is graph-ordered before
    the anchor; ``gemm_kernel`` then dispatches by canonical index (``act_idx``
    / ``kw_idx`` / ``fused_idx``), not a positional ``*extra`` split.

    Args:
        node: The linear / matmul / batched-matmul node, or the fused group.
        tiler: Tile-search backend, forwarded to ``get_tiling``.
        k_tiles: Reduction-tile count to use instead of the one the tile
            search picked. A sparse GEMM sets this to its CSR's slice count.

    Returns:
        A ``_GemmPlan``, or ``None``.
    """
    info = parse_fused_submodule(node, tiler)
    if info is not None:
        tiling, l3_order = info.tiling, info.l3_order
    else:
        tiling, l3_order = get_tiling(node, tiler)
    if info is None and tiling is None and not is_bmm(node):
        return None
    anchor = info.anchor if info is not None else node

    weight_node, transposed, weight_repeat, dequant = weight_transforms(
        anchor.args[1]
    )

    act = anchor.args[0].value.clone()
    weight = weight_node.value.clone()
    out = anchor.value  # the GEMM output (drives the M/N/K grid)
    if not isinstance(out, torch.Tensor) or act.ndim < 2 or weight.ndim < 2:
        return None

    M, K, N = act.shape[-2], act.shape[-1], out.shape[-1]
    if tiling is None:
        # A BMM keeps its batch dims as per-element tiles (size 1)
        if is_bmm(anchor):
            out_ts = (1,) * (out.ndim - 2) + tuple(out.shape[-2:])
        else:
            out_ts = tuple(out.shape)
        nk = 1
    else:
        out_tiling, nk = tiling[:-1], tiling[-1]  # batch.. + (n_m, n_n) , n_k
        out_ts = tuple(s // t for s, t in zip(out.shape, out_tiling))
    # A sparse GEMM must split K exactly as its CSR was produced -- column
    # indices are local to a slice and the engine cannot rebase them -- so it
    # overrides the tile search along that one dim.
    if k_tiles is not None:
        nk = k_tiles
    tk = K // nk
    tm, tn = int(out_ts[-2]), int(out_ts[-1])

    nb = out.ndim - 2
    l3_order = l3_order or GEMM_L3_ORDER
    gm, gn = (nb + l3_order.index(d) for d in GEMM_L3_ORDER)
    gk = nb + 2
    out_batch = tuple(out.shape[:nb])
    tb = tuple(int(x) for x in out_ts[:nb])
    counts = {"M": M // tm, "N": N // tn}
    grid = (
        tuple(b // t for b, t in zip(out_batch, tb))
        + tuple(counts[d] for d in l3_order)
        + (K // tk,)
    )

    ck = weight_is_ck(anchor)
    _proj = lambda n, k: (k, n) if ck else (n, k)

    def _batch(shape):
        """An operand's leading-batch ``(tiles, index_map, is_broadcast)``
        right-aligned to the output batch dims; a size-1 batch dim broadcasts
        (pinned to block 0).
        """
        ob = shape[:-2]
        off = nb - len(ob)
        tiles, imap, bcast = [], [], []
        for j, sz in enumerate(ob):
            g = off + j
            b = sz == 1 and out_batch[g] != 1
            tiles.append(1 if b else tb[g])
            imap.append(g)
            bcast.append(b)
        return tiles, imap, bcast

    def _spec(shape, mn_tiles, mn_map):
        """An ``_InputSpec`` for an operand whose batch dims follow ``shape``
        and whose trailing two (M/N and K) dims tile by ``mn_tiles`` onto grid
        dims ``mn_map``.
        """
        bt, bm, bb = _batch(shape)
        return _InputSpec(
            tuple(bt) + tuple(mn_tiles),
            tuple(bm) + tuple(mn_map),
            tuple(bb) + (False, False),
        )

    act_spec = _spec(act.shape, (tm, tk), (gm, gk))
    weight_spec = _spec(weight.shape, _proj(tn, tk), _proj(gn, gk))
    weight_spec.transposed = transposed
    weight_spec.repeat = weight_repeat
    bias_spec = _InputSpec((tn,), (gn,), (False,))
    # The output(s) tile onto the M/N grid dims (K reduction ``gk`` dropped); a
    # fused op may produce several (``quantize_mx``).
    out_index_map = tuple(range(nb)) + (gm, gn)
    if info is None:
        out_specs = [
            _OutputSpec(
                tuple(out.shape), tuple(tb) + (tm, tn), out_index_map, out.dtype
            )
        ]
    else:
        out_specs = list(info.out_specs)

    src = lambda n: n.meta.get("source_node", n)
    node_to_spec = {
        src(anchor.args[0]): (act, act_spec),
        src(weight_node): (weight, weight_spec),
    }

    # Bias [N] tiles along N (grid dim ``gn``); folded once on the k==0 step.
    bias_n = get_arg_value(anchor, 2, "bias")
    if bias_n is not None:
        node_to_spec[src(bias_n)] = (bias_n.value.clone(), bias_spec)

    # Microscaling (linear_mx / matmul_mx): per-block scales tile along the
    # reduction; codebooks load whole (None spec).  Each threads by keyword.
    bs = anchor.kwargs.get("block_size")
    scalar_kwargs = {
        k: v
        for k, v in anchor.kwargs.items()
        if not isinstance(v, torch.fx.Node)
    }
    kw_nodes = {}

    def add_kw_input(name: str, spec: _InputSpec | None) -> None:
        v = anchor.kwargs.get(name)
        if not isinstance(v, torch.fx.Node):
            return
        # A scale wears the same relayouts as the tensor it scales, so it peels
        # the same way.
        v, transposed, repeat, _ = weight_transforms(v)
        if spec is not None:
            spec.transposed = transposed
            spec.repeat = repeat
        if not hasattr(v, "value"):
            raise ValueError(
                f"Expected materialized value for FX node kwarg {name!r}"
            )
        kw_nodes[name] = src(v)
        node_to_spec[src(v)] = (v.value.clone(), spec)

    if anchor.target in (
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ):
        add_kw_input("input_scale", _spec(act.shape, (tm, tk // bs), (gm, gk)))
        add_kw_input(
            "weight_scale",
            _spec(weight.shape, _proj(tn, tk // bs), _proj(gn, gk)),
        )
        add_kw_input("input_code", None)
        add_kw_input("weight_code", None)

    # An outlier CSR rides the GEMM as three kwargs. The row pointers are one
    # continuous array per K slice, so this tile is a row *range* of it: tm+1
    # entries at stride tm, overlapping its neighbour by the boundary entry
    # both share. ``data`` / ``indices`` are a packed stream addressed by
    # values only known at run time, so they stay raw DRAM (``None``) and the
    # sparse builder fetches them by hand.
    if anchor.kwargs.get("A_indptr") is not None:
        ptr_spec = _InputSpec(
            tuple(tb) + (1, tm + 1),
            tuple(range(nb)) + (gk, gm),
            (False,) * (nb + 2),
            strides=tuple(tb) + (1, tm),
        )
        add_kw_input("A_indptr", ptr_spec)
        add_kw_input("A_data", None)
        add_kw_input("A_indices", None)

    # A packed KV cache reaches the GEMM through a ``dequantize``, which decodes
    # the weight *tile* in the kernel -- so the cache is fetched, and paid for,
    # packed.  Its scale / zero point block along one of the weight's own axes,
    # so they dice with it, that axis divided by the block; the codebook, indexed
    # by value rather than by position, loads whole.
    dq_nodes = {}
    if dequant is not None:
        dq_axes = {a % weight.ndim for a in get_arg_value(dequant, 3, "axes")}
        dq_bs = get_arg_value(dequant, 4, "block_size")
        k_dim, n_dim = (
            (weight.ndim - 2, weight.ndim - 1)
            if ck
            else (weight.ndim - 1, weight.ndim - 2)
        )
        dq_tile = _proj(
            tn // dq_bs if n_dim in dq_axes else tn,
            tk // dq_bs if k_dim in dq_axes else tk,
        )
        tables = quant_param_arg_nodes(dequant)
        for i, v in enumerate(dequant.args):
            if i == 0 or not isinstance(v, torch.fx.Node):
                continue
            spec = None
            if v not in tables:
                v, t, r, _ = weight_transforms(v)
                spec = _spec(v.value.shape, dq_tile, _proj(gn, gk))
                spec.transposed = t
                spec.repeat = r
            dq_nodes[i] = src(v)
            node_to_spec[src(v)] = (v.value.clone(), spec)

    # Fused post-op operands (residual, scale, …), keyed by their outer node.
    if info is not None:
        for s, spec in zip(info.in_nodes, info.in_specs):
            node_to_spec[s] = (s.value.clone(), spec)

    order = {n: i for i, n in enumerate(node.all_input_nodes)}
    assert len(node_to_spec) == len(order), (
        f"{node.name}: {len(order) - len(node_to_spec)} operand(s) reached the "
        f"GEMM with no role — every input needs a block spec "
        f"({sorted(n.name for n in order if n not in node_to_spec)})"
    )
    inputs = [node_to_spec[n][0] for n in node.all_input_nodes]
    in_specs = [node_to_spec[n][1] for n in node.all_input_nodes]

    act_idx = order[src(anchor.args[0])]
    weight_idx = order[src(weight_node)]
    bias_idx = order[src(bias_n)] if bias_n is not None else None
    kw_idx = {name: order[n] for name, n in kw_nodes.items()}
    fused_idx = [order[s] for s in info.in_nodes] if info is not None else []
    fused_gm = info.fused_gm if info is not None else None

    # The kernel body is traced, and dynamo refuses to look inside an FX node
    # there — so read the dequantize's call apart here: its scalar args as plain
    # Python, and the tile slot each of its tensor args comes from.
    dq_idx = {i: order[n] for i, n in dq_nodes.items()}
    dq_target = dequant.target if dequant is not None else None
    dq_args = [
        (
            None
            if isinstance(a, torch.fx.Node)
            else tuple(a) if isinstance(a, (list, tuple)) else a
        )
        for a in (dequant.args if dequant is not None else ())
    ]

    op = anchor.target
    num_k = K // tk  # reduction tiles (grid extent along ``gk``)

    def gemm_kernel(in_tiles, first):
        """The bare GEMM op on the current tiles, by canonical index.  On the
        ``first`` reduction step the bias folds straight in; later steps only
        accumulate partials, which must not re-add it.
        """
        act_tile = in_tiles[act_idx]
        weight_tile = in_tiles[weight_idx]
        if dq_target is not None:
            # Decode the packed tile in place of the weight -- the same call the
            # graph made, on tiles.  The group fuses it into the GEMM's kernel,
            # so it stores nothing of its own.
            args = list(dq_args)
            args[0] = weight_tile
            for i, j in dq_idx.items():
                args[i] = in_tiles[j]
            weight_tile = dq_target(*args)
        kw = {name: in_tiles[i] for name, i in kw_idx.items()}
        kw.update(scalar_kwargs)  # block_size / weight_layout / ...
        if bias_idx is None:
            return op(act_tile, weight_tile, **kw)
        bias = in_tiles[bias_idx] if first else None
        return op(act_tile, weight_tile, bias, **kw)

    return _GemmPlan(
        anchor=anchor,
        grid=grid,
        grid_dims=(gm, gn, gk),
        tile_k=tk,
        acc_shape=tuple(tb) + (tm, tn),
        inputs=tuple(inputs),
        in_specs=in_specs,
        out_specs=out_specs,
        kw_idx=kw_idx,
        gemm_kernel=gemm_kernel,
        fused_gm=fused_gm,
        fused_idx=fused_idx,
        outline_dps=num_k > 1 or info is not None or dequant is not None,
    )


def build_gemm(
    node,
    *,
    num_slots: int = _DEFAULT_NUM_SLOTS,
    accumulate_fp32: bool = False,
    single_buffer_tail: bool = False,
    async_pipeline: bool = True,
    tiler=None,
):
    """Pipeline builder for a linear / matmul / batched-matmul node — incl. the
    microscaling / codebook (``*_mx``) variants and a fused bias — over the
    cross-tile K reduction.  Grid ``(M, N, K)`` (or ``(B, M, N, K)``) tiles with
    K innermost; the kernel accumulates ``act_tile @ weight_tile`` into the
    output slot.  Returns the gm, or ``None``.
    """
    plan = _gemm_plan(node, tiler)
    if plan is None:
        return None

    scratch_specs, kernel = _gemm_scratch_and_kernel(
        plan.gemm_kernel,
        plan.fused_gm,
        reduction_dim=plan.grid_dims[2],
        num_k=plan.num_k,
        acc_shape=plan.acc_shape,
        in_specs=plan.in_specs,
        out_specs=plan.out_specs,
        fused_idx=plan.fused_idx,
        anchor=plan.anchor,
        accumulate_fp32=accumulate_fp32,
        chain_tail=node.meta.get("accumulate_fusible", False),
        async_pipeline=async_pipeline,
        single_buffer_tail=single_buffer_tail,
    )
    gm = build_pipelined_buffers(
        kernel,
        plan.grid,
        plan.in_specs,
        plan.out_specs,
        plan.inputs,
        scratch_specs=scratch_specs,
        num_slots=num_slots,
        async_pipeline=async_pipeline,
    )
    if plan.outline_dps:
        outline_dps_ops(gm)
    _stamp_anchor_meta(gm, plan.anchor)
    return gm


def _apply_relayout(node, *seqs, invert=False):
    """Permute each per-dim sequence (an ``index_map``, a tile shape, …) by a
    ``transpose`` / ``permute`` node's dim mapping, returning them in order.

    Forward perm gathers a destination sequence from the source (``dst[k] =
    src[perm[k]]``); ``invert=True`` gathers a source sequence from the
    destination (``build_pointwise`` uses this to turn a standalone relayout's
    output specs into its input load specs).  Handles transpose / permute, args
    or kwargs, negative dims.
    """
    ndim = len(seqs[0])
    if node.target is torch.ops.aten.permute.default:
        perm = [d % ndim for d in get_arg_value(node, 1, "dims")]
    else:  # aten.transpose.int
        d0 = get_arg_value(node, 1, "dim0") % ndim
        d1 = get_arg_value(node, 2, "dim1") % ndim
        perm = list(range(ndim))
        perm[d0], perm[d1] = perm[d1], perm[d0]
    if invert:
        inv = [0] * ndim
        for k, p in enumerate(perm):
            inv[p] = k
        perm = inv
    return tuple(tuple(s[p] for p in perm) for s in seqs)


def _naming_scratch(submod, names):
    """``submod`` with its batched reduction retargeted at the twin that names
    the scratch, taking one extra placeholder per region appended to the
    operands.  Returned unchanged when the group holds no such reduction; the
    original is never mutated (it is the graph's ``meta['submodule']``)."""
    if not names:
        return submod
    gm = copy_graph_module(submod)
    reduction = next(
        (n for n in gm.graph.nodes if reduction_op(n) is not None), None
    )
    if reduction is None:
        return submod

    # The scratch placeholders have to come last in the signature (the kernel
    # passes them after the loaded tiles) *and* before the reduction that reads
    # them.  Export can leave a lifted constant placeholder after the compute,
    # so gather the block to the front before appending to it.
    phs = [p for p in gm.graph.nodes if p.op == "placeholder"]
    anchor = phs[0]
    for ph in phs[1:]:
        anchor.append(ph)
        anchor = ph

    extra = []
    for name in names:
        with gm.graph.inserting_after(anchor):
            anchor = gm.graph.placeholder(f"scratch_{name}")
        extra.append(anchor)

    reduction.target = reduction_op(reduction)
    reduction.kwargs = {**reduction.kwargs, **dict(zip(names, extra))}
    gm.graph.lint()
    gm.recompile()
    return gm


def build_pointwise(node, *, num_slots: int = _DEFAULT_NUM_SLOTS, tiler=None):
    """Pipeline builder for a pointwise / batched-reduction node (elementwise
    ops, layernorm·softmax whose reduction dim is kept whole in the tile, and a
    standalone ``transpose`` / ``permute`` relayout).  Tiles the output grid and
    writes each output tile once (no cross-tile reduction).

    A batched reduction also reserves the on-chip regions its backend kernel
    keeps beside its tiles (``reduction_scratch``); the kernel ignores them, so
    the reservation *is* the declaration.  Returns the gm, or ``None``.
    """
    anchor = get_anchor_node(node)
    tiling = vector_op_tiling(node, tiler.config)
    if node.op != "call_module" and tiling is None:
        return None

    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]

    val = node.value
    outputs = val if isinstance(val, (list, tuple)) else (val,)

    output_shape = tuple(outputs[-1].shape)
    if tiling is None:
        tiling = (1,) * len(output_shape)
    grid = tuple(tiling)
    # ``compute_output_tiled_shapes`` dices each output by ``tiling``, with the
    # sparse-output handling a per-output ``compute_tiled_shape`` would miss.
    tiled_shape = compute_output_tiled_shapes(node, tiling)
    tiled_shape = (
        list(tiled_shape)
        if isinstance(node.value, (list, tuple))
        else [tiled_shape]
    )
    # The regions the backend keeps between its passes.  The kernel hands them
    # to the op, which names them and never reads them.
    scratch = reduction_scratch(node, tiled_shape, tiler.config.vector_lanes)
    scratch_names = [name for name, _, _ in scratch]

    if node.op == "call_module":
        submod = node.meta.get("submodule")
        if not isinstance(submod, torch.fx.GraphModule):
            return None
        # Codebook operands (whole): map the submodule's codebook placeholders
        # back to their outer input nodes.
        codebooks = set()
        for sn in submod.graph.nodes:
            if sn.op != "call_function":
                continue
            for cb in quant_param_arg_nodes(sn):
                codebooks.add(cb.meta.get("source_node", cb))

        compute = _naming_scratch(submod, scratch_names)

    else:
        # Resolve each op arg to a loaded-tile index (tensor operand) or a plain
        # constant *now* — the closure runs in the traced while_loop body, where
        # dynamo rejects FX-Node lookups.
        order = {n: i for i, n in enumerate(in_nodes)}
        _plain = lambda a: list(a) if isinstance(a, list) else a
        arg_slots = [
            order[a] if isinstance(a, torch.fx.Node) else None
            for a in node.args
        ]
        kw_slots = {
            k: order[v] if isinstance(v, torch.fx.Node) else None
            for k, v in node.kwargs.items()
        }
        op_args = [_plain(a) for a in node.args]
        op_kwargs = {k: _plain(v) for k, v in node.kwargs.items()}
        # A batched reduction is traced against the twin that names its
        # scratch, which takes the same operands plus one keyword per region.
        op = reduction_op(node) or node.target
        codebooks = quant_param_arg_nodes(node)
        n_scratch = len(scratch_names)

        def compute(*tiles):
            end = len(tiles) - n_scratch
            args = [
                tiles[i] if i is not None else a
                for i, a in zip(arg_slots, op_args)
            ]
            kwargs = {
                k: tiles[i] if i is not None else op_kwargs[k]
                for k, i in kw_slots.items()
            }
            kwargs.update(zip(scratch_names, tiles[end:]))
            return op(*args, **kwargs)

    if node.target in (
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ):
        # Standalone transpose / permute: store each output tile identity, but
        # load the single input from the transposed source (input dim ``j``
        # tiles along the grid dim its output image occupies — inverse perm),
        # and ``compute = op(tile)`` does the actual transpose.
        ndim = len(output_shape)
        in_tile, in_imap = _apply_relayout(
            node, tiled_shape[0], tuple(range(ndim)), invert=True
        )
        in_specs = [_InputSpec(in_tile, in_imap, (False,) * ndim)]
    else:
        in_specs = [
            (
                _compute_input_spec(tiling, tuple(n.shape))
                if n not in codebooks
                else None
            )
            for n in in_nodes
        ]
    out_specs = [
        _OutputSpec(tuple(o.shape), ts, tuple(range(o.ndim)), o.dtype)
        for o, ts in zip(outputs, tiled_shape)
    ]
    scratch_specs = [_ScratchSpec(shape, dtype) for _, shape, dtype in scratch]
    kernel = _map_kernel(compute, len(outputs), len(scratch_specs))
    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs,
        tuple(inputs),
        scratch_specs=scratch_specs,
        num_slots=num_slots,
    )

    if node.op == "call_module" and anchor is not None:
        outline_dps_ops(gm)
    return gm


_POOL2D_SUPPORTED = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.quantized_ops.max_pool2d.default,
}


def build_pool(node, *, num_slots: int = _DEFAULT_NUM_SLOTS, tiler=None):
    """Pipeline builder for a 2-D max/avg pool node, bare or fused with post-op
    pointwise ops: a map over the (N, C, oH, oW) output grid whose input tile is
    a strided receptive-field halo (boundary padding folded into the load), so
    the kernel pools each halo with ``padding=0``.

    Pool has no cross-tile reduction, so a fused submodule needs no anchor /
    tail split (unlike conv / gemm): the whole submodule is the per-tile
    compute (as in ``build_pointwise``), differing only in that the pool's own
    input loads the halo while the tail operands tile at the output block.
    Returns the gm, or ``None``.
    """
    anchor = get_anchor_node(node)
    if anchor.target not in _POOL2D_SUPPORTED:
        return None

    tiling = pool_op_tiling(node, tiler.config)
    if node.op != "call_module" and tiling is None:
        return None

    in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
    input_t = in_node.value.clone()

    val = node.value
    outputs = val if isinstance(val, (list, tuple)) else (val,)
    output_shape = tuple(outputs[-1].shape)

    in_dims = NCHW_TO_NHWC if anchor.meta.get("transposed", False) else None
    N, C, H, W = unproject(output_shape, in_dims)
    if tiling is None:
        nN, nH, nW, nC = 1, 1, 1, 1
    else:
        nN, nH, nW, nC = tiling
    tn, tc, toh, tow = N // nN, C // nC, H // nH, W // nW
    output_ts = project((tn, tc, toh, tow), in_dims)
    out_tiling = project((nN, nC, nH, nW), in_dims)

    # Geometry params, to size the halo / output / strides.  Only ``max_pool``
    # has a dilation arg; ``avg_pool``'s is implicitly 1.
    is_max = "max_pool" in str(anchor.target)
    kernel_size = get_arg_value(anchor, 1, "kernel_size")
    stride = get_arg_value(anchor, 2, "stride", [])
    padding = get_arg_value(anchor, 3, "padding", 0)
    dilation = get_arg_value(anchor, 4, "dilation", 1) if is_max else 1
    pad_value = float("-inf") if is_max else 0.0

    kH, kW = _pair(kernel_size)
    sh, sw = _pair(stride) if stride else (kH, kW)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)

    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1
    step_h, step_w = toh * sh, tow * sw

    grid = out_tiling
    in_spec = _InputSpec(
        tile_sizes=project((tn, tc, ih, iw), in_dims),
        index_map=(0, 1, 2, 3),
        is_broadcast=(False,) * 4,
        strides=project((tn, tc, step_h, step_w), in_dims),
        pad=project((0, 0, ph, pw), in_dims),
        pad_value=pad_value,
    )

    if node.op != "call_module":
        out_specs = [
            _OutputSpec(
                output_shape, output_ts, tuple(range(4)), outputs[-1].dtype
            )
        ]

        # Reuse the op's trailing args verbatim; bound here, not inside
        # ``compute`` — dynamo can't trace an FX-node attribute read.
        extra = tuple(anchor.args[4:])

        def compute(tile):
            # Padding zeroed (folded into the halo load).
            return anchor.target(tile, [kH, kW], [sh, sw], [0, 0], *extra)

        kernel = _map_kernel(compute, 1)
        return build_pipelined_buffers(
            kernel,
            grid,
            [in_spec],
            out_specs,
            (input_t,),
            num_slots=num_slots,
        )

    # Fused: run the whole submodule per tile.  The pool's input loads the halo;
    # every other operand tiles at the output block (codebooks / scalars whole).
    submod = node.meta["submodule"]
    codebooks = set()
    for sn in submod.graph.nodes:
        if sn.op == "call_function":
            for cb in quant_param_arg_nodes(sn):
                codebooks.add(cb.meta.get("source_node", cb))

    inputs, in_specs = [], []
    for n in node.all_input_nodes:
        inputs.append(n.value.clone())
        if n is in_node:
            in_specs.append(in_spec)
        elif n in codebooks or n.value.ndim == 0 or list(n.shape) == [1]:
            in_specs.append(None)
        else:
            in_specs.append(_compute_input_spec(out_tiling, tuple(n.shape)))

    # Dice each output via the canonical helper (multi / sparse outputs
    # handled), using the physical-order ``out_tiling``.
    tiled_shape = compute_output_tiled_shapes(node, out_tiling)
    tiled_shape = (
        list(tiled_shape)
        if isinstance(node.value, (list, tuple))
        else [tiled_shape]
    )
    out_specs = [
        _OutputSpec(tuple(o.shape), ts, tuple(range(o.ndim)), o.dtype)
        for o, ts in zip(outputs, tiled_shape)
    ]

    kernel = _map_kernel(submod, len(outputs))
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, out_specs, tuple(inputs), num_slots=num_slots
    )
    outline_dps_ops(gm)
    return gm
