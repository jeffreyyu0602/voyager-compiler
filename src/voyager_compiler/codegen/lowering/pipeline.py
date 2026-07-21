"""Unified Pallas-``pallas_call``-style kernel scheduler.

One scheduler subsuming the pointwise / pooling / GEMM bufferization builders:
given a ``kernel``, a ``grid``, and per-operand ``_InputSpec`` / ``_OutputSpec``
block specs, it emits a single rolled ``while_loop`` over the flattened grid.

Spec-driven for tile addressing, mutate-style for compute (Pallas ``out_ref``
semantics): each grid step loads every tiled input's current block into its SRAM
bank, calls ``kernel(grid_index, *in_banks, *out_banks)`` which writes each
output SRAM bank, then stores each out bank to DRAM.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.utils import (
    _HWIO,
    _InputSpec,
    _NHWC,
    _OutputSpec,
    _ScratchSpec,
    _build_fused_gm,
    _compute_input_spec,
    _finalize_exported_gm,
    _fuse_tail_in_body,
    _lenient_verifier,
    _project,
    _tag_loop_extents,
    _unproject,
    effect_cond,
    voyager,
)
from voyager_compiler.codegen.lowering.ops import (
    MemoryLevel,
    commit,
    oracle_disabled,
)
from voyager_compiler.codegen.lowering.tiling import get_tiling
from voyager_compiler.codegen.shape_prop import ShapeProp
from voyager_compiler.codegen.mapping_utils import (
    ancestors,
    is_bmm,
    is_conv2d,
    is_linear,
    is_matmul,
    quant_param_arg_nodes,
    repeat_of,
    swaps_last_two_dims,
    trailing_mha_perm,
)
from voyager_compiler.codegen.passes.tiling import compute_output_tiled_shapes
from voyager_compiler.codegen.passes.utils import _pair, get_arg_value

# Top-level: these modules do not import this one at module scope
# (bufferization imports the builders function-locally), so no cycle.
from voyager_compiler.codegen.mapping import get_anchor_node

_SRAM = int(MemoryLevel.SRAM)

# A microscaling quantize fuses onto the *end* of an MHA output relayout, so it
# is what the body returns and the permute sits one step above it.
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default

# Default software-pipeline depth (2 = double buffering).  Single source of
# truth for the ``num_banks`` default across the scheduler and op builders; a
# spec may override it per operand (``_InputSpec`` / ``_OutputSpec.num_banks``).
_DEFAULT_NUM_BANKS = 2


@dataclass
class _Window:
    """The ``num_banks``-dependent slice of a grid step's context: the current
    read slot and the depth-``D`` prefetch window (``D = num_banks - 1``).  One
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
    ``windows[self.num_banks]`` for its own depth.  Nothing is recomputed inside
    a scheduler — that would duplicate the traced ``delinearize_index`` nodes.
    """

    step: object
    cur: object
    next: object
    prev: object
    last: object
    windows: dict  # num_banks -> _Window


def select_bank(buf, slot):
    """One bank of a banked buffer (``[num_banks, *tile]``), as an explicit
    ``voyager.subview``: offset ``slot`` along the bank dim, the whole tile
    along the rest, and the bank dim dropped — it is not a tensor dim.  ``slot``
    may be a runtime value (``step % num_banks``).

    Said with ``buf[slot]`` this would be an ``aten.select``, indistinguishable
    from a model slicing a tensor — and the two mean opposite things: a bank
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

    Owns one tiled input's or one output's *window* — its ``num_banks``-deep
    SRAM ``bank`` and per-slot DMA ``sem`` (Pallas's ``window_ref`` +
    ``sem_recvs`` / ``sem_sends``) — plus the state machine that drives it: slot
    selection, copy / wait predicates, prologue priming, and producer /
    consumer / store cursor advancement.

    It holds **no mutable runtime state**: each method takes the loop-carried
    counter as a plain argument and returns the updated SymInt (the live cursors
    live in the ``while_loop`` operands), so the scheduler stays exportable.
    """

    _IN = "in"
    _OUT = "out"

    def __init__(self, kind, spec, grid, num_banks, bank, async_pipeline=False):
        self.kind = kind
        self.spec = spec
        self.grid = grid
        self.ndim = len(grid)
        self.num_steps = math.prod(grid)
        self.num_banks = num_banks
        # Prefetch distance (blocks ahead).
        self.D = max(0, num_banks - 2) if async_pipeline else num_banks - 1
        self.bank = bank  # SRAM window: [num_banks, *tile_sizes]
        # Per-slot async-DMA semaphore bank ([num_banks] int64).  An async
        # output store-sem starts with one credit per slot (``fill(1)``) so each
        # slot's first use has a token to consume — the commit always waits the
        # slot free, no warm-up guard needed.
        if kind == self._OUT and async_pipeline:
            self.sem = voyager.fill([], torch.int64, 1, banks=num_banks)
        else:
            self.sem = voyager.zeros([], torch.int64, banks=num_banks)

    # --- addressing (shared by both kinds) ----------------------------------

    def _tiled_dims(self):
        """``(d, g, r)`` for each operand dim that is dynamically indexed: it
        maps to a tiled grid dim ``g`` (``grid > 1``) and is not broadcast.
        Whole / broadcast / ``None``-mapped dims stay at block 0 and are left
        out.  ``r`` is how many consecutive grid steps read the *same* block —
        1 unless the operand repeats over ``g`` (a GQA head, say).
        """
        spec = self.spec
        bcast = getattr(spec, "is_broadcast", None)
        rep = getattr(spec, "repeat", None)
        for d, g in enumerate(spec.index_map):
            if (
                g is not None
                and self.grid[g] > 1
                and not (bcast is not None and bcast[d])
            ):
                yield d, g, (rep[d] if rep is not None else 1)

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
        for d, g, r in self._tiled_dims():
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
        for _, g, r in self._tiled_dims():
            term = self._block(cur[g], r) != self._block(next[g], r)
            differ = term if differ is None else (differ | term)
        return False if differ is None else differ

    def _innermost_tiled_dim(self):
        """The fastest-varying tiled grid dim (last dim with extent > 1, whose
        coord advances every step in row-major order); ``None`` if none.
        """
        tiled = [g for g in range(self.ndim) if self.grid[g] > 1]
        return tiled[-1] if tiled else None

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
        return any(g == inner and r == 1 for _, g, r in self._tiled_dims())

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
        tiled = list(self._tiled_dims())
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
        the dedup is a Python ``if``).  Each block's load signals ``post_count``
        times.  Return the seed producer count for a guarded input, or ``None``
        for an always-advance input (no cursor).
        """
        num_copies = 0
        prev_idx = None
        for p in range(min(self.D, self.num_steps)):
            idx = self._unravel(p)
            if p == 0 or self._indices_differ(prev_idx, idx):
                slot = num_copies % self.num_banks
                self._load_tile(
                    src,
                    select_bank(self.bank, slot),
                    idx,
                    select_bank(self.sem, slot),
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
        nb = self.num_banks
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
            # kernel's num_banks == 2): copy on the first step or a block change
            # (no prologue primes the first block).  Otherwise prefetch-gated.
            should_copy = (
                (w.first | differ) if self.D == 0 else (has_fetch & differ)
            )
            next_count = torch.sym_ite(should_copy, load_count + 1, load_count)
        # ``_check`` against the bank's own size lets the select bound
        # resolve on the unbacked step (needed for num_banks >= 3).
        torch._check(copy_slot < self.bank.size(0))
        effect_cond(
            should_copy,
            lambda: self._load_tile(
                src,
                select_bank(self.bank, copy_slot),
                w.fetch_idx,
                select_bank(self.sem, copy_slot),
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
            rs = ctx.windows[self.num_banks].cur_slot
            pred = None
        else:
            rs = wait_count % self.num_banks
            if self.spec.first_use_at_exit:
                # First read is the sweep's last step: defer the wait to
                # block-exit (the ``finished`` predicate).
                pred = ctx.last | self._indices_differ(ctx.cur, ctx.next)
            else:
                pred = (ctx.step == 0) | self._indices_differ(ctx.prev, ctx.cur)
        torch._check(rs < self.bank.size(0))
        _guarded_wait(select_bank(self.sem, rs), pred)
        return select_bank(self.bank, rs)

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
            slot = ctx.windows[self.num_banks].cur_slot
        else:
            slot = store_count % self.num_banks
        torch._check(slot < self.bank.size(0))
        if self.spec.first_use_at_exit:
            # Drain at block-exit (the write step): the ``finished`` predicate,
            # whose ``last`` term catches the final tile (``next`` is OOB).
            changed = ctx.last | self._indices_differ(ctx.cur, ctx.next)
        else:
            changed = self._indices_differ(ctx.prev, ctx.cur)
        pred = changed & (store_count >= self.num_banks)
        _guarded_wait(select_bank(self.sem, slot), pred)
        return select_bank(self.bank, slot), slot

    def copy_out(self, ctx, dst, store_count, out_slot, slot_idx):
        """Phase 5 — store the completed output tile ``out_slot`` to DRAM
        ``dst``, signaling its store semaphore, and return the advanced store
        counter.  Unconditional when the output advances every step; otherwise
        store-in-cond so a reduction writes once per output tile (when its block
        completes or on the last step).
        """
        sem = select_bank(self.sem, slot_idx)
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
        for j in range(self.num_banks):
            _guarded_wait(select_bank(self.sem, j), j < final_store_count)

    # --- async-kernel input protocol (AsyncPipelinedKernel) ------------------
    #
    # Reuses the base ``copy_in`` / ``prime_prologue`` / ``advance_consumer``
    # prefetch, but emits no input ``async_wait``: a tile's load semaphore
    # feeds ``commit.dependencies`` instead (``read_advancing`` /
    # ``read_reused`` are ``wait_in`` without the blocking wait).  Prefetch
    # distance is one less than the base (``D = num_banks - 2``): a tile is
    # consumed an iteration later, so a slot a prefetch overwrites was already
    # retired — reuse-WAR-safe.  An advancing input posts ``+1`` per load; a
    # reused one (held across an inner sweep) loads once per block and posts
    # ``+R`` (``reuse_count``), balancing a per-step consume.

    def read_advancing(self, ctx):
        """The current read slot's tile and its load semaphore (no wait) — the
        semaphore is handed to ``commit.dependencies``."""
        slot = ctx.windows[self.num_banks].cur_slot
        torch._check(slot < self.bank.size(0))
        return select_bank(self.bank, slot), select_bank(self.sem, slot)

    def read_reused(self, ctx, wait_count):
        """A reused input's current read slot (consumer cursor ``wait_count``)
        and its load semaphore, no wait — like ``wait_in`` without the blocking
        ``async_wait`` (the wait moves into ``commit.dependencies``)."""
        slot = wait_count % self.num_banks
        torch._check(slot < self.bank.size(0))
        return select_bank(self.bank, slot), select_bank(self.sem, slot)


class PipelinedKernel(torch.nn.Module):
    """Spec-driven, mutate-style kernel scheduler (see module docstring).

    ``kernel(grid_index, *in_banks, *out_banks)`` is the per-tile compute; it
    writes each output SRAM bank (via ``voyager.insert``) rather than
    returning a value.  A ``None`` input spec is a whole / scalar / codebook
    operand, passed through un-tiled.  ``num_banks`` is the software-pipeline
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
        num_banks: int = _DEFAULT_NUM_BANKS,
    ):
        super().__init__()
        if num_banks < 1:
            raise ValueError("num_banks must be >= 1")
        self.kernel = kernel
        self.grid = grid
        self.in_specs = in_specs
        self.out_specs = out_specs
        # Persistent, unbuffered, non-DMA SRAM refs (e.g. a reduction
        # accumulator); appended after the input/output refs in the kernel call.
        self.scratch_specs = tuple(scratch_specs)
        self.num_banks = num_banks
        self.ndim = len(self.grid)
        self.num_steps = math.prod(self.grid)

    def _num_banks(self, spec):
        """Resolve an operand's software-pipeline depth: its per-spec
        ``num_banks`` override, else the scheduler default.
        """
        n = self.num_banks if spec.num_banks is None else spec.num_banks
        if n < 1:
            raise ValueError("num_banks must be >= 1")
        return n

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
        and one banked ``_BufferedRef`` scheduler per operand, plus scratch
        SRAM.  Returns them as a tuple ``(out_bufs, tiled, in_counts,
        out_counts, in_refs, out_refs, scratch_banks)``.  Each input ref's
        prefetch distance
        follows ``self._async_pipeline`` (``num_banks - 2`` when async,
        ``num_banks - 1`` otherwise)."""
        out_bufs = [voyager.alloc(s.shape, s.dtype) for s in self.out_specs]
        tiled = [
            (inp, s) for inp, s in zip(inputs, self.in_specs) if s is not None
        ]
        # Per-operand software-pipeline depth: each operand's own ``num_banks``
        # (spec override, else the scheduler default), so a reused / low-reuse
        # operand can run a shallower or deeper pipeline than its peers.
        in_counts = [self._num_banks(s) for _, s in tiled]
        out_counts = [self._num_banks(s) for s in self.out_specs]
        # One SRAM bank per operand; separate pass so all bank ``alloc``s
        # precede the refs' semaphore ``zeros`` in graph order.
        in_banks = [
            voyager.alloc(s.tile_sizes, inp.dtype, _SRAM, banks=c)
            for (inp, s), c in zip(tiled, in_counts)
        ]
        out_banks = [
            voyager.alloc(s.tile_sizes, s.dtype, _SRAM, banks=c)
            for s, c in zip(self.out_specs, out_counts)
        ]
        # Per-operand schedulers: each owns its SRAM window (bank) and
        # allocates its own per-slot semaphore.
        in_refs = [
            _BufferedRef(
                _BufferedRef._IN,
                s,
                self.grid,
                c,
                bank,
                async_pipeline=self._async_pipeline,
            )
            for (inp, s), c, bank in zip(tiled, in_counts, in_banks)
        ]
        out_refs = [
            _BufferedRef(
                _BufferedRef._OUT,
                s,
                self.grid,
                c,
                bank,
                async_pipeline=self._async_pipeline,
            )
            for s, c, bank in zip(self.out_specs, out_counts, out_banks)
        ]
        # Scratch refs: single buffer (not ``num_banks``-deep — reused
        # immediately for the next tile's reduction while the output bank stays
        # buffered until its DMA drains), captured like ``out_banks``.
        scratch_banks = tuple(
            voyager.alloc(s.shape, s.dtype, _SRAM) for s in self.scratch_specs
        )
        return (
            out_bufs,
            tiled,
            in_counts,
            out_counts,
            in_refs,
            out_refs,
            scratch_banks,
        )

    def forward(self, *inputs):
        (
            out_bufs,
            tiled,
            in_counts,
            out_counts,
            in_refs,
            out_refs,
            scratch_banks,
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
            in_banks, i, g = [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_banks.append(inp)
                    continue
                ref = in_refs[i]
                if ref._advances_every_step():
                    in_banks.append(ref.wait_in(ctx, None))
                else:
                    in_banks.append(ref.wait_in(ctx, wait_counts[g]))
                    g += 1
                i += 1

            # 3. WAIT-OUT (reuse): drain each output slot's prior store before
            #    the kernel overwrites it.
            out_banks, out_slots = [], []
            for i in range(num_outputs):
                slot_ref, slot = out_refs[i].wait_out(ctx, store_counts[i])
                out_banks.append(slot_ref)
                out_slots.append(slot)

            # 4. KERNEL (mutate-style: writes the output bank slots).  Scratch
            #    refs follow the input/output args (Pallas's *index, *inputs,
            #    *outputs, *scratch convention).
            self.kernel(ctx.cur, *in_banks, *out_banks, *scratch_banks)

            # 5. COPY-OUT: store each completed output tile (guarded), signaling
            #    its store semaphore; advance the store counter.
            next_store_counts = []
            for i in range(num_outputs):
                next_store_counts.append(
                    out_refs[i].copy_out(
                        ctx,
                        out_bufs[i],
                        store_counts[i],
                        out_banks[i],
                        out_slots[i],
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
    Inputs prefetch ``num_banks - 2`` ahead (one less than the base), so a slot
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
            scratch_banks,
        ) = self._allocate(inputs)
        num_outputs = len(out_refs)
        distinct_counts = list(dict.fromkeys(in_counts + out_counts))

        # Compute-done semaphore bank: commit(s) posts slot s % 2, the lagged
        # retire of tile s waits it.
        done_sem = voyager.zeros([], torch.int64, banks=_DONE_DEPTH)

        # Per-operand reuse count: 1 = advancing (posts +1/step), R = reused
        # (loaded once per block, posts +R so a per-step consume balances it).
        reuse = [ref.reuse_count() for ref in in_refs]

        # Prologue: prime the first ``D = num_banks - 2`` prefetch positions per
        # input (deduped by block, posting the reuse count); a reused input
        # seeds its producer cursor, an advancing one carries none.
        init_load = []
        for i, (inp, _) in enumerate(tiled):
            c = in_refs[i].prime_prologue(inp, reuse[i])
            if c is not None:
                init_load.append(c)

        def _store_out(i, coord, store_count):
            """Store output ``i``'s tile to DRAM at grid ``coord`` from the slot
            of the tile that just finished — ``(store_count - 1) % num_banks``,
            one behind the tile the commit is on."""
            ref = out_refs[i]
            slot = (store_count - 1) % ref.num_banks
            ref._store_tile(
                select_bank(ref.bank, slot),
                out_bufs[i],
                coord,
                select_bank(ref.sem, slot),
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
            in_banks, in_sems, i, g = [], [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_banks.append(inp)
                    continue
                ref = in_refs[i]
                if ref._advances_every_step():
                    tile, sem = ref.read_advancing(ctx)
                else:
                    tile, sem = ref.read_reused(ctx, wait_counts[g])
                    g += 1
                in_banks.append(tile)
                in_sems.append(sem)
                i += 1

            # The output tile's ordinal is ``store_counts`` (tiles finished so
            # far); all its reduction rounds share slot ``ordinal % num_banks``
            # (they accumulate into it), rotating per tile.
            out_banks, out_sems = [], []
            for i in range(num_outputs):
                slot = store_counts[i] % out_refs[i].num_banks
                out_banks.append(select_bank(out_refs[i].bank, slot))
                out_sems.append(select_bank(out_refs[i].sem, slot))

            post = select_bank(done_sem, step % _DONE_DEPTH)

            # 3. Dispatch the tile's compute: the async kernel issues the
            #    ``commit`` itself — waiting the input load semaphores and, on
            #    the round it writes the output, that slot's store semaphore
            #    free (seeded with a credit, so first uses never underflow).
            self.kernel(
                ctx.cur,
                in_banks,
                out_banks,
                scratch_banks,
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
                    select_bank(done_sem, (step - 1) % _DONE_DEPTH)
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
            voyager.async_wait(select_bank(done_sem, last % _DONE_DEPTH))
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
    num_banks: int = _DEFAULT_NUM_BANKS,
    async_pipeline: bool = False,
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """Build the bufferized FX graph (a single rolled ``while_loop`` over
    ``voyager.*`` primitives) for ``kernel`` over ``grid``.  Mirrors
    ``build_pointwise_buffers``'s export / finalize / extent-tag flow.

    ``async_pipeline`` selects :class:`AsyncPipelinedKernel` (cross-tile
    ramp-up/ramp-down overlap via ``voyager.commit``) instead of the
    synchronous :class:`PipelinedKernel`.  ``kernel`` must then be an async
    kernel template (``_map_kernel(async_pipeline=True)``) that issues the
    ``commit`` itself.
    """
    cls = AsyncPipelinedKernel if async_pipeline else PipelinedKernel
    pattern = cls(
        kernel,
        grid,
        in_specs,
        out_specs,
        scratch_specs=scratch_specs,
        num_banks=num_banks,
    )
    with _lenient_verifier():
        gm = export_model(pattern, inputs, kwargs=kwargs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(0, pattern.num_steps, 1)]])
    # Stamp a concrete-offset ``.value`` on every node (incl. loop / cond
    # bodies) so the tail re-fusion's ShapeProp never sees export's symbolic
    # ``step % num_banks`` tile offset.
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
# (each result written into its output bank), accumulating across the reduction
# grid dim for a GEMM / conv and overwriting for a map.
# ---------------------------------------------------------------------------


# One fused output, as the builders take it before it becomes an
# ``_OutputSpec``: its DRAM shape, its SRAM tile, its dtype, and the output dim
# -> grid dim map -- ``None`` for the builder's default, and ``None`` on a dim
# that no grid index addresses (it is stored whole).
_FusedOutput = Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    torch.dtype,
    Optional[Tuple[Optional[int], ...]],
]


@dataclass
class _FusedInfo:
    """Parsed pieces of a fused ``call_module`` (GEMM/conv + post-op pointwise
    ops), for the GEMM / conv pipeline builders.

    ``anchor_node`` is the GEMM/conv reference op (inside the submodule, so its
    ``args`` are submodule placeholders whose ``meta['source_node']`` point back
    to the outer graph).  ``fused_gm`` runs the post-op ops as ``[acc, *fused]
    -> output(s)`` on the anchor's result tile, and is ``None`` when there is no
    tail to run -- a submodule can hold the anchor and nothing but its prelude
    (a GQA ``expand``), and it is still a ``_fused`` node.  ``input_values`` are
    the tensors those ops consume; ``in_specs[i]`` is that input's tile
    ``_InputSpec`` (or ``None`` for a whole input); ``in_sources[i]`` is its
    outer graph node, used to order operands canonically.  ``output_specs``
    holds one ``_FusedOutput`` per fused output -- several when the fused op
    returns a tuple (``quantize_mx``).
    """

    anchor_node: torch.fx.Node
    fused_gm: Optional[torch.fx.GraphModule]
    tiling: Optional[Tuple[int, ...]]
    input_values: List[torch.Tensor]
    in_specs: List[Optional[_InputSpec]]
    in_sources: List[torch.fx.Node]
    output_specs: List[_FusedOutput]


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


def _detect_mha_relayout(fused_ops, anchor, tiling, gm):
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
    grid_m, grid_n = nb, nb + 1
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
    projected to the output's physical layout).  The factors are stashed on
    ``_FusedInfo.tiling`` so the builder reuses them (no second tiler run).
    """
    if node.op != "call_module":
        return None
    submod = node.meta.get("submodule")
    anchor = get_anchor_node(node)
    is_conv = is_conv2d(anchor)

    ShapeProp(submod).propagate(
        *(n.value.clone() for n in node.all_input_nodes)
    )

    tiling = get_tiling(node, tiler)
    if tiling is None:
        out_tiling = None
        out_index_map = None
    elif is_conv:
        ny, nx, nk, _ = tiling  # logical (Y, X, K, C) counts
        odims = _NHWC if anchor.meta.get("transposed", False) else None
        out_tiling = _project((1, nk, ny, nx), odims)  # physical output counts
        out_index_map = _project((0, 1, 2, 3), odims)
    else:
        out_tiling = tiling[:-1]  # gemm (batch.., n_m, n_n, n_k) -> drop K
        out_index_map = tuple(range(len(out_tiling)))

    anchor_prelude = ancestors(anchor)
    fused_ops = []
    input_nodes, input_values, in_specs = [], [], []
    for sn in submod.graph.nodes:
        if sn is anchor or sn.op != "call_function" or sn in anchor_prelude:
            continue
        fused_ops.append(sn)
        codebooks = quant_param_arg_nodes(sn)
        for inp in sn.all_input_nodes:
            if (
                inp.op != "placeholder"
                or inp in input_nodes
                or inp in anchor_prelude
            ):
                continue
            input_nodes.append(inp)
            input_values.append(inp.value.clone())
            if inp in codebooks or inp.value.numel() == 1 or out_tiling is None:
                in_specs.append(None)  # whole operand
            else:
                in_specs.append(
                    _compute_input_spec(
                        out_tiling, tuple(inp.shape), out_index_map
                    )
                )

    fused_gm = (
        _build_fused_gm(submod, anchor, fused_ops, input_nodes)
        if fused_ops
        else None
    )
    in_sources = [n.meta.get("source_node", n) for n in input_nodes]

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
    # ``index_map`` is ``None`` (builder's default M/N mapping) except for the
    # MHA relayout handled below.
    output_specs = [
        (s, t, v.dtype, None) for s, t, v in zip(full_shapes, tiled_shape, vals)
    ]

    if not is_conv and fused_ops and tiling is not None:
        relayout = _detect_mha_relayout(fused_ops, anchor, tiling, fused_gm)
        if relayout is not None:
            out_tile, out_imap, data_shape = relayout
            output_specs = [
                (
                    s,
                    tuple(
                        t * d // full
                        for t, d, full in zip(out_tile, s, data_shape)
                    ),
                    v.dtype,
                    out_imap,
                )
                for s, v in zip(full_shapes, vals)
            ]

    return _FusedInfo(
        anchor,
        fused_gm,
        tiling,
        input_values,
        in_specs,
        in_sources,
        output_specs,
    )


def _map_kernel(
    compute: Callable, num_outputs: int, async_pipeline: bool = False
):
    """Map kernel (no cross-tile reduction): adapt a return-style
    ``compute(*in_banks) -> Tensor | tuple`` into the scheduler's mutate-style
    kernel, writing each result straight into its output bank.  Every num_k == 1
    op uses this; the reduction case uses ``_reduction_fused_kernel``.

    ``async_pipeline=False`` returns the synchronous :class:`PipelinedKernel`
    kernel (``kernel(grid_index, *in_banks, *out_banks)``, insert inline).
    ``async_pipeline=True`` returns the :class:`AsyncPipelinedKernel`
    counterpart: it *dispatches* the same compute with ``voyager.commit`` so
    the next tile's systolic-array ramp-up overlaps this tile's ramp-down.  The
    commit waits the input load semaphores and — every step, since a map writes
    the output each step — that slot's store semaphore free ("wait on whatever
    we write"; the store-sem is seeded with a credit so a slot's first use
    never underflows).
    """

    def inner(grid_index, *args):
        in_banks = args[: len(args) - num_outputs]
        out_banks = args[len(args) - num_outputs :]
        results = compute(*in_banks)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        for bank, value in zip(out_banks, results):
            voyager.insert(value, bank)

    if not async_pipeline:
        return inner

    def kernel(
        grid_index, in_banks, out_banks, scratch, in_sems, out_sems, post
    ):
        operands = [grid_index, *in_banks, *out_banks, *scratch]
        commit(inner, operands, dependencies=[*in_sems, *out_sems], post=post)

    return kernel


def _reduction_fused_kernel(
    compute: Callable,
    reduction_dim: int,
    last_idx: int,
    op_dtype: Optional[torch.dtype],
    out_specs: List[_OutputSpec],
    fused_gm: Optional[Callable],
    fused_operand_indices: List[int] = (),
    async_pipeline: bool = False,
):
    """Kernel for an op whose reduction needs > 1 tile (num_k > 1 GEMM / conv;
    the num_k == 1 map case uses ``_map_kernel``).

    ``compute(in_banks, first)`` runs the bare op on the current tiles; on the
    ``first`` step it folds the bias straight into the op (hardware does ``op +
    bias`` in one pass).  The bias rides only the first step — the same step
    that initializes the accumulator — so bias gate and reduction init collapse
    into the single reduction ``torch.cond`` (no nested bias-gate cond).  The
    partial accumulates into a scratch ref; on the last step the completed
    accumulator is cast to ``op_dtype`` (it may accumulate wider, e.g. fp32) and
    mapped through the fused tail (if any) into the output bank(s).

    ``async_pipeline=True`` returns the :class:`AsyncPipelinedKernel` variant.
    The accumulate (``_accumulate``, input-only) and the finalize
    (``_finalize``, output-writing) split across the round predicate, so the
    finalizing round's commit waits the output store-sem while the others wait
    only their inputs -- and neither branch carries a statically-dead sub-cond.
    """
    num_outputs = len(out_specs)

    def _split(args):
        # args = [*in_banks, *out_banks, scratch]; one scratch accumulator.
        n_in = len(args) - num_outputs - 1
        return args[:n_in], args[n_in : n_in + num_outputs], args[-1]

    def _to_acc(result, scratch):
        # Upcast a partial to the (possibly wider, e.g. fp32) accumulator dtype.
        return result if op_dtype is None else result.to(scratch.dtype)

    def _accumulate(grid_index, in_banks, scratch):
        """Fold this round's partial into the scratch accumulator: the op with
        bias on the first coord (initializing it), the bare op plus the running
        accumulator after."""

        def init():
            return _to_acc(compute(in_banks, True), scratch)

        def accumulate(prev=scratch):
            return _to_acc(compute(in_banks, False), scratch) + prev

        voyager.insert(
            torch.cond(grid_index[reduction_dim] == 0, init, accumulate),
            scratch,
        )

    def _finalize(in_banks, out_banks, scratch):
        """Cast the completed accumulator, apply the fused tail once, and store
        each output."""
        outs = scratch if op_dtype is None else scratch.to(op_dtype)
        if fused_gm is not None:
            fused = [in_banks[i] for i in fused_operand_indices]
            outs = fused_gm(outs, *fused)
        if not isinstance(outs, (tuple, list)):
            outs = (outs,)
        for bank, out in zip(out_banks, outs):
            voyager.insert(out, bank)

    def inner(grid_index, *args):
        in_banks, out_banks, scratch = _split(args)
        _accumulate(grid_index, in_banks, scratch)
        effect_cond(
            grid_index[reduction_dim] == last_idx,
            lambda: _finalize(in_banks, out_banks, scratch),
        )

    if not async_pipeline:
        return inner

    # The accumulate never touches the output and the finalize always does, so
    # commit them per round rather than one body in both branches: every round
    # commits the accumulate (input deps only); the last round commits
    # accumulate-then-finalize and also waits the output store-sem.
    def accumulate_body(grid_index, *args):
        in_banks, _out_banks, scratch = _split(args)
        _accumulate(grid_index, in_banks, scratch)

    def finalize_body(grid_index, *args):
        in_banks, out_banks, scratch = _split(args)
        partial = _to_acc(compute(in_banks, False), scratch)
        voyager.insert(partial + scratch, scratch)
        _finalize(in_banks, out_banks, scratch)

    def kernel(
        grid_index, in_banks, out_banks, scratch, in_sems, out_sems, post
    ):
        operands = [grid_index, *in_banks, *out_banks, *scratch]

        def on_last():
            commit(
                finalize_body,
                operands,
                dependencies=[*in_sems, *out_sems],
                post=post,
            )
            return 0

        def not_last():
            commit(
                accumulate_body, operands, dependencies=[*in_sems], post=post
            )
            return 0

        torch.cond(grid_index[reduction_dim] == last_idx, on_last, not_last)

    return kernel


def _reduction_inplace_kernel(
    compute: Callable, reduction_dim: int, async_pipeline: bool = False
):
    """Kernel for a reduction with nothing left to do once it completes — no
    cast (the accumulator's dtype is the output's) and no fused tail.  It
    accumulates straight into the output bank, so there is no scratch ref and no
    finalize step: the completed tile is already in the slot the store reads.

    ``_reduction_fused_kernel`` is the general case, where a cast or a tail must
    map the accumulator into the bank and so needs one of its own.

    ``async_pipeline=True`` returns the :class:`AsyncPipelinedKernel` variant:
    the accumulator *is* the output slot, first written on the ``0`` reduction
    coord, so the commit dispatch waits the slot free there.
    """

    def inner(grid_index, *args):
        *in_banks, out_bank = args

        def init():
            return compute(in_banks, True)

        def accumulate(prev=out_bank):
            return compute(in_banks, False) + prev

        voyager.insert(
            torch.cond(grid_index[reduction_dim] == 0, init, accumulate),
            out_bank,
        )

    if not async_pipeline:
        return inner

    # ``init`` writes the output-bank accumulator fresh (waits the slot free),
    # ``accumulate`` RMWs it in place (no output dep).  Split them across the
    # round-0 predicate so neither commit body carries the now-redundant
    # ``K == 0`` inner cond.
    def init_body(grid_index, *args):
        *in_banks, out_bank = args
        voyager.insert(compute(in_banks, True), out_bank)

    def accumulate_body(grid_index, *args):
        *in_banks, out_bank = args
        voyager.insert(compute(in_banks, False) + out_bank, out_bank)

    def kernel(
        grid_index, in_banks, out_banks, scratch, in_sems, out_sems, post
    ):
        operands = [grid_index, *in_banks, *out_banks, *scratch]

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
        s.num_banks = 1
        s.first_use_at_exit = True
    for i in fused_idx:
        if in_specs[i] is not None:
            in_specs[i].num_banks = 1
            in_specs[i].first_use_at_exit = True


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
    num_banks: int = _DEFAULT_NUM_BANKS,
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
    tiling = info.tiling if info is not None else get_tiling(node, tiler)
    if info is None and tiling is None:
        return None
    anchor = info.anchor_node if info is not None else node

    inp = anchor.args[0].value.clone()
    w = anchor.args[1].value.clone()
    out = anchor.value  # the conv output (drives the N/K/oH/oW grid)
    if inp.ndim != 4 or w.ndim != 4:
        return None
    groups = get_arg_value(anchor, 6, "groups", 1)
    if groups != 1:
        return None  # depthwise conv unsupported

    nhwc = anchor.meta.get("transposed", False)
    in_dims = _NHWC if nhwc else None
    w_dims = _HWIO if nhwc else None
    out_dims = _NHWC if nhwc else None

    N, C, H, W = _unproject(inp.shape, in_dims)
    K, _, kH, kW = _unproject(w.shape, w_dims)
    oH, oW = _unproject(out.shape, out_dims)[2:]

    if tiling is None:
        tiling = (1, 1, 1, 1)
    ny, nx, nk, nc = tiling
    tn, toh, tow, tc, tk = N, oH // ny, oW // nx, C // nc, K // nk
    num_k = nc

    sh, sw = _pair(get_arg_value(anchor, 3, "stride", 1))
    ph, pw = _pair(get_arg_value(anchor, 4, "padding", 0))
    dh, dw = _pair(get_arg_value(anchor, 5, "dilation", 1))
    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1

    # grid dims (logical): 0=N 1=K 2=oH 3=oW 4=C(reduction).  Batch is never
    # tiled.  The weight's kH/kW axes are loaded whole (index_map ``None``), so
    # there is no kernel-window grid dim.
    grid = (1, nk, ny, nx, nc)
    in_spec = _InputSpec(
        _project((tn, tc, ih, iw), in_dims),
        _project((0, 4, 2, 3), in_dims),  # N->0, C->4, H->oH(2), W->oW(3)
        (False,) * 4,
        strides=_project((tn, tc, toh * sh, tow * sw), in_dims),
        pad=_project((0, 0, ph, pw), in_dims),
        pad_value=0.0,
    )
    w_spec = _InputSpec(
        _project((tk, tc, kH, kW), w_dims),
        # K->1, C->4, kH/kW->None (loaded whole, mapped to no grid dim)
        _project((1, 4, None, None), w_dims),
        (False,) * 4,
    )
    bias_spec = _InputSpec((tk,), (1,), (False,))
    # The output(s) tile onto the (N, K, oH, oW) grid dims (C reduction
    # dropped); a fused op may produce several (``quantize_mx``).
    out_index_map = _project((0, 1, 2, 3), out_dims)
    if info is None:
        out_specs = [
            _OutputSpec(
                _project((N, K, oH, oW), out_dims),
                _project((tn, tk, toh, tow), out_dims),
                out_index_map,
                inp.dtype,
            )
        ]
    else:
        out_specs = [
            _OutputSpec(tuple(shape), tuple(tile), imap or out_index_map, dtype)
            for shape, tile, dtype, imap in info.output_specs
        ]

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
            _project((tn, tc // bs, ih, iw), in_dims),
            _project((0, 4, 2, 3), in_dims),
            (False,) * 4,
            strides=_project((tn, tc // bs, toh * sh, tow * sw), in_dims),
            pad=_project((0, 0, ph, pw), in_dims),
            pad_value=0.0,
        )
        wt_scale_qspec = _InputSpec(
            _project((tk, tc // bs, kH, kW), w_dims),
            _project((1, 4, None, None), w_dims),  # kH/kW whole -> None
            (False,) * 4,
        )
        add_kw_input("input_scale", in_scale_qspec)
        add_kw_input("weight_scale", wt_scale_qspec)
        add_kw_input("input_code", None)
        add_kw_input("weight_code", None)

    # Fused post-op operands (a residual, …), keyed by their outer node.
    if info is not None:
        for s, val, spec in zip(
            info.in_sources, info.input_values, info.in_specs
        ):
            node_to_spec[s] = (val, spec)

    order = {n: i for i, n in enumerate(node.all_input_nodes)}
    assert len(node_to_spec) == len(order), "conv operand shared across roles"
    inputs = [node_to_spec[n][0] for n in node.all_input_nodes]
    in_specs = [node_to_spec[n][1] for n in node.all_input_nodes]

    in_idx = order[src(anchor.args[0])]
    w_idx = order[src(anchor.args[1])]
    bias_idx = order[src(bias_n)] if bias_n is not None else None
    kw_idx = {name: order[n] for name, n in kw_nodes.items()}
    fused_idx = [order[s] for s in info.in_sources] if info is not None else []
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
        d0, d1, d2, d3 = _project((tn, tk, toh, tow), out_dims)
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

    def compute(*in_tiles):
        # num_k == 1 map: conv in one step, then the fused tail (if any).
        result = conv2d_kernel(in_tiles, True)
        if fused_gm is not None:
            return fused_gm(result, *[in_tiles[i] for i in fused_idx])
        return result

    acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
    if num_k == 1:
        scratch_specs = []
        kernel = _map_kernel(
            compute, len(out_specs), async_pipeline=async_pipeline
        )
    elif fused_gm is None and acc_dtype == out.dtype:
        scratch_specs = []
        kernel = _reduction_inplace_kernel(
            conv2d_kernel, reduction_dim=4, async_pipeline=async_pipeline
        )
    else:
        if single_buffer_tail and not async_pipeline:
            _single_buffer_reduction_operands(in_specs, out_specs, fused_idx)
        scratch_specs = [
            _ScratchSpec(_project((tn, tk, toh, tow), out_dims), acc_dtype)
        ]
        kernel = _reduction_fused_kernel(
            conv2d_kernel,
            reduction_dim=4,
            last_idx=num_k - 1,
            out_specs=out_specs,
            op_dtype=(out.dtype if acc_dtype != out.dtype else None),
            fused_gm=fused_gm,
            fused_operand_indices=fused_idx,
            async_pipeline=async_pipeline,
        )
    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs,
        tuple(inputs),
        scratch_specs=scratch_specs,
        num_banks=num_banks,
        async_pipeline=async_pipeline,
    )
    if num_k > 1 or info is not None:
        _fuse_tail_in_body(
            gm, anchor.target, fuse_anchor_with_tail=(num_k == 1)
        )
    _stamp_anchor_meta(gm, anchor)
    return gm


def _peel_weight(node: torch.fx.Node):
    """Resolve a GEMM weight operand through the ops fused onto it — a
    last-two-dim transpose, a grouped-query broadcast, a dequantize — to the
    external operand they read.

    Returns ``(node, transposed, repeat, dequant)``.  An attention ``Q @ Kᵀ``
    fuses ``K.transpose(-2, -1)`` onto the weight, and GQA fuses the repeat that
    turns 8 KV heads into 32.  Neither is emitted: ``transposed`` folds into the
    DMA (the fetch swaps its last two dims and ``async_copy`` ``.mT``s the tile
    into the bank), ``repeat`` into the block index (``grid_index // repeat[d]``,
    so four query heads share one KV tile).

    A ``dequantize`` -- a KIVI KV cache, packed in DRAM -- does *not* fold into
    the addressing, because it computes: it comes back for the builder to run on
    the fetched tile, which is what lets the cache stay packed all the way into
    the bank.

    Only ops over an external operand — a placeholder, i.e. something the fused
    submodule is handed rather than computes — peel; anything else comes back
    unchanged.
    """
    inner = node
    dequant = None
    if inner.target is torch.ops.quantized_ops.dequantize.default:
        dequant = inner
        inner = inner.args[0]

    # A ``Kᵀ`` sits under the decode, never over it (``_insert_transpose_op``
    # hoists it there).  The decode is none the wiser: the tile the fetch
    # transposes into the bank is the one it was written against.
    transposed = swaps_last_two_dims(inner)
    if transposed:
        inner = inner.args[0]

    if inner.op == "placeholder":
        return inner, transposed, None, dequant

    found = repeat_of(inner)
    if found is not None and found[0].op == "placeholder":
        source, _, repeat = found
        return source, transposed, repeat, dequant

    return node, False, None, None


def build_gemm(
    node,
    *,
    num_banks: int = _DEFAULT_NUM_BANKS,
    accumulate_fp32: bool = False,
    single_buffer_tail: bool = False,
    async_pipeline: bool = True,
    tiler=None,
):
    """Pipeline builder for a linear / matmul / batched-matmul node — incl. the
    microscaling / codebook (``*_mx``) variants and a fused bias — over the
    cross-tile K reduction.  Grid ``(M, N, K)`` (or ``(B, M, N, K)``) tiles with
    K innermost; the kernel accumulates ``act_tile @ weight_tile`` into the
    output bank.  Returns the gm, or ``None``.

    Operands are assembled in the fused node's ``all_input_nodes`` order so the
    positional splice in ``replace_node_with_graph_module`` binds each
    placeholder correctly even when a fused-tail operand is graph-ordered before
    the anchor; ``compute`` / ``gemm_kernel`` then dispatch by canonical index
    (``act_idx`` / ``kw_idx`` / ``fused_idx``), not a positional ``*extra``
    split.
    """
    info = parse_fused_submodule(node, tiler)
    tiling = info.tiling if info is not None else get_tiling(node, tiler)
    if info is None and tiling is None and not is_bmm(node):
        return None
    anchor = info.anchor_node if info is not None else node

    # The weight's fused relayouts (attention's ``Kᵀ``, GQA's head repeat) are
    # folded into how its tile is addressed rather than emitted; the spec itself
    # stays in the matmul (Kᵀ) layout.  A fused ``dequantize`` (a packed KV
    # cache) is compute, so it runs on the fetched tile instead.
    weight_node, transposed, weight_repeat, dequant = _peel_weight(
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
        tk = K
    else:
        out_tiling, nk = tiling[:-1], tiling[-1]  # batch.. + (n_m, n_n) , n_k
        out_ts = tuple(s // t for s, t in zip(out.shape, out_tiling))
        tk = K // nk
    tm, tn = int(out_ts[-2]), int(out_ts[-1])

    # torch matmul broadcasts the leading batch dims: each output batch dim is
    # its own grid dim (0..nb-1), then M / N / K are grid dims nb / nb+1 / nb+2
    # (K innermost).
    nb = out.ndim - 2
    gm, gn, gk = nb, nb + 1, nb + 2
    out_batch = tuple(out.shape[:nb])
    tb = tuple(int(x) for x in out_ts[:nb])
    grid = tuple(b // t for b, t in zip(out_batch, tb)) + (
        M // tm,
        N // tn,
        K // tk,
    )

    ck = is_matmul(anchor) != bool(anchor.meta.get("transposed", False))
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
        out_specs = [
            _OutputSpec(tuple(shape), tuple(tile), imap or out_index_map, dtype)
            for shape, tile, dtype, imap in info.output_specs
        ]

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
        v, transposed, repeat, _ = _peel_weight(v)
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
                v, t, r, _ = _peel_weight(v)
                spec = _spec(v.value.shape, dq_tile, _proj(gn, gk))
                spec.transposed = t
                spec.repeat = r
            dq_nodes[i] = src(v)
            node_to_spec[src(v)] = (v.value.clone(), spec)

    # Fused post-op operands (residual, scale, …), keyed by their outer node.
    if info is not None:
        for s, val, spec in zip(
            info.in_sources, info.input_values, info.in_specs
        ):
            node_to_spec[s] = (val, spec)

    order = {n: i for i, n in enumerate(node.all_input_nodes)}
    assert len(node_to_spec) == len(order), "gemm operand shared across roles"
    inputs = [node_to_spec[n][0] for n in node.all_input_nodes]
    in_specs = [node_to_spec[n][1] for n in node.all_input_nodes]

    act_idx = order[src(anchor.args[0])]
    weight_idx = order[src(weight_node)]
    bias_idx = order[src(bias_n)] if bias_n is not None else None
    kw_idx = {name: order[n] for name, n in kw_nodes.items()}
    fused_idx = [order[s] for s in info.in_sources] if info is not None else []
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

    def compute(*in_tiles):
        # num_k == 1 map: GEMM in one step, then the fused tail (if any).
        result = gemm_kernel(in_tiles, True)
        if fused_gm is not None:
            return fused_gm(result, *[in_tiles[i] for i in fused_idx])
        return result

    acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
    if num_k == 1:
        scratch_specs = []
        kernel = _map_kernel(
            compute, len(out_specs), async_pipeline=async_pipeline
        )
    elif fused_gm is None and acc_dtype == out.dtype:
        scratch_specs = []
        kernel = _reduction_inplace_kernel(
            gemm_kernel, reduction_dim=gk, async_pipeline=async_pipeline
        )
    else:
        if single_buffer_tail and not async_pipeline:
            _single_buffer_reduction_operands(in_specs, out_specs, fused_idx)
        scratch_specs = [_ScratchSpec(tuple(tb) + (tm, tn), acc_dtype)]
        kernel = _reduction_fused_kernel(
            gemm_kernel,
            reduction_dim=gk,
            last_idx=num_k - 1,
            out_specs=out_specs,
            op_dtype=(out.dtype if acc_dtype != out.dtype else None),
            fused_gm=fused_gm,
            fused_operand_indices=fused_idx,
            async_pipeline=async_pipeline,
        )
    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs,
        tuple(inputs),
        scratch_specs=scratch_specs,
        num_banks=num_banks,
        async_pipeline=async_pipeline,
    )
    if num_k > 1 or info is not None or dequant is not None:
        _fuse_tail_in_body(
            gm, anchor.target, fuse_anchor_with_tail=(num_k == 1)
        )
    _stamp_anchor_meta(gm, anchor)
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


def build_pointwise(node, *, num_banks: int = _DEFAULT_NUM_BANKS):
    """Pipeline builder for a pointwise / batched-reduction node (elementwise
    ops, layernorm·softmax whose reduction dim is kept whole in the tile, and a
    standalone ``transpose`` / ``permute`` relayout).  Tiles the output grid and
    writes each output tile once (no cross-tile reduction).  Returns the gm, or
    ``None``.
    """
    anchor = get_anchor_node(node)
    tiling = anchor.meta.get("l2_tiling") if anchor is not None else None
    if node.op != "call_module" and tiling is None:
        return None

    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]

    val = node.value
    outputs = val if isinstance(val, (list, tuple)) else (val,)

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

        compute = submod

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
        op = node.target
        codebooks = quant_param_arg_nodes(node)

        def compute(*tiles):
            args = [
                tiles[i] if i is not None else a
                for i, a in zip(arg_slots, op_args)
            ]
            kwargs = {
                k: tiles[i] if i is not None else op_kwargs[k]
                for k, i in kw_slots.items()
            }
            return op(*args, **kwargs)

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
    kernel = _map_kernel(compute, len(outputs))
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, out_specs, tuple(inputs), num_banks=num_banks
    )

    if node.op == "call_module" and anchor is not None:
        _fuse_tail_in_body(gm, anchor.target)
    return gm


_POOL2D_SUPPORTED = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.quantized_ops.max_pool2d.default,
}


def build_pool(node, *, num_banks: int = _DEFAULT_NUM_BANKS):
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

    tiling = anchor.meta.get("l2_tiling")
    if node.op != "call_module" and tiling is None:
        return None

    in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
    input_t = in_node.value.clone()

    val = node.value
    outputs = val if isinstance(val, (list, tuple)) else (val,)
    output_shape = tuple(outputs[-1].shape)

    in_dims = _NHWC if anchor.meta.get("transposed", False) else None
    N, C, H, W = _unproject(output_shape, in_dims)
    if tiling is None:
        nN, nH, nW, nC = 1, 1, 1, 1
    else:
        nN, nH, nW, nC = tiling
    tn, tc, toh, tow = N // nN, C // nC, H // nH, W // nW
    output_ts = _project((tn, tc, toh, tow), in_dims)
    out_tiling = _project((nN, nC, nH, nW), in_dims)

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
        tile_sizes=_project((tn, tc, ih, iw), in_dims),
        index_map=(0, 1, 2, 3),
        is_broadcast=(False,) * 4,
        strides=_project((tn, tc, step_h, step_w), in_dims),
        pad=_project((0, 0, ph, pw), in_dims),
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
            num_banks=num_banks,
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
        kernel, grid, in_specs, out_specs, tuple(inputs), num_banks=num_banks
    )
    _fuse_tail_in_body(gm, anchor.target)
    return gm
