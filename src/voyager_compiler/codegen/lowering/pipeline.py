"""Unified Pallas-``pallas_call``-style kernel scheduler.

One scheduler subsuming the pointwise / pooling / GEMM bufferization builders:
given a ``kernel``, a ``grid``, and per-operand ``_InputSpec`` / ``_OutputSpec``
block specs, it emits a single rolled ``while_loop`` over the flattened grid.

Spec-driven for tile addressing, mutate-style for compute (Pallas ``out_ref``
semantics): each grid step loads every tiled input's current block into its SRAM
bank, calls ``kernel(grid_index, *in_tiles, *out_banks)`` which writes each
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
    voyager,
)
from voyager_compiler.codegen.lowering.ops import MemoryLevel, oracle_disabled
from voyager_compiler.codegen.lowering.tiling import get_tiling
from voyager_compiler.codegen.shape_prop import ShapeProp
from voyager_compiler.codegen.mapping_utils import (
    ancestors,
    is_bmm,
    is_compute_op,
    is_conv2d,
    is_gemm_op,
    is_linear,
    is_matmul,
    quant_table_arg_nodes,
)
from voyager_compiler.codegen.passes.tiling import compute_output_tiled_shapes
from voyager_compiler.codegen.passes.utils import _pair, get_arg_value

# Top-level: these modules do not import this one at module scope
# (bufferization imports the builders function-locally), so no cycle.
from voyager_compiler.codegen.mapping import (
    get_anchor_node,
    is_mha_qkv_permute,
)

_SRAM = int(MemoryLevel.SRAM)

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


def _guarded_wait(sem, pred=None):
    """``async_wait(sem)`` guarded by ``pred``, so each slot's semaphore is
    waited exactly once per signaling copy (a counting semaphore underflows on a
    stray wait).  ``pred=None`` waits unconditionally — an operand whose block
    changes every step is already once-per-block.  The wait is wrapped in
    ``torch.cond`` (kept alive by ``has_side_effect``).
    """
    if pred is None:
        voyager.async_wait(sem)
        return

    def do():
        voyager.async_wait(sem)
        return 1

    def skip():
        return 0

    torch.cond(pred, do, skip)


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

    def __init__(self, kind, spec, grid, num_banks, bank):
        self.kind = kind
        self.spec = spec
        self.grid = grid
        self.ndim = len(grid)
        self.num_steps = math.prod(grid)
        self.num_banks = num_banks
        self.D = num_banks - 1  # prefetch distance (blocks ahead)
        self.bank = bank  # SRAM window: [num_banks, *tile_sizes]
        # Per-slot async-DMA semaphore bank ([num_banks] int64), owned by this
        # ref (Pallas ``BufferedRef.create``): the copy signals a slot and a
        # once-per-block-guarded wait consumes it, so the copy->wait dependency
        # rides the shared semaphore.  Never referenced outside this ref, so it
        # is allocated here rather than threaded through ``forward``.
        self.sem = voyager.zeros([num_banks], torch.int64)

    # --- addressing (shared by both kinds) ----------------------------------

    def _block_address(self, grid_idx):
        """The ``(dims, indices)`` addressing this operand's tile for
        ``async_copy`` at ``grid_idx``: a spec dim is dynamically indexed iff the
        grid dim it maps to is tiled (``grid > 1``) and isn't broadcast.  Whole
        / broadcast / ``None``-mapped dims stay at block 0 (omitted).  ``dims``
        is ``None`` when every dim is dynamic (``async_copy``'s "all dims"
        shorthand).
        """
        spec = self.spec
        bcast = getattr(spec, "is_broadcast", None)
        dims, indices = [], []
        for d, g in enumerate(spec.index_map):
            if (
                g is not None
                and self.grid[g] > 1
                and not (bcast is not None and bcast[d])
            ):
                dims.append(d)
                indices.append(grid_idx[g])
        if len(dims) < len(spec.index_map):
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
        spec = self.spec
        bcast = getattr(spec, "is_broadcast", None)
        differ = None
        for d, g in enumerate(spec.index_map):
            if (
                g is not None
                and self.grid[g] > 1
                and not (bcast is not None and bcast[d])
            ):
                term = cur[g] != next[g]
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
        no counter ``sym_ite``.
        """
        inner = self._innermost_tiled_dim()
        if inner is None:
            return False
        spec = self.spec
        bcast = getattr(spec, "is_broadcast", None)
        return any(
            g is not None
            and g == inner
            and not (bcast is not None and bcast[d])
            for d, g in enumerate(spec.index_map)
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

    def _load_tile(self, src, dst, grid_idx, sem):
        """Async-DMA ``src``'s tile at ``grid_idx`` into SRAM ``dst``, carrying
        the input halo (``strides`` / ``pad`` / ``pad_value``) and signaling the
        load semaphore ``sem``.
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
        )

    def _copy_in(self, src, slot, fetch_idx, should_copy, sem):
        """Async-DMA ``fetch_idx``'s block of ``src`` into SRAM ``slot``
        (signaling ``sem``) when ``should_copy``, else a no-op.

        Store-in-cond pattern: the ``do`` branch mutates the captured ``slot`` /
        ``sem`` in place and returns an unused ``int`` kept alive by
        ``has_side_effect(cond)``.  The caller advances the producer cursor
        under the same ``should_copy``.
        """

        def do():
            self._load_tile(src, slot, fetch_idx, sem)
            return 1

        def skip():
            return 0

        torch.cond(should_copy, do, skip)

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

    def _copy_out(self, src, dst, sc, cur, next, last, sem):
        """Store ``src`` -> ``dst``'s block at ``cur`` (signaling ``sem``) and
        return ``next_count``.  Unconditional when the output advances every
        step; otherwise guarded (store-in-cond) so a reduction writes once per
        output tile — when its block completes or on the last step.
        """
        if self._advances_every_step():
            self._store_tile(src, dst, cur, sem)
            return sc + 1

        should_store = self._indices_differ(cur, next) | last

        def do():
            self._store_tile(src, dst, cur, sem)
            return 1

        def skip():
            return 0

        torch.cond(should_store, do, skip)
        return torch.sym_ite(should_store, sc + 1, sc)

    # --- input phases (kind == _IN) -----------------------------------------

    def prime_prologue(self, src):
        """Prime the first ``D`` logical positions from DRAM ``src``,
        deduplicating reused blocks (positions are static concrete coords, so
        the dedup is a Python ``if``).  Return the seed producer count for a
        guarded input, or ``None`` for an always-advance input (no cursor).
        """
        num_copies = 0
        prev_idx = None
        for p in range(min(self.D, self.num_steps)):
            idx = self._unravel(p)
            if p == 0 or self._indices_differ(prev_idx, idx):
                slot = num_copies % self.num_banks
                self._load_tile(src, self.bank[slot], idx, self.sem[slot])
                num_copies += 1
            prev_idx = idx
        if self._advances_every_step():
            return None
        return num_copies

    def copy_in(self, ctx, src, load_count):
        """Phase 1 — prefetch the block ``D`` steps ahead of DRAM ``src`` into
        ``copy_slot``, signaling its load semaphore.  An always-advance input
        copies unconditionally (gated only by ``has_fetch``) and carries no
        cursor; a guarded one copies only when a new block enters the window
        edge and advances its producer cursor.  Returns the advanced producer
        count (guarded) or ``None`` (always-advance).
        """
        nb = self.num_banks
        w = ctx.windows[nb]
        if self._advances_every_step():
            copy_slot = (ctx.step + self.D) % nb
            should_copy = w.has_fetch
            next_count = None
        else:
            pc = load_count
            copy_slot = pc % nb
            differ = self._indices_differ(w.prev_edge, w.fetch_idx)
            # Single buffer: copy on the first step or a block change; else
            # prefetch-gated.
            should_copy = (
                (w.first | differ) if nb == 1 else (w.has_fetch & differ)
            )
            next_count = torch.sym_ite(should_copy, pc + 1, pc)
        # ``_check`` against the bank's own size lets the select bound
        # resolve on the unbacked step (needed for num_banks >= 3).
        torch._check(copy_slot < self.bank.size(0))
        self._copy_in(
            src,
            self.bank[copy_slot],
            w.fetch_idx,
            should_copy,
            self.sem[copy_slot],
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
        _guarded_wait(self.sem[rs], pred)
        return self.bank[rs]

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
        _guarded_wait(self.sem[slot], pred)
        return self.bank[slot], slot

    def copy_out(self, ctx, dst, store_count, out_slot, slot_idx):
        """Phase 5 — store the completed output tile ``out_slot`` to DRAM
        ``dst`` (guarded), signaling its store semaphore; return the advanced
        store counter.
        """
        return self._copy_out(
            out_slot,
            dst,
            store_count,
            ctx.cur,
            ctx.next,
            ctx.last,
            self.sem[slot_idx],
        )

    def drain(self, final_store_count):
        """Finalize — drain each slot's last (un-reused) store so the DRAM
        result is complete.  Slot ``j`` holds a pending store iff ``j <
        final_store_count`` (a small grid leaves the rest un-signaled).
        """
        for j in range(self.num_banks):
            _guarded_wait(self.sem[j], j < final_store_count)


class PipelinedKernel(torch.nn.Module):
    """Spec-driven, mutate-style kernel scheduler (see module docstring).

    ``kernel(grid_index, *in_tiles, *out_banks)`` is the per-tile compute; it
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

    def _alloc_in_bank(self, num_banks, tile_sizes, dtype):
        """One SRAM input bank: ``num_banks`` slots, leading bank dim."""
        return voyager.alloc([num_banks] + list(tile_sizes), dtype, _SRAM)

    def _alloc_out_bank(self, num_banks, tile_sizes, dtype):
        """One SRAM output bank: ``num_banks`` slots, leading bank dim.  The
        kernel accumulates into slot ``store_count % num_banks``; on a completed
        tile that slot is stored while the next tile accumulates into the other
        slot (store / compute overlap).
        """
        return voyager.alloc([num_banks] + list(tile_sizes), dtype, _SRAM)

    def forward(self, *inputs):
        # Pallas-style software pipelining (depth ``num_banks``, ``D =
        # num_banks - 1`` blocks ahead) with guarded DMA + per-slot semaphore
        # waits.  Each step is Pallas's ``copy_in, wait_in, kernel, copy_out``
        # (plus a ``wait_out`` reuse drain); the per-operand state machines live
        # in ``_BufferedRef`` and are driven here in phase order.  ``num_banks
        # == 2`` is one-step-ahead double buffering; ``num_banks == 1`` (D == 0)
        # is single buffering (copy current block, wait, read — no prefetch).
        out_bufs = [voyager.alloc(s.shape, s.dtype) for s in self.out_specs]
        tiled = [
            (inp, s) for inp, s in zip(inputs, self.in_specs) if s is not None
        ]
        # Per-operand software-pipeline depth: each operand's own ``num_banks``
        # (spec override, else the scheduler default), so a reused / low-reuse
        # operand can run a shallower or deeper pipeline than its peers.
        in_counts = [self._num_banks(s) for _, s in tiled]
        out_counts = [self._num_banks(s) for s in self.out_specs]
        in_banks = [
            self._alloc_in_bank(c, s.tile_sizes, inp.dtype)
            for (inp, s), c in zip(tiled, in_counts)
        ]
        out_banks = [
            self._alloc_out_bank(c, s.tile_sizes, s.dtype)
            for s, c in zip(self.out_specs, out_counts)
        ]
        # Per-operand schedulers: each owns its SRAM window (bank) and allocates
        # its own per-slot semaphore; the DRAM buffer is threaded in per phase
        # call (Pallas style).  Constructed here (after the banks, before the
        # scratch refs) so the semaphore ``zeros`` keep their original position.
        in_refs = [
            _BufferedRef(_BufferedRef._IN, s, self.grid, c, bank)
            for (inp, s), c, bank in zip(tiled, in_counts, in_banks)
        ]
        out_refs = [
            _BufferedRef(_BufferedRef._OUT, s, self.grid, c, bank)
            for s, c, bank in zip(self.out_specs, out_counts, out_banks)
        ]

        # Scratch refs: single buffer (not ``num_banks``-deep — reused
        # immediately for the next tile's reduction while the output bank stays
        # buffered until its DMA drains), captured like ``out_banks``.
        scratch_refs = tuple(
            voyager.alloc(s.shape, s.dtype, _SRAM) for s in self.scratch_specs
        )

        num_outputs = len(self.out_specs)
        # Distinct buffer counts in use (first-appearance order); one prefetch
        # window is built per distinct count each step (see ``body_fn``).
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
            cur = voyager.delinearize_index(step, self.grid)
            next = voyager.delinearize_index(step + 1, self.grid)
            prev = voyager.delinearize_index(step - 1, self.grid)
            last = step + 1 >= self.num_steps

            # One prefetch window per distinct buffer count: an operand of depth
            # ``count`` reads slot ``step % count`` and fetches ``count - 1``
            # blocks ahead (``prev_edge`` is the block one before the window
            # edge).  With a uniform depth this is a single window — the
            # pre-per-spec graph unchanged.
            windows = {}
            for count in distinct_counts:
                d = count - 1
                cur_slot = step % count
                fetch_step = step + d
                has_fetch = fetch_step < self.num_steps
                first = None
                if d == 0:
                    # single buffering: no prefetch — fetch the current block.
                    first = step == 0
                    fetch_idx = cur
                    prev_edge = prev
                elif d == 1:
                    fetch_idx, prev_edge = next, cur
                else:
                    fetch_idx = voyager.delinearize_index(fetch_step, self.grid)
                    prev_edge = voyager.delinearize_index(
                        fetch_step - 1, self.grid
                    )
                windows[count] = _Window(
                    cur_slot, fetch_idx, prev_edge, has_fetch, first
                )

            ctx = _StepCtx(step, cur, next, prev, last, windows)

            # 1. COPY-IN: prefetch each input; a guarded input advances its
            #    producer cursor (appended in ``tiled`` order), an
            #    always-advance one carries none.
            next_load_counts = []
            g = 0
            for i, (inp, spec) in enumerate(tiled):
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
            in_args, i, g = [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_args.append(inp)
                    continue
                ref = in_refs[i]
                if ref._advances_every_step():
                    in_args.append(ref.wait_in(ctx, None))
                else:
                    in_args.append(ref.wait_in(ctx, wait_counts[g]))
                    g += 1
                i += 1

            # 3. WAIT-OUT (reuse): drain each output slot's prior store before
            #    the kernel overwrites it.
            out_slots, out_slot_idx = [], []
            for i, spec in enumerate(self.out_specs):
                slot_ref, slot = out_refs[i].wait_out(ctx, store_counts[i])
                out_slots.append(slot_ref)
                out_slot_idx.append(slot)

            # 4. KERNEL (mutate-style: writes the output bank slots).  Scratch
            #    refs follow the input/output args (Pallas's *index, *inputs,
            #    *outputs, *scratch convention).
            self.kernel(cur, *in_args, *out_slots, *scratch_refs)

            # 5. COPY-OUT: store each completed output tile (guarded), signaling
            #    its store semaphore; advance the store counter.
            next_store_counts = []
            for i, spec in enumerate(self.out_specs):
                next_store_counts.append(
                    out_refs[i].copy_out(
                        ctx,
                        out_bufs[i],
                        store_counts[i],
                        out_slots[i],
                        out_slot_idx[i],
                    )
                )

            # 6. Consumer advance (guarded inputs), in ``tiled`` order.
            next_wait_counts, g = [], 0
            for i, (inp, spec) in enumerate(tiled):
                if in_refs[i]._advances_every_step():
                    continue
                next_wait_counts.append(
                    in_refs[i].advance_consumer(ctx, wait_counts[g])
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


def build_pipelined_buffers(
    kernel: Callable,
    grid: Tuple[int, ...],
    in_specs: List[Optional[_InputSpec]],
    out_specs: List[_OutputSpec],
    inputs: Tuple[torch.Tensor, ...],
    *,
    scratch_specs: Sequence[_ScratchSpec] = (),
    num_banks: int = _DEFAULT_NUM_BANKS,
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """Build the bufferized FX graph (a single rolled ``while_loop`` over
    ``voyager.*`` primitives) for ``kernel`` over ``grid``.  Mirrors
    ``build_pointwise_buffers``'s export / finalize / extent-tag flow.
    """
    pattern = PipelinedKernel(
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


class _FusedInfo:
    """Parsed pieces of a fused ``call_module`` (GEMM/conv + post-op pointwise
    ops), for the GEMM / conv pipeline builders.

    ``anchor_node`` is the GEMM/conv reference op (inside the submodule, so its
    ``args`` are submodule placeholders whose ``meta['source_node']`` point back
    to the outer graph).  ``fused_gm`` runs the post-op ops as ``[acc, *fused]
    -> output(s)`` on the anchor's result tile.  ``input_values`` are the
    tensors those ops consume; ``in_specs[i]`` is that input's tile
    ``_InputSpec`` (or ``None`` for a whole input); ``in_sources[i]`` is its
    outer graph node, used to order operands canonically.  ``output_specs`` is
    one ``(full_shape, tile_shape, dtype, index_map)`` per fused output (a tuple
    ⇒ multi-output, e.g. ``quantize_mx``).
    """

    def __init__(
        self,
        anchor_node,
        is_conv,
        fused_gm,
        tiling,
        input_values,
        in_specs,
        in_sources,
        output_specs,
    ):
        self.anchor_node = anchor_node
        self.is_conv = is_conv
        self.fused_gm = fused_gm
        self.tiling = tiling
        self.input_values = input_values
        self.in_specs = in_specs
        self.in_sources = in_sources
        self.output_specs = output_specs


def _retile_mha_view(fused_gm, nb, tm) -> None:
    """Rewrite the MHA relayout ``view`` in the fused body to tile dims.

    The original view's ``M`` dim is the full (untiled) M, so on a tile it
    scrambles the data — replace it with the tile's ``tm`` and let the split
    outer (heads) auto-size via ``-1``.  View dims are ``[*batch, M, H,
    head_dim]`` (M at index ``nb``).  The body isn't ShapeProp'd, so navigate
    output -> perm -> view by structure, not shape.
    """
    out = next(n for n in fused_gm.graph.nodes if n.op == "output")
    perm = out.args[0]
    if isinstance(perm, (list, tuple)):
        perm = perm[0]
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
    4-D tensor), return the relaid-out output ``(tile_sizes, index_map)`` (the
    tile stored to the permuted block, tiling the output on its own axes) and,
    for the projection case, retile the body's ``view`` in place; else ``None``
    (the caller keeps the default, un-permuted output spec).

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
    perm = fused_ops[-1]
    if not is_mha_qkv_permute(perm):
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
    return out_tile, out_imap


def parse_fused_submodule(node, tiler=None) -> Optional["_FusedInfo"]:
    """Parse a fused ``call_module`` ``node`` into a ``_FusedInfo``, or ``None``
    if its anchor is not a GEMM/conv.

    The submodule (``node.meta['submodule']``) holds a GEMM/conv anchor followed
    by post-op pointwise ops.  The fused operands / outputs tile at the output
    block, diced from the anchor's per-dim tile factors (``get_tiling``,
    projected to the output's physical layout).  The factors are stashed on
    ``_FusedInfo.tiling`` so the builder reuses them (no second tiler run).
    """
    submod = node.meta.get("submodule")
    anchor = get_anchor_node(node)
    is_conv = is_conv2d(anchor)
    if not is_conv and not is_gemm_op(anchor):
        return None

    ShapeProp(submod).propagate(
        *(n.value.clone() for n in node.all_input_nodes)
    )

    # ``tiling`` = the anchor's per-dim tile counts (logical).  ``out_tiling`` =
    # those in the output's physical layout with the reduction dim dropped (the
    # divisor that dices fused operands / outputs).  ``out_index_map`` = output
    # dim -> grid dim: conv projects the physical (NHWC) layout, gemm is
    # identity (K reduction is the last grid dim).  ``None`` => untiled (whole).
    tiling = get_tiling(node, tiler)
    if tiling is None:
        out_tiling = None
        out_index_map = None
    elif is_conv:
        ny, nx, _, nk = tiling  # logical (Y, X, C, K) counts
        odims = _NHWC if anchor.meta.get("transposed", False) else None
        out_tiling = _project((1, nk, ny, nx), odims)  # physical output counts
        out_index_map = _project((0, 1, 2, 3), odims)
    else:
        out_tiling = tiling[:-1]  # gemm (batch.., n_m, n_n, n_k) -> drop K
        out_index_map = tuple(range(len(out_tiling)))

    # Walk the fused pointwise chain after the anchor, collecting each op and,
    # for every new placeholder it consumes, the node, its cloned value, and its
    # tile ``_InputSpec`` (``None`` for a codebook / scalar passed whole, or an
    # untiled op).  The anchor's own operands (act / weight / scales) are
    # handled by the builder, not as fused post-op operands — exclude them here.
    anchor_prelude = ancestors(anchor)
    fused_ops = []
    input_nodes, input_values, in_specs = [], [], []
    for sn in submod.graph.nodes:
        if sn is anchor or sn.op != "call_function" or sn in anchor_prelude:
            continue
        fused_ops.append(sn)
        codebooks = quant_table_arg_nodes(sn)
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

    fused_gm = _build_fused_gm(submod, anchor, fused_ops, input_nodes)
    # Each fused operand's outer graph node, so the builder can order operands
    # canonically.
    in_sources = [n.meta.get("source_node", n) for n in input_nodes]

    multi_outputs = isinstance(node.value, (list, tuple))
    vals = list(node.value) if multi_outputs else [node.value]
    full_shapes = [tuple(v.shape) for v in vals]
    if out_tiling is None:
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
            out_tile, out_imap = relayout
            output_specs = [(full_shapes[0], out_tile, vals[0].dtype, out_imap)]

    return _FusedInfo(
        anchor,
        is_conv,
        fused_gm,
        tiling,
        input_values,
        in_specs,
        in_sources,
        output_specs,
    )


def _map_kernel(compute: Callable, num_outputs: int):
    """Map kernel (no cross-tile reduction): adapt a return-style
    ``compute(*in_tiles) -> Tensor | tuple`` into the scheduler's mutate-style
    ``kernel(grid_index, *in_tiles, *out_banks)``, writing each result straight
    into its output bank.  Every num_k == 1 op uses this; the reduction case
    uses ``_reduction_fused_kernel``.
    """

    def kernel(grid_index, *args):
        in_tiles = args[: len(args) - num_outputs]
        out_banks = args[len(args) - num_outputs :]
        results = compute(*in_tiles)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        for bank, value in zip(out_banks, results):
            voyager.insert(value, bank)

    return kernel


def _reduction_fused_kernel(
    compute: Callable,
    reduction_dim: int,
    last_idx: int,
    op_dtype: Optional[torch.dtype],
    out_specs: List[_OutputSpec],
    fused_gm: Optional[Callable],
    fused_operand_indices: List[int] = (),
):
    """Kernel for an op whose reduction needs > 1 tile (num_k > 1 GEMM / conv;
    the num_k == 1 map case uses ``_map_kernel``).

    ``compute(in_tiles, first)`` runs the bare op on the current tiles; on the
    ``first`` step it folds the bias straight into the op (hardware does ``op +
    bias`` in one pass).  The bias rides only the first step — the same step
    that initializes the accumulator — so bias gate and reduction init collapse
    into the single reduction ``torch.cond`` (no nested bias-gate cond).  The
    partial accumulates into a scratch ref; on the last step the completed
    accumulator is cast to ``op_dtype`` (it may accumulate wider, e.g. fp32) and
    mapped through the fused tail (if any) into the output bank(s).
    """
    num_outputs = len(out_specs)

    def kernel(grid_index, *args):
        n_in = len(args) - num_outputs - 1  # one scratch accumulator
        in_tiles = args[:n_in]
        out_banks = args[n_in : n_in + num_outputs]
        scratch = args[-1]  # the single scratch accumulator (Scratchpad)

        def to_acc(result):
            return result if op_dtype is None else result.to(scratch.dtype)

        def init():
            # First step: op with bias folded in, initializing the accumulator.
            return to_acc(compute(in_tiles, True))

        def accumulate(prev=scratch):
            # Later steps: bare op (no bias) + the running accumulator.
            return to_acc(compute(in_tiles, False)) + prev

        voyager.insert(
            torch.cond(grid_index[reduction_dim] == 0, init, accumulate),
            scratch,
        )

        # On the last reduction coord, cast the accumulator, apply the fused
        # tail once, and store each output; off the last step a no-op.
        # ``torch.cond`` captures ``scratch`` / fused operands automatically.
        def post_process():
            outs = scratch if op_dtype is None else scratch.to(op_dtype)
            if fused_gm is not None:
                fused = [in_tiles[i] for i in fused_operand_indices]
                outs = fused_gm(outs, *fused)
            if not isinstance(outs, (tuple, list)):
                outs = (outs,)
            for bank, out in zip(out_banks, outs):
                voyager.insert(out, bank)
            return 1

        def skip():
            return 0

        torch.cond(grid_index[reduction_dim] == last_idx, post_process, skip)

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


def _stamp_tile_compute_cycles(gm, anchor) -> None:
    """Copy the anchor's ``tile_compute_cycles`` -- what one of its tiled
    executions really costs, per the interstellar RuntimeCalculator -- onto
    every compute node of the nest just built, at every nesting level (loop
    body, cond branch).
    """
    cycles = anchor.meta.get("tile_compute_cycles")
    if cycles is None:
        return
    for m in gm.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        for n in m.graph.nodes:
            if n.op == "call_module" or is_compute_op(n):
                n.meta["tile_compute_cycles"] = cycles


def build_conv2d(
    node,
    *,
    num_banks: int = _DEFAULT_NUM_BANKS,
    accumulate_fp32: bool = False,
    single_buffer_tail: bool = False,
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
    info = (
        parse_fused_submodule(node, tiler) if node.op == "call_module" else None
    )
    if node.op == "call_module" and (info is None or not info.is_conv):
        return None

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

    # ``meta["transposed"]`` selects the systolic layout: NHWC feature maps +
    # HWIO weight.  Specs are logical NCHW/OIHW and ``_project``-ed onto each
    # operand's physical order; the grid stays logical (``None`` => identity).
    nhwc = anchor.meta.get("transposed", False)
    in_dims = _NHWC if nhwc else None
    w_dims = _HWIO if nhwc else None
    out_dims = _NHWC if nhwc else None

    N, C, H, W = _unproject(inp.shape, in_dims)
    K, _, kH, kW = _unproject(w.shape, w_dims)
    oH, oW = _unproject(out.shape, out_dims)[2:]

    # ``tiling`` is the logical output/reduction tile factors ``(n_y, n_x, n_c,
    # n_k)`` (``None`` => untiled); tile sizes are ``full // factor``.  Batch is
    # never tiled; ``n_c`` (the C reduction) is the kernel's ``num_k``.
    if tiling is None:
        tiling = (1, 1, 1, 1)
    ny, nx, nc, nk = tiling
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

    # Assemble operands in the fused node's ``all_input_nodes`` order (keyed by
    # each anchor placeholder's ``source_node``) and dispatch by canonical index,
    # so the placeholder splice binds correctly even when a fused-tail operand is
    # graph-ordered before the anchor (mirrors build_gemm).
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
        if info is not None:
            return info.fused_gm(result, *[in_tiles[i] for i in fused_idx])
        return result

    if num_k > 1:
        if single_buffer_tail:
            _single_buffer_reduction_operands(in_specs, out_specs, fused_idx)
        acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
        scratch_specs = [
            _ScratchSpec(_project((tn, tk, toh, tow), out_dims), acc_dtype)
        ]
        kernel = _reduction_fused_kernel(
            conv2d_kernel,
            reduction_dim=4,
            last_idx=num_k - 1,
            out_specs=out_specs,
            op_dtype=(out.dtype if acc_dtype != out.dtype else None),
            fused_gm=info.fused_gm if info is not None else None,
            fused_operand_indices=fused_idx,
        )
    else:
        scratch_specs = []
        kernel = _map_kernel(compute, len(out_specs))
    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs,
        tuple(inputs),
        scratch_specs=scratch_specs,
        num_banks=num_banks,
    )
    if num_k > 1 or info is not None:
        _fuse_tail_in_body(
            gm, anchor.target, fuse_anchor_with_tail=(num_k == 1)
        )
    _stamp_tile_compute_cycles(gm, anchor)
    return gm


def _detect_weight_transpose(node: torch.fx.Node):
    """Resolve a GEMM weight operand through a last-two-dim transpose / permute
    to the un-transposed external operand it relayouts.

    Returns ``(node, transposed)``.  An attention ``Q @ Kᵀ`` fuses
    ``K.transpose(-2, -1)`` onto the weight; it is peeled to ``(K, True)``.  The
    weight spec keeps the matmul (Kᵀ) layout, but the DRAM buffer is K's own
    ``(N, K)`` storage, so ``_load_tile`` swaps the fetch to read it and
    ``async_copy`` ``.mT``s the loaded tile into the bank.  Only a transpose of
    an external operand (placeholder) is peeled; anything else is returned
    unchanged as ``(operand, False)``.
    """
    if node.op != "call_function":
        return node, False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node) or src.op != "placeholder":
        return node, False
    ndim = src.value.ndim
    if node.target is torch.ops.aten.transpose.int:
        if {a % ndim for a in node.args[1:]} == {ndim - 2, ndim - 1}:
            return src, True
    elif node.target is torch.ops.aten.permute.default:
        swapped = list(range(ndim))
        swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
        if list(node.args[1]) == swapped:
            return src, True
    return node, False


def build_gemm(
    node,
    *,
    num_banks: int = _DEFAULT_NUM_BANKS,
    accumulate_fp32: bool = False,
    single_buffer_tail: bool = False,
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
    info = (
        parse_fused_submodule(node, tiler) if node.op == "call_module" else None
    )
    if node.op == "call_module" and info is None:
        return None

    tiling = info.tiling if info is not None else get_tiling(node, tiler)
    if info is None and tiling is None and not is_bmm(node):
        return None

    anchor = info.anchor_node if info is not None else node
    if not (is_linear(anchor) or is_matmul(anchor)):
        return None

    # A fused weight transpose (attention ``Q @ Kᵀ``): load the un-transposed
    # external operand and transpose the tile in the DMA (the weight spec's
    # ``transposed``); the spec itself stays in the matmul (Kᵀ) layout.
    weight_node, transposed = _detect_weight_transpose(anchor.args[1])

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

    # Weight storage layout: C-major ``(K, N)`` iff ``ck``, else K-major
    # ``(N, K)``.  matmul's weight is naturally C-major and linear's K-major;
    # ``meta["transposed"]`` flips it (``ck = is_matmul XOR transposed``).
    # ``_proj`` orders an (N, K) pair into that layout; weight and scale share
    # it.
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

    # Every kernel input maps to one entry of ``node.all_input_nodes`` via the
    # anchor placeholder's ``source_node``.  Collect ``outer_node -> (value,
    # spec)``, materialize ``inputs`` / ``in_specs`` in ``all_input_nodes``
    # order, and record each role's canonical tile index for ``compute``.
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

    op = anchor.target
    num_k = K // tk  # reduction tiles (grid extent along ``gk``)

    def gemm_kernel(in_tiles, first):
        """The bare GEMM op on the current tiles, by canonical index.  On the
        ``first`` reduction step the bias folds straight in; later steps only
        accumulate partials, which must not re-add it.
        """
        act_tile = in_tiles[act_idx]
        weight_tile = in_tiles[weight_idx]
        kw = {name: in_tiles[i] for name, i in kw_idx.items()}
        kw.update(scalar_kwargs)  # block_size / weight_layout / ...
        if bias_idx is None:
            return op(act_tile, weight_tile, **kw)
        bias = in_tiles[bias_idx] if first else None
        return op(act_tile, weight_tile, bias, **kw)

    def compute(*in_tiles):
        # num_k == 1 map: GEMM in one step, then the fused tail (if any).
        result = gemm_kernel(in_tiles, True)
        if info is not None:
            return info.fused_gm(result, *[in_tiles[i] for i in fused_idx])
        return result

    if num_k > 1:
        if single_buffer_tail:
            _single_buffer_reduction_operands(in_specs, out_specs, fused_idx)
        acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
        scratch_specs = [_ScratchSpec(tuple(tb) + (tm, tn), acc_dtype)]
        kernel = _reduction_fused_kernel(
            gemm_kernel,
            reduction_dim=gk,
            last_idx=num_k - 1,
            out_specs=out_specs,
            op_dtype=(out.dtype if acc_dtype != out.dtype else None),
            fused_gm=info.fused_gm if info is not None else None,
            fused_operand_indices=fused_idx,
        )
    else:
        scratch_specs = []
        kernel = _map_kernel(compute, len(out_specs))
    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs,
        tuple(inputs),
        scratch_specs=scratch_specs,
        num_banks=num_banks,
    )
    if num_k > 1 or info is not None:
        _fuse_tail_in_body(
            gm, anchor.target, fuse_anchor_with_tail=(num_k == 1)
        )
    _stamp_tile_compute_cycles(gm, anchor)
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
    # Tiling comes from the anchor's ``l2_tiling`` — per-output-dim tile counts
    # set by the vector L2 tiling pass.  A non-fused op with no ``l2_tiling`` is
    # bufferized whole-tensor elsewhere; a fused untiled submodule falls back to
    # whole tensor (trip-1) below.
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
            for cb in quant_table_arg_nodes(sn):
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
        codebooks = quant_table_arg_nodes(node)

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

    # Tiling comes from the anchor's ``l2_tiling`` — logical ``(n_N, n_H, n_W,
    # n_C)`` tile factors set by the pool L2 tiling pass.  A non-fused op with
    # no ``l2_tiling`` is bufferized whole-tensor elsewhere; a fused untiled
    # submodule falls back to whole tensor (trip-1) below.
    tiling = anchor.meta.get("l2_tiling")
    if node.op != "call_module" and tiling is None:
        return None

    in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
    input_t = in_node.value.clone()

    val = node.value
    outputs = val if isinstance(val, (list, tuple)) else (val,)
    output_shape = tuple(outputs[-1].shape)

    in_dims = _NHWC if anchor.meta.get("transposed", False) else None
    # ``_unproject`` gives the logical ``(N, C, H, W)`` output; tile sizes are
    # ``dim // factor``.  ``l2_tiling`` is logical ``(n_N, n_H, n_W, n_C)``
    # (``None`` => untiled).
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
            for cb in quant_table_arg_nodes(sn):
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
