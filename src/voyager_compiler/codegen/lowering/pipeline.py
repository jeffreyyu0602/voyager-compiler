"""Unified Pallas-``pallas_call``-style kernel scheduler.

One scheduler that subsumes the pointwise / pooling / GEMM bufferization
builders: given a ``kernel``, a ``grid``, and per-operand ``_InputSpec`` /
``_OutputSpec`` block specs, it emits a **single rolled** ``while_loop`` over
the flattened grid.

The scheduler is **spec-driven** for tile addressing and *mutate-style* for
compute (Pallas ``out_ref`` semantics): each grid step loads every tiled input's
current block into its SRAM bank, then calls ``kernel(grid_index, *in_tiles,
*out_banks)`` which **writes** each output SRAM bank, then stores each out bank
to DRAM.

  * An operand's *block index* at a grid point is the tuple of grid coords its
    ``index_map`` selects (dropping whole / broadcast dims); that is how the DMA
    tile is addressed.  An output whose ``index_map`` omits a grid dim does not
    move along it, so its bank stays live across that (reduction) sweep.
  * Each tiled input has a ``num_banks``-deep SRAM bank and runs a depth-``N``
    (``N = num_banks``) software pipeline ``D = N - 1`` blocks ahead: the
    prologue primes the first ``D`` distinct blocks (slots ``0..D-1``); each
    step *prefetches* the block ``D`` ahead (a guarded async DMA — skipped when
    no new block enters the window edge) while the kernel computes the current
    one.  A *guarded* input (block reused across some sweep) carries a producer
    ``copy_in`` and a consumer ``wait_in`` cursor that advance on different
    events; an always-advance input needs neither (read slot ``step % N``, copy
    slot ``(step + D) % N``).  At most ``N`` distinct blocks are in flight, so
    slots never alias — a later async-DMA pass overlaps the loads.  Slots are a
    runtime ``% num_banks`` — no unrolling.
  * Each output likewise has a ``num_banks``-deep SRAM bank and a carried store
    counter.  The kernel accumulates into slot ``store_count % num_banks``
    across the output tile's reduction sweep; the tile is *stored only when its
    block completes* (its block changes next, or the last step) — a guarded
    async DMA — and the counter then flips the slot, so a completed tile's store
    overlaps the next tile's compute (store / compute double buffering).  A
    reduction thus writes each output tile once instead of every step.
  * The kernel writes an out bank via ``voyager.copy_tile``, choosing initialize vs.
    accumulate per ``grid_index`` (e.g. initialize when the reduction coord is 0) —
    the scheduler stays oblivious to which output is a reduction.

Export-driven design notes (each verified against ``export_model``):
  * In-body bank writes go through the opaque ``voyager.copy_tile`` custom op,
    never raw aten in-place: ``while_loop`` (a HOP) rejects aten in-place
    mutation of captured tensors, but custom-op ``Tensor(a!)`` mutation of a
    *captured* (closed-over) bank is allowed and persists.
  * Accumulate-vs-initialize (a reduction) uses ``torch.cond`` in its *functional*
    form (each branch returns a fresh tensor) followed by a ``copy_tile`` write;
    a map overwrites directly (no cond).
  * A **guarded DMA** is ``torch.cond`` over ``voyager.async_copy`` (or a guarded
    ``async_wait``): the branch closes over the captured bank / semaphore and
    mutates them in place, returning an unused ``int``.  The cond is kept by
    ``has_side_effect(torch.ops.higher_order.cond)`` (registered in
    ``lowering/__init__.py``); the in-place mutation of a captured buffer inside
    a branch persists.  Load/store counters advance with
    ``torch.sym_ite(changed, count + 1, count)`` outside the cond.
  * DMA ordering uses **per-slot semaphores** (a captured ``[num_banks]`` int64
    bank per input/output, not loop-carried): ``async_copy(sem)`` increments and
    a once-per-block-guarded ``async_wait(sem)`` decrements, so the copy->wait
    dependency is a write-after-write on the shared semaphore, and the eager
    counting (``assert > 0``) is a runtime check that the schedule is balanced.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.utils import (
    _InputSpec,
    _OutputSpec,
    _ScratchSpec,
    _compute_input_spec,
    _finalize_exported_gm,
    _fuse_tail_in_body,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)
from voyager_compiler.codegen.lowering.ops import MemoryLevel

# ``mapping`` / ``lowering.utils`` / ``lowering.bufferization`` do not import
# this module at module scope (bufferization imports the builders function-
# locally), so these are top-level.
from voyager_compiler.codegen.mapping import get_anchor_node
from voyager_compiler.codegen.lowering.bufferization import (
    annotate_tensor_spaces,
)

_SRAM = int(MemoryLevel.SRAM)


class PipelinedKernel(torch.nn.Module):
    """Spec-driven, mutate-style kernel scheduler (see module docstring).

    ``kernel(grid_index, *in_tiles, *out_banks)`` is the per-tile compute; it
    **writes** each output SRAM bank (via ``voyager.copy_tile``) rather than
    returning a value. ``grid`` is the iteration space (tiles per
    dim); ``in_specs`` / ``out_specs`` are the per-operand block specs (a
    ``None`` input spec is a whole / scalar / codebook operand, passed through
    un-tiled).  ``num_banks`` is the software-pipeline depth (2 = double
    buffering): each tiled input gets a ``(num_banks, *tile)`` SRAM bank and
    each step prefetches the *next* block into the alternate slot while
    computing from the current one.
    """

    def __init__(
        self,
        kernel: Callable,
        grid: Tuple[int, ...],
        in_specs: List[Optional[_InputSpec]],
        out_specs: List[_OutputSpec],
        scratch_specs: Sequence[_ScratchSpec] = (),
        num_banks: int = 2,
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

    def _block_address(self, spec, grid_idx):
        """The ``(dims, indices)`` addressing an operand's tile for
        ``copy_tile`` at ``grid_idx``: a spec dim is dynamically indexed iff the
        grid dim it maps to is tiled (``grid > 1``) and isn't broadcast — those
        dims and their indices are returned.  Whole / broadcast dims stay at
        block 0 (omitted; ``copy_tile`` defaults them).  ``dims`` is ``None``
        when *every* dim is dynamic (``copy_tile``'s "all dims" shorthand).
        """
        bcast = getattr(spec, "is_broadcast", None)
        dims, indices = [], []
        for d, g in enumerate(spec.index_map):
            if self.grid[g] > 1 and not (bcast is not None and bcast[d]):
                dims.append(d)
                indices.append(grid_idx[g])
        # some whole / broadcast dims -> explicit dim list
        if len(dims) < len(spec.index_map):
            return dims, indices
        # every dim dynamic -> copy_tile's "all dims" shorthand
        return None, indices

    def _indices_differ(self, spec, cur, next):
        """Whether the operand's tile block changes between grid points ``cur``
        and ``next`` — the OR over its tiled (non-broadcast) dims of ``cur[g] !=
        next[g]``.  A chained ``|`` of ``SymBool``s — no Python ``any`` short-
        circuit (which would data-dependent-guard inside the traced loop) and no
        mixed-radix arithmetic, just the per-dim comparisons.  This is the load
        / store change predicate.
        """
        bcast = getattr(spec, "is_broadcast", None)
        differ = None
        for d, g in enumerate(spec.index_map):
            if self.grid[g] > 1 and not (bcast is not None and bcast[d]):
                term = cur[g] != next[g]
                differ = term if differ is None else (differ | term)
        # No tiled (non-broadcast) dim -> the block never changes.  Seeding the
        # OR with the first term (not ``False``) avoids a redundant ``False |
        # ...`` node in the traced loop.
        return False if differ is None else differ

    def _innermost_tiled_dim(self):
        """The fastest-varying tiled grid dim — the last dim with extent > 1,
        whose coord advances on *every* step in row-major delinearization
        (``None`` for a single-step grid).
        """
        tiled = [g for g in range(self.ndim) if self.grid[g] > 1]
        return tiled[-1] if tiled else None

    def _advances_every_step(self, spec):
        """Whether this operand's tile block changes on *every* grid step — i.e.
        it spans the innermost tiled grid dim (non-broadcast).  Known at build
        time from ``spec`` + ``grid``: when True the change predicate is the
        constant ``True``, so the DMA is emitted unconditionally — no
        ``torch.cond`` guard, no counter ``sym_ite``.
        """
        inner = self._innermost_tiled_dim()
        if inner is None:
            return False
        bcast = getattr(spec, "is_broadcast", None)
        return any(
            g == inner and not (bcast is not None and bcast[d])
            for d, g in enumerate(spec.index_map)
        )

    def _load_block(self, src, dst, spec, grid_idx, sem):
        """Async-DMA ``src``'s tile at ``grid_idx`` into SRAM ``dst``, carrying
        the input halo (``strides`` / ``pad`` / ``pad_value``), and signal the
        load semaphore ``sem`` (``sem[slot]``).  Used by the prologue and inside
        a guard's ``do`` branch.
        """
        dims, indices = self._block_address(spec, grid_idx)
        voyager.async_copy(
            src,
            dst,
            indices,
            spec.tile_sizes,
            sem,
            dims=dims,
            strides=spec.strides,
            pad=spec.pad,
            pad_value=spec.pad_value,
        )

    def _unravel(self, flat):
        """The row-major grid coords of flat index ``flat`` as plain Python
        ints — the build-time counterpart of ``voyager.delinearize_index``
        (mirrors ``ops._delinearize``), used for the *static* prologue
        positions so their block dedup is a Python ``if`` (no ``torch.cond``).
        """
        out = [0] * self.ndim
        for d in range(self.ndim - 1, -1, -1):
            flat, out[d] = divmod(flat, self.grid[d])
        return tuple(out)

    def _copy_in(self, dram_buf, slot, spec, fetch_idx, should_copy, sem):
        """Async-DMA ``fetch_idx``'s block of ``dram_buf`` into the SRAM bank
        ``slot`` (signaling ``sem``) when ``should_copy``, else a no-op.  The
        guarded copy is the store-in-cond pattern: the ``do`` branch closes over
        the captured ``slot`` / ``sem`` and mutates them in place (via
        ``async_copy``) and returns an ``int`` — the cond's output is unused but
        kept alive by ``has_side_effect(cond)``.  The caller advances the
        producer cursor (``sym_ite``) under the same ``should_copy``.
        """

        def do():
            self._load_block(dram_buf, slot, spec, fetch_idx, sem)
            return 1

        def skip():
            return 0

        torch.cond(should_copy, do, skip)

    def _store_block(self, input_buffers, output_buffers, spec, grid_idx, sem):
        """Async-DMA SRAM tile ``bank`` -> ``out_buf``'s block at ``grid_idx``,
        signaling ``sem``.  Used unconditionally (an output that stores every
        step) or inside a guard's ``do``.
        """
        dims, indices = self._block_address(spec, grid_idx)
        voyager.async_copy(
            input_buffers,
            output_buffers,
            indices,
            spec.tile_sizes,
            sem,
            dims=dims,
        )

    def _guarded_store(
        self, sram_buf, dram_buf, sc, spec, cur, next, last, sem
    ):
        """Store ``sram_buf`` -> ``dram_buf``'s block at ``cur`` (signaling
        ``sem``) and return ``next_count``.  Unconditional when the output
        advances every step (build-time constant); otherwise guarded (store-in-
        cond) so a reduction writes once per output tile — when its block
        completes (changes next) or on the last step.
        """
        if self._advances_every_step(spec):
            self._store_block(sram_buf, dram_buf, spec, cur, sem)
            return sc + 1

        should_store = self._indices_differ(spec, cur, next) | last

        def do():
            self._store_block(sram_buf, dram_buf, spec, cur, sem)
            return 1

        def skip():
            return 0

        torch.cond(should_store, do, skip)
        return torch.sym_ite(should_store, sc + 1, sc)

    def _guarded_wait(self, sem, pred=None):
        """``async_wait(sem)`` guarded by ``pred``, so each slot's semaphore is
        waited exactly once per signaling copy (a counting semaphore underflows
        on a stray wait).  Pallas's scheduler predicate: wait only when a new
        block is entered.  ``pred=None`` waits unconditionally — an operand whose
        block changes every step is already once-per-block.  The wait is wrapped
        in ``torch.cond`` (kept alive by ``has_side_effect``); the branch returns
        an unused ``int``.
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

    def _alloc_in_bank(self, tile_sizes, dtype):
        """One double-buffered SRAM input bank: ``num_banks`` slots, leading
        bank dim.
        """
        return voyager.alloc([self.num_banks] + list(tile_sizes), dtype, _SRAM)

    def _alloc_out_bank(self, tile_sizes, dtype):
        """One double-buffered SRAM output bank: ``num_banks`` slots, leading
        bank dim.  The kernel accumulates into slot ``store_count % num_banks``;
        on a completed tile that slot is stored while the next tile accumulates
        into the other slot (store / compute overlap).
        """
        return voyager.alloc([self.num_banks] + list(tile_sizes), dtype, _SRAM)

    def forward(self, *inputs):
        # Full Pallas-style software pipelining (depth ``num_banks``) with
        # *guarded* DMA + explicit semaphore waits.
        #
        # Each tiled input / output has a ``num_banks``-deep SRAM bank and a
        # captured per-bank ``[num_banks]`` int64 *semaphore* bank (one per slot,
        # NOT loop-carried).  The input pipeline runs ``D = num_banks - 1`` blocks
        # ahead: the prologue primes the first ``D`` distinct blocks (slots
        # ``0..D-1``); each step prefetches the block ``D`` ahead while the kernel
        # reads the current one.  A *guarded* input (block reused across some
        # sweep) carries two cursors — a producer ``copy_in`` and a consumer
        # ``wait_in`` — that advance on different events; an always-advance input
        # needs neither (its read / copy slots are pure functions of ``step``).
        # Each step (Pallas ``copy_in, wait_in, kernel, copy_out, wait_out``):
        # * COPY-IN (guarded prefetch) — async-load the block ``D`` ahead into
        #   ``copy_slot`` (skipped when no new block enters the window edge or
        #   the fetch is past the tail), signaling that slot's load semaphore.
        # * WAIT-IN — before the kernel reads ``read_slot`` (``step % nb`` for an
        #   always-advance input, the consumer cursor's slot for a guarded one),
        #   ``async_wait`` that slot's load semaphore.  Guarded once-per-block
        #   (Pallas ``has_changed | first_step``) so the counting semaphore is
        #   balanced; an always-advance input changes block every step (no guard).
        # * WAIT-OUT (reuse) — before the kernel reuses an output slot, wait on
        #   that slot's previous store so it has drained.  Guarded once-per-tile
        #   and only when the slot holds a prior store (``store_counts >= nb``).
        # * KERNEL — mutate-style, writes the captured output bank slots.
        # * COPY-OUT (guarded store) — async-store an output slot to DRAM when
        #   its block completes (block changes next, or last step), signaling its
        #   store semaphore.  After the loop a guarded finalize drains every
        #   output slot's last store before the result is returned.  Guards are
        #   ``torch.cond`` (kept by ``has_side_effect``); ``async_copy`` signals
        #   and ``async_wait`` waits the per-slot semaphore.  Slot indices are
        #   runtime ``% num_banks`` — no unrolling.  At ``num_banks == 2``
        #   (``D == 1``) this reduces to one-step-ahead double buffering;
        #   ``num_banks == 1`` (``D == 0``) is single buffering — copy the current
        #   block, wait on it, read it (no prefetch).
        #
        out_bufs = [voyager.alloc(s.shape, s.dtype) for s in self.out_specs]
        tiled = [
            (inp, s) for inp, s in zip(inputs, self.in_specs) if s is not None
        ]
        in_banks = [
            self._alloc_in_bank(s.tile_sizes, inp.dtype) for inp, s in tiled
        ]
        out_banks = [
            self._alloc_out_bank(s.tile_sizes, s.dtype) for s in self.out_specs
        ]
        # Per-slot async-DMA semaphores: one ``[num_banks]`` int64 bank per tiled
        # input (load) and per output (store), initialized to zero.  ``async_copy
        # (sem[slot])`` signals and ``async_wait(sem[slot])`` waits — the
        # per-slot semaphore persists across iterations (captured, mutated in
        # place like the SRAM banks), so it carries the copy->wait dependency the
        # old returned token used to thread through the loop's token vectors.
        sem_loads = [
            voyager.zeros([self.num_banks], torch.int64) for _ in tiled
        ]
        sem_stores = [
            voyager.zeros([self.num_banks], torch.int64) for _ in self.out_specs
        ]
        # Scratch refs: allocate once here (single buffer, not
        # ``num_banks``-deep — reused immediately for the next tile's reduction
        # while the output bank stays buffered until its DMA drains), captured
        # in the loop body like ``out_banks`` (not carried in the loop state).
        scratch_refs = tuple(
            voyager.alloc(s.shape, s.dtype, _SRAM) for s in self.scratch_specs
        )
        num_outputs = len(self.out_specs)
        nb = self.num_banks
        D = nb - 1  # prefetch distance (blocks ahead)

        # Prologue: prime the first ``D`` logical positions, deduplicating
        # reused blocks.  These positions are *static* (concrete grid coords),
        # so the dedup is a Python ``if`` — each distinct block is copied into
        # the next ring slot, signaling that slot's load semaphore.  An
        # always-advance input makes ``D`` distinct copies (slots ``0..D-1``); a
        # guarded one's prologue copy count seeds its producer cursor.
        init_copy_in = []  # guarded inputs only: prologue copy count
        for i, (inp, spec) in enumerate(tiled):
            num_copies = 0
            prev_idx = None
            for p in range(min(D, self.num_steps)):
                idx = self._unravel(p)
                if p == 0 or self._indices_differ(spec, prev_idx, idx):
                    slot = num_copies % nb
                    self._load_block(
                        inp, in_banks[i][slot], spec, idx, sem_loads[i][slot]
                    )
                    num_copies += 1
                prev_idx = idx
            if not self._advances_every_step(spec):
                init_copy_in.append(num_copies)

        def cond_fn(step, load_counts, wait_counts, store_counts):
            return step < self.num_steps

        def body_fn(step, load_counts, wait_counts, store_counts):
            cur = voyager.delinearize_index(step, self.grid)
            next = voyager.delinearize_index(step + 1, self.grid)
            prev = voyager.delinearize_index(step - 1, self.grid)
            last = step + 1 >= self.num_steps
            cur_slot = step % nb

            # The depth-``D`` prefetch window: fetch the block ``D`` steps
            # ahead, gated so no tail fetch runs past the grid end.
            # ``prev_edge`` is the block one before the window edge — a guarded
            # input copies only when a *new* block enters the edge.
            fetch_step = step + D
            has_fetch = fetch_step < self.num_steps
            if D == 0:
                # single buffering (num_banks == 1): no prefetch — fetch the
                # *current* block.
                first = step == 0
                fetch_idx = cur
                prev_edge = prev
            elif D == 1:
                # one-step pipeline: the window edge is just ``next`` / ``cur``
                fetch_idx, prev_edge = next, cur
            else:
                fetch_idx = voyager.delinearize_index(fetch_step, self.grid)
                prev_edge = voyager.delinearize_index(fetch_step - 1, self.grid)

            # 1. COPY-IN: prefetch into ``copy_slot``, signaling its load
            #    semaphore.  An always-advance input has no cursor (``copy_slot =
            #    (step + D) % nb``, ``should_copy = has_fetch``); a guarded one
            #    advances its producer cursor only when a copy fires.  When
            #    ``should_copy`` is false the block repeats at the window edge,
            #    so in-flight distinct blocks <= nb - 1 and ``copy_slot`` differs
            #    from the read slot — overwritten by a real copy before any read.
            next_load_counts = []
            g = 0
            for i, (inp, spec) in enumerate(tiled):
                if self._advances_every_step(spec):
                    copy_slot = (step + D) % nb
                    should_copy = has_fetch
                else:
                    pc = load_counts[g]
                    copy_slot = pc % nb
                    differ = self._indices_differ(spec, prev_edge, fetch_idx)
                    # D == 0 (single buffer): copy on the first step or when the
                    # block changes from the previous one; else prefetch-gated.
                    should_copy = (
                        (first | differ) if nb == 1 else (has_fetch & differ)
                    )
                    next_load_counts.append(
                        torch.sym_ite(should_copy, pc + 1, pc)
                    )
                    g += 1
                # ``_check`` against the bank's own size lets the select bound
                # resolve on the unbacked step (needed for num_banks >= 3).
                torch._check(copy_slot < in_banks[i].size(0))
                self._copy_in(
                    inp,
                    in_banks[i][copy_slot],
                    spec,
                    fetch_idx,
                    should_copy,
                    sem_loads[i][copy_slot],
                )

            # 2. WAIT-IN: wait on each input's read-slot load semaphore once per
            #    consumed block (Pallas ``has_changed | first_step``), then read
            #    it.  An always-advance input changes block every step, so its
            #    wait is unconditional; a reused input waits only on entering a
            #    new block.  ``None``-spec operands pass through in kernel order.
            in_args, i, g = [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_args.append(inp)
                    continue
                if self._advances_every_step(spec):
                    rs = cur_slot
                    pred = None
                else:
                    rs = wait_counts[g] % nb
                    g += 1
                    pred = (step == 0) | self._indices_differ(spec, prev, cur)
                torch._check(rs < in_banks[i].size(0))
                self._guarded_wait(sem_loads[i][rs], pred)
                in_args.append(in_banks[i][rs])
                i += 1

            # 3. WAIT-OUT (reuse): before the kernel reuses an output slot, wait
            #    on that slot's previous store — once per output tile (Pallas
            #    ``has_changed(output) & ~first``) and only when the slot already
            #    holds a prior store (``store_counts >= nb``); the first ``nb``
            #    uses of each slot have nothing to drain.
            out_slots, out_slot_idx = [], []
            for i, spec in enumerate(self.out_specs):
                if self._advances_every_step(spec):
                    slot = cur_slot
                else:
                    slot = store_counts[i] % nb
                torch._check(slot < out_banks[i].size(0))
                out_slot_idx.append(slot)
                pred = self._indices_differ(spec, prev, cur) & (
                    store_counts[i] >= nb
                )
                self._guarded_wait(sem_stores[i][slot], pred)
                out_slots.append(out_banks[i][slot])

            # 4. KERNEL (mutate-style: writes the output bank slots).  Scratch
            #    refs follow the input/output args (Pallas's *index, *inputs,
            #    *outputs, *scratch convention).
            self.kernel(cur, *in_args, *out_slots, *scratch_refs)

            # 5. COPY-OUT: store each completed output tile (guarded), signaling
            #    its store semaphore; advance the store counter (flip slot).
            next_store_counts = []
            for i, spec in enumerate(self.out_specs):
                new_sc = self._guarded_store(
                    out_slots[i],
                    out_bufs[i],
                    store_counts[i],
                    spec,
                    cur,
                    next,
                    last,
                    sem_stores[i][out_slot_idx[i]],
                )
                next_store_counts.append(new_sc)

            # 6. Consumer advance (guarded inputs): the current block is done
            #    when it changes next step or this is the last step.
            next_wait_counts, g = [], 0
            for inp, spec in tiled:
                if self._advances_every_step(spec):
                    continue
                wc = wait_counts[g]
                finished = last | self._indices_differ(spec, cur, next)
                next_wait_counts.append(torch.sym_ite(finished, wc + 1, wc))
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
        # result is complete.  ``store_counts`` is the total stores per output;
        # slot ``j`` holds a pending store iff ``j < total`` (a small grid stores
        # fewer than ``nb`` distinct slots, leaving the rest un-signaled).
        final_store_counts = final[3]
        for i in range(num_outputs):
            for j in range(nb):
                self._guarded_wait(sem_stores[i][j], j < final_store_counts[i])

        return out_bufs[0] if len(out_bufs) == 1 else tuple(out_bufs)


def build_pipelined_buffers(
    kernel: Callable,
    grid: Tuple[int, ...],
    in_specs: List[Optional[_InputSpec]],
    out_specs: List[_OutputSpec],
    inputs: Tuple[torch.Tensor, ...],
    *,
    scratch_specs: Sequence[_ScratchSpec] = (),
    num_banks: int = 2,
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
    # Memory-space annotation is deferred to the builders, *after* fusion — the
    # destination-passing validation needs the fused call_modules in place.
    return gm


# ---------------------------------------------------------------------------
# Op-family builders
#
# Each takes the FX ``node`` being lowered, reads its tiled tile-shapes from
# ``node.meta['tiled_shapes']`` (the tiler's per-tensor tile), and returns a
# bufferized ``GraphModule`` (over a rolled ``while_loop`` of ``voyager.*``
# primitives) that substitutes for the node, or ``None`` when the node isn't
# covered.  (The output count the substitution needs is derivable from
# ``node.value`` — a tuple for a multi-output op — so it isn't returned.) They
# mirror ``bufferization._build_for_*`` but target the pipelined scheduler: a
# return-style per-tile ``compute`` is wrapped *mutate-style* by
# ``_map_kernel`` (each result is written into its output bank), accumulating
# across the reduction grid dim for a GEMM / conv and overwriting for a map
# (pointwise / pool / layernorm·softmax).
# ---------------------------------------------------------------------------


class _FusedInfo:
    """Parsed pieces of a fused ``call_module`` (GEMM/conv + post-op fused
    pointwise ops), for the GEMM / conv pipeline builders.

    ``anchor_node`` is the GEMM/conv reference op (inside the submodule, so its
    ``args`` are submodule placeholders whose ``meta['source_node']`` point back
    to the outer graph; ``ShapeProp`` has populated their ``.value``).
    ``fused_gm`` runs the post-op pointwise ops as ``[acc, *fused] ->
    output(s)`` on the anchor's result tile.  ``fused_operands`` are the
    tensors those ops consume (a residual, a scale, …); ``fused_tiles[i]`` is
    that operand's per-dim tiled shape, or ``None`` for a whole operand.
    ``output_specs`` is one ``(full_shape, tile_shape, dtype)`` per fused output
    (a tuple ⇒ multi-output, e.g. ``quantize_mx``).
    """

    def __init__(
        self,
        anchor_node,
        is_conv,
        fused_gm,
        fused_operands,
        fused_tiles,
        output_specs,
    ):
        self.anchor_node = anchor_node
        self.is_conv = is_conv
        self.fused_gm = fused_gm
        self.fused_operands = fused_operands
        self.fused_tiles = fused_tiles
        self.output_specs = output_specs

    def operand_specs(self, out_shape, out_index_map):
        """Per fused operand, the ``(tensor, _InputSpec | None)`` for the GEMM /
        conv builders to append to the kernel's ``inputs`` / ``in_specs``: the
        operand tiled at the output block, each of its dims mapped to a grid dim
        via ``out_index_map`` (a whole codebook / scalar operand -> ``None``).
        The grid mapping is builder-specific (GEMM inserts the K reduction dim,
        conv projects the NHWC layout), so it is passed in.
        """
        pairs = []
        for operand, tile in zip(self.fused_operands, self.fused_tiles):
            if tile is None:
                pairs.append((operand, None))
                continue
            off = len(out_shape) - operand.ndim
            imap, tiles, bcast = [], [], []
            for d in range(operand.ndim):
                b = operand.shape[d] == 1 and out_shape[off + d] > 1
                imap.append(out_index_map[off + d])
                bcast.append(b)
                tiles.append(1 if b else int(tile[d]))
            pairs.append(
                (operand, _InputSpec(tuple(tiles), tuple(imap), tuple(bcast)))
            )
        return pairs


def parse_fused_submodule(node) -> Optional["_FusedInfo"]:
    """Parse a fused ``call_module`` ``node`` into a ``_FusedInfo``, or ``None``
    if its anchor is not a GEMM/conv (adapted from the retired
    ``_build_for_fused_submodule``).

    The submodule (``node.meta['submodule']``) holds a GEMM/conv anchor followed
    by post-op pointwise ops.  ``adjust_tiling`` node-keyed the SRAM-fit tiling
    onto the outer ``node`` (``meta['tiled_shapes']``, keyed by outer input
    nodes + ``node``); the fused operands tile at the output block, so their
    tiles come from ``tiled_shapes`` keyed by the outer ``source_node``.
    """
    from voyager_compiler.codegen.lowering.bufferization import (
        _codebook_arg_nodes,
    )
    from voyager_compiler.codegen.lowering.utils import _build_fused_gm
    from voyager_compiler.codegen.mapping_utils import is_conv2d, is_gemm_op
    from voyager_compiler.codegen.shape_prop import ShapeProp

    submod = node.meta.get("submodule")
    if not isinstance(submod, torch.fx.GraphModule):
        return None
    anchor = get_anchor_node(submod.graph.nodes)
    if anchor is None:
        return None
    is_conv = is_conv2d(anchor)
    if not is_conv and not is_gemm_op(anchor):
        return None

    # ShapeProp the submodule so its inner nodes (anchor + fused ops) carry
    # ``.value``; the main-graph ShapeProp does not populate submodule
    # internals.  Inputs come in ``all_input_nodes`` = placeholder order.
    ShapeProp(submod).propagate(
        *(n.value.clone() for n in node.all_input_nodes)
    )

    shapes = node.meta.get("tiled_shapes") or {}

    # Walk the ops after the anchor (the fused pointwise chain).  Collect each
    # op and, for every new placeholder it consumes, the operand tensor and its
    # tiled shape (or ``None`` for a codebook / scalar passed whole).
    reachable = {anchor}
    fused_ops, fused_inputs, fused_operands, fused_tiles = [], [], [], []
    for sn in submod.graph.nodes:
        if sn is anchor or sn.op != "call_function":
            continue
        if not any(inp in reachable for inp in sn.all_input_nodes):
            continue
        reachable.add(sn)
        fused_ops.append(sn)
        codebooks = _codebook_arg_nodes(sn)
        for inp in sn.all_input_nodes:
            if inp.op != "placeholder" or inp in fused_inputs:
                continue
            fused_inputs.append(inp)
            fused_operands.append(inp.value.clone())
            src = inp.meta.get("source_node", inp)
            if inp in codebooks or inp.value.numel() == 1:
                fused_tiles.append(None)  # whole operand
            else:
                fused_tiles.append(shapes.get(src, tuple(inp.value.shape)))

    fused_gm = _build_fused_gm(submod, anchor, fused_ops, fused_inputs)

    multi = isinstance(node.value, (list, tuple))
    vals = list(node.value) if multi else [node.value]
    full_shapes = [tuple(v.shape) for v in vals]
    keyed = shapes.get(node)  # None when the fused node is untiled
    if keyed is None:
        tile_shapes = full_shapes  # tile == full tensor (trip-1)
    else:
        tile_shapes = list(keyed) if multi else [keyed]
    output_specs = list(zip(full_shapes, tile_shapes, [v.dtype for v in vals]))

    return _FusedInfo(
        anchor, is_conv, fused_gm, fused_operands, fused_tiles, output_specs
    )


def _map_kernel(compute: Callable, num_outputs: int):
    """Map kernel (no cross-tile reduction): adapt a return-style
    ``compute(grid_index, *in_tiles) -> Tensor | tuple`` into the scheduler's
    mutate-style ``kernel(grid_index, *in_tiles, *out_banks)`` — run ``compute``
    once per tile and write each result straight into its output bank.  Every
    num_k == 1 op (gemm / conv / pointwise / pool) uses this; the cross-tile
    reduction case uses ``_reduction_fused_kernel``.
    """

    def kernel(grid_index, *args):
        in_tiles = args[: len(args) - num_outputs]
        out_banks = args[len(args) - num_outputs :]
        results = compute(grid_index, *in_tiles)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        for bank, value in zip(out_banks, results):
            voyager.copy_tile(value, bank, (0,) * bank.ndim, bank.shape)

    return kernel


def _reduction_fused_kernel(
    compute: Callable,
    reduction_dim: int,
    last_idx: int,
    op_dtype: Optional[torch.dtype],
    out_specs: List[_OutputSpec],
    fused_gm: Optional[Callable],
    num_fused_operands: int,
):
    """Kernel for an op whose reduction needs > 1 tile (every num_k > 1 GEMM /
    conv; the num_k == 1 map case uses ``_map_kernel``).

    ``compute(grid_index, in_tiles, first)`` runs the bare op (GEMM / conv) on
    the current tiles; on the ``first`` reduction step it folds the bias straight
    into the op (the hardware does ``op + bias`` in one pass).  The bias rides
    only the first reduction step — which is the same step that *initializes* the
    accumulator — so the bias gate and the reduction init collapse into the
    single reduction ``torch.cond`` (init = op-with-bias; accumulate =
    op-without-bias + scratch), with **no** nested bias-gate cond.  The partial
    accumulates into a scratch ref across the reduction sweep; on the last step
    the completed accumulator is cast to ``op_dtype`` (it may accumulate in a
    wider dtype, e.g. fp32) and mapped through the fused tail (if any) into the
    output bank(s).
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
            # First reduction step: op with the bias folded in, initializing
            # the accumulator.
            return to_acc(compute(grid_index, in_tiles, True))

        def accumulate(prev=scratch):
            # Later steps: bare op (no bias) added to the running accumulator.
            return to_acc(compute(grid_index, in_tiles, False)) + prev

        voyager.copy_tile(
            torch.cond(grid_index[reduction_dim] == 0, init, accumulate),
            scratch,
            (0,) * scratch.ndim,
            scratch.shape,
        )

        # On the last reduction coord, cast the now-complete accumulator, apply
        # the fused tail ONCE, and store each output; off the last step a no-op.
        # ``torch.cond`` captures the closed-over ``scratch`` / fused operands
        # automatically — no need to thread them through as branch operands.
        def post_process():
            outs = scratch if op_dtype is None else scratch.to(op_dtype)
            if fused_gm is not None:
                outs = fused_gm(outs, *in_tiles[n_in - num_fused_operands :])
            if not isinstance(outs, (tuple, list)):
                outs = (outs,)
            for bank, out in zip(out_banks, outs):
                voyager.copy_tile(out, bank, (0,) * bank.ndim, bank.shape)
            return 1

        def skip():
            return 0

        torch.cond(grid_index[reduction_dim] == last_idx, post_process, skip)

    return kernel


def build_conv2d(node, *, num_banks: int = 2, accumulate_fp32: bool = False):
    """Pipeline builder for a conv2d (groups=1) node — including the
    microscaling / codebook (``conv2d_mx``) variant, a fused bias, and the
    systolic NHWC layout — the input-channel (C) cross-tile reduction.  A map
    over the (N, K, oH, oW) output grid plus a reduction grid dim for C: the
    input is a strided receptive-field halo (pad-on-load, ``pad_value=0``), the
    weight is tiled on (K, C), and the kernel convolves each C-block and
    accumulates (initialize when the C coord is 0).  Grid ``(N, K, oH, oW, C, 1)`` —
    the trailing extent-1 dim holds the whole ``kH``/``kW`` weight dims; for
    ``num_k == 1`` the C dim is extent 1 (no reduction).  Specs are written in
    logical NCHW/OIHW terms and projected onto each operand's physical order
    (``meta["transposed"]`` selects NHWC input/output + HWIO weight), like
    gemm.py's TiledConv2d.  Returns the gm or ``None``.
    """
    from voyager_compiler.codegen.lowering.utils import (
        _HWIO,
        _NHWC,
        _phys_pos,
        _project,
        _unproject,
    )
    from voyager_compiler.codegen.passes.utils import _pair, get_arg_value

    # ``or {}`` (not a ``{}`` default): ``adjust_tiling`` sets the key to
    # ``None`` for an on-chip / untiled node, which a plain default wouldn't
    # catch.
    shapes = node.meta.get("tiled_shapes") or {}

    # A fused ``call_module`` (conv + post-op pointwise ops): read the op /
    # tiling off the conv anchor inside the submodule; ``anchor`` is the bare
    # ``node`` for a non-fused op.  A num_k == 1 fused op (the C reduction fits
    # one tile) applies the tail in ``compute``; num_k > 1 accumulates the conv
    # partials into a scratch ref and applies the tail once on the last C step.
    info = parse_fused_submodule(node) if node.op == "call_module" else None
    if node.op == "call_module" and (info is None or not info.is_conv):
        return None
    # A non-fused untiled op is bufferized elsewhere; a fused untiled op falls
    # back to whole-tensor tiles (trip-1 loops) below.
    if info is None and not shapes:
        return None
    anchor = info.anchor_node if info is not None else node

    inp = anchor.args[0].value.clone()
    w = anchor.args[1].value.clone()
    out = anchor.value  # the conv output (drives the N/K/oH/oW grid)
    if inp.ndim != 4 or w.ndim != 4:
        return None
    groups = get_arg_value(anchor, 6, "groups", 1)
    if groups != 1:
        return None  # doesn't support depthwise conv yet

    # ``meta["transposed"]`` selects the systolic layout: NHWC feature maps +
    # HWIO weight.  Specs are logical (NCHW/OIHW per-axis) and ``_project``-ed
    # onto each operand's physical order; the grid stays logical.  ``None`` dims
    # == logical NCHW (a plain identity projection).
    nhwc = anchor.meta.get("transposed", False)
    in_dims = _NHWC if nhwc else None
    w_dims = _HWIO if nhwc else None
    out_dims = _NHWC if nhwc else None

    # ``tiled_shapes`` is keyed by outer graph nodes: the output by ``node``,
    # the input by its outer ``source_node`` (== the node itself for a bare op).
    # A fused node that fits on-chip has no ``tiled_shapes`` entry — falling
    # back to the whole tensor makes every loop trip-1.
    in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
    out_keyed = shapes.get(node)
    if out_keyed is None:
        out_keyed = tuple(out.shape)  # untiled -> whole tensor (trip-1)
    elif info is not None and isinstance(node.value, (list, tuple)):
        out_keyed = out_keyed[
            -1
        ]  # the activation output's tile drives the grid
    # logical (tn, tk, toh, tow)
    out_ts = _unproject(out_keyed, out_dims)
    # logical (tn, tc, ., .)
    in_ts = _unproject(shapes.get(in_node, tuple(inp.shape)), in_dims)
    tn, tk, toh, tow = (int(x) for x in out_ts)
    N, C, H, W = _unproject(inp.shape, in_dims)
    K, _, kH, kW = _unproject(w.shape, w_dims)
    tc = int(in_ts[1])
    num_k = C // tc

    sh, sw = _pair(get_arg_value(anchor, 3, "stride", 1))
    ph, pw = _pair(get_arg_value(anchor, 4, "padding", 0))
    dh, dw = _pair(get_arg_value(anchor, 5, "dilation", 1))
    oH, oW = _unproject(out.shape, out_dims)[2:]
    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1

    # grid dims (logical): 0=N 1=K 2=oH 3=oW 4=C(reduction) 5=whole(kH/kW).
    grid = (N // tn, K // tk, oH // toh, oW // tow, num_k, 1)
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
        # K->1, C->4, kH/kW->5 (whole, extent 1)
        _project((1, 4, 5, 5), w_dims),
        (False,) * 4,
    )
    # The output(s) tile onto the (N, K, oH, oW) grid dims (the C reduction dim
    # 4 is dropped); a fused op may produce several (``quantize_mx``).
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
            _OutputSpec(tuple(shape), tuple(tile), out_index_map, dtype)
            for shape, tile, dtype in info.output_specs
        ]

    inputs, in_specs = [inp, w], [in_spec, w_spec]

    # Bias [K] tiles along K (grid dim 1); folded once on the C==0 step (below),
    # since later C blocks only accumulate partials into the bank.  It is the
    # first extra (``extra[0]``) when present.
    bias_n = get_arg_value(anchor, 2, "bias")
    if bias_n is not None:
        inputs.append(bias_n.value.clone())
        in_specs.append(_InputSpec((tk,), (1,), (False,)))

    # On the microscaling target (conv2d_mx) the per-block scales tile along C
    # (// block_size): input_scale shares the input halo's layout, weight_scale
    # the weight's; the codebook tables (only present on this target) load whole
    # (untiled, None spec).  Each threads to the op by kw.
    target = anchor.target
    bs = anchor.kwargs.get("block_size")

    kw_slots = {}

    def add_kw_input(name: str, spec: _InputSpec | None) -> None:
        v = anchor.kwargs.get(name)
        if not isinstance(v, torch.fx.Node):
            return

        if not hasattr(v, "value"):
            raise ValueError(
                f"Expected materialized value for FX node kwarg {name!r}"
            )

        kw_slots[name] = len(in_specs) - 2
        inputs.append(v.value.clone())
        in_specs.append(spec)

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
            _project((1, 4, 5, 5), w_dims),
            (False,) * 4,
        )

        add_kw_input("input_scale", in_scale_qspec)
        add_kw_input("weight_scale", wt_scale_qspec)

        for name in ("input_code", "weight_code"):
            add_kw_input(name, None)

    # Fused post-op operands (a residual, …): append each through ``in_specs``,
    # tiled at the output (N, K, oH, oW) block in the same physical layout as
    # the output.  A ``None`` spec is a whole (codebook / scalar) operand.
    if info is not None:
        for operand, spec in info.operand_specs(out.shape, out_index_map):
            inputs.append(operand)
            in_specs.append(spec)
    # The fused operands are the last inputs (appended above); the kernel /
    # compute pick them off the tail of ``*extra`` by count.
    num_fused = len(info.fused_operands) if info is not None else 0

    def _conv(in_tile, w_tile, bias, kw):
        return target(
            in_tile, w_tile, bias, [sh, sw], [0, 0], [dh, dw], groups, **kw
        )

    def _fix_stride(t):
        # ``torch.cond`` rejects a branch output whose stride it can't prove is a
        # simple product of sizes; a *strided* conv tile's spatial extent is the
        # symbolic ``(((H - kH)//stride) + 1)`` with a ``Max(1, .)`` clamp, whose
        # stride the cond can't represent (``.to`` / ``.contiguous`` / reshape
        # keep the ``Max(1, .)`` form).  Re-view with the **concrete** output tile
        # shape the builder already knows (``tn, tk, toh, tow``) and its dense
        # stride — no symbolic ``Max`` — via ``as_strided`` (a metadata-only NOP,
        # in ``is_nop``).  Needed independently of the accumulator cast, which the
        # reduction kernel drops when no dtype change is required.
        # TODO why this doesn't work.
        # n, c, h, w = t.shape
        # return torch.as_strided(
        #     t, size=(n, c, h, w), stride=(c * h * w, h * w, w, 1)
        # )
        d0, d1, d2, d3 = _project((tn, tk, toh, tow), out_dims)
        return torch.as_strided(
            t, size=(d0, d1, d2, d3), stride=(d1 * d2 * d3, d2 * d3, d3, 1)
        )

    def conv_partial(grid_index, in_tiles, first):
        """The bare conv op on the current tiles, dense-strided (``_fix_stride``)
        so it can feed a ``torch.cond`` branch.  On the ``first`` reduction step
        the [K] bias folds straight into the op (the hardware does conv + bias in
        one pass, taking the bias directly — no broadcast reshape); later steps
        only accumulate partials, which must not re-add it.
        """
        in_tile, w_tile, *extra = in_tiles
        kw = {name: extra[i] for name, i in kw_slots.items()}
        if bs is not None:
            kw["block_size"] = bs
        bias = extra[0] if (bias_n is not None and first) else None  # [K]
        return _fix_stride(_conv(in_tile, w_tile, bias, kw))

    def compute(grid_index, in_tile, w_tile, *extra):
        # num_k == 1 map: the conv completes in one step (bias folds straight in,
        # no reduction cond), then the fused tail (if any) is applied here.
        in_tiles = (in_tile, w_tile, *extra)
        result = conv_partial(grid_index, in_tiles, True)
        if info is not None:
            # The fused operands are the last ``num_fused`` extras.
            fused_tiles = extra[len(extra) - num_fused :]
            return info.fused_gm(result, *fused_tiles)
        return result

    if num_k > 1:
        # Tiled C reduction: accumulate the bare conv partials into a scratch
        # ref (the conv output tile); on the last C step the accumulator reaches
        # the output bank(s) — through the tail if fused, else stored directly.
        acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
        scratch_specs = [
            _ScratchSpec(_project((tn, tk, toh, tow), out_dims), acc_dtype)
        ]
        kernel = _reduction_fused_kernel(
            conv_partial,
            reduction_dim=4,
            last_idx=num_k - 1,
            out_specs=out_specs,
            op_dtype=(out.dtype if acc_dtype != out.dtype else None),
            fused_gm=info.fused_gm if info is not None else None,
            num_fused_operands=num_fused,
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
        # num_k > 1: the reduction splits work across ``torch.cond`` branches —
        # fuse the accumulate branch (anchor + cast + accumulate add), and the
        # finalize branch's tail when present (``fuse_anchor_with_tail=False``).
        # Runs even without a tail so the bare anchor isn't flagged by the
        # destination-passing check (its result feeds the add, not a store).
        # num_k == 1: the anchor + fused tail share the output body — one cone
        # (``fuse_anchor_with_tail=True``).
        _fuse_tail_in_body(
            gm, anchor.target, fuse_anchor_with_tail=(num_k == 1)
        )
    return gm


def build_gemm(node, *, num_banks: int = 2, accumulate_fp32: bool = False):
    """Pipeline builder for a linear / matmul / batched-matmul node — including
    the microscaling / codebook (``*_mx``) variants and a fused bias — covering
    the cross-tile K reduction the old pointwise engine couldn't do.  Grid ``(M,
    N, K)`` (or ``(B, M, N, K)`` for a batched matmul) tiles with K innermost;
    the kernel accumulates ``act_tile @ weight_tile`` into the output bank,
    initialize on ``k == 0``.  Returns the gm, or ``None`` (unsupported).
    """
    from voyager_compiler.codegen.mapping_utils import is_linear, is_matmul
    from voyager_compiler.codegen.passes.utils import get_arg_value

    shapes = node.meta.get("tiled_shapes") or {}

    # A fused ``call_module`` (GEMM + post-op pointwise ops): read the op /
    # tiling off the GEMM anchor inside the submodule, append the fused operands
    # through ``in_specs``, and apply the fused ops in ``compute``.  ``anchor``
    # is the bare ``node`` for a non-fused op.  A num_k == 1 fused op (the K
    # reduction fits one tile) applies the tail in ``compute``; num_k > 1
    # accumulates the GEMM partials into a scratch ref and applies the tail once
    # on the last K step.
    info = parse_fused_submodule(node) if node.op == "call_module" else None
    if node.op == "call_module" and info is None:
        return None
    # A non-fused untiled op is bufferized elsewhere (load / run / store whole);
    # a fused untiled op falls back to whole-tensor tiles (trip-1 loops) below.
    if info is None and not shapes:
        return None
    anchor = info.anchor_node if info is not None else node

    if not (is_linear(anchor) or is_matmul(anchor)):
        return None
    act = anchor.args[0].value.clone()
    weight = anchor.args[1].value.clone()
    out = anchor.value  # the GEMM output (drives the M/N/K grid)
    if not isinstance(out, torch.Tensor) or act.ndim < 2 or weight.ndim < 2:
        return None

    # ``tiled_shapes`` is keyed by outer graph nodes: the output by ``node``,
    # the activation by its outer ``source_node`` (== the node itself for a bare
    # op).  A fused node that fits on-chip has no entry — fall back to the whole
    # tensor so every loop is trip-1.
    in_node = anchor.args[0].meta.get("source_node", anchor.args[0])
    keyed_out = shapes.get(node)
    if keyed_out is None:
        out_ts = tuple(out.shape)  # untiled -> whole tensor (trip-1)
    elif info is not None and isinstance(node.value, (list, tuple)):
        out_ts = keyed_out[-1]  # the activation output's tile drives the grid
    else:
        out_ts = keyed_out
    in_ts = shapes.get(in_node, tuple(act.shape))  # (..batch.., tile_m, tile_k)
    tm, tn = int(out_ts[-2]), int(out_ts[-1])
    tk = int(in_ts[-1])
    M, K, N = act.shape[-2], act.shape[-1], out.shape[-1]

    # torch matmul broadcasts the leading ``N - 2`` batch dims: each output
    # batch dim is its own grid dim (0..nb-1), then M / N / K are grid dims nb /
    # nb+1 / nb+2 (K the innermost reduction). An operand's batch dims right-
    # align to the output's (a missing or size-1 batch dim broadcasts — pinned
    # to block 0); a 2-D weight (a batched *linear*) thus simply has no batch
    # dims to map.
    nb = out.ndim - 2
    gm, gn, gk = nb, nb + 1, nb + 2
    out_batch = tuple(out.shape[:nb])
    tb = tuple(int(x) for x in out_ts[:nb])
    grid = tuple(b // t for b, t in zip(out_batch, tb)) + (
        M // tm,
        N // tn,
        K // tk,
    )

    # Weight storage layout: C-major ``(K_reduction, N)`` iff ``weight_ck``,
    # else K-major ``(N, K_reduction)``.  matmul's weight is naturally C-major
    # and linear's K-major; ``meta["transposed"]`` (set by the relayout pass)
    # flips it — matching gemm.py's ``weight_ck = is_matmul XOR transposed``.
    # ``_proj`` orders an (output N, reduction K) pair into that layout (like
    # gemm.py's ``_wkc``); the weight and its scale share it.
    weight_ck = is_matmul(anchor) != bool(anchor.meta.get("transposed", False))
    _proj = lambda n, k: (k, n) if weight_ck else (n, k)

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
    # The output(s) tile onto the M/N grid dims (the K reduction dim ``gk`` is
    # dropped); a fused op may produce several (``quantize_mx``).
    out_index_map = tuple(range(nb)) + (gm, gn)
    if info is None:
        out_specs = [
            _OutputSpec(
                tuple(out.shape), tuple(tb) + (tm, tn), out_index_map, out.dtype
            )
        ]
    else:
        out_specs = [
            _OutputSpec(tuple(shape), tuple(tile), out_index_map, dtype)
            for shape, tile, dtype in info.output_specs
        ]

    inputs, in_specs = [act, weight], [act_spec, weight_spec]

    # Bias [N] tiles along N (grid dim ``gn``); folded once on the k==0 step
    # (below), since later steps only accumulate partials into the bank — adding
    # bias per-k would multiply-count it.  It is the first extra (``extra[0]``)
    # when present.
    bias_n = get_arg_value(anchor, 2, "bias")
    if bias_n is not None:
        inputs.append(bias_n.value.clone())
        in_specs.append(_InputSpec((tn,), (gn,), (False,)))

    # On the microscaling targets (linear_mx / matmul_mx) the per-block scales
    # tile along the reduction (// block_size, sharing the operand's batch +
    # block layout) and the codebook tables (input_code / weight_code, only
    # present on these targets) load whole (untiled, None spec); each threads to
    # the op by keyword.
    bs = anchor.kwargs.get("block_size")

    kw_slots = {}

    def add_kw_input(name: str, spec: _InputSpec | None) -> None:
        v = anchor.kwargs.get(name)
        if not isinstance(v, torch.fx.Node):
            return

        if not hasattr(v, "value"):
            raise ValueError(
                f"Expected materialized value for FX node kwarg {name!r}"
            )

        kw_slots[name] = len(in_specs) - 2
        inputs.append(v.value.clone())
        in_specs.append(spec)

    if anchor.target in (
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ):
        in_scale_qspec = _spec(act.shape, (tm, tk // bs), (gm, gk))
        wt_scale_qspec = _spec(weight.shape, _proj(tn, tk // bs), _proj(gn, gk))

        add_kw_input("input_scale", in_scale_qspec)
        add_kw_input("weight_scale", wt_scale_qspec)

        for name in ("input_code", "weight_code"):
            add_kw_input(name, None)

    op = anchor.target

    # Fused post-op operands (a residual, a scale, …): append each through
    # ``in_specs`` so the scheduler pipelines it like any tiled input, tiled at
    # the output (M/N) block.  A ``None`` tile is a whole (codebook / scalar)
    # operand passed through.  ``compute`` then applies ``info.fused_gm`` to the
    # GEMM result with these tiles.
    if info is not None:
        for operand, spec in info.operand_specs(out.shape, out_index_map):
            inputs.append(operand)
            in_specs.append(spec)
    # The fused operands are the *last* inputs (appended above), so the kernel
    # picks them off the tail of ``*extra`` by count.
    num_fused = len(info.fused_operands) if info is not None else 0

    num_k = K // tk  # reduction tiles (grid extent along ``gk``)

    def gemm_partial(grid_index, in_tiles, first):
        """The bare GEMM op on the current tiles.  On the ``first`` reduction
        step the bias folds straight into the op (``w @ x + bias`` in one pass);
        later steps only accumulate partials, which must not re-add it.
        """
        activation, weight_tile, *extra = in_tiles
        kw = {name: extra[i] for name, i in kw_slots.items()}
        if bs is not None:
            kw["block_size"] = bs
        if bias_n is None:
            return op(activation, weight_tile, **kw)
        bias = extra[0] if first else None  # bias [N] is the first extra
        return op(activation, weight_tile, bias, **kw)

    def compute(grid_index, activation, weight_tile, *extra):
        # num_k == 1 map: the GEMM completes in one step (bias folds straight in,
        # no reduction cond), then the fused tail (if any) is applied here.
        in_tiles = (activation, weight_tile, *extra)
        result = gemm_partial(grid_index, in_tiles, True)
        if info is not None:
            # The fused operands are the last ``num_fused`` extras.
            fused_tiles = extra[len(extra) - num_fused :]
            return info.fused_gm(result, *fused_tiles)
        return result

    if num_k > 1:
        # Tiled K reduction: accumulate the bare GEMM partials into a scratch
        # ref (the op output tile); on the last K step the accumulator reaches
        # the output bank(s) — through the tail if fused, else stored directly.
        acc_dtype = torch.float32 if accumulate_fp32 else out.dtype
        scratch_specs = [_ScratchSpec(tuple(tb) + (tm, tn), acc_dtype)]
        kernel = _reduction_fused_kernel(
            gemm_partial,
            reduction_dim=gk,
            last_idx=num_k - 1,
            out_specs=out_specs,
            op_dtype=(out.dtype if acc_dtype != out.dtype else None),
            fused_gm=info.fused_gm if info is not None else None,
            num_fused_operands=num_fused,
        )
    else:
        # num_k == 1: a map (the tail, if any, is applied inside ``compute``).
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
        # num_k > 1: the reduction splits work across ``torch.cond`` branches —
        # fuse the accumulate branch (anchor + cast + accumulate add), and the
        # finalize branch's tail when present (``fuse_anchor_with_tail=False``).
        # Runs even without a tail so the bare anchor isn't flagged by the
        # destination-passing check (its result feeds the add, not a store).
        # num_k == 1: the anchor + fused tail share the output body — one cone
        # (``fuse_anchor_with_tail=True``).
        _fuse_tail_in_body(
            gm, anchor.target, fuse_anchor_with_tail=(num_k == 1)
        )
    return gm


def build_pointwise(node, *, num_banks: int = 2):
    """Pipeline builder for a pointwise / batched-reduction node (elementwise
    ops, and layernorm·softmax whose reduction dim is kept whole in the tile).
    Tiles the output grid and writes each output tile once (no cross-tile
    reduction).  Returns the gm, or ``None``.
    """
    from voyager_compiler.codegen.lowering.bufferization import (
        _codebook_arg_nodes,
    )

    val = getattr(node, "value", None)
    if not isinstance(val, (torch.Tensor, list, tuple)):
        return None
    tiled_shapes = node.meta.get("tiled_shapes") or {}
    # A non-fused untiled op is bufferized elsewhere; a fused untiled submodule
    # falls back to whole-tensor tiles (trip-1) below.
    if node.op != "call_module" and not tiled_shapes:
        return None

    outputs = list(val) if isinstance(val, (list, tuple)) else [val]
    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]

    if node.op == "call_module":
        # Fused pointwise submodule (e.g. ``relu(x + residual)``): there is no
        # reduction, so the whole submodule is the per-tile compute — every
        # input tiles at the output block (codebooks / scalars passed whole) and
        # the submodule runs on the loaded tiles.  No anchor / fused-op split is
        # needed (unlike the GEMM / conv builders, whose reduction sweep keeps
        # the anchor separate).
        submod = node.meta.get("submodule")
        if not isinstance(submod, torch.fx.GraphModule):
            return None
        # Codebook operands (passed whole): map the submodule's codebook
        # placeholders back to their outer input nodes.
        codebooks = set()
        for sn in submod.graph.nodes:
            if sn.op != "call_function":
                continue
            for cb in _codebook_arg_nodes(sn):
                codebooks.add(cb.meta.get("source_node", cb))

        def compute(grid_index, *tiles):
            return submod(*tiles)

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
        codebooks = _codebook_arg_nodes(node)

        def compute(grid_index, *tiles):
            args = [
                tiles[i] if i is not None else a
                for i, a in zip(arg_slots, op_args)
            ]
            kwargs = {
                k: tiles[i] if i is not None else op_kwargs[k]
                for k, i in kw_slots.items()
            }
            return op(*args, **kwargs)

    output_ts = tiled_shapes.get(node)
    output_shape = tuple(outputs[-1].shape)
    if output_ts is None:
        output_ts = output_shape  # untiled -> whole tensor (trip-1)
    elif isinstance(val, (list, tuple)):
        output_ts = output_ts[-1]
    grid = tuple(s // t for s, t in zip(output_shape, output_ts))
    in_specs = [
        (
            _compute_input_spec(output_shape, output_ts, tuple(n.shape))
            if n not in codebooks
            else None
        )
        for n in in_nodes
    ]
    out_specs = [
        _OutputSpec(
            tuple(o.shape),
            tuple(min(t, s) for t, s in zip(output_ts, o.shape)),
            tuple(range(o.ndim)),
            o.dtype,
        )
        for o in outputs
    ]
    kernel = _map_kernel(compute, len(outputs))
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, out_specs, tuple(inputs), num_banks=num_banks
    )

    if node.op == "call_module":
        # Group the fused pointwise ops into one nested call_module for
        # codegen's ``fused_op``.  There is no reduction here, so (like the GEMM
        # / conv num_k == 1 path) the anchor — the first pointwise op, per
        # ``get_anchor_node`` — sits in the body with its tail, and the forward
        # cone from it is the whole chain.
        anchor = get_anchor_node(submod.graph.nodes)
        if anchor is not None:
            _fuse_tail_in_body(gm, anchor.target)
    return gm


_POOL2D_SUPPORTED = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
    # NHWC variant after the data-layout transform (same schema as aten's);
    # ``build_pool``'s ``"max_pool" in str(target)`` detection covers it.
    torch.ops.quantized_ops.max_pool2d.default,
}


def build_pool(node, *, num_banks: int = 2):
    """Pipeline builder for a 2-D max/avg pool node: a map over the (N, C, oH,
    oW) output grid whose input tile is a strided receptive-field *halo*
    (overlap = the kernel footprint) with the boundary padding folded into the
    load (``copy_tile``'s ``pad`` / ``pad_value``), so the kernel pools each
    halo with ``padding=0``.  Returns the gm, or ``None``.
    """
    from voyager_compiler.codegen.lowering.utils import (
        _NHWC,
        _project,
        _unproject,
    )
    from voyager_compiler.codegen.passes.utils import _pair, get_arg_value

    if node.target not in _POOL2D_SUPPORTED:
        return None
    nhwc = bool(node.meta.get("transposed", False))
    in_dims = _NHWC if nhwc else None
    input_t = node.args[0].value.clone()
    # ``or {}``: ``adjust_tiling`` may set the key to ``None`` (on-chip op).
    shapes = node.meta.get("tiled_shapes") or {}
    output_ts = shapes.get(node, tuple(node.value.shape))
    tn, tc, toh, tow = _unproject(output_ts, in_dims)

    kernel_size = get_arg_value(node, 1, "kernel_size")
    stride = get_arg_value(node, 2, "stride", [])
    padding = get_arg_value(node, 3, "padding", 0)
    if "max_pool" in str(node.target):
        dilation = get_arg_value(node, 4, "dilation", 1)
        ceil_mode = get_arg_value(node, 5, "ceil_mode", False)
        extra_args = (dilation, ceil_mode)
        pad_value = float("-inf")
    else:
        ceil_mode = get_arg_value(node, 4, "ceil_mode", False)
        count_include_pad = get_arg_value(node, 5, "count_include_pad") is None
        dilation = 1
        extra_args = (
            ceil_mode,
            count_include_pad,
            get_arg_value(node, 6, "divisor_override"),
        )
        pad_value = 0.0

    N, C, H, W = _unproject(input_t.shape, in_dims)
    kH, kW = _pair(kernel_size)
    sh, sw = _pair(stride) if stride else (kH, kW)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1

    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1
    step_h, step_w = toh * sh, tow * sw

    out_tile = _project((tn, tc, toh, tow), in_dims)
    out_shape = _project((N, C, oH, oW), in_dims)
    grid = tuple(s // t for s, t in zip(out_shape, out_tile))
    in_spec = _InputSpec(
        tile_sizes=_project((tn, tc, ih, iw), in_dims),
        index_map=(0, 1, 2, 3),
        is_broadcast=(False,) * 4,
        strides=_project((tn, tc, step_h, step_w), in_dims),
        pad=_project((0, 0, ph, pw), in_dims),
        pad_value=pad_value,
    )
    out_specs = [
        _OutputSpec(out_shape, out_tile, tuple(range(4)), input_t.dtype)
    ]

    def compute(grid_index, tile):
        return node.target(tile, [kH, kW], [sh, sw], [0, 0], *extra_args)

    kernel = _map_kernel(compute, 1)
    gm = build_pipelined_buffers(
        kernel, grid, [in_spec], out_specs, (input_t,), num_banks=num_banks
    )
    return gm
