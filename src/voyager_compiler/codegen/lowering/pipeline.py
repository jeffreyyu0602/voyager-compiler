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
  * The kernel writes an out bank via the ``write_out`` helper (or
    ``voyager.copy_tile`` directly), choosing reset vs. accumulate per
    ``grid_index`` (e.g. reset when the reduction coord is 0) — the scheduler
    stays oblivious to which output is a reduction.

Export-driven design notes (each verified against ``export_model``):
  * In-body bank writes go through the opaque ``voyager.copy_tile`` custom op,
    never raw aten in-place: ``while_loop`` (a HOP) rejects aten in-place
    mutation of captured tensors, but custom-op ``Tensor(a!)`` mutation of a
    *captured* (closed-over) bank is allowed and persists.
  * Accumulate-vs-reset (``write_out``) uses ``torch.cond`` in its *functional*
    form (each branch returns a fresh tensor) followed by a ``copy_tile`` write.
  * A **guarded DMA** is ``torch.cond`` over ``voyager.async_copy``: the mutated
    buffer is a cond *operand* (a captured/closed buffer mutated inside a branch
    is silently dropped on export — the docstring's "no mutation of global
    variables"), and the async *token* the copy returns is the cond's used
    result (carried in loop state) so the op is not pruned.  Load counters
    advance with ``torch.sym_ite(changed, count + 1, count)`` outside the cond.
    See ``test/cond_capture_demo.py`` for the captured-vs-operand contrast.
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.utils import (
    _InputSpec,
    _OutputSpec,
    _compute_input_spec,
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)
from voyager_compiler.codegen.lowering.ops import MemoryLevel

_SRAM = int(MemoryLevel.SRAM)


def _inert_token() -> torch.Tensor:
    """A 0-dim int token matching ``voyager.async_copy``'s return — the value a
    guard's *skip* branch returns so both ``torch.cond`` branches agree.  A
    tensor, not a bare int: a ``torch.cond`` branch can't return a ``SymInt``
    (it must be a tensor), which is the cost of the ``aten.zeros``.
    """
    return torch.zeros((), dtype=torch.int64)


def write_out(bank: torch.Tensor, value: torch.Tensor, accumulate) -> None:
    """Write ``value`` into the captured output SRAM ``bank`` (mutate-style
    kernel helper).

    ``value`` is pinned to the bank's tile shape (the kernel's result may carry
    input-derived symbolic sizes, e.g. a pooled halo tile).  ``accumulate`` (a
    bool / SymBool the kernel derives from ``grid_index`` — e.g.
    ``reduction_coord != 0``) selects add-into-bank vs. overwrite, as a
    functional ``torch.cond`` (fresh-tensor branches), then a single
    ``copy_tile`` writes the result back into the bank.
    """
    new_value = torch.cond(
        accumulate,
        lambda b=bank, v=value: b + v,
        lambda v=value: v.clone(),
    )
    voyager.copy_tile(new_value, bank, [0] * bank.ndim, list(bank.shape))


class PipelinedKernel(torch.nn.Module):
    """Spec-driven, mutate-style kernel scheduler (see module docstring).

    ``kernel(grid_index, *in_tiles, *out_banks)`` is the per-tile compute; it
    **writes** each output SRAM bank (via ``write_out`` / ``voyager.copy_tile``)
    rather than returning a value. ``grid`` is the iteration space (tiles per
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
        num_banks: int = 2,
    ):
        super().__init__()
        if num_banks < 1:
            raise ValueError("num_banks must be >= 1")
        self.kernel = kernel
        self.grid = grid
        self.in_specs = in_specs
        self.out_specs = out_specs
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

    def _indices_differ(self, spec, cur, nxt):
        """Whether the operand's tile block changes between grid points ``cur``
        and ``nxt`` — the OR over its tiled (non-broadcast) dims of ``cur[g] !=
        nxt[g]``.  A chained ``|`` of ``SymBool``s — no Python ``any`` short-
        circuit (which would data-dependent-guard inside the traced loop) and no
        mixed-radix arithmetic, just the per-dim comparisons.  This is the load
        / store change predicate.
        """
        bcast = getattr(spec, "is_broadcast", None)
        differ = False
        for d, g in enumerate(spec.index_map):
            if self.grid[g] > 1 and not (bcast is not None and bcast[d]):
                differ = differ | (cur[g] != nxt[g])
        return differ

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

    def _load_block(self, src, dst, spec, grid_idx):
        """Async-DMA ``src``'s tile at ``grid_idx`` into SRAM ``dst``, carrying
        the input halo (``strides`` / ``pad`` / ``pad_value``); returns the
        async token.  Used both by the prologue (token discarded) and inside a
        guard's ``do`` branch (token is the cond's used result).
        ``voyager.async_copy`` — not ``copy_tile`` — so the guarded form has a
        tensor result that matches its ``skip`` branch and keeps the op from
        being pruned.
        """
        dims, indices = self._block_address(spec, grid_idx)
        return voyager.async_copy(
            src,
            dst,
            indices,
            spec.tile_sizes,
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

    def _copy_in(self, dram_buf, slot, spec, fetch_idx, should_copy):
        """Async-DMA ``fetch_idx``'s block of ``dram_buf`` into the SRAM bank
        ``slot`` when ``should_copy``, else a no-op; return the async token (the
        copy's when it fires, an inert token when skipped).  ``slot`` is a
        ``torch.cond`` operand (a captured buffer mutated inside a branch is
        dropped on export), and the token is the cond's used result, so the op
        survives.  The caller advances the producer cursor (``sym_ite``) and
        stamps the token vector (``where``) under the same ``should_copy``.
        """

        def do(src, dst):
            return self._load_block(src, dst, spec, fetch_idx)

        def skip(src, dst):
            return _inert_token()

        return torch.cond(should_copy, do, skip, (dram_buf, slot))

    def _store_block(self, input_buffers, output_buffers, spec, grid_idx):
        """Async-DMA SRAM tile ``bank`` -> ``out_buf``'s block at ``grid_idx``;
        returns the token. Used unconditionally (an output that stores every
        step) or inside a guard's ``do``.
        """
        dims, indices = self._block_address(spec, grid_idx)
        return voyager.async_copy(
            input_buffers, output_buffers, indices, spec.tile_sizes, dims=dims
        )

    def _guarded_store(self, sram_buf, dram_buf, sc, spec, cur, nxt, last):
        """Store ``sram_buf`` -> ``dram_buf``'s block at ``cur`` and return
        ``(token, next_count)``. Unconditional when the output advances every
        step (build-time constant); otherwise guarded so a reduction writes once
        per output tile — when its block completes (changes next) or on the last
        step.  ``sram_buf`` / ``dram_buf`` are cond operands; the token is the
        used result.
        """
        if self._advances_every_step(spec):
            return self._store_block(sram_buf, dram_buf, spec, cur), sc + 1

        should_store = self._indices_differ(spec, cur, nxt) | last

        def do(src, dst):
            return self._store_block(src, dst, spec, cur)

        def skip(src, dst):
            return _inert_token()

        return (
            torch.cond(should_store, do, skip, (sram_buf, dram_buf)),
            torch.sym_ite(should_store, sc + 1, sc),
        )

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
        # carried per-bank *token vector* (one async-DMA token per slot).  The
        # input pipeline runs ``D = num_banks - 1`` blocks ahead: the prologue
        # primes the first ``D`` distinct blocks (slots ``0..D-1``); each step
        # prefetches the block ``D`` ahead while the kernel reads the current
        # one.  A *guarded* input (block reused across some sweep) carries two
        # cursors — a producer ``copy_in`` and a consumer ``wait_in`` — that
        # advance on different events; an always-advance input needs neither
        # (its read / copy slots are pure functions of ``step``).  Each step
        # (Pallas ``copy_in, wait_in, kernel, copy_out, wait_out``):
        # * COPY-IN (guarded prefetch) — async-load the block ``D`` ahead into
        #   ``copy_slot`` (skipped when no new block enters the window edge or
        #   the fetch is past the tail), recording the copy's token at that
        #   slot.
        # * WAIT-IN — before the kernel reads ``read_slot`` (``step % nb`` for
        #   an always-advance input, the consumer cursor's slot for a guarded
        #   one), ``async_wait`` on that slot's recorded token.
        # * WAIT-OUT (reuse) — before the kernel reuses an output slot,
        #   ``async_wait`` on that slot's previous store, so the store has
        #   drained before the slot is overwritten.
        # * KERNEL — mutate-style, writes the captured output bank slots.
        # * COPY-OUT (guarded store) — async-store an output slot to DRAM when
        #   its block completes (block changes next, or last step), recording
        #   the store's token at that slot. After the loop a final WAIT-OUT
        #   drains every output slot's last store before the result is returned.
        #   Guards are ``torch.cond`` over ``voyager.async_copy`` (the token is
        #   the cond's used result); the per-slot token vectors are updated with
        #   ``torch.where`` and read with a symbolic ``[slot]`` select.  Slot
        #   indices are runtime ``% num_banks`` — no unrolling.  At
        #   ``num_banks == 2`` (``D == 1``) this reduces to one-step-ahead
        #   double buffering; ``num_banks == 1`` (``D == 0``) is single
        #   buffering — copy the current block, wait on it, read it (no
        #   prefetch).
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
        nb = self.num_banks
        num_outputs = len(self.out_specs)
        D = nb - 1  # prefetch distance (blocks ahead)

        # Prologue: prime the first ``D`` logical positions, deduplicating
        # reused blocks.  These positions are *static* (concrete grid coords),
        # so the dedup is a Python ``if`` — each distinct block is copied into
        # the next ring slot and stamped into the input's token vector.  An
        # always-advance input makes ``D`` distinct copies (slots ``0..D-1``); a
        # guarded one's prologue copy count seeds its producer cursor.
        init_load_tokens = []
        init_copy_in = []  # guarded inputs only: prologue copy count
        for i, (inp, spec) in enumerate(tiled):
            vec = _inert_token()
            num_copies = 0
            prev_idx = None
            for p in range(min(D, self.num_steps)):
                idx = self._unravel(p)
                if p == 0 or self._indices_differ(spec, prev_idx, idx):
                    slot = num_copies % nb
                    tok = self._load_block(inp, in_banks[i][slot], spec, idx)
                    vec = torch.where(torch.arange(nb) == slot, tok, vec)
                    num_copies += 1
                prev_idx = idx
            # Single buffering (nb == 1) waits on the freshly-issued token, so
            # it carries no input token vector (see WAIT-IN) — nothing to seed.
            if nb > 1:
                init_load_tokens.append(vec)
            if not self._advances_every_step(spec):
                init_copy_in.append(num_copies)

        init_store_tokens = [
            torch.zeros(nb, dtype=torch.int64) for _ in range(num_outputs)
        ]

        def cond_fn(
            step, copy_counts, wait_counts, store_counts, in_toks, out_toks
        ):
            return step < self.num_steps

        def body_fn(
            step, copy_counts, wait_counts, store_counts, in_toks, out_toks
        ):
            cur = voyager.delinearize_index(step, self.grid)
            nxt = voyager.delinearize_index(step + 1, self.grid)
            last = step + 1 >= self.num_steps
            cur_slot = step % nb
            slots = torch.arange(nb)

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
                prev_edge = voyager.delinearize_index(step - 1, self.grid)
            elif D == 1:
                # one-step pipeline: the window edge is just ``nxt`` / ``cur``
                fetch_idx, prev_edge = nxt, cur
            else:
                fetch_idx = voyager.delinearize_index(fetch_step, self.grid)
                prev_edge = voyager.delinearize_index(fetch_step - 1, self.grid)

            # 1. COPY-IN: prefetch into ``copy_slot`` and stamp the token
            #    there.  An always-advance input has no cursor (``copy_slot =
            #    (step + D) % nb``, ``should_copy = has_fetch``); a guarded one
            #    advances its producer cursor only when a copy fires.  A skipped
            #    copy stamps an inert token, but the depth invariant guarantees
            #    ``copy_slot`` is then the next-to-fill slot (never a still-live
            #    one): when ``should_copy`` is false the block repeats at the
            #    window edge, so in-flight distinct blocks <= nb - 1 and
            #    ``copy_slot`` differs from the read slot — it is overwritten by
            #    a real copy before any read.
            next_copy_counts, next_in_toks, issued_tokens = [], [], []
            g = 0
            for i, (inp, spec) in enumerate(tiled):
                if self._advances_every_step(spec):
                    copy_slot = (step + D) % nb
                    should_copy = has_fetch
                else:
                    pc = copy_counts[g]
                    copy_slot = pc % nb
                    differ = self._indices_differ(spec, prev_edge, fetch_idx)
                    # D == 0 (single buffer): copy on the first step or when the
                    # block changes from the previous one; else prefetch-gated.
                    should_copy = (
                        (first | differ) if nb == 1 else (has_fetch & differ)
                    )
                    next_copy_counts.append(
                        torch.sym_ite(should_copy, pc + 1, pc)
                    )
                    g += 1
                # ``_check`` against the bank's own size lets the select bound
                # resolve on the unbacked step (needed for num_banks >= 3).
                torch._check(copy_slot < in_banks[i].size(0))
                token = self._copy_in(
                    inp,
                    in_banks[i][copy_slot],
                    spec,
                    fetch_idx,
                    should_copy,
                )
                # nb == 1 carries no input token vector (it is unused — WAIT-IN
                # waits on ``issued_tokens``), so don't produce a next value.
                if nb == 1:
                    issued_tokens.append(token)
                else:
                    next_in_toks.append(
                        torch.where(slots == copy_slot, token, in_toks[i])
                    )

            # 2. WAIT-IN: block on the token of each input's read slot, then
            #    read it.  Always-advance reads ``step % nb``; a guarded input
            #    reads its consumer cursor's slot.  ``None``-spec operands pass
            #    through in kernel arg order.
            in_args, i, g = [], 0, 0
            for inp, spec in zip(inputs, self.in_specs):
                if spec is None:
                    in_args.append(inp)
                    continue
                if self._advances_every_step(spec):
                    rs = cur_slot
                else:
                    rs = wait_counts[g] % nb
                    g += 1
                torch._check(rs < in_banks[i].size(0))
                # nb == 1 (single buffer): the copy issued this step is the
                # *current* tile, so wait on that token — the carried token is a
                # future tile only when nb >= 2.
                wait_token = issued_tokens[i] if nb == 1 else in_toks[i][rs]
                voyager.async_wait(wait_token)
                in_args.append(in_banks[i][rs])
                i += 1

            # 3. WAIT-OUT (reuse): block on each output slot's previous store
            #    before the kernel reuses (overwrites / re-accumulates into) it.
            out_slots, out_slot_idx = [], []
            for i, spec in enumerate(self.out_specs):
                if self._advances_every_step(spec):
                    slot = cur_slot
                else:
                    slot = store_counts[i] % nb
                torch._check(slot < out_banks[i].size(0))
                out_slot_idx.append(slot)
                # TODO: guard async_wait(0) using torch.cond?
                voyager.async_wait(out_toks[i][slot])
                out_slots.append(out_banks[i][slot])

            # 4. KERNEL (mutate-style: writes the output bank slots).
            self.kernel(cur, *in_args, *out_slots)

            # 5. COPY-OUT: store each completed output tile (guarded) and record
            #    the store's token at its source slot; advance the store counter
            #    (flip slot).
            next_store_counts, next_out_toks = [], []
            for i, spec in enumerate(self.out_specs):
                token, new_sc = self._guarded_store(
                    out_slots[i],
                    out_bufs[i],
                    store_counts[i],
                    spec,
                    cur,
                    nxt,
                    last,
                )
                next_out_toks.append(
                    torch.where(slots == out_slot_idx[i], token, out_toks[i])
                )
                next_store_counts.append(new_sc)

            # 6. Consumer advance (guarded inputs): the current block is done
            #    when it changes next step or this is the last step.
            next_wait_counts, g = [], 0
            for inp, spec in tiled:
                if self._advances_every_step(spec):
                    continue
                wc = wait_counts[g]
                finished = last | self._indices_differ(spec, cur, nxt)
                next_wait_counts.append(torch.sym_ite(finished, wc + 1, wc))
                g += 1

            return (
                step + 1,
                tuple(next_copy_counts),
                tuple(next_wait_counts),
                tuple(next_store_counts),
                tuple(next_in_toks),
                tuple(next_out_toks),
            )

        init = (
            0,
            # producer cursors, one per guarded input (prologue copy count)
            tuple(init_copy_in),
            # consumer cursors, one per guarded input (start at block 0)
            (0,) * len(init_copy_in),
            # store counters
            (0,) * num_outputs,
            # per-input [nb] token vectors (the prologue copies)
            tuple(init_load_tokens),
            # per-output [nb] token vectors (no store yet)
            tuple(init_store_tokens),
        )
        final = while_loop(cond_fn, body_fn, init)

        # Final WAIT-OUT: drain every output slot's last store so the DRAM
        # result is complete.
        final_store_tokens = final[5]
        for i in range(num_outputs):
            for slot in range(nb):
                voyager.async_wait(final_store_tokens[i][slot])

        return out_bufs[0] if len(out_bufs) == 1 else tuple(out_bufs)


def build_pipelined_buffers(
    kernel: Callable,
    grid: Tuple[int, ...],
    in_specs: List[Optional[_InputSpec]],
    out_specs: List[_OutputSpec],
    inputs: Tuple[torch.Tensor, ...],
    *,
    num_banks: int = 2,
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """Build the bufferized FX graph (a single rolled ``while_loop`` over
    ``voyager.*`` primitives) for ``kernel`` over ``grid``.  Mirrors
    ``build_pointwise_buffers``'s export / finalize / extent-tag flow.
    """
    pattern = PipelinedKernel(
        kernel, grid, in_specs, out_specs, num_banks=num_banks
    )
    with _lenient_verifier():
        gm = export_model(pattern, inputs, kwargs=kwargs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(0, pattern.num_steps, 1)]])
    # Annotate each tensor's memory space (DRAM vs Scratchpad) so the graph is
    # fully bufferized (threads through the while_loop body and the guarded
    # ``torch.cond`` regions).
    from voyager_compiler.codegen.lowering.bufferization import (
        annotate_tensor_spaces,
    )

    annotate_tensor_spaces(gm)
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
# ``_mutate_kernel`` (each result is written into its output bank), accumulating
# across the reduction grid dim for a GEMM / conv and overwriting for a map
# (pointwise / pool / layernorm·softmax).
# ---------------------------------------------------------------------------


def _mutate_kernel(
    compute: Callable, num_outputs: int, reduction_dim: Optional[int] = None
):
    """Adapt a return-style ``compute(grid_index, *in_tiles) -> Tensor | tuple``
    into the scheduler's mutate-style ``kernel(grid_index, *in_tiles,
    *out_banks)``: run ``compute`` and write each result into its output bank
    via ``write_out``.  ``reduction_dim`` (a grid dim) makes the write
    *accumulate* over that sweep (reset when its coord is 0); ``None``
    overwrites (a map).
    """

    def kernel(grid_index, *args):
        in_tiles = args[: len(args) - num_outputs]
        out_banks = args[len(args) - num_outputs :]
        results = compute(grid_index, *in_tiles)
        if not isinstance(results, (tuple, list)):
            results = (results,)
        accumulate = (
            reduction_dim is not None and grid_index[reduction_dim] != 0
        )
        for bank, value in zip(out_banks, results):
            write_out(bank, value, accumulate)

    return kernel


def build_conv2d(node, *, num_banks: int = 2):
    """Pipeline builder for a conv2d (groups=1) node — including the
    microscaling / codebook (``conv2d_mx``) variant, a fused bias, and the
    systolic NHWC layout — the input-channel (C) cross-tile reduction.  A map
    over the (N, K, oH, oW) output grid plus a reduction grid dim for C: the
    input is a strided receptive-field halo (pad-on-load, ``pad_value=0``), the
    weight is tiled on (K, C), and the kernel convolves each C-block and
    accumulates (reset when the C coord is 0).  Grid ``(N, K, oH, oW, C, 1)`` —
    the trailing extent-1 dim holds the whole ``kH``/``kW`` weight dims; for
    ``num_c == 1`` the C dim is extent 1 (no reduction).  Specs are written in
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

    shapes = node.meta.get("tiled_shapes", {})
    inp = node.args[0].value.clone()
    w = node.args[1].value.clone()
    out = node.value
    if inp.ndim != 4 or w.ndim != 4:
        return None
    groups = get_arg_value(node, 6, "groups", 1)
    if groups != 1:
        return None  # doesn't support depthwise conv yet

    # ``meta["transposed"]`` selects the systolic layout: NHWC feature maps +
    # HWIO weight.  Specs are logical (NCHW/OIHW per-axis) and ``_project``-ed
    # onto each operand's physical order; the grid stays logical.  ``None`` dims
    # == logical NCHW (a plain identity projection).
    nhwc = node.meta.get("transposed", False)
    in_dims = _NHWC if nhwc else None
    w_dims = _HWIO if nhwc else None
    out_dims = _NHWC if nhwc else None

    # logical (tn, tk, toh, tow)
    out_ts = _unproject(shapes.get(node), out_dims)
    # logical (tn, tc, ., .)
    in_ts = _unproject(shapes.get(node.args[0]), in_dims)
    tn, tk, toh, tow = (int(x) for x in out_ts)
    N, C, H, W = _unproject(inp.shape, in_dims)
    K, _, kH, kW = _unproject(w.shape, w_dims)
    tc = int(in_ts[1])
    num_c = C // tc

    sh, sw = _pair(get_arg_value(node, 3, "stride", 1))
    ph, pw = _pair(get_arg_value(node, 4, "padding", 0))
    dh, dw = _pair(get_arg_value(node, 5, "dilation", 1))
    oH, oW = _unproject(out.shape, out_dims)[2:]
    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1

    # grid dims (logical): 0=N 1=K 2=oH 3=oW 4=C(reduction) 5=whole(kH/kW).
    grid = (N // tn, K // tk, oH // toh, oW // tow, num_c, 1)
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
    out_spec = _OutputSpec(
        _project((N, K, oH, oW), out_dims),
        _project((tn, tk, toh, tow), out_dims),
        _project((0, 1, 2, 3), out_dims),
        inp.dtype,
    )

    inputs, in_specs = [inp, w], [in_spec, w_spec]
    # Operand index into the kernel's ``*extra`` (the args after the input,
    # weight tiles).
    extra_index = lambda: len(in_specs) - 2

    # Bias [K] tiles along K (grid dim 1); folded once on the C==0 step (below),
    # since later C blocks only accumulate partials into the bank.
    bias_n = get_arg_value(node, 2, "bias")
    if bias_n is not None:
        bias_slot = extra_index()
        inputs.append(bias_n.value.clone())
        in_specs.append(_InputSpec((tk,), (1,), (False,)))

    # Bias broadcasts over the output's channel (K) dim — its physical position
    # depends on layout.
    bias_shape = [1, 1, 1, 1]
    bias_shape[_phys_pos(1, out_dims)] = -1

    # On the microscaling target (conv2d_mx) the per-block scales tile along C
    # (// block_size): input_scale shares the input halo's layout, weight_scale
    # the weight's; the codebook tables (only present on this target) load whole
    # (untiled, None spec).  Each threads to the op by kw.
    target = node.target
    bs = node.kwargs.get("block_size")

    kw_slots = {}

    def add_kw_input(name: str, spec: _InputSpec | None) -> None:
        v = node.kwargs.get(name)
        if not isinstance(v, torch.fx.Node):
            return

        if not hasattr(v, "value"):
            raise ValueError(
                f"Expected materialized value for FX node kwarg {name!r}"
            )

        kw_slots[name] = extra_index()
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

    def compute(grid_index, in_tile, w_tile, *extra):
        kw = {name: extra[i] for name, i in kw_slots.items()}
        if bs is not None:
            kw["block_size"] = bs
        result = target(
            in_tile, w_tile, None, [sh, sw], [0, 0], [dh, dw], groups, **kw
        )
        if bias_n is not None:
            # Add bias once, on the first C-reduction step: gate the small
            # contiguous bias addend through the cond, then add it (gating
            # ``result`` trips cond's dense-output check on the conv tile's
            # symbolic spatial stride).
            bias_tile = extra[bias_slot].reshape(bias_shape)
            addend = torch.cond(
                grid_index[4] == 0,
                # clone: a cond branch can't return its input
                lambda: bias_tile.clone(),
                lambda: torch.zeros_like(bias_tile),
            )
            result = result + addend
        return result

    kernel = _mutate_kernel(compute, 1, reduction_dim=4 if num_c > 1 else None)
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, [out_spec], tuple(inputs), num_banks=num_banks
    )
    return gm


def build_gemm(node, *, num_banks: int = 2):
    """Pipeline builder for a linear / matmul / batched-matmul node — including
    the microscaling / codebook (``*_mx``) variants and a fused bias — covering
    the cross-tile K reduction the old pointwise engine couldn't do.  Grid ``(M,
    N, K)`` (or ``(B, M, N, K)`` for a batched matmul) tiles with K innermost;
    the kernel accumulates ``act_tile @ weight_tile`` into the output bank,
    reset on ``k == 0``.  Returns the gm, or ``None`` (unsupported).
    """
    from voyager_compiler.codegen.mapping_utils import is_linear, is_matmul
    from voyager_compiler.codegen.passes.utils import get_arg_value

    shapes = node.meta.get("tiled_shapes")
    if not shapes or not (is_linear(node) or is_matmul(node)):
        return None
    act = node.args[0].value.clone()
    weight = node.args[1].value.clone()
    out = node.value
    if not isinstance(out, torch.Tensor) or act.ndim < 2 or weight.ndim < 2:
        return None

    out_ts = shapes.get(node)  # (..batch.., tile_m, tile_n)
    in_ts = shapes.get(node.args[0])  # (..batch.., tile_m, tile_k)
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
    weight_ck = is_matmul(node) != bool(node.meta.get("transposed", False))
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
    out_spec = _OutputSpec(
        tuple(out.shape),
        tuple(tb) + (tm, tn),
        tuple(range(nb)) + (gm, gn),
        out.dtype,
    )

    inputs, in_specs = [act, weight], [act_spec, weight_spec]
    # Operand index into the kernel's ``*extra`` (the args after activation,
    # weight).
    extra_index = lambda: len(in_specs) - 2

    # Bias [N] tiles along N (grid dim ``gn``); folded once on the k==0 step
    # (below), since later steps only accumulate partials into the bank — adding
    # bias per-k would multiply-count it.
    bias_n = get_arg_value(node, 2, "bias")
    if bias_n is not None:
        bias_slot = extra_index()
        inputs.append(bias_n.value.clone())
        in_specs.append(_InputSpec((tn,), (gn,), (False,)))

    # On the microscaling targets (linear_mx / matmul_mx) the per-block scales
    # tile along the reduction (// block_size, sharing the operand's batch +
    # block layout) and the codebook tables (input_code / weight_code, only
    # present on these targets) load whole (untiled, None spec); each threads to
    # the op by keyword.
    bs = node.kwargs.get("block_size")

    kw_slots = {}

    def add_kw_input(name: str, spec: _InputSpec | None) -> None:
        v = node.kwargs.get(name)
        if not isinstance(v, torch.fx.Node):
            return

        if not hasattr(v, "value"):
            raise ValueError(
                f"Expected materialized value for FX node kwarg {name!r}"
            )

        kw_slots[name] = extra_index()
        inputs.append(v.value.clone())
        in_specs.append(spec)

    if node.target in (
        torch.ops.quantized_ops.linear_mx.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ):
        in_scale_qspec = _spec(act.shape, (tm, tk // bs), (gm, gk))
        wt_scale_qspec = _spec(weight.shape, _proj(tn, tk // bs), _proj(gn, gk))

        add_kw_input("input_scale", in_scale_qspec)
        add_kw_input("weight_scale", wt_scale_qspec)

        for name in ("input_code", "weight_code"):
            add_kw_input(name, None)

    op = node.target

    def compute(grid_index, activation, weight_tile, *extra):
        kw = {name: extra[i] for name, i in kw_slots.items()}
        if bs is not None:
            kw["block_size"] = bs
        result = op(activation, weight_tile, **kw)
        if bias_n is not None:
            # Add bias once, on the first reduction step: gate the (small,
            # contiguous) bias addend through the cond, then add it to the
            # result — adding bias per-k would multiply-count it.  (Gating
            # ``result`` itself trips cond's dense-output check on symbolic
            # shapes.)
            bias_tile = extra[bias_slot]  # [N]
            addend = torch.cond(
                grid_index[gk] == 0,
                # clone: a cond branch can't return its input
                lambda: bias_tile.clone(),
                lambda: torch.zeros_like(bias_tile),
            )
            result = result + addend
        return result

    kernel = _mutate_kernel(compute, 1, reduction_dim=gk)
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, [out_spec], tuple(inputs), num_banks=num_banks
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
    tiled_shapes = node.meta.get("tiled_shapes")
    if not tiled_shapes:
        return None

    outputs = list(val) if isinstance(val, (list, tuple)) else [val]
    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]

    # Resolve each op arg to a loaded-tile index (tensor operand) or a plain
    # constant *now* — the closure runs in the traced while_loop body, where
    # dynamo rejects FX-Node lookups.
    order = {n: i for i, n in enumerate(in_nodes)}
    _plain = lambda a: list(a) if isinstance(a, list) else a
    arg_slots = [
        order[a] if isinstance(a, torch.fx.Node) else None for a in node.args
    ]
    kw_slots = {
        k: order[v] if isinstance(v, torch.fx.Node) else None
        for k, v in node.kwargs.items()
    }
    op_args = [_plain(a) for a in node.args]
    op_kwargs = {k: _plain(v) for k, v in node.kwargs.items()}
    op = node.target

    def compute(grid_index, *tiles):
        args = [
            tiles[i] if i is not None else a for i, a in zip(arg_slots, op_args)
        ]
        kwargs = {
            k: tiles[i] if i is not None else op_kwargs[k]
            for k, i in kw_slots.items()
        }
        return op(*args, **kwargs)

    output_ts = tiled_shapes.get(node)
    if isinstance(val, (list, tuple)):
        output_ts = output_ts[-1]
    output_shape = tuple(outputs[-1].shape)
    grid = tuple(s // t for s, t in zip(output_shape, output_ts))
    codebooks = _codebook_arg_nodes(node)
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
    kernel = _mutate_kernel(compute, len(outputs))
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, out_specs, tuple(inputs), num_banks=num_banks
    )
    return gm


_POOL2D_SUPPORTED = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.avg_pool2d.default,
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
    shapes = node.meta.get("tiled_shapes", {})
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

    kernel = _mutate_kernel(compute, 1)
    gm = build_pipelined_buffers(
        kernel, grid, [in_spec], out_specs, (input_t,), num_banks=num_banks
    )
    return gm
