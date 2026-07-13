"""FA3-style flash-attention bufferization builder (cross-sweep pipelined).

A standalone sibling of ``attention.build_attention`` that overlaps the
systolic (matrix) array with the vector unit, the way FlashAttention-3
overlaps Hopper's tensor cores with its multi-function units.  The
baseline kernel serializes the two units inside every KV step (QKᵀ GEMM →
softmax chain → PV GEMM); this builder restructures the schedule so both
units always have work:

1. **Deferred rescale.**  The accumulator update is
   ``o += P_prev @ V_prev`` *then* ``o *= alpha``: entering a step, ``o``
   (all earlier blocks) and ``P_prev`` share the previous running-max
   scale, so accumulate-then-rescale moves the whole sum to the new scale
   at once — same math, but the PV GEMM no longer depends on the current
   block's softmax.
2. **One-step skew, fused across sweeps.**  Each loop iteration runs the
   QKᵀ GEMM of the *current* KV block and the PV GEMM of the *previous*
   step's block back-to-back on the matrix unit while the vector unit
   softmaxes the current scores.  Unlike the FA3 paper's Algorithm 2
   (which drains each Q tile's pipeline before the next — its per-tile
   bubbles are hidden by ping-ponging two warpgroups, impossible in one
   instruction stream), the skew here crosses Q-tile and head boundaries:
   a tile's leftover PV GEMM and its ``o/l`` finalize execute during the
   NEXT tile's first iteration, so the units never idle at boundaries.

The pipeline is peeled, matching Algorithm 2's shape: a **prologue**
(traced once before the ``while_loop``) primes the DMA and runs the very
first QKᵀ + softmax; the loop then runs ``num_steps - 1`` completely
uniform iterations (counter starting at 1); an **epilogue** (traced once
after the loop) runs the final tile's leftover PV GEMM, finalize, and
output store.  This is implemented as a dedicated scheduler
(``_FA3Pipeline``) rather than through the generic ``PipelinedKernel`` —
the schedule needs a lagged V stream, a lagged output store, and the
peeled iterations, none of which the generic scheduler models; the FA3
case has fixed operand roles and depths, so the explicit DMA bookkeeping
below is small.

The matrix unit computes only ``A @ B``, so the P@V is split: a pure GEMM
into ``pv_buf`` on the matrix unit, then an accumulate into ``o`` on the
vector unit.  Only the DMA (``async_copy``) and the GEMMs (QKᵀ, P@V) are
asynchronous — issued onto their unit and left running while the instruction
stream continues; the softmax chain and the accumulate/rescale are
synchronous.  So a semaphore is needed exactly where a synchronous op must
wait on an asynchronous producer: an ``insert`` carrying a semaphore posts it
when that GEMM finishes and ``voyager.async_wait`` consumes it (the eager
impls assert every wait has a matching prior post, so tests validate it):

  ``sem_scores``  the QKᵀ GEMM wrote S       → softmax may read it.
  ``sem_pv``      the P@V GEMM wrote pv_buf  → the accumulate may read it.

(softmax → the next step's P@V needs no semaphore: softmax is synchronous, so
it has already written P by the time the stream issues that P@V.)

Per loop iteration (``N = num_kv_blocks``, ``kv = cur[gkv]``), in program
order — the P@V GEMM ([B]) runs on the matrix unit while the vector unit
softmaxes ([C]), then the vector unit lands the P@V into ``o`` ([D]/[E]):

  [A] S = (Q @ Kᵀ)·scale (+ mask) -> s_buf[step % 2];   GEMM, post sem_scores
  [B] wait V DMA;  pv_buf = s_buf[(step-1) % 2] @ V_prev; GEMM, post sem_pv
  [D] kv == 0 only (a Q-tile boundary): wait sem_pv; o += pv_buf; drain the
      previous output store; (o / l) -> o_bank; store o_bank to the PREVIOUS
      tile's DRAM rows (the finalize belongs to the tile that just ended);
      reset m/o/l.
  [C] wait sem_scores;  softmax chain (rowmax, m, alpha, P, rowsum, l).
  [E] kv >= 1 only (vector unit, after [C]): wait sem_pv;  the deferred
      rescale fused with the accumulate, o = alpha·(o + pv_buf).

Because [B] is issued (async) before the synchronous [C], the matrix unit runs
the P@V while the vector unit softmaxes — the two overlap on every non-boundary
step. ``pv_buf`` is single-buffered: the accumulate ([D]/[E]) is synchronous
and reads ``pv_buf`` before the stream issues the next step's P@V, so the write
never races the read. ``s_buf`` needs two slots, though: [A] writes the next
block's scores into ``s_buf[step % 2]`` while [B] is still reading the previous
block's probabilities from ``s_buf[(step-1) % 2]`` — two async GEMMs' operands
live at once (the later overwrite is safe by matrix-unit program order).
``o/l/m`` are single-buffered — only the synchronous vector ops touch them.

DMA schedule (exact — every block fetched once and consumed once): per
iteration one K block and one V block, V's stream one step behind K's
(matching its one-step-behind consumption).  K prefetches the next step's
block (gated off on the last iteration); V fetches the CURRENT step's
block into the slot the next step reads; Q prefetches the next sweep's
tile at each sweep's last iteration.  The prologue primes K0/Q0 (+ the
K1/V0 prefetches); the epilogue consumes V's one remaining block.
"""

import math

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.attention import (
    _MASK_FILL,
    _fuse_passes,
)
from voyager_compiler.codegen.lowering.ops import MemoryLevel, oracle_disabled
from voyager_compiler.codegen.lowering.pipeline import (
    _guarded_wait,
    select_bank,
)
from voyager_compiler.codegen.lowering.utils import (
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)
from voyager_compiler.codegen.shape_prop import ShapeProp
from voyager_compiler.codegen.passes.utils import get_arg_value

_SRAM = int(MemoryLevel.SRAM)


def _unravel(flat, basis):
    """Row-major coordinates of Python-int ``flat`` in ``basis`` (the
    build-time counterpart of ``voyager.delinearize_index``)."""
    out = [0] * len(basis)
    for d in range(len(basis) - 1, -1, -1):
        out[d] = flat % basis[d]
        flat //= basis[d]
    return out


class _FA3Pipeline(torch.nn.Module):
    """The FA3 scheduler: prologue → uniform ``while_loop`` → epilogue
    (see the module docstring for the schedule).  Traced whole by
    ``torch.export``; every slot index outside the loop is a Python int.
    """

    def __init__(
        self,
        *,
        grid,
        num_kv_blocks,
        tq,
        tkv,
        head_dim,
        scale,
        has_mask,
        mask_is_bool,
        mask_dyn,
        out_shape,
        out_dtype,
        acc_dtype,
    ):
        super().__init__()
        self.grid = tuple(grid)  # (*batch, num_q_blocks, num_kv_blocks)
        self.num_steps = math.prod(grid)
        self.N = num_kv_blocks
        self.nb = len(grid) - 2
        self.gq, self.gkv = self.nb, self.nb + 1
        self.tq, self.tkv, self.d = tq, tkv, head_dim
        self.scale = scale
        self.has_mask = has_mask
        self.mask_is_bool = mask_is_bool
        # Mask: list of (dram_dim, grid_dim) pairs for its dynamic dims
        # (broadcast batch dims are pinned static-0 and never appear).
        self.mask_dyn = mask_dyn
        self.out_shape = tuple(out_shape)
        self.out_dtype = out_dtype
        self.acc_dtype = acc_dtype
        # Q / K / V share the DRAM layout [*batch, seq, head_dim]: the
        # dynamic dims are the >1-block batch dims plus (for the >1 case)
        # the sequence dim; ``head_dim`` is always loaded whole.
        self.batch_dyn = [i for i in range(self.nb) if grid[i] > 1]

    # --- DMA helpers (all take explicit coords; slots may be SymInts) ---

    def _block_address(self, coords, grid_dim, block_count):
        """The (dims, indices) block address of a [*batch, seq, d] DRAM
        tensor at grid point ``coords`` (the dynamically indexed dims and
        their block indices; the sequence dim is driven by ``grid_dim``,
        ``head_dim`` is always loaded whole).  The custom-schedule
        counterpart of ``_BufferedRef._block_address``."""
        dims = list(self.batch_dyn)
        idx = [coords[i] for i in self.batch_dyn]
        if block_count > 1:
            dims.append(self.nb)  # the sequence dim of the DRAM tensor
            idx.append(coords[grid_dim])
        return dims, idx

    def _load_q(self, q, bank, slot, sem, coords):
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        unit = (1,) * self.nb
        voyager.async_copy(
            q,
            select_bank(bank, slot),
            idx,
            unit + (self.tq, self.d),
            select_bank(sem, slot),
            dims,
        )

    def _load_k(self, k, bank, slot, sem, coords):
        # Transposed load: the DRAM buffer is K's own [.., tkv, d] block,
        # ``.mT``-ed into the bank's Kᵀ [.., d, tkv] tile by the DMA.
        dims, idx = self._block_address(coords, self.gkv, self.grid[self.gkv])
        unit = (1,) * self.nb
        voyager.async_copy(
            k,
            select_bank(bank, slot),
            idx,
            unit + (self.d, self.tkv),
            select_bank(sem, slot),
            dims,
            None,
            True,
        )

    def _load_v(self, v, bank, slot, sem, coords):
        dims, idx = self._block_address(coords, self.gkv, self.grid[self.gkv])
        unit = (1,) * self.nb
        voyager.async_copy(
            v,
            select_bank(bank, slot),
            idx,
            unit + (self.tkv, self.d),
            select_bank(sem, slot),
            dims,
        )

    def _load_mask(self, mask, bank, slot, sem, coords):
        dims = [d for d, _ in self.mask_dyn]
        idx = [coords[g] for _, g in self.mask_dyn]
        munit = (1,) * (mask.ndim - 2)
        voyager.async_copy(
            mask,
            select_bank(bank, slot),
            idx,
            munit + (self.tq, self.tkv),
            select_bank(sem, slot),
            dims,
        )

    def _store_out(self, tile, out, sem, coords):
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        unit = (1,) * self.nb
        voyager.async_copy(
            tile, out, idx, unit + (self.tq, self.d), select_bank(sem, 0), dims
        )

    # --- compute helpers (shared by prologue / loop / epilogue) ---------

    def _gemm_a(self, q_tile, k_tile, mask_tile, s_slot, sem_scores):
        """[A]: S = (Q @ Kᵀ)·scale (+ mask) -> s_slot; post sem_scores."""
        s = torch.matmul(q_tile, k_tile) * self.scale
        if self.has_mask:
            if self.mask_is_bool:
                s = torch.where(mask_tile, s, _MASK_FILL)
            else:
                s = s + mask_tile
        voyager.insert(s, s_slot, semaphore=sem_scores)

    def _softmax(self, s_slot, m, l, row_tmp, alpha, sem_scores):
        """[C]'s chain: waits for S, then rowmax / m / alpha / P / rowsum
        / l on ``s_slot`` in place (the baseline passes 2-7)."""
        voyager.async_wait(sem_scores)
        voyager.insert(torch.amax(s_slot, dim=-1, keepdim=True), row_tmp)
        voyager.insert(torch.maximum(m, row_tmp), row_tmp)
        voyager.insert(torch.exp(m - row_tmp), alpha)
        voyager.insert(row_tmp.clone(), m)
        voyager.insert(torch.exp(s_slot - row_tmp), s_slot)
        voyager.insert(torch.sum(s_slot, dim=-1, keepdim=True), row_tmp)
        voyager.insert(alpha * l + row_tmp, l)

    def _reset(self, m, l, o):
        voyager.insert(torch.full_like(m, _MASK_FILL), m)
        voyager.insert(torch.zeros_like(l), l)
        voyager.insert(torch.zeros_like(o), o)

    # --------------------------------------------------------------------

    def forward(self, q, k, v, mask=None):
        grid, N, nb = self.grid, self.N, self.nb
        num_steps = self.num_steps
        unit = (1,) * nb
        out = voyager.alloc(self.out_shape, self.out_dtype)

        # SRAM banks (2 slots each) + per-slot DMA semaphores; the output
        # bank is single-slotted (written once per Q tile).
        q_bank = voyager.alloc([*unit, self.tq, self.d], q.dtype, _SRAM, 2)
        q_sem = voyager.zeros([], torch.int64, banks=2)
        k_bank = voyager.alloc([*unit, self.d, self.tkv], k.dtype, _SRAM, 2)
        k_sem = voyager.zeros([], torch.int64, banks=2)
        v_bank = voyager.alloc([*unit, self.tkv, self.d], v.dtype, _SRAM, 2)
        v_sem = voyager.zeros([], torch.int64, banks=2)
        if self.has_mask:
            munit = (1,) * (mask.ndim - 2)
            m_bank = voyager.alloc(
                [*munit, self.tq, self.tkv], mask.dtype, _SRAM, 2
            )
            m_sem = voyager.zeros([], torch.int64, banks=2)
        out_bank = voyager.alloc([*unit, self.tq, self.d], out.dtype, _SRAM, 1)
        out_sem = voyager.zeros([], torch.int64, banks=1)

        # Running softmax state (single-buffered — see module docstring)
        # and the parity-double-buffered scores/probabilities tile.
        acc = self.acc_dtype
        m = voyager.alloc([*unit, self.tq, 1], acc, _SRAM)
        l = voyager.alloc([*unit, self.tq, 1], acc, _SRAM)
        o = voyager.alloc([*unit, self.tq, self.d], acc, _SRAM)
        s_buf = voyager.alloc([*unit, self.tq, self.tkv], acc, _SRAM, 2)
        pv_buf = voyager.alloc([*unit, self.tq, self.d], acc, _SRAM)
        row_tmp = voyager.alloc([*unit, self.tq, 1], acc, _SRAM)
        alpha = voyager.alloc([*unit, self.tq, 1], acc, _SRAM)
        sem_scores = voyager.zeros([1], torch.int64)
        sem_pv = voyager.zeros([1], torch.int64)

        # ---- prologue: prime the DMA and run step 0's [A] + [C] --------
        c0 = _unravel(0, grid)
        self._load_q(q, q_bank, 0, q_sem, c0)
        self._load_k(k, k_bank, 0, k_sem, c0)
        if self.has_mask:
            self._load_mask(mask, m_bank, 0, m_sem, c0)
        if num_steps > 1:
            c1 = _unravel(1, grid)
            self._load_k(k, k_bank, 1, k_sem, c1)
            if self.has_mask:
                self._load_mask(mask, m_bank, 1, m_sem, c1)
            if N == 1:
                # Step 0 is also its sweep's LAST step, so the uniform
                # pattern's end-of-sweep Q prefetch belongs here too.
                self._load_q(q, q_bank, 1, q_sem, c1)
        # V's stream runs one step behind K's: block 0 lands in the slot
        # step 1 reads (slot 1); nothing is fetched for step 0, which
        # consumes no V.
        self._load_v(v, v_bank, 1 % 2, v_sem, c0)

        self._reset(m, l, o)
        voyager.async_wait(select_bank(q_sem, 0))
        voyager.async_wait(select_bank(k_sem, 0))
        if self.has_mask:
            voyager.async_wait(select_bank(m_sem, 0))
        self._gemm_a(
            select_bank(q_bank, 0),
            select_bank(k_bank, 0),
            select_bank(m_bank, 0) if self.has_mask else None,
            select_bank(s_buf, 0),
            sem_scores,
        )
        self._softmax(select_bank(s_buf, 0), m, l, row_tmp, alpha, sem_scores)

        # ---- the uniform loop: t = 1 .. num_steps - 1 -------------------
        def cond_fn(step):
            return step < num_steps

        def body_fn(step):
            cur = voyager.delinearize_index(step, grid)
            prev = voyager.delinearize_index(step - 1, grid)
            nxt = voyager.delinearize_index(step + 1, grid)
            kv = cur[self.gkv]
            cur_slot = step % 2
            nxt_slot = (step + 1) % 2
            torch._check(cur_slot < 2)
            torch._check(nxt_slot < 2)

            # DMA phase: K (and mask) prefetch the next step's block,
            # gated off on the last iteration; V fetches the CURRENT
            # step's block into the slot the next step reads (the lag);
            # Q prefetches the next sweep's tile at each sweep's end.
            def k_fetch():
                self._load_k(k, k_bank, nxt_slot, k_sem, nxt)
                if self.has_mask:
                    self._load_mask(mask, m_bank, nxt_slot, m_sem, nxt)
                return 1

            torch.cond(step + 1 < num_steps, k_fetch, lambda: 0)
            self._load_v(v, v_bank, nxt_slot, v_sem, cur)

            q_next_slot = ((step + 1) // N) % 2
            torch._check(q_next_slot < 2)

            def q_fetch():
                self._load_q(q, q_bank, q_next_slot, q_sem, nxt)
                return 1

            torch.cond(
                (kv == N - 1) & (step + 1 < num_steps), q_fetch, lambda: 0
            )

            # Waits, just-in-time per Algorithm 2: K (and mask) before the
            # S GEMM here; V only later, right before the P@V GEMM — so
            # the S GEMM can issue while V's DMA is still in flight.  Q is
            # waited once per sweep, at its first iteration.
            voyager.async_wait(select_bank(k_sem, cur_slot))
            if self.has_mask:
                voyager.async_wait(select_bank(m_sem, cur_slot))
            q_slot = (step // N) % 2
            torch._check(q_slot < 2)
            _guarded_wait(select_bank(q_sem, q_slot), kv == 0)

            # [A] current block's scores on the matrix unit.
            self._gemm_a(
                select_bank(q_bank, q_slot),
                select_bank(k_bank, cur_slot),
                select_bank(m_bank, cur_slot) if self.has_mask else None,
                select_bank(s_buf, cur_slot),
                sem_scores,
            )

            # [B] the lagged P@V on the matrix unit: previous step's
            # probabilities × its V block (in this step's V read slot) into
            # pv_buf.
            prev_slot = (step - 1) % 2
            torch._check(prev_slot < 2)
            voyager.async_wait(select_bank(v_sem, cur_slot))
            voyager.insert(
                torch.matmul(
                    select_bank(s_buf, prev_slot), select_bank(v_bank, cur_slot)
                ),
                pv_buf,
                semaphore=sem_pv,
            )

            # [D] Q-tile boundary: land the previous tile's last P@V into o,
            # finalize (o / l) to ITS rows, reset for the new tile.  Runs
            # before [C] so the new tile's softmax sees fresh m/l.
            def boundary():
                voyager.async_wait(sem_pv)
                voyager.insert(o + pv_buf, o)
                # Drain the previous boundary's store before overwriting
                # the bank (no prior store exists at the first boundary).
                _guarded_wait(select_bank(out_sem, 0), step >= 2 * N)
                voyager.insert(
                    (o / l).to(self.out_dtype), select_bank(out_bank, 0)
                )
                self._store_out(select_bank(out_bank, 0), out, out_sem, prev)
                self._reset(m, l, o)
                return 1

            torch.cond(kv == 0, boundary, lambda: 0)

            # [C] softmax of the current block on the vector unit — runs
            # (synchronously) while the matrix [B] GEMM is still in flight.
            self._softmax(
                select_bank(s_buf, cur_slot), m, l, row_tmp, alpha, sem_scores
            )

            # [E] deferred rescale fused with the P@V accumulate: o = alpha·(o
            # + pv).  Runs after softmax (kv >= 1 only) so the vector unit
            # isn't blocked on sem_pv before the softmax.
            def rescale():
                voyager.async_wait(sem_pv)
                voyager.insert(alpha * (o + pv_buf), o)
                return 1

            torch.cond(kv > 0, rescale, lambda: 0)
            return (step + 1,)

        while_loop(cond_fn, body_fn, (1,))

        # ---- epilogue: the final tile's leftover P@V + finalize --------
        c_last = _unravel(num_steps - 1, grid)
        v_slot = num_steps % 2
        # [B] the final block's P@V on the matrix unit.
        voyager.async_wait(select_bank(v_sem, v_slot))
        voyager.insert(
            torch.matmul(
                select_bank(s_buf, (num_steps - 1) % 2),
                select_bank(v_bank, v_slot),
            ),
            pv_buf,
            semaphore=sem_pv,
        )
        # land it into o on the vector unit, then finalize.
        voyager.async_wait(sem_pv)
        voyager.insert(o + pv_buf, o)
        if num_steps > N:  # a prior boundary store exists (static)
            voyager.async_wait(select_bank(out_sem, 0))
        voyager.insert((o / l).to(self.out_dtype), select_bank(out_bank, 0))
        self._store_out(select_bank(out_bank, 0), out, out_sem, c_last)
        voyager.async_wait(select_bank(out_sem, 0))  # drain
        return out


def build_attention_fa3(
    node,
    *,
    accumulate_fp32: bool = True,
    tiler=None,
):
    """FA3-style pipeline builder for an
    ``aten.scaled_dot_product_attention`` node.

    Returns the bufferized ``GraphModule`` (prologue + rolled
    ``while_loop`` + epilogue over ``voyager.*`` primitives, see the
    module docstring), or ``None`` when uncovered (missing tiling,
    dropout / GQA, unsupported rank).  ``tiler`` is accepted for
    signature parity but unused — attention tiling comes from
    ``node.meta['l2_tiling']`` (``(num_q_blocks, num_kv_blocks)``).
    """
    if (
        node.op != "call_function"
        or node.target
        is not torch.ops.aten.scaled_dot_product_attention.default
    ):
        return None

    q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
    mask_node = get_arg_value(node, 3, "attn_mask", None)
    dropout_p = get_arg_value(node, 4, "dropout_p", 0.0)
    is_causal = get_arg_value(node, 5, "is_causal", False)
    scale = node.kwargs.get("scale", None)
    if node.kwargs.get("enable_gqa", False) or dropout_p:
        return None  # GQA / dropout unsupported
    if is_causal:
        raise NotImplementedError(
            "is_causal attention is not supported yet; pass an explicit "
            "additive attn_mask instead"
        )

    tiling = node.meta.get("l2_tiling")
    if tiling is None:
        return None
    num_q_blocks, num_kv_blocks = tiling

    q = q_node.value.clone()
    k = k_node.value.clone()
    v = v_node.value.clone()
    out = node.value
    if q.ndim < 2 or q.ndim != k.ndim or q.ndim != v.ndim:
        return None

    batch = tuple(q.shape[:-2])
    nb = len(batch)
    Sq, d = q.shape[-2], q.shape[-1]
    Skv = k.shape[-2]
    tq, tkv = Sq // num_q_blocks, Skv // num_kv_blocks
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    grid = batch + (num_q_blocks, num_kv_blocks)
    gq, gkv = nb, nb + 1

    mask_is_bool = False
    mask_dyn = []
    if isinstance(mask_node, torch.fx.Node):
        mask = mask_node.value
        mask_is_bool = mask.dtype == torch.bool
        # Mask [*mbatch, Sq, Skv]: dynamic dims are the >1-block batch
        # dims (a size-1 batch dim broadcasts, pinned to block 0) plus
        # the q / kv dims when tiled.
        mb = tuple(mask.shape[:-2])
        off = nb - len(mb)
        for j, sz in enumerate(mb):
            g = off + j
            if sz != 1 and grid[g] > 1:
                mask_dyn.append((j, g))
        if num_q_blocks > 1:
            mask_dyn.append((len(mb), gq))
        if num_kv_blocks > 1:
            mask_dyn.append((len(mb) + 1, gkv))
    else:
        mask_node = None

    pattern = _FA3Pipeline(
        grid=grid,
        num_kv_blocks=num_kv_blocks,
        tq=tq,
        tkv=tkv,
        head_dim=d,
        scale=float(scale),
        has_mask=mask_node is not None,
        mask_is_bool=mask_is_bool,
        mask_dyn=mask_dyn,
        out_shape=tuple(out.shape),
        out_dtype=out.dtype,
        acc_dtype=torch.float32 if accumulate_fp32 else out.dtype,
    )

    # ``all_input_nodes`` is the SDPA operands in fixed arg order (query,
    # key, value, then optional attn_mask) — the order ``forward`` takes.
    inputs = tuple(n.value.clone() for n in node.all_input_nodes)
    with _lenient_verifier():
        gm = export_model(pattern, inputs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(1, pattern.num_steps, 1)]])
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*inputs)
    _fuse_passes(gm)
    return gm
