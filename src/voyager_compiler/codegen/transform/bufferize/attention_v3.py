"""FA3-style flash-attention bufferization builder (cross-sweep pipelined).

A dedicated scheduler (``_FA3Pipeline``) that overlaps the systolic (matrix)
array with the vector unit: each iteration runs the current block's QKᵀ matmul
and the previous block's P@V matmul on the matrix unit while the vector unit
softmaxes, a one-step skew fused across Q-tile / head boundaries.  The pipeline
is peeled — a prologue, ``num_steps - 1`` uniform loop iterations, an epilogue.
The matmuls (QKᵀ, P@V) and the DMA are asynchronous; the softmax chain and the
accumulate/rescale are synchronous.  Each async matmul is dispatched with
``voyager.commit`` — its input load semaphores are the ``dependencies`` and it
posts a done-semaphore (``sem_scores`` for QKᵀ, ``sem_pv`` for P@V) that
``voyager.async_wait`` consumes where a synchronous op reads the result.

Per loop iteration (``N = num_kv_blocks``, ``kv = cur[gkv]``), in program order:

  [A] S = (Q @ Kᵀ)·scale (+ mask) -> s_buf[step % 2];  commit, post sem_scores
  [B] pv_buf = p_buf[(step-1) % 2] @ V_prev;  commit (dep V DMA), post sem_pv
  [C] kv == 0 only (a Q-tile boundary): wait sem_pv; o += pv_buf; drain the
      previous output store; (o / l) -> o_slots; store o_slots to the PREVIOUS
      tile's DRAM rows (the finalize belongs to the tile that just ended);
      reset m/o/l.
  [D] wait sem_scores;  softmax chain (rowmax, m, alpha, P, rowsum, l).
  [E] kv >= 1 only (vector unit, after [D]): wait sem_pv;  the deferred
      rescale fused with the accumulate, o = alpha·(o + pv_buf).

The probabilities ``p_buf`` are ``s_buf`` itself -- [D] exponentiates the
scores in place -- unless the softmax runs at a different dtype than the
operands (``accumulate_fp32`` over bf16 inputs), when P is written to its
own buffer at V's dtype so the matrix unit reads it as an operand.

Microscaling operands (an ``sdpa_mx`` node): Q, K and V are codes with
block scales, and each scale streams through slots of its own beside its
data, K's transposed by the DMA like K.  Both products are ``matmul_mx``.
[D] exponentiates S in place, sums the exact exponentials into ``l``, then
one more vector pass quantizes them along the keys with the query's
parameters into P's codes (``p_buf``) and block scales (``p_scale``).
"""

import logging
import math

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.transform.bufferize.attention import (
    _MASK_FILL,
    _fuse_passes,
    fold_mask_tensor,
    plan_gqa_fold,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    MemoryLevel,
    commit,
    oracle_disabled,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    _guarded_wait,
    get_slot,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)
from voyager_compiler.codegen.transform.tiling import attention_op_tiling
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import ShapeProp

_SRAM = int(MemoryLevel.SRAM)
_ATTENTION_OPS = (
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.quantized_ops.sdpa_mx.default,
)
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
_PRODUCT_OPS = (
    torch.ops.aten.matmul.default,
    torch.ops.quantized_ops.matmul_mx.default,
)

logger = logging.getLogger(__name__)


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
        g_head,
        sq_orig,
        q_fold_shape,
        out_unfold_shape,
        mask_broadcast,
        mask_fold_shape,
        operand_index,
        block_size,
        probs_quant_max,
        force_scale_power_of_two,
    ):
        super().__init__()
        # ``forward`` takes the node's operands in ``all_input_nodes`` order;
        # ``operand_index`` names each position.  ``block_size`` is the MX
        # block along the contraction axes, ``None`` for unquantized
        # operands; the rest is what P's ``quantize_mx`` takes.
        self.operand_index = operand_index
        self.block_size = block_size
        self.probs_quant_max = probs_quant_max
        self.force_scale_power_of_two = force_scale_power_of_two
        # GQA fold: the ``g_head`` query heads sharing a KV head are folded
        # into the query rows (see ``build_attention_fa3``), so the pipeline
        # runs plain MHA over the KV-head batch and K/V load once per KV head.
        # ``forward`` reshapes q / mask in and the output back out.
        self.g_head = g_head
        self.sq_orig = sq_orig
        self.q_fold_shape = tuple(q_fold_shape)
        self.out_unfold_shape = tuple(out_unfold_shape)
        self.mask_broadcast = mask_broadcast
        self.mask_fold_shape = (
            tuple(mask_fold_shape) if mask_fold_shape is not None else None
        )
        self.grid = tuple(grid)  # (*kv_batch, num_q_blocks, num_kv_blocks)
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

    def _load_sweep(self, src, slots, slot, sem, coords, tile):
        """DMA ``src``'s ``tile`` at ``coords``' query block into
        ``slots[slot]``.  The block is loaded once per sweep but read on
        every one of the sweep's N steps, so its semaphore posts N times to
        balance the per-step [A] consume."""
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        voyager.async_copy(
            src,
            get_slot(slots, slot),
            idx,
            (1,) * self.nb + tuple(tile),
            get_slot(sem, slot),
            dims,
            post_count=self.N,
        )

    def _load_step(self, src, slots, slot, sem, coords, tile, transposed):
        """DMA ``src``'s ``tile`` at ``coords``' key block into
        ``slots[slot]``.  ``tile`` is the DRAM block in the buffer's own
        order; ``transposed`` has the DMA ``.mT`` it into the slot (the Kᵀ
        tiles, as ``PipelinedKernel._load_tile`` does)."""
        dims, idx = self._block_address(coords, self.gkv, self.grid[self.gkv])
        voyager.async_copy(
            src,
            get_slot(slots, slot),
            idx,
            (1,) * self.nb + tuple(tile),
            get_slot(sem, slot),
            dims,
            transposed=transposed,
        )

    def _load_mask(self, mask, slots, slot, sem, coords):
        dims = [d for d, _ in self.mask_dyn]
        idx = [coords[g] for _, g in self.mask_dyn]
        munit = (1,) * (mask.ndim - 2)
        voyager.async_copy(
            mask,
            get_slot(slots, slot),
            idx,
            munit + (self.tq, self.tkv),
            get_slot(sem, slot),
            dims,
        )

    def _store_out(self, tile, out, sem, coords):
        dims, idx = self._block_address(coords, self.gq, self.grid[self.gq])
        unit = (1,) * self.nb
        voyager.async_copy(
            tile, out, idx, unit + (self.tq, self.d), get_slot(sem, 0), dims
        )

    # --- compute helpers (shared by prologue / loop / epilogue) ---------

    def _matmul(self, *tiles):
        """The matrix unit's product of two tiles, ``a @ b``: ``(a, b)``, or
        under MX ``(a, a_scale, b, b_scale, a_code, b_code)`` -- each
        operand's codes, then its block scales, then the codebooks."""
        if self.block_size is None:
            a, b = tiles
            return torch.matmul(a, b)
        a, a_scale, b, b_scale, a_code, b_code = tiles
        return torch.ops.quantized_ops.matmul_mx(
            a,
            b,
            input_scale=a_scale,
            weight_scale=b_scale,
            block_size=self.block_size,
            input_code=a_code,
            weight_code=b_code,
        )

    def _matmul_qk(self, operands, s_slot, sem_scores, deps):
        """[A]: commit S = (Q @ Kᵀ)·scale (+ mask) -> s_slot.  ``operands``
        are the product's tiles (``_matmul``), then the mask tile when
        there is one; ``deps`` the load semaphores it waits on.  It posts
        ``sem_scores`` when the matmul retires.  The body branches on
        ``has_mask`` at build time, so the traced subgraph carries no
        runtime cond."""
        scale, has_mask = self.scale, self.has_mask
        mask_is_bool, matmul = self.mask_is_bool, self._matmul

        def body(*args):
            *tiles, s = args
            if has_mask:
                *tiles, mask = tiles
            out = matmul(*tiles) * scale
            if has_mask:
                if mask_is_bool:
                    out = torch.where(mask, out, _MASK_FILL)
                else:
                    out = out + mask
            voyager.insert(out, s)

        commit(body, [*operands, s_slot], dependencies=deps, post=sem_scores)

    def _matmul_pv(self, operands, pv_buf, sem_pv, deps):
        """[B]: commit pv_buf = P @ V on the matrix unit; ``operands`` are
        the product's tiles (``_matmul``).  ``deps`` are the V load
        semaphores; it posts ``sem_pv`` when the matmul retires (the
        probabilities are written synchronously, so need no dep)."""
        matmul = self._matmul

        def body(*args):
            *tiles, pv = args
            voyager.insert(matmul(*tiles), pv)

        commit(body, [*operands, pv_buf], dependencies=deps, post=sem_pv)

    def _softmax(
        self, s_slot, p_slot, m, l, row_tmp, alpha, sem_scores, p_scale, probs
    ):
        """[D]'s chain: waits for S, then rowmax / m / alpha / P / rowsum
        / l (the baseline passes 2-7).  Unquantized, P lands in ``p_slot``
        -- ``s_slot`` itself unless the probabilities have their own buffer
        -- and the rowsum reads it back, so ``l`` sums the P the matmul
        will see.  Under MX, S is exponentiated in place and summed exact,
        then one more pass quantizes it along the keys with the query's
        parameters -- ``probs``: the lookup table, the scale codebook and
        the midpoints -- into ``p_slot``'s codes and ``p_scale``."""
        mx = self.block_size is not None
        voyager.async_wait(sem_scores)
        voyager.insert(torch.amax(s_slot, dim=-1, keepdim=True), row_tmp)
        voyager.insert(torch.maximum(m, row_tmp), row_tmp)
        voyager.insert(torch.exp(m - row_tmp), alpha)
        voyager.insert(row_tmp.clone(), m)
        exp_slot = s_slot if mx else p_slot
        voyager.insert(torch.exp(s_slot - row_tmp), exp_slot)
        voyager.insert(torch.sum(exp_slot, dim=-1, keepdim=True), row_tmp)
        voyager.insert(alpha * l + row_tmp, l)
        if mx:
            qmap, scale_qmap, code = probs
            scale, codes = torch.ops.quantized_ops.quantize_mx(
                s_slot,
                qmap,
                [-1],
                self.block_size,
                self.probs_quant_max,
                self.force_scale_power_of_two,
                scale_qmap,
                code,
            )
            voyager.insert(scale, p_scale)
            voyager.insert(codes, p_slot)

    def _reset(self, m, l, o):
        voyager.insert(torch.full_like(m, _MASK_FILL), m)
        voyager.insert(torch.zeros_like(l), l)
        voyager.insert(torch.zeros_like(o), o)

    def _fold_mask(self, mask):
        return fold_mask_tensor(
            mask,
            mask_broadcast=self.mask_broadcast,
            sq_orig=self.sq_orig,
            g_head=self.g_head,
            mask_fold_shape=self.mask_fold_shape,
        )

    # --------------------------------------------------------------------

    def forward(self, *operands):
        index = self.operand_index
        q, k, v = (operands[index[n]] for n in ("query", "key", "value"))
        mask = operands[index["attn_mask"]] if "attn_mask" in index else None
        bs = self.block_size
        mx = bs is not None
        if mx:
            q_scale, k_scale, v_scale = (
                operands[index[n]]
                for n in ("query_scale", "key_scale", "value_scale")
            )
            codes = [
                operands[index["input_code"]],
                operands[index["weight_code"]],
            ]
            probs = tuple(
                operands[index[n]] if n in index else None
                for n in ("probs_qmap", "probs_scale_qmap", "probs_code")
            )
        # GQA fold: reshape q (and mask) so the shared query heads become extra
        # query rows; the output is reshaped back at the end.
        if self.g_head > 1:
            q = q.reshape(self.q_fold_shape)
            if mx:
                q_scale = q_scale.reshape(
                    (*self.q_fold_shape[:-1], q_scale.shape[-1])
                )
            if mask is not None:
                mask = self._fold_mask(mask)
        grid, N, nb = self.grid, self.N, self.nb
        num_steps = self.num_steps
        unit = (1,) * nb
        tq, tkv, d = self.tq, self.tkv, self.d
        out = voyager.alloc(self.out_shape, self.out_dtype)

        # SRAM slots (2 slots each) + per-slot DMA semaphores; under MX each
        # operand's block scales stream beside it through slots of their
        # own.  The output slot is single-slotted (written once per Q tile).
        def stream(src, tile):
            slots = voyager.alloc([*unit, *tile], src.dtype, _SRAM, 2)
            return slots, voyager.zeros([], torch.int64, num_slots=2)

        q_slots, q_sem = stream(q, (tq, d))
        k_slots, k_sem = stream(k, (d, tkv))
        v_slots, v_sem = stream(v, (tkv, d))
        if mx:
            q_scale_slots, q_scale_sem = stream(q_scale, (tq, d // bs))
            k_scale_slots, k_scale_sem = stream(k_scale, (d // bs, tkv))
            v_scale_slots, v_scale_sem = stream(v_scale, (tkv // bs, d))
        if self.has_mask:
            munit = (1,) * (mask.ndim - 2)
            m_slots = voyager.alloc([*munit, tq, tkv], mask.dtype, _SRAM, 2)
            m_sem = voyager.zeros([], torch.int64, num_slots=2)
        out_slots = voyager.alloc([*unit, tq, d], out.dtype, _SRAM, 1)
        out_sem = voyager.zeros([], torch.int64, num_slots=1)

        # Running softmax state (single-buffered — see module docstring)
        # and the parity-double-buffered scores/probabilities tile.
        acc = self.acc_dtype
        m = voyager.alloc([*unit, tq, 1], acc, _SRAM)
        l = voyager.alloc([*unit, tq, 1], acc, _SRAM)
        o = voyager.alloc([*unit, tq, d], acc, _SRAM)
        s_buf = voyager.alloc([*unit, tq, tkv], acc, _SRAM, 2)
        # P: under MX its codes and block scales (see the module docstring);
        # else S exponentiated in place, unless the softmax runs at a dtype
        # the matrix unit cannot take as an operand.
        p_scale = None
        if mx:
            p_buf = voyager.alloc([*unit, tq, tkv], q.dtype, _SRAM, 2)
            p_scale = voyager.alloc(
                [*unit, tq, tkv // bs], q_scale.dtype, _SRAM, 2
            )
        elif acc != v.dtype:
            p_buf = voyager.alloc([*unit, tq, tkv], v.dtype, _SRAM, 2)
        else:
            p_buf = s_buf
        pv_buf = voyager.alloc([*unit, tq, d], acc, _SRAM)
        row_tmp = voyager.alloc([*unit, tq, 1], acc, _SRAM)
        alpha = voyager.alloc([*unit, tq, 1], acc, _SRAM)
        sem_scores = voyager.zeros([1], torch.int64)
        sem_pv = voyager.zeros([1], torch.int64)

        # The three DMA streams, each with its scales under MX.  K's DRAM
        # block (and its scales') is transposed by the DMA into the Kᵀ tile
        # the product reads; the mask block rides with K.
        def load_q(slot, coords):
            self._load_sweep(q, q_slots, slot, q_sem, coords, (tq, d))
            if mx:
                self._load_sweep(
                    q_scale,
                    q_scale_slots,
                    slot,
                    q_scale_sem,
                    coords,
                    (tq, d // bs),
                )

        def load_k(slot, coords):
            self._load_step(k, k_slots, slot, k_sem, coords, (tkv, d), True)
            if mx:
                self._load_step(
                    k_scale,
                    k_scale_slots,
                    slot,
                    k_scale_sem,
                    coords,
                    (tkv, d // bs),
                    True,
                )
            if self.has_mask:
                self._load_mask(mask, m_slots, slot, m_sem, coords)

        def load_v(slot, coords):
            self._load_step(v, v_slots, slot, v_sem, coords, (tkv, d), False)
            if mx:
                self._load_step(
                    v_scale,
                    v_scale_slots,
                    slot,
                    v_scale_sem,
                    coords,
                    (tkv // bs, d),
                    False,
                )

        # The tiles each product reads at the given slots, and the load
        # semaphores it waits on.
        def qk_operands(q_slot, k_slot):
            q_tile = get_slot(q_slots, q_slot)
            k_tile = get_slot(k_slots, k_slot)
            deps = [get_slot(k_sem, k_slot), get_slot(q_sem, q_slot)]
            if mx:
                tiles = [
                    q_tile,
                    get_slot(q_scale_slots, q_slot),
                    k_tile,
                    get_slot(k_scale_slots, k_slot),
                    *codes,
                ]
                deps += [
                    get_slot(k_scale_sem, k_slot),
                    get_slot(q_scale_sem, q_slot),
                ]
            else:
                tiles = [q_tile, k_tile]
            if self.has_mask:
                tiles.append(get_slot(m_slots, k_slot))
                deps.append(get_slot(m_sem, k_slot))
            return tiles, deps

        def pv_operands(p_slot, v_slot):
            p_tile = get_slot(p_buf, p_slot)
            v_tile = get_slot(v_slots, v_slot)
            deps = [get_slot(v_sem, v_slot)]
            if mx:
                tiles = [
                    p_tile,
                    get_slot(p_scale, p_slot),
                    v_tile,
                    get_slot(v_scale_slots, v_slot),
                    *codes,
                ]
                deps.append(get_slot(v_scale_sem, v_slot))
            else:
                tiles = [p_tile, v_tile]
            return tiles, deps

        def softmax(slot):
            self._softmax(
                get_slot(s_buf, slot),
                get_slot(p_buf, slot),
                m,
                l,
                row_tmp,
                alpha,
                sem_scores,
                get_slot(p_scale, slot) if mx else None,
                probs if mx else None,
            )

        # ---- prologue: prime the DMA and run step 0's [A] + [D] --------
        c0 = _unravel(0, grid)
        load_q(0, c0)
        load_k(0, c0)
        if num_steps > 1:
            c1 = _unravel(1, grid)
            load_k(1, c1)
            if N == 1:
                # Step 0 is also its sweep's LAST step, so the uniform
                # pattern's end-of-sweep Q prefetch belongs here too.
                load_q(1, c1)
        # V's stream runs one step behind K's: block 0 lands in the slot
        # step 1 reads (slot 1); nothing is fetched for step 0, which
        # consumes no V.
        load_v(1 % 2, c0)

        self._reset(m, l, o)
        tiles, deps = qk_operands(0, 0)
        self._matmul_qk(tiles, get_slot(s_buf, 0), sem_scores, deps)
        softmax(0)

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
                load_k(nxt_slot, nxt)
                return 1

            torch.cond(step + 1 < num_steps, k_fetch, lambda: 0)
            load_v(nxt_slot, cur)

            q_next_slot = ((step + 1) // N) % 2
            torch._check(q_next_slot < 2)

            def q_fetch():
                load_q(q_next_slot, nxt)
                return 1

            torch.cond(
                (kv == N - 1) & (step + 1 < num_steps), q_fetch, lambda: 0
            )

            # [A] current block's scores on the matrix unit: its K (and mask)
            # load semaphores are per-step commit dependencies; Q's is too, but
            # Q is loaded once per sweep, so its load posts N times (see
            # _load_sweep) to balance the per-step consume.  V is not a
            # dependency here — it feeds [B], so [A] issues while V's DMA is
            # in flight.
            q_slot = (step // N) % 2
            torch._check(q_slot < 2)
            tiles, deps = qk_operands(q_slot, cur_slot)
            self._matmul_qk(tiles, get_slot(s_buf, cur_slot), sem_scores, deps)

            # [B] the lagged P@V on the matrix unit: previous step's
            # probabilities × its V block (in this step's V read slot) into
            # pv_buf; the V load semaphore is the commit dependency.
            prev_slot = (step - 1) % 2
            torch._check(prev_slot < 2)
            tiles, deps = pv_operands(prev_slot, cur_slot)
            self._matmul_pv(tiles, pv_buf, sem_pv, deps)

            # [C] Q-tile boundary: land the previous tile's last P@V into o,
            # finalize (o / l) to ITS rows, reset for the new tile.  Runs
            # before [D] so the new tile's softmax sees fresh m/l.
            def boundary():
                voyager.async_wait(sem_pv)
                voyager.insert(o + pv_buf, o)
                # Drain the previous boundary's store before overwriting
                # the slot (no prior store exists at the first boundary).
                _guarded_wait(get_slot(out_sem, 0), step >= 2 * N)
                voyager.insert(
                    (o / l).to(self.out_dtype), get_slot(out_slots, 0)
                )
                self._store_out(get_slot(out_slots, 0), out, out_sem, prev)
                self._reset(m, l, o)
                return 1

            torch.cond(kv == 0, boundary, lambda: 0)

            # [D] softmax of the current block on the vector unit — runs
            # (synchronously) while the matrix [B] matmul is still in flight.
            softmax(cur_slot)

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
        tiles, deps = pv_operands((num_steps - 1) % 2, v_slot)
        self._matmul_pv(tiles, pv_buf, sem_pv, deps)
        # land it into o on the vector unit, then finalize.
        voyager.async_wait(sem_pv)
        voyager.insert(o + pv_buf, o)
        if num_steps > N:  # a prior boundary store exists (static)
            voyager.async_wait(get_slot(out_sem, 0))
        voyager.insert((o / l).to(self.out_dtype), get_slot(out_slots, 0))
        self._store_out(get_slot(out_slots, 0), out, out_sem, c_last)
        voyager.async_wait(get_slot(out_sem, 0))  # drain
        if self.g_head > 1:
            out = out.reshape(self.out_unfold_shape)
        return out


def _operand_index(node):
    """Position of each named operand -- query, key, value, attn_mask and
    the MX kwargs -- among ``node.all_input_nodes``, the order
    ``_FA3Pipeline.forward`` receives them."""
    operands = {
        "query": node.args[0],
        "key": node.args[1],
        "value": node.args[2],
    }
    mask = get_arg_value(node, 3, "attn_mask", None)
    if isinstance(mask, torch.fx.Node):
        operands["attn_mask"] = mask
    operands.update(
        (name, n)
        for name, n in node.kwargs.items()
        if isinstance(n, torch.fx.Node)
    )
    inputs = list(node.all_input_nodes)
    return {name: inputs.index(n) for name, n in operands.items()}


def _stamp_probs_dtypes(gm, dtypes):
    """Stamp ``dtypes`` -- P's ``(scale, codes)`` logical dtypes, the
    query's -- on every ``quantize_mx`` in ``gm`` and its nested graphs:
    no node of the original graph carries them for the bufferizer's
    propagation to read."""
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is _QUANTIZE_MX:
            n.meta["dtype"] = dtypes
        elif n.op in ("get_attr", "call_module"):
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _stamp_probs_dtypes(sub, dtypes)


def _stamp_product_tilings(gm, products, tkv):
    """Copy the products' mapping metadata (``attention_op_tiling``) onto
    every kernel of ``gm`` that runs one -- the bare product op or the
    fused ``call_module`` around it, at every nesting level -- the way the
    GEMM builder stamps its nest, so the emitter and the reporting model
    see the mapping the kernel runs.  A product is told by its output: the
    scores are ``tkv`` wide, the context ``head_dim`` wide."""
    named = dict(gm.named_modules())
    for n in gm.graph.nodes:
        sub = named.get(str(n.target)) if n.op != "call_function" else None
        if n.op == "get_attr" and isinstance(sub, torch.fx.GraphModule):
            _stamp_product_tilings(sub, products, tkv)
            continue
        if n.op == "call_function" and n.target in _PRODUCT_OPS:
            anchor = n
        elif isinstance(sub, torch.fx.GraphModule):
            anchor = next(
                (x for x in sub.graph.nodes if x.target in _PRODUCT_OPS), None
            )
        else:
            anchor = None
        if anchor is None:
            continue
        which = "scores" if anchor.value.shape[-1] == tkv else "context"
        n.meta.update(products[which])


def build_attention_fa3(
    node,
    *,
    accumulate_fp32: bool = False,
    tiler=None,
):
    """FA3-style pipeline builder for an
    ``aten.scaled_dot_product_attention`` node, or its ``sdpa_mx`` twin
    whose operands are MX codes with block scales.

    Returns the bufferized ``GraphModule`` (prologue + rolled
    ``while_loop`` + epilogue over ``voyager.*`` primitives, see the
    module docstring), or ``None`` when uncovered (unsupported rank, an
    unfoldable GQA layout, no tiling and no ``tiler`` to choose one).  GQA
    is supported by folding the head group into the query rows (below);
    ``dropout_p`` is ignored (identity at inference).  ``accumulate_fp32``
    runs the softmax and the output accumulator in fp32 instead of the
    output dtype, at the cost of a separate probabilities buffer at V's
    dtype when the two differ.  The block counts
    ``(num_q_blocks, num_kv_blocks)`` come from ``node.meta['l2_tiling']``
    when set, else ``attention_op_tiling`` chooses them under
    ``tiler.config``.
    """
    if node.op != "call_function" or node.target not in _ATTENTION_OPS:
        return None

    q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
    mask_node = get_arg_value(node, 3, "attn_mask", None)
    dropout_p = get_arg_value(node, 4, "dropout_p", 0.0)
    is_causal = get_arg_value(node, 5, "is_causal", False)
    scale = node.kwargs.get("scale", None)
    if dropout_p:
        logger.warning(
            "%s: dropout_p=%s ignored (attention dropout is identity at "
            "inference; lowering the dropout-free kernel)",
            node.name,
            dropout_p,
        )
    if is_causal:
        raise NotImplementedError(
            "is_causal attention is not supported yet; pass an explicit "
            "additive attn_mask instead"
        )

    block_size = node.kwargs.get("block_size")  # None: unquantized

    q = q_node.value.clone()
    k = k_node.value.clone()
    v = v_node.value.clone()
    out = node.value
    if q.ndim < 2 or q.ndim != k.ndim or q.ndim != v.ndim:
        return None

    nb = q.ndim - 2
    Sq, d = q.shape[-2], q.shape[-1]
    Skv = k.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    acc_dtype = torch.float32 if accumulate_fp32 else out.dtype

    # GQA folds the head group into the query rows so K/V load once per KV head
    # and reuse across the group (see ``plan_gqa_fold``); ``forward`` applies
    # the q / mask / output reshapes.  The fold only needs the query block
    # count to check divisibility, and a count the tiler picks divides by
    # construction, so an unset tiling plans at one block.
    mask_val = mask_node.value if isinstance(mask_node, torch.fx.Node) else None
    tiling = node.meta.get("l2_tiling")
    plan = plan_gqa_fold(
        q, k, v, mask_val, out.shape, tiling[0] if tiling else 1
    )
    if plan is None:
        return None
    kbatch, g_head = plan.kbatch, plan.g_head
    if tiling is None:
        if tiler is None:
            return None
        tiling = attention_op_tiling(
            node,
            tiler,
            sq_eff=plan.sq_eff,
            kv_batch=kbatch,
            acc_dtype=acc_dtype,
        )
    num_q_blocks, num_kv_blocks = tiling
    tq, tkv = plan.sq_eff // num_q_blocks, Skv // num_kv_blocks
    grid = kbatch + (num_q_blocks, num_kv_blocks)
    gq, gkv = nb, nb + 1

    # Mask [*mbatch, Sq, Skv]: dynamic dims are the >1-block batch dims (a
    # size-1 batch dim broadcasts, pinned to block 0) plus the q / kv dims when
    # tiled.  Under the fold the mask's batch dims are the folded ones.
    mask_is_bool = False
    mask_dyn = []
    if isinstance(mask_node, torch.fx.Node):
        mask = mask_node.value
        mask_is_bool = mask.dtype == torch.bool
        mb = tuple(mask.shape[:-2])
        if g_head > 1:
            mb = plan.mask_fold_shape[:-2]
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
        g_head=g_head,
        sq_orig=Sq,
        q_fold_shape=plan.q_fold_shape,
        out_unfold_shape=plan.out_unfold_shape,
        mask_broadcast=plan.mask_broadcast,
        mask_fold_shape=plan.mask_fold_shape,
        out_shape=plan.out_fold_shape,
        out_dtype=out.dtype,
        acc_dtype=acc_dtype,
        operand_index=_operand_index(node),
        block_size=block_size,
        probs_quant_max=node.kwargs.get("probs_quant_max"),
        force_scale_power_of_two=node.kwargs.get(
            "force_scale_power_of_two", False
        ),
    )

    # ``all_input_nodes`` is the order ``forward`` takes its operands
    # (``_operand_index``).
    inputs = tuple(n.value.clone() for n in node.all_input_nodes)
    with _lenient_verifier():
        gm = export_model(pattern, inputs)
    gm = _finalize_exported_gm(gm)
    _tag_loop_extents(gm, [[(1, pattern.num_steps, 1)]])
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*inputs)
    _fuse_passes(gm)
    if block_size is not None:
        _stamp_probs_dtypes(
            gm,
            (
                node.kwargs["query_scale"].meta.get("dtype"),
                q_node.meta.get("dtype"),
            ),
        )
    products = node.meta.get("product_tilings")
    if products is not None:
        _stamp_product_tilings(gm, products, tkv)
    return gm
