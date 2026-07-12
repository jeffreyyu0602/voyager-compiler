"""Flash-attention bufferization builder (the ``attention`` member of the
gemm / pointwise / attention builder family).

Lowers a single ``aten.scaled_dot_product_attention`` node into one rolled,
software-pipelined ``while_loop`` over the ``PipelinedKernel`` scheduler,
computing an *online* (running-max / running-sum) softmax over a KV reduction
grid dim — so the full ``[Sq, Skv]`` scores matrix never materializes (the
Pallas-TPU flash-attention structure).

Grid ``(*batch, n_q, n_kv)`` with ``kv`` innermost = the reduction dim.  Per kv
step the kernel is a sequence of **hardware passes**, each either a single op or
a fused *elementwise chain* / GEMM-with-epilogue, writing its result to an SRAM
buffer (fusion is hardware-aware: only elementwise chains and a GEMM + its
elementwise epilogue collapse into one pass; every reduction and every matmul is
its own pass).  The running softmax state lives in three persistent scratch refs
(``m`` max, ``l`` denominator, ``o`` accumulator); three transient SRAM buffers
(``s_buf`` scores→probs, ``row_tmp`` per-row stat, ``alpha`` rescale) carry the
intermediates between passes.

Passes per kv step (``m,l,o`` reset at ``kv==0``; the output normalized and
stored at ``kv==last``):

  1. ``S = (Q@Kᵀ)·scale + mask``             GEMM + elementwise epilogue → s_buf
  2. ``row_tmp = rowmax(s_buf)``             reduction
  3. ``row_tmp = maximum(m, row_tmp)``       elementwise (row_tmp := m_new)
  4. ``alpha = exp(m - row_tmp)``            elementwise (old m + m_new)
  4b ``m = row_tmp``                         copy (update running max)
  5. ``s_buf = exp(s_buf - row_tmp)``        elementwise (S→P, uses m_new)
  6. ``row_tmp = rowsum(s_buf)``             reduction (row_tmp := blk_sum)
  7. ``l = alpha·l + row_tmp``               elementwise chain
  8. ``o = alpha·o``                         elementwise
  9. ``o = o + s_buf@V``                     GEMM + accumulate

``K`` is transposed in ``async_copy`` (``_InputSpec.transposed``), so no ``.mT``
op appears in the body.  An additive / boolean ``attn_mask`` is a tensor operand
added into pass 1's epilogue.  ``is_causal`` is not supported yet (the hardware
lacks on-chip causal masking); callers pass an explicit additive mask instead.
"""

import math
import operator

import torch

from voyager_compiler.codegen.lowering.pipeline import (
    _DEFAULT_NUM_BANKS,
    build_pipelined_buffers,
)
from voyager_compiler.codegen.lowering.utils import (
    _InputSpec,
    _OutputSpec,
    _ScratchSpec,
    voyager,
)
from voyager_compiler.codegen.passes.utils import get_arg_value

# Additive fill for masked-out score positions (drives ``exp`` to ~0 without
# ``-inf`` NaNs when a whole row is masked); Pallas uses the same trick.
_MASK_FILL = -0.7 * torch.finfo(torch.float32).max

_INSERT = voyager.insert.default


def _flash_attention_kernel(
    has_mask: bool,
    mask_is_bool: bool,
    reduction_dim: int,
    last_idx: int,
    scale: float,
    out_dtype: torch.dtype,
):
    """Build the mutate-style per-tile kernel ``kernel(grid_index, *in_tiles,
    o_bank, m, l, o, s_buf, row_tmp, alpha)`` for flash attention (see module
    docstring for the pass sequence).  ``in_tiles`` are the SDPA operands in arg
    order — ``q, kᵀ, v`` then the optional ``attn_mask``.
    """

    def kernel(grid_index, *args):
        # args = (*in_tiles, o_bank, m, l, o, s_buf, row_tmp, alpha)
        m, l, o, s_buf, row_tmp, alpha = args[-6:]
        o_bank = args[-7]
        in_tiles = args[:-7]
        # kT is already Kᵀ ([.., d, tkv]) via the transposed DMA.
        q, kT, v = in_tiles[0], in_tiles[1], in_tiles[2]

        kv = grid_index[reduction_dim]

        # Pass 0 (guard kv==0): reset the running softmax state.
        def reset():
            voyager.insert(torch.full_like(m, _MASK_FILL), m)
            voyager.insert(torch.zeros_like(l), l)
            voyager.insert(torch.zeros_like(o), o)
            return 1

        torch.cond(kv == 0, reset, lambda: 0)

        # Pass 1: S = (Q @ Kᵀ)·scale + mask  -> s_buf.  The mask is a tensor
        # operand (additive, or boolean select), so this stays a fused
        # GEMM + elementwise epilogue.
        s = torch.matmul(q, kT) * scale
        if has_mask:
            mask = in_tiles[3]  # attn_mask is the 4th SDPA operand
            if mask_is_bool:
                s = torch.where(mask, s, torch.full_like(s, _MASK_FILL))
            else:
                s = s + mask
        voyager.insert(s, s_buf)

        # Pass 2: row-max of the scores -> row_tmp.
        voyager.insert(torch.amax(s_buf, dim=-1, keepdim=True), row_tmp)
        # Pass 3: row_tmp := m_new = maximum(m, row_tmp)  (in place).
        voyager.insert(torch.maximum(m, row_tmp), row_tmp)
        # Pass 4: alpha = exp(m_old − m_new)  (old m from scratch + m_new).
        voyager.insert(torch.exp(m - row_tmp), alpha)
        # Pass 4b: update the running max (m := m_new).  Read after pass 4's
        # old-m read; pass 5 uses row_tmp (still m_new), so m is not re-read.
        voyager.insert(row_tmp.clone(), m)
        # Pass 5: s_buf := P = exp(S − m_new).
        voyager.insert(torch.exp(s_buf - row_tmp), s_buf)
        # Pass 6: row_tmp := blk_sum = rowsum(P).
        voyager.insert(torch.sum(s_buf, dim=-1, keepdim=True), row_tmp)
        # Pass 7: l := alpha·l + blk_sum.
        voyager.insert(alpha * l + row_tmp, l)
        # Pass 8: o := alpha·o  (rescale the accumulator).
        voyager.insert(alpha * o, o)
        # Pass 9: o := o + P@V.
        voyager.insert(o + torch.matmul(s_buf, v), o)

        # Pass 10 (guard kv==last): normalize and store the output tile.
        def finalize():
            voyager.insert((o / l).to(out_dtype), o_bank)
            return 1

        torch.cond(kv == last_idx, finalize, lambda: 0)

    return kernel


def _fuse_passes(gm: torch.fx.GraphModule) -> None:
    """Collapse each hardware-fusable pass (a multi-op elementwise chain, or a
    GEMM + its elementwise epilogue) into one ``call_module``, recursing into
    the ``while_loop`` body and ``cond`` branches.

    A pass is the backward compute cone feeding a single ``insert`` — and by
    construction each cone is exactly one hardware pass (the kernel writes every
    intermediate to a buffer, so a cone never spans two matmuls or crosses a
    reduction).  A node joins the cone only when **all** its users are already
    in it, so a value read by a later pass stays a boundary input, not absorbed.
    Single-op cones (reductions, a lone ``maximum`` / ``mul``) are left as-is.
    """
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph

    for n in list(gm.graph.nodes):
        if n.op == "get_attr":
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _fuse_passes(sub)

    # A node absorbable into a pass's compute cone.  ``select.int`` is the
    # bank-slot tile read (``bank[step % num_banks]``) — the load boundary, not
    # compute; stopping there keeps its integer slot index out of the group's
    # inputs (it would break the fused submodule's tensor-only ShapeProp).
    _boundary = (_INSERT, operator.getitem, torch.ops.aten.select.int)

    def _is_compute(n):
        return (
            isinstance(n, torch.fx.Node)
            and n.op == "call_function"
            and n.target not in _boundary
        )

    for cp in [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is _INSERT
    ]:
        src = cp.args[0]
        if not _is_compute(src):
            continue
        group = [src]
        changed = True
        while changed:
            changed = False
            for nd in list(group):
                for inp in nd.all_input_nodes:
                    if (
                        _is_compute(inp)
                        and inp not in group
                        and all(u in group for u in inp.users)
                    ):
                        group.append(inp)
                        changed = True
        if len(group) >= 2:
            _create_and_insert_subgraph(group, gm)
    gm.graph.lint()
    gm.recompile()


def build_attention(
    node,
    *,
    num_banks: int = _DEFAULT_NUM_BANKS,
    accumulate_fp32: bool = True,
    tiler=None,
):
    """Pipeline builder for an ``aten.scaled_dot_product_attention`` node.

    Returns the bufferized ``GraphModule`` (a rolled ``while_loop`` over
    ``voyager.*`` primitives) implementing flash attention over a KV reduction
    grid dim, or ``None`` when uncovered (missing tiling, dropout / GQA,
    unsupported rank).  ``tiler`` is accepted for signature parity but unused —
    attention tiling comes from ``node.meta['l2_tiling']`` (``(n_q, n_kv)``).
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
        # On-chip causal masking (generating the triangular mask from the
        # tile's block indices) is not supported by the hardware yet; callers
        # must pass an explicit additive ``attn_mask`` instead.
        raise NotImplementedError(
            "is_causal attention is not supported yet; pass an explicit "
            "additive attn_mask instead"
        )

    tiling = node.meta.get("l2_tiling")
    if tiling is None:
        return None
    n_q, n_kv = tiling

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
    tq, tkv = Sq // n_q, Skv // n_kv
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Grid: (*batch, n_q, n_kv); kv (last) is the reduction sweep.
    gq, gkv = nb, nb + 1
    grid = batch + (n_q, n_kv)
    last_idx = n_kv - 1

    # Batch dims are never tiled (one tile per batch element): tile size 1,
    # mapped to their own grid dim.  ``head_dim d`` is loaded whole (index_map
    # ``None``), like a single-tile GEMM's contraction dim.
    batch_map = tuple(range(nb))
    unit = (1,) * nb
    false = (False,) * nb

    # Q: [.., tq, d] indexed by the q block only (guarded, reused over kv).
    q_spec = _InputSpec(
        unit + (tq, d), batch_map + (gq, None), false + (False, False)
    )
    # K: loaded transposed to Kᵀ ([.., d, tkv]).  The spec is in matmul (Kᵀ)
    # order; ``transposed`` tells ``async_copy`` the DRAM buffer is its (tkv, d)
    # transpose (swap the fetch, ``.mT`` into the bank) — as the GEMM weight.
    k_spec = _InputSpec(
        unit + (d, tkv),
        batch_map + (None, gkv),
        false + (False, False),
    )
    k_spec.transposed = True
    # V: [.., tkv, d] indexed by the kv block (streamed).
    v_spec = _InputSpec(
        unit + (tkv, d), batch_map + (gkv, None), false + (False, False)
    )

    spec_by_node = {q_node: q_spec, k_node: k_spec, v_node: v_spec}

    mask_is_bool = False
    if isinstance(mask_node, torch.fx.Node):
        mask = mask_node.value
        mask_is_bool = mask.dtype == torch.bool
        # Mask [*mbatch, Sq, Skv] tiles [tq, tkv] on the q / kv grid dims; a
        # size-1 batch dim broadcasts (pinned to block 0).
        mb = tuple(mask.shape[:-2])
        off = nb - len(mb)
        mtile, mimap, mbcast = [], [], []
        for j, sz in enumerate(mb):
            g = off + j
            bcast = sz == 1 and batch[g] != 1
            mtile.append(1)
            mimap.append(g)
            mbcast.append(bcast)
        spec_by_node[mask_node] = _InputSpec(
            tuple(mtile) + (tq, tkv),
            tuple(mimap) + (gq, gkv),
            tuple(mbcast) + (False, False),
        )
    else:
        mask_node = None

    # ``all_input_nodes`` is the SDPA operands in fixed arg order (query, key,
    # value, then optional attn_mask), which the kernel unpacks positionally.
    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]
    in_specs = [spec_by_node[n] for n in in_nodes]

    # Output tile [.., tq, d]; the kv reduction dim is dropped (index_map has no
    # ``gkv``), so the bank stays live across the sweep, written once at the
    # last kv step — single-buffered, drained at block-exit.
    out_spec = _OutputSpec(
        tuple(out.shape),
        unit + (tq, d),
        batch_map + (gq, None),
        out.dtype,
        num_banks=1,
        first_use_at_exit=True,
    )

    acc = torch.float32 if accumulate_fp32 else out.dtype
    scratch_specs = [
        _ScratchSpec(unit + (tq, 1), acc),  # m  running max
        _ScratchSpec(unit + (tq, 1), acc),  # l  running denominator
        _ScratchSpec(unit + (tq, d), acc),  # o  accumulator
        _ScratchSpec(unit + (tq, tkv), acc),  # s_buf  scores -> probs
        _ScratchSpec(unit + (tq, 1), acc),  # row_tmp  per-row stat
        _ScratchSpec(unit + (tq, 1), acc),  # alpha  rescale factor
    ]

    kernel = _flash_attention_kernel(
        has_mask=mask_node is not None,
        mask_is_bool=mask_is_bool,
        reduction_dim=gkv,
        last_idx=last_idx,
        scale=float(scale),
        out_dtype=out.dtype,
    )

    gm = build_pipelined_buffers(
        kernel,
        grid,
        in_specs,
        out_specs=[out_spec],
        inputs=tuple(inputs),
        scratch_specs=scratch_specs,
        num_banks=num_banks,
    )
    _fuse_passes(gm)
    return gm
