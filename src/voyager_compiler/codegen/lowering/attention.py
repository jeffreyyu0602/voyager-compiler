"""
Bufferization builders for scaled-dot-product attention (flash attention).

Replaces ``aten.scaled_dot_product_attention`` with an explicit, executable
*bufferized FX graph*: an output buffer (``voyager.alloc``), an outer
``while_loop`` over the ``(B, H, query-block)`` grid, and an inner ``while_loop``
over key blocks implementing the online-softmax flash-attention recurrence, with
``voyager.*`` tile loads/stores and plain integer loop counters.

Two variants, both in the same int-loop / static-addressing style as ``gemm.py``
and ``pointwise.py`` (and reusing their shared helpers):

  * ``TiledAttention``        — straightforward sequential flash attention.
  * ``FlashAttentionPipelined``  — software-pipelined (triple-buffered) kernel
    that overlaps DMA load / QK-matmul / softmax+PV across key blocks; falls back
    to the sequential kernel for fewer than four key blocks.

Both match ``scaled_dot_product_attention(query, key, value, scale=scale)``.
"""

import math
from typing import List, Optional

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.common import (
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)


class TiledAttention(torch.nn.Module):
    """
    Bufferized tiled flash attention (non-causal), matching
    ``scaled_dot_product_attention(query, key, value, scale=scale)``.

      query (B, H, N, d)   key / value (B, H, M, d)   output (B, H, N, d)

    The outer grid tiles the query rows in blocks of ``Br``; the inner reduction
    tiles the key/value rows in blocks of ``Bc`` and threads the online-softmax
    accumulators ``(O, l, m)`` across iterations.  ``scale`` defaults to
    ``1/sqrt(d)`` (the SDPA convention).
    """

    def __init__(
        self,
        *,
        scale: Optional[float] = None,
        accumulate_fp32: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.accumulate_fp32 = accumulate_fp32

    def _outer_loop_body(self, b, h, i, query, key, value, Br, Bc):
        d = query.shape[-1]
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype

        # Query tile is loaded once and reused across all key blocks.
        Qi = voyager.load_tile(query, (b, h, i), (1, 1, Br, d), dims=(0, 1, 2))

        # Online-softmax accumulators: output, row-sum, row-max.
        O_i = voyager.zero_tile((1, 1, Br, d), acc_dtype)
        l_i = voyager.zero_tile((1, 1, Br, 1), acc_dtype)
        m_i = voyager.zero_tile((1, 1, Br, 1), acc_dtype) + float("-inf")

        def cond_fn(j, O_buf, l_buf, m_buf):
            return j < key.shape[2] // Bc

        def body_fn(j, O_buf, l_buf, m_buf):
            # Compute scale inside the body: a closed-over Python float is not a
            # valid while_loop additional-input (only Tensor / int / SymInt).
            scale = self.scale if self.scale is not None else 1.0 / math.sqrt(d)
            Kj = voyager.load_tile(
                key, (b, h, j), (1, 1, Bc, d), dims=(0, 1, 2)
            )
            Vj = voyager.load_tile(
                value, (b, h, j), (1, 1, Bc, d), dims=(0, 1, 2)
            )

            scores = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale
            if self.accumulate_fp32 and scores.dtype != torch.float32:
                scores = scores.to(torch.float32)

            m_new = torch.maximum(m_buf, scores.amax(dim=-1, keepdim=True))
            alpha = torch.exp(m_buf - m_new)
            P = torch.exp(scores - m_new)
            l_new = alpha * l_buf + P.sum(dim=-1, keepdim=True)
            O_new = alpha * O_buf + torch.matmul(P, Vj.to(P.dtype))
            return (j + 1, O_new, l_new, m_new)

        _, O_i, l_i, m_i = while_loop(cond_fn, body_fn, (0, O_i, l_i, m_i))

        out = O_i / l_i
        if self.accumulate_fp32 and out.dtype != query.dtype:
            out = out.to(query.dtype)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int = 64,
        Bc: int = 128,
    ):
        B, H, N, d = query.shape
        Tr = N // Br

        O = voyager.alloc((B, H, N, d), query.dtype)

        def cond_fn(b, h, i, O_buf):
            return b < query.shape[0]

        def body_fn(b, h, i, O_buf):
            O_tile = self._outer_loop_body(b, h, i, query, key, value, Br, Bc)
            O_buf = voyager.store_tile(
                O_tile, O_buf, (b, h, i), (1, 1, Br, d), dims=(0, 1, 2)
            )
            # Advance the (b, h, i) grid (one fused increment op).
            b_next, h_next, i_next = voyager.increment_indices(
                (b, h, i), (B, H, Tr)
            )
            return (b_next, h_next, i_next, O_buf)

        # Loop counters are plain ints (0); no index tensors are materialized.
        _, _, _, final_O = while_loop(cond_fn, body_fn, (0, 0, 0, O))
        return final_O


class FlashAttentionPipelined(torch.nn.Module):
    """
    Software-pipelined flash attention (FlashAttention-V3-style triple buffering),
    int-index / ``voyager.*`` refactor.

    Each key block flows through three pipeline stages — DMA load, QK matmul,
    softmax + PV matmul — kept in flight simultaneously so softmax latency is
    hidden behind the matmuls.  ``_prologue`` primes the pipeline (tiles 0/1/2),
    the steady-state ``_kernel`` runs inside the inner ``while_loop``, and
    ``_epilogue`` drains the final three tiles.  For fewer than four key blocks
    (``Tc <= 3``) it falls back to the sequential kernel.

    Matches ``scaled_dot_product_attention(query, key, value, scale=scale)``.
    """

    def __init__(
        self,
        *,
        scale: Optional[float] = None,
        accumulate_fp32: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.accumulate_fp32 = accumulate_fp32

    # -- per-stage helpers -----------------------------------------------------

    def _load_kv(self, key, value, Bc, idx, dims, static):
        # Fuse the K transpose into the DMA load: Kj is (1, 1, d, Bc).
        d = key.shape[-1]
        Kj = voyager.load_tile(
            key, idx, (1, 1, Bc, d), dims=dims, static_indices=static, transposed=True
        )
        Vj = voyager.load_tile(
            value, idx, (1, 1, Bc, d), dims=dims, static_indices=static
        )
        return Kj, Vj

    def _dma_load(self, b, h, j, key, value, Bc):
        # Dynamic key block (steady-state kernel): ``j`` is a loop-counter Node, so
        # the whole (b, h, j) index is dynamic.
        return self._load_kv(key, value, Bc, (b, h, j), (0, 1, 2), None)

    def _dma_load_const(self, b, h, j, key, value, Bc):
        # Constant key block (prologue): ``j`` is a Python int, so pin that dim via
        # static_indices and keep the dynamic index all loop counters (b, h).  A
        # mixed [b, h, <int>] index list could not be serialized by the codegen,
        # and a SymInt loop counter is indistinguishable from an int by type — so
        # the constant-vs-dynamic split lives at the call site, not a type check.
        return self._load_kv(key, value, Bc, (b, h), (0, 1), (0, 0, j, 0))

    def _qk_matmul(self, Qi, Kj):
        return torch.matmul(Qi, Kj)  # Kj already transposed to (1, 1, d, Bc)

    def _online_softmax(self, O_i, l_i, m_i, scores):
        d = O_i.shape[-1]
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(d)
        scores = scores * scale
        if self.accumulate_fp32 and scores.dtype != torch.float32:
            scores = scores.to(torch.float32)

        m_block = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))
        alpha = torch.exp(m_i - m_block)
        P = torch.exp(scores - m_block)
        l_i = alpha * l_i + P.sum(dim=-1, keepdim=True)
        O_i = alpha * O_i
        return P, O_i, l_i, m_block

    def _pv_matmul(self, O_i, P, Vj):
        return O_i + torch.matmul(P, Vj.to(P.dtype))

    # -- pipeline stages -------------------------------------------------------

    def _prologue(self, b, h, Qi, key, value, Bc, O_i, l_i, m_i):
        # Fill the 3-deep pipeline with key tiles 0, 1, 2 (constant blocks).
        K0, V0 = self._dma_load_const(b, h, 0, key, value, Bc)
        scores0 = self._qk_matmul(Qi, K0)
        K1, V1 = self._dma_load_const(b, h, 1, key, value, Bc)
        P0, O_i, l_i, m_i = self._online_softmax(O_i, l_i, m_i, scores0)
        scores1 = self._qk_matmul(Qi, K1)
        K2, V2 = self._dma_load_const(b, h, 2, key, value, Bc)
        # State: accumulators + tile0 (PV-ready) + tile1 (softmax-ready) +
        # tile2 (QK-ready).
        return (O_i, l_i, m_i, P0, V0, scores1, V1, K2, V2)

    def _kernel(self, b, h, j, state, Qi, key, value, Bc):
        O_i, l_i, m_i, P0, V0, scores1, V1, K2, V2 = state
        O_i = self._pv_matmul(O_i, P0, V0)                      # drain tile (PV)
        P1, O_i, l_i, m_i = self._online_softmax(O_i, l_i, m_i, scores1)
        K3, V3 = self._dma_load(b, h, j, key, value, Bc)        # prefetch tile j
        scores2 = self._qk_matmul(Qi, K2)                       # QK next tile
        # Clone the values shifted to a different pipeline slot so the while_loop
        # body does not alias its carried inputs.
        return (O_i, l_i, m_i, P1, V1.clone(), scores2, V2.clone(), K3, V3)

    def _epilogue(self, state, Qi):
        O_i, l_i, m_i, P0, V0, scores1, V1, K2, V2 = state
        O_i = self._pv_matmul(O_i, P0, V0)
        P1, O_i, l_i, m_i = self._online_softmax(O_i, l_i, m_i, scores1)
        scores2 = self._qk_matmul(Qi, K2)
        O_i = self._pv_matmul(O_i, P1, V1)
        P2, O_i, l_i, m_i = self._online_softmax(O_i, l_i, m_i, scores2)
        O_i = self._pv_matmul(O_i, P2, V2)
        return O_i, l_i, m_i

    # -- outer-body variants ---------------------------------------------------

    def _outer_loop_body_pipelined(self, b, h, i, query, key, value, Br, Bc):
        d = query.shape[-1]
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype

        Qi = voyager.load_tile(query, (b, h, i), (1, 1, Br, d), dims=(0, 1, 2))
        O_i = voyager.zero_tile((1, 1, Br, d), acc_dtype)
        l_i = voyager.zero_tile((1, 1, Br, 1), acc_dtype)
        m_i = voyager.zero_tile((1, 1, Br, 1), acc_dtype) + float("-inf")

        state = self._prologue(b, h, Qi, key, value, Bc, O_i, l_i, m_i)

        def cond_fn(j, *st):
            return j < key.shape[2] // Bc

        def body_fn(j, *st):
            new = self._kernel(b, h, j, st, Qi, key, value, Bc)
            return (j + 1, *new)

        # Steady state starts at tile 3 (prologue consumed 0/1/2).
        result = while_loop(cond_fn, body_fn, (3, *state))
        O_i, l_i, m_i = self._epilogue(result[1:], Qi)

        out = O_i / l_i
        if self.accumulate_fp32 and out.dtype != query.dtype:
            out = out.to(query.dtype)
        return out

    def _outer_loop_body(self, b, h, i, query, key, value, Br, Bc):
        # Sequential fallback (used when there are <= 3 key blocks).
        d = query.shape[-1]
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype

        Qi = voyager.load_tile(query, (b, h, i), (1, 1, Br, d), dims=(0, 1, 2))
        O_i = voyager.zero_tile((1, 1, Br, d), acc_dtype)
        l_i = voyager.zero_tile((1, 1, Br, 1), acc_dtype)
        m_i = voyager.zero_tile((1, 1, Br, 1), acc_dtype) + float("-inf")

        def cond_fn(j, O_buf, l_buf, m_buf):
            return j < key.shape[2] // Bc

        def body_fn(j, O_buf, l_buf, m_buf):
            scale = self.scale if self.scale is not None else 1.0 / math.sqrt(d)
            Kj = voyager.load_tile(
                key, (b, h, j), (1, 1, Bc, d), dims=(0, 1, 2)
            )
            Vj = voyager.load_tile(
                value, (b, h, j), (1, 1, Bc, d), dims=(0, 1, 2)
            )
            scores = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale
            if self.accumulate_fp32 and scores.dtype != torch.float32:
                scores = scores.to(torch.float32)
            m_new = torch.maximum(m_buf, scores.amax(dim=-1, keepdim=True))
            alpha = torch.exp(m_buf - m_new)
            P = torch.exp(scores - m_new)
            l_new = alpha * l_buf + P.sum(dim=-1, keepdim=True)
            O_new = alpha * O_buf + torch.matmul(P, Vj.to(P.dtype))
            return (j + 1, O_new, l_new, m_new)

        _, O_i, l_i, m_i = while_loop(cond_fn, body_fn, (0, O_i, l_i, m_i))

        out = O_i / l_i
        if self.accumulate_fp32 and out.dtype != query.dtype:
            out = out.to(query.dtype)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int = 64,
        Bc: int = 128,
    ):
        B, H, N, d = query.shape
        M = key.shape[2]
        Tr = N // Br
        Tc = M // Bc

        # The pipeline needs four key blocks to fill/drain; otherwise sequential.
        outer_body = (
            self._outer_loop_body_pipelined if Tc > 3 else self._outer_loop_body
        )

        O = voyager.alloc((B, H, N, d), query.dtype)

        def cond_fn(b, h, i, O_buf):
            return b < query.shape[0]

        def body_fn(b, h, i, O_buf):
            O_tile = outer_body(b, h, i, query, key, value, Br, Bc)
            O_buf = voyager.store_tile(
                O_tile, O_buf, (b, h, i), (1, 1, Br, d), dims=(0, 1, 2)
            )
            # Advance the (b, h, i) grid (one fused increment op).
            b_next, h_next, i_next = voyager.increment_indices(
                (b, h, i), (B, H, Tr)
            )
            return (b_next, h_next, i_next, O_buf)

        _, _, _, final_O = while_loop(cond_fn, body_fn, (0, 0, 0, O))
        return final_O


def build_attention_buffers(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tile_sizes: List[int],
    *,
    scale: Optional[float] = None,
    accumulate_fp32: bool = True,
    pipelined: bool = False,
) -> torch.fx.GraphModule:
    """
    Build the bufferized FX graph (while_loop nest over voyager.* primitives) for
    flash attention.  ``tile_sizes`` is ``[Br, Bc]`` (query-row / key-col blocks).
    ``pipelined`` selects the software-pipelined kernel.
    """
    Br, Bc = tile_sizes
    cls = FlashAttentionPipelined if pipelined else TiledAttention
    pattern = cls(scale=scale, accumulate_fp32=accumulate_fp32)
    with _lenient_verifier():
        gm = export_model(pattern, (query, key, value, Br, Bc))
    gm = _finalize_exported_gm(gm)

    B, H, N, d = query.shape
    Tc = key.shape[2] // Bc
    # Pipelined steady state runs key blocks 3..Tc; sequential runs 0..Tc.
    inner = [(3, Tc, 1)] if (pipelined and Tc > 3) else [Tc]
    _tag_loop_extents(
        gm,
        [
            [B, H, N // Br],  # outer (b, h, query-block) grid
            inner,            # inner key-block reduction
        ],
    )
    return gm
