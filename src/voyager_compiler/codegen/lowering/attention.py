import itertools
import math
import operator
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.fx.node import map_arg
from torch.utils._pytree import tree_flatten
from torch._higher_order_ops import while_loop
from google.protobuf import text_format

import voyager_compiler.decomposed
from voyager_compiler.quantize_pt2e import export_model
from voyager_compiler.codegen import ShapeProp, is_nop
from voyager_compiler.codegen.lowering.ir import (
    IndexValue,
    Module,
    Loops,
    NameGenerator,
    Operation,
    IRNode,
    TensorBox,
    _propagate_dtype,
)
from voyager_compiler.codegen.lowering.allocator import run_memory_pass
from voyager_compiler.codegen.lowering.codegen import generate_proto


aten = torch.ops.aten
quantized_ops = torch.ops.quantized_ops


def _create_causal_mask(i: int, j: int, Br: int, Bc: int, device=None):
    """
    Returns a boolean mask of shape [Bm, Bn] where True means "allowed".
    """
    i = torch.arange(Br, device=device).unsqueeze(1) + i  # [Br, 1]
    j = torch.arange(Bc, device=device).unsqueeze(0) + j  # [1, Bc]
    return (j <= i)  # [Br, Bc]


class FlashAttention(torch.nn.Module):
    """
    torch.export friendly Flash Attention implementation for compiler lowering.

    The auxiliary quantization tensors (qmap/codebooks/etc.) are passed in as
    arguments and registered as buffers for quantizing attention probs on the fly.
    QKV scales and codebooks are are passed as forward args during runtime.
    """

    def __init__(
        self,
        *,
        qmap: Optional[torch.Tensor] = None,
        axes: Optional[List[int]] = None,
        block_size: int = None,
        quant_max: Optional[float] = None,
        force_scale_power_of_two: bool = False,
        scale_qmap: Optional[torch.Tensor] = None,
        output_code: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        accumulate_fp32: bool = False,
    ):
        super().__init__()
        self.register_buffer("qmap", qmap)
        self.register_buffer("input_code", input_code)
        self.register_buffer("scale_qmap", scale_qmap)
        self.register_buffer("output_code", output_code)

        self.axes = axes
        self.block_size = block_size
        self.quant_max = quant_max
        self.force_scale_power_of_two = force_scale_power_of_two
        self.accumulate_fp32 = accumulate_fp32

    def _inner_loop_body(
        self,
        indices,
        O_i: torch.Tensor,
        l_i: torch.Tensor,
        m_i: torch.Tensor,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        d = key.shape[-1]
        b, h, i, j = indices

        Kj = quantized_ops.load_tile(key, [b, h, j], [1, 1, Bc, d], [0, 1, 2])
        Vj = quantized_ops.load_tile(value, [b, h, j], [1, 1, Bc, d], [0, 1, 2])

        use_quant = (
            query_scale is not None
            and key_scale is not None
            and value_scale is not None
        )

        if use_quant:
            # Load per-block scales for K and V
            Kj_scale = quantized_ops.load_tile(
                key_scale,
                [b, h, j],
                [1, 1, Bc, d // self.block_size],
                [0, 1, 2],
            )
            Vj_scale = quantized_ops.load_tile(
                value_scale,
                [b, h, j],
                [1, 1, Bc // self.block_size, d],
                [0, 1, 2],
            )

            scores = quantized_ops.matmul_mx(
                Qi,
                Kj.transpose(-1, -2),
                input_scale=query_scale,
                weight_scale=Kj_scale.transpose(-1, -2),
                block_size=self.block_size,
                input_code=query_code,
                weight_code=key_code,
            )
        else:
            scores = torch.matmul(Qi, Kj.transpose(-1, -2))

        scores = scores * (1 / math.sqrt(d))
        # mask = _create_causal_mask(i * Br, j * Bc, Br, Bc, device=key.device)
        # scores = scores.masked_fill(~mask, -float("inf"))

        # Numerically stable accumulation
        if self.accumulate_fp32 and scores.dtype != torch.float32:
            scores = scores.to(torch.float32)

        m_block = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))

        alpha = torch.exp(m_i - m_block)
        P = torch.exp(scores - m_block)
        l_i = alpha * l_i + P.sum(dim=-1, keepdim=True)

        if use_quant:
            P_scale, P_q = quantized_ops.quantize_mx(
                P,
                qmap=self.qmap,
                axes=self.axes,
                block_size=self.block_size,
                quant_max=self.quant_max,
                force_scale_power_of_two=self.force_scale_power_of_two,
                scale_qmap=self.scale_qmap,
                output_code=self.output_code,
            )

            attention_probs = quantized_ops.matmul_mx(
                P_q,
                Vj,
                input_scale=P_scale,
                weight_scale=Vj_scale,
                block_size=self.block_size,
                input_code=self.input_code,
                weight_code=value_code,
            )
        else:
            attention_probs = torch.matmul(P, Vj)

        O_i = alpha * O_i + attention_probs

        # Perform a nop to copy new maximum to the correct memory location
        m_i = m_block.add(0)

        return O_i, l_i, m_i

    def _outer_loop_body(
        self,
        indices,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        **kwargs,
    ):
        B, H, N, d = query.shape
        b, h, i, init_j = indices

        Qi = quantized_ops.load_tile(query, [b, h, i], [1, 1, Br, d], [0, 1, 2])

        Qi_scale = None
        if (query_scale := kwargs.get("query_scale")) is not None:
            Qi_scale = quantized_ops.load_tile(
                query_scale,
                [b, h, i],
                [1, 1, Br, d // self.block_size],
                [0, 1, 2],
            )

        # Accumulator dtype choice
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype
        O_i = torch.zeros((1, 1, Br, d), device=query.device, dtype=acc_dtype)
        l_i = torch.zeros((1, 1, Br, 1), device=query.device, dtype=acc_dtype)
        m_i = torch.full((1, 1, Br, 1), -float("inf"), device=query.device, dtype=acc_dtype)

        def cond_fn(b, h, i, j, *args):
            return j < N // Bc

        def body_fn(b, h, i, j, *args):
            additional_inputs = (Qi, key, value, Br, Bc)
            additional_kwargs = {
                **kwargs,
                "query_scale": Qi_scale,
            }
            next_state = self._inner_loop_body(
                [b, h, i, j], *args, *additional_inputs, **additional_kwargs
            )
            return (b, h, i, j + 1) + next_state

        *_, O_i, l_i, m_i = while_loop(
            cond_fn,
            body_fn,
            (b, h, i, init_j, O_i, l_i, m_i),
        )

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
        **kwargs,
    ):
        B, H, N, d = query.shape
        Tr = N // Br

        # Allocate output buffer
        O = torch.empty((B, H, N, d), device=query.device, dtype=query.dtype)

        def cond_fn(b, h, i, j, O_buf):
            return b < B

        def body_fn(b, h, i, j, O_buf):
            indices = [b, h, i, j]
            O_tile = self._outer_loop_body(
                indices,
                query=query,
                key=key,
                value=value,
                Br=Br,
                Bc=Bc,
                **kwargs,
            )
            O_buf = quantized_ops.store_tile(
                O_tile, O_buf, [b, h, i], [1, 1, Br, d], [0, 1, 2]
            )
            new_indices = quantized_ops.increment_indices(
                indices, [B + 1, H, Tr, 1]
            )
            return (*new_indices, O_buf)

        factory_kwargs = {"device": query.device, "dtype": torch.int32}
        initial_state = (
            torch.tensor(0, **factory_kwargs),  # b
            torch.tensor(0, **factory_kwargs),  # h
            torch.tensor(0, **factory_kwargs),  # i
            torch.tensor(0, **factory_kwargs),  # j
            O,
        )

        *_, final_O = while_loop(cond_fn, body_fn, initial_state)
        return final_O


class FlashAttentionPipelined(torch.nn.Module):
    """
    torch.export friendly Flash Attention V3 implementation for compiler lowering.
    Features a software-pipelined attention kernel triple buffering to hide
    Softmax latency.

    The auxiliary quantization tensors (qmap/codebooks/etc.) are passed in as
    arguments and registered as buffers for quantizing attention probs on the fly.
    QKV scales and codebooks are are passed as forward args during runtime.
    """

    def __init__(
        self,
        *,
        qmap: Optional[torch.Tensor] = None,
        axes: Optional[List[int]] = None,
        block_size: int = None,
        quant_max: Optional[float] = None,
        force_scale_power_of_two: bool = False,
        scale_qmap: Optional[torch.Tensor] = None,
        output_code: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        accumulate_fp32: bool = False,
        flatten_loops: bool = False,
    ):
        super().__init__()
        self.register_buffer("qmap", qmap)
        self.register_buffer("input_code", input_code)
        self.register_buffer("scale_qmap", scale_qmap)
        self.register_buffer("output_code", output_code)

        self.axes = axes
        self.block_size = block_size
        self.quant_max = quant_max
        self.force_scale_power_of_two = force_scale_power_of_two
        self.accumulate_fp32 = accumulate_fp32
        self.flatten_loops = flatten_loops

    def _load_tile(
        self,
        buffer: torch.Tensor,
        indices: List[int],
        tile_shape: List[int],
        transposed: bool = False
    ):
        if self.flatten_loops or all(isinstance(idx, int) for idx in indices):
            return quantized_ops.load_tile(
                buffer,
                [],
                tile_shape,
                [],
                static_indices=[*indices, 0],
                transposed=transposed,
            )

        return quantized_ops.load_tile(
            buffer,
            indices,
            tile_shape,
            [0, 1, 2],
            transposed=transposed
        )

    def _store_tile(
        self,
        tile: torch.Tensor,
        buffer: torch.Tensor,
        indices: List[int],
        tile_shape: List[int]
    ):
        if self.flatten_loops:
            return quantized_ops.store_tile(
                tile,
                buffer,
                [],
                tile_shape,
                [],
                static_indices=[*indices, 0]
            )

        return quantized_ops.store_tile(
            tile, buffer, indices, tile_shape, [0, 1, 2]
        )

    def _dma_load(
        self,
        indices,
        key: torch.Tensor,
        value: torch.Tensor,
        Bc: int,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
    ):
        d = key.shape[-1]
        b, h, i, j = indices

        # Fuse transpose op into DMA load
        Kj = self._load_tile(key, [b, h, j], [1, 1, Bc, d], True)
        Vj = self._load_tile(value, [b, h, j], [1, 1, Bc, d])

        if key_scale is None or value_scale is None:
            return Kj, Vj, None, None

        Kj_scale = self._load_tile(
            key_scale, [b, h, j], [1, 1, Bc, d // self.block_size], True
        )

        Vj_scale = self._load_tile(
            value_scale, [b, h, j], [1, 1, Bc // self.block_size, d]
        )

        return Kj, Vj, Kj_scale, Vj_scale

    def _qk_matmul(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
    ):
        if query_scale is not None and key_scale is not None:
            scores = quantized_ops.matmul_mx(
                query,
                key,
                input_scale=query_scale,
                weight_scale=key_scale,
                block_size=self.block_size,
                input_code=query_code,
                weight_code=key_code,
            )
        else:
            scores = torch.matmul(query, key)

        # TODO Perform FP8 quantization?

        return scores

    def _online_softmax(
        self,
        O_i: torch.Tensor,
        l_i: torch.Tensor,
        m_i: torch.Tensor,
        scores: torch.Tensor,
        scaling: float,
        use_quant: bool = False,
        Br: int = None,
        Bc: int = None,
    ):
        scores = scores * scaling
        # mask = _create_causal_mask(i * Br, j * Bc, Br, Bc, device=key.device)
        # scores = scores.masked_fill(~mask, -float("inf"))

        # Numerically stable accumulation
        if self.accumulate_fp32 and scores.dtype != torch.float32:
            scores = scores.to(torch.float32)

        m_block = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))

        alpha = torch.exp(m_i - m_block)
        P = torch.exp(scores - m_block)
        l_i = alpha * l_i + P.sum(dim=-1, keepdim=True)
        O_i = alpha * O_i

        if not use_quant:
            return None, P, O_i, l_i, m_i

        scale, output = quantized_ops.quantize_mx(
            P,
            qmap=self.qmap,
            axes=self.axes,
            block_size=self.block_size,
            quant_max=self.quant_max,
            force_scale_power_of_two=self.force_scale_power_of_two,
            scale_qmap=self.scale_qmap,
            output_code=self.output_code,
        )

        return scale, output, O_i, l_i, m_block

    def _pv_matmul(
        self,
        O_i: torch.Tensor,
        attention_probs: torch.Tensor,
        value: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
    ):
        if input_scale is not None and value_scale is not None:
            attention_probs = quantized_ops.matmul_mx(
                attention_probs,
                value,
                input_scale=input_scale,
                weight_scale=value_scale,
                block_size=self.block_size,
                input_code=self.input_code,
                weight_code=value_code,
            )
        else:
            attention_probs = torch.matmul(attention_probs, value)

        return O_i + attention_probs

    def _inner_loop_body_sequential(
        self,
        indices: List[int],
        O_i: torch.Tensor,
        l_i: torch.Tensor,
        m_i: torch.Tensor,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        scaling: float,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
    ):
        Kj, Vj, Kj_scale, Vj_scale = self._dma_load(
            indices, key, value, Bc, key_scale, value_scale
        )

        scores = self._qk_matmul(
            query=Qi,
            key=Kj,
            query_scale=query_scale,
            key_scale=Kj_scale,
            query_code=query_code,
            key_code=key_code,
        )

        use_quant = query_scale is not None
        scale, P, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores, scaling, use_quant, Br, Bc
        )

        O_i = self._pv_matmul(
            O_i,
            attention_probs=P,
            value=Vj,
            input_scale=scale,
            value_scale=Vj_scale,
            value_code=value_code
        )

        return O_i, l_i, m_i

    def _outer_loop_body(
        self,
        indices: List[int],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        **kwargs,
    ):
        B, H, N, d = query.shape
        b, h, i, init_j = indices

        # Load Query tile
        Qi = self._load_tile(query, [b, h, i], [1, 1, Br, d])

        Qi_scale = None
        if (query_scale := kwargs.get("query_scale")) is not None:
            Qi_scale = self._load_tile(
                query_scale, [b, h, i], [1, 1, Br, d // self.block_size]
            )

        # Initialize Accumulators
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype
        O_i = torch.zeros((1, 1, Br, d), device=query.device, dtype=acc_dtype)
        l_i = torch.zeros((1, 1, Br, 1), device=query.device, dtype=acc_dtype)
        m_i = torch.full((1, 1, Br, 1), -float("inf"), device=query.device, dtype=acc_dtype)

        additional_inputs = (Qi, key, value, Br, Bc)
        additional_kwargs = {
            **kwargs,
            "scaling": 1 / math.sqrt(d),
            "query_scale": Qi_scale,
        }

        Tc = N // Bc

        if self.flatten_loops:
            for j in range(Tc):
                O_i, l_i, m_i = self._inner_loop_body_sequential(
                    [b, h, i, j],
                    O_i,
                    l_i,
                    m_i,
                    *additional_inputs,
                    **additional_kwargs,
                )
        else:
            def cond_fn(b, h, i, j, *args):
                return j < Tc

            def body_fn(b, h, i, j, *args):
                next_start = self._inner_loop_body_sequential(
                    [b, h, i, j], *args, *additional_inputs, **additional_kwargs
                )
                return (b, h, i, j + 1) + next_start

            *_, O_i, l_i, m_i = while_loop(
                cond_fn,
                body_fn,
                (b, h, i, init_j, O_i, l_i, m_i),
            )

        # Normalize output
        out = O_i / l_i
        if self.accumulate_fp32 and out.dtype != query.dtype:
            out = out.to(query.dtype)

        return out

    def _prologue(
        self,
        O_i: torch.Tensor,
        l_i: torch.Tensor,
        m_i: torch.Tensor,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        scaling: float,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # ---------------------------------------------------------
        # Time Step 0: Fill Pipeline Stage 1
        # ---------------------------------------------------------

        # Stage 1: Fetch Tile 0 from global memory
        K0, V0, K0_scale, V0_scale = self._dma_load(
            [0, 0, 0, 0], key, value, Bc, key_scale, value_scale
        )

        # ---------------------------------------------------------
        # Time Step 1: Fill Pipeline Stages 1 & 2
        # ---------------------------------------------------------

        # Stage 2: Compute QK^T for Tile 0
        scores0 = self._qk_matmul(
            Qi,
            K0,
            query_scale=query_scale,
            key_scale=K0_scale,
            query_code=query_code,
            key_code=key_code,
        )

        # Tile 1 DMA load
        K1, V1, K1_scale, V1_scale = self._dma_load(
            [0, 0, 0, 1], key, value, Bc, key_scale, value_scale
        )

        # ---------------------------------------------------------
        # Time Step 2: Fill Pipeline Stages 1, 2, & 3
        # ---------------------------------------------------------

        # Stage 3: Online Softmax for Tile 0
        use_quant = query_scale is not None
        scale0, attention_probs0, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores0, scaling, use_quant=use_quant, Br=Br, Bc=Bc
        )

        # Tile 1 QK matmul
        scores1 = self._qk_matmul(
            Qi,
            K1,
            query_scale=query_scale,
            key_scale=K1_scale,
            query_code=query_code,
            key_code=key_code,
        )

        # Tile 2 DMA load
        K2, V2, K2_scale, V2_scale = self._dma_load(
            [0, 0, 0, 2], key, value, Bc, key_scale, value_scale
        )

        # Pipeline is now primed.
        # State 0 represents Tile 0 (ready for PV Matmul)
        # State 1 represents Tile 1 (ready for Softmax)
        # State 2 represents Tile 2 (ready for QK Matmul)
        return (
            (O_i, l_i, m_i),
            (scale0, attention_probs0, V0, V0_scale),
            (scores1, V1, V1_scale),
            (K2, V2, K2_scale, V2_scale),
        )

    def _kernel(
        self,
        indices,
        states: Tuple,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        scaling: float,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        O_i, l_i, m_i = states[0]
        scale0, attention_probs0, V0, V0_scale = states[1]
        scores1, V1, V1_scale = states[2]
        K2, V2, K2_scale, V2_scale = states[3]
        use_quant = query_scale is not None

        # Tile j PV matmul
        O_i = self._pv_matmul(
            O_i,
            attention_probs0,
            V0,
            input_scale=scale0,
            value_scale=V0_scale,
            value_code=value_code,
        )

        # Tile j + 1 Softmax
        scale1, attention_probs1, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores1, scaling, use_quant=use_quant, Br=Br, Bc=Bc
        )

        # Tile j + 3 DMA load
        K3, V3, K3_scale, V3_scale = self._dma_load(
            indices, key, value, Bc, key_scale, value_scale
        )

        # Tile j + 2 QK matmul
        scores2 = self._qk_matmul(
            Qi,
            K2,
            query_scale=query_scale,
            key_scale=K2_scale,
            query_code=query_code,
            key_code=key_code,
        )

        return (
            (O_i, l_i, m_i),
            (scale1, attention_probs1, V1, V1_scale),
            (scores2, V2, V2_scale),
            (K3, V3, K3_scale, V3_scale),
        )

    def _epilogue(
        self,
        states: Tuple,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        scaling: float,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        O_i, l_i, m_i = states[0]
        scale0, attention_probs0, V0, V0_scale = states[1]
        scores1, V1, V1_scale = states[2]
        K2, V2, K2_scale, V2_scale = states[3]
        use_quant = query_scale is not None

        # ---------------------------------------------------------
        # Epilogue Step 1: Drain state0, advance state1 & state2
        # ---------------------------------------------------------

        # Tile j + 1: PV matmul
        O_i = self._pv_matmul(
            O_i,
            attention_probs0,
            V0,
            input_scale=scale0,
            value_scale=V0_scale,
            value_code=value_code,
        )

        # Tile j + 2: Softmax
        scale1, attention_probs1, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores1, scaling, use_quant=use_quant, Br=Br, Bc=Bc
        )

        # Tile j + 3: QK matmul
        scores2 = self._qk_matmul(
            Qi,
            K2,
            query_scale=query_scale,
            key_scale=K2_scale,
            query_code=query_code,
            key_code=key_code,
        )

        # ---------------------------------------------------------
        # Epilogue Step 2: Drain the new state1, advance state2
        # ---------------------------------------------------------

        # Tile j + 2: PV matmul
        O_i = self._pv_matmul(
            O_i,
            attention_probs1,
            V1,
            input_scale=scale1,
            value_scale=V1_scale,
            value_code=value_code,
        )

        # Tile j + 3: Softmax
        scale2, attention_probs2, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores2, scaling, use_quant=use_quant, Br=Br, Bc=Bc
        )

        # ---------------------------------------------------------
        # Epilogue Step 3: Drain the final state2
        # ---------------------------------------------------------

        # Tile j + 3: PV matmul
        O_i = self._pv_matmul(
            O_i,
            attention_probs2,
            V2,
            input_scale=scale2,
            value_scale=V2_scale,
            value_code=value_code,
        )

        # The pipeline is now fully drained.
        return O_i, l_i, m_i

    def _outer_loop_body_pipelined(
        self,
        indices,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        **kwargs,
    ):
        B, H, N, d = query.shape
        b, h, i, init_j = indices

        Qi = self._load_tile(query, [b, h, i], [1, 1, Br, d])

        Qi_scale = None
        if (query_scale := kwargs.get("query_scale")) is not None:
            Qi_scale = self._load_tile(
                query_scale, [b, h, i], [1, 1, Br, d // self.block_size]
            )

        # Accumulator dtype choice
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype
        O_i = torch.zeros((1, 1, Br, d), device=query.device, dtype=acc_dtype)
        l_i = torch.zeros((1, 1, Br, 1), device=query.device, dtype=acc_dtype)
        m_i = torch.full((1, 1, Br, 1), -float("inf"), device=query.device, dtype=acc_dtype)

        additional_inputs = (Qi, key, value, Br, Bc)
        additional_kwargs = {
            **kwargs,
            "scaling": 1 / math.sqrt(d),
            "query_scale": Qi_scale,
        }

        state = self._prologue(
            O_i,
            l_i,
            m_i,
            *additional_inputs,
            **additional_kwargs,
        )

        if self.flatten_loops:
            for j in range(3, N // Bc):
                state = self._kernel(
                    [b, h, i, j],
                    state,
                    *additional_inputs,
                    **additional_kwargs,
                )
        else:
            def cond_fn(b, h, i, j, loop_state):
                return j < N // Bc

            def body_fn(b, h, i, j, loop_state):
                next_state = self._kernel(
                    [b, h, i, j],
                    loop_state,
                    *additional_inputs,
                    **additional_kwargs,
                )
                return b, h, i, j + 1, next_state

            *_, state = while_loop(cond_fn, body_fn, (b, h, i, init_j, state))

        O_i, l_i, m_i = self._epilogue(
            state,
            *additional_inputs,
            **additional_kwargs,
        )

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
        **kwargs,
    ):
        B, H, N, d = query.shape
        Tr = N // Br
        Tc = N // Bc

        outer_body_fn = (
            self._outer_loop_body_pipelined if Tc > 3 else self._outer_loop_body
        )

        # Allocate output buffer
        O = torch.empty((B, H, N, d), device=query.device, dtype=query.dtype)

        if self.flatten_loops:
            for b, h, i in itertools.product(range(B), range(H), range(Tr)):
                O_tile = outer_body_fn(
                    [b, h, i, 3],
                    query=query,
                    key=key,
                    value=value,
                    Br=Br,
                    Bc=Bc,
                    **kwargs,
                )
                self._store_tile(O_tile, O, [b, h, i], [1, 1, Br, d])

            return O

        def cond_fn(b, h, i, j, O_buf):
            return b < B

        def body_fn(b, h, i, j, O_buf):
            indices = [b, h, i, j]
            O_tile = outer_body_fn(
                indices,
                query=query,
                key=key,
                value=value,
                Br=Br,
                Bc=Bc,
                **kwargs,
            )
            O_buf = self._store_tile(
                O_tile, O_buf, [b, h, i], [1, 1, Br, d]
            )
            new_indices = quantized_ops.increment_indices(
                indices, [B + 1, H, Tr, 1]
            )
            return (*new_indices, O_buf)

        factory_kwargs = {"device": query.device, "dtype": torch.int32}
        initial_state = (
            torch.tensor(0, **factory_kwargs),  # b
            torch.tensor(0, **factory_kwargs),  # h
            torch.tensor(0, **factory_kwargs),  # i
            torch.tensor(3 if Tc > 3 else 0, **factory_kwargs),  # j
            O,
        )

        loop_result = while_loop(cond_fn, body_fn, initial_state)
        return loop_result[-1]


def _should_use_dram(node, named_modules):
    if node.target == quantized_ops.store_tile.default:
        return True

    for user in node.users:
        if user.target == quantized_ops.load_tile.default:
            return True
        elif user.target == quantized_ops.store_tile.default:
            if user.args[1] == node:
                return True
        elif user.target == torch.ops.higher_order.while_loop:
            loop_body = named_modules[user.args[1].target]
            loop_inputs = user.args[2] + user.args[3]

            placeholders = [
                n for n in loop_body.graph.nodes if n.op == "placeholder"
            ]
            assert len(placeholders) == len(loop_inputs)

            index = loop_inputs.index(node)

            if _should_use_dram(placeholders[index], named_modules):
                return True

    return False


def _lower_nested_loops(
    model: torch.fx.GraphModule,
    loop_bounds,
    env,
    namer,
    index_values=None,
    depth=0,
):
    named_modules = dict(model.named_modules())

    output_node = next(n for n in model.graph.nodes if n.op == "output")
    outputs_order = {n: i for i, n in enumerate(output_node.args[0])}

    num_indices = len(index_values) if index_values is not None else 0

    model.graph.print_tabular()

    def load_arg(a):
        return map_arg(a, lambda n: n.value)

    body: List[IRNode] = []

    for node in model.graph.nodes:
        if node.target == torch.ops.higher_order.while_loop:
            loop_body = named_modules[node.args[1].target]
            carried_inputs = node.args[2]
            loop_body_inputs = node.args[2] + node.args[3]

            example_inputs = load_arg(loop_body_inputs)
            ShapeProp(loop_body).propagate(*example_inputs)

            # Indices are created beforehand. Exclude them in init_args.
            current_init_args = [env[n] for n in carried_inputs[num_indices:]]

            placeholders = [
                n for n in loop_body.graph.nodes if n.op == "placeholder"
            ]
            assert len(placeholders) == len(loop_body_inputs)

            # Link iter_args to variables in the outer scope
            inner_env = env.copy()
            inner_iter_vars = []
            for idx, p in enumerate(placeholders):
                if idx < num_indices:
                    inner_env[p] = index_values[idx]
                elif idx < len(carried_inputs):
                    stmt = Operation.from_fx_node(p, inner_env, namer)
                    inner_iter_vars.append(inner_env[p])
                else:
                    inner_env[p] = env[loop_body_inputs[idx]]

            stmts, yields = _lower_nested_loops(
                loop_body,
                loop_bounds,
                env=inner_env,
                namer=namer,
                index_values=index_values,
                depth=depth + 1
            )

            loop_outputs = [
                TensorBox(
                    name=namer.new_tensor(),
                    shape=tuple(v.shape),
                    dtype=v.dtype,
                    space=y.space,
                    producer_op=y.producer_op,
                )
                for y, v in zip(yields, node.value[num_indices:])
            ]

            indices_count = sum(len(bounds) for bounds in loop_bounds[:depth])
            bounds = loop_bounds[depth]

            for i in reversed(range(len(bounds))):
                is_outermost = (i == 0)
                is_innermost = (i == len(bounds) - 1)

                stmts = Loops(
                    index=index_values[indices_count + i],
                    start=bounds[i][0],
                    end=bounds[i][1],
                    step=1,
                    body=stmts if isinstance(stmts, list) else [stmts],
                    init_args=current_init_args if is_outermost else inner_iter_vars,
                    iter_vars=inner_iter_vars,
                    yields=yields if is_innermost else inner_iter_vars,
                    outputs=loop_outputs if is_outermost else inner_iter_vars,
                )

            body.append(stmts)
            env[node] = index_values + loop_outputs
        elif node.op == "call_function":
            # Skip emitting code for loop indices increment
            index = outputs_order.get(node)
            if (
                depth > 0
                and index is not None
                and index < len(index_values)
                or node.target == quantized_ops.increment_indices.default
            ):
                continue

            if is_nop(node):
                env[node] = env[node.all_input_nodes[0]]
            elif node.target == operator.getitem:
                idx = node.args[1]
                env[node] = env[node.all_input_nodes[0]][idx]
            else:
                use_dram = _should_use_dram(node, named_modules)
                mem_loc = "DRAM" if use_dram else "Scratchpad"
                op = Operation.from_fx_node(node, env, namer, mem_space=mem_loc)
                body.append(op)

    outputs = []
    for i, n in enumerate(output_node.args[0]):
        if depth == 0 or i >= len(index_values):
            outputs.append(env[n])

    return body, outputs


def lower_flash_attention(pattern, query, key, value, Br=64, Bc=128, kwargs=None):
    gm = export_model(
        pattern,
        (query, key, value, Br, Bc),
        kwargs=kwargs,
    )

    # Remove Br and Bc from the graph since they're compile-time constants
    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()

    example_args = (query, key, value)
    flatten_args, spec = tree_flatten((example_args, kwargs))

    ShapeProp(gm).propagate(*flatten_args)

    namer = NameGenerator()
    env = {}
    placeholders = []
    parameters = []
    index_values = []
    start = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ir_node = Operation.from_fx_node(node, env, namer)
            placeholders.extend(ir_node.outputs)
        elif node.op == "get_attr":
            target = node.target
            if target.startswith("lifted_tensor"):
                ssa_index = IndexValue(name=namer.new_index(), expr=target)
                env[node] = ssa_index
                index_values.append(ssa_index)
                param = getattr(gm, target)
                start.append(param.item())
            elif isinstance(getattr(gm, target), torch.Tensor):
                ir_node = Operation.from_fx_node(node, env, namer)
                parameters.extend(ir_node.outputs)

    B, H, N, d = query.shape
    loop_bounds = [B, H, N // Br, N // Bc]
    start = [0] * (len(loop_bounds) - len(start)) + start
    loop_bounds = list(zip(start, loop_bounds))
    loop_bounds = (tuple(loop_bounds[0:3]), (loop_bounds[3],))

    print("Module inputs:")
    for k, v in env.items():
        print(f"{k} -> {v}")

    body, outputs = _lower_nested_loops(
        gm,
        loop_bounds,
        env=env,
        namer=namer,
        index_values=index_values,
    )

    module = Module(
        "main",
        args=placeholders,
        params=parameters,
        body=body,
        results=outputs
    )
    print("\nFinal IR:")
    print(module.format())

    return module, gm


if __name__ == "__main__":
    """
    Usage:
    python voyager-compiler/src/voyager_compiler/codegen/lowering/attention.py \
        --output_dir test/compiler/networks/flash_attention/MXNF4
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="A script to process data and save results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory where the output files will be saved",
        default="test/compiler/networks/flash_attention"
    )
    parser.add_argument(
        "--Br", type=int, default=512, help="Row tile size for Flash Attention"
    )
    parser.add_argument(
        "--Bc", type=int, default=128, help="Column tile size for Flash Attention"
    )
    parser.add_argument(
        "--accumulate_fp32",
        action="store_true",
        help="Whether to perform accumulation in fp32 for improved numerical stability"
    )
    parser.add_argument(
        "--flatten_loops",
        action="store_true",
        help="Whether to flatten the outer loops into a single loop with dynamic bounds"
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    B, H, N, D = 1, 32, 1024, 64
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    flash_attention_pattern = FlashAttention()

    out_ref = F.scaled_dot_product_attention(q, k, v)
    out_gold = flash_attention_pattern(q, k, v, Br=args.Br, Bc=args.Bc)

    print(out_ref)
    print(out_gold)

    # Compare in fp32 for a fair error metric
    max_err = (out_gold.float() - out_ref.float()).abs().max().item()
    rms_err = torch.mean((out_gold.float() - out_ref.float()) ** 2).sqrt().item()
    print(f"max_err={max_err:.6g}  rms_err={rms_err:.6g}\n\n")

    example_inputs = (q.cpu(), k.cpu(), v.cpu(), args.Br, args.Bc)
    module, gm = lower_flash_attention(flash_attention_pattern, *example_inputs)
    run_memory_pass(module)

    os.makedirs(args.output_dir, exist_ok=True)

    proto = generate_proto(module, gm, example_inputs)
    with open(os.path.join(args.output_dir, 'fp16_model.txt'), "w") as f:
        f.write(text_format.MessageToString(proto))


    from voyager_compiler.fake_quantize import get_quantization_map

    nf4_qmap, nf4_code = get_quantization_map("nf4_6", device=device)
    midpoints = (nf4_code[:-1] + nf4_code[1:]) / 2

    scale_qmap = get_quantization_map("fp8_e5m3", device=device)

    quant_kwargs = {
        "qmap": nf4_qmap,
        "axes": [-1],
        "block_size": 64,
        "quant_max": 31,
        "force_scale_power_of_two": False,
        "scale_qmap": scale_qmap,
        "output_code": midpoints,
    }

    qs, qq = quantized_ops.quantize_mx(q, **quant_kwargs)
    ks, kq = quantized_ops.quantize_mx(k, **quant_kwargs)
    vs, vq = quantized_ops.quantize_mx(v, **quant_kwargs)

    flash_attention_quantized_pattern = FlashAttentionPipelined(
        **quant_kwargs,
        input_code=nf4_code.clone(),
        accumulate_fp32=args.accumulate_fp32,
        flatten_loops=args.flatten_loops,
    )

    example_inputs = (qq, kq, vq, args.Br, args.Bc)
    example_kwargs = {
        "query_scale": qs,
        "key_scale": ks,
        "value_scale": vs,
        "query_code": nf4_code.clone(),
        "key_code": nf4_code.clone(),
        "value_code": nf4_code.clone(),
    }
    input_dtypes = [
        "int4", "int4", "int4", "fp8_e5m3", "fp8_e5m3", "fp8_e5m3",
        "int6", "int6", "int6", "int4", "fp8_e5m3", None, None
    ]

    out_gold = flash_attention_quantized_pattern(
        *example_inputs, **example_kwargs
    )

    print(out_ref)
    print(out_gold)

    # Compare in fp32 for a fair error metric
    max_err = (out_gold.float() - out_ref.float()).abs().max().item()
    rms_err = torch.mean((out_gold.float() - out_ref.float()) ** 2).sqrt().item()
    print(f"max_err={max_err:.6g}  rms_err={rms_err:.6g}\n\n")

    # Move inputs to CPU for export
    example_inputs = tuple(
        x.cpu() if isinstance(x, torch.Tensor) else x for x in example_inputs
    )
    example_kwargs = {k: v.cpu() for k, v in example_kwargs.items()}

    module, gm = lower_flash_attention(
        flash_attention_quantized_pattern.cpu(),
        *example_inputs,
        kwargs=example_kwargs,
    )
    _propagate_dtype(module, input_dtypes=input_dtypes)
    run_memory_pass(module)

    flatten_args, spec = tree_flatten((example_inputs[:3], example_kwargs))
    output_dir = os.path.join(args.output_dir, "tensor_files")
    proto = generate_proto(module, gm, flatten_args, output_dir=output_dir)

    with open(os.path.join(args.output_dir, 'model.txt'), "w") as f:
        f.write(text_format.MessageToString(proto))

    operations = [
        op.op.name if op.WhichOneof('op_type') == 'op' else op.fused_op.name
        for op in proto.ops if op.op.op != 'nop'
    ]

    with open(os.path.join(args.output_dir, 'layers.txt'), 'w') as f:
        f.write('\n'.join(operations))
