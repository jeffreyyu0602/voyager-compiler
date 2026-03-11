import itertools
import math
from typing import List, Optional, Tuple

import torch
from torch._higher_order_ops import while_loop

import voyager_compiler.decomposed
from voyager_compiler.quantize_pt2e import export_model
from voyager_compiler.codegen.lowering.ir import Module, NameGenerator
from voyager_compiler.codegen.lowering.lowering_utils import _lower_nested_loops, _prepare_ir_graph


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

        is_mx_op = (
            query_scale is not None
            and key_scale is not None
            and value_scale is not None
        )

        if is_mx_op:
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

        if is_mx_op:
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
            attention_probs = torch.matmul(P, Vj.to(P.dtype))

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

        additional_inputs = (Qi, key, value, Br, Bc)
        additional_kwargs = {**kwargs, "query_scale": Qi_scale}

        def cond_fn(b, h, i, j, *args):
            N = additional_inputs[1].shape[2]
            Bc = additional_inputs[-1]
            return j < N // Bc

        def body_fn(b, h, i, j, *args):
            next_state = self._inner_loop_body(
                [b, h, i, j], *args, *additional_inputs, **additional_kwargs
            )
            return (b.clone(), h.clone(), i.clone(), j + 1) + next_state

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
        is_mx_op: bool = False,
        Br: int = None,
        Bc: int = None,
    ):
        d = O_i.shape[-1]
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
        O_i = alpha * O_i

        if not is_mx_op:
            return None, P, O_i, l_i, m_block

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
            attention_probs = torch.matmul(attention_probs, value.to(attention_probs.dtype))

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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
    ):
        is_mx_op = query_scale is not None

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

        scale, P, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores, is_mx_op, Br, Bc
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
        additional_kwargs = {**kwargs, "query_scale": Qi_scale}

        if self.flatten_loops:
            for j in range(N // Bc):
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
                N = additional_inputs[1].shape[2]
                Bc = additional_inputs[-1]
                return j < N // Bc

            def body_fn(b, h, i, j, *args):
                next_start = self._inner_loop_body_sequential(
                    [b, h, i, j], *args, *additional_inputs, **additional_kwargs
                )
                return (b.clone(), h.clone(), i.clone(), j + 1) + next_start

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
        is_mx_op = query_scale is not None
        b, h, i = indices

        # When b/h are tensors (inside while_loop), j must also be a tensor to
        # avoid mixed int/tensor index lists in the dynamic load_tile path.
        if torch.is_tensor(b):
            j0 = torch.zeros_like(b)
            j1 = j0 + 1
            j2 = j0 + 2
        else:
            j0, j1, j2 = 0, 1, 2

        # ---------------------------------------------------------
        # Time Step 0: Fill Pipeline Stage 1
        # ---------------------------------------------------------

        # Stage 1: Fetch Tile 0 from global memory
        K0, V0, K0_scale, V0_scale = self._dma_load(
            [b, h, i, j0], key, value, Bc, key_scale, value_scale
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
            [b, h, i, j1], key, value, Bc, key_scale, value_scale
        )

        # ---------------------------------------------------------
        # Time Step 2: Fill Pipeline Stages 1, 2, & 3
        # ---------------------------------------------------------

        # Stage 3: Online Softmax for Tile 0
        scale0, attention_probs0, O_i, l_i, m_i = self._online_softmax(
            O_i, l_i, m_i, scores0, is_mx_op=is_mx_op, Br=Br, Bc=Bc
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
            [b, h, i, j2], key, value, Bc, key_scale, value_scale
        )

        # Pipeline is now primed.
        # State 0 represents Tile 0 (ready for PV Matmul)
        # State 1 represents Tile 1 (ready for Softmax)
        # State 2 represents Tile 2 (ready for QK Matmul)
        if not is_mx_op:
            return (
                (O_i, l_i, m_i),
                (attention_probs0, V0),
                (scores1, V1),
                (K2, V2),
            )

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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        is_mx_op = query_scale is not None
        if not is_mx_op:
            flat_states = list(itertools.chain.from_iterable(states))
            O_i, l_i, m_i, attention_probs0, V0, scores1, V1, K2, V2 = flat_states
            scale0, V0_scale, V1_scale, K2_scale, V2_scale = [None] * 5
        else:
            O_i, l_i, m_i = states[0]
            scale0, attention_probs0, V0, V0_scale = states[1]
            scores1, V1, V1_scale = states[2]
            K2, V2, K2_scale, V2_scale = states[3]

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
            O_i, l_i, m_i, scores1, is_mx_op=is_mx_op, Br=Br, Bc=Bc
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

        if not is_mx_op:
            return (
                (O_i, l_i, m_i),
                (attention_probs1, V1.clone()),
                (scores2, V2.clone()),
                (K3, V3),
            )

        return (
            (O_i, l_i, m_i),
            (scale1, attention_probs1, V1.clone(), V1_scale.clone()),
            (scores2, V2.clone(), V2_scale.clone()),
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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        is_mx_op = query_scale is not None
        if not is_mx_op:
            flat_states = list(itertools.chain.from_iterable(states))
            O_i, l_i, m_i, attention_probs0, V0, scores1, V1, K2, V2 = flat_states
            scale0, V0_scale, V1_scale, K2_scale, V2_scale = [None] * 5
        else:
            O_i, l_i, m_i = states[0]
            scale0, attention_probs0, V0, V0_scale = states[1]
            scores1, V1, V1_scale = states[2]
            K2, V2, K2_scale, V2_scale = states[3]

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
            O_i, l_i, m_i, scores1, is_mx_op=is_mx_op, Br=Br, Bc=Bc
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
            O_i, l_i, m_i, scores2, is_mx_op=is_mx_op, Br=Br, Bc=Bc
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
        additional_kwargs = {**kwargs, "query_scale": Qi_scale}

        state = self._prologue(
            [b, h, i],
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
                N = additional_inputs[1].shape[2]
                Bc = additional_inputs[-1]
                return j < N // Bc

            def body_fn(b, h, i, j, loop_state):
                next_state = self._kernel(
                    [b, h, i, j],
                    loop_state,
                    *additional_inputs,
                    **additional_kwargs,
                )
                return b.clone(), h.clone(), i.clone(), j + 1, next_state

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
                indices, [B + 1, H, Tr, 1], [0, 0, 0, 3 if Tc > 3 else 0]
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

    namer = NameGenerator()
    env = {}
    placeholders, parameters, index_values, starts = _prepare_ir_graph(
        gm, (query, key, value), namer, env, kwargs=kwargs
    )

    B, H, N, _ = query.shape
    num_tiles = (B, H, N // Br, N // Bc)
    # Pad starts on the left with 0 for any dims whose lifted_tensor was not
    # found (e.g. FlashAttention where all indices start at 0).
    # FlashAttentionPipelined sets j_start = 3 when Tc > 3.
    starts = [0] * (len(num_tiles) - len(starts)) + starts
    loop_bounds = (
        tuple((s, n) for s, n in zip(starts[:3], num_tiles[:3])),  # outer: b, h, i
        ((starts[3], num_tiles[3]),),                               # inner: j
    )

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
        results=outputs,
    )
    return module, gm
