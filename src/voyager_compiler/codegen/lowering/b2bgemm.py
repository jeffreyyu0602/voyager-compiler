from typing import Optional, List

import torch
from torch._higher_order_ops.while_loop import while_loop

import voyager_compiler.decomposed
from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.ir import Module, NameGenerator
from voyager_compiler.codegen.lowering.lowering_utils import _lower_nested_loops, _prepare_ir_graph

quantized_ops = torch.ops.quantized_ops


def _load_tile(
    buffer: torch.Tensor,
    indices: List[int],
    tile_shape: List[int],
    dims: List[int],
    transposed: bool = False,
    flatten_loops: bool = False,
):
    if flatten_loops or all(isinstance(idx, int) for idx in indices):
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
        dims,
        transposed=transposed,
    )


class TiledGEMM(torch.nn.Module):
    """
    torch.export-friendly tiled GEMM implementation for compiler lowering.

    Tiles over (B, X, K) outer grid and C reduction dimension, with optional
    FP32 accumulation, block-wise MX quantization, batched weight (for BMM),
    and weight transpose on load (for matmul-convention ops).

    Weight layout conventions
    -------------------------
    batched_weight=False, transpose_weight=False (default, linear-style):
        weight shape: (K, C)   op: linear(input_tile, weight_tile) = input @ weight.T
    batched_weight=False, transpose_weight=True  (matmul-style):
        weight shape: (K, C)   tile loaded as (C, K) via .mT
                               op: matmul(input_tile, weight_tile) = input @ weight.T
    batched_weight=True,  transpose_weight=True  (BMM-style):
        weight shape: (B, K, C) tile loaded as (B, C, K) via .mT
                               op: matmul(input_tile, weight_tile) = batched input @ weight.T
    """

    def __init__(
        self,
        target: torch._ops.OpOverload,
        *,
        block_size: int = None,
        accumulate_fp32: bool = False,
        flatten_loops: bool = False,
        batched_weight: bool = False,
        transpose_weight: bool = False,
    ):
        super().__init__()
        self.target = target
        self.block_size = block_size
        self.accumulate_fp32 = accumulate_fp32
        self.flatten_loops = flatten_loops
        self.batched_weight = batched_weight
        self.transpose_weight = transpose_weight

    def _reduction_loop_body(
        self,
        indices: List[torch.Tensor],
        tile_sizes: List[int],
        psum: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        b, x, k, c = indices
        tile_b, tile_x, tile_k, tile_c = tile_sizes

        input_tile = _load_tile(
            input,
            [b, x, c],
            [tile_b, tile_x, tile_c],
            [0, 1, 2],
            flatten_loops=self.flatten_loops,
        )

        if self.batched_weight:
            weight_tile = _load_tile(
                weight,
                [b, k, c],
                [tile_b, tile_k, tile_c],
                [0, 1, 2],
                transposed=self.transpose_weight,
                flatten_loops=self.flatten_loops,
            )
        else:
            weight_tile = _load_tile(
                weight,
                [k, c],
                [tile_k, tile_c],
                [0, 1],
                transposed=self.transpose_weight,
                flatten_loops=self.flatten_loops,
            )

        args = (input_tile, weight_tile)

        if input_scale is not None and weight_scale is not None:
            input_scale_tile = _load_tile(
                input_scale,
                [b, x, c],
                [tile_b, tile_x, tile_c // self.block_size],
                [0, 1, 2],
                flatten_loops=self.flatten_loops,
            )

            if self.batched_weight:
                weight_scale_tile = _load_tile(
                    weight_scale,
                    [b, k, c],
                    [tile_b, tile_k, tile_c // self.block_size],
                    [0, 1, 2],
                    transposed=self.transpose_weight,
                    flatten_loops=self.flatten_loops,
                )
            else:
                weight_scale_tile = _load_tile(
                    weight_scale,
                    [k, c],
                    [tile_k, tile_c // self.block_size],
                    [0, 1],
                    transposed=self.transpose_weight,
                    flatten_loops=self.flatten_loops,
                )

            kwargs = {
                "input_scale": input_scale_tile,
                "weight_scale": weight_scale_tile,
                "block_size": self.block_size,
                "input_code": input_code,
                "weight_code": weight_code,
            }

            out_tile = self.target(*args, **kwargs)
        else:
            out_tile = self.target(*args)

        if self.accumulate_fp32 and out_tile.dtype != torch.float32:
            out_tile = out_tile.to(torch.float32)

        return psum + out_tile

    def _outer_loop_body(
        self,
        indices: List[torch.Tensor],
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        tile_b: int,
        tile_x: int,
        tile_c: int,
        tile_k: int,
        **kwargs,
    ):
        b, x, k, init_c = indices

        acc_dtype = torch.float32 if self.accumulate_fp32 else input.dtype
        psum = torch.zeros((tile_b, tile_x, tile_k), device=input.device, dtype=acc_dtype)

        def cond_fn(b_idx, x_idx, k_idx, c_idx, psum_buf):
            return c_idx < input.shape[-1] // tile_c

        def body_fn(b_idx, x_idx, k_idx, c_idx, psum_buf):
            next_psum = self._reduction_loop_body(
                [b_idx, x_idx, k_idx, c_idx],
                [tile_b, tile_x, tile_k, tile_c],
                psum_buf,
                input,
                weight,
                **kwargs,
            )
            return (b_idx.clone(), x_idx.clone(), k_idx.clone(), c_idx + 1, next_psum)

        *_, final_psum = while_loop(
            cond_fn,
            body_fn,
            (b, x, k, init_c, psum),
        )

        if bias is not None:
            bias_tile = quantized_ops.load_tile(bias, [k], [tile_k], [0])
            final_psum = final_psum + bias_tile

        if self.accumulate_fp32 and final_psum.dtype != input.dtype:
            final_psum = final_psum.to(input.dtype)

        return final_psum

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tile_b: int = 1,
        tile_x: int = 64,
        tile_c: int = 128,
        tile_k: int = 64,
        **kwargs,
    ):
        """
        input shape:  (B, X, C)
        weight shape: (K, C)      when batched_weight=False
                      (B, K, C)   when batched_weight=True
        bias shape:   (K,)
        """
        B, X, C = input.shape
        K = weight.shape[1] if self.batched_weight else weight.shape[0]

        num_b = B // tile_b
        num_x = X // tile_x
        num_k = K // tile_k

        O = torch.empty((B, X, K), device=input.device, dtype=input.dtype)

        def cond_fn(b, x, k, c, O_buf):
            return b < B // tile_b

        def body_fn(b, x, k, c, O_buf):
            indices = [b, x, k, c]
            O_tile = self._outer_loop_body(
                indices,
                input=input,
                weight=weight,
                bias=bias,
                tile_b=tile_b,
                tile_x=tile_x,
                tile_c=tile_c,
                tile_k=tile_k,
                **kwargs,
            )

            O_buf = quantized_ops.store_tile(
                O_tile, O_buf, [b, x, k], [tile_b, tile_x, tile_k], [0, 1, 2]
            )

            # Increment 3D grid (b, x, k); c has bound 1 and always resets to 0
            new_indices = quantized_ops.increment_indices(
                indices, [num_b + 1, num_x, num_k, 1]
            )

            return (*new_indices, O_buf)

        factory_kwargs = {"device": input.device, "dtype": torch.int32}
        initial_state = (
            torch.tensor(0, **factory_kwargs),  # b
            torch.tensor(0, **factory_kwargs),  # x
            torch.tensor(0, **factory_kwargs),  # k
            torch.tensor(0, **factory_kwargs),  # c (inner loop start)
            O,
        )

        *_, final_O = while_loop(cond_fn, body_fn, initial_state)
        return final_O


def lower_gemm(pattern, tile_sizes, input, weight, bias=None, kwargs=None):
    """
    Lower a TiledGEMM pattern to Voyager IR.

    Args:
        pattern:    TiledGEMM instance (carries batched_weight/transpose_weight flags).
        tile_sizes: [tile_b, tile_x, tile_c, tile_k]
        input:      CPU tensor, shape (B, X, C)
        weight:     CPU tensor, shape (K, C) or (B, K, C) for batched_weight
        bias:       CPU tensor, shape (K,), or None
        kwargs:     Extra keyword args forwarded to forward() (e.g. MX scales)
    """
    gm = export_model(
        pattern,
        (input, weight, bias, *tile_sizes),
        kwargs=kwargs,
    )

    # Remove compile-time constant inputs (unused placeholders from tile_sizes)
    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()

    namer = NameGenerator()
    env = {}
    # starts discarded: TiledGEMM.forward always initialises all loop indices to 0
    placeholders, parameters, index_values, _ = _prepare_ir_graph(
        gm, (input, weight, bias), namer, env, kwargs=kwargs
    )

    B, X, C = input.shape
    batched_weight = getattr(pattern, "batched_weight", False)
    K = weight.shape[1] if batched_weight else weight.shape[0]

    tile_b, tile_x, tile_c, tile_k = tile_sizes
    assert B % tile_b == 0, f"B={B} not divisible by tile_b={tile_b}"
    assert X % tile_x == 0, f"X={X} not divisible by tile_x={tile_x}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert C % tile_c == 0, f"C={C} not divisible by tile_c={tile_c}"

    # Outer loop group: (b, x, k); inner reduction loop: (c,).
    # All loops start at 0 — the initial_state scalars in TiledGEMM.forward
    # are always torch.tensor(0, ...), so no non-zero start is possible.
    num_tiles = (B // tile_b, X // tile_x, K // tile_k, C // tile_c)
    loop_bounds = (
        tuple((0, n) for n in num_tiles[:3]),   # outer: b, x, k
        ((0, num_tiles[3]),),                    # inner reduction: c
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
