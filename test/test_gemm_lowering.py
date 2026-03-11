"""
Tests for TiledGEMM forward pass (numerical correctness) and lower_gemm IR generation.

Covered cases:
  - linear / linear_mx (weight shape K×C, op = linear)
  - matmul / matmul_mx (weight shape K×C, loaded transposed, op = matmul)
  - batched BMM       (weight shape B×K×C, batched_weight=True)
  - MHA-style BMM     (Q×K^T after reshaping to 3-D batch)
  - varied batch shapes and tile sizes
"""

import pytest
import torch
import torch.nn.functional as F

import voyager_compiler.decomposed  # registers quantized_ops
from voyager_compiler.codegen.lowering.b2bgemm import TiledGEMM, lower_gemm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_numerical(gold, out, max_err_tol, rms_err_tol, label=""):
    g = gold.float()
    o = out.float()
    max_err = (g - o).abs().max().item()
    rms_err = ((g - o) ** 2).mean().sqrt().item()
    assert max_err <= max_err_tol, (
        f"{label}: max_err {max_err:.4g} > tol {max_err_tol}"
    )
    assert rms_err <= rms_err_tol, (
        f"{label}: rms_err {rms_err:.4g} > tol {rms_err_tol}"
    )


def _lower_and_print(label, pattern, tile_sizes, input, weight, bias=None, kwargs=None):
    """Run lower_gemm and print the resulting IR (for inspection)."""
    module, _ = lower_gemm(
        pattern=pattern,
        tile_sizes=tile_sizes,
        input=input,
        weight=weight,
        bias=bias,
        kwargs=kwargs,
    )
    print(f"\n=== {label} ===")
    print(module.format())
    return module


# ---------------------------------------------------------------------------
# 1. Linear (no quantization)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K", [
    (1, 64, 128, 64),
    (2, 128, 256, 128),
])
@pytest.mark.parametrize("with_bias", [True, False])
def test_linear(B, X, C, K, with_bias):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(K, C, dtype=dtype)
    bias   = torch.randn(K, dtype=dtype) if with_bias else None

    gold = F.linear(input, weight, bias)

    model = TiledGEMM(torch.ops.aten.linear.default, accumulate_fp32=True)
    tile_b, tile_x, tile_c, tile_k = 1, X // 2, C // 2, K // 2
    out = model(input, weight, bias,
                tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k)

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"linear B={B} bias={with_bias}")

    _lower_and_print(
        f"linear B={B} bias={with_bias}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        bias=bias.cpu() if bias is not None else None,
        kwargs={"accumulate_fp32": True},
    )


# ---------------------------------------------------------------------------
# 2. Matmul (no quantization) — weight stored as (K, C), loaded transposed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K", [
    (1, 64, 128, 64),
    (2, 64, 128, 64),
])
def test_matmul(B, X, C, K):
    torch.manual_seed(1)
    dtype = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(K, C, dtype=dtype)   # stored (K, C)

    # Gold: input @ weight.T
    gold = torch.matmul(input, weight.T)

    model = TiledGEMM(
        torch.ops.quantized_ops.matmul,
        accumulate_fp32=True,
        transpose_weight=True,   # tile loaded as (C, K) so matmul(input, tile) works
    )
    tile_b, tile_x, tile_c, tile_k = 1, X // 2, C // 2, K // 2
    out = model(input, weight, bias=None,
                tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k)

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"matmul B={B}")

    _lower_and_print(
        f"matmul B={B}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        kwargs={"accumulate_fp32": True},
    )


# ---------------------------------------------------------------------------
# 3. Linear MX (block-wise microscaling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K,block_size", [
    (1, 64, 128, 64, 32),
    (1, 64, 256, 128, 32),
])
def test_linear_mx(B, X, C, K, block_size):
    torch.manual_seed(2)
    dtype = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(K, C, dtype=dtype)
    bias   = torch.randn(K, dtype=dtype)

    # Block-wise scales (one scale per block along the C dimension)
    input_scale  = torch.ones(B, X, C // block_size, dtype=dtype)
    weight_scale = torch.ones(K, C // block_size, dtype=dtype)

    # Gold: linear with unit scales is identical to plain linear
    gold = F.linear(input, weight, bias)

    model = TiledGEMM(
        torch.ops.quantized_ops.linear_mx,
        block_size=block_size,
        accumulate_fp32=True,
    )
    tile_b, tile_x, tile_c, tile_k = 1, X // 2, C // 2, K // 2
    out = model(
        input, weight, bias,
        tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"linear_mx B={B} bs={block_size}")

    _lower_and_print(
        f"linear_mx B={B} bs={block_size}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        bias=bias.cpu(),
        kwargs={
            "input_scale": input_scale.cpu(),
            "weight_scale": weight_scale.cpu(),
            "accumulate_fp32": True,
        },
    )


# ---------------------------------------------------------------------------
# 4. Matmul MX — weight loaded transposed + block-wise scales
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K,block_size", [
    (1, 64, 128, 64, 32),
])
def test_matmul_mx(B, X, C, K, block_size):
    torch.manual_seed(3)
    dtype = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(K, C, dtype=dtype)   # stored (K, C)

    # Block-wise scales — both in (K, C//block_size) layout matching weight
    input_scale  = torch.ones(B, X, C // block_size, dtype=dtype)
    weight_scale = torch.ones(K, C // block_size, dtype=dtype)

    # Gold: input @ weight.T with unit scales
    gold = torch.matmul(input, weight.T)

    model = TiledGEMM(
        torch.ops.quantized_ops.matmul_mx,
        block_size=block_size,
        accumulate_fp32=True,
        transpose_weight=True,
    )
    tile_b, tile_x, tile_c, tile_k = 1, X // 2, C // 2, K // 2
    out = model(
        input, weight, bias=None,
        tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"matmul_mx B={B} bs={block_size}")

    _lower_and_print(
        f"matmul_mx B={B} bs={block_size}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        kwargs={
            "input_scale": input_scale.cpu(),
            "weight_scale": weight_scale.cpu(),
            "accumulate_fp32": True,
        },
    )


# ---------------------------------------------------------------------------
# 5. Batched GEMM (BMM) — weight has a batch dimension
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K", [
    (2, 64, 128, 64),
    (4, 32, 64, 32),
])
def test_bmm(B, X, C, K):
    """BMM: each batch has its own weight matrix. weight shape (B, K, C)."""
    torch.manual_seed(4)
    dtype = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(B, K, C, dtype=dtype)  # (B, K, C)

    # Gold: batched input @ weight.transpose(-1,-2) = (B, X, K)
    gold = torch.bmm(input, weight.transpose(-1, -2))

    model = TiledGEMM(
        torch.ops.quantized_ops.matmul,
        accumulate_fp32=True,
        batched_weight=True,
        transpose_weight=True,
    )
    tile_b, tile_x, tile_c, tile_k = 1, X // 2, C // 2, K // 2
    out = model(input, weight, bias=None,
                tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k)

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"BMM B={B}")

    _lower_and_print(
        f"BMM B={B}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        kwargs={"accumulate_fp32": True},
    )


# ---------------------------------------------------------------------------
# 6. MHA-style BMM: Q @ K^T after reshaping (B, H, T, C) → (B*H, T, C)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,H,T,C_head", [
    (2, 4, 64, 32),
])
def test_mha_bmm(B, H, T, C_head):
    """
    Q × K^T for multi-head attention, reshaped to (B*H, T, C_head).

    Q and K both have shape (B*H, T, C_head).
    Result shape: (B*H, T, T).
    """
    torch.manual_seed(5)
    dtype  = torch.bfloat16
    BH     = B * H

    Q = torch.randn(BH, T, C_head, dtype=dtype)
    K = torch.randn(BH, T, C_head, dtype=dtype)   # stored as (BH, K_dim=T, C=C_head)

    # Gold: Q @ K^T
    gold = torch.bmm(Q, K.transpose(-1, -2))

    # tile_k tiles the K_dim (= T) dimension of the key
    tile_bh = 1
    tile_t  = T // 2
    tile_c  = C_head // 2
    tile_k  = T // 2

    model = TiledGEMM(
        torch.ops.quantized_ops.matmul,
        accumulate_fp32=True,
        batched_weight=True,
        transpose_weight=True,
    )
    out = model(Q, K, bias=None,
                tile_b=tile_bh, tile_x=tile_t, tile_c=tile_c, tile_k=tile_k)

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"MHA BMM B={B} H={H}")

    _lower_and_print(
        f"MHA BMM B={B} H={H}",
        model,
        [tile_bh, tile_t, tile_c, tile_k],
        Q.cpu(), K.cpu(),
        kwargs={"accumulate_fp32": True},
    )


# ---------------------------------------------------------------------------
# 7. Batched GEMM MX (BMM with block-wise microscaling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K,block_size", [
    (2, 64, 128, 64, 32),
    (4, 32, 64, 32, 16),
])
def test_bmm_mx(B, X, C, K, block_size):
    """BMM with block-wise MX scales. Weight is batched (B, K, C)."""
    torch.manual_seed(7)
    dtype = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(B, K, C, dtype=dtype)

    # Batched block-wise scales along the C (reduction) dimension
    input_scale  = torch.ones(B, X, C // block_size, dtype=dtype)
    weight_scale = torch.ones(B, K, C // block_size, dtype=dtype)  # batched!

    # Gold: batched matmul with unit scales ≈ plain BMM
    gold = torch.bmm(input, weight.transpose(-1, -2))

    model = TiledGEMM(
        torch.ops.quantized_ops.matmul_mx,
        block_size=block_size,
        accumulate_fp32=True,
        batched_weight=True,
        transpose_weight=True,
    )
    tile_b, tile_x, tile_c, tile_k = 1, X // 2, C // 2, K // 2
    out = model(
        input, weight, bias=None,
        tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"BMM MX B={B} bs={block_size}")

    _lower_and_print(
        f"BMM MX B={B} bs={block_size}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        kwargs={
            "input_scale": input_scale.cpu(),
            "weight_scale": weight_scale.cpu(),
            "accumulate_fp32": True,
        },
    )


# ---------------------------------------------------------------------------
# 8. Varied batch sizes for linear (stress-test tile divisibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,X,C,K,tile_b,tile_x,tile_c,tile_k", [
    (4, 128, 256, 128,  2, 64, 128, 64),
    (1, 256, 512, 256,  1, 64, 128, 64),
])
def test_linear_varied_tiles(B, X, C, K, tile_b, tile_x, tile_c, tile_k):
    torch.manual_seed(6)
    dtype  = torch.bfloat16
    input  = torch.randn(B, X, C, dtype=dtype)
    weight = torch.randn(K, C, dtype=dtype)
    bias   = torch.randn(K, dtype=dtype)

    gold = F.linear(input, weight, bias)

    model = TiledGEMM(torch.ops.aten.linear.default, accumulate_fp32=True)
    out = model(input, weight, bias,
                tile_b=tile_b, tile_x=tile_x, tile_c=tile_c, tile_k=tile_k)

    _check_numerical(gold, out, max_err_tol=2, rms_err_tol=0.5,
                     label=f"linear varied B={B} X={X} C={C} K={K}")

    _lower_and_print(
        f"linear varied B={B} X={X} C={C} K={K}",
        model,
        [tile_b, tile_x, tile_c, tile_k],
        input.cpu(), weight.cpu(),
        bias=bias.cpu(),
        kwargs={"accumulate_fp32": True},
    )
