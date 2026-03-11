"""
Tests for FlashAttention / FlashAttentionPipelined lowering to Voyager IR.

Covered cases:
  - FlashAttention (plain, no quantization) — numerical + IR
  - FlashAttentionPipelined (plain, no quantization) — pipelined and sequential paths
  - FlashAttention MX (microscaling) — unit-scale smoke test vs standard SDPA
  - FlashAttentionPipelined MX — pipelined and sequential paths vs non-pipelined MX gold
  - Various (B, H, N, d) shapes and tile sizes
"""

import pytest
import torch
import torch.nn.functional as F

import voyager_compiler.decomposed  # registers quantized_ops
from voyager_compiler.codegen.lowering.attention import (
    FlashAttention,
    FlashAttentionPipelined,
    lower_flash_attention,
)


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


def _lower_and_print(label, pattern, query, key, value, Br, Bc, kwargs=None):
    """Run lower_flash_attention and print the resulting IR (for inspection)."""
    module, _ = lower_flash_attention(
        pattern, query, key, value, Br=Br, Bc=Bc, kwargs=kwargs
    )
    print(f"\n=== {label} ===")
    print(module.format())
    return module


def _make_identity_qmap():
    """
    Build a 65536-entry bfloat16 lookup table where entry i holds the bfloat16
    value whose bit pattern is i, making vmap(x, qmap) == x for all bfloat16 x.
    """
    return torch.arange(65536, dtype=torch.int16).view(torch.bfloat16)


def _make_mx_scales(q, k, v, block_size):
    """
    Compute block-wise MX scale tensors for Q, K, V.

    Shapes (with input [B, H, N, d]):
        query_scale / key_scale : [B, H, N, d // block_size]  (per-block along d)
        value_scale             : [B, H, N // block_size, d]  (per-block along N)

    axes=[3] groups blocks of `block_size` elements along the d axis for Q/K.
    axes=[2] groups blocks of `block_size` rows along the N axis for V (the
    reduction axis of the subsequent P×V matmul).
    """
    calc = torch.ops.quantized_ops.calculate_mx_qparam
    # Compute in float32 for precision, then cast back to the input dtype.
    query_scale = calc(q.float(), axes=[3], block_size=block_size, quant_max=1.0).to(q.dtype)
    key_scale   = calc(k.float(), axes=[3], block_size=block_size, quant_max=1.0).to(k.dtype)
    value_scale = calc(v.float(), axes=[2], block_size=block_size, quant_max=1.0).to(v.dtype)
    return query_scale, key_scale, value_scale


# ---------------------------------------------------------------------------
# 1. FlashAttention (plain, no quantization)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,H,N,d,Br,Bc", [
    (1, 4, 256, 64, 64, 128),
    (2, 8, 256, 32, 64, 128),
])
def test_flash_attention_plain(B, H, N, d, Br, Bc):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(B, H, N, d, dtype=dtype)
    k = torch.randn(B, H, N, d, dtype=dtype)
    v = torch.randn(B, H, N, d, dtype=dtype)

    gold = F.scaled_dot_product_attention(q, k, v)

    model = FlashAttention(accumulate_fp32=True)
    out = model(q, k, v, Br=Br, Bc=Bc)
    _check_numerical(gold, out, max_err_tol=0.1, rms_err_tol=0.01,
                     label=f"FlashAttention B={B} H={H} N={N} d={d}")

    # IR lowering uses while_loop
    _lower_and_print(
        f"FlashAttention B={B} H={H} N={N} d={d}",
        model.cpu(), q.cpu(), k.cpu(), v.cpu(), Br, Bc,
    )


# ---------------------------------------------------------------------------
# 2. FlashAttentionPipelined (plain, no quantization)
#    Tc > 3  → pipelined path (_outer_loop_body_pipelined)
#    Tc <= 3 → sequential path (_outer_loop_body)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,H,N,d,Br,Bc", [
    (1, 4, 512, 64, 64, 128),   # Tc = 512//128 = 4 > 3 → pipelined
    (1, 4, 256, 64, 64, 128),   # Tc = 256//128 = 2 ≤ 3 → sequential
])
def test_flash_attention_pipelined_plain(B, H, N, d, Br, Bc):
    torch.manual_seed(1)
    dtype = torch.bfloat16
    q = torch.randn(B, H, N, d, dtype=dtype)
    k = torch.randn(B, H, N, d, dtype=dtype)
    v = torch.randn(B, H, N, d, dtype=dtype)

    gold = F.scaled_dot_product_attention(q, k, v)

    Tc = N // Bc
    model = FlashAttentionPipelined(accumulate_fp32=True)
    out = model(q, k, v, Br=Br, Bc=Bc)
    _check_numerical(gold, out, max_err_tol=0.1, rms_err_tol=0.01,
                     label=f"FlashAttentionPipelined B={B} H={H} N={N} d={d} Tc={Tc}")

    # IR lowering uses while_loop (flatten_loops=False)
    _lower_and_print(
        f"FlashAttentionPipelined B={B} H={H} N={N} d={d} Tc={Tc}",
        model.cpu(), q.cpu(), k.cpu(), v.cpu(), Br, Bc,
    )


# ---------------------------------------------------------------------------
# 3. FlashAttention MX (microscaling, no codebooks)
#    Unit QKV scales + identity qmap → the MX round-trip (normalize → lookup
#    → rescale) should introduce only small bfloat16 rounding error, so the
#    output must be close to F.scaled_dot_product_attention.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,H,N,d,Br,Bc,block_size", [
    (1, 4, 256, 64, 64, 128, 32),
    (2, 4, 256, 64, 64, 128, 32),
])
def test_flash_attention_mx(B, H, N, d, Br, Bc, block_size):
    torch.manual_seed(10)
    dtype = torch.bfloat16
    q = torch.randn(B, H, N, d, dtype=dtype)
    k = torch.randn(B, H, N, d, dtype=dtype)
    v = torch.randn(B, H, N, d, dtype=dtype)

    gold = F.scaled_dot_product_attention(q, k, v)

    # Unit scales: MX matmuls reduce to plain matmuls; only P-quantization
    # introduces a small round-trip error via the identity qmap.
    query_scale = torch.ones(B, H, N, d // block_size, dtype=dtype)
    key_scale   = torch.ones(B, H, N, d // block_size, dtype=dtype)
    value_scale = torch.ones(B, H, N // block_size, d, dtype=dtype)

    model = FlashAttention(
        qmap=_make_identity_qmap(),
        axes=[3],
        block_size=block_size,
        quant_max=1.0,
        accumulate_fp32=False,  # keep bfloat16 throughout; fp32 upcasting
        # in online softmax causes dtype mismatch in the MX matmul_mx path
    )
    out = model(q, k, v, Br=Br, Bc=Bc,
                query_scale=query_scale, key_scale=key_scale, value_scale=value_scale)

    _check_numerical(gold, out, max_err_tol=0.2, rms_err_tol=0.05,
                     label=f"FlashAttention MX B={B} H={H} N={N} d={d} bs={block_size}")

    _lower_and_print(
        f"FlashAttention MX B={B} H={H} N={N} d={d} bs={block_size}",
        model.cpu(), q.cpu(), k.cpu(), v.cpu(), Br, Bc,
        kwargs={
            "query_scale": query_scale.cpu(),
            "key_scale":   key_scale.cpu(),
            "value_scale": value_scale.cpu(),
        },
    )


# ---------------------------------------------------------------------------
# 4. FlashAttentionPipelined MX (microscaling, no codebooks)
#    Tc > 3  → pipelined path (_outer_loop_body_pipelined)
#    Tc <= 3 → sequential path (_outer_loop_body)
#    Gold: non-pipelined FlashAttention with the same MX config and real
#    (non-unit) scales, verifying pipelined == sequential numerically.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,H,N,d,Br,Bc,block_size", [
    (1, 4, 512, 64, 64, 128, 32),   # Tc=4 > 3 → pipelined
    (1, 4, 256, 64, 64, 128, 32),   # Tc=2 ≤ 3 → sequential
])
def test_flash_attention_pipelined_mx(B, H, N, d, Br, Bc, block_size):
    torch.manual_seed(11)
    dtype = torch.bfloat16
    q = torch.randn(B, H, N, d, dtype=dtype)
    k = torch.randn(B, H, N, d, dtype=dtype)
    v = torch.randn(B, H, N, d, dtype=dtype)

    Tc = N // Bc
    qmap = _make_identity_qmap()
    query_scale, key_scale, value_scale = _make_mx_scales(q, k, v, block_size)

    mx_ctor_kwargs = dict(
        qmap=qmap,
        axes=[3],
        block_size=block_size,
        quant_max=1.0,
        accumulate_fp32=False,  # keep bfloat16 throughout; fp32 upcasting
        # in online softmax causes dtype mismatch in the MX matmul_mx path
    )
    scale_fwd_kwargs = dict(
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
    )

    # Gold: non-pipelined FlashAttention with the same MX config.
    gold_model = FlashAttention(**mx_ctor_kwargs)
    gold = gold_model(q, k, v, Br=Br, Bc=Bc, **scale_fwd_kwargs)

    model = FlashAttentionPipelined(**mx_ctor_kwargs)
    out = model(q, k, v, Br=Br, Bc=Bc, **scale_fwd_kwargs)

    _check_numerical(gold, out, max_err_tol=0.05, rms_err_tol=0.01,
                     label=f"FlashAttentionPipelined MX B={B} H={H} N={N} d={d} Tc={Tc} bs={block_size}")

    _lower_and_print(
        f"FlashAttentionPipelined MX B={B} H={H} N={N} d={d} Tc={Tc} bs={block_size}",
        model.cpu(), q.cpu(), k.cpu(), v.cpu(), Br, Bc,
        kwargs={k_: v_.cpu() for k_, v_ in scale_fwd_kwargs.items()},
    )
