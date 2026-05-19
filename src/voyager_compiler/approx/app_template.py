# create corresponding templates
import torch

def get_compute_dtype(target_dtype):
    if target_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return torch.bfloat16
    elif target_dtype in [torch.int8, torch.uint8]:
        return torch.int32
    elif target_dtype == torch.float16:
        return torch.float32 # Good practice for stable fp16 accumulation
    else:
        return target_dtype  # Default to native (fp32, bf16, etc.)

def linear_app_template(x, coeffs, dtype):
    a, b = coeffs

    x_cast = x.to(dtype)
    a_cast = a.to(dtype)
    b_cast = b.to(dtype)

    compute_dtype = get_compute_dtype(dtype)

    ax_compute = a_cast.to(compute_dtype) * x_cast.to(compute_dtype)
    ax_cast = ax_compute.to(dtype)

    result_compute = ax_cast.to(compute_dtype) + b_cast.to(compute_dtype)
    final_result = result_compute.to(dtype)

    return final_result


def quadratic_app_template(x, coeffs, dtype):
    a, b, c = coeffs

    x_cast = x.to(dtype)
    a_cast = a.to(dtype)
    b_cast = b.to(dtype)
    c_cast = c.to(dtype)

    compute_dtype = get_compute_dtype(dtype)

    x_sq_compute = x_cast.to(compute_dtype) * x_cast.to(compute_dtype)
    x_sq_cast = x_sq_compute.to(dtype)

    ax2_compute = a_cast.to(compute_dtype) * x_sq_cast.to(compute_dtype)
    ax2_cast = ax2_compute.to(dtype)

    bx_compute = b_cast.to(compute_dtype) * x_cast.to(compute_dtype)
    bx_cast = bx_compute.to(dtype)

    ax2_plus_bx_compute = ax2_cast.to(compute_dtype) + bx_cast.to(compute_dtype)
    ax2_plus_bx_cast = ax2_plus_bx_compute.to(dtype)

    result_compute = ax2_plus_bx_cast.to(compute_dtype) + c_cast.to(compute_dtype)
    final_result = result_compute.to(dtype)

    return final_result

def quadratic_app_synth(x, coeffs, dtype):
    """Factored quadratic: (c1*x + c2) * (x + c3)
    From SMT: fp.mul(fp.add(c2, fp.mul(x, c1)), fp.add(x, c3))
    """
    c1, c2, c3 = coeffs

    x_cast = x.to(dtype)
    c1_cast = c1.to(dtype)
    c2_cast = c2.to(dtype)
    c3_cast = c3.to(dtype)

    compute_dtype = get_compute_dtype(dtype)

    # c1 * x
    c1x_compute = x_cast.to(compute_dtype) * c1_cast.to(compute_dtype)
    c1x_cast = c1x_compute.to(dtype)

    # c1*x + c2
    left_compute = c1x_cast.to(compute_dtype) + c2_cast.to(compute_dtype)
    left_cast = left_compute.to(dtype)

    # x + c3
    right_compute = x_cast.to(compute_dtype) + c3_cast.to(compute_dtype)
    right_cast = right_compute.to(dtype)

    # (c1*x + c2) * (x + c3)
    result_compute = left_cast.to(compute_dtype) * right_cast.to(compute_dtype)
    final_result = result_compute.to(dtype)

    return final_result


def _cast_to_precision(val, prec):
    """Cast tensor to the specified precision level using round-nearest-even.

    prec options:
      - 'native': no cast (stay in current dtype, typically float32)
      - 'bf16': cast to bfloat16 (RNE is PyTorch's default rounding mode)
      - 'fp15': round to 6 mantissa bits (8 exp + 6 mant) using RNE
      - 'fp14': round to 5 mantissa bits (8 exp + 5 mant) using RNE
    """
    if prec == 'native':
        return val
    elif prec == 'bf16':
        return val.to(torch.bfloat16).to(val.dtype)
    elif prec in ('fp15', 'fp14', 'fp13'):
        # RNE rounding to reduced mantissa bits:
        # BF16 has 7 mantissa bits. fp15 = 6, fp14 = 5, fp13 = 4.
        # We drop the lowest k bits with proper RNE rounding.
        #
        # Strategy: cast to bf16 first (RNE), then round the mantissa
        # by adding/subtracting a power of 2 that forces rounding at
        # the desired bit position (the "round by addition" trick).
        bf = val.to(torch.bfloat16)
        # Number of mantissa bits to drop
        drop = {'fp15': 1, 'fp14': 2, 'fp13': 3}[prec]
        # The unit-in-last-place for the target precision relative to bf16's ULP:
        # Adding and subtracting 2^(mantissa_bits - target_bits - 1) forces RNE
        # at the target bit. But simpler: use float32 round-trip with masking.
        #
        # Proper RNE via float arithmetic:
        # For fp15 (drop 1 bit): round bf16 value to nearest even at bit 0
        f32 = bf.to(torch.float32)
        # Get the ULP of each value at bf16 precision, then scale
        # Actually, the cleanest approach: use torch rounding
        # Truncate to target bits by: multiply by 2^(7-drop), round, divide back
        # This gives exact RNE at the target precision.
        scale = float(2 ** (7 - drop))  # fp15: 2^6=64, fp14: 2^5=32
        # Round mantissa: shift so target LSB is at integer boundary
        # bf16 mantissa is 7 bits, so the value in bf16 has ULP = 2^(exp-7)
        # We want ULP = 2^(exp-7+drop)
        # Trick: val * scale rounds the mantissa to (7-drop) bits when cast to bf16
        rounded = (f32 * scale).to(torch.bfloat16).to(torch.float32) / scale
        return rounded.to(val.dtype)
    else:
        raise ValueError(f"Unknown precision: {prec}")


def quadratic_app_synth_mp(x, coeffs, dtype, precisions=None):
    """Factored quadratic with mixed precision per operation: (c1*x + c2) * (x + c3)

    precisions: tuple of (mul1_prec, add1_prec, add2_prec, mul2_prec)
        Each is one of: 'native', 'bf16', 'fp15', 'fp14'
        Default (None) uses bf16 for all (same as quadratic_app_synth).

    RNE (round-nearest-ties-to-even) is used for all precision reductions,
    matching the hardware rounding mode (roundNearestTiesToEven in SMT-LIB).
    """
    if precisions is None:
        precisions = ('bf16', 'bf16', 'bf16', 'bf16')
    mul1_prec, add1_prec, add2_prec, mul2_prec = precisions

    c1, c2, c3 = coeffs

    x_cast = x.to(dtype)
    c1_cast = c1.to(dtype)
    c2_cast = c2.to(dtype)
    c3_cast = c3.to(dtype)

    compute_dtype = get_compute_dtype(dtype)

    # c1 * x → mul1
    c1x = c1_cast.to(compute_dtype) * x_cast.to(compute_dtype)
    c1x = _cast_to_precision(c1x, mul1_prec)

    # c1*x + c2 → add1
    left = c1x.to(compute_dtype) + c2_cast.to(compute_dtype)
    left = _cast_to_precision(left, add1_prec)

    # x + c3 → add2
    right = x_cast.to(compute_dtype) + c3_cast.to(compute_dtype)
    right = _cast_to_precision(right, add2_prec)

    # (c1*x + c2) * (x + c3) → mul2
    result = left.to(compute_dtype) * right.to(compute_dtype)
    result = _cast_to_precision(result, mul2_prec)

    return result


def quadratic_stable_app_template(x, coeffs, dtype):
    n, a, b, c = coeffs
    x_cast = x.to(dtype)
    n_cast = n.to(dtype)
    a_cast = a.to(dtype)
    b_cast = b.to(dtype)
    c_cast = c.to(dtype)

    compute_dtype = get_compute_dtype(dtype)
    x_plus_n_compute = x_cast.to(compute_dtype) + n_cast.to(compute_dtype)
    x_plus_n_cast = x_plus_n_compute.to(dtype)

    x_plus_n_sq_compute = x_plus_n_cast.to(compute_dtype) * x_plus_n_cast.to(compute_dtype)
    x_plus_n_sq_cast = x_plus_n_sq_compute.to(dtype)

    ax2_compute = a_cast.to(compute_dtype) * x_plus_n_sq_cast.to(compute_dtype)
    ax2_cast = ax2_compute.to(dtype)

    bx_compute = b_cast.to(compute_dtype) * x_plus_n_cast.to(compute_dtype)
    bx_cast = bx_compute.to(dtype)

    ax2_plus_bx_compute = ax2_cast.to(compute_dtype) + bx_cast.to(compute_dtype)
    ax2_plus_bx_cast = ax2_plus_bx_compute.to(dtype)

    result_compute = ax2_plus_bx_cast.to(compute_dtype) + c_cast.to(compute_dtype)
    final_result = result_compute.to(dtype)

    return final_result