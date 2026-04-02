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