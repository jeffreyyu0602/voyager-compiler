"""Custom operator registrations and the layouts they work in.

Importing this package defines the ``quantized_ops`` torch.library: the
decomposed quantize / dequantize / microscaling ops, plus hardware-layout
twins of the aten operators the data-layout pass retargets.  Import it for
that side effect before referencing any ``torch.ops.quantized_ops.*`` target —
the op namespace resolves lazily, so a missing registration surfaces at call
time, not at import.
"""

from .quantized import (  # noqa: F401  (defines quantized_ops.*)
    calculate_mx_qparam,
    dequantize,
    expand,
    filter_outlier,
    quantize,
    quantize_mx,
    quantize_mx_outlier,
    quantized_ops_lib,
    slice_csr_tensor,
    spmm_csr,
    vmap,
)
from .layout import (
    DEFAULT_GEMM_WEIGHT_LAYOUT,
    DEFAULT_LAYOUT_POLICY,
    GEMM_OP_VARIANTS,
    GEMM_WEIGHT_LAYOUTS,
    HWIO_TO_OIHW,
    LAYOUT_POLICIES,
    NCHW_TO_NHWC,
    NHWC_OP_VARIANTS,
    NHWC_TO_NCHW,
    OIHW_TO_HWIO,
    POLICY_GEMM_WEIGHT_LAYOUT,
    project,
    unproject,
)

__all__ = [
    "DEFAULT_GEMM_WEIGHT_LAYOUT",
    "DEFAULT_LAYOUT_POLICY",
    "GEMM_OP_VARIANTS",
    "GEMM_WEIGHT_LAYOUTS",
    "HWIO_TO_OIHW",
    "LAYOUT_POLICIES",
    "NCHW_TO_NHWC",
    "NHWC_OP_VARIANTS",
    "NHWC_TO_NCHW",
    "OIHW_TO_HWIO",
    "POLICY_GEMM_WEIGHT_LAYOUT",
    "calculate_mx_qparam",
    "dequantize",
    "expand",
    "filter_outlier",
    "project",
    "quantize",
    "quantize_mx",
    "quantize_mx_outlier",
    "quantized_ops_lib",
    "slice_csr_tensor",
    "spmm_csr",
    "unproject",
    "vmap",
]
