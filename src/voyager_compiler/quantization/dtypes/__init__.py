"""Numeric formats a tensor can be quantized to.

Each module maps bfloat16 bit patterns onto the representable values of one
format, which ``fake_quantize`` turns into a lookup table.
"""

from .fp8 import (
    _quantize_elemwise_core,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
)
from .normal_float import create_normal_map, quantize_to_nf
from .posit import quantize_to_posit

__all__ = [
    "_quantize_elemwise_core",
    "create_normal_map",
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "quantize_to_nf",
    "quantize_to_posit",
]
