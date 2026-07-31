"""Quantization: choose a spec per node, observe, then bake in quant/dequant.

The PT2E flow (``quantize_pt2e``) is the one the compiler consumes; the eager
module-swap flow (``quantize``) is for QAT experiments.  Both are configured
with the comma-separated spec strings ``QuantizationSpec.from_str`` parses.
"""

from .fake_quantize import FusedAmaxObsFakeQuantize, get_quantization_map
from .qconfig import QConfig, get_qconfig
from .quantize import (
    convert,
    get_conv_bn_layers,
    prepare,
    propagate_config,
    quantize,
    replace_softmax,
)
from .quantize_pt2e import (
    convert_pt2e,
    derive_bias_qparams_fn,
    get_default_quantizer,
    prepare_pt2e,
    sink_obs_or_fq,
    swap_matmul_inputs,
)
from .qspec import QScheme
from .quantizer.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)
from .quantizer.xnnpack_quantizer_utils import QuantizationConfig

__all__ = [
    "DerivedQuantizationSpec",
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationConfig",
    "QScheme",
    "QuantizationSpec",
    "convert",
    "convert_pt2e",
    "derive_bias_qparams_fn",
    "get_conv_bn_layers",
    "get_default_quantizer",
    "get_qconfig",
    "get_quantization_map",
    "prepare",
    "prepare_pt2e",
    "propagate_config",
    "quantize",
    "replace_softmax",
    "sink_obs_or_fq",
    "swap_matmul_inputs",
]
