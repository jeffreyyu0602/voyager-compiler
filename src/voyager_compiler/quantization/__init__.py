"""Quantization: choose a spec per node, observe, then bake in quant/dequant.

The PT2E flow (``quantize_pt2e``) is the one the compiler consumes; the eager
module-swap flow (``quantize``) is for QAT experiments.  Both are configured
with the comma-separated spec strings ``QuantizationSpec.from_str`` parses.
"""

from voyager_compiler.quantization.fake_quantize import (
    FusedAmaxObsFakeQuantize,
    get_quantization_map,
)
from voyager_compiler.quantization.gptq import (
    CACHE_RESERVE,
    compensate_weight,
    gptq,
)
from voyager_compiler.quantization.lcq import (
    CodebookGrid,
    Weighting,
    Histogram,
    codebook_qmap,
    fit_codebooks,
    load_codebooks,
    normal_float_levels,
    optimal_codebook,
    to_integer_codebook,
)
from voyager_compiler.quantization.qconfig import QConfig, get_qconfig
from voyager_compiler.quantization.qspec import QScheme
from voyager_compiler.quantization.quantize import (
    convert,
    get_conv_bn_layers,
    prepare,
    propagate_config,
    quantize,
    replace_softmax,
)
from voyager_compiler.quantization.quantize_pt2e import (
    convert_pt2e,
    derive_bias_qparams_fn,
    get_default_quantizer,
    prepare_pt2e,
    sink_obs_or_fq,
    swap_matmul_inputs,
)
from voyager_compiler.quantization.quantizer.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

__all__ = [
    "CACHE_RESERVE",
    "CodebookGrid",
    "Weighting",
    "Histogram",
    "DerivedQuantizationSpec",
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QScheme",
    "QuantizationConfig",
    "QuantizationSpec",
    "codebook_qmap",
    "compensate_weight",
    "convert",
    "convert_pt2e",
    "derive_bias_qparams_fn",
    "fit_codebooks",
    "get_conv_bn_layers",
    "get_default_quantizer",
    "get_qconfig",
    "get_quantization_map",
    "gptq",
    "load_codebooks",
    "normal_float_levels",
    "optimal_codebook",
    "prepare",
    "prepare_pt2e",
    "propagate_config",
    "quantize",
    "replace_softmax",
    "sink_obs_or_fq",
    "swap_matmul_inputs",
    "to_integer_codebook",
]
