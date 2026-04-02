# we statically store the parameters for the approximation templates
# each template contains the following fields:
# - tag: a string tag that mark some property of the template
# - name: the target function
# - the template used (stored in app_template.py)
# - the param datatype: fp8, bf16, in8, etc.
# - the param values: a list of floats or ints that are used in the approximation formula
# - - the params are for segments, containing start, end, and coeffs for each segment

import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict

SUPPORTED_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "uint8": torch.uint8
}

@dataclass
class SegmentConfig:
    """Stores the boundaries and coefficients for a single approximation segment."""
    start: float
    end: float
    # Note: Based on the previous numpy code, these are ordered as (c0, c1, c2)
    # which corresponds to the formula: c0 + c1*x + c2*x^2
    coeffs: Tuple[float, ...]

@dataclass
class AppTemplateConfig:
    """Defines the full approximation template for a specific function."""
    tag: str
    name: str
    template_name: str
    param_dtype: torch.dtype
    segments: List[SegmentConfig]

# =============================================================================
# Configuration Registry
# =============================================================================

# We store all templates in a list. You can easily convert this to a dict
# keyed by `name` or `tag` later if you need fast lookups.
APPROXIMATION_REGISTRY: List[AppTemplateConfig] = [
    AppTemplateConfig(
        tag="kartik-thesis",
        name="elu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -2.331, (0.016, 0.15, -0.642)),
            SegmentConfig(-2.331, -0.95, (0.098, 0.531, -0.198)),
            SegmentConfig(-0.95, 0.081, (0.318, 0.95, 0.001)),
            SegmentConfig(0.081, 3.972, (0.000, 1.001, -0.001)),
            SegmentConfig(3.972, 5.0, (0.001, 0.992, 0.017)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="exp",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -4.286, (-0.023, -0.197, -0.409)),
            SegmentConfig(-4.286, -2.060, (0.023, 0.196, 0.432)),
            SegmentConfig(-2.060, -0.750, (0.124, 0.612, 0.860)),
            SegmentConfig(-0.750, 0.196, (0.382, 0.999, 1.005)),
            SegmentConfig(0.196, 1.0, (0.879, 0.804, 1.024)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="gelu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -3.691, (0.013, 0.114, 0.240)),
            SegmentConfig(-3.691, -0.934, (-0.027, -0.187, -0.315)),
            SegmentConfig(-0.934, 0.925, (0.341, 0.501, 0.006)),
            SegmentConfig(0.925, 4.010, (-0.025, 1.178, -0.307)),
            SegmentConfig(4.010, 5.0, (0.032, 0.718, 0.616)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="mish",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -3.481, (-0.030, -0.301, -0.791)),
            SegmentConfig(-3.481, -1.285, (-0.001, -0.095, -0.432)),
            SegmentConfig(-1.285, 0.947, (0.266, 0.591, 0.008)),
            SegmentConfig(0.947, 4.000, (-0.016, 1.127, -0.245)),
            SegmentConfig(4.000, 5.0, (0.009, 0.926, 0.156)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="relu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, 0.0, (0.000, 0.000, 0.000)),
            SegmentConfig(0.0, 5.0, (0.000, 1.000, 0.000)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="selu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -1.658, (0.044, 0.379, -0.925)),
            SegmentConfig(-1.658, 0.048, (0.367, 1.450, -0.037)),
            SegmentConfig(0.048, 0.128, (-2.695, 1.741, -0.044)),
            SegmentConfig(0.128, 3.907, (0.000, 1.051, 0.000)),
            SegmentConfig(3.907, 5.0, (0.000, 1.050, 0.000)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="sigmoid",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -2.676, (0.012, 0.115, 0.287)),
            SegmentConfig(-2.676, -0.359, (0.042, 0.278, 0.505)),
            SegmentConfig(-0.359, 0.356, (0.000, 0.248, 0.500)),
            SegmentConfig(0.356, 2.673, (-0.042, 0.279, 0.495)),
            SegmentConfig(2.673, 5.0, (-0.012, 0.115, 0.713)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="silu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -3.495, (-0.028, -0.282, -0.750)),
            SegmentConfig(-3.495, -1.376, (0.002, -0.073, -0.384)),
            SegmentConfig(-1.376, 1.401, (0.210, 0.499, 0.010)),
            SegmentConfig(1.401, 3.992, (-0.003, 1.095, -0.407)),
            # SegmentConfig(3.992, 10.0, (-0.041, 1.400, -1.017)),
            SegmentConfig(3.992, 5.0, (-0.041, 1.400, -1.017)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="softplus",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -3.247, (0.006, 0.071, 0.198)),
            SegmentConfig(-3.247, -1.542, (0.037, 0.269, 0.519)),
            SegmentConfig(-1.542, 1.633, (0.112, 0.499, 0.697)),
            SegmentConfig(1.633, 3.897, (0.029, 0.770, 0.477)),
            SegmentConfig(3.897, 5.0, (-0.009, 1.066, -0.100)),
        ]
    ),

    AppTemplateConfig(
        tag="kartik-thesis",
        name="tanh",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-5.0, -1.681, (0.013, 0.103, -0.803)),
            SegmentConfig(-1.681, -0.022, (0.297, 1.058, -0.001)),
            SegmentConfig(-0.022, 0.020, (4.687, 1.255, 0.002)),
            SegmentConfig(0.020, 1.676, (-0.299, 1.060, 0.000)),
            SegmentConfig(1.676, 5.0, (-0.013, 0.105, 0.801)),
        ]
    )
]

def get_all_names_by_tag(tag: str) -> list[str]:
    """Return all function names registered under *tag*."""
    return [c.name for c in APPROXIMATION_REGISTRY if c.tag == tag]


# Helper function to easily fetch configs in your execution code
def get_app_config(name: str, tag: str = "kartik-thesis") -> AppTemplateConfig:
    """Retrieves the template configuration for a given function and tag."""
    for config in APPROXIMATION_REGISTRY:
        if config.name == name and config.tag == tag:
            return config
    raise ValueError(f"No configuration found for name='{name}' and tag='{tag}'")