"""
Registry of TIMM models for the approx benchmark, annotated with the
activation functions each model actually uses during its forward pass.

The `activations` list controls which approx bindings are applied when
benchmarking a model — only functions the model actually calls are bound,
so results are not polluted by irrelevant patches.

Functions match names in APPROXIMATION_REGISTRY (app_param.py):
    relu, gelu, silu, sigmoid, tanh, elu, selu, mish, softplus, exp

`note` is optional context (sensitivity, paper reference, etc.).
`skip` marks models excluded from automated sweeps (with a reason).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    arch: str
    activations: List[str]
    family: str
    note: str = ""
    skip: bool = False
    skip_reason: str = ""
    has_exp_softmax: bool = False  # True if model has interceptable softmax (F.softmax or SDPA)


MODELS: List[ModelConfig] = [
    # -------------------------------------------------------------------------
    # Standard CNNs — ReLU baseline
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="resnet18",
        activations=["relu"],
        family="standard_cnn",
        note="Baseline: zero-overhead ReLU approximation.",
    ),
    ModelConfig(
        arch="resnet50",
        activations=["relu"],
        family="standard_cnn",
        note="Baseline: zero-overhead ReLU approximation.",
    ),
    ModelConfig(
        arch="resnet101",
        activations=["relu"],
        family="standard_cnn",
        note="Baseline: zero-overhead ReLU approximation.",
    ),

    # -------------------------------------------------------------------------
    # Legacy — ReLU, heavy compute
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="vgg16",
        activations=["relu"],
        family="legacy",
        note="Heavy redundant compute; tests ReLU approx at scale.",
    ),
    ModelConfig(
        arch="vgg19",
        activations=["relu"],
        family="legacy",
    ),

    # -------------------------------------------------------------------------
    # Efficient / Mobile — SiLU + Sigmoid (SE blocks)
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="efficientnet_b0",
        activations=["silu", "sigmoid"],
        family="efficient",
        note="49× SiLU + 16× Sigmoid (SE gates). Demonstrated 45.1% speedup.",
    ),
    ModelConfig(
        arch="efficientnet_b4",
        activations=["silu", "sigmoid"],
        family="efficient",
        note="96× SiLU + 32× Sigmoid (SE gates). Richer sigmoid coverage than B0.",
    ),

    # -------------------------------------------------------------------------
    # Mobile / NAS — Hardswish + Hardsigmoid (NOT in approx registry)
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="mobilenetv3_large_100",
        activations=[],          # hardsigmoid/hardswish not in registry
        family="mobile",
        note="Uses Hardswish (21×) + Hardsigmoid (8×) + ReLU. "
             "Not covered by current approx registry.",
        skip=True,
        skip_reason="hardsigmoid/hardswish not in APPROXIMATION_REGISTRY",
    ),
    ModelConfig(
        arch="hardcorenas_a",
        activations=[],
        family="mobile",
        note="Hardswish (15×) + Hardsigmoid (8×). Extreme sensitivity candidate.",
        skip=True,
        skip_reason="hardsigmoid/hardswish not in APPROXIMATION_REGISTRY",
    ),
    ModelConfig(
        arch="hardcorenas_b",
        activations=[],
        family="mobile",
        note="Hardswish (25×) + Hardsigmoid (4×). Extreme sensitivity candidate.",
        skip=True,
        skip_reason="hardsigmoid/hardswish not in APPROXIMATION_REGISTRY",
    ),

    # -------------------------------------------------------------------------
    # Modern CNNs — SiLU / GELU
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="resnext26ts",
        activations=["silu"],
        family="modern_cnn",
        note="27× SiLU. Peak speedup 3.3× reported in evaluations.",
    ),
    ModelConfig(
        arch="convnext_tiny",
        activations=["gelu"],
        family="modern_cnn",
        note="18× F.gelu (called directly, not via nn.GELU). SOTA CNN standard.",
    ),
    ModelConfig(
        arch="convnext_base",
        activations=["gelu"],
        family="modern_cnn",
    ),

    # -------------------------------------------------------------------------
    # RegNet — ReLU / SiLU (industrial design space)
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="regnetx_008",
        activations=["relu"],
        family="regnet",
        note="49× ReLU. RegNetX uses pure ReLU throughout.",
    ),
    ModelConfig(
        arch="regnety_040",
        activations=["relu"],
        family="regnet",
        note="89× ReLU. RegNetY adds SE blocks but TIMM uses ReLU in SE gates.",
    ),

    # -------------------------------------------------------------------------
    # Transformers — GELU
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="vit_tiny_patch16_224",
        activations=["gelu"],
        family="transformer",
        has_exp_softmax=True,  # SDPA → manual
    ),
    ModelConfig(
        arch="vit_small_patch16_224",
        activations=["gelu"],
        family="transformer",
        has_exp_softmax=True,
    ),
    ModelConfig(
        arch="vit_base_patch16_224",
        activations=["gelu"],
        family="transformer",
        note="Standard transformer benchmark.",
        has_exp_softmax=True,
    ),
    ModelConfig(
        arch="swin_tiny_patch4_window7_224",
        activations=["gelu"],
        family="transformer",
        has_exp_softmax=True,  # F.softmax
    ),
    ModelConfig(
        arch="swinv2_large_window12to16_192to256",
        activations=["gelu"],
        family="transformer",
        note="Requires 256px input. Large version shows near-lossless accuracy.",
        skip=True,
        skip_reason="Non-standard input resolution (256px); handle separately.",
    ),
    ModelConfig(
        arch="deit_tiny_patch16_224",
        activations=["gelu"],
        family="transformer",
        has_exp_softmax=True,  # SDPA → manual
    ),
    ModelConfig(
        arch="deit_base_patch16_224",
        activations=["gelu"],
        family="transformer",
        has_exp_softmax=True,
    ),

    # -------------------------------------------------------------------------
    # Hybrids / Sensitive — SiLU
    # -------------------------------------------------------------------------
    ModelConfig(
        arch="mobilevit_s",
        activations=["silu"],
        family="hybrid",
        note="34× SiLU. Identified as most sensitive to PPA approximation.",
    ),
    ModelConfig(
        arch="halonet50ts",
        activations=["silu"],
        family="hybrid",
        note="51× SiLU.",
        has_exp_softmax=True,  # Tensor.softmax, intercepted
    ),
    ModelConfig(
        arch="lambda_resnet50ts",
        activations=["silu"],
        family="hybrid",
        note="51× SiLU. Requires more breakpoints (32) for accuracy.",
        has_exp_softmax=True,  # F.softmax
    ),
    ModelConfig(
        arch="sebotnet33ts_256",
        activations=["silu", "relu"],
        family="hybrid",
        note="34× SiLU + 6× ReLU. Achieves lossless accuracy with minimal breakpoints.",
        has_exp_softmax=True,  # Tensor.softmax, intercepted
    ),
    ModelConfig(
        arch="crossvit_base_240",
        activations=["gelu"],
        family="hybrid",
        note="27× GELU. Lossless accuracy with minimal breakpoints.",
        has_exp_softmax=True,  # Tensor.softmax, intercepted
    ),
    ModelConfig(
        arch="mixer_b16_224",
        activations=["gelu"],
        family="hybrid",
        note="24× GELU (MLP-Mixer).",
    ),
]


def get_active_models() -> List[ModelConfig]:
    """Return models that are not skipped."""
    return [m for m in MODELS if not m.skip]


def get_models_by_family(family: str) -> List[ModelConfig]:
    return [m for m in MODELS if m.family == family and not m.skip]


def get_models_by_activation(fn: str) -> List[ModelConfig]:
    return [m for m in MODELS if fn in m.activations and not m.skip]
