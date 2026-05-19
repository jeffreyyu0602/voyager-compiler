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
    precisions: Tuple[str, ...] = ()  # e.g. ('fp14', 'fp15', 'native', 'native') for mixed-prec templates

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
            # SegmentConfig(-10, -4.286, (-0.023, -0.197, -0.409)),
            SegmentConfig(-5.0, -4.286, (-0.023, -0.197, -0.409)),
            SegmentConfig(-4.286, -2.060, (0.023, 0.196, 0.432)),
            SegmentConfig(-2.060, -0.750, (0.124, 0.612, 0.860)),
            SegmentConfig(-0.750, 0.196, (0.382, 0.999, 1.005)),
            SegmentConfig(0.196, 0.1, (0.879, 0.804, 1.024)),
            # SegmentConfig(0.196, 1.0, (0.879, 0.804, 1.024)),
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
            SegmentConfig(-8.0, -3.495, (-0.028, -0.282, -0.750)),
            # SegmentConfig(-5.0, -3.495, (-0.028, -0.282, -0.750)),
            SegmentConfig(-3.495, -1.376, (0.002, -0.073, -0.384)),
            SegmentConfig(-1.376, 1.401, (0.210, 0.499, 0.010)),
            SegmentConfig(1.401, 3.992, (-0.003, 1.095, -0.407)),
            SegmentConfig(3.992, 8.0, (-0.041, 1.400, -1.017)),
            # SegmentConfig(3.992, 5.0, (-0.041, 1.400, -1.017)),
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
    ),

    AppTemplateConfig(
        tag="formal",
        name="elu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.21875, (0.006317138671875, 0.07958984375, -0.7578125)),
            SegmentConfig(-2.21875, -0.8515625, (0.08447265625, 0.494140625, -0.2197265625)),
            SegmentConfig(-0.8515625, 0.111328125, (0.36328125, 0.9765625, 0.0)),
            SegmentConfig(0.111328125, 8.0, (0.0, 1.0078125, 0.0)),
        ]
    ),

    AppTemplateConfig( #ULP 2, -3 (7 segments)
        tag="formal",
        name="exp",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-10.0, -4.1875, (0.000881195068359375, 0.0145263671875, 0.058837890625)),
            SegmentConfig(-4.1875, -2.90625, (0.01190185546875, 0.11328125, 0.28125)),
            SegmentConfig(-2.90625, -2.140625, (0.033203125, 0.251953125, 0.50390625)),
            SegmentConfig(-2.140625, -1.8359375, (-0.03271484375, 0.0, 0.267578125)),
            SegmentConfig(-1.8359375, -1.40625, (-0.000385284423828125, 0.189453125, 0.5078125)),
            SegmentConfig(-1.40625, -0.54296875, (0.1689453125, 0.7109375, 0.91015625)),
            SegmentConfig(-0.54296875, 0.1, (0.5, 1.0234375, 0.99609375)),
        ]
    ),

    # AppTemplateConfig(  # ULP scale2.0, -10 17 segments
    #     tag="formal",
    #     name="exp",
    #     template_name="quadratic_app_template",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-10.0, -8.5, (-2.2351741790771484e-08, 0.00010776519775390625, 0.00110626220703125)),
    #         SegmentConfig(-8.5, -7.75, (-1.9073486328125e-05, 1.6874484496081765e-21, 0.00156402587890625)),
    #         SegmentConfig(-7.75, -7.03125, (2.4557113647460938e-05, 0.000934600830078125, 0.006195068359375)),
    #         SegmentConfig(-7.03125, -6.5625, (4.234834705130197e-12, 0.00103759765625, 0.0081787109375)),
    #         SegmentConfig(-6.5625, -6.25, (-5.173683166503906e-05, 0.000865936279296875, 0.00933837890625)),
    #         SegmentConfig(-6.25, -5.8125, (-0.0002117156982421875, -9.202957153320312e-05, 0.00958251953125)),
    #         SegmentConfig(-5.8125, -5.46875, (-0.0003662109375, -0.000621795654296875, 0.01171875)),
    #         SegmentConfig(-5.46875, -5.0625, (-0.0003376007080078125, 0.00110626220703125, 0.0203857421875)),
    #         SegmentConfig(-5.0625, -4.59375, (-0.000400543212890625, 0.00372314453125, 0.035400390625)),
    #         SegmentConfig(-4.59375, -4.09375, (3.0279159545898438e-05, 0.01373291015625, 0.072265625)),
    #         SegmentConfig(-4.09375, -3.59375, (0.00125885009765625, 0.0306396484375, 0.12060546875)),
    #         SegmentConfig(-3.59375, -3.109375, (-0.00061798095703125, 0.03076171875, 0.1455078125)),
    #         SegmentConfig(-3.109375, -2.609375, (5.245208740234375e-05, 0.06005859375, 0.2294921875)),
    #         SegmentConfig(-2.609375, -2.140625, (0.00021839141845703125, 0.09326171875, 0.314453125)),
    #         SegmentConfig(-2.140625, -1.625, (0.0255126953125, 0.2490234375, 0.53125)),
    #         SegmentConfig(-1.625, -0.859375, (0.3046875, 0.59765625, 0.83984375)),
    #         SegmentConfig(-0.859375, 0.1, (0.3046875, 0.9296875, 0.99609375)),
    #     ]
    # ),

    # AppTemplateConfig( #ULP 1.5, -10
    #     tag="formal",
    #     name="exp",
    #     template_name="quadratic_app_template",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-10.0, -8.75, (-4.887580871582031e-06, 0.0, 0.000522613525390625)),
    #         SegmentConfig(-8.75, -8.0, (-0.0, 0.0002307891845703125, 0.002166748046875)),
    #         SegmentConfig(-8.0, -7.53125, (-2.7060508728027344e-05, 0.0, 0.0020599365234375)),
    #         SegmentConfig(-7.53125, -7.03125, (-4.506111145019531e-05, -2.3245811462402344e-05, 0.0029144287109375)),
    #         SegmentConfig(-7.03125, -6.5625, (-0.0, 0.00103759765625, 0.0081787109375)),
    #         SegmentConfig(-6.5625, -6.21875, (-7.963180541992188e-05, 0.000659942626953125, 0.0091552734375)),
    #         SegmentConfig(-6.21875, -5.78125, (-4.935264587402344e-05, 0.00183868408203125, 0.01531982421875)),
    #         SegmentConfig(-5.78125, -5.5625, (-0.000118255615234375, 0.0015869140625, 0.0162353515625)),
    #         SegmentConfig(-5.5625, -5.25, (-0.000949859619140625, -0.005950927734375, 9.1552734375e-05)),
    #         SegmentConfig(-5.25, -4.8125, (-0.0011749267578125, -0.00518798828125, 0.01025390625)),
    #         SegmentConfig(-4.8125, -4.53125, (-0.000553131103515625, 0.003936767578125, 0.039794921875)),
    #         SegmentConfig(-4.53125, -4.09375, (1.2934207916259766e-05, 0.0140380859375, 0.07373046875)),
    #         SegmentConfig(-4.09375, -3.84375, (-0.0023956298828125, 0.0, 0.056640625)),
    #         SegmentConfig(-3.84375, -3.59375, (-0.0030059814453125, -0.0, 0.06591796875)),
    #         SegmentConfig(-3.59375, -3.296875, (-0.0, 0.0303955078125, 0.13671875)),
    #         SegmentConfig(-3.296875, -3.03125, (-0.006591796875, 0.0, 0.1083984375)),
    #         SegmentConfig(-3.03125, -2.609375, (0.000507354736328125, 0.062255859375, 0.2314453125)),
    #         SegmentConfig(-2.609375, -2.296875, (-0.0, 0.08251953125, 0.2890625)),
    #         SegmentConfig(-2.296875, -1.90625, (0.00017547607421875, 0.1201171875, 0.375)),
    #         SegmentConfig(-1.90625, -1.4765625, (0.0228271484375, 0.26171875, 0.5625)),
    #         SegmentConfig(-1.4765625, -1.0625, (-0.0137939453125, 0.2451171875, 0.6171875)),
    #         SegmentConfig(-1.0625, -0.140625, (0.25, 0.859375, 0.9765625)),
    #         SegmentConfig(-0.140625, 0.1, (0.74609375, 1.046875, 0.99609375)),
    #     ]
    # ),

    AppTemplateConfig(
        tag="formal",
        name="gelu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.875, (-0.0003681182861328125, -0.004913330078125, -0.0147705078125)),
            SegmentConfig(-2.875, -2.21875, (-0.019287109375, -0.1318359375, -0.224609375)),
            SegmentConfig(-2.21875, -1.328125, (-0.040283203125, -0.25, -0.384765625)),
            SegmentConfig(-1.328125, -0.6171875, (0.1259765625, 0.1787109375, -0.10595703125)),
            SegmentConfig(-0.6171875, 0.8515625, (0.3671875, 0.498046875, 0.00154876708984375)),
            SegmentConfig(0.8515625, 8.0, (-0.0128173828125, 1.125, -0.265625)),
        ]
    ),

    AppTemplateConfig(
        tag="formal",
        name="mish",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.71875, (-0.003997802734375, -0.061767578125, -0.2421875)),
            SegmentConfig(-4.71875, -3.578125, (-0.00921630859375, -0.125, -0.42578125)),
            SegmentConfig(-3.578125, -2.625, (-0.006378173828125, -0.1259765625, -0.466796875)),
            SegmentConfig(-2.625, -1.15625, (0.03173828125, 0.0291748046875, -0.322265625)),
            SegmentConfig(-1.15625, -0.169921875, (0.255859375, 0.5546875, -0.00677490234375)),
            SegmentConfig(-0.169921875, 0.89453125, (0.28515625, 0.6015625, 0.0)),
            SegmentConfig(0.89453125, 8.0, (-0.0091552734375, 1.1015625, -0.2275390625)),
        ]
    ),

    AppTemplateConfig(
        tag="formal",
        name="relu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, 0.0, (0.0, 0.0, 0.0)),
            SegmentConfig(0.0, 100.0, (0.0, 1.0, 0.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal",
        name="sigmoid",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.84375, (0.0019378662109375, 0.0272216796875, 0.09521484375)),
            SegmentConfig(-3.84375, -2.828125, (0.0, 0.034423828125, 0.1513671875)),
            SegmentConfig(-2.828125, -1.5078125, (0.0361328125, 0.248046875, 0.470703125)),
            SegmentConfig(-1.5078125, 0.59375, (0.0272216796875, 0.25, 0.494140625)),
            SegmentConfig(0.59375, 4.71875, (-0.0294189453125, 0.234375, 0.5234375)),
            SegmentConfig(4.71875, 8.0, (0.0, 0.0, 1.0)),
        ]
    ),

    # AppTemplateConfig(  # previous formal silu
    #     tag="formal",
    #     name="silu",
    #     template_name="quadratic_app_template",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-8.0, -4.875, (-0.003936767578125, -0.060791015625, -0.23828125)),
    #         SegmentConfig(-4.875, -3.734375, (-0.0093994140625, -0.1259765625, -0.42578125)),
    #         SegmentConfig(-3.734375, -2.6875, (-0.007476806640625, -0.1259765625, -0.453125)),
    #         SegmentConfig(-2.6875, -1.1953125, (0.0322265625, 0.049072265625, -0.26953125)),
    #         SegmentConfig(-1.1953125, -0.169921875, (0.1923828125, 0.455078125, -0.0079345703125)),
    #         SegmentConfig(-0.169921875, 1.234375, (0.228515625, 0.50390625, 0.0)),
    #         SegmentConfig(1.234375, 8.0, (-0.000583648681640625, 1.0625, -0.36328125)),
    #     ]
    # ),

    AppTemplateConfig(
        tag="formal",
        name="silu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.75, (-0.004058837890625, -0.0625, -0.244140625)),
            SegmentConfig(-4.75, -3.65625, (-0.00177001953125, -0.06103515625, -0.2890625)),
            SegmentConfig(-3.65625, -2.5, (-0.00634765625, -0.123046875, -0.455078125)),
            SegmentConfig(-2.5, -1.2421875, (0.0341796875, 0.052734375, -0.26953125)),
            SegmentConfig(-1.2421875, -0.15625, (0.1923828125, 0.455078125, -0.00738525390625)),
            SegmentConfig(-0.15625, 1.34375, (0.2197265625, 0.5078125, -0.0)),
            SegmentConfig(1.34375, 20.0, (-0.002471923828125, 1.0703125, -0.357421875)),
        ]
    ),

    AppTemplateConfig(
        tag="formal",
        name="softplus",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.90625, (0.00189971923828125, 0.026611328125, 0.09326171875)),
            SegmentConfig(-3.90625, -2.890625, (0.00408935546875, 0.061279296875, 0.1953125)),
            SegmentConfig(-2.890625, -2.21875, (0.00799560546875, 0.1123046875, 0.310546875)),
            SegmentConfig(-2.21875, -1.53125, (0.0302734375, 0.2490234375, 0.50390625)),
            SegmentConfig(-1.53125, 2.21875, (0.1123046875, 0.498046875, 0.6953125)),
            SegmentConfig(2.21875, 8.0, (0.0, 0.96484375, 0.181640625)),
        ]
    ),

    AppTemplateConfig(
        tag="formal",
        name="tanh",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -1.875, (0.00201416015625, 0.0224609375, -0.93359375)),
            SegmentConfig(-1.875, -0.46875, (0.275390625, 1.0, -0.03466796875)),
            SegmentConfig(-0.46875, 0.169921875, (0.09716796875, 0.9921875, -0.00054931640625)),
            SegmentConfig(0.169921875, 1.65625, (-0.322265625, 1.09375, -0.008056640625)),
            SegmentConfig(1.65625, 6.6875, (-0.006988525390625, 0.06640625, 0.8515625)),
            SegmentConfig(6.6875, 8.0, (0.0, 0.0, 0.98828125)),
        ]
    ),
    # =========================================================================
    # formal_synth tag: dagpoly_add template (c1*x + c2) * (x + c3)
    # =========================================================================

    AppTemplateConfig(
        tag="formal_synth",
        name="elu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.234375, (0.006439208984375, 0.1201171875, -6.34375)),
            SegmentConfig(-2.234375, -0.5, (0.1484375, 0.703125, -0.1337890625)),
            SegmentConfig(-0.5, 0.1015625, (0.380859375, -0.0, 2.578125)),
            SegmentConfig(0.1015625, 8.0, (-2.7550648847397363e-40, 1.0078125, 0.0)),
        ]
    ),

    AppTemplateConfig(  # ULP scale2.0, 8 segments
        tag="formal_synth",
        name="exp",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-10.0, -4.09375, (0.00096893310546875, 0.0068359375, 9.3125)),
            SegmentConfig(-4.09375, -2.859375, (0.007568359375, 0.0390625, 6.03125)),
            SegmentConfig(-2.859375, -2.109375, (0.0205078125, 0.103515625, 4.09375)),
            SegmentConfig(-2.109375, -1.625, (-9.499490261077881e-07, -2.726912498474121e-06, -163840.0)),
            SegmentConfig(-1.625, -1.09375, (0.062255859375, 0.20703125, 3.46875)),
            SegmentConfig(-1.09375, -0.640625, (-0.0172119140625, 0.419921875, 1.84375)),
            SegmentConfig(-0.640625, -0.1005859375, (0.1513671875, 0.5, 1.9375)),
            SegmentConfig(-0.1005859375, 8.0, (1.3775324423698682e-39, 1.0, 1.0)),
        ]
    ),

    # AppTemplateConfig(  # ULP scale1.5, 18 segments
    #     tag="formal_synth",
    #     name="exp",
    #     template_name="quadratic_app_synth",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-10.0, -8.25, (3.170967102050781e-05, 0.0003414154052734375, 11.375)),
    #         SegmentConfig(-8.25, -7.53125, (1.2814998626708984e-05, 0.000469207763671875, 8.9375)),
    #         SegmentConfig(-7.53125, -7.03125, (-0.00011873245239257812, -0.000965118408203125, 0.296875)),
    #         SegmentConfig(-7.03125, -6.53125, (0.0002727508544921875, 0.002410888671875, 8.8125)),
    #         SegmentConfig(-6.53125, -6.0625, (0.0004367828369140625, 0.00372314453125, 8.1875)),
    #         SegmentConfig(-6.0625, -5.5625, (0.000518798828125, 0.00537109375, 7.09375)),
    #         SegmentConfig(-5.5625, -5.0625, (0.00125885009765625, 0.009521484375, 7.0625)),
    #         SegmentConfig(-5.0625, -4.59375, (0.00103759765625, 0.01171875, 6.03125)),
    #         SegmentConfig(-4.59375, -4.0625, (0.0032196044921875, 0.019287109375, 6.8125)),
    #         SegmentConfig(-4.0625, -3.59375, (0.00238037109375, 0.02783203125, 5.0)),
    #         SegmentConfig(-3.59375, -3.109375, (0.005706787109375, 0.046875, 4.625)),
    #         SegmentConfig(-3.109375, -2.609375, (0.01385498046875, 0.072265625, 4.625)),
    #         SegmentConfig(-2.609375, -2.109375, (0.0206298828125, 0.111328125, 3.875)),
    #         SegmentConfig(-2.109375, -1.609375, (0.036865234375, 0.166015625, 3.46875)),
    #         SegmentConfig(-1.609375, -1.0859375, (0.064453125, 0.21875, 3.328125)),
    #         SegmentConfig(-1.0859375, -0.63671875, (0.006439208984375, 0.01220703125, 65.0)),
    #         SegmentConfig(-0.63671875, -0.1484375, (0.1572265625, 0.486328125, 1.9921875)),
    #         SegmentConfig(-0.1484375, 0.1, (0.0, 0.98046875, 1.0234375)),
    #     ]
    # ),

    AppTemplateConfig(
        tag="formal_synth",
        name="gelu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.984375, (-1.0376153603865179e-20, -5.802175688691957e-20, 7.149464408450662e+16)),
            SegmentConfig(-2.984375, -1.96875, (-0.027587890625, -0.0927734375, 3.171875)),
            SegmentConfig(-1.96875, -0.984375, (-0.0019683837890625, -0.004669189453125, 60.5)),
            SegmentConfig(-0.984375, -0.220703125, (0.275390625, -0.0091552734375, 1.546875)),
            SegmentConfig(-0.220703125, 0.76171875, (0.375, 0.5, 0.0)),
            SegmentConfig(0.76171875, 8.0, (-0.01123046875, 1.1171875, -0.2353515625)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth",
        name="mish",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.96875, (-0.0025177001953125, -0.0218505859375, 8.4375)),
            SegmentConfig(-4.96875, -3.71875, (-0.0079345703125, -0.051513671875, 7.6875)),
            SegmentConfig(-3.71875, -1.75, (-0.0137939453125, -0.09521484375, 5.71875)),
            SegmentConfig(-1.75, -0.6328125, (0.1396484375, -0.034423828125, 2.734375)),
            SegmentConfig(-0.6328125, 0.765625, (0.30078125, 0.00048828125, 1.9765625)),
            SegmentConfig(0.765625, 8.0, (-0.004150390625, 0.000743865966796875, -256.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth",
        name="relu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, 0.0009765625, (-2.7550648847397363e-40, -2.407412430484045e-35, 15.9375)),
            SegmentConfig(0.0009765625, 100.0, (1.8417693326376007e-25, 2.1693674893577825e-29, 5.477945120128788e+24)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth",
        name="sigmoid",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.71875, (0.0019683837890625, 0.01544189453125, 6.40625)),
            SegmentConfig(-3.71875, -2.46875, (0.010498046875, 0.053955078125, 5.1875)),
            SegmentConfig(-2.46875, -1.546875, (0.0228271484375, 0.1015625, 4.15625)),
            SegmentConfig(-1.546875, 0.462890625, (0.031005859375, 0.1611328125, 3.078125)),
            SegmentConfig(0.462890625, 4.15625, (-0.033935546875, 0.30859375, 1.6484375)),
            SegmentConfig(4.15625, 8.0, (7.573064690121713e-28, 1.6478988765704848e-24, 6.044629098073146e+23)),
        ]
    ),

    # AppTemplateConfig(  # previous formal_synth silu
    #     tag="formal_synth",
    #     name="silu",
    #     template_name="quadratic_app_synth",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-8.0, -4.96875, (-0.0024871826171875, -0.0218505859375, 8.375)),
    #         SegmentConfig(-4.96875, -3.5, (-0.0086669921875, -0.058837890625, 7.03125)),
    #         SegmentConfig(-3.5, -1.7421875, (-0.00714111328125, -0.09326171875, 5.0)),
    #         SegmentConfig(-1.7421875, -0.7265625, (0.10595703125, -0.03076171875, 2.953125)),
    #         SegmentConfig(-0.7265625, 1.2734375, (0.2314453125, 0.498046875, 0.0029296875)),
    #         SegmentConfig(1.2734375, 8.0, (-0.000518798828125, 1.0625, -0.34765625)),
    #     ]
    # ),

    AppTemplateConfig(
        tag="formal_synth",
        name="silu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.96875, (-0.0024871826171875, -0.0218505859375, 8.375)),
            SegmentConfig(-4.96875, -3.5, (-0.0086669921875, -0.058837890625, 7.03125)),
            SegmentConfig(-3.5, -1.7421875, (-0.00714111328125, -0.09326171875, 5.0)),
            SegmentConfig(-1.7421875, -0.7265625, (0.10595703125, -0.03076171875, 2.953125)),
            SegmentConfig(-0.7265625, 1.2734375, (0.2314453125, 0.498046875, 0.0029296875)),
            # SegmentConfig(1.2734375, 20.0, (-0.0035552978515625, 0.00124359130859375, -304.0)),
            SegmentConfig(1.2734375, 100.0, (-0.0035552978515625, 0.00124359130859375, -304.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth",
        name="softplus",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.75, (0.00188446044921875, 0.0130615234375, 7.3125)),
            SegmentConfig(-3.75, -2.734375, (0.004486083984375, 0.048095703125, 4.4375)),
            SegmentConfig(-2.734375, -2.0, (0.013916015625, 0.10107421875, 3.703125)),
            SegmentConfig(-2.0, -1.2421875, (0.033935546875, 0.11328125, 4.75)),
            SegmentConfig(-1.2421875, -0.46484375, (0.049560546875, 0.2578125, 2.515625)),
            SegmentConfig(-0.46484375, 1.0859375, (0.09228515625, 0.2392578125, 2.921875)),
            SegmentConfig(1.0859375, 8.0, (0.017822265625, 0.796875, 0.58203125)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth",
        name="tanh",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -1.8203125, (0.0029449462890625, -0.0380859375, 24.0)),
            SegmentConfig(-1.8203125, -0.498046875, (0.287109375, 1.03125, -0.02587890625)),
            SegmentConfig(-0.498046875, 0.10888671875, (0.12451171875, 1.0, 0.0)),
            SegmentConfig(0.10888671875, 1.5625, (-0.337890625, 1.109375, -0.0087890625)),
            SegmentConfig(1.5625, 5.09375, (-0.0147705078125, 0.1787109375, 4.40625)),
            SegmentConfig(5.09375, 8.0, (0.0, -0.0079345703125, -132.0)),
        ]
    ),

    # =========================================================================
    # formal_synth_sol4 tag: dagpoly_add with mul1=fp14, add1=fp15
    # =========================================================================

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="elu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -2.53125, (0.004486083984375, 0.0966796875, -8.375)),
            SegmentConfig(-2.53125, -0.9609375, (0.08642578125, 0.5234375, -0.4609375)),
            SegmentConfig(-0.9609375, -0.0712890625, (0.3125, 0.93359375, -0.0057373046875)),
            SegmentConfig(-0.0712890625, 8.0, (0.0, 1.0078125, 0.00098419189453125)),
        ]
    ),

    # AppTemplateConfig(  # previous sol4 exp (ULP scale2.0, 8 segments)
    #     tag="formal_synth_sol4",
    #     name="exp",
    #     template_name="quadratic_app_synth_mp",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     precisions=('fp14', 'fp15', 'native', 'native'),
    #     segments=[
    #         SegmentConfig(-10.0, -4.0625, (0.0009918212890625, 0.0091552734375, 6.96875)),
    #         SegmentConfig(-4.0625, -2.859375, (0.008056640625, 0.04150390625, 5.875)),
    #         SegmentConfig(-2.859375, -2.296875, (1.8367099231598242e-40, 0.0771484375, 3.578125)),
    #         SegmentConfig(-2.296875, -1.8359375, (4.9591167925315254e-39, 0.125, 3.09375)),
    #         SegmentConfig(-1.8359375, -1.375, (2.1742607714259066e-12, 0.1982421875, 2.625)),
    #         SegmentConfig(-1.375, -0.87890625, (-0.00157928466796875, 0.328125, 2.125)),
    #         SegmentConfig(-0.87890625, -0.388671875, (0.00787353515625, 0.5078125, 1.703125)),
    #         SegmentConfig(-0.388671875, 0.1, (0.2080078125, 0.5859375, 1.7109375)),
    #     ]
    # ),

    AppTemplateConfig(  # ULP scale1.5, 23 segments
        tag="formal_synth_sol4",
        name="exp",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-10.0, -8.5, (2.753734588623047e-05, 0.0003185272216796875, 10.8125)),
            SegmentConfig(-8.5, -7.78125, (4.1425228118896484e-06, 0.0003185272216796875, 9.1875)),
            SegmentConfig(-7.78125, -7.25, (2.60770320892334e-06, 0.00054931640625, 8.5625)),
            SegmentConfig(-7.25, -6.8125, (2.7865171432495117e-06, 0.000885009765625, 8.0625)),
            SegmentConfig(-6.8125, -6.5, (-9.183549615799121e-41, 0.00124359130859375, 7.6875)),
            SegmentConfig(-6.5, -6.0625, (2.8014183044433594e-05, 0.0019989013671875, 7.3125)),
            SegmentConfig(-6.0625, -5.8125, (5.1021575927734375e-05, 0.002655029296875, 7.0625)),
            SegmentConfig(-5.8125, -5.4375, (2.7865171432495117e-06, 0.0035400390625, 6.65625)),
            SegmentConfig(-5.4375, -5.0625, (0.000331878662109375, 0.006591796875, 6.34375)),
            SegmentConfig(-5.0625, -4.78125, (8.296966552734375e-05, 0.007171630859375, 6.0)),
            SegmentConfig(-4.78125, -4.5, (0.000644683837890625, 0.01220703125, 5.6875)),
            SegmentConfig(-4.5, -4.09375, (0.000644683837890625, 0.0155029296875, 5.375)),
            SegmentConfig(-4.09375, -3.765625, (-0.000919342041015625, 0.0155029296875, 4.96875)),
            SegmentConfig(-3.765625, -3.34375, (-3.1948089599609375e-05, 0.028564453125, 4.5625)),
            SegmentConfig(-3.34375, -2.953125, (0.0, 0.043212890625, 4.15625)),
            SegmentConfig(-2.953125, -2.578125, (0.0, 0.0625, 3.78125)),
            SegmentConfig(-2.578125, -2.203125, (-0.0, 0.0908203125, 3.40625)),
            SegmentConfig(-2.203125, -1.7734375, (-0.0004482269287109375, 0.138671875, 2.984375)),
            SegmentConfig(-1.7734375, -1.359375, (0.000202178955078125, 0.205078125, 2.59375)),
            SegmentConfig(-1.359375, -0.90625, (0.064453125, 0.302734375, 2.53125)),
            SegmentConfig(-0.90625, -0.640625, (0.1044921875, 0.26171875, 3.328125)),
            SegmentConfig(-0.640625, -0.251953125, (-0.006561279296875, 0.6484375, 1.4375)),
            SegmentConfig(-0.251953125, 0.1, (-0.0194091796875, 0.9375, 1.0703125)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="gelu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -2.984375, (2.3877229001077715e-39, -1.2572470676959876e-29, 1.5474250491067253e+26)),
            SegmentConfig(-2.984375, -2.0, (-0.0267333984375, -0.0888671875, 3.234375)),
            SegmentConfig(-2.0, -1.2109375, (-0.037841796875, -0.1357421875, 2.75)),
            SegmentConfig(-1.2109375, -0.74609375, (0.11865234375, -0.06298828125, 1.875)),
            SegmentConfig(-0.74609375, -0.201171875, (0.3046875, 0.44921875, -0.020751953125)),
            SegmentConfig(-0.201171875, 0.76953125, (0.365234375, 0.498046875, 0.002960205078125)),
            SegmentConfig(0.76953125, 7.3125, (-0.01397705078125, 1.1171875, -0.2353515625)),
            SegmentConfig(7.3125, 8.0, (-3.981590270996094e-05, 0.9921875, -0.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="mish",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -5.21875, (-0.00193023681640625, -0.018798828125, 8.25)),
            SegmentConfig(-5.21875, -4.0, (-0.006683349609375, -0.0517578125, 6.84375)),
            SegmentConfig(-4.0, -3.0, (-0.007659912109375, -0.0849609375, 5.3125)),
            SegmentConfig(-3.0, -1.484375, (0.0027923583984375, -0.1044921875, 4.28125)),
            SegmentConfig(-1.484375, -0.7421875, (0.12255859375, 0.34375, -0.36328125)),
            SegmentConfig(-0.7421875, 0.0400390625, (0.287109375, 0.5859375, -0.00183868408203125)),
            SegmentConfig(0.0400390625, 1.140625, (0.2451171875, 0.625, -0.001739501953125)),
            SegmentConfig(1.140625, 5.125, (0.0, 1.046875, -0.1611328125)),
            SegmentConfig(5.125, 8.0, (1.011145580784821e-20, 0.9921875, 0.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="relu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, 0.0009765625, (1.1663108012064884e-38, -8.3570301503772e-39, 65536.0)),
            SegmentConfig(0.0009765625, 100.0, (-3.2877107624560854e-38, 0.99609375, -0.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="sigmoid",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.75, (0.00183868408203125, 0.011962890625, 7.875)),
            SegmentConfig(-3.75, -2.5625, (0.00994873046875, 0.0556640625, 4.90625)),
            SegmentConfig(-2.5625, -1.7734375, (0.01458740234375, 0.1015625, 3.65625)),
            SegmentConfig(-1.7734375, -0.77734375, (0.0311279296875, 0.138671875, 3.5)),
            SegmentConfig(-0.77734375, 1.28125, (-0.00118255615234375, 0.236328125, 2.109375)),
            SegmentConfig(1.28125, 4.4375, (-0.0247802734375, 0.255859375, 2.203125)),
            SegmentConfig(4.4375, 8.0, (-3.903258515484599e-24, -6.829203137237796e-21, -1.4757395258967641e+20)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="silu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -5.25, (-0.00174713134765625, -0.0147705078125, 9.6875)),
            SegmentConfig(-5.25, -3.96875, (-0.0030975341796875, -0.04541015625, 6.125)),
            SegmentConfig(-3.96875, -2.875, (-0.00958251953125, -0.0849609375, 5.5)),
            SegmentConfig(-2.875, -1.4921875, (0.01177978515625, -0.08740234375, 4.125)),
            SegmentConfig(-1.4921875, -0.66796875, (0.12060546875, 0.34375, -0.203125)),
            SegmentConfig(-0.66796875, 0.9453125, (0.2412109375, 0.498046875, -0.0)),
            SegmentConfig(0.9453125, 4.65625, (0.0169677734375, 0.9921875, -0.283203125)),
            SegmentConfig(4.65625, 20.0, (-0.0, 0.99609375, -0.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="softplus",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.75, (0.0019989013671875, 0.0145263671875, 6.8125)),
            SegmentConfig(-3.75, -2.734375, (0.006805419921875, 0.0498046875, 4.6875)),
            SegmentConfig(-2.734375, -2.0, (0.0147705078125, 0.1005859375, 3.75)),
            SegmentConfig(-2.0, -1.3203125, (0.0322265625, 0.15234375, 3.4375)),
            SegmentConfig(-1.3203125, -0.609375, (0.050537109375, 0.234375, 2.71875)),
            SegmentConfig(-0.609375, 0.451171875, (0.08447265625, 0.208984375, 3.328125)),
            SegmentConfig(0.451171875, 3.171875, (0.07177734375, 0.484375, 1.3671875)),
            SegmentConfig(3.171875, 8.0, (-0.00160980224609375, 1.015625, -0.0)),
        ]
    ),

    AppTemplateConfig(
        tag="formal_synth_sol4",
        name="tanh",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -1.796875, (0.0031890869140625, 0.07421875, -12.25)),
            SegmentConfig(-1.796875, -0.7421875, (0.2138671875, 0.8671875, -0.1611328125)),
            SegmentConfig(-0.7421875, -0.053955078125, (0.271484375, 1.0703125, 0.00445556640625)),
            SegmentConfig(-0.053955078125, 0.59765625, (-0.1787109375, 1.015625, -0.0)),
            SegmentConfig(0.59765625, 1.515625, (-0.306640625, 1.046875, 0.0244140625)),
            SegmentConfig(1.515625, 4.09375, (-0.0255126953125, 0.244140625, 2.9375)),
            SegmentConfig(4.09375, 8.0, (6.906634997391003e-26, -5.3998350387461647e-20, -1.8446744073709552e+19)),
        ]
    ),

    # =========================================================================
    # formal_quad_syn_sol7: quadratic_app_synth_mp (mul1=fp15, add1=fp14, add2=native, mul2=fp15)
    # =========================================================================

    AppTemplateConfig(  # 7 segs, from parallel_quad_syn_sol7_elu.txt
        tag="formal_quad_syn_sol7",
        name="elu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -2.671875, (-2.5331974029541016e-06, 0.00057220458984375, -1640.0)),
            SegmentConfig(-2.671875, -1.4921875, (0.00396728515625, 0.16015625, -3.65625)),
            SegmentConfig(-1.4921875, -0.86328125, (-0.00390625, 0.267578125, -1.3515625)),
            SegmentConfig(-0.86328125, -0.48046875, (-0.0, 0.515625, -0.26953125)),
            SegmentConfig(-0.48046875, -0.263671875, (-0.0, 0.68359375, -0.0771484375)),
            SegmentConfig(-0.263671875, 0.0390625, (0.453125, 1.0078125, 0.00048828125)),
            SegmentConfig(0.0390625, 8.0, (-1.3755815063409838e-18, 1.0, 0.0)),
        ]
    ),

    # AppTemplateConfig(  # 10 segs, from parallel_quad_syn_sol7_exp.txt
    #     tag="formal_quad_syn_sol7",
    #     name="exp",
    #     template_name="quadratic_app_synth_mp",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     precisions=('fp15', 'fp14', 'native', 'fp15'),
    #     segments=[
    #         SegmentConfig(-10.0, -4.09375, (0.000942230224609375, 0.00909423828125, 6.96875)),
    #         SegmentConfig(-4.09375, -2.890625, (0.00775146484375, 0.048583984375, 4.9375)),
    #         SegmentConfig(-2.890625, -2.359375, (-0.00616455078125, 0.06298828125, 3.5625)),
    #         SegmentConfig(-2.359375, -1.84375, (-0.000835418701171875, 0.1279296875, 3.0625)),
    #         SegmentConfig(-1.84375, -1.421875, (-0.00762939453125, 0.18359375, 2.640625)),
    #         SegmentConfig(-1.421875, -0.96875, (0.0751953125, 0.263671875, 2.921875)),
    #         SegmentConfig(-0.96875, -0.6328125, (0.109375, 0.29296875, 2.953125)),
    #         SegmentConfig(-0.6328125, -0.2734375, (0.1142578125, 0.52734375, 1.78125)),
    #         SegmentConfig(-0.2734375, 0.0157470703125, (0.051513671875, 0.06298828125, 15.9375)),
    #         SegmentConfig(0.0157470703125, 0.1, (-2.277520304718182e-37, 1.0, 1.0)),
    #     ]
    # ),

    AppTemplateConfig(  # ULP scale1.5, 28 segs, from parallel_quad_syn_sol7_scale1.5_exp.txt
        tag="formal_quad_syn_sol7",
        name="exp",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-10.0, -8.375, (1.2576580047607422e-05, 0.00020503997802734375, 10.4375)),
            SegmentConfig(-8.375, -7.6875, (-2.2351741790771484e-06, 0.0003261566162109375, 9.0)),
            SegmentConfig(-7.6875, -7.1875, (6.198883056640625e-05, 0.000957489013671875, 8.625)),
            SegmentConfig(-7.1875, -6.8125, (-6.580352783203125e-05, 0.0004863739013671875, 7.96875)),
            SegmentConfig(-6.8125, -6.46875, (-0.00019359588623046875, 0.00017833709716796875, 7.53125)),
            SegmentConfig(-6.46875, -6.0625, (2.181529998779297e-05, 0.001922607421875, 7.34375)),
            SegmentConfig(-6.0625, -5.6875, (-9.183549615799121e-41, 0.0027618408203125, 6.90625)),
            SegmentConfig(-5.6875, -5.3125, (0.0, 0.0038299560546875, 6.5625)),
            SegmentConfig(-5.3125, -4.875, (-0.00054931640625, 0.0035858154296875, 6.0625)),
            SegmentConfig(-4.875, -4.5, (0.00146484375, 0.01385498046875, 6.0)),
            SegmentConfig(-4.5, -4.09375, (-0.000408172607421875, 0.01214599609375, 5.28125)),
            SegmentConfig(-4.09375, -3.796875, (-0.0, 0.0194091796875, 4.9375)),
            SegmentConfig(-3.796875, -3.40625, (0.0002117156982421875, 0.0267333984375, 4.65625)),
            SegmentConfig(-3.40625, -3.109375, (0.005615234375, 0.047607421875, 4.5625)),
            SegmentConfig(-3.109375, -2.796875, (-0.0, 0.051025390625, 3.984375)),
            SegmentConfig(-2.796875, -2.46875, (0.00799560546875, 0.07958984375, 3.859375)),
            SegmentConfig(-2.46875, -2.109375, (-6.780028343200684e-07, 0.09912109375, 3.3125)),
            SegmentConfig(-2.109375, -1.890625, (0.0004634857177734375, 0.1435546875, 2.9375)),
            SegmentConfig(-1.890625, -1.609375, (-0.0120849609375, 0.1669921875, 2.671875)),
            SegmentConfig(-1.609375, -1.375, (-0.0185546875, 0.2177734375, 2.40625)),
            SegmentConfig(-1.375, -1.109375, (0.041748046875, 0.294921875, 2.421875)),
            SegmentConfig(-1.109375, -0.90234375, (-0.048828125, 0.341796875, 1.9375)),
            SegmentConfig(-0.90234375, -0.7109375, (-0.0615234375, -0.1015625, -7.78125)),
            SegmentConfig(-0.7109375, -0.484375, (0.00714111328125, 0.50390625, 1.7109375)),
            SegmentConfig(-0.484375, -0.34765625, (0.1669921875, 0.52734375, 1.84375)),
            SegmentConfig(-0.34765625, -0.1865234375, (4.4345855712890625e-05, 0.75390625, 1.2890625)),
            SegmentConfig(-0.1865234375, -0.046142578125, (0.255859375, 0.494140625, 2.046875)),
            SegmentConfig(-0.046142578125, 0.1, (5.693800761795455e-39, 1.0546875, 0.94140625)),
        ]
    ),

    AppTemplateConfig(  # 12 segs, from parallel_quad_syn_sol7_gelu.txt
        tag="formal_quad_syn_sol7",
        name="gelu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -2.984375, (-1.9879294811569497e-28, -1.0926725019580474e-19, 1.7873661021126656e+16)),
            SegmentConfig(-2.984375, -2.03125, (-0.025146484375, -0.08056640625, 3.4375)),
            SegmentConfig(-2.03125, -1.2421875, (-0.026123046875, -0.1357421875, 2.53125)),
            SegmentConfig(-1.2421875, -0.91796875, (-0.016357421875, -0.07080078125, 3.921875)),
            SegmentConfig(-0.91796875, -0.56640625, (0.0001430511474609375, -0.0029449462890625, 55.5)),
            SegmentConfig(-0.56640625, -0.33984375, (0.125, 0.294921875, -0.1572265625)),
            SegmentConfig(-0.33984375, 0.1337890625, (0.396484375, 0.50390625, -0.0009765625)),
            SegmentConfig(0.1337890625, 0.294921875, (2.765655517578125e-05, 0.6484375, -0.0220947265625)),
            SegmentConfig(0.294921875, 0.546875, (0.016357421875, 0.76171875, -0.055908203125)),
            SegmentConfig(0.546875, 0.78125, (0.0, 0.94140625, -0.134765625)),
            SegmentConfig(0.78125, 1.6328125, (1.8367099231598242e-40, 1.0859375, -0.228515625)),
            SegmentConfig(1.6328125, 8.0, (0.003875732421875, 0.984375, -0.03515625)),
        ]
    ),

    AppTemplateConfig(  # 13 segs, from parallel_quad_syn_sol7_mish.txt
        tag="formal_quad_syn_sol7",
        name="mish",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -5.0625, (-0.0023651123046875, -0.0208740234375, 8.4375)),
            SegmentConfig(-5.0625, -3.71875, (-0.00665283203125, -0.0595703125, 6.21875)),
            SegmentConfig(-3.71875, -2.796875, (-0.0137939453125, -0.0927734375, 5.78125)),
            SegmentConfig(-2.796875, -1.4921875, (0.0005950927734375, -0.10546875, 4.34375)),
            SegmentConfig(-1.4921875, -0.8671875, (0.0003719329833984375, -0.00885009765625, 34.0)),
            SegmentConfig(-0.8671875, -0.5390625, (-0.0030364990234375, 0.16015625, -0.9296875)),
            SegmentConfig(-0.5390625, -0.279296875, (0.25, -0.00408935546875, 2.21875)),
            SegmentConfig(-0.279296875, -0.11669921875, (-4.3713696171203817e-38, 0.470703125, -0.0269775390625)),
            SegmentConfig(-0.11669921875, 0.2080078125, (0.333984375, 0.58984375, 0.00048828125)),
            SegmentConfig(0.2080078125, 0.38671875, (-8.404155023538761e-22, 0.76171875, -0.02734375)),
            SegmentConfig(0.38671875, 0.69140625, (9.183549615799121e-41, 0.88671875, -0.076171875)),
            SegmentConfig(0.69140625, 1.765625, (0.08984375, 0.85546875, -0.08984375)),
            SegmentConfig(1.765625, 8.0, (0.003448486328125, 0.98046875, -0.03515625)),
        ]
    ),

    AppTemplateConfig(  # 2 segs, from parallel_quad_syn_sol7_relu.txt
        tag="formal_quad_syn_sol7",
        name="relu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, 0.001953125, (-2.6007757969005233e-30, 0.000244140625, 3.90625)),
            SegmentConfig(0.001953125, 100.0, (0.0, 0.9921875, 2.9802322387695312e-08)),
        ]
    ),

    AppTemplateConfig(  # 9 segs, from parallel_quad_syn_sol7_sigmoid.txt
        tag="formal_quad_syn_sol7",
        name="sigmoid",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -3.734375, (0.002105712890625, 0.0157470703125, 6.5)),
            SegmentConfig(-3.734375, -2.75, (0.00457763671875, 0.0458984375, 4.5)),
            SegmentConfig(-2.75, -2.0, (0.015625, 0.09033203125, 4.0)),
            SegmentConfig(-2.0, -1.71875, (-0.002532958984375, 0.11669921875, 2.96875)),
            SegmentConfig(-1.71875, -1.0390625, (0.0257568359375, 0.1474609375, 3.171875)),
            SegmentConfig(-1.0390625, -0.111328125, (0.004913330078125, 0.2197265625, 2.25)),
            SegmentConfig(-0.111328125, 1.109375, (-0.003326416015625, 0.2490234375, 1.9921875)),
            SegmentConfig(1.109375, 2.765625, (-0.0008544921875, 0.1376953125, 4.4375)),
            SegmentConfig(2.765625, 8.0, (3.0517578125e-05, 0.0037384033203125, 250.0)),
        ]
    ),

    AppTemplateConfig(  # 14 segs, from parallel_quad_syn_sol7_silu.txt
        tag="formal_quad_syn_sol7",
        name="silu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -5.0, (-0.002471923828125, -0.0216064453125, 8.4375)),
            SegmentConfig(-5.0, -3.75, (-0.007568359375, -0.058349609375, 6.5625)),
            SegmentConfig(-3.75, -2.71875, (-0.01080322265625, -0.08935546875, 5.46875)),
            SegmentConfig(-2.71875, -1.671875, (0.003570556640625, -0.09521484375, 4.3125)),
            SegmentConfig(-1.671875, -1.1171875, (0.008056640625, -0.049560546875, 5.90625)),
            SegmentConfig(-1.1171875, -0.7265625, (0.00677490234375, 0.11865234375, -1.390625)),
            SegmentConfig(-0.7265625, -0.3984375, (0.0147705078125, 0.251953125, -0.271484375)),
            SegmentConfig(-0.3984375, -0.248046875, (-9.183549615799121e-41, 0.341796875, -0.0712890625)),
            SegmentConfig(-0.248046875, 0.23046875, (0.224609375, 0.50390625, 0.00146484375)),
            SegmentConfig(0.23046875, 0.443359375, (0.01806640625, 0.6484375, -0.0361328125)),
            SegmentConfig(0.443359375, 0.84765625, (0.09765625, 0.66796875, -0.068359375)),
            SegmentConfig(0.84765625, 1.5234375, (1.8367099231598242e-40, 0.96484375, -0.240234375)),
            SegmentConfig(1.5234375, 4.03125, (6.183981895446777e-07, 1.0859375, -0.3828125)),
            SegmentConfig(4.03125, 100.0, (-0.0, 1.0078125, -0.05078125)),
        ]
    ),

    AppTemplateConfig(  # 11 segs, from parallel_quad_syn_sol7_softplus.txt
        tag="formal_quad_syn_sol7",
        name="softplus",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -3.734375, (0.002197265625, 0.0164794921875, 6.34375)),
            SegmentConfig(-3.734375, -2.71875, (0.007781982421875, 0.054931640625, 4.59375)),
            SegmentConfig(-2.71875, -2.03125, (0.01953125, 0.09130859375, 4.34375)),
            SegmentConfig(-2.03125, -1.609375, (-0.010986328125, 0.12109375, 2.890625)),
            SegmentConfig(-1.609375, -1.171875, (-0.0089111328125, 0.1943359375, 2.46875)),
            SegmentConfig(-1.171875, -0.734375, (0.021728515625, 0.259765625, 2.328125)),
            SegmentConfig(-0.734375, -0.416015625, (-0.0174560546875, 0.3828125, 1.71875)),
            SegmentConfig(-0.416015625, 0.072265625, (-0.01019287109375, 0.44921875, 1.5234375)),
            SegmentConfig(0.072265625, 0.78125, (0.06103515625, 0.443359375, 1.546875)),
            SegmentConfig(0.78125, 2.140625, (-0.00738525390625, 0.82421875, 0.6015625)),
            SegmentConfig(2.140625, 8.0, (0.00390625, 0.94921875, 0.15234375)),
        ]
    ),

    AppTemplateConfig(  # 9 segs, from parallel_quad_syn_sol7_tanh.txt
        tag="formal_quad_syn_sol7",
        name="tanh",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'fp15'),
        segments=[
            SegmentConfig(-8.0, -1.7265625, (-6.854534149169922e-06, 0.00125885009765625, -760.0)),
            SegmentConfig(-1.7265625, -0.984375, (0.125, -0.057373046875, 5.1875)),
            SegmentConfig(-0.984375, -0.55859375, (0.35546875, 1.1328125, 0.009765625)),
            SegmentConfig(-0.55859375, -0.1826171875, (0.216796875, 1.03125, -0.0009765625)),
            SegmentConfig(-0.1826171875, 0.26953125, (-0.0576171875, 0.99609375, 0.00048828125)),
            SegmentConfig(0.26953125, 0.5859375, (-0.21484375, 1.0234375, 0.0009765625)),
            SegmentConfig(0.5859375, 1.2890625, (-0.35546875, 1.1328125, -0.009765625)),
            SegmentConfig(1.2890625, 2.1875, (-1.8367099231598242e-40, 0.1279296875, 5.46875)),
            SegmentConfig(2.1875, 8.0, (-5.86781247029804e-24, -1.6609786700064796e-21, -5.810724383218509e+20)),
        ]
    ),

    # =========================================================================
    # formal_quad_syn_sol5: quadratic_app_synth_mp (mul1=fp15, add1=fp14, add2=native, mul2=native)
    # =========================================================================

    AppTemplateConfig(  # 6 segs, from parallel_quad_syn_sol5_elu.txt
        tag="formal_quad_syn_sol5",
        name="elu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -2.484375, (-4.00543212890625e-05, 0.008544921875, -106.5)),
            SegmentConfig(-2.484375, -1.265625, (0.00089263916015625, 0.1787109375, -2.796875)),
            SegmentConfig(-1.265625, -0.6953125, (0.00946044921875, 0.396484375, -0.60546875)),
            SegmentConfig(-0.6953125, -0.248046875, (0.287109375, 0.91015625, -0.0157470703125)),
            SegmentConfig(-0.248046875, 0.1416015625, (0.333984375, 0.97265625, -0.001007080078125)),
            SegmentConfig(0.1416015625, 8.0, (8.301928852682406e-38, 1.0078125, -3.8381180422460484e-21)),
        ]
    ),

    # AppTemplateConfig(  # 9 segs, from parallel_quad_syn_sol5_exp.txt
    #     tag="formal_quad_syn_sol5",
    #     name="exp",
    #     template_name="quadratic_app_synth_mp",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     precisions=('fp15', 'fp14', 'native', 'native'),
    #     segments=[
    #         SegmentConfig(-10.0, -4.09375, (0.000942230224609375, 0.00897216796875, 6.96875)),
    #         SegmentConfig(-4.09375, -3.09375, (-9.183549615799121e-41, 0.0281982421875, 4.625)),
    #         SegmentConfig(-3.09375, -2.359375, (0.0019378662109375, 0.06787109375, 3.796875)),
    #         SegmentConfig(-2.359375, -1.8671875, (-9.183549615799121e-41, 0.12158203125, 3.125)),
    #         SegmentConfig(-1.8671875, -1.375, (-0.00738525390625, 0.185546875, 2.625)),
    #         SegmentConfig(-1.375, -0.87890625, (-0.009033203125, 0.3203125, 2.125)),
    #         SegmentConfig(-0.87890625, -0.61328125, (-0.009765625, 0.54296875, 1.609375)),
    #         SegmentConfig(-0.61328125, -0.140625, (-0.014892578125, 0.70703125, 1.359375)),
    #         SegmentConfig(-0.140625, 0.1, (3.122406869371701e-39, 0.96484375, 1.03125)),
    #     ]
    # ),

    AppTemplateConfig(  # ULP scale1.5, 23 segs, from parallel_quad_syn_sol5_scale1.5_exp.txt
        tag="formal_quad_syn_sol5",
        name="exp",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-10.0, -8.5, (1.1801719665527344e-05, 0.00019741058349609375, 10.4375)),
            SegmentConfig(-8.5, -7.78125, (2.4139881134033203e-06, 0.00030517578125, 9.1875)),
            SegmentConfig(-7.78125, -7.28125, (-8.153915405273438e-05, -1.990795135498047e-05, 8.4375)),
            SegmentConfig(-7.28125, -6.8125, (7.808208465576172e-06, 0.000865936279296875, 8.125)),
            SegmentConfig(-6.8125, -6.5, (-0.00014400482177734375, 0.0002574920654296875, 7.71875)),
            SegmentConfig(-6.5, -6.0625, (1.704692840576172e-05, 0.0018768310546875, 7.34375)),
            SegmentConfig(-6.0625, -5.71875, (-1.9099388737231493e-11, 0.0027313232421875, 6.90625)),
            SegmentConfig(-5.71875, -5.3125, (2.8371810913085938e-05, 0.003997802734375, 6.5625)),
            SegmentConfig(-5.3125, -4.8125, (-0.000591278076171875, 0.0036163330078125, 6.03125)),
            SegmentConfig(-4.8125, -4.46875, (-1.8367099231598242e-40, 0.00958251953125, 5.65625)),
            SegmentConfig(-4.46875, -4.03125, (-0.0004024505615234375, 0.01275634765625, 5.25)),
            SegmentConfig(-4.03125, -3.59375, (-0.000911712646484375, 0.0196533203125, 4.78125)),
            SegmentConfig(-3.59375, -3.203125, (0.0030517578125, 0.039794921875, 4.53125)),
            SegmentConfig(-3.203125, -2.859375, (-0.002593994140625, 0.044189453125, 3.96875)),
            SegmentConfig(-2.859375, -2.484375, (0.007080078125, 0.07763671875, 3.84375)),
            SegmentConfig(-2.484375, -2.109375, (0.0, 0.10009765625, 3.3125)),
            SegmentConfig(-2.109375, -1.7734375, (9.183549615799121e-41, 0.1416015625, 2.96875)),
            SegmentConfig(-1.7734375, -1.34375, (-0.0057373046875, 0.201171875, 2.5625)),
            SegmentConfig(-1.34375, -1.046875, (-0.0, 0.294921875, 2.21875)),
            SegmentConfig(-1.046875, -0.734375, (-0.0038299560546875, 0.431640625, 1.8515625)),
            SegmentConfig(-0.734375, -0.404296875, (-2.7550648847397363e-40, 0.55859375, 1.578125)),
            SegmentConfig(-0.404296875, -0.09130859375, (1.8367099231598242e-40, 0.76171875, 1.2734375)),
            SegmentConfig(-0.09130859375, 0.1, (3.03057137321371e-39, 1.0078125, 0.99609375)),
        ]
    ),

    AppTemplateConfig(  # 10 segs, from parallel_quad_syn_sol5_gelu.txt
        tag="formal_quad_syn_sol5",
        name="gelu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.0, (4.2881042954748955e-21, 3.959879028413854e-20, -6.58651445502935e+16)),
            SegmentConfig(-3.0, -2.03125, (-0.025146484375, -0.08740234375, 3.140625)),
            SegmentConfig(-2.03125, -1.203125, (-0.03369140625, -0.1357421875, 2.65625)),
            SegmentConfig(-1.203125, -0.82421875, (0.001129150390625, -0.08349609375, 2.828125)),
            SegmentConfig(-0.82421875, -0.498046875, (0.25, -0.0184326171875, 1.5703125)),
            SegmentConfig(-0.498046875, -0.2734375, (0.01263427734375, 0.2255859375, -0.2158203125)),
            SegmentConfig(-0.2734375, 0.384765625, (0.37890625, 0.498046875, 0.001220703125)),
            SegmentConfig(0.384765625, 1.2578125, (0.2294921875, 0.65234375, -0.051513671875)),
            SegmentConfig(1.2578125, 3.328125, (-1.3775324423698682e-39, 1.0859375, -0.2373046875)),
            SegmentConfig(3.328125, 8.0, (8.265194654219209e-40, 0.9921875, -0.042724609375)),
        ]
    ),

    AppTemplateConfig(  # 10 segs, from parallel_quad_syn_sol5_mish.txt
        tag="formal_quad_syn_sol5",
        name="mish",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -5.09375, (-0.00238037109375, -0.0206298828125, 8.5625)),
            SegmentConfig(-5.09375, -3.8125, (-0.006927490234375, -0.056884765625, 6.5)),
            SegmentConfig(-3.8125, -2.96875, (-0.00030517578125, -0.07177734375, 5.0)),
            SegmentConfig(-2.96875, -1.484375, (-0.00042724609375, -0.10107421875, 4.46875)),
            SegmentConfig(-1.484375, -0.80859375, (0.1220703125, 0.34375, -0.36328125)),
            SegmentConfig(-0.80859375, -0.33203125, (0.25390625, 0.54296875, -0.025390625)),
            SegmentConfig(-0.33203125, 0.00970458984375, (0.26171875, 0.57421875, -0.0029449462890625)),
            SegmentConfig(0.00970458984375, 0.5703125, (0.279296875, 0.60546875, -0.000972747802734375)),
            SegmentConfig(0.5703125, 2.21875, (0.06787109375, 0.88671875, -0.0966796875)),
            SegmentConfig(2.21875, 8.0, (6.079673767089844e-06, 0.98828125, 0.021728515625)),
        ]
    ),

    AppTemplateConfig(  # 2 segs, from parallel_quad_syn_sol5_relu.txt
        tag="formal_quad_syn_sol5",
        name="relu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, 0.001953125, (-2.7550648847397363e-40, 0.0, 1.2379400392853803e+27)),
            SegmentConfig(0.001953125, 100.0, (2.8469003808977276e-39, 0.9921875, 3.457069396972656e-05)),
        ]
    ),

    AppTemplateConfig(  # 8 segs, from parallel_quad_syn_sol5_sigmoid.txt
        tag="formal_quad_syn_sol5",
        name="sigmoid",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.734375, (0.002227783203125, 0.0159912109375, 6.5)),
            SegmentConfig(-3.734375, -2.609375, (0.009033203125, 0.054931640625, 4.75)),
            SegmentConfig(-2.609375, -1.9609375, (0.00616455078125, 0.09033203125, 3.515625)),
            SegmentConfig(-1.9609375, -1.1796875, (0.00701904296875, 0.14453125, 2.875)),
            SegmentConfig(-1.1796875, -0.1513671875, (-0.00191497802734375, 0.2216796875, 2.203125)),
            SegmentConfig(-0.1513671875, 1.375, (-0.0018463134765625, 0.2333984375, 2.140625)),
            SegmentConfig(1.375, 3.703125, (-0.031005859375, 0.287109375, 1.875)),
            SegmentConfig(3.703125, 8.0, (6.428484731059385e-40, 3.933906555175781e-06, 249856.0)),
        ]
    ),

    AppTemplateConfig(  # 11 segs, from parallel_quad_syn_sol5_silu.txt
        tag="formal_quad_syn_sol5",
        name="silu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -5.0, (-0.002471923828125, -0.0213623046875, 8.4375)),
            SegmentConfig(-5.0, -3.703125, (-0.00634765625, -0.058837890625, 6.1875)),
            SegmentConfig(-3.703125, -2.65625, (-0.005126953125, -0.08740234375, 4.96875)),
            SegmentConfig(-2.65625, -1.4921875, (0.00167083740234375, -0.08935546875, 4.53125)),
            SegmentConfig(-1.4921875, -0.9140625, (-0.000179290771484375, 0.00457763671875, -55.0)),
            SegmentConfig(-0.9140625, -0.482421875, (0.125, -0.01483154296875, 2.9375)),
            SegmentConfig(-0.482421875, 0.283203125, (0.2451171875, 0.498046875, 0.0)),
            SegmentConfig(0.283203125, 0.73828125, (0.03759765625, 0.68359375, -0.05615234375)),
            SegmentConfig(0.73828125, 2.765625, (0.07958984375, 0.80078125, -0.1650390625)),
            SegmentConfig(2.765625, 25.75, (-0.002777099609375, 1.0625, -0.283203125)),
            SegmentConfig(25.75, 100.0, (0.0, 0.99609375, 0.0)),
        ]
    ),

    AppTemplateConfig(  # 9 segs, from parallel_quad_syn_sol5_softplus.txt
        tag="formal_quad_syn_sol5",
        name="softplus",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.734375, (0.002105712890625, 0.01361083984375, 7.4375)),
            SegmentConfig(-3.734375, -2.75, (0.008544921875, 0.052978515625, 4.8125)),
            SegmentConfig(-2.75, -2.0, (0.01953125, 0.09814453125, 4.125)),
            SegmentConfig(-2.0, -1.4375, (-0.00738525390625, 0.14453125, 2.78125)),
            SegmentConfig(-1.4375, -0.796875, (-0.002899169921875, 0.251953125, 2.265625)),
            SegmentConfig(-0.796875, -0.103515625, (0.0128173828125, 0.365234375, 1.828125)),
            SegmentConfig(-0.103515625, 0.8046875, (0.04833984375, 0.458984375, 1.5078125)),
            SegmentConfig(0.8046875, 2.71875, (0.003692626953125, 0.79296875, 0.65234375)),
            SegmentConfig(2.71875, 8.0, (-0.0, 1.0078125, 0.052734375)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_quad_syn_sol5_tanh.txt
        tag="formal_quad_syn_sol5",
        name="tanh",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp15', 'fp14', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -1.96875, (5.525307997912802e-25, 0.00555419921875, -172.0)),
            SegmentConfig(-1.96875, -1.171875, (0.1123046875, 0.58984375, -0.6171875)),
            SegmentConfig(-1.171875, -0.302734375, (0.35546875, 1.1171875, 0.01123046875)),
            SegmentConfig(-0.302734375, 0.28515625, (0.02685546875, 0.98046875, -0.0)),
            SegmentConfig(0.28515625, 0.984375, (-0.32421875, 1.0859375, -0.00494384765625)),
            SegmentConfig(0.984375, 1.8515625, (-0.1201171875, 0.62109375, 0.53125)),
            SegmentConfig(1.8515625, 8.0, (-2.1010387558846903e-22, -3.218725199566341e-20, -2.9831843931702166e+19)),
        ]
    ),

    # =========================================================================
    # formal_quad_syn_sol4: quadratic_app_synth_mp (mul1=fp14, add1=fp15, add2=native, mul2=native)
    # =========================================================================

    AppTemplateConfig(  # 4 segs, from parallel_quad_syn_sol4_elu.txt
        tag="formal_quad_syn_sol4",
        name="elu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -2.53125, (0.005340576171875, -0.039794921875, 19.875)),
            SegmentConfig(-2.53125, -0.8359375, (0.09716796875, 0.5625, -0.357421875)),
            SegmentConfig(-0.8359375, 0.001953125, (0.345703125, 0.9609375, -0.00194549560546875)),
            SegmentConfig(0.001953125, 8.0, (1.4870003537901937e-36, 1.0, -0.0)),
        ]
    ),

    # AppTemplateConfig(  # 8 segs, from parallel_quad_syn_sol4_exp.txt
    #     tag="formal_quad_syn_sol4",
    #     name="exp",
    #     template_name="quadratic_app_synth_mp",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     precisions=('fp14', 'fp15', 'native', 'native'),
    #     segments=[
    #         SegmentConfig(-10.0, -4.09375, (0.00092315673828125, 0.00872802734375, 7.0)),
    #         SegmentConfig(-4.09375, -2.859375, (0.008056640625, 0.046875, 5.1875)),
    #         SegmentConfig(-2.859375, -2.28125, (-0.0, 0.0771484375, 3.578125)),
    #         SegmentConfig(-2.28125, -1.8359375, (-0.0, 0.1259765625, 3.09375)),
    #         SegmentConfig(-1.8359375, -1.375, (-0.00051116943359375, 0.19921875, 2.625)),
    #         SegmentConfig(-1.375, -0.87890625, (-0.007415771484375, 0.318359375, 2.125)),
    #         SegmentConfig(-0.87890625, -0.37109375, (0.12451171875, 0.27734375, 3.328125)),
    #         SegmentConfig(-0.37109375, 0.1, (0.2197265625, 0.515625, 1.9453125)),
    #     ]
    # ),

    AppTemplateConfig(  # ULP scale1.5, 24 segs, from parallel_quad_syn_sol4_scale1.5_exp.txt
        tag="formal_quad_syn_sol4",
        name="exp",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-10.0, -8.5, (2.753734588623047e-05, 0.0003185272216796875, 10.8125)),
            SegmentConfig(-8.5, -7.78125, (4.1425228118896484e-06, 0.0003185272216796875, 9.1875)),
            SegmentConfig(-7.78125, -7.25, (2.60770320892334e-06, 0.00054931640625, 8.5625)),
            SegmentConfig(-7.25, -6.8125, (2.7865171432495117e-06, 0.000885009765625, 8.0625)),
            SegmentConfig(-6.8125, -6.5, (-9.183549615799121e-41, 0.00124359130859375, 7.6875)),
            SegmentConfig(-6.5, -6.0625, (2.8014183044433594e-05, 0.0019989013671875, 7.3125)),
            SegmentConfig(-6.0625, -5.8125, (5.1021575927734375e-05, 0.002655029296875, 7.0625)),
            SegmentConfig(-5.8125, -5.4375, (2.7865171432495117e-06, 0.0035400390625, 6.65625)),
            SegmentConfig(-5.4375, -5.0625, (0.000331878662109375, 0.006591796875, 6.34375)),
            SegmentConfig(-5.0625, -4.78125, (8.296966552734375e-05, 0.007171630859375, 6.0)),
            SegmentConfig(-4.78125, -4.5, (0.000644683837890625, 0.01220703125, 5.6875)),
            SegmentConfig(-4.5, -4.09375, (0.000644683837890625, 0.0155029296875, 5.375)),
            SegmentConfig(-4.09375, -3.765625, (-0.000919342041015625, 0.0155029296875, 4.96875)),
            SegmentConfig(-3.765625, -3.34375, (-3.1948089599609375e-05, 0.028564453125, 4.5625)),
            SegmentConfig(-3.34375, -2.953125, (0.0, 0.043212890625, 4.15625)),
            SegmentConfig(-2.953125, -2.578125, (0.0, 0.0625, 3.78125)),
            SegmentConfig(-2.578125, -2.203125, (-0.0, 0.0908203125, 3.40625)),
            SegmentConfig(-2.203125, -1.7734375, (-0.0004482269287109375, 0.138671875, 2.984375)),
            SegmentConfig(-1.7734375, -1.359375, (-0.0004253387451171875, 0.205078125, 2.59375)),
            SegmentConfig(-1.359375, -0.90625, (0.064453125, 0.3046875, 2.53125)),
            SegmentConfig(-0.90625, -0.6171875, (-0.095703125, 0.4609375, 1.640625)),
            SegmentConfig(-0.6171875, -0.341796875, (9.183549615799121e-41, 0.6171875, 1.484375)),
            SegmentConfig(-0.341796875, 0.016845703125, (-0.01336669921875, 0.85546875, 1.15625)),
            SegmentConfig(0.016845703125, 0.1, (9.09171411964113e-39, 0.9296875, 1.0859375)),
        ]
    ),

    AppTemplateConfig(  # 8 segs, from parallel_quad_syn_sol4_gelu.txt
        tag="formal_quad_syn_sol4",
        name="gelu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -2.984375, (2.3877229001077715e-39, -1.2572470676959876e-29, 1.5474250491067253e+26)),
            SegmentConfig(-2.984375, -2.0, (-0.0267333984375, -0.0888671875, 3.234375)),
            SegmentConfig(-2.0, -1.2109375, (-0.037841796875, -0.134765625, 2.78125)),
            SegmentConfig(-1.2109375, -0.62109375, (0.12255859375, -0.05810546875, 1.875)),
            SegmentConfig(-0.62109375, 0.0693359375, (0.349609375, 0.482421875, -0.00107574462890625)),
            SegmentConfig(0.0693359375, 0.83203125, (0.357421875, 0.5078125, 0.0009765625)),
            SegmentConfig(0.83203125, 7.3125, (-0.0159912109375, 1.140625, -0.2470703125)),
            SegmentConfig(7.3125, 8.0, (-0.00136566162109375, 1.0, -0.0)),
        ]
    ),

    AppTemplateConfig(  # 8 segs, from parallel_quad_syn_sol4_mish.txt
        tag="formal_quad_syn_sol4",
        name="mish",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -5.21875, (-0.00193023681640625, -0.018798828125, 8.25)),
            SegmentConfig(-5.21875, -4.0, (-0.006683349609375, -0.0517578125, 6.84375)),
            SegmentConfig(-4.0, -3.0, (-0.00872802734375, -0.0869140625, 5.375)),
            SegmentConfig(-3.0, -1.453125, (0.0008544921875, -0.103515625, 4.375)),
            SegmentConfig(-1.453125, -0.73046875, (0.12158203125, 0.341796875, -0.35546875)),
            SegmentConfig(-0.73046875, 0.0537109375, (0.287109375, 0.5859375, -0.00177001953125)),
            SegmentConfig(0.0537109375, 1.140625, (0.2451171875, 0.62890625, -0.001495361328125)),
            SegmentConfig(1.140625, 8.0, (-0.0091552734375, 1.0859375, -0.2001953125)),
        ]
    ),

    AppTemplateConfig(  # 2 segs, from parallel_quad_syn_sol4_relu.txt
        tag="formal_quad_syn_sol4",
        name="relu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, 0.0009765625, (1.1663108012064884e-38, -8.3570301503772e-39, 65536.0)),
            SegmentConfig(0.0009765625, 100.0, (-1.3183191012563879e-23, 9.183549615799121e-41, -7.555786372591432e+22)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_quad_syn_sol4_sigmoid.txt
        tag="formal_quad_syn_sol4",
        name="sigmoid",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.75, (0.0021820068359375, 0.01611328125, 6.46875)),
            SegmentConfig(-3.75, -2.6875, (0.00860595703125, 0.051513671875, 4.84375)),
            SegmentConfig(-2.6875, -1.9921875, (0.00023937225341796875, 0.080078125, 3.46875)),
            SegmentConfig(-1.9921875, -1.2421875, (0.0272216796875, 0.1240234375, 3.703125)),
            SegmentConfig(-1.2421875, 0.5703125, (0.0223388671875, 0.19140625, 2.59375)),
            SegmentConfig(0.5703125, 3.453125, (-0.041259765625, 0.333984375, 1.4765625)),
            SegmentConfig(3.453125, 8.0, (-2.4199485778808594e-05, 0.00787353515625, 123.0)),
        ]
    ),

    AppTemplateConfig(  # 8 segs, from parallel_quad_syn_sol4_silu.txt
        tag="formal_quad_syn_sol4",
        name="silu",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -5.25, (-0.00174713134765625, -0.0147705078125, 9.6875)),
            SegmentConfig(-5.25, -3.96875, (-0.0030975341796875, -0.04541015625, 6.125)),
            SegmentConfig(-3.96875, -2.875, (-0.00958251953125, -0.0849609375, 5.5)),
            SegmentConfig(-2.875, -1.4921875, (0.01177978515625, -0.08740234375, 4.125)),
            SegmentConfig(-1.4921875, -0.6796875, (0.12158203125, 0.34375, -0.2060546875)),
            SegmentConfig(-0.6796875, 0.9453125, (0.2431640625, 0.50390625, -0.0)),
            SegmentConfig(0.9453125, 4.5625, (0.0177001953125, 0.9921875, -0.28515625)),
            SegmentConfig(4.5625, 100.0, (-0.0, 0.99609375, -0.0)),
        ]
    ),

    AppTemplateConfig(  # 8 segs, from parallel_quad_syn_sol4_softplus.txt
        tag="formal_quad_syn_sol4",
        name="softplus",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -3.75, (0.0019989013671875, 0.0145263671875, 6.8125)),
            SegmentConfig(-3.75, -2.734375, (0.006805419921875, 0.0498046875, 4.6875)),
            SegmentConfig(-2.734375, -2.0, (0.0147705078125, 0.1005859375, 3.75)),
            SegmentConfig(-2.0, -1.34375, (0.032470703125, 0.15234375, 3.4375)),
            SegmentConfig(-1.34375, -0.640625, (0.052978515625, 0.216796875, 2.921875)),
            SegmentConfig(-0.640625, 0.255859375, (0.07958984375, 0.27734375, 2.5)),
            SegmentConfig(0.255859375, 2.90625, (0.07861328125, 0.453125, 1.4765625)),
            SegmentConfig(2.90625, 8.0, (9.183549615799121e-41, 0.984375, 0.08251953125)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_quad_syn_sol4_tanh.txt
        tag="formal_quad_syn_sol4",
        name="tanh",
        template_name="quadratic_app_synth_mp",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        precisions=('fp14', 'fp15', 'native', 'native'),
        segments=[
            SegmentConfig(-8.0, -1.765625, (0.0031890869140625, 0.07421875, -12.25)),
            SegmentConfig(-1.765625, -0.7421875, (0.2138671875, 0.8671875, -0.1611328125)),
            SegmentConfig(-0.7421875, -0.02001953125, (0.251953125, 1.046875, 0.002777099609375)),
            SegmentConfig(-0.02001953125, 0.65234375, (-0.2216796875, 1.0234375, -0.0)),
            SegmentConfig(0.65234375, 1.8515625, (-0.2490234375, 0.94921875, 0.08056640625)),
            SegmentConfig(1.8515625, 7.0625, (6.5267086029052734e-06, 0.00096893310546875, 988.0)),
            SegmentConfig(7.0625, 8.0, (-2.0384788513183594e-05, -0.00244140625, -396.0)),
        ]
    ),

    # =========================================================================
    # formal_quad_syn: quadratic_app_synth
    # =========================================================================

    AppTemplateConfig(  # 4 segs, from parallel_quad_syn_elu.txt
        tag="formal_quad_syn",
        name="elu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.234375, (0.006439208984375, 0.1201171875, -6.34375)),
            SegmentConfig(-2.234375, -0.5, (0.1484375, 0.703125, -0.1337890625)),
            SegmentConfig(-0.5, 0.11474609375, (0.375, 0.9765625, -0.0)),
            SegmentConfig(0.11474609375, 8.0, (-7.346839692639297e-40, 0.9921875, 0.0)),
        ]
    ),

    # AppTemplateConfig(  # 8 segs, from parallel_quad_syn_exp.txt
    #     tag="formal_quad_syn",
    #     name="exp",
    #     template_name="quadratic_app_synth",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-10.0, -4.09375, (0.00096893310546875, 0.0068359375, 9.3125)),
    #         SegmentConfig(-4.09375, -2.859375, (0.007568359375, 0.0390625, 6.03125)),
    #         SegmentConfig(-2.859375, -2.109375, (0.0159912109375, 0.06103515625, 6.46875)),
    #         SegmentConfig(-2.109375, -1.625, (0.037109375, 0.1318359375, 4.34375)),
    #         SegmentConfig(-1.625, -1.1171875, (0.045654296875, 0.125, 5.46875)),
    #         SegmentConfig(-1.1171875, -0.640625, (0.003326416015625, 0.00634765625, 123.5)),
    #         SegmentConfig(-0.640625, -0.1181640625, (0.1318359375, 0.234375, 4.125)),
    #         SegmentConfig(-0.1181640625, 0.1, (-0.0, 1.0703125, 0.9375)),
    #     ]
    # ),

    AppTemplateConfig(  # ULP scale1.5, 19 segs, from parallel_quad_syn_scale1.5_exp.txt
        tag="formal_quad_syn",
        name="exp",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-10.0, -8.25, (3.170967102050781e-05, 0.0003414154052734375, 11.375)),
            SegmentConfig(-8.25, -7.53125, (1.2814998626708984e-05, 0.000469207763671875, 8.9375)),
            SegmentConfig(-7.53125, -7.03125, (-0.00011873245239257812, -0.000965118408203125, 0.296875)),
            SegmentConfig(-7.03125, -6.53125, (0.0002727508544921875, 0.002410888671875, 8.8125)),
            SegmentConfig(-6.53125, -6.0625, (0.0004367828369140625, 0.00372314453125, 8.1875)),
            SegmentConfig(-6.0625, -5.5625, (0.000518798828125, 0.00537109375, 7.09375)),
            SegmentConfig(-5.5625, -5.0625, (0.00125885009765625, 0.009521484375, 7.0625)),
            SegmentConfig(-5.0625, -4.59375, (0.00103759765625, 0.01171875, 6.03125)),
            SegmentConfig(-4.59375, -4.0625, (0.0032196044921875, 0.019287109375, 6.8125)),
            SegmentConfig(-4.0625, -3.59375, (0.00238037109375, 0.02783203125, 5.0)),
            SegmentConfig(-3.59375, -3.109375, (0.005706787109375, 0.046875, 4.625)),
            SegmentConfig(-3.109375, -2.609375, (0.01385498046875, 0.072265625, 4.625)),
            SegmentConfig(-2.609375, -2.109375, (0.0206298828125, 0.111328125, 3.875)),
            SegmentConfig(-2.109375, -1.59375, (0.0311279296875, 0.1005859375, 5.53125)),
            SegmentConfig(-1.59375, -1.1171875, (0.06201171875, 0.2197265625, 3.265625)),
            SegmentConfig(-1.1171875, -0.8359375, (0.0, 0.376953125, 1.9765625)),
            SegmentConfig(-0.8359375, -0.392578125, (0.12890625, 0.302734375, 3.046875)),
            SegmentConfig(-0.392578125, 0.09375, (0.2138671875, 0.44140625, 2.265625)),
            SegmentConfig(0.09375, 0.1, (-0.0, -1.8341016046388524e-29, -5.942112188569825e+28)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_quad_syn_gelu.txt
        tag="formal_quad_syn",
        name="gelu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.984375, (-1.0376153603865179e-20, -5.802175688691957e-20, 7.149464408450662e+16)),
            SegmentConfig(-2.984375, -1.984375, (-0.02392578125, -0.07421875, 3.65625)),
            SegmentConfig(-1.984375, -0.99609375, (-0.005218505859375, -0.12109375, 2.390625)),
            SegmentConfig(-0.99609375, -0.2451171875, (0.267578125, -0.0108642578125, 1.5625)),
            SegmentConfig(-0.2451171875, 0.828125, (0.373046875, 0.5, 0.0)),
            SegmentConfig(0.828125, 8.0, (-0.01416015625, 1.1328125, -0.2451171875)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_quad_syn_mish.txt
        tag="formal_quad_syn",
        name="mish",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.96875, (-0.0025177001953125, -0.0218505859375, 8.4375)),
            SegmentConfig(-4.96875, -3.71875, (-0.0079345703125, -0.051513671875, 7.6875)),
            SegmentConfig(-3.71875, -1.75, (-0.0137939453125, -0.09521484375, 5.71875)),
            SegmentConfig(-1.75, -0.6328125, (0.1396484375, -0.034423828125, 2.734375)),
            SegmentConfig(-0.6328125, 0.8046875, (0.296875, 0.59375, 0.001953125)),
            SegmentConfig(0.8046875, 8.0, (-0.009765625, 1.1015625, -0.2041015625)),
        ]
    ),

    AppTemplateConfig(  # 2 segs, from parallel_quad_syn_relu.txt
        tag="formal_quad_syn",
        name="relu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, 0.0009765625, (-2.7550648847397363e-40, -2.407412430484045e-35, 15.9375)),
            SegmentConfig(0.0009765625, 100.0, (1.8417693326376007e-25, 2.1693674893577825e-29, 5.477945120128788e+24)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_quad_syn_sigmoid.txt
        tag="formal_quad_syn",
        name="sigmoid",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.71875, (0.0019683837890625, 0.01544189453125, 6.40625)),
            SegmentConfig(-3.71875, -2.484375, (0.00958251953125, 0.056884765625, 4.75)),
            SegmentConfig(-2.484375, -1.7421875, (0.0186767578125, 0.07080078125, 5.59375)),
            SegmentConfig(-1.7421875, -0.38671875, (0.03271484375, 0.1416015625, 3.484375)),
            SegmentConfig(-0.38671875, 2.5625, (-0.032958984375, 0.30859375, 1.6328125)),
            SegmentConfig(2.5625, 8.0, (-0.004241943359375, -0.0380859375, -21.75)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_quad_syn_silu.txt
        tag="formal_quad_syn",
        name="silu",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.96875, (-0.0024871826171875, -0.0218505859375, 8.375)),
            SegmentConfig(-4.96875, -3.5, (-0.0086669921875, -0.058837890625, 7.03125)),
            SegmentConfig(-3.5, -1.7265625, (-0.007598876953125, -0.0380859375, 12.375)),
            SegmentConfig(-1.7265625, -0.482421875, (0.138671875, 0.37890625, -0.11669921875)),
            SegmentConfig(-0.482421875, 1.328125, (0.228515625, 0.498046875, 0.002960205078125)),
            SegmentConfig(1.328125, 31.0, (-0.0023345947265625, 1.0703125, -0.349609375)),
            SegmentConfig(31.0, 100.0, (-9.276845958083868e-10, 1.0078125, -0.625)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_quad_syn_softplus.txt
        tag="formal_quad_syn",
        name="softplus",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.75, (0.00188446044921875, 0.0130615234375, 7.3125)),
            SegmentConfig(-3.75, -2.734375, (0.004486083984375, 0.048095703125, 4.4375)),
            SegmentConfig(-2.734375, -2.0, (0.013916015625, 0.10107421875, 3.703125)),
            SegmentConfig(-2.0, -1.2421875, (0.033935546875, 0.11328125, 4.75)),
            SegmentConfig(-1.2421875, -0.35546875, (0.06298828125, 0.1875, 3.53125)),
            SegmentConfig(-0.35546875, 2.890625, (0.09423828125, 0.224609375, 3.109375)),
            SegmentConfig(2.890625, 8.0, (-0.0, 1.0078125, 0.0)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_quad_syn_tanh.txt
        tag="formal_quad_syn",
        name="tanh",
        template_name="quadratic_app_synth",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -1.8203125, (0.003082275390625, -0.038818359375, 23.5)),
            SegmentConfig(-1.8203125, -0.44140625, (0.29296875, -0.00592041015625, 3.546875)),
            SegmentConfig(-0.44140625, 0.09375, (0.12451171875, -0.0, 8.0625)),
            SegmentConfig(0.09375, 1.6328125, (-0.3203125, 1.0859375, -0.0054931640625)),
            SegmentConfig(1.6328125, 5.5, (-0.010009765625, 0.1416015625, 5.875)),
            SegmentConfig(5.5, 8.0, (-9.183549615799121e-41, 1.1754943508222875e-37, 8.598443597733675e+36)),
        ]
    ),

    # =========================================================================
    # formal_std_quad: quadratic_app_template
    # =========================================================================

    AppTemplateConfig(  # 4 segs, from parallel_std_quad_elu.txt
        tag="formal_std_quad",
        name="elu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.234375, (0.00537109375, 0.0703125, -0.77734375)),
            SegmentConfig(-2.234375, -0.64453125, (0.12451171875, 0.61328125, -0.138671875)),
            SegmentConfig(-0.64453125, 0.1044921875, (0.375, 0.98046875, 0.0)),
            SegmentConfig(0.1044921875, 8.0, (-1.1938614500538858e-39, 0.98828125, 0.0)),
        ]
    ),

    # AppTemplateConfig(  # 6 segs, from parallel_std_quad_exp.txt
    #     tag="formal_std_quad",
    #     name="exp",
    #     template_name="quadratic_app_template",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-10.0, -4.09375, (0.000934600830078125, 0.01531982421875, 0.061767578125)),
    #         SegmentConfig(-4.09375, -2.9375, (0.00885009765625, 0.09326171875, 0.2490234375)),
    #         SegmentConfig(-2.9375, -2.109375, (0.033203125, 0.25, 0.5)),
    #         SegmentConfig(-2.109375, -1.5859375, (0.0245361328125, 0.2490234375, 0.53515625)),
    #         SegmentConfig(-1.5859375, -0.85546875, (0.1318359375, 0.6171875, 0.8515625)),
    #         SegmentConfig(-0.85546875, 0.1, (0.3125, 0.93359375, 1.0)),
    #     ]
    # ),

    AppTemplateConfig(  # 23 segs, from parallel_std_quad_exp15_exp.txt
        tag="formal_std_quad",
        name="exp",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-10.0, -8.75, (-4.887580871582031e-06, 0.0, 0.000522613525390625)),
            SegmentConfig(-8.75, -8.0, (-0.0, 0.0002307891845703125, 0.002166748046875)),
            SegmentConfig(-8.0, -7.53125, (-2.7060508728027344e-05, 0.0, 0.0020599365234375)),
            SegmentConfig(-7.53125, -7.03125, (-4.506111145019531e-05, -1.728534698486328e-05, 0.002960205078125)),
            SegmentConfig(-7.03125, -6.5625, (-0.0, 0.00103759765625, 0.0081787109375)),
            SegmentConfig(-6.5625, -6.3125, (-0.0001087188720703125, 0.00023555755615234375, 0.00762939453125)),
            SegmentConfig(-6.3125, -5.8125, (-0.00020313262939453125, -4.2438507080078125e-05, 0.00958251953125)),
            SegmentConfig(-5.8125, -5.5625, (-0.0002803802490234375, -8.335337042808533e-08, 0.012451171875)),
            SegmentConfig(-5.5625, -5.3125, (-0.00035858154296875, 0.0, 0.01495361328125)),
            SegmentConfig(-5.3125, -4.96875, (-0.00022411346435546875, 0.003570556640625, 0.0301513671875)),
            SegmentConfig(-4.96875, -4.59375, (-0.000789642333984375, 0.0, 0.0264892578125)),
            SegmentConfig(-4.59375, -4.21875, (-0.0, 0.0118408203125, 0.064453125)),
            SegmentConfig(-4.21875, -3.84375, (-0.0021514892578125, 0.0007171630859375, 0.0556640625)),
            SegmentConfig(-3.84375, -3.59375, (-0.0032501220703125, 0.0, 0.0693359375)),
            SegmentConfig(-3.59375, -3.34375, (0.0, 0.0301513671875, 0.1357421875)),
            SegmentConfig(-3.34375, -3.0625, (-0.00634765625, 0.0, 0.10595703125)),
            SegmentConfig(-3.0625, -2.765625, (-0.0047607421875, 0.0244140625, 0.166015625)),
            SegmentConfig(-2.765625, -2.359375, (-0.00054168701171875, 0.07421875, 0.271484375)),
            SegmentConfig(-2.359375, -1.9296875, (0.00020599365234375, 0.11865234375, 0.37109375)),
            SegmentConfig(-1.9296875, -1.6171875, (-0.0, 0.166015625, 0.46484375)),
            SegmentConfig(-1.6171875, -1.28125, (-0.01153564453125, 0.2001953125, 0.55078125)),
            SegmentConfig(-1.28125, -0.38671875, (0.20703125, 0.78515625, 0.9453125)),
            SegmentConfig(-0.38671875, 0.1, (0.58984375, 1.0390625, 0.99609375)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_std_quad_gelu.txt
        tag="formal_std_quad",
        name="gelu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.984375, (-0.0, 2.424457098570968e-38, -0.001953125)),
            SegmentConfig(-2.984375, -2.25, (-0.0155029296875, -0.111328125, -0.197265625)),
            SegmentConfig(-2.25, -1.2421875, (-0.040771484375, -0.25, -0.3828125)),
            SegmentConfig(-1.2421875, -0.498046875, (0.1650390625, 0.255859375, -0.0693359375)),
            SegmentConfig(-0.498046875, 0.7578125, (0.37890625, 0.498046875, 0.0)),
            SegmentConfig(0.7578125, 7.25, (-0.01470947265625, 1.1328125, -0.26953125)),
            SegmentConfig(7.25, 8.0, (0.00048828125, 1.03125, -0.25)),
        ]
    ),

    AppTemplateConfig(  # 7 segs, from parallel_std_quad_mish.txt
        tag="formal_std_quad",
        name="mish",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.75, (-0.004058837890625, -0.0625, -0.244140625)),
            SegmentConfig(-4.75, -3.625, (-0.00921630859375, -0.1240234375, -0.421875)),
            SegmentConfig(-3.625, -2.75, (-7.915496826171875e-05, -0.0849609375, -0.400390625)),
            SegmentConfig(-2.75, -1.25, (0.01953125, -0.0184326171875, -0.3671875)),
            SegmentConfig(-1.25, -0.2373046875, (0.2392578125, 0.53125, -0.01275634765625)),
            SegmentConfig(-0.2373046875, 1.015625, (0.275390625, 0.6015625, 0.001708984375)),
            SegmentConfig(1.015625, 8.0, (-0.0037841796875, 1.0625, -0.185546875)),
        ]
    ),

    AppTemplateConfig(  # 2 segs, from parallel_std_quad_relu.txt
        tag="formal_std_quad",
        name="relu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, 0.0009765625, (1.57160684466362e-08, 1.8189894035458565e-11, -8.344650268554688e-07)),
            SegmentConfig(0.0009765625, 100.0, (-4.81482486096809e-35, 0.98828125, 0.00096893310546875)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_std_quad_sigmoid.txt
        tag="formal_std_quad",
        name="sigmoid",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.703125, (0.0022430419921875, 0.031005859375, 0.1064453125)),
            SegmentConfig(-3.703125, -2.71875, (0.007354736328125, 0.08544921875, 0.23828125)),
            SegmentConfig(-2.71875, -1.4140625, (0.0361328125, 0.25, 0.474609375)),
            SegmentConfig(-1.4140625, 0.52734375, (0.031982421875, 0.255859375, 0.494140625)),
            SegmentConfig(0.52734375, 4.5, (-0.03125, 0.2421875, 0.515625)),
            SegmentConfig(4.5, 8.0, (6.934897101018578e-12, -2.3646862246096134e-10, 1.0)),
        ]
    ),

    AppTemplateConfig(  # 8 segs, from parallel_std_quad_silu.txt
        tag="formal_std_quad",
        name="silu",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.75, (-0.004058837890625, -0.0625, -0.244140625)),
            SegmentConfig(-4.75, -3.65625, (-0.00177001953125, -0.06103515625, -0.2890625)),
            SegmentConfig(-3.65625, -2.5, (-0.00634765625, -0.123046875, -0.455078125)),
            SegmentConfig(-2.5, -1.2265625, (0.040283203125, 0.07568359375, -0.25)),
            SegmentConfig(-1.2265625, -0.185546875, (0.189453125, 0.451171875, -0.0087890625)),
            SegmentConfig(-0.185546875, 1.296875, (0.2265625, 0.50390625, -1.1368683772161603e-13)),
            SegmentConfig(1.296875, 29.875, (-0.0023956298828125, 1.0703125, -0.357421875)),
            SegmentConfig(29.875, 100.0, (-0.0, 0.9921875, -0.1767578125)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_std_quad_softplus.txt
        tag="formal_std_quad",
        name="softplus",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -3.703125, (0.0022430419921875, 0.031005859375, 0.1064453125)),
            SegmentConfig(-3.703125, -2.703125, (0.0032806396484375, 0.0615234375, 0.205078125)),
            SegmentConfig(-2.703125, -2.0625, (-9.441375732421875e-05, 0.083984375, 0.291015625)),
            SegmentConfig(-2.0625, -1.4765625, (0.0303955078125, 0.2490234375, 0.50390625)),
            SegmentConfig(-1.4765625, 2.40625, (0.11083984375, 0.498046875, 0.69921875)),
            SegmentConfig(2.40625, 8.0, (-0.0, 0.9921875, 0.09228515625)),
        ]
    ),

    AppTemplateConfig(  # 6 segs, from parallel_std_quad_tanh.txt
        tag="formal_std_quad",
        name="tanh",
        template_name="quadratic_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -1.7265625, (0.00390625, 0.042724609375, -0.890625)),
            SegmentConfig(-1.7265625, -0.30859375, (0.30078125, 1.0546875, -0.00592041015625)),
            SegmentConfig(-0.30859375, 0.263671875, (0.015625, 0.984375, -2.938735877055719e-39)),
            SegmentConfig(0.263671875, 1.640625, (-0.3203125, 1.0859375, -0.0038909912109375)),
            SegmentConfig(1.640625, 6.5, (-0.00775146484375, 0.07177734375, 0.84375)),
            SegmentConfig(6.5, 8.0, (0.0, 0.0, 0.984375)),
        ]
    ),
    # =========================================================================
    # formal_linear: linear_app_template (b*x + c)
    # =========================================================================

    AppTemplateConfig(  # 8 segs, from parallel_linear_app_elu.txt
        tag="formal_linear",
        name="elu",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.921875, (0.0096435546875, -0.93359375)),
            SegmentConfig(-2.921875, -1.7109375, (0.095703125, -0.66796875)),
            SegmentConfig(-1.7109375, -1.0, (0.26171875, -0.37890625)),
            SegmentConfig(-1.0, -0.7109375, (0.375, -0.2490234375)),
            SegmentConfig(-0.7109375, -0.5, (0.498046875, -0.150390625)),
            SegmentConfig(-0.5, -0.2490234375, (0.69140625, -0.052001953125)),
            SegmentConfig(-0.2490234375, -0.0732421875, (0.84765625, -0.01031494140625)),
            SegmentConfig(-0.0732421875, 8.0, (0.9921875, 0.00048828125)),
        ]
    ),

    AppTemplateConfig(  # 11 segs, from parallel_linear_app_exp.txt
        tag="formal_linear",
        name="exp",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-10.0, -5.0625, (0.000827789306640625, 0.00860595703125)),
            SegmentConfig(-5.0625, -3.59375, (0.01446533203125, 0.07763671875)),
            SegmentConfig(-3.59375, -2.84375, (0.04052734375, 0.171875)),
            SegmentConfig(-2.84375, -2.28125, (0.0791015625, 0.28125)),
            SegmentConfig(-2.28125, -1.8359375, (0.1240234375, 0.384765625)),
            SegmentConfig(-1.8359375, -1.4765625, (0.185546875, 0.5)),
            SegmentConfig(-1.4765625, -1.1171875, (0.259765625, 0.61328125)),
            SegmentConfig(-1.1171875, -0.87890625, (0.365234375, 0.73046875)),
            SegmentConfig(-0.87890625, -0.5078125, (0.4765625, 0.8359375)),
            SegmentConfig(-0.5078125, -0.06640625, (0.74609375, 0.97265625)),
            SegmentConfig(-0.06640625, 0.10009765625, (0.9765625, 0.99609375)),
        ]
    ),

    AppTemplateConfig(  # 16 segs, from parallel_linear_app_gelu.txt
        tag="formal_linear",
        name="gelu",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -2.984375, (-0.0, -0.001953125)),
            SegmentConfig(-2.984375, -2.375, (-0.0250244140625, -0.078125)),
            SegmentConfig(-2.375, -1.9765625, (-0.0634765625, -0.1708984375)),
            SegmentConfig(-1.9765625, -1.453125, (-0.10986328125, -0.263671875)),
            SegmentConfig(-1.453125, -0.9765625, (-0.1171875, -0.27734375)),
            SegmentConfig(-0.9765625, -0.6953125, (-0.031494140625, -0.193359375)),
            SegmentConfig(-0.6953125, -0.484375, (0.076171875, -0.1171875)),
            SegmentConfig(-0.484375, -0.376953125, (0.1572265625, -0.07568359375)),
            SegmentConfig(-0.376953125, -0.2060546875, (0.271484375, -0.031982421875)),
            SegmentConfig(-0.2060546875, -0.03173828125, (0.400390625, -0.004638671875)),
            SegmentConfig(-0.03173828125, 0.10498046875, (0.5234375, -0.0)),
            SegmentConfig(0.10498046875, 0.265625, (0.625, -0.0081787109375)),
            SegmentConfig(0.265625, 0.51171875, (0.77734375, -0.047119140625)),
            SegmentConfig(0.51171875, 1.0, (0.9765625, -0.1494140625)),
            SegmentConfig(1.0, 3.359375, (1.0859375, -0.23828125)),
            SegmentConfig(3.359375, 8.0, (1.0, 0.0)),
        ]
    ),

    AppTemplateConfig(  # 17 segs, from parallel_linear_app_mish.txt
        tag="formal_linear",
        name="mish",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -5.75, (-0.006805419921875, -0.055419921875)),
            SegmentConfig(-5.75, -4.625, (-0.0238037109375, -0.1533203125)),
            SegmentConfig(-4.625, -3.78125, (-0.047119140625, -0.26171875)),
            SegmentConfig(-3.78125, -3.234375, (-0.06591796875, -0.3359375)),
            SegmentConfig(-3.234375, -2.265625, (-0.099609375, -0.4453125)),
            SegmentConfig(-2.265625, -1.484375, (-0.10107421875, -0.453125)),
            SegmentConfig(-1.484375, -0.98046875, (-0.0078125, -0.314453125)),
            SegmentConfig(-0.98046875, -0.66796875, (0.1259765625, -0.181640625)),
            SegmentConfig(-0.66796875, -0.41015625, (0.265625, -0.08642578125)),
            SegmentConfig(-0.41015625, -0.197265625, (0.412109375, -0.026123046875)),
            SegmentConfig(-0.197265625, 0.00390625, (0.53515625, -0.001678466796875)),
            SegmentConfig(0.00390625, 0.1484375, (0.6484375, -0.001953125)),
            SegmentConfig(0.1484375, 0.26953125, (0.703125, -0.0079345703125)),
            SegmentConfig(0.26953125, 0.51171875, (0.8046875, -0.03125)),
            SegmentConfig(0.51171875, 1.25, (0.984375, -0.1201171875)),
            SegmentConfig(1.25, 5.65625, (1.046875, -0.1748046875)),
            SegmentConfig(5.65625, 8.0, (1.0, 0.01171875)),
        ]
    ),

    AppTemplateConfig(  # 2 segs, from parallel_linear_app_relu.txt
        tag="formal_linear",
        name="relu",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, 0.001953125, (0.000125885009765625, 0.0)),
            SegmentConfig(0.001953125, 100.0, (1.0, -2.938735877055719e-39)),
        ]
    ),

    AppTemplateConfig(  # 10 segs, from parallel_linear_app_sigmoid.txt
        tag="formal_linear",
        name="sigmoid",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.625, (0.002838134765625, 0.0211181640625)),
            SegmentConfig(-4.625, -3.5, (0.0167236328125, 0.0859375)),
            SegmentConfig(-3.5, -2.703125, (0.041015625, 0.171875)),
            SegmentConfig(-2.703125, -2.140625, (0.07568359375, 0.265625)),
            SegmentConfig(-2.140625, -1.6484375, (0.10986328125, 0.33984375)),
            SegmentConfig(-1.6484375, -1.0625, (0.158203125, 0.421875)),
            SegmentConfig(-1.0625, -0.2392578125, (0.216796875, 0.486328125)),
            SegmentConfig(-0.2392578125, 1.2734375, (0.23046875, 0.5)),
            SegmentConfig(1.2734375, 2.796875, (0.10888671875, 0.65234375)),
            SegmentConfig(2.796875, 8.0, (0.0101318359375, 0.9296875)),
        ]
    ),

    # AppTemplateConfig(  # 18 segs, from parallel_linear_app_silu.txt
    #     tag="formal_linear",
    #     name="silu",
    #     template_name="linear_app_template",
    #     param_dtype=SUPPORTED_DTYPES["bf16"],
    #     segments=[
    #         SegmentConfig(-8.0, -5.75, (-0.006591796875, -0.05419921875)),
    #         SegmentConfig(-5.75, -4.71875, (-0.021728515625, -0.142578125)),
    #         SegmentConfig(-4.71875, -3.8125, (-0.04443359375, -0.25)),
    #         SegmentConfig(-3.8125, -3.15625, (-0.06689453125, -0.337890625)),
    #         SegmentConfig(-3.15625, -1.625, (-0.09375, -0.423828125)),
    #         SegmentConfig(-1.625, -1.25, (-0.046142578125, -0.33984375)),
    #         SegmentConfig(-1.25, -0.98046875, (0.01068115234375, -0.26171875)),
    #         SegmentConfig(-0.98046875, -0.7265625, (0.11181640625, -0.1591796875)),
    #         SegmentConfig(-0.7265625, -0.5, (0.19140625, -0.095703125)),
    #         SegmentConfig(-0.5, -0.244140625, (0.3203125, -0.0308837890625)),
    #         SegmentConfig(-0.244140625, -0.035400390625, (0.4296875, -0.004119873046875)),
    #         SegmentConfig(-0.035400390625, 0.1318359375, (0.51953125, -0.0)),
    #         SegmentConfig(0.1318359375, 0.267578125, (0.5703125, -0.0031890869140625)),
    #         SegmentConfig(0.267578125, 0.53125, (0.6796875, -0.030517578125)),
    #         SegmentConfig(0.53125, 1.03125, (0.83203125, -0.109375)),
    #         SegmentConfig(1.03125, 2.578125, (1.0390625, -0.318359375)),
    #         SegmentConfig(2.578125, 14.9375, (1.03125, -0.228515625)),
    #         SegmentConfig(14.9375, 20.0, (1.0078125, 0.0)),
    #     ]
    # ),

    AppTemplateConfig(  # 18 segs, from parallel_linear_app_silu_silu.txt
        tag="formal_linear",
        name="silu",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -5.75, (-0.006591796875, -0.05419921875)),
            SegmentConfig(-5.75, -4.71875, (-0.021728515625, -0.142578125)),
            SegmentConfig(-4.71875, -3.8125, (-0.04443359375, -0.25)),
            SegmentConfig(-3.8125, -3.15625, (-0.06689453125, -0.337890625)),
            SegmentConfig(-3.15625, -1.625, (-0.09375, -0.423828125)),
            SegmentConfig(-1.625, -1.25, (-0.046142578125, -0.33984375)),
            SegmentConfig(-1.25, -0.98046875, (0.01068115234375, -0.26171875)),
            SegmentConfig(-0.98046875, -0.7265625, (0.11181640625, -0.1591796875)),
            SegmentConfig(-0.7265625, -0.5, (0.19140625, -0.095703125)),
            SegmentConfig(-0.5, -0.244140625, (0.3203125, -0.0308837890625)),
            SegmentConfig(-0.244140625, -0.035400390625, (0.4296875, -0.004119873046875)),
            SegmentConfig(-0.035400390625, 0.1318359375, (0.51953125, -0.0)),
            SegmentConfig(0.1318359375, 0.267578125, (0.5703125, -0.0031890869140625)),
            SegmentConfig(0.267578125, 0.53125, (0.6796875, -0.030517578125)),
            SegmentConfig(0.53125, 1.03125, (0.83203125, -0.109375)),
            SegmentConfig(1.03125, 2.578125, (1.0390625, -0.318359375)),
            SegmentConfig(2.578125, 15.0, (1.03125, -0.228515625)),
            SegmentConfig(15.0, 100.0, (0.9921875, -0.0)),
        ]
    ),

    AppTemplateConfig(  # 12 segs, from parallel_linear_app_softplus.txt
        tag="formal_linear",
        name="softplus",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -4.75, (0.001922607421875, 0.015869140625)),
            SegmentConfig(-4.75, -3.5, (0.016357421875, 0.0849609375)),
            SegmentConfig(-3.5, -2.71875, (0.04296875, 0.1787109375)),
            SegmentConfig(-2.71875, -2.21875, (0.07666015625, 0.271484375)),
            SegmentConfig(-2.21875, -1.6953125, (0.12451171875, 0.376953125)),
            SegmentConfig(-1.6953125, -1.2421875, (0.1826171875, 0.4765625)),
            SegmentConfig(-1.2421875, -0.74609375, (0.26171875, 0.578125)),
            SegmentConfig(-0.74609375, -0.2431640625, (0.36328125, 0.66015625)),
            SegmentConfig(-0.2431640625, 0.384765625, (0.5, 0.69921875)),
            SegmentConfig(0.384765625, 1.390625, (0.69140625, 0.62890625)),
            SegmentConfig(1.390625, 4.25, (0.91015625, 0.330078125)),
            SegmentConfig(4.25, 8.0, (1.0078125, -0.00787353515625)),
        ]
    ),

    AppTemplateConfig(  # 11 segs, from parallel_linear_app_tanh.txt
        tag="formal_linear",
        name="tanh",
        template_name="linear_app_template",
        param_dtype=SUPPORTED_DTYPES["bf16"],
        segments=[
            SegmentConfig(-8.0, -1.96875, (0.0057373046875, -0.96484375)),
            SegmentConfig(-1.96875, -1.25, (0.1484375, -0.67578125)),
            SegmentConfig(-1.25, -0.8828125, (0.341796875, -0.416015625)),
            SegmentConfig(-0.8828125, -0.474609375, (0.6640625, -0.1328125)),
            SegmentConfig(-0.474609375, -0.2470703125, (0.8515625, -0.03515625)),
            SegmentConfig(-0.2470703125, 0.294921875, (0.98828125, 2.168404344971009e-19)),
            SegmentConfig(0.294921875, 0.5, (0.87109375, 0.032470703125)),
            SegmentConfig(0.5, 0.83984375, (0.6796875, 0.125)),
            SegmentConfig(0.83984375, 1.2890625, (0.408203125, 0.34375)),
            SegmentConfig(1.2890625, 2.0, (0.1630859375, 0.65234375)),
            SegmentConfig(2.0, 8.0, (0.0045166015625, 0.96875)),
        ]
    ),
    # === AUTO-GENERATED ENTRIES BELOW (from SMT solver logs) ===
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