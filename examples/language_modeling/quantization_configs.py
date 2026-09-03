import logging

import torch

from voyager_compiler import QuantizationSpec, QuantizationConfig
from voyager_compiler.quantization import FusedAmaxObsFakeQuantize

logger = logging.getLogger(__name__)


# Microscaling geometry every spec shares: 64 elements to a block, taken
# along the contraction axis.  The value operand of an attention matmul
# contracts along -2, so it blocks there.
BLOCKING = "qs=microscaling,bs=64"

# The same, with each block scale itself quantized to fp8.  A config
# named `_scale_bf16` is the one that leaves the scale alone.
MICROSCALING = f"{BLOCKING},scale=fp8_e5m3"

# MXNF4: 4-bit NormalFloat decoded to a 6-bit integer codebook.
MXNF4_SPEC = f"nf4_6,{MICROSCALING},ax=-1"
MXNF4_VALUE_SPEC = f"nf4_6,{MICROSCALING},ax=-2"

# Plain 6-bit integers, which the attention operands carry in the variants
# that quantize them separately from the linears.
INT6_SPEC = f"int6,{MICROSCALING},ax=-1"
INT6_VALUE_SPEC = f"int6,{MICROSCALING},ax=-2"

QUANTIZATION_CONFIGS = {}

QUANTIZATION_CONFIGS["mxnf4"] = {
    torch.nn.Linear: [MXNF4_SPEC, MXNF4_SPEC],
    torch.ops.aten.matmul.default: [MXNF4_SPEC, MXNF4_VALUE_SPEC],
}

# Attribution configs: quantize one side at a time, so the weight term, the
# activation term, and the interaction between them can be separated.
QUANTIZATION_CONFIGS["w4a16"] = {
    torch.nn.Linear: [None, MXNF4_SPEC],
    torch.ops.aten.matmul.default: [None, None],
}
QUANTIZATION_CONFIGS["w16a4"] = {
    torch.nn.Linear: [MXNF4_SPEC, None],
    torch.ops.aten.matmul.default: [MXNF4_SPEC, MXNF4_VALUE_SPEC],
}

# Attention operands at int6 rather than NormalFloat.  The linears here keep
# NormalFloat's float levels (`nf4`), not the int6 projection every other
# config deploys.
QUANTIZATION_CONFIGS["mxnf4_attn_int6"] = {
    torch.nn.Linear: [MXNF4_SPEC, MXNF4_SPEC],
    torch.ops.aten.matmul.default: [INT6_SPEC, INT6_VALUE_SPEC],
}

# ... and with `lm_head` reading an int6 activation as well.
QUANTIZATION_CONFIGS["mxnf4_attn_head_int6"] = {
    torch.nn.Linear: [MXNF4_SPEC, MXNF4_SPEC],
    torch.ops.aten.matmul.default: [INT6_SPEC, INT6_VALUE_SPEC],
    ("lm_head", torch.ops.aten.linear.default, 0): [INT6_SPEC, MXNF4_SPEC],
}

QUANTIZATION_CONFIGS["mxint6"] = {
    torch.nn.Linear: [INT6_SPEC, MXNF4_SPEC],
    torch.ops.aten.matmul.default: [INT6_SPEC, INT6_VALUE_SPEC],
    ("lm_head", torch.ops.aten.linear.default, 0): [INT6_SPEC, MXNF4_SPEC],
}

# Outlier filtering on the linears only: each activation sets aside its
# largest 1% before quantizing.  Attention and the `lm_head` activation stay
# dense at int6, as in `mxnf4_attn_head_int6`, which is this config's dense
# twin.
QUANTIZATION_CONFIGS["mxnf4_outlier"] = {
    torch.nn.Linear: [f"{MXNF4_SPEC},opct=0.01", MXNF4_SPEC],
    torch.ops.aten.matmul.default: [INT6_SPEC, INT6_VALUE_SPEC],
    ("lm_head", torch.ops.aten.linear.default, 0): [INT6_SPEC, MXNF4_SPEC],
}

# The same linears, plus the attention key and value: the matmuls' second
# operand sets aside its largest 1% too, with the attention operands kept at
# NormalFloat.  A side-stream on the column operand makes the lowering swap
# each matmul so the CSR lands on the row side, transposing the scores.
QUANTIZATION_CONFIGS["mxnf4_outlier_kv"] = {
    torch.nn.Linear: [f"{MXNF4_SPEC},opct=0.01", MXNF4_SPEC],
    torch.ops.aten.matmul.default: [
        MXNF4_SPEC,
        f"{MXNF4_VALUE_SPEC},opct=0.01",
    ],
    ("lm_head", torch.ops.aten.linear.default, 0): [INT6_SPEC, MXNF4_SPEC],
}

# The attention side-stream on the row operand instead: Q carries it on the
# first matmul and P @ V has none, so no matmul is swapped.
QUANTIZATION_CONFIGS["mxnf4_outlier_q"] = {
    **QUANTIZATION_CONFIGS["mxnf4_outlier_kv"],
    ("self_attn", torch.ops.aten.matmul.default, 0): [
        f"{MXNF4_SPEC},opct=0.01",
        MXNF4_VALUE_SPEC,
    ],
    ("self_attn", torch.ops.aten.matmul.default, 1): [
        MXNF4_SPEC,
        MXNF4_VALUE_SPEC,
    ],
}


def set_qconfig(quantizer, qconfigs, force_scale_power_of_two=False):
    def make_qspec(spec):
        if spec is None:
            return None
        quant_spec = QuantizationSpec.from_str(spec)
        quant_spec.observer_or_fake_quant_ctr = (
            FusedAmaxObsFakeQuantize.with_args(
                force_scale_power_of_two=force_scale_power_of_two,
            )
        )
        return quant_spec

    for key, qspec in qconfigs.items():
        if qspec is None:
            qconfig = None
        elif isinstance(qspec, str):
            quant_spec = make_qspec(qspec)
            qconfig = QuantizationConfig(quant_spec, None, quant_spec, None)
        else:
            num_specs = len(qspec)

            if num_specs not in (2, 3):
                raise ValueError(f"Invalid qspec: {qspec}")

            activation = make_qspec(qspec[0])
            weight = make_qspec(qspec[1])
            bias = make_qspec(qspec[2]) if num_specs == 3 else None

            qconfig = QuantizationConfig(activation, None, weight, bias)

        if isinstance(key, tuple):
            logger.info(
                f"Setting qconfig for module name, object type and order: {key}"
            )
            quantizer.set_module_name_object_type_order(*key, qconfig)
        elif isinstance(key, str):
            logger.info(f"Setting qconfig for module name: {key}")
            quantizer.set_module_name(key, qconfig)
        elif isinstance(key, type) and issubclass(key, torch.nn.Module):
            logger.info(f"Setting qconfig for module type: {key}")
            quantizer.set_module_type(key, qconfig)
        elif isinstance(key, torch._ops.OpOverload):
            logger.info(f"Setting qconfig for op overload: {key}")
            quantizer.set_object_type(key, qconfig)
        else:
            raise ValueError(f"Invalid module name or type: {key}")

    return quantizer
