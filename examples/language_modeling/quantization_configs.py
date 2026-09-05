import logging
import re

import torch
from torchao.quantization.pt2e.quantizer.utils import annotate_output_qspec

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

# Plain integers and fp4 with the block scale a power of two (``fp8_e8m0``);
# the arms in ``POWER_OF_TWO_SCALE`` build their quantizer with that flag.
INT8_SPEC = f"int8,{BLOCKING},ax=-1"
INT8_VALUE_SPEC = f"int8,{BLOCKING},ax=-2"
INT4_SPEC = f"int4,{BLOCKING},ax=-1"
INT4_VALUE_SPEC = f"int4,{BLOCKING},ax=-2"
FP4_SPEC = f"fp4_e2m1,{BLOCKING},ax=-1"
FP4_VALUE_SPEC = f"fp4_e2m1,{BLOCKING},ax=-2"

# 8-bit integer activations beside 4-bit NormalFloat weights decoded to an
# 8-bit integer codebook.
MXINT8_ACT_SPEC = f"int8,{MICROSCALING},ax=-1"
MXINT8_VALUE_SPEC = f"int8,{MICROSCALING},ax=-2"
MXNF4_INT8_SPEC = f"nf4_8,{MICROSCALING},ax=-1"

QUANTIZATION_CONFIGS = {}

# Reference points.  ``bf16`` quantizes nothing; ``mxint8``, ``mxint4`` and
# ``mxfp4`` are the plain MX deployments with power-of-two block scales
# (``POWER_OF_TWO_SCALE``); ``mxnf4_int8`` keeps 4-bit weights through an
# 8-bit codebook under 8-bit activations.
QUANTIZATION_CONFIGS["bf16"] = {
    torch.nn.Linear: [None, None],
    torch.ops.aten.matmul.default: [None, None],
}
QUANTIZATION_CONFIGS["mxint8"] = {
    torch.nn.Linear: [INT8_SPEC, INT8_SPEC],
    torch.ops.aten.matmul.default: [INT8_SPEC, INT8_VALUE_SPEC],
}
QUANTIZATION_CONFIGS["mxint4"] = {
    torch.nn.Linear: [INT4_SPEC, INT4_SPEC],
    torch.ops.aten.matmul.default: [INT4_SPEC, INT4_VALUE_SPEC],
}
QUANTIZATION_CONFIGS["mxfp4"] = {
    torch.nn.Linear: [FP4_SPEC, FP4_SPEC],
    torch.ops.aten.matmul.default: [FP4_SPEC, FP4_VALUE_SPEC],
}
QUANTIZATION_CONFIGS["mxnf4_int8"] = {
    torch.nn.Linear: [MXINT8_ACT_SPEC, MXNF4_INT8_SPEC],
    torch.ops.aten.matmul.default: [MXINT8_ACT_SPEC, MXINT8_VALUE_SPEC],
}
POWER_OF_TWO_SCALE = {"mxint8", "mxint4", "mxfp4"}

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

# NF4 weights under int6 activations everywhere
QUANTIZATION_CONFIGS["mxnf4_int6"] = {
    torch.nn.Linear: [INT6_SPEC, MXNF4_SPEC],
    torch.ops.aten.matmul.default: [INT6_SPEC, INT6_VALUE_SPEC],
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


# KIVI's 2-bit KV cache: keys grouped along the sequence (per channel), values
# along the head dim (per token).  Only the completed chunks of a cache are
# quantized: ``split_kv_cache`` keeps the chunk being filled in a
# full-precision residual, and the compiler bakes the main cache quantized
# and quantizes each chunk as it completes (``quant_folding``).
# The group size, which is also the chunk the residual folds by: a
# residual length must be a multiple of it.  The scale and zero point are
# both stored as fp8_e4m3 (signed; the e5m3 scale format is unsigned and
# cannot hold a zero point).
KIVI_BLOCK_SIZE = 64
# KIVI's residual: how many positions of a cache stay in full precision
# while their chunk fills.
KIVI_RESIDUAL_LENGTH = 128
KIVI_KEY_SPEC = (
    f"uint2,bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,ax=-2,scale=fp8_e4m3"
)
KIVI_VALUE_SPEC = (
    f"uint2,bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,ax=-1,scale=fp8_e4m3"
)
# The attention matmuls under KIVI re-encode the main cache as int6
# microscaling for the MXU.  It is decoded and re-encoded in one fused
# dequantize, which needs the int6 scale constant across each affine block:
# 64x64 blocks cover both the key's 64-token and the value's 64-channel
# groups.  The residual is read in the cache's own dtype.
KIVI_QUERY_SPEC = (
    f"int6,qs=microscaling,bs={KIVI_BLOCK_SIZE},ax=-1,scale=fp8_e5m3"
)
KIVI_CACHE_MX_SPEC = (
    f"int6,qs=microscaling,bs={KIVI_BLOCK_SIZE},ax=(-2,-1),scale=fp8_e5m3"
)
_KV_CACHE = re.compile(r"^(key|value)_cache_\d+$")


def set_residual_attention_qconfig(quantizer):
    """Leave the residual halves of a split decode graph's attention
    unquantized.

    ``split_kv_cache`` leaves four matmuls per layer, in graph order the main
    and residual halves of ``q @ K^T`` then of ``P @ V``.  The main halves
    keep whatever the quantizer gives attention matmuls; the residual halves
    (orders 1 and 3) are not quantized at all, in either operand.

    Args:
        quantizer: The quantizer being configured; edited in place.
    """
    for order in (1, 3):
        quantizer.set_module_name_object_type_order(
            "self_attn", torch.ops.aten.matmul.default, order, None
        )


def set_kivi_attention_qconfig(quantizer):
    """Point the attention matmuls of a split decode graph at their KIVI
    operands: the main halves read an int6 query against the int6 re-encode
    of the 2-bit cache, the residual halves are not quantized.

    Args:
        quantizer: The quantizer being configured; edited in place.
    """
    query = QuantizationSpec.from_str(KIVI_QUERY_SPEC)
    cache_mx = QuantizationSpec.from_str(KIVI_CACHE_MX_SPEC)
    main = QuantizationConfig(query, None, cache_mx, None)
    for order in (0, 2):
        quantizer.set_module_name_object_type_order(
            "self_attn", torch.ops.aten.matmul.default, order, main
        )
    set_residual_attention_qconfig(quantizer)


def annotate_kivi_cache(gm):
    """Quantize the main KV caches of a split decode graph to KIVI's 2-bit
    layout.

    The observer sits on the main cache's read into the attention, so the
    whole buffer -- completed chunks and the zeros above them -- is
    quantized on the way in, and ``quant_folding`` can bake it and move the
    quantize onto each chunk's fold.  The residual buffers are not annotated.

    Args:
        gm: Decode graph from ``convert_and_export_with_cache``, after
            ``split_kv_cache``; annotated in place.

    Returns:
        The number of caches annotated.

    Raises:
        RuntimeError: A cache is still written directly, i.e. the graph was
            not split.
    """
    key_qspec = QuantizationSpec.from_str(KIVI_KEY_SPEC)
    value_qspec = QuantizationSpec.from_str(KIVI_VALUE_SPEC)
    count = 0
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        match = _KV_CACHE.match(str(node.target))
        if match is None:
            continue
        writes = [
            u
            for u in node.users
            if u.target is torch.ops.aten.index_copy_.default
        ]
        if writes:
            # The fold writes chunk ``c`` back through a ``where`` on the
            # completing-step predicate; a token written straight into the
            # cache means the graph was never split.
            if writes[0].args[3].target is not torch.ops.aten.where.self:
                raise RuntimeError(
                    f"{node.target} is written directly: split the graph "
                    "with split_kv_cache before annotating the caches"
                )
            continue  # the fold's handle on the buffer, not a read
        residual = gm.get_buffer(f"{node.target}_residual")
        if residual.shape[-2] % KIVI_BLOCK_SIZE:
            raise ValueError(
                f"{node.target}: a residual of {residual.shape[-2]} positions "
                f"is not a multiple of the {KIVI_BLOCK_SIZE}-token group"
            )
        annotate_output_qspec(
            node, key_qspec if match.group(1) == "key" else value_qspec
        )
        count += 1
    return count


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
