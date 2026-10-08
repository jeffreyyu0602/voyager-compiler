import logging
import re

import torch
from torchao.quantization.pt2e.quantizer.utils import annotate_output_qspec

from voyager_compiler import QuantizationSpec, QuantizationConfig

logger = logging.getLogger(__name__)


def mx_spec(dtype, scale, axis, block_size=64):
    """The spec of ``dtype`` in 64-element microscaling blocks.

    Args:
        dtype: Element dtype, e.g. ``int8`` or ``lut4_to_int6``.
        scale: How each block's scale is stored: ``MX_POT_SCALE``,
            ``MX_E5M3_SCALE`` or ``MX_E4M3_GLOBAL_SCALE``.
        axis: Axis the blocks run along, the operand's contraction axis.

    Returns:
        The spec string.
    """
    return f"{dtype},qs=microscaling,bs={block_size},{scale},ax={axis}"


def uniform_config(dtype, scale):
    """A config quantizing every linear and matmul operand to ``dtype``.

    A matmul's right-hand operand contracts along -2, so it blocks there.

    Args:
        dtype: Element dtype of every operand.
        scale: How each block's scale is stored.

    Returns:
        The config, as ``QUANTIZATION_CONFIGS`` holds it.
    """
    lhs, rhs = mx_spec(dtype, scale, -1), mx_spec(dtype, scale, -2)
    return {
        torch.nn.Linear: [lhs, lhs],
        torch.ops.aten.matmul.default: [lhs, rhs],
    }


MX_POT_SCALE = "power_2_scale=1"
MX_E5M3_SCALE = "scale=fp8_e5m3"
MX_E4M3_GLOBAL_SCALE = "scale=fp8_e4m3,global_scale=1"
NVFP4_SPEC = mx_spec("fp4_e2m1", MX_E4M3_GLOBAL_SCALE, -1, block_size=16)
NVFP4_RHS_SPEC = mx_spec("fp4_e2m1", MX_E4M3_GLOBAL_SCALE, -2, block_size=16)
MXINT6_E5M3_SPEC = mx_spec("int6", MX_E5M3_SCALE, -1)
MXINT6_E5M3_RHS_SPEC = mx_spec("int6", MX_E5M3_SCALE, -2)
MXLUT4_INT6_E5M3_SPEC = mx_spec("lut4_to_int6", MX_E5M3_SCALE, -1)
MXLUT4_INT6_E5M3_RHS_SPEC = mx_spec("lut4_to_int6", MX_E5M3_SCALE, -2)

QUANTIZATION_CONFIGS = {}

QUANTIZATION_CONFIGS["bf16"] = {
    torch.nn.Linear: [None, None],
    torch.ops.aten.matmul.default: [None, None],
}

QUANTIZATION_CONFIGS["mxint8_pot"] = uniform_config("int8", MX_POT_SCALE)
QUANTIZATION_CONFIGS["mxint4_pot"] = uniform_config("int4", MX_POT_SCALE)
QUANTIZATION_CONFIGS["mxfp4_pot"] = uniform_config("fp4_e2m1", MX_POT_SCALE)
QUANTIZATION_CONFIGS["mxlut4_int6_pot"] = uniform_config(
    "lut4_to_int6", MX_POT_SCALE
)
QUANTIZATION_CONFIGS["nvfp4"] = {
    torch.nn.Linear: [NVFP4_SPEC, NVFP4_SPEC],
    torch.ops.aten.matmul.default: [NVFP4_SPEC, NVFP4_RHS_SPEC],
}

QUANTIZATION_CONFIGS["mxlut4_int6_e5m3"] = {
    **uniform_config("lut4_to_int6", MX_E5M3_SCALE),
    # Flash attention (``--attn_implementation sdpa``): the specs the two
    # attention matmuls take, on the one node that stands for both.
    torch.ops.aten.scaled_dot_product_attention.default: [
        MXLUT4_INT6_E5M3_SPEC,
        MXLUT4_INT6_E5M3_RHS_SPEC,
    ],
}

# Attribution configs: quantize one side at a time, so the weight term, the
# activation term, and the interaction between them can be separated.
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_w_only"] = {
    torch.nn.Linear: [None, MXLUT4_INT6_E5M3_SPEC],
    torch.ops.aten.matmul.default: [None, None],
}
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_a_only"] = {
    torch.nn.Linear: [MXLUT4_INT6_E5M3_SPEC, None],
    torch.ops.aten.matmul.default: [
        MXLUT4_INT6_E5M3_SPEC,
        MXLUT4_INT6_E5M3_RHS_SPEC,
    ],
}

# Attention operands at plain int6 rather than through the lookup table.
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_attn_int6"] = {
    torch.nn.Linear: [MXLUT4_INT6_E5M3_SPEC, MXLUT4_INT6_E5M3_SPEC],
    torch.ops.aten.matmul.default: [MXINT6_E5M3_SPEC, MXINT6_E5M3_RHS_SPEC],
}

# ... and with `lm_head` reading an int6 activation as well.
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_attn_head_int6"] = {
    torch.nn.Linear: [MXLUT4_INT6_E5M3_SPEC, MXLUT4_INT6_E5M3_SPEC],
    torch.ops.aten.matmul.default: [MXINT6_E5M3_SPEC, MXINT6_E5M3_RHS_SPEC],
    ("lm_head", torch.ops.aten.linear.default, 0): [
        MXINT6_E5M3_SPEC,
        MXLUT4_INT6_E5M3_SPEC,
    ],
}

# NF4 weights under int6 activations everywhere
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_a_int6"] = {
    torch.nn.Linear: [MXINT6_E5M3_SPEC, MXLUT4_INT6_E5M3_SPEC],
    torch.ops.aten.matmul.default: [MXINT6_E5M3_SPEC, MXINT6_E5M3_RHS_SPEC],
}

# Outlier filtering on the linears only: each activation sets aside its
# largest 1% before quantizing.  Attention and the `lm_head` activation stay
# dense at int6, as in `mxlut4_int6_e5m3_attn_head_int6`, which is this
# config's dense twin.
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_outlier"] = {
    torch.nn.Linear: [
        f"{MXLUT4_INT6_E5M3_SPEC},opct=0.01",
        MXLUT4_INT6_E5M3_SPEC,
    ],
    torch.ops.aten.matmul.default: [MXINT6_E5M3_SPEC, MXINT6_E5M3_RHS_SPEC],
    ("lm_head", torch.ops.aten.linear.default, 0): [
        MXINT6_E5M3_SPEC,
        MXLUT4_INT6_E5M3_SPEC,
    ],
}

# The same linears, plus the attention key and value: the matmuls' second
# operand sets aside its largest 1% too, with the attention operands kept at
# NormalFloat.  A side-stream on the column operand makes the lowering swap
# each matmul so the CSR lands on the row side, transposing the scores.
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_outlier_kv"] = {
    torch.nn.Linear: [
        f"{MXLUT4_INT6_E5M3_SPEC},opct=0.01",
        MXLUT4_INT6_E5M3_SPEC,
    ],
    torch.ops.aten.matmul.default: [
        MXLUT4_INT6_E5M3_SPEC,
        f"{MXLUT4_INT6_E5M3_RHS_SPEC},opct=0.01",
    ],
    ("lm_head", torch.ops.aten.linear.default, 0): [
        MXINT6_E5M3_SPEC,
        MXLUT4_INT6_E5M3_SPEC,
    ],
}

# The attention side-stream on the row operand instead: Q carries it on the
# first matmul and P @ V has none, so no matmul is swapped.
QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_outlier_q"] = {
    **QUANTIZATION_CONFIGS["mxlut4_int6_e5m3_outlier_kv"],
    ("self_attn", torch.ops.aten.matmul.default, 0): [
        f"{MXLUT4_INT6_E5M3_SPEC},opct=0.01",
        MXLUT4_INT6_E5M3_RHS_SPEC,
    ],
    ("self_attn", torch.ops.aten.matmul.default, 1): [
        MXLUT4_INT6_E5M3_SPEC,
        MXLUT4_INT6_E5M3_RHS_SPEC,
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
# Entry width of a KIVI cache.  2 is KIVI's own setting; 4 buys accuracy
# back at twice the cache traffic, with the group geometry unchanged.
KIVI_CACHE_BITS = 2


def kivi_cache_spec(bits, role):
    """The dtype string for one KIVI cache tensor.

    Args:
        bits: Entry width of the stored cache.
        role: ``"key"``, grouped along the sequence (``ax=-2``), or
            ``"value"``, grouped along the head dim (``ax=-1``).

    Returns:
        The spec string that quantizes that cache.
    """
    ax = -2 if role == "key" else -1
    return (
        f"uint{bits},bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,"
        f"ax={ax},scale=fp8_e4m3"
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


def annotate_kivi_cache(gm, bits=KIVI_CACHE_BITS):
    """Quantize the main KV caches of a split decode graph to KIVI's
    grouped-affine layout.

    The observer sits on the main cache's read into the attention, so the
    whole buffer -- completed chunks and the zeros above them -- is
    quantized on the way in, and ``quant_folding`` can bake it and move the
    quantize onto each chunk's fold.  The residual buffers are not annotated.

    Args:
        gm: Decode graph from ``convert_and_export_with_cache``, after
            ``split_kv_cache``; annotated in place.
        bits: Entry width of the stored cache.

    Returns:
        The number of caches annotated.

    Raises:
        RuntimeError: A cache is still written directly, i.e. the graph was
            not split.
    """
    key_qspec = QuantizationSpec.from_str(kivi_cache_spec(bits, "key"))
    value_qspec = QuantizationSpec.from_str(kivi_cache_spec(bits, "value"))
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


def set_qconfig(quantizer, qconfigs):
    def make_qspec(spec):
        return None if spec is None else QuantizationSpec.from_str(spec)

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
