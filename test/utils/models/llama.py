"""Llama prefill / decode export and lowering for ``test_codegen.py``.

Both stages are exported straight from Hugging Face's ``AutoModelForCausalLM``
-- embeddings, decoder layers, final norm and ``lm_head`` all in one graph --
the same way ``benchmarks/common.py`` builds its sweep graphs.  Prefill keeps
only the last position's logits (``logits_to_keep=1``) so the vocabulary
projection lowers as a matrix-vector product; decode is one token over a
static KV cache captured by ``convert_and_export_with_cache``, and a
speculative-decoding verification step (``llama_verify``) is the same export
over ``--spec_length`` tokens at once.
"""

import logging
import math
import os
import re
import sys

import torch
from datasets import load_dataset
from torch._export.utils import _disable_aten_to_metadata_assertions
from torch.utils._pytree import tree_flatten
from torchao.quantization.pt2e.quantizer.utils import annotate_output_qspec
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StaticCache,
)
from transformers.integrations.executorch import convert_and_export_with_cache

from voyager_compiler import (
    QuantizationConfig,
    QuantizationSpec,
    ShapeProp,
    compile,
    convert_pt2e,
    export_model,
    prepare_pt2e,
    split_kv_cache,
    transform,
)
from voyager_compiler.codegen import (
    remove_softmax_dtype_cast,
    replace_rmsnorm_with_layer_norm,
)

from .utils import get_compile_args, get_transform_args

# ``set_qconfig`` lives in the language-modeling example, not the package.
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../examples/language_modeling"
        )
    )
)
from quantization_configs import (  # noqa: E402
    KIVI_BLOCK_SIZE,
    KIVI_RESIDUAL_LENGTH,
    QUANTIZATION_CONFIGS,
    set_kivi_attention_qconfig,
    set_qconfig,
    set_residual_attention_qconfig,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

# Generation slots the decode KV cache holds beyond the prefilled context.
DECODE_MAX_GEN = 128

# The KV-cache buffers of a split decode graph: the main caches and the
# residuals ``split_kv_cache`` puts beside them.
_KV_BUFFER = re.compile(r"^(key|value)_cache_\d+(_residual)?$")

# The rotary embedding's ``inv_freq @ position`` matmul is not an MXU op and
# stays unquantized.  A regex, so it matches both the prefill scope
# (``model.rotary_emb``) and the executorch wrapper's (``model.model...``).
_ROTARY_SCOPE = r"model\.rotary_emb"

# The 2-bit KV cache the compiler lowers: KIVI's groups (keys along the
# sequence per channel, values along the head dim per token), with the
# group scale and zero point in fp8_e5m3, the format of the block scale
# the fused dequantize reads them beside.
_KIVI_KEY_SPEC = (
    f"uint2,bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,ax=-2,scale=fp8_e5m3"
)
_KIVI_VALUE_SPEC = (
    f"uint2,bs={KIVI_BLOCK_SIZE},qs=group_wise_affine,ax=-1,scale=fp8_e5m3"
)
_KV_CACHE = re.compile(r"^(key|value)_cache_\d+$")


def load_model(args):
    """Load the causal LM (one decoder layer under
    ``--compile_single_layer``) and its tokenizer."""
    if args.model_name_or_path is None:
        args.model_name_or_path = DEFAULT_MODEL

    extra = {"num_hidden_layers": 1} if args.compile_single_layer else {}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation=args.attn_implementation,
        **extra,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer


def _prompt_ids(tokenizer, length):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    return encodings.input_ids[:, :length]


def residual_length(args):
    """Positions of a decode KV cache kept in full precision while their
    chunk fills: ``--residual_length``, or ``KIVI_RESIDUAL_LENGTH``; a
    multiple of the 2-bit group (which the int6 block matches)."""
    length = args.residual_length or KIVI_RESIDUAL_LENGTH
    if length % KIVI_BLOCK_SIZE:
        raise ValueError(
            f"--residual_length {length} is not a multiple of the "
            f"{KIVI_BLOCK_SIZE}-token KIVI group"
        )
    return length


def max_cache_len(args, config):
    """Context plus generation budget, rounded up to a vector-lane multiple
    so the KV tensors stay block-aligned, and to a whole number of residual
    chunks."""
    block = math.lcm(config.vector_lanes, residual_length(args))
    raw = args.context_length + DECODE_MAX_GEN
    return -(-raw // block) * block


def build_prefill(model, tokenizer, args):
    """Export the whole model over ``context_length`` prompt tokens with no
    cache; ``logits_to_keep=1`` slices the hidden states before ``lm_head``
    so it lowers as a GEMV.  Returns ``(gm, example_args, example_kwargs)``."""
    input_ids = _prompt_ids(tokenizer, args.context_length)
    example_args = (input_ids,)
    example_kwargs = {
        "return_dict": False,
        "use_cache": False,
        "logits_to_keep": 1,
    }
    gm = export_model(model, example_args, example_kwargs)
    return gm, example_args, example_kwargs


def build_decode(model, tokenizer, args, config, tokens):
    """Export one decode step of ``tokens`` positions from ``cache_position =
    context_length`` over a static BF16 KV cache of ``max_cache_len`` slots,
    via Hugging Face's ``convert_and_export_with_cache``, then prefill the
    exported cache with the ``context_length`` prompt tokens so calibration
    and the baked cache see real contents.  The caches are then split into
    completed chunks and a full-precision residual of ``residual_length(args)``
    positions.  ``tokens`` is 1 for plain decode; a speculative-decoding
    verification step writes the draft's tokens at once, and must not cross
    a residual chunk boundary.  Returns ``(gm, (), example_kwargs)``.
    """
    length = residual_length(args)
    if tokens > DECODE_MAX_GEN:
        raise ValueError(
            f"a {tokens}-token step exceeds the {DECODE_MAX_GEN} generation "
            "slots the cache holds past the context"
        )
    if args.context_length % length + tokens > length:
        raise ValueError(
            f"a {tokens}-token step from position {args.context_length} "
            f"crosses a {length}-position residual chunk boundary"
        )
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        cache_config={
            "batch_size": 1,
            "max_cache_len": max_cache_len(args, config),
        },
    )
    # The tokens past the prompt, so the step reads real ids.
    prompt = _prompt_ids(tokenizer, args.context_length + tokens)
    input_ids = prompt[:, -tokens:]
    cache_position = torch.arange(
        args.context_length, args.context_length + tokens
    )
    # Strict export bakes in aten._assert_tensor_metadata guards (the
    # attention softmax's float32); remove_softmax_dtype_cast later rewrites
    # that softmax to bf16, so the guards must be suppressed at export.
    with _disable_aten_to_metadata_assertions():
        ep = convert_and_export_with_cache(
            model,
            example_input_ids=input_ids,
            example_cache_position=cache_position,
        )
    gm = ep.module()

    prompt = prompt[:, : args.context_length]
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len(args, config),
        dtype=model.dtype,
    )
    model(prompt, past_key_values=cache, use_cache=True)
    for i, layer in enumerate(cache.layers):
        getattr(gm, f"key_cache_{i}").copy_(layer.keys)
        getattr(gm, f"value_cache_{i}").copy_(layer.values)

    split_kv_cache(gm, length, args.context_length)

    example_kwargs = {"input_ids": input_ids, "cache_position": cache_position}
    return gm, (), example_kwargs


def annotate_kivi_cache(gm):
    """Quantize the main KV caches of a split decode graph to the 2-bit
    layout the compiler lowers (``_KIVI_KEY_SPEC`` / ``_KIVI_VALUE_SPEC``).

    The observer sits on the main cache's read into the attention, so the
    whole buffer -- completed chunks and the zeros above them -- is
    quantized on the way in, and ``quant_folding`` bakes it and moves the
    quantize onto each chunk's fold.  The fold's own handle on the buffer,
    whose ``index_copy_`` writes a ``where``, is not a read; the residual
    buffers are not annotated.

    Args:
        gm: Decode graph after ``split_kv_cache``; annotated in place.

    Raises:
        RuntimeError: A cache is written directly, i.e. the graph was not
            split.
    """
    key_qspec = QuantizationSpec.from_str(_KIVI_KEY_SPEC)
    value_qspec = QuantizationSpec.from_str(_KIVI_VALUE_SPEC)
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
            if writes[0].args[3].target is not torch.ops.aten.where.self:
                raise RuntimeError(
                    f"{node.target} is written directly: split the graph "
                    "with split_kv_cache before annotating the caches"
                )
            continue
        annotate_output_qspec(
            node, key_qspec if match.group(1) == "key" else value_qspec
        )


def kv_cache_state(gm):
    """A copy of every KV-cache buffer, to put back with
    ``restore_kv_cache`` before the graph is run again.

    The decode step is not idempotent: the step that completes a chunk folds
    the residual into the main cache, and a second run of it would then read
    the chunk from both.
    """
    return {
        name: buffer.clone()
        for name, buffer in gm.named_buffers()
        if _KV_BUFFER.match(name)
    }


def restore_kv_cache(gm, state):
    for name, saved in state.items():
        getattr(gm, name).copy_(saved)


def quantize_model(model, tokenizer, quantizer, vector_stages, args):
    """Export and quantize the stage ``args.model`` names (``llama_prefill`` /
    ``llama_decode`` / ``llama_verify``), stopping short of ``transform``.
    Returns ``(gm, example_args, example_kwargs, old_output, transform_args,
    compile_args, kv_state)`` -- the converted graph with shapes propagated,
    its example inputs and reference output, the keyword sets ``transform``
    and ``compile`` take, and the KV-cache state to restore before running
    the graph again (``restore_kv_cache``)."""
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)
    config = transform_args["config"]

    is_decode = args.model in (
        "llama_decode",
        "llama_decode_kivi",
        "llama_verify",
    )
    tokens = args.spec_length if args.model == "llama_verify" else 1
    if is_decode:
        gm, example_args, example_kwargs = build_decode(
            model, tokenizer, args, config, tokens
        )
    else:
        gm, example_args, example_kwargs = build_prefill(model, tokenizer, args)
    kv_state = kv_cache_state(gm)

    remove_softmax_dtype_cast(gm)

    hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
    seq = tokens if is_decode else 128
    example_input = torch.randn(1, seq, hidden_size, dtype=model.dtype)
    replace_rmsnorm_with_layer_norm(
        gm, model.model.layers[0].input_layernorm, (example_input,)
    )

    quantizer.set_module_name_object_type_order(
        _ROTARY_SCOPE, torch.ops.aten.matmul.default, 0, None
    )

    if args.qconfig is not None:
        set_qconfig(quantizer, QUANTIZATION_CONFIGS[args.qconfig])

    if args.model == "llama_decode_kivi":
        set_kivi_attention_qconfig(quantizer)
        annotate_kivi_cache(gm)
    elif is_decode:
        set_residual_attention_qconfig(quantizer)

    if args.qconfig is not None or args.model == "llama_decode_kivi":
        fp8_qspec = QuantizationSpec.from_str(
            "fp8_e4m3,qs=per_tensor_symmetric,qmax=240"
        )
        qconfig = QuantizationConfig(fp8_qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.softmax.int, qconfig)
        quantizer.set_object_type(torch.ops.aten.layer_norm.default, qconfig)

    # The HF export builds the causal mask in-graph: a ``where`` over the
    # boolean mask that the attention scores' ``add`` reads.  In prefill the
    # mask is a constant, so annotating the ``where`` makes convert_pt2e emit
    # quantize -> dequantize on it and the constant fold leaves an int1
    # constant plus the dequantize.  Decode rebuilds the mask from
    # ``cache_position`` every step and keeps it in bf16.
    if args.quantize_attention_mask and not is_decode:
        qspec = QuantizationSpec.from_str("int1,qs=per_tensor_symmetric,qmax=1")
        masks = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.where.ScalarOther
            and any(u.target is torch.ops.aten.add.Tensor for u in n.users)
        ]
        if not masks:
            raise RuntimeError("no causal-mask where node feeds an add")
        for mask in masks:
            annotate_output_qspec(mask, qspec)

    gm = prepare_pt2e(gm, quantizer, example_args, example_kwargs)

    for _ in range(2):
        restore_kv_cache(gm, kv_state)
        gm(*example_args, **example_kwargs)
    restore_kv_cache(gm, kv_state)

    convert_pt2e(gm, args.bias)

    flatten_args, _ = tree_flatten((example_args, example_kwargs))
    old_output = ShapeProp(gm).propagate(*flatten_args)
    return (
        gm,
        example_args,
        example_kwargs,
        old_output,
        transform_args,
        compile_args,
        kv_state,
    )


def quantize_and_dump_model(model, tokenizer, quantizer, vector_stages, args):
    """Export, quantize, transform and compile the stage ``args.model`` names
    (``llama_prefill`` / ``llama_decode``).  Returns ``(gm, old_output,
    new_output)``; ``new_output`` is ``None`` unless ``--debug`` re-runs the
    lowered graph."""
    (
        gm,
        example_args,
        example_kwargs,
        old_output,
        transform_args,
        compile_args,
        _,
    ) = quantize_model(model, tokenizer, quantizer, vector_stages, args)

    transform(gm, example_args, example_kwargs, **transform_args)
    compile(gm, example_args, example_kwargs, **compile_args)
    gm.graph.print_tabular()

    new_output = gm(*example_args, **example_kwargs) if args.debug else None
    return gm, old_output, new_output
