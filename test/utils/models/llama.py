"""Llama prefill / decode export and lowering for ``test_codegen.py``.

Both stages are exported straight from Hugging Face's ``AutoModelForCausalLM``
-- embeddings, decoder layers, final norm and ``lm_head`` all in one graph --
the same way ``benchmarks/common.py`` builds its sweep graphs.  Prefill keeps
only the last position's logits (``logits_to_keep=1``) so the vocabulary
projection lowers as a matrix-vector product; decode is one token over a
static KV cache captured by ``convert_and_export_with_cache``.
"""

import logging
import os
import sys

import torch
from datasets import load_dataset
from torch._export.utils import _disable_aten_to_metadata_assertions
from torch.utils._pytree import tree_flatten
from torchao.quantization.pt2e.quantizer.utils import annotate_output_qspec
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.integrations.executorch import convert_and_export_with_cache

from voyager_compiler import (
    QuantizationConfig,
    QuantizationSpec,
    ShapeProp,
    compile,
    convert_pt2e,
    export_model,
    prepare_pt2e,
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
    QUANTIZATION_CONFIGS,
    set_qconfig,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

# Generation slots the decode KV cache holds beyond the prefilled context.
DECODE_MAX_GEN = 128

# The rotary embedding's ``inv_freq @ position`` matmul is not an MXU op and
# stays unquantized.  A regex, so it matches both the prefill scope
# (``model.rotary_emb``) and the executorch wrapper's (``model.model...``).
_ROTARY_SCOPE = r"model\.rotary_emb"


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


def max_cache_len(args, config):
    """Context plus generation budget, rounded up to a vector-lane multiple
    so the KV tensors stay block-aligned."""
    block = config.vector_lanes
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


def build_decode(model, tokenizer, args, config):
    """Export one decode step at ``cache_position = [context_length]`` over a
    static BF16 KV cache of ``max_cache_len`` slots, via Hugging Face's
    ``convert_and_export_with_cache``.  Returns ``(gm, (), example_kwargs)``.
    """
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        cache_config={
            "batch_size": 1,
            "max_cache_len": max_cache_len(args, config),
        },
    )
    # The first token past the prompt, so decode reads a real id.
    input_ids = _prompt_ids(tokenizer, args.context_length + 1)[:, -1:]
    cache_position = torch.tensor([args.context_length], dtype=torch.long)
    # Strict export bakes in aten._assert_tensor_metadata guards (the
    # attention softmax's float32); remove_softmax_dtype_cast later rewrites
    # that softmax to bf16, so the guards must be suppressed at export.
    with _disable_aten_to_metadata_assertions():
        ep = convert_and_export_with_cache(
            model,
            example_input_ids=input_ids,
            example_cache_position=cache_position,
        )
    example_kwargs = {"input_ids": input_ids, "cache_position": cache_position}
    return ep.module(), (), example_kwargs


def quantize_model(model, tokenizer, quantizer, vector_stages, args):
    """Export and quantize the stage ``args.model`` names (``llm_prefill`` /
    ``llm_decode``), stopping short of ``transform``.  Returns ``(gm,
    example_args, example_kwargs, old_output, transform_args,
    compile_args)`` -- the converted graph with shapes propagated, its example
    inputs and reference output, and the keyword sets ``transform`` and
    ``compile`` take."""
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)
    config = transform_args["config"]

    is_decode = args.model == "llm_decode"
    if is_decode:
        transform_args["context_len"] = args.context_length
        transform_args["max_new_tokens"] = DECODE_MAX_GEN
        gm, example_args, example_kwargs = build_decode(
            model, tokenizer, args, config
        )
    else:
        gm, example_args, example_kwargs = build_prefill(model, tokenizer, args)

    remove_softmax_dtype_cast(gm)

    hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
    seq = 1 if is_decode else 128
    example_input = torch.randn(1, seq, hidden_size, dtype=model.dtype)
    replace_rmsnorm_with_layer_norm(
        gm, model.model.layers[0].input_layernorm, (example_input,)
    )

    quantizer.set_module_name_object_type_order(
        _ROTARY_SCOPE, torch.ops.aten.matmul.default, 0, None
    )

    if args.qconfig is not None:
        set_qconfig(quantizer, QUANTIZATION_CONFIGS[args.qconfig])

        fp8_qspec = QuantizationSpec.from_str(
            "fp8_e4m3,qs=per_tensor_symmetric,qmax=240"
        )
        qconfig = QuantizationConfig(fp8_qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.softmax.int, qconfig)
        quantizer.set_object_type(torch.ops.aten.layer_norm.default, qconfig)

    # The HF export builds the causal mask in-graph: a ``where`` over the
    # boolean mask that the attention scores' ``add`` reads.  Annotating its
    # output makes convert_pt2e emit quantize -> dequantize on it; the
    # constant fold then leaves an int1 constant plus the dequantize.
    if args.quantize_attention_mask:
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
        gm(*example_args, **example_kwargs)

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
    )


def quantize_and_dump_model(model, tokenizer, quantizer, vector_stages, args):
    """Export, quantize, transform and compile the stage ``args.model`` names
    (``llm_prefill`` / ``llm_decode``).  Returns ``(gm, old_output,
    new_output)``; ``new_output`` is ``None`` unless ``--debug`` re-runs the
    lowered graph."""
    (
        gm,
        example_args,
        example_kwargs,
        old_output,
        transform_args,
        compile_args,
    ) = quantize_model(model, tokenizer, quantizer, vector_stages, args)

    transform(gm, example_args, example_kwargs, **transform_args)
    compile(gm, example_args, example_kwargs, **compile_args)
    gm.graph.print_tabular()

    new_output = gm(*example_args, **example_kwargs) if args.debug else None
    return gm, old_output, new_output
