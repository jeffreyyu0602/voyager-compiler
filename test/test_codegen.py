import argparse
import logging
import os
import re
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils._pytree import tree_flatten
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_output_qspec as _annotate_output_qspec,
)
from torchvision import models, transforms
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    StaticCache,
)
from utils.dataset import glue, imagenet
from utils.models import bert, mobilebert, torchvision_models, vit
from utils.models.utils import get_compile_args, get_transform_args

from voyager_compiler import (
    OpMatcher,
    QuantizationConfig,
    QuantizationSpec,
    ShapeProp,
    TorchExportableModuleWithStaticCache,
    add_compile_args,
    add_quantization_args,
    compile,
    convert_and_export_with_split_cache,
    convert_pt2e,
    export_model,
    extract_input_preprocessor,
    fuse_dequantize_quantize,
    fuse_operator,
    get_default_quantizer,
    prepare_pt2e,
    print_node_scope_tabular,
    sink_obs_or_fq,
    swap_llama_attention,
    transform,
)
from voyager_compiler.codegen import (
    remove_softmax_dtype_cast,
    replace_rmsnorm_with_layer_norm,
)
from voyager_compiler.codegen.node_info import is_fully_connected

logger = logging.getLogger()


def _is_bf16_fc(node):
    # BF16 FC are ran on vector unit and thus cannot be fused
    if hasattr(node, "value") and is_fully_connected(node):
        input_node = node.args[0]
        return input_node.meta.get("dtype") is None
    return False


def _is_spmm(node):
    return node.kwargs.get("A_data") is not None


def _can_fuse(node):
    return not _is_spmm(node) and not _is_bf16_fc(node)


def _is_constant_div(node):
    if node.target != torch.ops.aten.div.Tensor:
        return True

    divisor = node.args[1]
    if isinstance(divisor, torch.fx.Node):
        return divisor.value.numel() == 1

    return True


MXU_OPS = ["conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"]
QUANT_OPS = ["quantize", "quantize_mx", "quantize_mx_outlier"]

VECTOR_PIPELINE = [
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("add", "sub", "mul", "div", predicate=_is_constant_div),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div", predicate=_is_constant_div),
        OpMatcher(*QUANT_OPS, "mul", "div"),
    ],
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("gelu", "sigmoid", "silu", "tanh", "hardtanh"),
        OpMatcher(*QUANT_OPS, "mul", "div"),
    ],
    # Fused SpMM operation will use the first stage in the pipeline
    [
        OpMatcher(*MXU_OPS, predicate=_is_spmm),
        OpMatcher("dequantize"),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div"),
        OpMatcher(*QUANT_OPS),
    ],
    [
        OpMatcher("layer_norm", "softmax"),
        OpMatcher(*QUANT_OPS),
    ],
]


def get_llama_qconfig(bs=64, outlier_pct=None):
    if outlier_pct is None:
        return {
            torch.nn.Linear: [
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            ],
            torch.ops.aten.matmul.default: [
                f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"int6,qs=microscaling,bs={bs},ax=-2,scale=fp8_e5m3",
            ],
            (r"lm_head", torch.ops.aten.linear.default, 0): [
                f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            ],
        }
    else:
        return {
            torch.nn.Linear: [
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3,opct={outlier_pct}",
                f"nf4_6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
            ],
            torch.ops.aten.matmul.default: [
                f"int6,qs=microscaling,bs={bs},ax=-1,scale=fp8_e5m3",
                f"nf4_6,qs=microscaling,bs={bs},ax=-2,scale=fp8_e5m3,othr=6.0",
            ],
        }


def main():
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, precision=10)
    torch.set_num_threads(32)
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help=(
            "Path to pretrained model or model identifier from "
            "huggingface.co/models."
        ),
    )
    parser.add_argument(
        "--task_name",
        default="sst2",
        help="Name of the task to load the dataset",
    )
    parser.add_argument(
        "--model_output_dir",
        required=True,
        help="Output directory for generated tensor files",
    )
    parser.add_argument(
        "--dump_dataset",
        action="store_true",
        help="Whether to save the dataset for later use.",
    )
    parser.add_argument(
        "--dataset_output_dir", help="Output directory for dataset files"
    )
    parser.add_argument(
        "--dump_tensors",
        action="store_true",
        help="Whether to save intermediate outputs for verification.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Re-run the lowered graph and compare its output against the "
            "pre-transform reference.  Re-running the graph is expensive on a "
            "large model, so this verification is off by default."
        ),
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Context length for the LLM decoding.",
    )
    parser.add_argument(
        "--compile_single_layer",
        action="store_true",
        help=(
            "Only compile for a single encoder/decoder layer in Transformer "
            "models."
        ),
    )
    parser.add_argument(
        "--enable_mixed_precision",
        action="store_true",
        help="Use mixed precision quantization scheme to quantize LLMs.",
    )
    parser.add_argument(
        "--outlier_pct",
        type=float,
        default=None,
        help="Percentage of outliers to filter when quantizing activations.",
    )
    parser.add_argument(
        "--use_maxpool_2x2",
        action="store_true",
        help="Whether to use 2x2 maxpool for resnet18 and resnet50.",
    )
    parser.add_argument(
        "--conv2d_im2col",
        action="store_true",
        help=(
            "Whether to transform Conv2d operations with small input channels "
            "into linear operations using im2col."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to run the pytorch evaluation during compilation",
    )
    parser.add_argument(
        "--quantize_attention_mask",
        action="store_true",
        help="Whether to quantize Transformer attention mask to binary values.",
    )
    parser.add_argument(
        "--quantize_fc",
        action="store_true",
        help="Whether to quantize the fully connected layers.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level.",
    )
    add_quantization_args(parser)
    add_compile_args(parser)
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    transform_args = get_transform_args(args, VECTOR_PIPELINE)
    compile_args = get_compile_args(args)
    config = transform_args["config"]

    if args.model in models.__dict__:
        model = torchvision_models.load_model(args)

        if args.dump_dataset or args.evaluate:
            imagenet_dataset = imagenet.retrieve_dataset(1000, "resnet")
            if args.evaluate:
                torchvision_models.evaluate(model, imagenet_dataset)
        else:
            imagenet_dataset = imagenet.retrieve_dataset(10, "resnet")

        gm, old_output, new_output, preprocess_fn = (
            torchvision_models.quantize_and_dump_model(
                model=model,
                quantizer=quantizer,
                calibration_data=imagenet_dataset,
                vector_stages=VECTOR_PIPELINE,
                args=args,
            )
        )

        if args.dump_dataset:
            preprocessed_imagenet = imagenet.dump_imagenet(
                args.dataset_output_dir,
                imagenet_dataset,
                "resnet",
                preprocess_fn,
                torch_dtype,
            )

        if args.evaluate:
            torchvision_models.evaluate(gm, preprocessed_imagenet)
    elif args.model == "mobilebert":
        model, tokenizer = mobilebert.load_model(args)

        eval_dataset, train_dataset = glue.retrieve_dataset(
            model, tokenizer, args
        )

        if args.evaluate:
            mobilebert.evaluate(model, eval_dataset)

        if args.dump_dataset:
            preprocessed_dataset = glue.dump_dataset(
                args.dataset_output_dir, eval_dataset, model
            )

        gm, old_output, new_output = mobilebert.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=train_dataset,
            vector_stages=VECTOR_PIPELINE,
            args=args,
        )

        if args.evaluate:
            mobilebert.evaluate_gm(gm, preprocessed_dataset)

    elif args.model == "bert":
        model, tokenizer = bert.load_model(args)

        eval_dataset, train_dataset = glue.retrieve_dataset(
            model, tokenizer, args
        )

        if args.evaluate:
            bert.evaluate(model, eval_dataset)

        if args.dump_dataset:
            preprocessed_dataset = glue.dump_dataset(
                args.dataset_output_dir, eval_dataset, model
            )

        gm, old_output, new_output = bert.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=train_dataset,
            vector_stages=VECTOR_PIPELINE,
            args=args,
        )

        if args.evaluate:
            bert.evaluate_gm(gm, preprocessed_dataset)

    elif args.model == "llm_prefill" or args.model == "llm_decode":
        from transformers import AutoModelForCausalLM

        if args.model_name_or_path is None:
            args.model_name_or_path = "meta-llama/Llama-3.1-8B"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            attn_implementation=args.attn_implementation,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

        input_ids = encodings.input_ids[:, : args.context_length]

        past_key_values = None

        block = config.vector_lanes
        raw = args.context_length + 128
        max_cache_len = -(-raw // block) * block

        if args.model == "llm_decode":
            transform_args["context_len"] = args.context_length
            transform_args["max_new_tokens"] = 128

            past_key_values = StaticCache(
                config=model.config,
                max_batch_size=1,
                max_cache_len=max_cache_len,
                dtype=model.dtype,
            )

            with torch.no_grad():
                outputs = model(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )

            input_ids = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )
            past_key_values = outputs.past_key_values

        inputs_embeds = model.model.embed_tokens(input_ids)

        past_seen_tokens = (
            past_key_values.get_seq_length().item()
            if past_key_values is not None
            else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        position_ids = cache_position.unsqueeze(0)

        causal_mask = TorchExportableModuleWithStaticCache._prepare_4d_causal_attention_mask_with_cache_position(
            None,
            sequence_length=inputs_embeds.shape[1],
            target_length=max_cache_len,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            cache_position=cache_position,
            batch_size=inputs_embeds.shape[0],
        )

        if args.model == "llm_prefill":
            causal_mask = causal_mask[:, :, :, : args.context_length]

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(
            inputs_embeds, position_ids
        )

        example_args = (
            inputs_embeds,
            causal_mask,
            position_embeddings,
            cache_position,
        )

        class LlamaWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model.model
                self.lm_head = model.lm_head

                self.static_cache = past_key_values

                if self.static_cache is not None:
                    for i in range(len(self.static_cache.layers)):
                        self.register_buffer(
                            f"key_cache_{i}",
                            self.static_cache.layers[i].keys,
                            persistent=False,
                        )
                        self.register_buffer(
                            f"value_cache_{i}",
                            self.static_cache.layers[i].values,
                            persistent=False,
                        )

            def forward(
                self,
                hidden_states,
                attention_mask,
                position_embeddings,
                cache_position=None,
            ):
                for decoder_layer in self.model.layers:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                        past_key_values=self.static_cache,
                        cache_position=cache_position,
                    )
                    hidden_states = layer_outputs[0]

                    if args.compile_single_layer:
                        break

                logits = self.lm_head(hidden_states)
                return logits

        gm = export_model(LlamaWrapper(), example_args)

        remove_softmax_dtype_cast(gm)

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 128, hidden_size, dtype=model.dtype)
        replace_rmsnorm_with_layer_norm(
            gm, model.model.layers[0].input_layernorm, (example_input,)
        )

        if args.enable_mixed_precision:
            qconfig = get_llama_qconfig(args.pe_array_size[1], args.outlier_pct)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            target_path = os.path.join(
                script_dir, "../examples/language_modeling"
            )
            sys.path.append(os.path.abspath(target_path))

            from quantization_configs import set_qconfig

            set_qconfig(quantizer, qconfig)

            fp8_qspec = QuantizationSpec.from_str(
                "fp8_e4m3,qs=per_tensor_symmetric,qmax=240"
            )
            qconfig = QuantizationConfig(fp8_qspec, None, None, None)
            quantizer.set_object_type(torch.ops.aten.softmax.int, qconfig)
            quantizer.set_object_type(
                torch.ops.aten.layer_norm.default, qconfig
            )

        if args.quantize_attention_mask:
            qspec = QuantizationSpec.from_str(
                "int1,qs=per_tensor_symmetric,qmax=1"
            )
            attention_mask = next(
                iter(n for n in gm.graph.nodes if n.target == "attention_mask")
            )
            _annotate_output_qspec(attention_mask, qspec)

        gm = prepare_pt2e(gm, quantizer, example_args)

        for _ in range(2):
            gm(*example_args)

        convert_pt2e(gm, args.bias)

        flatten_args, _ = tree_flatten(example_args)
        old_output = ShapeProp(gm).propagate(*flatten_args)

        if args.quantize_attention_mask:
            gm, mask_preprocess_fn = extract_input_preprocessor(
                gm, "attention_mask"
            )
            h, mask, pos, cache = example_args
            example_args = (h, mask_preprocess_fn(mask), pos, cache)

        transform(gm, example_args, **transform_args)
        compile(gm, example_args, **compile_args)
        gm.graph.print_tabular()
        new_output = gm(*example_args) if args.debug else None
    elif args.model == "llm_kivi":
        from transformers import AutoModelForCausalLM

        if args.model_name_or_path is None:
            args.model_name_or_path = "meta-llama/Llama-3.1-8B"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            attn_implementation=args.attn_implementation,
        ).eval()

        if args.compile_single_layer:
            layers_to_keep = model.model.layers[:1]
            model.model.layers = nn.ModuleList(layers_to_keep)

            if hasattr(model, "config"):
                model.config.num_hidden_layers = len(model.model.layers)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

        input_ids = encodings.input_ids[:, : args.context_length]

        block = config.vector_lanes
        # context + generation budget, rounded up to a block_size multiple.
        raw = args.context_length + 128
        max_cache_len = -(-raw // block) * block

        max_length = args.context_length
        max_gen = max_cache_len - max_length

        swap_llama_attention(model)

        from torch._export.utils import _disable_aten_to_metadata_assertions

        with _disable_aten_to_metadata_assertions():
            gm = convert_and_export_with_split_cache(
                model, max_len=max_length, max_new_tokens=max_gen
            ).module()

        # Run decode once to fill in the KV caches
        output = TorchExportableModuleWithStaticCache.generate(
            model,
            prompt_token_ids=input_ids,
            max_new_tokens=max_gen,
            min_length=max_length + 1,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.encode("\n", add_special_tokens=False)[-1],
            ],
            model_decode=gm,
        )[0]

        remove_softmax_dtype_cast(gm)

        hidden_size = model.model.layers[0].input_layernorm.weight.shape[-1]
        example_input = torch.randn(1, 1, hidden_size, dtype=model.dtype)
        replace_rmsnorm_with_layer_norm(
            gm, model.model.layers[0].input_layernorm, (example_input,)
        )

        quantizer.set_object_type(torch.ops.aten.matmul.default, None)

        act0 = QuantizationSpec.from_str(
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3"
        )
        act1 = QuantizationSpec.from_str(
            "int6,qs=microscaling,bs=64,ax=(-2,-1),scale=fp8_e5m3"
        )
        qconfig = QuantizationConfig(act0, None, act1, None)

        for layer_idx in range(model.config.num_hidden_layers):
            module_name = f"model.model.layers.{layer_idx}.self_attn"
            quantizer.set_module_name_object_type_order(
                module_name, torch.ops.aten.matmul.default, 0, qconfig
            )
            quantizer.set_module_name_object_type_order(
                module_name, torch.ops.aten.matmul.default, 2, qconfig
            )

        fp8_qspec = QuantizationSpec.from_str(
            "fp8_e4m3,qs=per_tensor_symmetric,qmax=240"
        )
        qconfig = QuantizationConfig(fp8_qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.softmax.int, qconfig)
        quantizer.set_object_type(torch.ops.aten.layer_norm.default, qconfig)

        if args.quantize_attention_mask:
            qspec = QuantizationSpec.from_str(
                "int1,qs=per_tensor_symmetric,qmax=1"
            )
            attention_mask = next(
                iter(n for n in gm.graph.nodes if n.target == "attention_mask")
            )
            _annotate_output_qspec(attention_mask, qspec)

        key_qspec = QuantizationSpec.from_str(
            "uint2,bs=64,qs=group_wise_affine,ax=-2,scale=fp8_e5m3"
        )
        value_qspec = QuantizationSpec.from_str(
            "uint2,bs=64,qs=group_wise_affine,ax=-1,scale=fp8_e5m3"
        )

        for node in gm.graph.nodes:
            match = re.match(r"^(key|value)_cache_(\d+)$", str(node.target))
            if node.op == "get_attr" and match is not None:
                _annotate_output_qspec(
                    node, key_qspec if match.group(1) == "key" else value_qspec
                )

        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
        example_cache_position_residual = torch.tensor([0], dtype=torch.long)
        example_attention_mask = torch.ones(
            (1, max_cache_len), dtype=torch_dtype
        )[None, None, :, :]
        example_kwargs = {
            "input_ids": example_input_ids,
            "cache_position": example_cache_position,
            "cache_position_residual": example_cache_position_residual,
            "attention_mask": example_attention_mask,
        }

        gm = prepare_pt2e(gm, quantizer)

        for _ in range(2):
            gm(**example_kwargs)

        sink_obs_or_fq(gm)
        convert_pt2e(gm, eliminate_no_effect=False)

        flatten_args, _ = tree_flatten(example_kwargs)
        old_output = ShapeProp(gm).propagate(*flatten_args)

        if args.quantize_attention_mask:
            gm, mask_preprocess_fn = extract_input_preprocessor(
                gm, "attention_mask"
            )
            example_kwargs["attention_mask"] = mask_preprocess_fn(
                example_kwargs["attention_mask"]
            )

        fuse_dequantize_quantize(gm)

        transform(gm, (), example_kwargs, **transform_args)
        compile(gm, (), example_kwargs, **compile_args)

        gm.graph.print_tabular()
        new_output = gm(**example_kwargs) if args.debug else None
    elif args.model == "vit":
        model = vit.load_model(args)

        if args.dump_dataset or args.evaluate:
            imagenet_dataset = imagenet.retrieve_dataset(1000, "vit")
            if args.evaluate:
                vit.evaluate(model, imagenet_dataset)
        else:
            imagenet_dataset = imagenet.retrieve_dataset(10, "vit")

        gm, old_output, new_output, preprocess_fn = vit.quantize_and_dump_model(
            model=model,
            quantizer=quantizer,
            calibration_data=imagenet_dataset,
            vector_stages=VECTOR_PIPELINE,
            args=args,
        )

        if args.dump_dataset:
            preprocessed_imagenet = imagenet.dump_imagenet(
                args.dataset_output_dir,
                imagenet_dataset,
                "vit",
                preprocess_fn,
                torch_dtype,
            )

        if args.evaluate:
            vit.evaluate(gm, preprocessed_imagenet)
    else:
        raise ValueError(f"Model {args.model} not supported")

    if new_output is None:
        print("Skipping output verification (pass --debug to run it)")
        return

    try:
        assert torch.all(old_output == new_output)
        print("Results match")
    except Exception as e:
        print(e)
        print(old_output)
        print(new_output)


if __name__ == "__main__":
    main()
