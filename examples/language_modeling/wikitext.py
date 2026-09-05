import argparse
import logging

import torch
from accelerate.utils import get_max_memory
from datasets import load_dataset

from common import (
    add_model_args,
    evaluate_perplexity,
    load_model_and_tokenizer,
)
from quantization_configs import QUANTIZATION_CONFIGS, set_qconfig
from torchao.quantization.pt2e import FakeQuantizeBase
from voyager_compiler import (
    add_experiment_args,
    get_default_quantizer,
    prepare_pt2e,
    convert_pt2e,
    with_execution_context,
    print_node_scope_tabular,
    get_device_map,
    dispatch_model,
    insert_align_device_nodes,
    sink_obs_or_fq,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    add_model_args(parser)
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Stride for processing the data",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Output directory for histograms"
    )
    parser.add_argument(
        "--qconfig", default=None, help="Quantization scheme for the model"
    )
    parser.add_argument(
        "--reserved_memory",
        type=int,
        default=8,
        help="GPU memory reserved for storing activations",
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="Print node scope information",
    )
    add_experiment_args(parser)
    return parser.parse_args()


def setup_quantized_model(
    model_id,
    quantizer,
    max_length,
    torch_dtype,
    device=None,
    reserved_memory=8,
    print_model=False,
):
    """Load model, prepare the quantized graph module, and dispatch to GPU(s).

    When device is not None (single GPU), moves the model to that device.
    When device is None (multi-GPU), dispatches the graph module across
    available GPUs after prepare_pt2e, reserving reserved_memory GiB per GPU
    for activations.

    Returns (model, tokenizer).
    """
    model, tokenizer = load_model_and_tokenizer(
        model_id, torch_dtype, attn_implementation="eager"
    )

    if device is not None:
        model.to(device)

    input_ids = torch.randint(
        0, model.config.vocab_size, (1, max_length), device=device
    )
    labels = input_ids.clone()
    example_args = (input_ids,)
    example_kwargs = {"labels": labels, "use_cache": False}
    chunk_dim = torch.export.Dim("chunk_dim", min=2, max=max_length // 64)
    dynamic_shapes = {
        "input_ids": {1: chunk_dim * 64},
        "labels": {1: chunk_dim * 64},
        "use_cache": None,
    }

    with torch.no_grad():
        gm = prepare_pt2e(
            model, quantizer, example_args, example_kwargs, dynamic_shapes
        )

    if print_model:
        gm.graph.print_tabular()
        print_node_scope_tabular(gm)

    sink_obs_or_fq(gm)

    if device is None:
        reserved_bytes = reserved_memory * 1024**3
        max_memory = {
            k: v - reserved_bytes
            for k, v in get_max_memory().items()
            if isinstance(k, int) and v > reserved_bytes
        }
        device_map = get_device_map(gm, max_memory)
        dispatch_model(gm, device_map)

        for node in list(gm.graph.nodes):
            if node.op not in ["placeholder", "output"] and not node.users:
                gm.graph.erase_node(node)

        insert_align_device_nodes(gm, (input_ids, labels))

    return gm, tokenizer


@with_execution_context
def main(args):
    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else None

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        weight=args.weight,
        bias=args.bias,
        record_histogram=args.record_histogram,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )
    quantizer.set_module_name("model.rotary_emb", None)

    if (qconfig := QUANTIZATION_CONFIGS.get(args.qconfig)) is not None:
        set_qconfig(quantizer, qconfig, args.force_scale_power_of_two)

    model, tokenizer = setup_quantized_model(
        args.model_id,
        quantizer,
        args.max_length,
        args.torch_dtype,
        device=device,
        reserved_memory=args.reserved_memory,
        print_model=args.print_model,
    )

    if args.calibration_steps > 0:
        validation = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="validation"
        )
        calib_encodings = tokenizer(
            "\n\n".join(validation["text"]), return_tensors="pt"
        )

        evaluate_perplexity(
            model,
            calib_encodings,
            args.max_length,
            args.stride,
            device,
            num_steps=args.calibration_steps,
        )

        for module in model.modules():
            if isinstance(module, FakeQuantizeBase):
                module.disable_observer()

    if args.convert_model:
        model = convert_pt2e(model)

    model.graph.print_tabular()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    ppl = evaluate_perplexity(
        model, test_encodings, args.max_length, args.stride, device
    )

    print(f"model:      {args.model_id}")
    print(f"max length: {args.max_length}")
    print(f"stride:     {args.stride}")
    print(f"perplexity: {ppl.item()}")


if __name__ == "__main__":
    main(parse_args())
