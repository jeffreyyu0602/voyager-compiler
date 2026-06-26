import argparse
import logging
import os

import torch
from accelerate.utils import get_max_memory
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization_configs import QUANTIZATION_CONFIGS, set_qconfig
from voyager_compiler import (
    add_experiment_args,
    get_default_quantizer,
    prepare_pt2e,
    convert_pt2e,
    plot_histogram,
    plot_layer_range,
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
    parser.add_argument(
        '--model_id', required=True, help='Pretrained model identifier'
    )
    parser.add_argument(
        '--max_length', type=int, default=1024, help='Maximum sequence length'
    )
    parser.add_argument(
        '--stride', type=int, default=512, help='Stride for processing the data'
    )
    parser.add_argument(
        '--output_dir', default=None, help='Output directory for histograms'
    )
    parser.add_argument(
        '--torch_dtype',
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help=(
            "Override the default `torch.dtype` and load the model under this "
            "dtype. If `auto` is passed, the dtype will be automatically "
            "derived from the model's weights."
        )
    )
    parser.add_argument(
        '--qconfig', default=None, help='Quantization scheme for the model'
    )
    parser.add_argument(
        '--reserved_memory',
        type=int,
        default=8,
        help='GPU memory reserved for storing activations'
    )
    parser.add_argument(
        '--print_model', action='store_true', help='Print node scope information'
    )
    add_experiment_args(parser)
    return parser.parse_args()


def setup_quantized_model(
    model_id,
    quantizer,
    max_length,
    device=None,
    dtype=None,
    reserved_memory=8,
    print_model=False
):
    """Load model, prepare the quantized graph module, and dispatch to GPU(s).

    When device is not None (single GPU), moves the model to that device.
    When device is None (multi-GPU), dispatches the graph module across available
    GPUs after prepare_pt2e, reserving reserved_memory GiB per GPU for activations.

    Returns (model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        reserved_bytes = reserved_memory * 1024 ** 3
        max_memory = {
            k: v - reserved_bytes for k, v in get_max_memory().items()
            if isinstance(k, int) and v > reserved_bytes
        }
        device_map = get_device_map(gm, max_memory)
        dispatch_model(gm, device_map)

        for node in list(gm.graph.nodes):
            if node.op not in ["placeholder", "output"] and not node.users:
                gm.graph.erase_node(node)

        insert_align_device_nodes(gm, (input_ids, labels))

    return gm, tokenizer


def evaluate_perplexity(model, encodings, max_length, stride, device, num_steps=None):
    """Sliding-window perplexity evaluation. Returns a scalar tensor.

    If num_steps is set, exits early after that many windows — useful for
    activation observer calibration (caller can ignore the returned value).
    """
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    # Subtract max_length from seq_len to ensure that the last window has length max_length
    for i, begin_loc in enumerate(tqdm(range(0, seq_len - max_length, stride))):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, use_cache=False)

            # loss is calculated using CrossEntropyLoss which averages over valid
            # labels N.B. the model only calculates loss over trg_len - 1 labels,
            # because it internally shifts the labels to the left by 1.
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len or (num_steps is not None and i == num_steps - 1):
            break

    return torch.exp(torch.stack(nlls).mean())


@with_execution_context
def main(args):
    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else None
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

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
        device=device,
        dtype=torch_dtype,
        reserved_memory=args.reserved_memory,
        print_model=args.print_model,
    )

    if args.calibration_steps > 0:
        validation = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        calib_encodings = tokenizer("\n\n".join(validation["text"]), return_tensors="pt")

        evaluate_perplexity(
            model,
            calib_encodings,
            args.max_length,
            args.stride,
            device,
            num_steps=args.calibration_steps,
        )

        for module in model.modules():
            if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
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

    if args.record_histogram and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_histogram(model, args.output_dir)
        plot_layer_range(model, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
