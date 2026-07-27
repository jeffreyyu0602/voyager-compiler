import re
import torch
from tqdm import tqdm

from transformers import ViTForImageClassification
from transformers.utils import logging

from voyager_compiler import (
    DerivedQuantizationSpec,
    QuantizationConfig,
    QuantizationSpec,
    convert_pt2e,
    export_model,
    prepare_pt2e,
    transform,
    compile,
    derive_bias_qparams_fn,
    extract_input_preprocessor,
    fuse_operator,
)
from voyager_compiler.codegen import (
    pad_vit_embeddings_output,
    remove_softmax_dtype_cast,
    remove_zero_attention_mask,
)
from voyager_compiler.pt2e_utils import get_conv_bn_layers

from .utils import get_compile_args, get_transform_args

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_model(args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model_name_or_path = (
        args.model_name_or_path or "google/vit-base-patch16-224"
    )

    return ViTForImageClassification.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    )


def quantize_and_dump_model(
    model, quantizer, calibration_data, vector_stages, args
):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    modules_to_fuse = get_conv_bn_layers(model)
    if len(modules_to_fuse) > 0:
        model = torch.ao.quantization.fuse_modules(
            model, modules_to_fuse, inplace=True
        )

    quantizer.set_module_name("classifier", None)

    if args.activation is not None and "microscaling" in args.activation:
        dtype = args.activation.split(",")[0]
        match = re.fullmatch(r"nf(\d+)(?:_(\d+))?", dtype, re.IGNORECASE)
        if match is not None and match.group(2) is not None:
            dtype = f"int{match.group(2)}"
        qspec = QuantizationSpec.from_str(f"{dtype},qs=per_tensor_symmetric")

        bias_qspec = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=None,
        )

        qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name(
            "^vit.embeddings.patch_embeddings.projection$", qconfig
        )

    example_args = (calibration_data[0]["image"].to(torch_dtype),)
    vector_lanes = (
        args.pe_array_size[1] if args.pe_array_size is not None else None
    )

    gm = export_model(model, example_args)
    remove_zero_attention_mask(gm, example_args)
    pad_vit_embeddings_output(
        gm, model.vit.embeddings, example_args, unroll=vector_lanes
    )

    gm = prepare_pt2e(gm, quantizer)

    remove_softmax_dtype_cast(gm)

    for i in tqdm(range(args.calibration_steps), desc="Calibrating ViT"):
        inputs = calibration_data[i]["image"]
        with torch.no_grad():
            gm(inputs.to(torch_dtype))

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args).logits

    transform(gm, example_args, **transform_args, skip_op_fusion=True)

    gm, preprocess_fn = extract_input_preprocessor(gm)
    example_args = (preprocess_fn(example_args[0]),)

    fuse_operator(gm, vector_stages)
    gm.graph.print_tabular()

    new_output = gm(*example_args).logits if args.debug else None

    compile(gm, example_args, **compile_args)
    return gm, old_output, new_output, preprocess_fn


def evaluate(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for image_label_pair in tqdm(dataset, desc="Evaluating ViT"):
            # for running the original model without the preprocessing function
            # applied to the dataset
            image = image_label_pair["image"].to(device)
            label = image_label_pair["label"]

            outputs = model(image)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            if prediction.item() == label:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Vit Accuracy: {accuracy:.4f}")
