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
    replace_conv2d_with_im2col,
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
from voyager_compiler.quantization.quantize import get_conv_bn_layers

from .utils import get_compile_args, get_transform_args

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def is_timm_model(args):
    """Whether the checkpoint is a timm one, loaded through timm itself.

    A ``timm/`` repository holds a timm-native checkpoint: its config names
    a timm architecture and its weights are keyed the timm way, so
    ``ViTForImageClassification`` cannot load it.
    """
    name = args.model_name_or_path
    return name is not None and name.startswith("timm/")


class TimmEmbeddings(torch.nn.Module):
    """The patch, class-token and position embeddings of a timm ViT.

    ``pad_vit_embeddings_output`` locates the embedding output by matching a
    pattern traced from a module. timm computes the class-token concat and
    the position add in ``_pos_embed``, a method rather than a module, so
    the pattern is assembled here.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model._pos_embed(self.model.patch_embed(pixel_values))


def get_logits(output):
    """timm returns the logits tensor; transformers wraps it in an output."""
    return output.logits if hasattr(output, "logits") else output


def load_model(args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    if is_timm_model(args):
        import timm

        model = timm.create_model(
            args.model_name_or_path.removeprefix("timm/"), pretrained=True
        )
        # The array consumes the attention matmuls individually; timm's
        # fused path exports as one scaled_dot_product_attention node.
        for block in model.blocks:
            block.attn.fused_attn = False
        return model.eval().to(torch_dtype)

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

    timm_model = is_timm_model(args)

    quantizer.set_module_name("head" if timm_model else "classifier", None)

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
            "^patch_embed.proj$"
            if timm_model
            else "^vit.embeddings.patch_embeddings.projection$",
            qconfig,
        )

    example_args = (calibration_data[0]["image"].to(torch_dtype),)
    vector_lanes = (
        args.pe_array_size[1] if args.pe_array_size is not None else None
    )

    embeddings = (
        TimmEmbeddings(model) if timm_model else model.vit.embeddings
    )

    gm = export_model(model, example_args)
    remove_zero_attention_mask(gm, example_args)
    pad_vit_embeddings_output(
        gm, embeddings, example_args, unroll=vector_lanes
    )

    if args.conv2d_im2col:
        replace_conv2d_with_im2col(gm)

    gm = prepare_pt2e(gm, quantizer)

    remove_softmax_dtype_cast(gm)

    for i in tqdm(range(args.calibration_steps), desc="Calibrating ViT"):
        inputs = calibration_data[i]["image"]
        with torch.no_grad():
            gm(inputs.to(torch_dtype))

    convert_pt2e(gm, args.bias)

    old_output = get_logits(gm(*example_args))

    transform(gm, example_args, **transform_args, skip_op_fusion=True)

    gm, preprocess_fn = extract_input_preprocessor(gm)
    example_args = (preprocess_fn(example_args[0]),)

    fuse_operator(gm, vector_stages)
    gm.graph.print_tabular()

    new_output = get_logits(gm(*example_args)) if args.debug else None

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

            logits = get_logits(model(image))
            prediction = torch.argmax(logits, dim=-1)
            if prediction.item() == label:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Vit Accuracy: {accuracy:.4f}")
