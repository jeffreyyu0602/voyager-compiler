# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler workload adapter for the public YOLOv5n-0.5 checkpoint."""

import os
import re
import sys
import types
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from voyager_compiler import (
    DerivedQuantizationSpec,
    QScheme,
    QuantizationConfig,
    QuantizationSpec,
    compile,
    convert_pt2e,
    derive_bias_qparams_fn,
    export_model,
    extract_input_preprocessor,
    finalize_pt2e_calibration,
    prepare_pt2e,
    transform,
)
from voyager_compiler.quantization.quantize import get_conv_bn_layers

from utils.pairs import yolo5n_0_5_magnet_alpha as alpha_policy

from .utils import get_compile_args, get_transform_args

WORKLOAD = "yolo5n-0.5"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
YOLO_ROOT_ENVIRONMENT_VARIABLE = "VOYAGER_YOLO_ROOT"
DEFAULT_YOLO_ROOT = REPOSITORY_ROOT / "libraries" / "yolov5-face"
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "libraries" / "weights" / "yolov5n-0.5.pt"
)
INPUT_SIZE = 192


class _InputChannelPaddedConv2d(nn.Conv2d):
    """Conv2d whose logical input is zero-padded to a physical channel word."""

    def __init__(self, conv, padded_in_channels):
        if conv.groups != 1 or padded_in_channels < conv.in_channels:
            raise ValueError(
                "Input-channel padding requires a non-grouped expansion"
            )
        super().__init__(
            in_channels=padded_in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        self.logical_in_channels = conv.in_channels
        self.input_channel_padding = padded_in_channels - conv.in_channels
        padded_weight = F.pad(
            conv.weight.detach(),
            (0, 0, 0, 0, 0, self.input_channel_padding),
        )
        self.weight = nn.Parameter(
            padded_weight,
            requires_grad=conv.weight.requires_grad,
        )
        if conv.bias is not None:
            self.bias = nn.Parameter(
                conv.bias.detach().clone(),
                requires_grad=conv.bias.requires_grad,
            )
        self.train(conv.training)

    def forward(self, inputs):
        inputs = F.pad(inputs, (0, 0, 0, 0, 0, self.input_channel_padding))
        return self._conv_forward(inputs, self.weight, self.bias)


def _clear_yolo_imports():
    """Avoid resolving a previously imported, incompatible YOLO package."""
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]


def _install_import_stubs():
    """Stub optional plotting/profiling dependencies unused during export."""
    if "seaborn" not in sys.modules:
        try:
            __import__("seaborn")
        except ImportError:
            sys.modules["seaborn"] = types.ModuleType("seaborn")

    if "thop" not in sys.modules:
        try:
            __import__("thop")
        except ImportError:
            thop_stub = types.ModuleType("thop")
            thop_stub.profile = lambda *args, **kwargs: (0, 0)
            thop_stub.clever_format = lambda values, *args, **kwargs: values
            sys.modules["thop"] = thop_stub


def _load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"].float().eval()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        from models.yolo import Model

        model = Model(checkpoint["model_cfg_yaml:"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.float().eval()
    if isinstance(checkpoint, nn.Module):
        return checkpoint.float().eval()
    raise ValueError(f"Unsupported YOLO checkpoint format: {path}")


def _rewrite_first_layer_grayscale(model):
    first_name = None
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_name = name
            first_conv = module
            break

    if first_conv is None or first_name is None:
        raise ValueError("YOLO model has no Conv2d input layer")
    if first_conv.in_channels == 1:
        return first_name
    if first_conv.in_channels != 3 or first_conv.groups != 1:
        raise ValueError(
            "Expected the first YOLO Conv2d to be an RGB, non-grouped layer; "
            f"found in_channels={first_conv.in_channels}, groups={first_conv.groups}"
        )

    grayscale = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=1,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
        device=first_conv.weight.device,
        dtype=first_conv.weight.dtype,
    )
    with torch.no_grad():
        grayscale.weight.copy_(
            first_conv.weight.detach().sum(dim=1, keepdim=True)
        )
        if first_conv.bias is not None:
            grayscale.bias.copy_(first_conv.bias.detach())
    grayscale.train(first_conv.training)

    parent_name, _, child_name = first_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, grayscale)
    return first_name


def _replace_silu_with_relu(module):
    for child_name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, child_name, nn.ReLU())
        else:
            _replace_silu_with_relu(child)


def _fuse_conv_bn(model):
    modules_to_fuse = get_conv_bn_layers(model)
    if modules_to_fuse:
        torch.ao.quantization.fuse_modules(
            model,
            modules_to_fuse,
            inplace=True,
        )


def _project_depthwise_scalar_bias(model):
    """Project fused depthwise biases onto Alpha's broadcast scalar."""
    projected = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if module.groups != module.in_channels:
                continue
            if module.out_channels != module.in_channels:
                continue
            if module.bias is None:
                continue
            scalar = module.bias.detach().mean()
            module.bias.copy_(scalar.expand_as(module.bias))
            projected += 1
    return projected


def _rewrite_input_channel_padding(model):
    for module_name, padded_channels in alpha_policy.INPUT_CHANNEL_PADDING:
        conv = model.get_submodule(module_name)
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(
                f"{module_name} must be Conv2d, got {type(conv).__name__}"
            )
        model.set_submodule(
            module_name,
            _InputChannelPaddedConv2d(conv, padded_channels),
        )


def _ensure_detect_helpers(model):
    for module in model.modules():
        if module.__class__.__name__ != "Detect":
            continue
        if not hasattr(module, "idents"):
            idents = []
            for index in range(getattr(module, "nl", 0)):
                ident = nn.Identity()
                setattr(module, f"ident_{index}", ident)
                idents.append(ident)
            module.idents = idents
        if not hasattr(module, "output_probes"):
            probes = []
            for index in range(getattr(module, "nl", 0)):
                probe = nn.Identity()
                setattr(module, f"output_probe_{index}", probe)
                probes.append(probe)
            module.output_probes = nn.ModuleList(probes)


def _set_raw_head_export(model):
    for module in model.modules():
        if hasattr(module, "export"):
            module.export = False
            module.train()
        if hasattr(module, "export_cat"):
            module.export_cat = False
            module.train()


def load_model(args, torch_dtype=torch.float32):
    yolo_root = Path(
        os.environ.get(YOLO_ROOT_ENVIRONMENT_VARIABLE, DEFAULT_YOLO_ROOT)
    ).resolve()
    if not yolo_root.is_dir():
        raise FileNotFoundError(
            f"YOLO source tree not found at {yolo_root}. Set "
            f"{YOLO_ROOT_ENVIRONMENT_VARIABLE} or place it under libraries/."
        )

    yolo_root_string = str(yolo_root)
    if yolo_root_string in sys.path:
        sys.path.remove(yolo_root_string)
    sys.path.insert(0, yolo_root_string)

    checkpoint = Path(args.model_name_or_path or DEFAULT_CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found at {checkpoint}; pass "
            "--model_name_or_path or place yolov5n-0.5.pt under "
            "libraries/weights/."
        )
    args.model_name_or_path = str(checkpoint)

    voyager_utils = {
        name: module
        for name, module in sys.modules.items()
        if name == "utils" or name.startswith("utils.")
    }
    _clear_yolo_imports()
    _install_import_stubs()
    try:
        model = _load_checkpoint(checkpoint)
    finally:
        _clear_yolo_imports()
        sys.modules.update(voyager_utils)
        sys.path.remove(yolo_root_string)

    first_layer = _rewrite_first_layer_grayscale(model)
    _replace_silu_with_relu(model)
    _fuse_conv_bn(model)
    _project_depthwise_scalar_bias(model)
    _rewrite_input_channel_padding(model)
    _ensure_detect_helpers(model)
    if args.bf16:
        model.bfloat16()
    _set_raw_head_export(model)

    first_conv = model.get_submodule(first_layer)
    if not isinstance(first_conv, nn.Conv2d) or first_conv.in_channels != 1:
        raise RuntimeError(
            "Failed to rewrite the first YOLO layer to grayscale"
        )
    example_args = (
        torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE, dtype=torch_dtype),
    )
    return model, example_args


def _uses_quantization(args):
    return args.pt2e or any(
        spec is not None
        for spec in (
            args.activation,
            args.output_activation,
            args.weight,
            args.bias,
        )
    )


def configure_quantizer(quantizer, args):
    """Apply the per-layer Alpha operand policy to the target quantizer."""
    if not _uses_quantization(args):
        return quantizer

    int8_activation = QuantizationSpec.from_str(
        alpha_policy.W8A8_ACTIVATION_QSPEC
    )
    int8_weight = QuantizationSpec.from_str(alpha_policy.W8_WEIGHT_QSPEC)
    depthwise_weight = QuantizationSpec.from_str(
        alpha_policy.DEPTHWISE_W8_WEIGHT_QSPEC
    )
    bias = DerivedQuantizationSpec(
        derived_from=None,
        derive_qparams_fn=derive_bias_qparams_fn,
        dtype=None,
    )
    int8_config = QuantizationConfig(
        int8_activation,
        None,
        int8_weight,
        bias,
    )
    depthwise_config = QuantizationConfig(
        int8_activation,
        None,
        depthwise_weight,
        bias,
    )
    for layer in alpha_policy.W8_LAYERS:
        quantizer.set_module_name(f"^{re.escape(layer)}$", int8_config)
    for layer in alpha_policy.DEPTHWISE_W8_LAYERS:
        quantizer.set_module_name(f"^{re.escape(layer)}$", depthwise_config)

    head_input = replace(
        QuantizationSpec.from_str(args.activation),
        ch_axis=1,
    )
    head_weight = replace(
        QuantizationSpec.from_str(args.weight),
        ch_axis=1,
    )
    if (
        head_weight.coarse is not None
        and head_weight.coarse.qscheme == QScheme.PER_CHANNEL_SYMMETRIC
    ):
        head_weight = replace(
            head_weight,
            coarse=replace(head_weight.coarse, ch_axis=0),
        )
    head_config = QuantizationConfig(
        head_input,
        int8_activation,
        head_weight,
        None,
    )
    for layer in alpha_policy.HEAD_LAYERS:
        quantizer.set_module_name(f"^{re.escape(layer)}$", head_config)

    nongemm = QuantizationSpec.from_str(alpha_policy.NONGEMM_QSPEC)
    nongemm_config = QuantizationConfig(nongemm, None, None, None)
    quantizer.set_object_type(torch.ops.aten.add.Tensor, nongemm_config)
    quantizer.set_object_type(torch.ops.aten.add_.Tensor, nongemm_config)
    return quantizer


def retrieve_dataset(args, example_args):
    """Return deterministic synthetic calibration frames when requested."""
    if args.calibration_steps <= 0 or not _uses_quantization(args):
        return ()
    return tuple(
        (torch.randn_like(example_args[0]),)
        for _ in range(args.calibration_steps)
    )


def quantize_and_dump_model(
    model,
    example_args,
    quantizer,
    calibration_data,
    vector_stages,
    args,
):
    transform_args = get_transform_args(args, vector_stages)
    compile_args = get_compile_args(args)

    if _uses_quantization(args):
        gm = prepare_pt2e(
            model,
            quantizer,
            example_args,
            region_qspec=QuantizationSpec.from_str(alpha_policy.NONGEMM_QSPEC),
            target=args.target,
        )
        for inputs in tqdm(calibration_data, desc="Calibrating YOLOv5n-0.5"):
            with torch.no_grad():
                gm(*inputs)
        finalize_pt2e_calibration(gm, target=args.target)
        convert_pt2e(gm, args.bias, target=args.target)
    else:
        gm = export_model(model, example_args)

    gm.graph.eliminate_dead_code()
    gm.recompile()
    old_output = gm(*example_args)

    transform(gm, example_args, **transform_args, skip_op_fusion=True)
    gm, preprocess_fn = extract_input_preprocessor(gm)
    compiled_args = (preprocess_fn(*example_args),)
    new_output = gm(*compiled_args) if args.debug else None

    compile(gm, compiled_args, **compile_args)
    return gm, old_output, new_output, preprocess_fn


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_YOLO_ROOT",
    "WORKLOAD",
    "configure_quantizer",
    "load_model",
    "quantize_and_dump_model",
    "retrieve_dataset",
]
