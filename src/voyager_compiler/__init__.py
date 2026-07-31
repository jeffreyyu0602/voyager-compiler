"""Lower a quantized PyTorch model onto a Voyager-generated accelerator.

``transform()`` runs the hardware-lowering passes over an exported FX graph;
``compile()`` bufferizes, plans memory and emits the ``voyager`` IR.  Both
operate in place and leave the graph executable, so every stage can be checked
numerically against the original.
"""

import os
from typing import Callable, Optional, Tuple

import torch
from google.protobuf import text_format
from torch.fx import Node
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

# Defines torch.ops.quantized_ops.* (and the hardware-layout twins).  This must
# come first: the imports below reference those targets, and the op namespace
# resolves lazily, so an unregistered target fails at call time rather than
# here.
from . import ops as _register_ops  # noqa: F401

from .cli_args import (
    add_compile_args,
    add_experiment_args,
    add_quantization_args,
)
from .codegen import (
    deduplicate_nodes,
    eliminate_reshape_with_no_effect,
    extract_input_preprocessor,
    fold_constant_generators,
    fuse_dequantize_quantize,
    fuse_operator,
    fuse_quantize_dequantize_with_previous_op,
    gen_compute_graph,
    inline_autocast_modules,
    normalize_conv2d_layout,
    normalize_gemm_weight_layout,
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    pad_vit_embeddings_output,
    remove_prunable_ops,
    remove_softmax_dtype_cast,
    remove_zero_attention_mask,
    rename_nodes_with_param_names,
    replace_conv2d_with_im2col,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
    split_dense_spmm_node,
)
from .codegen.transform.bufferize import (
    bufferize_graph,
    compute_op_names,
    flush_tensor_files,
    gen_code_bufferized,
    gen_compute_graph_bufferized,
    plan_memory,
    print_bufferized_graph,
)
from .codegen.transform.tiling import (
    DEFAULT_RUNTIME_TOLERANCE,
    build_interstellar_tiler,
)
from .export_utils import (
    TorchExportableModuleWithStaticCache,
    convert_and_export_with_split_cache,
    export_model,
    get_aten_graph_module,
    get_node_name_to_scope,
    print_node_scope_tabular,
)
from .hardware_config import AcceleratorConfig
from .modeling import (
    dispatch_model,
    get_device_map,
    insert_align_device_nodes,
)
from .quantization import (
    DerivedQuantizationSpec,
    FusedAmaxObsFakeQuantize,
    QConfig,
    QScheme,
    QuantizationConfig,
    QuantizationSpec,
    convert,
    convert_pt2e,
    derive_bias_qparams_fn,
    get_default_quantizer,
    get_qconfig,
    prepare,
    prepare_pt2e,
    propagate_config,
    quantize,
    replace_softmax,
    sink_obs_or_fq,
)
from .quantization.dtypes import (
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
    quantize_to_nf,
    quantize_to_posit,
)
from .quantization.modules import swap_llama_attention
from .shape_prop import ShapeProp, fetch_attr, propagate_shape
from .utils import with_execution_context

__all__ = [
    "AcceleratorConfig",
    "DerivedQuantizationSpec",
    "FusedAmaxObsFakeQuantize",
    "OpMatcher",
    "QConfig",
    "QScheme",
    "QuantizationConfig",
    "QuantizationSpec",
    "ShapeProp",
    "TorchExportableModuleWithStaticCache",
    "add_compile_args",
    "add_experiment_args",
    "add_quantization_args",
    "compile",
    "convert",
    "convert_and_export_with_split_cache",
    "convert_pt2e",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "dispatch_model",
    "export_model",
    "extract_input_preprocessor",
    "fetch_attr",
    "fuse_dequantize_quantize",
    "fuse_operator",
    "get_aten_graph_module",
    "get_default_quantizer",
    "get_device_map",
    "get_node_name_to_scope",
    "get_qconfig",
    "insert_align_device_nodes",
    "pad_vit_embeddings_output",
    "prepare",
    "prepare_pt2e",
    "print_node_scope_tabular",
    "propagate_config",
    "propagate_shape",
    "quantize",
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "quantize_to_nf",
    "quantize_to_posit",
    "remove_softmax_dtype_cast",
    "remove_zero_attention_mask",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "replace_softmax",
    "sink_obs_or_fq",
    "swap_llama_attention",
    "transform",
    "with_execution_context",
]


class qscheme: ...


# Defined in voyager_compiler/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
group_wise_affine: qscheme = QScheme.GROUP_WISE_AFFINE


def _get_op_overload(op_name: str):
    all_overloads = []
    for lib in [torch.ops.aten, torch.ops.quantized_ops]:
        # Also check inplace version of the op (e.g., "add_" for "add")
        for name in [op_name, f"{op_name}_"]:
            if (packet := getattr(lib, name, None)) is None:
                continue
            all_overloads.extend(
                [getattr(packet, name) for name in packet.overloads()]
            )
    return all_overloads


class OpMatcher:
    targets: Tuple[torch._ops.OpOverload]
    predicate: Optional[Callable[[Node], bool]] = None

    def __init__(self, *ops, predicate=None):
        self.predicate = predicate

        # Resolve symbolic ops
        targets = []
        for op in ops:
            targets.extend(_get_op_overload(op))

        # Freeze resolved targets
        self.targets = tuple(targets)

    def matches(self, node: Node) -> bool:
        if node.target not in self.targets:
            return False

        return self.predicate(node) if self.predicate else True


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
    config=None,
    transform_layout=False,
    transpose_fc=False,
    skip_op_fusion=False,
    fuse_reshape=True,
    split_spmm=False,
    use_fake_mode=True,
    context_len=None,
    max_gen=None,
):
    if example_kwargs is None:
        example_kwargs = {}

    # A null config (no hardware) skips padding and tiling.
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    fake_mode = (
        FakeTensorMode(allow_non_fake_inputs=True) if use_fake_mode else None
    )

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # Fold input-free ``arange`` / ``zeros`` (RoPE setup) into ``get_attr``.
    fold_constant_generators(model)

    # Flatten the autocast / no_grad wrap HOPs ``torch.export`` leaves behind.
    inline_autocast_modules(model)

    # Delete identity ops: full slices, unit expands, no-op casts, p=0 dropout.
    remove_prunable_ops(model)

    fuse_quantize_dequantize_with_previous_op(model, context_len, max_gen)

    # Pad dimensions to the hardware unrolling.
    if config.pe_array_size is not None:
        pad_matrix_op_dimensions(model, *config.pe_array_size)

    # Systolic-array-friendly operand layouts.
    if transform_layout:
        normalize_conv2d_layout(model)

    normalize_gemm_weight_layout(
        model,
        mm_layout="ck" if transform_layout else "kc",
        mv_layout="ck" if transpose_fc else "kc",
    )

    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # Drop reshapes that do not change tensor semantics.
    eliminate_reshape_with_no_effect(model)

    if split_spmm:
        split_dense_spmm_node(model)

    if config.pe_array_size is not None:
        pad_vector_op_dimensions(model, config.vector_lanes)

    # Fuse op sequences (e.g. Conv+ReLU) into one kernel.
    if not skip_op_fusion:
        fuse_operator(model, patterns, fuse_reshape)

    rename_nodes_with_param_names(model)
    deduplicate_nodes(model)

    return model


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    config=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    runtime_tolerance=None,
):
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    os.makedirs(output_dir, exist_ok=True)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    tolerance = (
        DEFAULT_RUNTIME_TOLERANCE
        if runtime_tolerance is None
        else runtime_tolerance
    )
    tiler = build_interstellar_tiler(config, runtime_tolerance=tolerance)

    gen_compute_graph(
        model, os.path.join(output_dir, output_file + "_prelowered")
    )

    bufferize_graph(model, pipelined=config.double_buffered_l2, tiler=tiler)

    plan_memory(model, config)
    print_bufferized_graph(model)

    path = os.path.join(output_dir, "tensor_files")
    params = gen_code_bufferized(
        model, flatten_args, path if dump_tensors else None
    )

    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(text_format.MessageToString(params))
    with open(os.path.join(output_dir, "layers.txt"), "w") as f:
        f.write("\n".join(compute_op_names(model)))

    gen_compute_graph_bufferized(
        model,
        os.path.join(output_dir, output_file),
        timeout=5 * 60,
    )

    flush_tensor_files()
    return params
