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
from torch.utils._pytree import tree_flatten

# Defines torch.ops.quantized_ops.* (and the hardware-layout twins).  This must
# come first: the imports below reference those targets, and the op namespace
# resolves lazily, so an unregistered target fails at call time rather than
# here.
from voyager_compiler import ops as _register_ops  # noqa: F401
from voyager_compiler.cli_args import (
    add_compile_args,
    add_experiment_args,
    add_quantization_args,
)
from voyager_compiler.codegen import (
    deduplicate_nodes,
    extract_input_preprocessor,
    fold_constant_generators,
    fuse_dequantize_quantize,
    fuse_operator,
    fuse_quantize_dequantize_with_producer,
    gen_compute_graph,
    inline_autocast_modules,
    normalize_conv2d_layout,
    normalize_gemm_weight_layout,
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    pad_vit_embeddings_output,
    remove_fp32_casts,
    remove_prunable_ops,
    remove_zero_attention_mask,
    rename_nodes_with_param_names,
    replace_conv2d_with_im2col,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
    scalarize_index_arithmetic,
    split_kv_cache,
)
from voyager_compiler.codegen.transform.bufferize import (
    bufferize_graph,
    flush_tensor_files,
    gen_code_bufferized,
    plan_memory,
    print_bufferized_graph,
    print_layer_table,
    shared_dram_layout,
)
from voyager_compiler.codegen.transform.tiling import (
    DEFAULT_RUNTIME_TOLERANCE,
    build_interstellar_tiler,
)
from voyager_compiler.export_utils import (
    export_model,
    get_aten_graph_module,
    get_conv_bn_layers,
    get_node_name_to_scope,
    print_node_scope_tabular,
)
from voyager_compiler.hardware_config import AcceleratorConfig
from voyager_compiler.modeling import (
    dispatch_model,
    get_device_map,
    insert_align_device_nodes,
)
from voyager_compiler.ops.layout import (
    DEFAULT_GEMM_WEIGHT_LAYOUT,
    DEFAULT_LAYOUT_POLICY,
    POLICY_GEMM_WEIGHT_LAYOUT,
)
from voyager_compiler.quantization import (
    DerivedQuantizationSpec,
    DirectCastFakeQuantize,
    FusedAmaxObsFakeQuantize,
    GroupWiseAffineFakeQuantize,
    MXFakeQuantize,
    QScheme,
    QuantizationConfig,
    QuantizationSpec,
    TrainingQuantizers,
    capture_training,
    convert_pt2e,
    derive_bias_qparams_fn,
    disable_observers,
    fold_conv_bn_qat,
    freeze_cache_reads,
    freeze_weights,
    get_default_quantizer,
    gradient_program,
    make_spec,
    prepare_from_args,
    prepare_pt2e,
    prepare_qat_pt2e,
    prepare_training,
    set_batch_norm_training,
    set_training,
    sink_obs_or_fq,
    update_program,
)
from voyager_compiler.quantization.dtypes import (
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.shape_prop import (
    ShapeProp,
    fake_like,
    fetch_attr,
    propagate_shape,
)
from voyager_compiler.utils import with_execution_context

__all__ = [
    "AcceleratorConfig",
    "DerivedQuantizationSpec",
    "DirectCastFakeQuantize",
    "FusedAmaxObsFakeQuantize",
    "GroupWiseAffineFakeQuantize",
    "MXFakeQuantize",
    "OpMatcher",
    "QScheme",
    "QuantizationConfig",
    "QuantizationSpec",
    "ShapeProp",
    "TrainingQuantizers",
    "add_compile_args",
    "add_experiment_args",
    "add_quantization_args",
    "capture_training",
    "compile",
    "convert_pt2e",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "disable_observers",
    "dispatch_model",
    "export_model",
    "extract_input_preprocessor",
    "fetch_attr",
    "fold_conv_bn_qat",
    "freeze_cache_reads",
    "freeze_weights",
    "fuse_dequantize_quantize",
    "fuse_operator",
    "get_aten_graph_module",
    "get_conv_bn_layers",
    "get_default_quantizer",
    "get_device_map",
    "get_node_name_to_scope",
    "gradient_program",
    "insert_align_device_nodes",
    "make_spec",
    "pad_vit_embeddings_output",
    "prepare_from_args",
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "prepare_training",
    "print_node_scope_tabular",
    "propagate_shape",
    "quantize_to_nf",
    "quantize_to_posit",
    "remove_fp32_casts",
    "remove_zero_attention_mask",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "replace_rmsnorm_with_layer_norm",
    "scalarize_index_arithmetic",
    "set_batch_norm_training",
    "set_training",
    "shared_dram_layout",
    "sink_obs_or_fq",
    "split_kv_cache",
    "transform",
    "update_program",
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
    layout_policy=DEFAULT_LAYOUT_POLICY,
    gemv_weight_layout=DEFAULT_GEMM_WEIGHT_LAYOUT,
    skip_op_fusion=False,
    fuse_reshape=True,
    keep_fp32=True,
):
    """Lower ``model`` in place through the graph-level passes.

    The passes run on fake copies of the example inputs and read only their
    shapes (see ``shape_prop``).

    Args:
        model: The exported graph, rewritten in place.
        example_args: Example positional inputs.
        example_kwargs: Example keyword inputs; ``None`` for none.
        patterns: The fusion patterns ``fuse_operator`` applies, each a list
            of ``OpMatcher`` from anchor to tail.
        config: The accelerator's ``AcceleratorConfig``; ``None`` models no
            hardware and skips padding.
        layout_policy: The data-layout policy: ``"systolic"`` puts conv2d in
            NHWC, and the policy picks the matrix-matrix GEMM weight layout.
        gemv_weight_layout: The weight layout of a matrix-vector GEMM.
        skip_op_fusion: Skip operator fusion.
        fuse_reshape: Let fusion fold a trailing reshape or permute into
            its producer's output.
        keep_fp32: ``False`` computes in 16-bit float where the model casts
            up to float32 (``remove_fp32_casts``).

    Returns:
        ``model``.
    """
    if example_kwargs is None:
        example_kwargs = {}

    # A null config (no hardware) skips padding and tiling.
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    if not keep_fp32:
        remove_fp32_casts(model)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*map(fake_like, flatten_args))

    fold_constant_generators(model)
    inline_autocast_modules(model)
    remove_prunable_ops(model)
    scalarize_index_arithmetic(model)
    fuse_quantize_dequantize_with_producer(model)

    if config.pe_array_size is not None:
        pad_matrix_op_dimensions(model, config.pe_array_size)

    if layout_policy == "systolic":
        normalize_conv2d_layout(model)

    normalize_gemm_weight_layout(
        model,
        mm_layout=POLICY_GEMM_WEIGHT_LAYOUT[layout_policy],
        mv_layout=gemv_weight_layout,
    )

    if config.pe_array_size is not None:
        pad_vector_op_dimensions(model, config.vector_lanes)

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
    dram_layout=None,
    accumulate_fp32=False,
):
    """Bufferize ``model``, plan its memory and write its ``voyager`` IR.

    Writes ``model.txt`` (the IR as text), ``layers.txt`` (a table of its
    layers) and ``<output_file>.svg`` (the compute graph) to ``output_dir``.

    Args:
        model: A graph ``transform`` lowered, bufferized in place.
        example_args: The inputs ``model`` runs on, so one it writes, such
            as a gradient buffer, is overwritten.
        example_kwargs: Keyword inputs; ``None`` for none.
        config: The accelerator's ``AcceleratorConfig``; ``None`` models no
            hardware.
        output_dir: The directory the files are written to.
        output_file: The compute-graph drawing's basename.
        dump_tensors: Also write every tensor's values under
            ``output_dir/tensor_files``.
        runtime_tolerance: How much longer than the fastest modeled tiling
            a tiling may run and still be chosen, as a fraction; ``None``
            takes ``DEFAULT_RUNTIME_TOLERANCE``.
        dram_layout: The DRAM ranges ``shared_dram_layout`` agreed for the
            inputs ``model`` shares with other programs, such as a training
            step's gradient and update programs; ``None`` places every
            input freely.
        accumulate_fp32: Accumulate a GEMM split along K in float32 rather
            than its output dtype.

    Returns:
        The ``voyager`` IR ``Model`` written to ``model.txt``.
    """
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    os.makedirs(output_dir, exist_ok=True)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*map(fake_like, flatten_args))

    gen_compute_graph(model, os.path.join(output_dir, output_file))

    tolerance = (
        DEFAULT_RUNTIME_TOLERANCE
        if runtime_tolerance is None
        else runtime_tolerance
    )
    tiler = build_interstellar_tiler(
        config, runtime_tolerance=tolerance, accumulate_fp32=accumulate_fp32
    )
    bufferize_graph(model, pipelined=config.double_buffered_l2, tiler=tiler)
    plan_memory(model, config, dram_layout)
    print_bufferized_graph(model)

    path = os.path.join(output_dir, "tensor_files")
    params = gen_code_bufferized(
        model, flatten_args, path if dump_tensors else None
    )

    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(text_format.MessageToString(params))
    with open(os.path.join(output_dir, "layers.txt"), "w") as f:
        f.write(print_layer_table(model, params, to_string=True))

    flush_tensor_files()
    return params
