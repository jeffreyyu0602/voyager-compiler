# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lower a quantized PyTorch model onto a Voyager-generated accelerator.

``transform()`` runs the hardware-lowering passes over an exported FX graph;
``compile()`` bufferizes, plans memory and emits the ``voyager`` IR.  Both
operate in place and leave the graph executable, so every stage can be checked
numerically against the original.
"""

import logging
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
    remove_prunable_ops,
    remove_softmax_dtype_cast,
    remove_zero_attention_mask,
    bind_operand_sources,
    rename_nodes_with_param_names,
    replace_conv2d_with_im2col,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
)
from voyager_compiler.codegen.artifacts import dump_compile_artifacts
from voyager_compiler.codegen.execution_order import (
    DEFAULT_EXECUTION_MODES,
    require_execution_mode,
    validate_execution_order_mode,
)
from voyager_compiler.op_matcher import OpMatcher, get_op_overload
from voyager_compiler.codegen.targets import (
    apply_graph_rewrites as apply_target_graph_rewrites,
)
from voyager_compiler.codegen.targets import (
    execution_fusion_patterns as target_execution_fusion_patterns,
)
from voyager_compiler.codegen.targets import (
    execution_modes as target_execution_modes,
)
from voyager_compiler.codegen.targets import (
    layer_table as target_layer_table,
)
from voyager_compiler.codegen.targets import (
    scheduler_providers as target_scheduler_providers,
)
from voyager_compiler.codegen.targets import (
    terminal_codegen as target_terminal_codegen,
)
from voyager_compiler.codegen.targets import (
    unsupported_policy as target_unsupported_policy,
)
from voyager_compiler.codegen.transform.operator_fusion import (
    fuse_execution_instructions,
)
from voyager_compiler.codegen.transform.unit_assignment import (
    assign_compute_units,
    issuable_group,
)
from voyager_compiler.codegen.transform.bufferize import (
    bufferize_graph,
    flush_tensor_files,
    gen_code_bufferized,
    plan_memory,
    print_bufferized_graph,
    print_layer_table,
)
from voyager_compiler.codegen.transform.tiling import (
    DEFAULT_RUNTIME_TOLERANCE,
    build_interstellar_tiler,
)
from voyager_compiler.export_utils import (
    TorchExportableModuleWithStaticCache,
    convert_and_export_with_split_cache,
    export_model,
    get_aten_graph_module,
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
    FusedAmaxObsFakeQuantize,
    QConfig,
    QScheme,
    QuantizationConfig,
    QuantizationSpec,
    convert,
    convert_pt2e,
    derive_bias_qparams_fn,
    finalize_pt2e_calibration,
    get_default_quantizer,
    get_qconfig,
    prepare,
    prepare_pt2e,
    propagate_config,
    quantize,
    replace_softmax,
    sink_obs_or_fq,
)
from voyager_compiler.quantization.dtypes import (
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.quantization.modules import swap_llama_attention
from voyager_compiler.shape_prop import ShapeProp, fetch_attr, propagate_shape
from voyager_compiler.utils import with_execution_context

__all__ = [
    "AcceleratorConfig",
    "DerivedQuantizationSpec",
    "FusedAmaxObsFakeQuantize",
    "OpMatcher",
    "get_op_overload",
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
    "finalize_pt2e_calibration",
    "dispatch_model",
    "export_model",
    "extract_input_preprocessor",
    "fetch_attr",
    "fuse_dequantize_quantize",
    "fuse_operator",
    "issuable_group",
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


#: Re-exported from the leaf module so a target package can import them
#: without waiting on this one to finish initialising.  See
#: :mod:`voyager_compiler.op_matcher`.
_get_op_overload = get_op_overload


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
    context_len=None,
    max_new_tokens=None,
    target=None,
    rename_fused_operands=False,
):
    if example_kwargs is None:
        example_kwargs = {}

    # A null config (no hardware) skips padding and tiling.
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    fold_constant_generators(model)
    inline_autocast_modules(model)
    remove_prunable_ops(model)
    fuse_quantize_dequantize_with_producer(model, context_len, max_new_tokens)

    if config.pe_array_size is not None:
        pad_matrix_op_dimensions(model, *config.pe_array_size, config=config)

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
        fuse_operator(
            model,
            patterns,
            fuse_reshape,
            group_is_issuable=issuable_group(config),
        )

    rename_nodes_with_param_names(model)
    # Always record which outer node each fused placeholder is: an edge in
    # meta, invisible to the emitted IR. Rewriting the placeholder's *name* is
    # the opt-in half, because a node name is what TensorBox.node serializes.
    bound = bind_operand_sources(model, rename=rename_fused_operands)
    logging.getLogger(__name__).info(
        "bound %d fused-submodule placeholders (rename=%s)",
        bound, rename_fused_operands,
    )
    deduplicate_nodes(model)

    # Last, with the graph settled: a target rewrites what it must, and
    # declares any call its hardware cannot run.  A target with no backend
    # hooks leaves the graph exactly as the untargeted compiler produced it.
    apply_target_graph_rewrites(model, config, target)

    # Then group what the machine issues as one instruction. After the
    # rewrites, because these patterns name the operations the rewrites just
    # built; and unconditionally, because a target's instruction is not an
    # optimization -- ``skip_op_fusion`` turns off a performance pass, not the
    # difference between a record existing and not. A target that declares no
    # such sequence does nothing here, which is every target on the shared
    # path.
    instructions = fuse_execution_instructions(
        model, target_execution_fusion_patterns(target)
    )
    if instructions:
        logging.getLogger(__name__).info(
            "grouped %d declared target instructions", instructions
        )

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
    target=None,
    local_scheduler=None,
    scratchpad_mapping=None,
    execution_order=None,
):
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    # Reject a combination nothing can honour, and then reject one the path
    # about to run does not implement. The second question is asked of the
    # *path*, not of the compiler: a whole-model residency is not what the
    # bufferized planner does -- it packs lifetimes, which is the
    # per-operation answer -- so accepting the flag and quietly doing that
    # instead would report a placement the endpoint's addressing does not
    # describe. A target with its own terminal codegen answers for itself,
    # and answers with what it has implemented rather than with what its
    # machine wants.
    #
    # Resolved here, before anything is created, so that an unhonourable
    # request still fails before the compile leaves a directory behind.
    execution_order, scratchpad_mapping = validate_execution_order_mode(
        execution_order, scratchpad_mapping
    )
    emit_terminal = target_terminal_codegen(target)
    require_execution_mode(
        execution_order,
        scratchpad_mapping,
        target_execution_modes(target)
        if emit_terminal is not None
        else DEFAULT_EXECUTION_MODES,
    )

    os.makedirs(output_dir, exist_ok=True)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    # Refuse what no declared unit can issue, before anything is drawn or
    # built. This is the last point at which the graph is still the one a
    # reader recognizes -- fused, not yet bufferized -- so it is both where
    # the question is answerable and where the answer is worth showing.
    assign_compute_units(model, config)

    gen_compute_graph(model, os.path.join(output_dir, output_file))

    tolerance = (
        DEFAULT_RUNTIME_TOLERANCE
        if runtime_tolerance is None
        else runtime_tolerance
    )
    # The target chooses which scheduler plans its machine; one that names
    # none gets the engine the compiler ships with.
    # The target chooses the *set*; the command line chooses within it. A
    # target carrying one engine is unaffected by naming it or not.
    schedule = target_scheduler_providers(target)
    path = os.path.join(output_dir, "tensor_files")
    if emit_terminal is not None:
        # A target whose collateral is not the loop-nest IR ends here instead.
        # It does not bufferize: its operators have no bufferized lowering, so
        # the shared path would not produce worse output, it would produce
        # none. See ``codegen.targets.terminal_codegen``.
        #
        # The provider *set* is handed over; the per-compile ``TilerContext``
        # is not built here. That context is the shared engine's view of one
        # machine, and it is an input to ``bufferize_graph`` and to nothing
        # else -- so building it before the branch made every target pay for
        # a per-compile GEMM engine, including one whose operations are
        # claimed per operation and never tiled. Alpha declares exactly that:
        # a live ``operation_provider`` and no entry in ``providers``, on
        # which ``resolve`` raises. Constructing it eagerly therefore refused
        # the terminal path for the absence of an object the terminal path
        # does not use.
        params = emit_terminal(
            model,
            flatten_args,
            config=config,
            schedule=schedule,
            local_scheduler=local_scheduler,
            scratchpad_mapping=scratchpad_mapping,
            execution_order=execution_order,
            runtime_tolerance=tolerance,
            tensor_dir=path if dump_tensors else None,
        )
    else:
        bufferize_graph(
            model,
            pipelined=config.double_buffered_l2,
            tiler=schedule.resolve(local_scheduler)(
                config, runtime_tolerance=tolerance
            ),
            unsupported_policy=target_unsupported_policy(target),
        )
        plan_memory(model, config)
        print_bufferized_graph(model)

        params = gen_code_bufferized(
            model, flatten_args, path if dump_tensors else None
        )

    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(text_format.MessageToString(params))
    # A target whose collateral is not the loop-nest IR writes its own layer
    # list. The shared writer describes a bufferized nest and reads ``op.name``
    # from each emitted operation -- a field the flat message does not have --
    # so for such a target it raises rather than merely mis-describing.
    write_layers = target_layer_table(target)
    with open(os.path.join(output_dir, "layers.txt"), "w") as f:
        f.write(
            write_layers(model, params)
            if write_layers is not None
            else print_layer_table(model, params, to_string=True)
        )

    dump_compile_artifacts(model, config, output_dir, target=target)

    flush_tensor_files()
    return params
