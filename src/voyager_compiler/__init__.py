from google.protobuf import text_format
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from .cli_args import *
from .codegen import *
from .codegen.mapping_utils import flush_tensor_files
from .hardware import AcceleratorConfig
from .decomposed import *
from .fake_quantize import *
from .fp8 import *
from .histogram import *
from .llm_utils import *
from .normal_float import *
from .posit import *
from .pt2e_utils import *
from .qconfig import *
from .quantize import *
from .quantize_pt2e import *
from .quantizer import *
from .utils import *

__all__ = [
    "FusedAmaxObsFakeQuantize",
    "QConfig",
    "QuantizationSpec",
    "TorchExportableModuleWithStaticCache",
    "add_compile_args",
    "add_experiment_args",
    "add_quantization_args",
    "convert",
    "convert_and_export_with_split_cache",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "dispatch_model",
    "dtype_byte_size",
    "export_model",
    "fetch_attr",
    "generate",
    "get_aten_graph_module",
    "get_default_quantizer",
    "get_device_map",
    "get_node_name_to_scope",
    "get_qconfig",
    "insert_align_device_nodes",
    "plot_histogram",
    "plot_layer_range",
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
    "replace_softmax",
    "sink_obs_or_fq",
    "swap_llama_attention",
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
    use_interstellar_tiling=False,
    bufferize=False,
    context_len=None,
    max_gen=None,
):
    if example_kwargs is None:
        example_kwargs = {}

    # A null config (no hardware specified) skips padding / tiling, as passing
    # no ``unroll_dims`` / ``cache_size`` used to.
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    fake_mode = (
        FakeTensorMode(allow_non_fake_inputs=True) if use_fake_mode else None
    )

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # -------------------------------------------------------------------------
    # 1. Lowering & Decomposition
    # -------------------------------------------------------------------------
    # Break down complex operators (like MultiHeadAttention) into simpler
    # primitives and handle memory copy/concat operations.

    # Fold constant generators (input-free ``arange`` / ``zeros`` / … from e.g.
    # RoPE position setup) into ``get_attr`` buffers so they are not lowered or
    # scheduled as compute ops.
    fold_constant_generators(model)

    # Flatten autocast / no_grad (set_grad_enabled) wrap HOPs that torch.export
    # leaves around e.g. Llama's RoPE, so the wrap is not lowered as an op.
    inline_autocast_modules(model)

    # Delete identity ops (full slices, unit expands, same-dtype casts, zero-prob
    # dropout) that survive fusion — fewer nodes for every later pass and the
    # bufferizer to process.
    remove_prunable_ops(model)

    if not bufferize:
        split_multi_head_attention(model)
        convert_expand_to_memory_copy(model)
        convert_cat_and_stack_as_stack_on_dim0(model)
        convert_cat_with_mismatched_shapes_to_stack(model)

    fuse_quantize_dequantize_with_previous_op(
        model, bufferize, context_len, max_gen
    )

    # -------------------------------------------------------------------------
    # 2. Hardware Alignment (Padding)
    # -------------------------------------------------------------------------
    # Pad dimensions to align with hardware unrolling constraints (SIMD, systolic
    # array dimensions, etc.) to ensure efficient execution.
    if config.pe_array_size is not None:
        pad_matrix_op_dimensions(model, *config.pe_array_size)

    # -------------------------------------------------------------------------
    # 3. Matrix Operation Tiling
    # -------------------------------------------------------------------------
    # Apply L2 tiling logic specifically for matrix operations (GEMM/Conv) to
    # optimize for the specific cache size and memory bank configuration. This
    # annotates each GEMM/conv with ``l2_tiling`` (the bufferized builders read
    # it).
    if config.scratchpad_size is not None:
        run_matrix_op_l2_tiling(
            model,
            config,
            use_interstellar_tiling=use_interstellar_tiling,
        )

    # -------------------------------------------------------------------------
    # 4. Data Layout Transformation
    # -------------------------------------------------------------------------
    # Transform GEMM and convolution inputs/weights into layouts friendly
    # for systolic-array based hardware (e.g., transposing weights).

    if transform_layout:
        normalize_conv2d_layout(model)

    normalize_gemm_weight_layout(
        model,
        mm_layout="ck" if transform_layout else "kc",
        mv_layout="ck" if transpose_fc else "kc",
    )

    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # Remove redundant reshapes that have no effect on tensor semantics
    eliminate_reshape_with_no_effect(model)

    # -------------------------------------------------------------------------
    # 5. Vector Operation Tiling
    # -------------------------------------------------------------------------
    # TODO: Used for unit test. Will be removed in the future
    from .codegen.passes.lowering import split_dense_spmm_node

    if split_spmm:
        split_dense_spmm_node(model)

    if config.pe_array_size is not None:
        pad_vector_op_dimensions(model, config.vector_lanes)

    # Apply L2 tiling logic for vector-based operations.
    if config.scratchpad_size is not None:
        run_pool_op_l2_tiling(model, config)
        run_vector_op_l2_tiling(model, config)

    # -------------------------------------------------------------------------
    # 6. Operator Fusion
    # -------------------------------------------------------------------------
    # Perform final operator lowering and fuse sequences of operations (e.g.,
    # Conv+ReLU) into single kernels to reduce memory access overhead.

    if not skip_op_fusion:
        fuse_operator(model, patterns, fuse_reshape, bufferize)

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
    dump_snapshot=False,
    bufferize: bool = False,
):
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    os.makedirs(output_dir, exist_ok=True)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    # ---------------------------------------------------------------------
    # Optional add-on: bufferized FX lowering + loop-aware code generation.
    # Rewrites tiled GEMM/pointwise nodes into explicit while_loop nests over
    # buf.* primitives, then emits protobuf / graph / text from that FX graph.
    # Terminal alternative to the per-node run_memory_mapping + gen_code path.
    # ---------------------------------------------------------------------
    if bufferize:
        from .codegen.lowering import (
            bufferize_graph,
            gen_code_bufferized,
            gen_compute_graph_bufferized,
            plan_memory,
            print_bufferized_graph,
            report,
        )
        from .codegen.lowering.codegen import compute_op_names
        from .codegen.lowering.tiling import build_interstellar_tiler

        tiler = build_interstellar_tiler(config)

        gen_compute_graph(
            model, os.path.join(output_dir, output_file + "_prelowered")
        )

        # Reuse the tiler's double-buffering decision: when L2 is double-buffered the
        # tiles already assume two-buffer occupancy, so emit software-pipelined loops.
        bufferize_graph(model, pipelined=config.double_buffered_l2, tiler=tiler)

        # Assign concrete DRAM / Scratchpad addresses to the explicit buffers/tiles
        # (greedy best-fit DRAM reuse; bank-aware, region-scoped scratchpad).  Writes
        # meta['memory'] / meta['scratchpad'] that the proto emitter reads.
        plan_memory(model, config)
        print_bufferized_graph(model)

        # Estimate latency / DRAM traffic from the scheduled graph and dump a
        # live Excel workbook + Perfetto trace alongside the protobuf.
        result = report(
            model,
            config,
            output_dir=output_dir,
            basename=output_file,
        )
        print(
            f"  total latency      : {result.total_latency:,} cycles\n"
            f"  DRAM read  bytes   : {result.dram_read_bytes:,}\n"
            f"  DRAM write bytes   : {result.dram_write_bytes:,}\n"
            f"  DRAM total bytes   : "
            f"{result.dram_read_bytes + result.dram_write_bytes:,}\n"
            f"  scheduled events   : {len(result.records):,}"
        )

        path = os.path.join(output_dir, "tensor_files")
        params = gen_code_bufferized(
            model, flatten_args, path if dump_tensors else None
        )
        with open(os.path.join(output_dir, "model.txt"), "w") as f:
            f.write(text_format.MessageToString(params))
        with open(os.path.join(output_dir, "layers.txt"), "w") as f:
            f.write("\n".join(compute_op_names(model)))
        with open(os.path.join(output_dir, "bufferized_graph.txt"), "w") as f:
            f.write(print_bufferized_graph(model, to_string=True))
        gen_compute_graph_bufferized(
            model,
            os.path.join(output_dir, output_file),
            timeout=5 * 60,
        )
        flush_tensor_files()
        return params

    total_memory = (
        None
        if config.dram_size is None
        else int(config.dram_size * 1024**3)  # GB -> bytes
    )
    allocator = MemoryAllocator(total_memory, bank_width=config.bank_width)
    run_memory_mapping(model, config, allocator)

    if dump_snapshot:
        allocator.dump_snapshots(os.path.join(output_dir, "memory.png"))

    path = os.path.join(output_dir, "tensor_files")
    params = gen_code(model, flatten_args, path if dump_tensors else None)

    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(text_format.MessageToString(params))

    operations = [
        op.op.name if op.WhichOneof("op_type") == "op" else op.fused_op.name
        for op in params.ops
        if op.op.op != "nop"
    ]

    with open(os.path.join(output_dir, "layers.txt"), "w") as f:
        f.write("\n".join(operations))

    if len(model.graph.nodes) < 10000:
        gen_compute_graph(model, os.path.join(output_dir, output_file))

    flush_tensor_files()
    return params
