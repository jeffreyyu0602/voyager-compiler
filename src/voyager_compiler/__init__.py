from google.protobuf import text_format
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from .cli_args import *
from .codegen import *
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
    unroll_dims=None,
    transform_layout=False,
    transpose_fc=False,
    cache_size=None,
    num_banks=None,
    skip_op_fusion=False,
    fuse_reshape=True,
    split_spmm=False,
    use_fake_mode=True,
    use_interstellar_tiling=False,
    bufferize=False,
):
    if example_kwargs is None:
        example_kwargs = {}

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

    if not bufferize:
        split_multi_head_attention(model)

    # TODO Disabled for large models. This will be removed in the future once
    # we can handle stack/cat using DMA properly.
    if len(model.graph.nodes) < 10000:
        convert_expand_to_memory_copy(model)
        convert_cat_and_stack_as_stack_on_dim0(model)
        convert_cat_with_mismatched_shapes_to_stack(model)

    fuse_quantize_dequantize_with_previous_op(model)

    # -------------------------------------------------------------------------
    # 2. Hardware Alignment (Padding)
    # -------------------------------------------------------------------------
    # Pad dimensions to align with hardware unrolling constraints (SIMD, systolic
    # array dimensions, etc.) to ensure efficient execution.
    if unroll_dims is not None:
        pad_matrix_op_dimensions(model, *unroll_dims)
        pad_vector_op_dimensions(model, unroll_dims[1])

    # -------------------------------------------------------------------------
    # 3. Matrix Operation Tiling
    # -------------------------------------------------------------------------
    # Apply L2 tiling logic specifically for matrix operations (GEMM/Conv) to
    # optimize for the specific cache size and memory bank configuration. This
    # annotates each GEMM/conv with ``l2_tiling`` (the bufferized builders read
    # it).  Skipped only under ``use_interstellar_tiling``, where the
    # bufferization lowering tiles each GEMM/conv on demand (interstellar)
    # instead of splitting it here.
    if cache_size is not None and not use_interstellar_tiling:
        run_matrix_op_l2_tiling(model, unroll_dims, cache_size, num_banks)

    # -------------------------------------------------------------------------
    # 4. Data Layout Transformation
    # -------------------------------------------------------------------------
    # Transform GEMM and convolution inputs/weights into layouts friendly
    # for systolic-array based hardware (e.g., transposing weights).

    if transform_layout:
        transpose_conv2d_inputs_and_weights(model)

    transpose_linear_weights(model, transform_layout, transpose_fc)

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

    # Apply L2 tiling logic for vector-based operations.
    if cache_size is not None:
        run_pool_op_l2_tiling(model, unroll_dims[1], cache_size, num_banks)
        run_vector_op_l2_tiling(model, unroll_dims[1], cache_size, num_banks)

    # -------------------------------------------------------------------------
    # 6. Operator Fusion
    # -------------------------------------------------------------------------
    # Perform final operator lowering and fuse sequences of operations (e.g.,
    # Conv+ReLU) into single kernels to reduce memory access overhead.

    if not skip_op_fusion:
        fuse_operator(model, patterns, fuse_reshape)

    rename_nodes_with_param_names(model)
    deduplicate_nodes(model)

    return model


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    total_memory=None,
    cache_size=None,
    num_banks=None,
    bank_width=None,
    unroll_dims=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    dump_snapshot=False,
    input_buffer_size: int = None,
    weight_buffer_size: int = None,
    accum_buffer_size: int = None,
    double_buffered_accum_buffer: bool = False,
    double_buffered_l2: bool = False,
    dram_size: int = None,
    dram_bandwidth=200,
    frequency=1.0,
    bufferize: bool = False,
):
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
        from .codegen.lowering.tiling import build_interstellar_tiler

        # Build the interstellar tiler once; the bufferization builders tile each
        # GEMM/conv on demand from it (no separate tiling pass / meta annotation).
        # A caller may forward ``dram_size`` / ``unroll_dims`` as ``None``
        # explicitly (e.g. the argparse defaults), so fall back here -> the tiler
        # is always built under bufferize.  Per-node dtype widths are read off the
        # nodes inside the tiler, so none are passed here.
        if dram_size is None:
            dram_size = 1 << 34
        if unroll_dims is None:
            unroll_dims = (16, 16)
        tiler = build_interstellar_tiler(
            unroll_dims,
            input_buffer_size=input_buffer_size,
            weight_buffer_size=weight_buffer_size,
            accum_buffer_size=accum_buffer_size,
            scratchpad_size=cache_size,
            dram_size=dram_size,
            dram_bandwidth=dram_bandwidth,
            frequency=frequency,
            double_buffered_accum_buffer=double_buffered_accum_buffer,
            double_buffered_l2=double_buffered_l2,
            num_banks=num_banks,
        )

        # Reuse the tiler's double-buffering decision: when L2 is double-buffered the
        # tiles already assume two-buffer occupancy, so emit software-pipelined loops.
        bufferize_graph(model, pipelined=double_buffered_l2, tiler=tiler)

        # Assign concrete DRAM / Scratchpad addresses to the explicit buffers/tiles
        # (greedy best-fit DRAM reuse; bank-aware, region-scoped scratchpad).  Writes
        # meta['memory'] / meta['scratchpad'] that the proto emitter reads.
        plan_memory(
            model,
            cache_size,
            num_banks=num_banks,
            bank_width=bank_width,
            unroll_dims=unroll_dims,
        )
        print_bufferized_graph(model)

        # Estimate latency / DRAM traffic from the scheduled graph and dump a
        # live Excel workbook + Perfetto trace alongside the protobuf.
        result = report(
            model,
            bytes_per_cycle=dram_bandwidth / frequency,
            unroll=unroll_dims,
            output_dir=output_dir,
            basename=output_file,
        )
        print(
            f"  total latency      : {result.total_latency:,} cycles\n"
            f"  DRAM read  bytes   : {result.dram_read_bytes:,}\n"
            f"  DRAM write bytes   : {result.dram_write_bytes:,}\n"
            f"  scheduled events   : {len(result.records):,}"
        )

        path = os.path.join(output_dir, "tensor_files")
        params = gen_code_bufferized(
            model, flatten_args, path if dump_tensors else None
        )
        with open(os.path.join(output_dir, "model.txt"), "w") as f:
            f.write(text_format.MessageToString(params))
        with open(os.path.join(output_dir, "bufferized_graph.txt"), "w") as f:
            f.write(print_bufferized_graph(model, to_string=True))
        gen_compute_graph_bufferized(
            model,
            os.path.join(output_dir, output_file),
            timeout=5 * 60,
        )
        return params

    allocator = MemoryAllocator(total_memory, bank_width=bank_width)
    run_memory_mapping(
        model,
        allocator,
        cache_size,
        num_banks,
        bank_width,
        unroll_dims,
        double_buffered_accum_buffer=double_buffered_accum_buffer,
        input_buffer_size=input_buffer_size,
        weight_buffer_size=weight_buffer_size,
        accum_buffer_size=accum_buffer_size,
    )

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

    return params
