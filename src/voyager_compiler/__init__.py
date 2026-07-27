from google.protobuf import text_format
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from .cli_args import *
from .codegen import *
from .codegen.transform.bufferize import (
    bufferize_graph,
    gen_code_bufferized,
    gen_compute_graph_bufferized,
    plan_memory,
    print_bufferized_graph,
)
from .codegen.transform.bufferize.emit import (
    compute_op_names,
    flush_tensor_files,
)
from .codegen.transform.bufferize.tiling import (
    DEFAULT_RUNTIME_TOLERANCE,
    build_interstellar_tiler,
)
from .codegen.transform.rewrites import split_dense_spmm_node
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
