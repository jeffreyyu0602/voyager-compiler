import argparse

from voyager_compiler.hardware_config import (
    DEFAULT_ACCUM_BUFFER_SIZE,
    DEFAULT_DOUBLE_BUFFERED_L2,
    DEFAULT_DRAM_ACCESS_LATENCY_NS,
    DEFAULT_DRAM_BANDWIDTH_GBS,
    DEFAULT_DRAM_ENERGY_PJ_PER_BIT,
    DEFAULT_DRAM_SIZE_GB,
    DEFAULT_FREQUENCY_GHZ,
    DEFAULT_INPUT_BUFFER_SIZE,
    DEFAULT_PE_ARRAY_SIZE,
    DEFAULT_SCRATCHPAD_OFFSET,
    DEFAULT_WEIGHT_BUFFER_SIZE,
    AcceleratorConfig,
)
from voyager_compiler.ops.layout import (
    DEFAULT_GEMM_WEIGHT_LAYOUT,
    DEFAULT_LAYOUT_POLICY,
    GEMM_WEIGHT_LAYOUTS,
    LAYOUT_POLICIES,
)
from voyager_compiler.utils import SLURM_ARGS

__all__ = [
    "add_quantization_args",
    "add_compile_args",
    "add_experiment_args",
]


qconfig_help_string = """
Input arguments as a comma-separated list. The first argument must specify \
the dtype.
Subsequent arguments can be specified using either abbreviations or full names.
Abbreviations and their full names:
  - qs: qscheme
  - qmax: quant_max
  - ahl: amax_history_len
  - ax: ch_axis
  - bs: block_size
  - pot: power_2_scale
  - sr: stochastic_rounding
  - gs: global_scale

Example usage:
  --params int8,qscheme=qscheme1,quant_max=127,amax_history_len=50,\
ch_axis=0,block_size=32
or
  --params int8,qs=qscheme1,qmax=127,ahl=50,ax=0,bs=32

Parameter details:
  - dtype (str): Data type (e.g., int8, int4, fp8_e4m3, fp8_e5m2, fp4_e2m1, \
posit8_1)
  - qscheme (str): Quantization scheme
  - quant_max (float): Maximum quantization value
  - amax_history_len (int): Length of the amax history (default: 50)
  - ch_axis (int): Channel axis (default: 0)
  - block_size (int): Block size (default: 32)
  - power_2_scale (0/1): Round each scale to a power of two (default: 0)
  - stochastic_rounding (0/1): Round each element at random, unbiased;
    microscaling only (default: 0)
  - global_scale (0/1): Scale the whole tensor by its amax in FP32 before
    quantizing the block scales, as NVFP4 does; microscaling with a
    scale dtype only (default: 0)
"""


def _gemm_kinds(value: str):
    """The GEMM kinds a comma-separated list names."""
    kinds = tuple(value.split(","))
    if not set(kinds) <= {"fprop", "dgrad", "wgrad"}:
        raise argparse.ArgumentTypeError(
            f"{value}: expected a subset of fprop,dgrad,wgrad"
        )
    return kinds


def add_quantization_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation",
        default=None,
        help=(
            "Activation quantization specification. Comma-separated "
            "key=value pairs "
            "using abbreviations or full names. See below for details:\n"
            + qconfig_help_string
        ),
    )
    parser.add_argument(
        "--weight",
        default=None,
        help=("Weight quantization specification. Format same as activation."),
    )
    parser.add_argument(
        "--error",
        default=None,
        help=(
            "Gradient quantization spec, in the activation's format. Setting "
            "it quantizes the backward pass too: the backward GEMMs of every "
            "quantized linear and matmul (not yet convolution)."
        ),
    )
    parser.add_argument(
        "--random_hadamard_transform",
        type=_gemm_kinds,
        default=(),
        help=(
            "GEMM kinds whose two operands are rotated by a random Hadamard "
            "transform along their contraction axis before they are "
            "quantized: a comma-separated subset of fprop, dgrad, wgrad "
            "(dgrad and wgrad need --error). Microscaling specs only."
        ),
    )
    parser.add_argument(
        "--bias",
        default=None,
        help=("Bias quantization specification. Format same as activation."),
    )
    parser.add_argument(
        "--residual",
        default=None,
        help="Residual quantization specification. Format same as activation.",
    )
    parser.add_argument(
        "--calibration_steps",
        type=int,
        default=0,
        help="Number of calibration steps for PTQ",
    )
    parser.add_argument(
        "--convert_model",
        action="store_true",
        help="Whether to convert the model to quantized model.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 (mixed) precision instead of 32-bit float.",
    )
    return parser


def add_experiment_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Run quantized inference or training."
        )
    add_quantization_args(parser)
    # ----------------------------------------------------
    # Wandb and logging arguments
    # ----------------------------------------------------
    parser.add_argument(
        "--project",
        default=None,
        help="The name of the project where the new run will be sent.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help=(
            "A short display name for this run, which is this run will be "
            "identified in the UI."
        ),
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="A unique ID for a wandb run, used for resuming.",
    )
    parser.add_argument(
        "--sweep_config_json",
        type=str,
        default=None,
        help="Inline JSON string for W&B sweep configuration",
    )
    parser.add_argument(
        "--sweep_id",
        default=None,
        help=(
            "The unique identifier for a sweep generated by W&B CLI or Python "
            "SDK."
        ),
    )
    parser.add_argument(
        "--sweep_count",
        type=int,
        default=None,
        help="The number of sweep config trials to try.",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help=(
            "Set the logging file. If not specified, the log will be printed "
            "to stdout."
        ),
    )
    # ----------------------------------------------------
    # Training arguments
    # ----------------------------------------------------
    parser.add_argument("--gpu", type=int, default=None, help="GPU to use.")
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training"
    )
    parser.add_argument(
        "--sgd", action="store_true", help="Whether to use SGD optimizer."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of warmup steps in the lr scheduler.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Number of Tranformer encoder layers to use.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="The dimension of the low-rank matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="The scaling factor for the low-rank matrices.",
    )
    parser.add_argument(
        "--target_modules",
        type=lambda x: x.split(","),
        default="query,value",
        help=(
            "The modules (for example, attention blocks) to apply the LoRA "
            "update matrices."
        ),
    )
    parser.add_argument(
        "--peft_model_id",
        default=None,
        help="Name of path of pre-trained peft adapter.",
    )
    # ----------------------------------------------------
    # Slurm arguments
    # ----------------------------------------------------
    subparsers = parser.add_subparsers(
        help="sub-command help", dest="execution_mode"
    )
    parser_slurm = subparsers.add_parser("slurm", help="slurm command help")
    for k, v in SLURM_ARGS.items():
        parser_slurm.add_argument("--" + k, **v)
    subparsers.add_parser("bash", help="bash command help")
    return parser


def add_compile_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # -- memory hierarchy ---------------------------------------------------
    parser.add_argument(
        "--scratchpad_size",
        type=int,
        default=None,
        help="Total L2 SRAM size (bytes).",
    )
    parser.add_argument(
        "--scratchpad_offset",
        type=int,
        default=DEFAULT_SCRATCHPAD_OFFSET,
        help="Bytes reserved at the base of the L2 SRAM, for a program the "
        "accelerator shares the scratchpad with; allocations start above "
        "it. Must be a multiple of the bank size.",
    )
    parser.add_argument(
        "--num_banks",
        type=int,
        default=None,
        help="Number of banks in the accelerator.",
    )
    parser.add_argument(
        "--bank_width",
        type=int,
        default=None,
        help="Memory bank width (bytes) for memory planning.",
    )
    parser.add_argument(
        "--independent_memory_ports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use independent external interface timing for SA or CIM; "
        "retain banked allocation and emit INDEPENDENT harness mode.",
    )
    parser.add_argument(
        "--double_buffered_l2",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DOUBLE_BUFFERED_L2,
        help="Overlap DRAM I/O with compute (ping-pong); halves L2 for tiling.",
    )
    parser.add_argument(
        "--input_buffer_size",
        type=int,
        default=DEFAULT_INPUT_BUFFER_SIZE,
        help="Input buffer size per IC dim (# elements).",
    )
    parser.add_argument(
        "--weight_buffer_size",
        type=int,
        default=DEFAULT_WEIGHT_BUFFER_SIZE,
        help="Weight buffer size per OC dim (# elements).",
    )
    parser.add_argument(
        "--accum_buffer_size",
        type=int,
        default=DEFAULT_ACCUM_BUFFER_SIZE,
        help="Accum buffer size per OC dim (# elements).",
    )
    parser.add_argument(
        "--double_buffered_accum_buffer",
        action="store_true",
        help="Use a double-buffered accumulation buffer (interstellar tiling).",
    )
    parser.add_argument(
        "--dram_size",
        type=float,
        default=DEFAULT_DRAM_SIZE_GB,
        help="DRAM capacity in GB; enables the 4-level interstellar hierarchy.",
    )
    parser.add_argument(
        "--dram_bandwidth",
        type=float,
        default=DEFAULT_DRAM_BANDWIDTH_GBS,
        help="Typical mobile SoC DRAM bandwidth in GB/s.",
    )
    parser.add_argument(
        "--dram_access_latency",
        type=float,
        default=DEFAULT_DRAM_ACCESS_LATENCY_NS,
        help="DRAM access latency (ns); per-transfer cycles = latency * "
        "frequency.",
    )
    parser.add_argument(
        "--dram_energy_per_bit",
        type=float,
        default=DEFAULT_DRAM_ENERGY_PJ_PER_BIT,
        help="DRAM access energy (pJ/bit) for the reporting energy columns.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY_GHZ,
        help="Clock frequency in GHz (with --dram_bandwidth -> bytes/cycle).",
    )

    # -- data layout + hardware unrolling -----------------------------------
    parser.add_argument(
        "--layout_policy",
        choices=LAYOUT_POLICIES,
        default=DEFAULT_LAYOUT_POLICY,
        help="Operand layouts (activation / conv weight / matmul weight): "
        "pytorch = NCHW/OIHW/KC, systolic or cim = NHWC/HWIO/CK.",
    )
    parser.add_argument(
        "--gemv_weight_layout",
        choices=GEMM_WEIGHT_LAYOUTS,
        default=DEFAULT_GEMM_WEIGHT_LAYOUT,
        help="Matrix-vector GEMM weight layout: kc is [out, contraction], "
        "ck is [contraction, out].",
    )
    parser.add_argument(
        "--pe_array_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default=DEFAULT_PE_ARRAY_SIZE,
        help="Matrix input/output lane counts, e.g. 16,16. For CIM, these "
        "must match the macro geometry and weight datatype.",
    )
    parser.add_argument(
        "--vector_unit_width",
        type=int,
        default=None,
        help="Vector unit lane count; defaults to the PE array columns.",
    )
    parser.add_argument(
        "--matrix_vector_unit_width",
        type=int,
        default=None,
        help=(
            "Matrix-vector unit width in elements; defaults to the PE array "
            "columns."
        ),
    )
    parser.add_argument(
        "--accumulator_width",
        type=int,
        default=None,
        help="Channels the vector unit fetches per pooling request "
        "(ACCUMULATOR_WIDTH); defaults to the vector unit lane count.",
    )

    # -- matrix backend + CIM geometry -------------------------------------
    parser.add_argument(
        "--matrix_backend",
        type=int,
        choices=(0, 1),
        default=AcceleratorConfig.matrix_backend,
        help="Matrix backend: 0 = systolic, 1 = CIM.",
    )
    # Keep defaults on AcceleratorConfig, shared by CLI and Python callers.
    for name, help_text in (
        ("cim_macro_input_lanes", "Physical input lanes per macro."),
        ("cim_macro_output_lanes", "Physical output lanes per macro."),
        ("cim_weight_sets", "Resident weight sets per macro."),
        ("cim_base_a_width", "Macro input width (bits)."),
        ("cim_base_b_width", "Macro weight width (bits)."),
        ("cim_base_c_width", "Macro accumulation width (bits)."),
        ("cim_macro_write_input_lanes", "Input lanes per weight write (1)."),
        ("cim_mac_latency", "Macro MAC latency (cycles)."),
        ("cim_mode", "Macro mode: 0 = bit-parallel, 1 = bit-serial."),
        ("cim_tile_input_axis_elements", "Elements along a tile's input axis."),
        (
            "cim_tile_output_axis_elements",
            "Elements along a tile's output axis.",
        ),
        ("cim_input_axis_tiles", "Tiles along the array input axis."),
        ("cim_output_axis_tiles", "Tiles along the array output axis."),
        ("cim_a_port_tiles", "Input tiles per A beat; default: full axis."),
        ("cim_b_port_tiles", "Output tiles per B beat; default: full axis."),
        ("cim_c_port_tiles", "Output tiles per C beat; default: full axis."),
        ("cim_c_beat_layout", "C beat layout; CIMProcessor requires 1."),
        (
            "cim_array_result_slots",
            "Result slots per output tile; default: input tile count.",
        ),
        ("cim_local_accum_contexts", "Live local accumulation contexts."),
    ):
        parser.add_argument(
            f"--{name}",
            type=int,
            default=getattr(AcceleratorConfig, name),
            help=help_text,
        )
    parser.add_argument(
        "--cim_signed",
        action=argparse.BooleanOptionalAction,
        default=AcceleratorConfig.cim_signed,
        help="Signed integer operands; disable for unsigned operands.",
    )

    # -- tiling / lowering --------------------------------------------------
    parser.add_argument(
        "--disable_reshape_fusion",
        action="store_true",
        help="Do not fuse reshape with the following GEMM in Transformers.",
    )
    parser.add_argument(
        "--remove_fp32_casts",
        action="store_true",
        help="Compute in 16-bit float where the model casts up to float32, "
        "for an accelerator with no float32 datapath.",
    )
    parser.add_argument(
        "--accumulate_fp32",
        action="store_true",
        help="Accumulate a GEMM split along K in float32 rather than its "
        "output dtype, as training needs.",
    )
    parser.add_argument(
        "--runtime_tolerance",
        type=float,
        default=None,  # -> DEFAULT_RUNTIME_TOLERANCE (0.02) in compile()
        help="How much longer than the best modeled runtime an interstellar "
        "tiling may take and still be chosen, as a fraction; among those, the "
        "one with the lowest modeled access energy wins.  0 = only the "
        "fastest, with energy breaking exact-runtime ties "
        "(default: 0.02).",
    )

    # -- reporting (timing / DRAM-traffic estimator) ------------------------
    parser.add_argument(
        "--report",
        action="store_true",
        help="After compile, estimate the schedule and dump <basename>.xlsx + "
        ".perfetto.json.",
    )
    parser.add_argument(
        "--report_output_dir",
        default=".",
        help="Directory for the reporting workbook / trace.",
    )
    parser.add_argument(
        "--report_basename",
        default="schedule",
        help="Base filename for the reporting outputs.",
    )
    parser.add_argument(
        "--calib_in",
        default=None,
        help="A filled-in calibration sheet (a standalone form or a previous "
        "<basename>.xlsx): price compute ops by its RTL measurements and "
        "carry its power numbers into the Operation Table.",
    )
    return parser
