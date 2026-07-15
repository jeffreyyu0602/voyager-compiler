"""Shared building blocks for the Llama design-space sweeps.

The benchmark runner (``runner.py``) drives the same compile + estimate
pipeline that ``test/test_llama_report.py`` runs, only parameterized by a
:class:`SweepConfig` instead of argparse.  The heavy lifting -- loading a model,
exporting a prefill *or* decode graph, quantizing, tiling, bufferizing, planning
memory, and estimating the schedule -- lives here so the driver stays short.

The estimate is *ideal*: no accelerator is ever run; ``estimate_schedule`` walks
the bufferized graph and returns latency (cycles) and DRAM traffic, now split
into weight / activation / KV-cache bytes.

Weight / activation precision follows the accelerator's MX formats, keyed by
bit-width:

* 16 -> BF16 (unquantized, runs on a bf16 PE array)
* 8  -> MXINT8 with a power-of-two block scale (the baseline)
* 4  -> MXFP4 (``fp4_e2m1`` element, ``fp8_e4m3`` block scale)

The MX group (block) size is not a free knob -- it must match the PE array, so
``group = max(pe)``.

The KV cache is kept BF16 (unquantized) for now: microscaling a stored KV cache
is unsupported (it feeds a repeat_kv slice, not an MXU op).  A 16-bit cache uses
the simple HF export; the KIVI split-cache + ``group_wise_affine`` machinery for
quantized KV is kept (see ``build_decode`` / ``_annotate_kv_cache``) but off
until the bufferizer can lower an in-place cache-write cone.
"""

import argparse
import multiprocessing
import operator
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from typing import (
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.integrations.executorch import (
    convert_and_export_with_cache,
)

import voyager_compiler  # noqa: F401  registers voyager.*
from torch._export.utils import _disable_aten_to_metadata_assertions

from voyager_compiler import (
    OpMatcher,
    QuantizationSpec,
    convert_and_export_with_split_cache,
    convert_pt2e,
    export_model,
    get_default_quantizer,
    prepare_pt2e,
    remove_softmax_dtype_cast,
    replace_rmsnorm_with_layer_norm,
    swap_llama_attention,
    transform,
)
from voyager_compiler.codegen.lowering import (
    bufferize_graph,
    plan_memory,
)
from voyager_compiler.codegen.lowering.reporting import (
    compress_schedule,
    estimate_schedule,
    write_excel_report,
    write_perfetto,
)
from voyager_compiler.codegen.lowering.tiling import build_interstellar_tiler
from voyager_compiler.codegen.mapping_utils import (
    is_compute_op,
    is_fully_connected,
)
from voyager_compiler.codegen.shape_prop import ShapeProp

try:
    # torchao helper the KIVI 2-bit KV path annotates cache buffers with.
    # (Lived in torch.ao.quantization.quantizer.utils before torch 2.12.)
    from torchao.quantization.pt2e.quantizer.utils import (
        annotate_output_qspec as _annotate_output_qspec,
    )
except ImportError:  # pragma: no cover - older torch layout
    _annotate_output_qspec = None

# Since torch 2.9 a `cond` over a buffer-mutating body is rejected as
# "training with in-place input or buffer mutations" unless grad is off.
torch.set_grad_enabled(False)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

# -- baseline design point (spec §1), matching test_llama_report.py -------
# ``cache_size`` is the tiler's L2 scratchpad; double buffering makes the
# effective on-chip SRAM ``2 * cache_size`` (so 1 MiB here == 2 MB effective).
SRAM_BANK_SIZE = 128 * 1024  # physical bank size, held fixed across sweeps
BASELINE_CACHE_SIZE = 1 * 1024 * 1024
BASELINE_NUM_BANKS = 8
BASELINE_BANK_WIDTH = 64
BASELINE_PE = (64, 64)
BASELINE_FREQUENCY_GHZ = 1.0
BASELINE_DRAM_BANDWIDTH_GBS = 50.0
BASELINE_DRAM_ACCESS_LATENCY_NS = 100.0
BASELINE_PROMPT_LEN = 1024

# Max generation length added to the decode KV cache: the cache must hold the
# ``kv_len`` context plus room for generation, so ``max_cache_len = kv_len +
# DECODE_MAX_GEN`` (matching test_codegen's ``context + 128``).  For the KIVI
# split cache this is also the residual window (``max_new_tokens``).
DECODE_MAX_GEN = 128


# -------------------------------------------------------------------------
# Operator-fusion pipeline (copied verbatim from test_llama_report.py): fold
# each MXU op's trailing dequant / activation / requantization into one fused
# kernel.  Stages that match no node are skipped.
# -------------------------------------------------------------------------
def _is_spmm(node):
    return node.kwargs.get("A_data") is not None


def _is_bf16_fc(node):
    # bf16 FCs run on the vector unit and thus cannot be fused.
    if hasattr(node, "value") and is_fully_connected(node):
        return node.args[0].meta.get("dtype") is None
    return False


def _can_fuse(node):
    return not _is_spmm(node) and not _is_bf16_fc(node)


def _is_constant_div(node):
    if node.target != torch.ops.aten.div.Tensor:
        return True
    divisor = node.args[1]
    if isinstance(divisor, torch.fx.Node):
        return divisor.value.numel() == 1
    return True


MXU_OPS = ["conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"]
QUANT_OPS = ["quantize", "quantize_mx", "quantize_mx_outlier"]

FUSION_PIPELINE = [
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("add", "sub", "mul", "div", predicate=_is_constant_div),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div", predicate=_is_constant_div),
        OpMatcher(*QUANT_OPS, "mul", "div"),
    ],
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("gelu", "sigmoid", "silu", "tanh", "hardtanh"),
        OpMatcher(*QUANT_OPS, "mul", "div"),
    ],
    [
        OpMatcher("layer_norm", "softmax"),
        OpMatcher(*QUANT_OPS),
    ],
]


# -------------------------------------------------------------------------
# Design point
# -------------------------------------------------------------------------
@dataclass
class SweepConfig:
    """One point in the design space.

    ``mode`` selects the graph shape: ``"prefill"`` feeds the whole model an
    ``input_ids`` of length ``prompt_len`` (``use_cache=False``); ``"decode"``
    measures one token over a KIVI split cache whose main cache holds ``kv_len``
    entries.  Bit-widths are each one of {16, 8, 4} (and 2 for KV only)."""

    model_id: str = DEFAULT_MODEL
    mode: str = "prefill"  # "prefill" | "decode"
    prompt_len: int = BASELINE_PROMPT_LEN
    kv_len: int = BASELINE_PROMPT_LEN  # decode: prepopulated KV length
    batch: int = 1

    weight_bits: int = 8
    act_bits: int = 8
    # KV cache is kept BF16 (unquantized): microscaling a stored KV cache is not
    # supported (it feeds a repeat_kv slice, not an MXU op), so KV precision is
    # not swept for now.
    kv_bits: int = 16

    pe: Tuple[int, int] = BASELINE_PE
    cache_size: int = BASELINE_CACHE_SIZE
    num_banks: int = BASELINE_NUM_BANKS
    bank_width: int = BASELINE_BANK_WIDTH
    frequency_ghz: float = BASELINE_FREQUENCY_GHZ
    dram_bandwidth_gbs: float = BASELINE_DRAM_BANDWIDTH_GBS
    dram_access_latency_ns: float = BASELINE_DRAM_ACCESS_LATENCY_NS

    # Attention backend.  "eager" keeps attention as ordinary matmuls (the only
    # backend wired into the bufferize/report path).
    attn_implementation: str = "eager"

    # Single-buffer a >1-tile reduction's output + fused post-op operands
    # (last-K-step-only), trading SRAM for the prefetch; off => double-buffered.
    single_buffer_tail: bool = False

    # Build only this many decoder layers; None uses the real count.
    num_layers_override: Optional[int] = None

    @property
    def unroll(self) -> Tuple[int, int]:
        return tuple(self.pe)

    @property
    def group(self) -> int:
        """MX block size -- pinned to the PE array."""
        return max(self.pe)

    @classmethod
    def baseline(cls, **overrides) -> "SweepConfig":
        return replace(cls(), **overrides)


def add_config_args(parser: argparse.ArgumentParser, defaults=None, exclude=()):
    """Add a ``--field-name`` flag for every SweepConfig field, defaulting to
    SweepConfig's default (or ``defaults``).  Bools become ``--x/--no-x``;
    ``pe`` takes two ints.  ``exclude`` names fields a script defines itself
    (e.g. a ``--mode`` with a ``both`` option); sweep scripts add extra flags
    on top."""
    base = defaults or SweepConfig()
    hints = get_type_hints(SweepConfig)
    for f in fields(SweepConfig):
        if f.name in exclude:
            continue
        typ = hints[f.name]
        if get_origin(typ) is Union:  # Optional[T] -> T
            typ = next(a for a in get_args(typ) if a is not type(None))
        flag = "--" + f.name.replace("_", "-")
        default = getattr(base, f.name)
        if typ is bool:
            parser.add_argument(
                flag,
                dest=f.name,
                default=default,
                action=argparse.BooleanOptionalAction,
            )
        elif get_origin(typ) is tuple:
            parser.add_argument(
                flag,
                dest=f.name,
                type=int,
                nargs=len(get_args(typ)),
                default=default,
            )
        else:
            parser.add_argument(flag, dest=f.name, type=typ, default=default)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Estimate via run_design_point_fast: compile two small layer "
        "counts and extrapolate, instead of the full model.",
    )
    parser.add_argument(
        "--probe-layers",
        dest="probe_layers",
        type=int,
        nargs=2,
        default=list(LAYER_PROBE),
        help="Layer counts the --fast path compiles and extrapolates from.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run this many design points in parallel (1 = serial).",
    )
    parser.add_argument(
        "--threads-per-job",
        dest="threads_per_job",
        type=int,
        default=None,
        help="Torch threads per parallel job (default: cores // jobs).",
    )
    parser.add_argument(
        "--log-dir",
        dest="log_dir",
        default=None,
        help="Directory for per-point logs when --jobs > 1.",
    )
    return parser


def config_from_args(args, **overrides) -> "SweepConfig":
    """Build a SweepConfig from parsed args; ``overrides`` win."""
    names = {f.name for f in fields(SweepConfig)}
    kw = {k: v for k, v in vars(args).items() if k in names}
    kw.update(overrides)
    if isinstance(kw.get("pe"), list):
        kw["pe"] = tuple(kw["pe"])
    return SweepConfig(**kw)


# -------------------------------------------------------------------------
# Precision -> dtype-string mapping
# -------------------------------------------------------------------------
def dtype_spec(bits: int, group: int) -> Optional[str]:
    """The MXU element format for a weight/activation operand at ``bits``.
    Returns ``None`` for 16-bit (bf16, left unquantized)."""
    if bits == 16:
        return None
    if bits == 8:
        return f"int8,qs=microscaling,bs={group}"
    if bits == 4:
        return f"fp4_e2m1,qs=microscaling,bs={group},scale=fp8_e4m3"
    raise ValueError(f"unsupported weight/activation bits: {bits}")


def _kv_cache_spec(bits: int, group: int, role: str) -> Optional[str]:
    """The dtype string for a *main* KV-cache tensor at ``bits``.  KIVI
    quantizes keys per-channel (``ax=-2``) and values per-token (``ax=-1``) of
    the ``(N, H, S, D)`` cache.  Returns ``None`` for 16-bit (full precision).
    """
    ax = -2 if role == "key" else -1
    if bits == 16:
        return None
    if bits == 8:
        return f"int8,qs=microscaling,bs={group},ax={ax}"
    if bits == 4:
        return f"fp4_e2m1,qs=microscaling,bs={group},ax={ax},scale=fp8_e4m3"
    if bits == 2:
        return f"uint2,qs=group_wise_affine,bs={group},ax={ax},scale=fp8_e5m3"
    raise ValueError(f"unsupported kv bits: {bits}")


def build_quantizer(cfg: SweepConfig):
    """A quantizer for ``cfg``'s weight / activation precision (microscaling for
    8/4-bit, unquantized bf16 for 16-bit).  KV-cache precision is *not* set here
    -- it is applied by annotating the main cache tensors in
    ``_annotate_kv_cache`` (KIVI keeps the full-precision residual untouched).
    """
    group = cfg.group
    return get_default_quantizer(
        input_activation=dtype_spec(cfg.act_bits, group),
        weight=dtype_spec(cfg.weight_bits, group),
        force_scale_power_of_two=True,
    )


# -------------------------------------------------------------------------
# Model / graph builders
# -------------------------------------------------------------------------
def _load_model(cfg: SweepConfig):
    extra = {}
    if cfg.num_layers_override is not None:
        extra["num_hidden_layers"] = cfg.num_layers_override
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
        **extra,
    ).eval()


def _prompt_ids(cfg: SweepConfig, tokenizer, length: int):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    ids = encodings.input_ids[:, :length]
    if cfg.batch > 1:
        ids = ids.repeat(cfg.batch, 1)
    return ids


def build_prefill(cfg: SweepConfig):
    """Export a prefill graph: the whole ``AutoModelForCausalLM`` fed
    ``input_ids`` of shape ``(batch, prompt_len)`` with ``use_cache=False``.
    Returns ``(gm, model, example_args, example_kwargs)`` -- the exported graph
    module and the underlying model."""
    model = _load_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    input_ids = _prompt_ids(cfg, tokenizer, cfg.prompt_len)
    example_args = (input_ids,)
    example_kwargs = {"return_dict": False, "use_cache": False}
    gm = export_model(model, example_args, example_kwargs)
    return gm, model, example_args, example_kwargs


def build_decode(cfg: SweepConfig):
    """Export a single decode step over a KV cache of length ``kv_len``.

    A **BF16 (16-bit) cache** does not need the KIVI split cache, so it takes
    the simpler HF export (``_build_decode_bf16``).  A **quantized cache** needs
    the split cache -- a quantizable main cache plus a full-precision residual,
    exported via ``_build_decode_split``.  KV quant is off for now (kv_bits is
    always 16), but the split path is kept for when the bufferizer can lower an
    in-place cache write.

    Returns ``(gm, model, example_args, example_kwargs)``."""
    if cfg.kv_bits == 16:
        return _build_decode_bf16(cfg)
    return _build_decode_split(cfg)


def _build_decode_bf16(cfg: SweepConfig):
    """Simple full-model decode export for a BF16 cache, via HF's
    ``convert_and_export_with_cache`` (whole model -- embeddings, every layer,
    final norm, lm_head -- no residual / KIVI swap).  The cache holds the
    ``kv_len`` context plus ``DECODE_MAX_GEN`` generation slots, rounded up to
    the MX ``block_size`` (= ``cfg.group``) so the KV tensors stay block-aligned
    and need no attention padding.  The decode token is at ``cache_position =
    [kv_len]``.  Contents don't affect the estimate, only shapes."""
    model = _load_model(cfg)
    block = cfg.group
    # context + generation budget, rounded up to a block_size multiple.
    raw = cfg.kv_len + DECODE_MAX_GEN
    max_cache_len = -(-raw // block) * block
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        cache_config={
            "batch_size": cfg.batch,
            "max_cache_len": max_cache_len,
        },
    )
    input_ids = torch.ones((cfg.batch, 1), dtype=torch.long)
    cache_position = torch.tensor([cfg.kv_len], dtype=torch.long)
    # Strict export bakes in aten._assert_tensor_metadata guards (e.g. the
    # attention softmax's dtype=float32); ``remove_softmax_dtype_cast`` later
    # rewrites that softmax to bf16, so those stale guards must be suppressed at
    # export or they fail at calibration.
    with _disable_aten_to_metadata_assertions():
        ep = convert_and_export_with_cache(
            model,
            example_input_ids=input_ids,
            example_cache_position=cache_position,
        )
    gm = ep.module()
    example_kwargs = {"input_ids": input_ids, "cache_position": cache_position}
    return gm, model, (), example_kwargs


def _build_decode_split(cfg: SweepConfig):
    """KIVI split-cache decode export (for a *quantized* KV cache).

    ``convert_and_export_with_split_cache`` (llm_utils) exports the whole model
    with a quantizable **main** cache of length ``kv_len`` plus a full-precision
    **residual** cache of length ``DECODE_MAX_GEN`` (the recent-token
    window KIVI keeps full precision).  Attention is swapped to
    ``LlamaAttentionKIVI`` and the causal mask is passed in precomputed, so no
    in-graph mask control flow is traced.  One step at ``cache_position =
    [kv_len]``; contents don't affect the estimate, only shapes."""
    model = _load_model(cfg)
    swap_llama_attention(model)

    max_len = cfg.kv_len
    max_new = DECODE_MAX_GEN
    max_cache_len = max_len + max_new

    input_ids = torch.ones((cfg.batch, 1), dtype=torch.long)
    cache_position = torch.tensor([cfg.kv_len], dtype=torch.long)
    cache_position_residual = torch.tensor([0], dtype=torch.long)
    attention_mask = torch.ones((cfg.batch, max_cache_len), dtype=model.dtype)[
        None, None, :, :
    ]

    with _disable_aten_to_metadata_assertions():
        ep = convert_and_export_with_split_cache(
            model,
            max_len=max_len,
            max_new_tokens=max_new,
            example_input_ids=input_ids,
            example_cache_position=cache_position,
            example_cache_position_residual=cache_position_residual,
            example_attention_mask=attention_mask,
        )
    gm = ep.module()

    example_kwargs = {
        "input_ids": input_ids,
        "cache_position": cache_position,
        "cache_position_residual": cache_position_residual,
        "attention_mask": attention_mask,
    }
    return gm, model, (), example_kwargs


# Main KV-cache buffers only (``key_cache_0`` ...); the ``key_cache_residual_*``
# tensors are deliberately excluded -- KIVI keeps the residual full precision.
_MAIN_KV_CACHE = re.compile(r"^(key|value)_cache_(\d+)$")


def _annotate_kv_cache(gm, cfg: SweepConfig) -> int:
    """Annotate the *main* KV-cache tensors with ``cfg.kv_bits``' KIVI spec
    (keys per-channel, values per-token), leaving the full-precision residual
    cache untouched.  A no-op for 16-bit KV.  Raises if the graph names the
    cache tensors unexpectedly so a mismatch surfaces loudly."""
    if cfg.kv_bits == 16:
        return 0
    if _annotate_output_qspec is None:
        raise RuntimeError("torch.ao _annotate_output_qspec unavailable")
    key_qspec = QuantizationSpec.from_str(
        _kv_cache_spec(cfg.kv_bits, cfg.group, "key")
    )
    value_qspec = QuantizationSpec.from_str(
        _kv_cache_spec(cfg.kv_bits, cfg.group, "value")
    )
    n = 0
    for node in gm.graph.nodes:
        m = _MAIN_KV_CACHE.match(str(node.target))
        if node.op == "get_attr" and m is not None:
            _annotate_output_qspec(
                node, key_qspec if m.group(1) == "key" else value_qspec
            )
            n += 1
    if n == 0:
        raise RuntimeError(
            "KV quant: no main key/value cache get_attrs matched -- the "
            "exported decode graph names them differently; inspect the graph "
            "and update _MAIN_KV_CACHE"
        )
    return n


# -------------------------------------------------------------------------
# The compile + estimate pipeline
# -------------------------------------------------------------------------
@dataclass
class Metrics:
    """The result of estimating one graph -- a whole design point or a single
    operator block.

    A whole-graph run leaves the runtime-breakdown fields
    (``compute / memory / overlap / stall``) and the block identity
    (``name / count``) at their defaults; the per-module path fills them,
    because it holds the ``ScheduleResult`` and computes ``_time_split`` while a
    whole-graph run keeps only the summary scalars."""

    total_latency: int
    dram_read_bytes: int
    dram_write_bytes: int
    dram_weight_bytes: int
    dram_activation_bytes: int
    dram_kv_bytes: int
    scratchpad_bytes: int
    num_layers: int
    num_params: int
    # Runtime breakdown (per-module only): a schedule's makespan split into
    # compute-bound / memory-bound / overlapped / stalled cycles.
    compute: int = 0
    memory: int = 0
    overlap: int = 0
    stall: int = 0
    # Block identity (per-module only): the block's name and how many times it
    # runs in the whole model.
    name: str = ""
    count: int = 1

    @property
    def dram_total_bytes(self) -> int:
        return self.dram_read_bytes + self.dram_write_bytes


def _scale(m: "Metrics", count: int) -> "Metrics":
    """Scale a per-block estimate to its whole-model contribution: multiply the
    *additive* metrics (latency, DRAM bytes, the runtime breakdown) by ``count``
    -- how many times the block runs -- and record ``count``.  ``scratchpad`` is
    a peak, not a sum, so it (and the model-shape fields) is left as-is, the
    same way ``run_design_point_fast`` extrapolates latency/DRAM but takes
    scratchpad as-is."""
    return replace(
        m,
        total_latency=m.total_latency * count,
        dram_read_bytes=m.dram_read_bytes * count,
        dram_write_bytes=m.dram_write_bytes * count,
        dram_weight_bytes=m.dram_weight_bytes * count,
        dram_activation_bytes=m.dram_activation_bytes * count,
        dram_kv_bytes=m.dram_kv_bytes * count,
        compute=m.compute * count,
        memory=m.memory * count,
        overlap=m.overlap * count,
        stall=m.stall * count,
        count=count,
    )


def _frontend(cfg: SweepConfig):
    """Export -> quantize -> transform -> ShapeProp for ``cfg``, and build the
    interstellar tiler.  Returns ``(gm, model, tiler)`` -- the graph is
    transformed and tile-ready but *not* yet whole-graph bufferized, so both the
    whole-graph and the per-module paths can share this."""
    unroll = cfg.unroll
    is_decode = cfg.mode == "decode"
    # The builders return an already-exported graph (prefill via export_model,
    # decode via convert_and_export_with_split_cache).
    if is_decode:
        gm, model, example_args, example_kwargs = build_decode(cfg)
    else:
        gm, model, example_args, example_kwargs = build_prefill(cfg)

    remove_softmax_dtype_cast(gm)

    hidden = model.model.layers[0].input_layernorm.weight.shape[-1]
    seq = 1 if is_decode else 128
    example_input = torch.randn(1, seq, hidden, dtype=model.dtype)
    replace_rmsnorm_with_layer_norm(
        gm, model.model.layers[0].input_layernorm, (example_input,)
    )

    # KV-cache precision is applied by annotating the main cache tensors (KIVI);
    # it only exists in decode.  Prefill has no persistent cache, so kv_bits is
    # not applicable there.
    if is_decode:
        _annotate_kv_cache(gm, cfg)

    quantizer = build_quantizer(cfg)
    gm = prepare_pt2e(gm, quantizer, example_args, example_kwargs)
    for _ in range(2):
        gm(*example_args, **example_kwargs)
    convert_pt2e(gm, None)

    transform(
        gm,
        example_args,
        example_kwargs=example_kwargs,
        patterns=FUSION_PIPELINE,
        unroll_dims=unroll,
        transform_layout=True,
        cache_size=cfg.cache_size,
        num_banks=cfg.num_banks,
        use_interstellar_tiling=True,
        bufferize=True,
    )

    flat_args, _ = torch.utils._pytree.tree_flatten(
        (example_args, example_kwargs)
    )
    ShapeProp(gm).propagate(*flat_args)

    bytes_per_cycle = cfg.dram_bandwidth_gbs / cfg.frequency_ghz
    tiler = build_interstellar_tiler(
        unroll,
        input_buffer_size=1024,
        weight_buffer_size=1024,
        accum_buffer_size=1024,
        scratchpad_size=cfg.cache_size,
        dram_size=64.0,
        dram_bandwidth=bytes_per_cycle,
        double_buffered_l2=True,
        num_banks=cfg.num_banks,
    )
    return gm, model, tiler


def _plan(cfg, gm):
    # Double buffering: the planner sees twice the cache and banks.
    return plan_memory(
        gm,
        cfg.cache_size * 2,
        num_banks=cfg.num_banks * 2,
        bank_width=cfg.bank_width,
        unroll_dims=cfg.unroll,
    )


def _compile(cfg: SweepConfig):
    """Front end + whole-graph bufferize + memory plan for ``cfg``.
    Returns ``(gm, model, plan)`` ready for ``estimate_schedule``."""
    gm, model, tiler = _frontend(cfg)
    bufferize_graph(
        gm,
        pipelined=True,
        tiler=tiler,
        single_buffer_tail=cfg.single_buffer_tail,
    )
    plan = _plan(cfg, gm)
    return gm, model, plan


def _num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def run_design_point(cfg: SweepConfig) -> Metrics:
    """Compile ``cfg`` and return its ideal latency + DRAM traffic."""
    gm, model, plan = _compile(cfg)
    r = estimate_schedule(
        gm,
        dram_bandwidth=cfg.dram_bandwidth_gbs,
        dram_access_latency=cfg.dram_access_latency_ns,
        frequency=cfg.frequency_ghz,
        unroll=cfg.unroll,
    )
    return Metrics(
        total_latency=r.total_latency,
        dram_read_bytes=r.dram_read_bytes,
        dram_write_bytes=r.dram_write_bytes,
        dram_weight_bytes=r.dram_weight_bytes,
        dram_activation_bytes=r.dram_activation_bytes,
        dram_kv_bytes=r.dram_kv_bytes,
        scratchpad_bytes=plan.scratchpad_bytes,
        num_layers=model.config.num_hidden_layers,
        num_params=_num_params(model),
    )


LAYER_PROBE = (1, 2)


def run_design_point_fast(
    cfg: SweepConfig, probe_layers: Tuple[int, int] = LAYER_PROBE
) -> Metrics:
    """Compile two small layer counts and linearly extrapolate each metric to
    the real ``num_hidden_layers`` (scratchpad, flat in layers, is taken as-is).
    """
    k1, k2 = probe_layers
    if not 0 < k1 < k2:
        raise ValueError(
            f"probe_layers must satisfy 0 < k1 < k2, got {probe_layers}"
        )
    n = AutoConfig.from_pretrained(cfg.model_id).num_hidden_layers
    m1 = run_design_point(replace(cfg, num_layers_override=k1))
    m2 = run_design_point(replace(cfg, num_layers_override=k2))

    def extrap(x1: int, x2: int) -> int:
        return round(x1 + (n - k1) * (x2 - x1) / (k2 - k1))

    return Metrics(
        total_latency=extrap(m1.total_latency, m2.total_latency),
        dram_read_bytes=extrap(m1.dram_read_bytes, m2.dram_read_bytes),
        dram_write_bytes=extrap(m1.dram_write_bytes, m2.dram_write_bytes),
        dram_weight_bytes=extrap(m1.dram_weight_bytes, m2.dram_weight_bytes),
        dram_activation_bytes=extrap(
            m1.dram_activation_bytes, m2.dram_activation_bytes
        ),
        dram_kv_bytes=extrap(m1.dram_kv_bytes, m2.dram_kv_bytes),
        scratchpad_bytes=m2.scratchpad_bytes,
        num_layers=n,
        num_params=extrap(m1.num_params, m2.num_params),
    )


def safe_run(
    cfg: SweepConfig, label: str, fast=False, probe_layers=LAYER_PROBE
) -> Optional[Metrics]:
    """Run one design point, logging and skipping (returning ``None``) on any
    failure -- so a missing / gated model or an unsupported point does not abort
    the whole sweep.  ``fast`` extrapolates from two small layer counts."""
    print(f"[run] {label}", flush=True)
    try:
        if fast:
            return run_design_point_fast(cfg, tuple(probe_layers))
        return run_design_point(cfg)
    except Exception as e:  # noqa: BLE001 - a sweep must survive one bad point
        print(f"  [skip] {label}: {type(e).__name__}: {e}", flush=True)
        return None


def _run_point_worker(task):
    """Pool worker: divert this point's stdout+stderr (Python logging and the
    C/torch output alike) to its own log file at the fd level, pin threads, then
    run it.  Top-level so it is picklable under the spawn start method."""
    cfg, label, fast, probe_layers, threads, log_path = task
    logfd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    os.close(logfd)
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    m = safe_run(cfg, label, fast=fast, probe_layers=probe_layers)
    return m, time.perf_counter() - t0


def _print_trial_times(tasks, times):
    print("\n[timing] per-trial wall time:")
    for (_, label), t in zip(tasks, times):
        stamp = f"{t:.1f}s" if t is not None else "-"
        print(f"  {stamp:>9}  {label}")
    done = [t for t in times if t is not None]
    if done:
        print(f"  slowest {max(done):.1f}s  sum {sum(done):.1f}s")


def run_points_parallel(
    tasks,
    jobs=1,
    threads_per_job=None,
    log_dir=None,
    fast=False,
    probe_layers=LAYER_PROBE,
    tag=None,
):
    """Run ``tasks`` -- each a ``(cfg, label)`` -- and return their Metrics in
    the same order (``None`` for a failed / skipped point).

    ``jobs == 1`` runs serially with console logging (unchanged).  ``jobs > 1``
    runs a spawn process pool, diverting each point's output to
    ``<log_dir>/<tag>_<i>_<label>.log`` and printing one console status line per
    completion, so parallel logs never interleave.  ``tag`` names the sweep, so
    logs written beside the tables cannot collide between sweeps."""
    if jobs <= 1:
        results, times = [], []
        for cfg, label in tasks:
            t0 = time.perf_counter()
            results.append(
                safe_run(cfg, label, fast=fast, probe_layers=probe_layers)
            )
            times.append(time.perf_counter() - t0)
        _print_trial_times(tasks, times)
        return results

    threads = threads_per_job or max(1, (os.cpu_count() or 1) // jobs)
    log_dir = log_dir or RESULTS_DIR
    os.makedirs(log_dir, exist_ok=True)
    # OMP/MKL read these at import; spawned children inherit the parent env.
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    def log_path(i, label):
        safe = re.sub(r"[^\w.-]+", "_", label).strip("_") or f"point_{i}"
        stem = f"{tag}_{i:03d}_{safe}" if tag else f"{i:03d}_{safe}"
        return os.path.join(log_dir, f"{stem}.log")

    worker_tasks = [
        (cfg, label, fast, tuple(probe_layers), threads, log_path(i, label))
        for i, (cfg, label) in enumerate(tasks)
    ]
    results = [None] * len(tasks)
    times = [None] * len(tasks)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futs = {
            ex.submit(_run_point_worker, t): i
            for i, t in enumerate(worker_tasks)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            label, log = tasks[i][1], worker_tasks[i][5]
            try:
                results[i], times[i] = fut.result()
            except Exception as e:  # noqa: BLE001 - worker died; keep going
                print(f"  [crash] {label}: {type(e).__name__}: {e}", flush=True)
                continue
            tag = "done" if results[i] is not None else "skip"
            print(f"  [{tag}] {label} ({times[i]:.1f}s) -> {log}", flush=True)
    _print_trial_times(tasks, times)
    return results


# -------------------------------------------------------------------------
# Per-module operator breakdown (copied from test_llama_report.py)
# -------------------------------------------------------------------------
def _is_block(n):
    """A block worth reporting: a fused ``call_module`` kernel, or a bare
    compute ``call_function`` (skip control / nop / data-movement ops)."""
    if n.op == "call_module":
        return True
    if n.op != "call_function":
        return False
    return is_compute_op(n) and isinstance(
        getattr(n, "value", None), (torch.Tensor, list, tuple)
    )


def _layer_block_nodes(gm, layer):
    """The compute blocks of decoder ``layer``: every ``_is_block`` node in
    graph order between this layer's ``input_layernorm`` and the next one's."""
    nodes = list(gm.graph.nodes)

    def ln_index(l):
        # Prefill names layers ``model_layers_{l}_``; the decode wrapper nests
        # one level deeper (``model_model_layers_{l}_``).  Match the
        # ``layers_{l}_input_layernorm`` segment anywhere so both export shapes
        # work (``layers_0_`` does not substring-match ``layers_10_``).
        pat = f"layers_{l}_input_layernorm"
        return next(
            (
                i
                for i, n in enumerate(nodes)
                if n.op != "get_attr" and pat in n.name
            ),
            None,
        )

    start = ln_index(layer)
    if start is None:
        raise ValueError(f"layer {layer} input_layernorm not found")
    end = ln_index(layer + 1) or len(nodes)
    return [n for n in nodes[start:end] if _is_block(n)]


def _extract_standalone(gm, node):
    """A one-block ``GraphModule``: ``node`` with every operand promoted to a
    DRAM placeholder (value + dtype meta carried).  A multi-output node gets
    ``getitem`` outputs so the splice routes correctly."""
    root = torch.nn.Module()
    g = torch.fx.Graph()
    remap = {}
    for inp in node.all_input_nodes:
        ph = g.placeholder(inp.name)
        ph.value = inp.value
        ph.shape = getattr(inp, "shape", None)
        for k, v in inp.meta.items():
            if k != "source_node":
                ph.meta[k] = v
        remap[inp] = ph

    if node.op == "call_module":
        submod = getattr(gm, node.target)
        for p in submod.graph.nodes:
            if p.op == "placeholder" and p.meta.get("source_node") in remap:
                p.meta["source_node"] = remap[p.meta["source_node"]]
        setattr(root, node.target, submod)

    new = g.node_copy(node, lambda n: remap[n])
    new.value = node.value
    new.shape = getattr(node, "shape", None)
    if isinstance(node.value, (list, tuple)):
        outs = []
        for i, v in enumerate(node.value):
            gi = g.call_function(operator.getitem, (new, i))
            gi.value, gi.shape = v, tuple(v.shape)
            outs.append(gi)
        g.output(tuple(outs))
    else:
        g.output(new)

    sub = torch.fx.GraphModule(root, g)
    sub.graph.lint()
    sub.recompile()
    return sub


def _union_len(intervals):
    """Wall-clock covered by ``[start, end)`` intervals (their union)."""
    total = cur_end = 0
    started = False
    for s, e in sorted(intervals):
        if not started or s > cur_end:
            total += e - s
            cur_end = e
            started = True
        elif e > cur_end:
            total += e - cur_end
            cur_end = e
    return total


def _time_split(result):
    """Split a schedule's makespan into ``(compute, memory, overlap, stall)``
    cycles.  Must be called on the *uncompressed* estimate."""
    comp = [
        (r.start, r.end)
        for r in result.records
        if r.latency_kind == "compute" and r.end > r.start
    ]
    dram = [
        (r.start, r.end)
        for r in result.records
        if r.latency_kind == "dram" and r.end > r.start
    ]
    c, m = _union_len(comp), _union_len(dram)
    overlap = c + m - _union_len(comp + dram)
    stall = max(0, result.total_latency - (c + m - overlap))
    return c - overlap, m - overlap, overlap, stall


def _tail_block_nodes(gm):
    """Compute blocks *after* the decoder stack: the final norm and ``lm_head``.
    ``embed_tokens`` is an ``aten.embedding`` lookup (a memory op), so it is not
    a compute block and carries no latency -- it is deliberately excluded."""
    out = []
    for n in gm.graph.nodes:
        if not _is_block(n):
            continue
        low = n.name.lower()
        if "lm_head" in low or ("norm" in low and "layers" not in low):
            out.append(n)
    return out


def _estimate_block(cfg, model, gm, tiler, node, common_kw, dump_dir):
    """Bufferize + plan + estimate one block standalone -> a ``Metrics`` (count
    1), through the *same* ``estimate_schedule`` the whole-graph path uses --
    the only addition is ``_time_split`` for the runtime breakdown, which this
    path can compute because it still holds the ``ScheduleResult``.  With
    ``dump_dir`` set, also write an xlsx + perfetto pair for the block."""
    sub = _extract_standalone(gm, node)
    bufferize_graph(
        sub,
        pipelined=True,
        tiler=tiler,
        single_buffer_tail=cfg.single_buffer_tail,
    )
    plan = _plan(cfg, sub)
    # Split off the *uncompressed* estimate, then compress + dump.
    r = estimate_schedule(sub, **common_kw)
    compute, memory, overlap, stall = _time_split(r)
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        compress_schedule(r)
        write_excel_report(r, os.path.join(dump_dir, f"{node.name}.xlsx"))
        write_perfetto(r, os.path.join(dump_dir, f"{node.name}.perfetto.json"))
    return Metrics(
        total_latency=r.total_latency,
        dram_read_bytes=r.dram_read_bytes,
        dram_write_bytes=r.dram_write_bytes,
        dram_weight_bytes=r.dram_weight_bytes,
        dram_activation_bytes=r.dram_activation_bytes,
        dram_kv_bytes=r.dram_kv_bytes,
        scratchpad_bytes=plan.scratchpad_bytes,
        num_layers=model.config.num_hidden_layers,
        num_params=_num_params(model),
        compute=compute,
        memory=memory,
        overlap=overlap,
        stall=stall,
        name=node.name,
        count=1,
    )


def report_per_module(cfg, layer, dump_dir=None):
    """Operator-level breakdown: bufferize + plan + estimate each block of
    decoder ``layer`` on its own (keeping *raw*, per-block metrics), plus the
    tail blocks (final norm, ``lm_head``).

    Each block's additive metrics are scaled to their whole-model contribution
    by its ``count`` -- ``num_hidden_layers`` for a decoder-layer block (it
    repeats identically in every layer), 1 for a tail block (norm / lm_head).
    With ``dump_dir`` set, also write an xlsx + perfetto pair per block."""
    gm, model, tiler = _frontend(cfg)
    n_layers = model.config.num_hidden_layers
    common_kw = dict(
        dram_bandwidth=cfg.dram_bandwidth_gbs,
        dram_access_latency=cfg.dram_access_latency_ns,
        frequency=cfg.frequency_ghz,
        unroll=cfg.unroll,
    )
    rows = []
    for node in _layer_block_nodes(gm, layer):
        m = _estimate_block(cfg, model, gm, tiler, node, common_kw, dump_dir)
        rows.append(_scale(m, n_layers))
    for node in _tail_block_nodes(gm):
        m = _estimate_block(cfg, model, gm, tiler, node, common_kw, dump_dir)
        rows.append(m)  # tail runs once (count 1); no scaling
    return rows


def _display_name(name):
    """A short label for a block: drop leading ``model_`` scopes and the
    ``layers_<N>_`` prefix and noise tokens, then space-separate (e.g.
    ``model_model_layers_0_self_attn_q_proj_fused`` -> ``self attn q proj``,
    ``model_lm_head`` -> ``lm head``)."""
    name = re.sub(r"^(model_)+", "", name)
    name = re.sub(r"^layers_\d+_", "", name)
    tokens = [t for t in name.split("_") if t and t not in {"default", "fused"}]
    return " ".join(tokens) or name


# -------------------------------------------------------------------------
# Results directory
# -------------------------------------------------------------------------
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)


def default_out_dir() -> str:
    """A fresh run directory under ``RESULTS_DIR``, named for the time it was
    made -- so an interrupted sweep cannot leave a stale ``.xlsx`` behind that
    reads as current.  ``runner.py`` also repoints ``results/latest`` at the
    run dir it uses.
    """
    return os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%H%M%S"))
