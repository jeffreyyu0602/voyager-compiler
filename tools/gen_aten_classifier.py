#!/usr/bin/env python
"""Generate the ATen compute / elementwise classifiers.

Exports a corpus of popular Hugging Face and timm models, harvests every ATen
operator that appears in the exported graphs, decomposes each one down to the
Core ATen operator set, and classifies it by which core operators it bottoms
out in.  Emits ``src/voyager_compiler/codegen/aten_classifier.py``, whose
``is_compute_op`` / ``is_elementwise_op`` are plain frozenset membership tests.

Hugging Face models go through ``transformers.exporters.DynamoExporter``, which
applies per-architecture patches before ``torch.export`` (data-dependent expert
loops, in-place ops, mask checks) and splits a generative model into its prefill
and decode graphs.  timm models are plain ``nn.Module`` and use ``torch.export``
directly.  Attention is forced to ``eager`` so the harvested vocabulary is the
explicit matmul / softmax chain the compiler lowers, not ``sdpa``.

Usage::

    python tools/gen_aten_classifier.py --hf-count 200 --timm-count 50

The classification rule, for a candidate op ``c`` with core decomposition
leaves ``D(c)``:

  * ``c`` is **compute** when at least one leaf is in ``CORE_ATEN_COMPUTE_IR``.
    An op whose decomposition does no arithmetic is pure data movement and must
    not be bufferized as a kernel.
  * ``c`` is **elementwise** when it is compute, *every* compute leaf is in
    ``CORE_ATEN_ELEMENTWISE_IR``, and every remaining leaf is in
    ``CORE_ATEN_PURE_IR`` (a reshape, a broadcast or a materialized constant).
    The last clause is what stops an op from passing on the strength of one
    harmless ``mul`` while quietly also doing a ``gather`` or a ``cat``.

Elementwise is therefore a subset of compute.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
import torch.nn as nn
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
)
from transformers.exporters import DynamoConfig, DynamoExporter

logger = logging.getLogger("gen_aten_classifier")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "tools" / "artifacts"
GENERATED_FILE = (
    REPO_ROOT / "src" / "voyager_compiler" / "codegen" / "aten_classifier.py"
)

BEGIN_MARKER = "# --- BEGIN GENERATED ATEN CLASSIFIER ---"
END_MARKER = "# --- END GENERATED ATEN CLASSIFIER ---"

DEFAULT_HF_COUNT = 200
DEFAULT_TIMM_COUNT = 50
DEFAULT_MAX_WORKERS = min(32, (os.cpu_count() or 8) // 4)
EXPORT_TIMEOUT_S = 600
# Trim every model to this many transformer blocks before exporting: the op
# vocabulary is what we harvest, and it does not depend on depth.
CORPUS_NUM_LAYERS = 2
SEQ_LEN = 8
# Export with symbolic shapes (``Dim.AUTO`` on every dimension).  The op
# vocabulary is what we harvest and it does not depend on the shapes, but a
# symbolic trace exercises the same path a real deployment would.
EXPORT_DYNAMIC = True


# ---------------------------------------------------------------------------
# Core ATen IR seeds
#
# Every entry below is transcribed from the Core ATen IR table at
# https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_ir.html
# An entry without an explicit overload means the ``default`` overload.  The
# four lists partition ``CORE_ATEN_IR``; ``_check_partition`` enforces that.
# ---------------------------------------------------------------------------

# Leaves that do arithmetic: one output element per input element, no
# reduction and no accumulation across positions.
CORE_ATEN_ELEMENTWISE_IR = (
    "abs",
    "acos",
    "acosh",
    "add.Scalar",
    "add.Tensor",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atan2.out",
    "atanh",
    "bitwise_and.Scalar",
    "bitwise_and.Tensor",
    "bitwise_not",
    "bitwise_or.Scalar",
    "bitwise_or.Tensor",
    "bitwise_xor.Scalar",
    "bitwise_xor.Tensor",
    "ceil",
    "clamp",
    "clamp.Tensor",
    "cos",
    "cosh",
    "div.Scalar",
    "div.Scalar_mode",
    "div.Tensor",
    "div.Tensor_mode",
    "elu",
    "eq.Scalar",
    "eq.Tensor",
    "erf",
    "exp",
    "expm1",
    "floor",
    "fmod.Scalar",
    "fmod.Tensor",
    "ge.Scalar",
    "ge.Tensor",
    "gelu",
    "gt.Scalar",
    "gt.Tensor",
    "hardtanh",
    "isinf",
    "isnan",
    "le.Scalar",
    "le.Tensor",
    "leaky_relu",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt.Scalar",
    "lt.Tensor",
    "maximum",
    "minimum",
    "mul.Scalar",
    "mul.Tensor",
    "ne.Scalar",
    "ne.Tensor",
    "neg",
    "pow.Scalar",
    "pow.Tensor_Scalar",
    "pow.Tensor_Tensor",
    "reciprocal",
    "relu",
    "remainder.Scalar",
    "remainder.Tensor",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sub.Scalar",
    "sub.Tensor",
    "tan",
    "tanh",
    "trunc",
    "where.self",
)

# Arithmetic leaves that are *not* elementwise: matmuls, convolutions,
# reductions, normalizations and pooling.
_CORE_ATEN_COMPUTE_ONLY_IR = (
    "_adaptive_avg_pool2d",
    "_adaptive_avg_pool2d_backward",
    "_adaptive_avg_pool3d",
    "_cdist_forward",
    "_embedding_bag",
    "_fft_c2r",
    "_fft_r2c",
    "_log_softmax",
    "_native_batch_norm_legit",
    "_native_batch_norm_legit.no_stats",
    "_native_batch_norm_legit_no_training",
    "_pdist_forward",
    "_softmax",
    "adaptive_avg_pool1d",
    "addmm",
    "amax",
    "amin",
    "any",
    "any.dim",
    "any.dims",
    "argmax",
    "argmin",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "bmm",
    "convolution",
    "convolution_backward",
    "cumsum",
    "embedding_dense_backward",
    "grid_sampler_2d",
    "max.dim",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "max_pool3d_with_indices",
    "mean",
    "mean.dim",
    "min.dim",
    "mm",
    "native_group_norm",
    "native_group_norm_backward",
    "native_layer_norm",
    "native_layer_norm_backward",
    "prod",
    "prod.dim_int",
    "sort",
    "sum.dim_IntList",
    "topk",
    "upsample_bilinear2d.vec",
    "var.correction",
    "var.dim",
)

CORE_ATEN_COMPUTE_IR = CORE_ATEN_ELEMENTWISE_IR + _CORE_ATEN_COMPUTE_ONLY_IR

# Leaves that only reshape, relabel metadata, or materialize a constant.  They
# may appear in an elementwise op's decomposition without disqualifying it.
#
# ``slice.Tensor`` and ``select.int`` are deliberately absent: they subset a
# tensor, so admitting them would let a shape-shrinking op such as ``aten.diff``
# (leaves ``{slice, sub}``) pass as elementwise.  Genuine broadcasting uses
# ``expand`` and ``view``.  Note constants (``full``, ``scalar_tensor``) are
# pure but *generators* (``arange``, ``rand``) are not -- that asymmetry is what
# admits ``masked_fill`` while rejecting ``one_hot``.
CORE_ATEN_PURE_IR = (
    "_local_scalar_dense",
    "_to_copy",
    "alias",
    "as_strided",
    "clone",
    "copy",
    "empty.memory_format",
    "empty_strided",
    "expand",
    "fill.Scalar",
    "full",
    "full_like",
    "permute",
    "resize_",
    "scalar_tensor",
    "squeeze.dim",
    "squeeze.dims",
    "sym_is_contiguous",
    "sym_numel",
    "sym_size.int",
    "sym_storage_offset",
    "sym_stride.int",
    "unsqueeze",
    "view",
)

# Leaves that touch data but do no arithmetic.
CORE_ATEN_DATA_MOVEMENT_IR = (
    "arange.start_step",
    "cat",
    "col2im",
    "constant_pad_nd",
    "diagonal",
    "embedding",
    "flip",
    "gather",
    "index.Tensor",
    "index_put",
    "index_select",
    "masked_scatter",
    "native_dropout",
    "nonzero",
    "rand",
    "randn",
    "randperm",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "repeat",
    "replication_pad2d",
    "replication_pad3d",
    "scatter.src",
    "scatter.value",
    "scatter_add",
    "scatter_reduce.two",
    "select.int",
    "select_scatter",
    "slice.Tensor",
    "slice_scatter",
    "split_with_sizes",
    "upsample_nearest2d.vec",
)

CORE_ATEN_IR = (
    CORE_ATEN_COMPUTE_IR + CORE_ATEN_PURE_IR + CORE_ATEN_DATA_MOVEMENT_IR
)

# Non-core leaves that ``run_decompositions`` leaves behind and that carry no
# data-flow meaning.  Stripped from a decomposition before classification.
IGNORED_LEAVES = frozenset(
    {
        "_operator.getitem",
        "_operator.ge",
        "_operator.le",
        "_operator.lt",
        "_operator.gt",
        "_operator.eq",
        "_operator.add",
        "_operator.sub",
        "_operator.mul",
        "_operator.floordiv",
        "aten._assert_tensor_metadata.default",
        "aten._assert_scalar.default",
        "aten.sym_constrain_range_for_size.default",
        "aten._assert_async.msg",
    }
)


def _check_partition() -> None:
    """The four seed lists must exactly partition the Core ATen IR."""
    compute = set(CORE_ATEN_COMPUTE_IR)
    pure = set(CORE_ATEN_PURE_IR)
    moves = set(CORE_ATEN_DATA_MOVEMENT_IR)
    elementwise = set(CORE_ATEN_ELEMENTWISE_IR)

    for name, a, b in (
        ("compute/pure", compute, pure),
        ("compute/data-movement", compute, moves),
        ("pure/data-movement", pure, moves),
    ):
        overlap = a & b
        assert not overlap, f"{name} seeds overlap: {sorted(overlap)}"

    assert elementwise <= compute, "elementwise seed escapes the compute seed"
    assert len(CORE_ATEN_IR) == len(set(CORE_ATEN_IR)), "duplicate core entry"
    assert len(compute) == len(CORE_ATEN_COMPUTE_IR), "duplicate compute entry"


# ---------------------------------------------------------------------------
# Op-target resolution
# ---------------------------------------------------------------------------


def resolve(target: str):
    """``"aten.add.Tensor"`` / ``"add.Tensor"`` / ``"add"`` -> an OpOverload.

    Returns ``None`` when the installed torch lacks the op or the overload.
    """
    name = target[len("aten.") :] if target.startswith("aten.") else target
    parts = name.split(".")
    packet = getattr(torch.ops.aten, parts[0], None)
    if packet is None:
        return None
    overload = parts[1] if len(parts) > 1 else "default"
    return getattr(packet, overload, None)


def resolve_all(targets) -> set:
    """Resolve each name, warning about (and dropping) the ones this torch
    build does not carry."""
    resolved = set()
    for t in targets:
        op = resolve(t)
        if op is None:
            logger.warning("core aten op %r not present in this torch", t)
        else:
            resolved.add(op)
    return resolved


def cross_check_core_tag(core_ops: set) -> None:
    """Compare the transcribed docs list against ``torch.Tag.core`` and log the
    symmetric difference.  The docs page tracks a newer torch than the pin, so
    a small diff is expected and informational."""
    tagged = set()
    for name in dir(torch.ops.aten):
        packet = getattr(torch.ops.aten, name, None)
        if not isinstance(packet, torch._ops.OpOverloadPacket):
            continue
        for ov in packet.overloads():
            op = getattr(packet, ov)
            try:
                if torch.Tag.core in op.tags:
                    tagged.add(op)
            except (AttributeError, RuntimeError):
                continue
    only_docs = sorted(str(o) for o in core_ops - tagged)
    only_tag = sorted(str(o) for o in tagged - core_ops)
    logger.info(
        "core-tag cross-check: %d docs ops, %d torch.Tag.core ops",
        len(core_ops),
        len(tagged),
    )
    if only_docs:
        logger.info("  in docs list but not Tag.core: %s", ", ".join(only_docs))
    if only_tag:
        logger.info("  in Tag.core but not docs list: %s", ", ".join(only_tag))


# ---------------------------------------------------------------------------
# Step 1: model corpus -> ATen op histogram
# ---------------------------------------------------------------------------

_TEXT_TAGS = {
    "text-generation",
    "fill-mask",
    "text-classification",
    "token-classification",
    "question-answering",
    "sentence-similarity",
    "feature-extraction",
    "zero-shot-classification",
    "summarization",
    "translation",
    "text2text-generation",
}
_IMAGE_TAGS = {
    "image-classification",
    "image-segmentation",
    "object-detection",
    "image-feature-extraction",
    "semantic-segmentation",
    "depth-estimation",
}
SUPPORTED_TAGS = _TEXT_TAGS | _IMAGE_TAGS

_DEPTH_ATTRS = (
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "num_encoder_layers",
    "num_decoder_layers",
    "encoder_layers",
    "decoder_layers",
)

# A 100B+ MoE materializes hundreds of expert weight tensors even at two layers,
# which is what times the export out.  The expert *count* does not change the op
# vocabulary, only how many times each op is instantiated, so trim it too --
# never below the router's top-k, which would make the config inconsistent.
_EXPERT_ATTRS = ("num_experts", "num_local_experts", "n_routed_experts")
_TOPK_ATTRS = (
    "num_experts_per_tok",
    "num_experts_per_token",
    "top_k_experts",
    "top_k",
)
CORPUS_NUM_EXPERTS = 2


def _is_timm(info) -> bool:
    tags = set(getattr(info, "tags", None) or ())
    return "timm" in tags or info.id.startswith("timm/")


def discover_models(hf_count: int, timm_count: int) -> list:
    """The most-downloaded HF models with a supported input contract, plus the
    most-downloaded timm models.  Returns ``[(repo_id, source), ...]``."""
    api = HfApi()
    picked: list = []
    seen: set = set()

    for info in api.list_models(sort="downloads", limit=hf_count * 4):
        if len(picked) >= hf_count:
            break
        if _is_timm(info) or info.id in seen:
            continue
        if getattr(info, "pipeline_tag", None) not in SUPPORTED_TAGS:
            continue
        seen.add(info.id)
        picked.append((info.id, "hf"))
    logger.info("discovered %d hugging-face models", len(picked))

    n_timm = 0
    for info in api.list_models(
        author="timm", sort="downloads", limit=timm_count * 2
    ):
        if n_timm >= timm_count:
            break
        if info.id in seen:
            continue
        seen.add(info.id)
        picked.append((info.id, "timm"))
        n_timm += 1
    logger.info("discovered %d timm models", n_timm)
    return picked


def _sub_configs(config):
    """``config`` and every nested ``PretrainedConfig`` beneath it.

    A multimodal model keeps its depth on ``text_config`` / ``vision_config``,
    not at the top level, so a non-recursive walk silently shrinks nothing and
    we export a 26B model in full.
    """
    yield config
    for value in vars(config).values():
        if isinstance(value, PretrainedConfig):
            yield from _sub_configs(value)


def _shrink(config) -> None:
    for cfg in _sub_configs(config):
        for attr in _DEPTH_ATTRS:
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > CORPUS_NUM_LAYERS:
                setattr(cfg, attr, CORPUS_NUM_LAYERS)

        top_k = max(
            (
                v
                for attr in _TOPK_ATTRS
                if isinstance(v := getattr(cfg, attr, None), int) and v > 0
            ),
            default=1,
        )
        floor = max(CORPUS_NUM_EXPERTS, top_k)
        for attr in _EXPERT_ATTRS:
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > floor:
                setattr(cfg, attr, floor)


def _image_size(model_id: str) -> tuple:
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        return (224, 224)
    size = getattr(proc, "size", None) or {}
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return (int(size["height"]), int(size["width"]))
        edge = size.get("shortest_edge")
        if edge:
            return (int(edge), int(edge))
    return (224, 224)


def _is_causal_lm(config, tag: str) -> bool:
    architectures = tuple(getattr(config, "architectures", None) or ())
    return tag == "text-generation" or any(
        name.endswith("ForCausalLM") for name in architectures
    )


def _export_causal_lm(config) -> dict:
    """Prefill and decode export to *different* operator sets, so harvest both.

    ``export_for_generation`` runs ``generate(max_new_tokens=2)`` once and hooks
    ``forward`` to capture the real prefill and decode kwargs, then exports each
    stage.  Decode is where the cache-update vocabulary lives (``index_copy_``,
    ``copy_``, ``slice``, ``cat``) along with the in-place ops that a plain
    functional export never produces.  A multimodal model is split further, into
    one program per encoder / projector / language-model / lm_head.
    """
    # ``get_text_config()`` is the config itself for a plain text model and the
    # nested text tower for a multimodal one, whose top level has no vocab_size.
    vocab = getattr(config.get_text_config(), "vocab_size", None)
    if not isinstance(vocab, int) or vocab <= 0:
        raise ValueError("causal LM without a vocab_size")

    # Eager attention, not ``sdpa``: the compiler lowers the explicit
    # matmul / softmax / div chain, so that is the vocabulary to harvest.
    # (Flash and flex attention are not exportable on any backend.)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        return DynamoExporter().export_for_generation(
            model,
            {
                "input_ids": torch.randint(0, vocab, (1, SEQ_LEN)),
                "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
            },
            config=DynamoConfig(dynamic=EXPORT_DYNAMIC),
        )


def _export_hf(model_id: str) -> dict:
    """``{variant: ExportedProgram}`` for a Hugging Face repo, built from config
    and metadata only -- weights are never downloaded and remote code is never
    executed."""
    info = HfApi().model_info(model_id)
    tag = getattr(info, "pipeline_tag", None)
    if tag not in SUPPORTED_TAGS:
        raise ValueError(f"unsupported pipeline_tag {tag!r}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    _shrink(config)

    if tag in _TEXT_TAGS and _is_causal_lm(config, tag):
        try:
            return _export_causal_lm(config)
        except Exception as exc:  # noqa: BLE001 - fall back to the plain export
            logger.warning(
                "%s: causal-LM export failed (%s), falling back to prefill",
                model_id,
                exc,
            )

    model = AutoModel.from_config(
        config, trust_remote_code=False, attn_implementation="eager"
    ).eval()

    if tag in _IMAGE_TAGS:
        height, width = _image_size(model_id)
        channels = getattr(config, "num_channels", 3) or 3
        inputs = {"pixel_values": torch.randn(1, channels, height, width)}
    else:
        vocab = getattr(config.get_text_config(), "vocab_size", None)
        if not isinstance(vocab, int) or vocab <= 0:
            raise ValueError("text model without a vocab_size")
        input_ids = torch.randint(0, vocab, (1, SEQ_LEN))
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
        }
        if getattr(config, "is_encoder_decoder", False):
            inputs["decoder_input_ids"] = input_ids.clone()

    # ``prepare_model_and_inputs`` pops ``use_cache`` / ``return_dict`` and
    # applies them to the config, so a decoder no longer returns a
    # ``DynamicCache`` that ``torch.export`` cannot flatten.
    with torch.no_grad():
        return {
            "forward": DynamoExporter().export(
                model, inputs, config=DynamoConfig(dynamic=EXPORT_DYNAMIC)
            )
        }


def _export_timm(model_id: str) -> dict:
    name = model_id.split("/", 1)[1] if "/" in model_id else model_id
    model = timm.create_model(name, pretrained=False).eval()
    config = timm.data.resolve_data_config({}, model=model)
    channels, height, width = config.get("input_size", (3, 224, 224))
    inputs = (torch.randn(1, channels, height, width),)
    with torch.no_grad():
        return {"forward": torch.export.export(model, inputs, strict=False)}


def _spec(value):
    """Serialize one FX argument into a replayable template, or raise."""
    if isinstance(value, torch.fx.Node):
        val = value.meta.get("val")
        # A symbolic size argument specializes to the value it held while
        # tracing; an unbacked symbol has none and raises.
        if isinstance(val, torch.SymInt):
            return int(val)
        if isinstance(val, torch.SymFloat):
            return float(val)
        if isinstance(val, torch.SymBool):
            return bool(val)
        if not isinstance(val, torch.Tensor):
            raise TypeError("non-tensor node argument")
        shape = [int(s) for s in val.shape]  # raises on unbacked SymInt
        return {"__tensor__": [shape, str(val.dtype)]}
    if isinstance(value, torch.dtype):
        return {"__dtype__": str(value)}
    if isinstance(value, torch.device):
        return {"__device__": str(value)}
    if isinstance(value, torch.layout):
        return {"__layout__": str(value)}
    if isinstance(value, torch.memory_format):
        return {"__memory_format__": str(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_spec(v) for v in value]
    raise TypeError(f"unserializable argument {type(value).__name__}")


def _record_example(node) -> dict | None:
    try:
        return {
            "args": [_spec(a) for a in node.args],
            "kwargs": {k: _spec(v) for k, v in node.kwargs.items()},
        }
    except (TypeError, ValueError, RuntimeError):
        return None


def export_one(model_id: str, source: str) -> dict:
    """Export one model -- every variant of it -- and harvest its ATen ops plus
    one replayable example call per op.  Raises on failure; caller records it.
    """
    programs = (
        _export_timm(model_id) if source == "timm" else _export_hf(model_id)
    )

    ops: set = set()
    examples: dict = {}
    for exported in programs.values():
        for node in exported.graph.nodes:
            if node.op != "call_function":
                continue
            target = str(node.target)
            if not target.startswith("aten."):
                continue
            ops.add(target)
            if target not in examples:
                example = _record_example(node)
                if example is not None:
                    examples[target] = example
    return {
        "id": model_id,
        "source": source,
        "variants": sorted(programs),
        "ops": sorted(ops),
        "ex": examples,
    }


def _worker(model_id: str, source: str, shard: Path) -> None:
    record = export_one(model_id, source)
    shard.write_text(json.dumps(record))


def build_corpus(models: list, max_workers: int, shard_dir: Path) -> tuple:
    """Run one export subprocess per model.  ``torch.export`` mutates global
    dispatcher state and can hang or segfault, so each model gets its own
    interpreter and a hard timeout; a failure never takes the pool down.

    Returns ``(records, failures)``.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    records: list = []
    failures: list = []

    def run(item):
        model_id, source = item
        shard = shard_dir / (model_id.replace("/", "__") + ".json")
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--export-one",
                model_id,
                "--source",
                source,
                "--shard",
                str(shard),
            ],
            capture_output=True,
            text=True,
            timeout=EXPORT_TIMEOUT_S,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0 or not shard.exists():
            tail = (proc.stderr or "").strip().splitlines()
            raise RuntimeError(tail[-1] if tail else "export subprocess failed")
        return json.loads(shard.read_text())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, item): item for item in models}
        for future in as_completed(futures):
            model_id, source = futures[future]
            try:
                record = future.result()
            except subprocess.TimeoutExpired:
                logger.warning("export FAILED %s: timeout", model_id)
                failures.append(
                    (model_id, "TimeoutExpired", "export timed out")
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - one model must not stop us
                logger.warning(
                    "export FAILED %s: %s: %s",
                    model_id,
                    type(exc).__name__,
                    exc,
                )
                failures.append((model_id, type(exc).__name__, str(exc)))
            else:
                logger.info(
                    "export ok %s (%s): %d aten ops",
                    model_id,
                    source,
                    len(record["ops"]),
                )
                records.append(record)
    return records, failures


# ---------------------------------------------------------------------------
# Step 2: classify each candidate by its Core ATen decomposition
# ---------------------------------------------------------------------------


def _make_tensor(shape, dtype_str: str) -> torch.Tensor:
    dtype = getattr(torch, dtype_str.split(".")[-1])
    if dtype.is_floating_point:
        return torch.rand(shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.zeros(shape, dtype=dtype)
    # Integers stand in for indices, so zero is always in range.
    return torch.zeros(shape, dtype=dtype)


def _instantiate(template, tensors: list):
    """Rebuild an argument from its template, appending any tensor it needs to
    ``tensors`` and leaving a positional slot marker in its place."""
    if isinstance(template, list):
        return [_instantiate(t, tensors) for t in template]
    if isinstance(template, dict):
        if "__tensor__" in template:
            shape, dtype = template["__tensor__"]
            tensors.append(_make_tensor(shape, dtype))
            return _Slot(len(tensors) - 1)
        if "__dtype__" in template:
            return getattr(torch, template["__dtype__"].split(".")[-1])
        if "__device__" in template:
            return torch.device(template["__device__"])
        if "__layout__" in template:
            return getattr(torch, template["__layout__"].split(".")[-1])
        if "__memory_format__" in template:
            return getattr(torch, template["__memory_format__"].split(".")[-1])
    return template


class _Slot:
    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index


def _fill(value, tensors):
    if isinstance(value, _Slot):
        return tensors[value.index]
    if isinstance(value, list):
        return [_fill(v, tensors) for v in value]
    return value


class _OneOp(nn.Module):
    """A module whose forward is a single call to ``op``, with the recorded
    literal arguments baked in and the tensor arguments taken positionally."""

    def __init__(self, op, args, kwargs) -> None:
        super().__init__()
        self.op = op
        self.arg_template = args
        self.kwarg_template = kwargs

    def forward(self, *tensors):
        args = [_fill(a, tensors) for a in self.arg_template]
        kwargs = {k: _fill(v, tensors) for k, v in self.kwarg_template.items()}
        return self.op(*args, **kwargs)


def decompose_to_core(op, example: dict) -> set | None:
    """The set of Core ATen leaves ``op`` decomposes into, or ``None`` when the
    op cannot be exported (data-dependent output, exotic overload).

    ``ExportedProgram.run_decompositions()`` with no arguments applies the
    default table, which targets the Core ATen operator set, and it is already
    transitive -- ``hardswish`` reaches ``{add, clamp, mul, div}`` in one call.
    """
    tensors: list = []
    args = [_instantiate(a, tensors) for a in example["args"]]
    kwargs = {k: _instantiate(v, tensors) for k, v in example["kwargs"].items()}
    if not tensors:
        return None  # a pure factory call; nothing to trace through

    module = _OneOp(op, args, kwargs)
    with torch.no_grad():
        exported = torch.export.export(module, tuple(tensors), strict=False)
        decomposed = exported.run_decompositions()

    leaves = set()
    for node in decomposed.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        name = (
            f"{target.__module__}.{target.__name__}"
            if not isinstance(target, torch._ops.OpOverload)
            else str(target)
        )
        if name in IGNORED_LEAVES:
            continue
        leaves.add(
            target if isinstance(target, torch._ops.OpOverload) else name
        )
    return leaves


def _counterpart(op):
    """The other member of ``op``'s in-place / functional pair, or ``None``.

    ``aten.add.Tensor`` <-> ``aten.add_.Tensor``, in whichever direction ``op``
    is not.
    """
    name = str(op)[len("aten.") :]
    packet, _, overload = name.partition(".")
    packet = packet[:-1] if packet.endswith("_") else packet + "_"
    return resolve(f"{packet}.{overload}")


def select_by_decomposition(
    candidates: set,
    seed: set,
    allowed_non_compute: set,
    compute_seed: set,
    core_ir: set,
    examples: dict,
) -> tuple:
    """Candidates whose Core ATen decomposition bottoms out in ``seed``.

    A candidate qualifies when its decomposition leaves include at least one op
    in ``compute_seed``, every such compute leaf is also in ``seed``, and every
    remaining leaf is in ``allowed_non_compute``.  A candidate that is already
    a Core ATen op *is* its own decomposition, so it qualifies exactly when it
    is a seed member -- never trace one.  Returns
    ``(qualifying, leaves_by_candidate)``.
    """
    qualifying: set = set()
    leaves_by_candidate: dict = {}

    for op in sorted(candidates, key=str):
        if op in core_ir:
            if op in seed:
                qualifying.add(op)
            leaves_by_candidate[op] = {op}
            continue

        example = examples.get(str(op))
        if example is None:
            logger.warning("no recorded example for %s", op)
            continue
        try:
            leaves = decompose_to_core(op, example)
        except Exception as exc:  # noqa: BLE001 - an op we simply cannot trace
            logger.warning(
                "cannot decompose %s: %s: %s", op, type(exc).__name__, exc
            )
            continue
        if not leaves:
            continue
        leaves_by_candidate[op] = leaves

        compute_leaves = leaves & compute_seed
        if not compute_leaves:
            continue
        if not compute_leaves <= seed:
            continue
        if not (leaves - compute_seed) <= allowed_non_compute:
            continue
        qualifying.add(op)

    return qualifying, leaves_by_candidate


def _all_overloads() -> set:
    """Every ATen overload this torch registers -- the optional broad candidate
    set, classified wherever an example can be synthesized."""
    ops = set()
    for name in dir(torch.ops.aten):
        packet = getattr(torch.ops.aten, name, None)
        if not isinstance(packet, torch._ops.OpOverloadPacket):
            continue
        for overload in packet.overloads():
            ops.add(getattr(packet, overload))
    return ops


def _counterparts(ops: set) -> set:
    """The counterpart of each op that has one."""
    return {other for op in ops if (other := _counterpart(op)) is not None}


def classify(
    histogram: dict, examples: dict, core: dict, all_overloads: bool = False
) -> dict:
    """Run the selector twice: once for compute, once for elementwise.

    Only ops the corpus actually observed are decomposed, since those are the
    ones whose arguments we can replay.  A Core ATen op needs no test: it *is*
    its own decomposition, so the seed lists give its class directly.  Nor does
    a counterpart: an in-place op computes exactly what its functional form
    computes and then stores the result in its first operand, so the two always
    share a class.

        tested = observed - core
        compute = classified(tested) | core_compute, closed under counterparts
        elemwise = classified(tested) | core_elemwise, closed under counterparts
    """
    candidates = set()
    for target in histogram:
        op = resolve(target)
        if op is None:
            logger.warning("observed op %r no longer resolves", target)
            continue
        candidates.add(op)

    candidates -= core["all"]
    if all_overloads:
        candidates |= _all_overloads()

    logger.info("classifying %d candidate aten overloads", len(candidates))

    compute, compute_leaves = select_by_decomposition(
        candidates,
        seed=core["compute"],
        allowed_non_compute=core["all"],
        compute_seed=core["compute"],
        core_ir=core["all"],
        examples=examples,
    )
    compute |= core["compute"]
    compute |= _counterparts(compute)
    logger.info("classified %d compute ops", len(compute))

    elementwise, _ = select_by_decomposition(
        candidates,
        seed=core["elementwise"],
        allowed_non_compute=core["pure"],
        compute_seed=core["compute"],
        core_ir=core["all"],
        examples=examples,
    )
    elementwise |= core["elementwise"]
    elementwise |= _counterparts(elementwise)
    logger.info("classified %d elementwise ops", len(elementwise))

    assert elementwise <= compute, "elementwise must be a subset of compute"

    unknown = {
        str(op): sorted(str(x) for x in leaves - core["all"])
        for op, leaves in compute_leaves.items()
        if not leaves <= core["all"]
    }
    for op, leaves in unknown.items():
        logger.warning("%s decomposes to non-core leaves %s", op, leaves)

    return {"compute": compute, "elementwise": elementwise}


# ---------------------------------------------------------------------------
# Step 3: emit the classifier module
# ---------------------------------------------------------------------------

_MODULE_TEMPLATE = '''"""ATen compute / elementwise classifiers.

GENERATED FILE -- DO NOT EDIT MANUALLY.  Regenerate with::

    python tools/gen_aten_classifier.py

An op is **compute** when its Core ATen decomposition performs arithmetic
anywhere, and **elementwise** when all of that arithmetic is elementwise and
the rest of the decomposition only reshapes, broadcasts or materializes
constants.  Elementwise is a subset of compute.

Generated:              {timestamp}
Torch version:          {torch_version}
Model corpus size:      {corpus_size}
  Hugging Face count:   {hf_count}
  timm count:           {timm_count}
  exported models:      {exported}
  failed models:        {failed}
Unique ATen ops seen:   {unique_ops}
Classified compute:     {n_compute}
Classified elementwise: {n_elementwise}
"""

import functools

import torch
from torch.fx import Node

{begin}
_GENERATED_COMPUTE_OPS = (
{compute_ops})

_GENERATED_ELEMENTWISE_OPS = (
{elementwise_ops})
{end}

# Hand-maintained.  ``quantized_ops`` are CompositeExplicitAutograd custom ops
# with no decomposition -- they survive ``run_decompositions`` unchanged, so the
# generator cannot classify them and never tries to.
_QUANTIZED_COMPUTE_OPS = (
    "_adaptive_avg_pool2d",
    "adaptive_avg_pool2d",
    "avg_pool2d",
    "conv2d",
    "conv2d_mx",
    "dequantize",
    "layer_norm",
    "linear",
    "linear_mx",
    "matmul",
    "matmul_mx",
    "max_pool2d",
    "quantize",
    "quantize_mx",
    "quantize_mx_outlier",
    "spmm_csr",
    "vmap",
)

_QUANTIZED_ELEMENTWISE_OPS = ("dequantize", "quantize", "vmap")


def _quantized(names):
    """Resolve ``quantized_ops`` targets, skipping any not yet registered."""
    ops = set()
    for name in names:
        packet = getattr(torch.ops.quantized_ops, name, None)
        if packet is not None:
            ops.add(packet.default)
    return ops


# The custom ops register on package import, after this module is first read,
# so both unions are built on first predicate call rather than at import time.
@functools.cache
def _compute_ops() -> frozenset:
    return frozenset(_GENERATED_COMPUTE_OPS) | _quantized(
        _QUANTIZED_COMPUTE_OPS
    )


@functools.cache
def _elementwise_ops() -> frozenset:
    return frozenset(_GENERATED_ELEMENTWISE_OPS) | _quantized(
        _QUANTIZED_ELEMENTWISE_OPS
    )


def is_compute_op(node: Node) -> bool:
    """Whether ``node`` performs arithmetic -- a kernel, not data movement."""
    return node.target in _compute_ops()


def is_elementwise_op(node: Node) -> bool:
    """Whether ``node`` maps one output element per input element."""
    return node.target in _elementwise_ops()
'''


def _render_ops(ops: set) -> str:
    """One op per line, sorted by target name.

    The sort is the review mechanism: because the emitted order is stable, a
    regenerated file diffs against the committed one as a clean set of added
    and removed lines, so any membership change is obvious in code review.
    """
    return "".join(
        f"    torch.ops.{op},\n" for op in sorted(str(o) for o in ops)
    )


def emit(result: dict, stats: dict) -> None:
    text = _MODULE_TEMPLATE.format(
        begin=BEGIN_MARKER,
        end=END_MARKER,
        compute_ops=_render_ops(result["compute"]),
        elementwise_ops=_render_ops(result["elementwise"]),
        n_compute=len(result["compute"]),
        n_elementwise=len(result["elementwise"]),
        **stats,
    )
    GENERATED_FILE.write_text(text)
    logger.info("wrote %s", GENERATED_FILE)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--line-length",
            "80",
            "--target-version",
            "py312",
            "--quiet",
            str(GENERATED_FILE),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-count", type=int, default=DEFAULT_HF_COUNT)
    parser.add_argument("--timm-count", type=int, default=DEFAULT_TIMM_COUNT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--reuse-histogram",
        action="store_true",
        help="skip the corpus export and classify the existing artifacts",
    )
    parser.add_argument(
        "--all-overloads",
        action="store_true",
        help="also classify every registered aten overload, not just the "
        "observed ones (slower, and the extra ops are unobserved in practice)",
    )
    parser.add_argument("--export-one", metavar="MODEL_ID")
    parser.add_argument("--source", default="hf", choices=("hf", "timm"))
    parser.add_argument("--shard", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    if args.export_one:
        _worker(args.export_one, args.source, args.shard)
        return

    _check_partition()
    core = {
        "all": resolve_all(CORE_ATEN_IR),
        "compute": resolve_all(CORE_ATEN_COMPUTE_IR),
        "elementwise": resolve_all(CORE_ATEN_ELEMENTWISE_IR),
        "pure": resolve_all(CORE_ATEN_PURE_IR),
    }
    cross_check_core_tag(core["all"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    histogram_path = ARTIFACT_DIR / "aten_ops_histogram.json"
    examples_path = ARTIFACT_DIR / "aten_op_examples.json"
    jsonl_path = ARTIFACT_DIR / "aten_ops_by_model.jsonl"
    failures_path = ARTIFACT_DIR / "export_failures.log"
    stats_path = ARTIFACT_DIR / "corpus_stats.json"

    if args.reuse_histogram:
        histogram = json.loads(histogram_path.read_text())
        examples = json.loads(examples_path.read_text())
        stats = json.loads(stats_path.read_text())
    else:
        started = time.time()
        models = discover_models(args.hf_count, args.timm_count)
        records, failures = build_corpus(
            models, args.max_workers, ARTIFACT_DIR / "shards"
        )
        logger.info(
            "exported %d/%d models in %.1fs",
            len(records),
            len(models),
            time.time() - started,
        )

        histogram = Counter()
        examples = {}
        with jsonl_path.open("w") as handle:
            for record in records:
                histogram.update(record["ops"])
                for target, example in record["ex"].items():
                    examples.setdefault(target, example)
                handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "source": record["source"],
                            "variants": record["variants"],
                            "ops": record["ops"],
                        }
                    )
                    + "\n"
                )
        histogram = dict(
            sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        histogram_path.write_text(json.dumps(histogram, indent=2))
        examples_path.write_text(json.dumps(examples, indent=2))
        failures_path.write_text(
            "\n".join(f"{i}\t{kind}\t{msg}" for i, kind, msg in failures)
        )
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "torch_version": torch.__version__,
            "corpus_size": len(models),
            "hf_count": args.hf_count,
            "timm_count": args.timm_count,
            "exported": len(records),
            "failed": len(failures),
            "unique_ops": len(histogram),
        }
        stats_path.write_text(json.dumps(stats, indent=2))

    result = classify(histogram, examples, core, args.all_overloads)
    emit(result, stats)

    print()
    print("=" * 68)
    print(f"models discovered:         {stats['corpus_size']}")
    print(f"models exported:           {stats['exported']}")
    print(f"models failed:             {stats['failed']}")
    print(f"unique aten ops observed:  {stats['unique_ops']}")
    print(f"classified compute:        {len(result['compute'])}")
    print(f"classified elementwise:    {len(result['elementwise'])}")
    print("=" * 68)


if __name__ == "__main__":
    main()
