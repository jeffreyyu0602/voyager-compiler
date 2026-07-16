"""Unified design-space sweep: every axis in one run, one aggregate workbook.

Replaces the individual ``sweep_context / sweep_model_size / sweep_quant /
sweep_family / sweep_hardware`` scripts.  All of them radiate from the same
baseline design point (Llama-3.1-8B, 64x64 PE, 1 MB cache, W8A8, length 1024),
so run separately they recompile that point many times over.  Here every axis's
points are collected, **deduplicated by config** (a ``SweepConfig`` fully
determines a compile, so ``astuple(cfg)`` is the key), and run through one
shared parallel pool -- the baseline point compiles once and its result feeds
every sweep that shares it.

The output is a single ``results.xlsx`` with one named sheet per sweep:

* ``baseline_prefill`` / ``baseline_decode`` -- the per-module runtime
  breakdown (stacked-bar chart), via :mod:`per_module_latency_chart`.
* ``context`` / ``model_size`` / ``quant`` / ``family`` / ``hardware`` --
  total latency (seconds) + total DRAM traffic (GB) per config point, as the
  dual-axis chart in :mod:`latency_dram_chart` (prefill + decode; hardware's
  pe / sram / bw sub-sweeps each get their own pair).

    python benchmarks/runner.py
    python benchmarks/runner.py --only quant --only hardware
    python benchmarks/runner.py --jobs 8 --fast
"""

import argparse
import os
from collections import namedtuple
from dataclasses import astuple

import common
import latency_dram_chart as ldc

# -- axis definitions (moved from the individual sweep scripts) ---------------
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]

MODEL_SIZES = [
    ("meta-llama/Llama-3.2-1B", "1B"),
    ("meta-llama/Llama-3.2-3B", "3B"),
    ("meta-llama/Llama-2-7b-hf", "7B"),
    ("meta-llama/Llama-3.1-8B", "8B"),
    ("meta-llama/Llama-2-13b-hf", "13B"),
]

# (label, weight_bits, act_bits) -- KV is fixed BF16.
QUANT_CONFIGS = [
    ("W16A16", 16, 16),
    ("W8A8", 8, 8),
    ("W4A8", 4, 8),
    ("W4A4", 4, 4),
]

FAMILY_MODELS = [
    ("meta-llama/Llama-3.1-8B", "Llama 3.1"),
    ("mistralai/Mistral-7B-v0.3", "Mistral"),
    ("Qwen/Qwen2.5-7B", "Qwen 2.5"),
    ("google/gemma-2-9b", "Gemma 2"),
]

PE_ARRAYS = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
SRAM_EFFECTIVE_MB = [0.5, 1, 2, 4, 8, 16]
BANDWIDTH_SCALES = [0.125, 0.25, 0.5, 1, 2, 4]

MODES = ("prefill", "decode")

# The sweep whose model is fixed at the baseline carries it in the chart title;
# model-varying sweeps (model_size / family) leave the prefix blank.
BASELINE_PREFIX = "Llama 3.1 8B"

# One design point: which sheet/group it lands in, its category label + mode,
# and the config to compile.
Point = namedtuple("Point", "sheet prefix group axis_title label mode cfg")

# Sheet order in the workbook (baseline breakdown sheets are prepended).
SHEET_ORDER = ["context", "model_size", "quant", "family", "hardware"]


def build_points(args):
    """Every whole-graph design point across all axes, as a flat list."""
    pts = []

    for mode in MODES:
        for length in CONTEXT_LENGTHS:
            cfg = common.config_from_args(
                args, mode=mode, prompt_len=length, kv_len=length
            )
            pts.append(
                Point(
                    "context",
                    BASELINE_PREFIX,
                    "",
                    "Context Length",
                    str(length),
                    mode,
                    cfg,
                )
            )

    for model_id, label in MODEL_SIZES:
        for mode in MODES:
            cfg = common.config_from_args(args, model_id=model_id, mode=mode)
            pts.append(
                Point(
                    "model_size",
                    "",
                    "",
                    "Model Size",
                    label,
                    mode,
                    cfg,
                )
            )

    for label, w, a in QUANT_CONFIGS:
        for mode in MODES:
            cfg = common.config_from_args(
                args, mode=mode, weight_bits=w, act_bits=a
            )
            pts.append(
                Point(
                    "quant", BASELINE_PREFIX, "", "Precision", label, mode, cfg
                )
            )

    for model_id, label in FAMILY_MODELS:
        for mode in MODES:
            cfg = common.config_from_args(args, model_id=model_id, mode=mode)
            pts.append(
                Point("family", "", "", "Model Family", label, mode, cfg)
            )

    for pe in PE_ARRAYS:
        for mode in MODES:
            cfg = common.config_from_args(args, mode=mode, pe=pe)
            pts.append(
                Point(
                    "hardware",
                    BASELINE_PREFIX,
                    "pe",
                    "PE Array Size",
                    f"{pe[0]}x{pe[1]}",
                    mode,
                    cfg,
                )
            )
    for mb in SRAM_EFFECTIVE_MB:
        # Effective SRAM = 2 * cache_size (double buffering).
        cache_size = int(mb * 1024 * 1024) // 2
        num_banks = max(1, cache_size // common.SRAM_BANK_SIZE)
        for mode in MODES:
            cfg = common.config_from_args(
                args, mode=mode, cache_size=cache_size, num_banks=num_banks
            )
            pts.append(
                Point(
                    "hardware",
                    BASELINE_PREFIX,
                    "sram",
                    "On-chip SRAM Size",
                    f"{mb}MB",
                    mode,
                    cfg,
                )
            )
    for scale in BANDWIDTH_SCALES:
        gbs = common.BASELINE_DRAM_BANDWIDTH_GBS * scale
        for mode in MODES:
            cfg = common.config_from_args(
                args, mode=mode, dram_bandwidth_gbs=gbs
            )
            pts.append(
                Point(
                    "hardware",
                    BASELINE_PREFIX,
                    "bw",
                    "DRAM Bandwidth",
                    f"{scale:g}x",
                    mode,
                    cfg,
                )
            )
    return pts


def _metrics_dict(m):
    """A Metrics -> the uniform field dict the metric sheet writes."""
    return {
        "total_latency": m.total_latency,
        "dram_total": m.dram_total_bytes,
        "dram_read": m.dram_read_bytes,
        "dram_write": m.dram_write_bytes,
        "dram_weight": m.dram_weight_bytes,
        "dram_activation": m.dram_activation_bytes,
        "dram_kv": m.dram_kv_bytes,
        "scratchpad": m.scratchpad_bytes,
        "num_layers": m.num_layers,
        "num_params": m.num_params,
    }


def _metric_sheets(points, metric_by_key):
    """Group the (surviving) design points back into ordered MetricSheets."""
    sheets = []
    for name in SHEET_ORDER:
        spts = [p for p in points if p.sheet == name]
        if not spts:
            continue
        groups = []
        seen = []
        for p in spts:  # preserve first-seen group order
            if p.group not in seen:
                seen.append(p.group)
        for gkey in seen:
            gpts = [p for p in spts if p.group == gkey]
            rows = []
            for p in gpts:
                m = metric_by_key.get(astuple(p.cfg))
                if m is None:
                    continue
                rows.append(
                    {
                        "group": gkey,
                        "point": p.label,
                        "mode": p.mode,
                        **_metrics_dict(m),
                    }
                )
            if rows:
                groups.append(ldc.MetricGroup(gpts[0].axis_title, rows))
        if groups:
            sheets.append(ldc.MetricSheet(name, spts[0].prefix, groups))
    return sheets


def _baseline_sheets(args):
    """The per-module breakdown sheets (prefill + decode), or [] on failure.
    With ``--dump``, also write an xlsx + perfetto pair per block."""
    sheets = []
    for mode, head in (("prefill", "Prefill"), ("decode", "Decode")):
        cfg = common.config_from_args(args, mode=mode)
        dump_dir = (
            os.path.join(args.out, f"baseline_{mode}_blocks")
            if args.dump
            else None
        )
        try:
            rows = common.report_per_module(cfg, args.layer, dump_dir=dump_dir)
        except Exception as e:  # noqa: BLE001 - a bad baseline must not abort
            print(f"  [skip] baseline {mode}: {type(e).__name__}: {e}")
            continue
        title = f"{BASELINE_PREFIX} {head} Latency (cycles)"
        sheets.append(
            ldc.BreakdownSheet(
                f"baseline_{mode}", title, rows, common._display_name
            )
        )
    return sheets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # The axes drive these fields, so they are not user flags.
    common.add_config_args(
        p,
        exclude={
            "mode",
            "model_id",
            "weight_bits",
            "act_bits",
            "prompt_len",
            "kv_len",
            "pe",
            "cache_size",
            "num_banks",
            "dram_bandwidth_gbs",
        },
    )
    p.add_argument(
        "--only",
        action="append",
        choices=["baseline", *SHEET_ORDER],
        help="Restrict to these sheets (repeatable); default: all.",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Decoder layer for the baseline per-module breakdown.",
    )
    p.add_argument(
        "--dump",
        action="store_true",
        help="Also write an xlsx + perfetto pair per baseline block, into "
        "<out>/baseline_<mode>_blocks/.",
    )
    p.add_argument("--out", default=common.default_out_dir())
    return p.parse_args()


def main():
    args = parse_args()
    selected = set(args.only) if args.only else {"baseline", *SHEET_ORDER}

    points = [p for p in build_points(args) if p.sheet in selected]

    # Deduplicate configs: the baseline point is shared across most sweeps, so
    # compile each distinct config once and fan the result back out.
    unique = {}
    for p in points:
        key = astuple(p.cfg)
        unique.setdefault(key, (p.cfg, f"{p.sheet}:{p.label}:{p.mode}"))
    keys = list(unique)
    print(
        f"[sweep] {len(points)} design points -> "
        f"{len(keys)} unique compiles"
    )

    results = common.run_points_parallel(
        [unique[k] for k in keys],
        args.jobs,
        args.threads_per_job,
        args.log_dir or args.out,
        fast=args.fast,
        probe_layers=args.probe_layers,
        tag="sweep",
    )
    metric_by_key = dict(zip(keys, results))

    sheets = []
    if "baseline" in selected:
        sheets += _baseline_sheets(args)
    sheets += _metric_sheets(points, metric_by_key)

    os.makedirs(args.out, exist_ok=True)
    # Point results/latest at this run (a fresh timestamped dir by default), so
    # the newest aggregate is always reachable at a stable path.
    latest = os.path.join(common.RESULTS_DIR, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(args.out), latest)
    except OSError:
        pass  # a non-default --out on a filesystem without symlinks: skip

    path = os.path.join(args.out, "results.xlsx")
    ldc.write_aggregate(path, sheets)
    print(
        f"\nwrote {os.path.abspath(path)}  "
        f"({len(sheets)} sheets: {', '.join(s.name for s in sheets)})"
    )


if __name__ == "__main__":
    main()
