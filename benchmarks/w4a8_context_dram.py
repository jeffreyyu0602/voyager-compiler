"""W4A8 DRAM traffic of Llama 3.1 8B vs. context length, 128 to 64k.

``runner.py``'s context axis is pinned to the W8A8 baseline and stops at
8192, and its ``weight_bits`` / ``act_bits`` are sweep axes rather than user
flags.  This driver runs the same compile + estimate pipeline
(``common.run_points_parallel`` -> ``estimate_schedule``) over the baseline
accelerator (64x64 PE, 2 MB SRAM, 64 GB/s DRAM) at W4A8 -- MXFP4 weights,
MXINT8 activations, a BF16 KV cache -- for every power-of-two context length
from 128 to 65536, in prefill and decode.

The output is a ``results.xlsx`` in the runner's format (one ``context``
sheet), so ``plot_results_mpl.py`` draws the latency + DRAM chart and the
Weight / Activation / KV-cache breakdown unmodified, and a ``table.md``
beside it with the activation and KV-cache shares of total DRAM traffic:

    python benchmarks/w4a8_context_dram.py --fast --jobs 2 \\
        --threads-per-job 16 --out benchmarks/results/w4a8_ctx
    python benchmarks/plot_results_mpl.py \\
        --workbook benchmarks/results/w4a8_ctx/results.xlsx \\
        --out benchmarks/figures/w4a8_context

The long points are expensive, so a sweep can be split across runs:
``--lengths`` / ``--modes`` restrict this run, and ``--merge`` carries the
points of an earlier run's workbook that this one does not recompute.

``--calibration-form`` also writes the RTL calibration form for the run's
kernels (what to simulate, and where to enter the measured cycles); the
filled-in form is passed back with ``--calibration`` to price the compute
ops by the measurements.
"""

import argparse
import glob
import json
import os
from dataclasses import replace

import common
import latency_dram_chart as ldc
from openpyxl import load_workbook
from runner import _metrics_dict
from voyager_compiler.codegen.reporting import (
    KernelRow,
    write_calibration_form,
)

CONTEXT_LENGTHS = [2**k for k in range(7, 17)]  # 128 .. 65536
MODES = ("prefill", "decode")
WEIGHT_BITS = 4
ACT_BITS = 8

SHEET = "context"
PREFIX = "Llama 3.1 8B W4A8"
AXIS_TITLE = "Context Length"

CYCLES_PER_SECOND = 1e9
BYTES_PER_GB = 1e9


def build_points(args):
    """Every design point of this run as ``(label, mode, length, cfg)``,
    longest first so the parallel pool starts the slow compiles before the
    quick ones.  The compute-graph SVG is skipped unless ``--dump-graphs``:
    a render that outlives ``common``'s timeout leaves an orphaned ``dot``
    process behind."""
    points = []
    for mode in args.modes:
        for length in sorted(args.lengths, reverse=True):
            cfg = common.config_from_args(
                args,
                mode=mode,
                prompt_len=length,
                kv_len=length,
                weight_bits=WEIGHT_BITS,
                act_bits=ACT_BITS,
            )
            if not args.dump_graphs:
                cfg = replace(cfg, dump_dir=None)
            if args.calibration_form:
                cfg = replace(cfg, calibration_rows_dir=rows_dir(args))
            points.append((f"{SHEET}:{length}:{mode}", mode, length, cfg))
    return points


def rows_dir(args) -> str:
    return os.path.join(args.out, "calibration_rows")


def write_form(args) -> None:
    """Assemble the RTL calibration form from the kernel rows every point of
    this run dumped."""
    by_point = {}
    for path in sorted(glob.glob(os.path.join(rows_dir(args), "*.json"))):
        with open(path) as f:
            dumped = json.load(f)
        rows = by_point.setdefault(dumped["point"], [])
        rows.extend(KernelRow(**r) for r in dumped["rows"])
    write_calibration_form(by_point, args.calibration_form)
    print(
        f"[calibration form] {len(by_point)} points -> "
        f"{args.calibration_form}"
    )


def previous_rows(path):
    """The ``context`` rows of an earlier run's workbook, keyed by
    ``(point, mode)``."""
    ws = load_workbook(path, data_only=True)[SHEET]
    header, *body = ws.values
    rows = {}
    for row in body:
        if row[0] is None and row[1] is None:
            continue  # the blank separator row after the group
        r = dict(zip(header, row))
        rows[(str(r["point"]), r["mode"])] = r
    return rows


def _pct(part, total):
    return f"{100.0 * (part or 0) / total:.1f}" if total else "-"


def share_table(rows):
    """Markdown lines: activation and KV-cache share of total DRAM traffic per
    context length, prefill beside decode."""
    by_key = {(r["point"], r["mode"]): r for r in rows}
    lines = [
        "| Context | Prefill activation % | Prefill KV cache % "
        "| Decode activation % | Decode KV cache % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for length in sorted({int(r["point"]) for r in rows}):
        cells = []
        for mode in MODES:
            r = by_key.get((str(length), mode))
            if r is None:
                cells += ["-", "-"]
                continue
            total = r["dram_total"] or 0
            cells.append(_pct(r["dram_activation"], total))
            cells.append(_pct(r["dram_kv"], total))
        lines.append(f"| {length} | " + " | ".join(cells) + " |")
    return lines


def bytes_table(rows):
    """Markdown lines: every point's DRAM traffic in GB by class, and its
    latency in seconds at 1 GHz."""
    lines = [
        "| Mode | Context | Total GB | Weight GB | Activation GB "
        "| KV cache GB | Latency s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        gb = [
            (r[f] or 0) / BYTES_PER_GB
            for f in ("dram_total", "dram_weight", "dram_activation", "dram_kv")
        ]
        seconds = (r["total_latency"] or 0) / CYCLES_PER_SECOND
        lines.append(
            f"| {r['mode']} | {r['point']} | "
            + " | ".join(f"{v:,.2f}" for v in gb)
            + f" | {seconds:,.3f} |"
        )
    return lines


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The context axis and the precision are fixed by this driver.
    common.add_config_args(
        p,
        exclude={"mode", "prompt_len", "kv_len", "weight_bits", "act_bits"},
    )
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
        help="Context lengths to compile this run (default: 128 .. 65536).",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Graph shapes to compile this run (default: both).",
    )
    p.add_argument(
        "--merge",
        help="An earlier run's results.xlsx; its points that this run does "
        "not recompute are carried into the output workbook and table.",
    )
    p.add_argument(
        "--dump-graphs",
        dest="dump_graphs",
        action="store_true",
        help="Also render each point's compute graph SVG into --out.",
    )
    p.add_argument(
        "--calibration-form",
        dest="calibration_form",
        help="Also write the RTL calibration form for this run's kernels "
        "(one row per distinct kernel signature) to this .xlsx.",
    )
    p.add_argument("--out", default=common.default_out_dir())
    return p.parse_args()


def main():
    args = parse_args()
    points = build_points(args)
    print(f"[w4a8] {len(points)} design points")

    os.makedirs(args.out, exist_ok=True)
    prov = common.write_provenance(args.out)
    print(
        f"[provenance] {prov['short']} "
        f"({'dirty' if prov['dirty'] else 'clean'}) -> "
        f"{os.path.join(args.out, 'provenance.txt')}"
    )
    if prov["dirty"]:
        print(
            "  WARNING: uncommitted changes present; results are not "
            "reproducible from the commit alone (see uncommitted.diff)."
        )

    metrics = common.run_points_parallel(
        [(cfg, label) for label, _, _, cfg in points],
        args.jobs,
        args.threads_per_job,
        args.log_dir or args.out,
        fast=args.fast,
        probe_layers=args.probe_layers,
        tag="w4a8",
    )

    if args.calibration_form:
        write_form(args)

    rows = previous_rows(args.merge) if args.merge else {}
    for (label, mode, length, _), m in zip(points, metrics):
        if m is None:
            print(f"  [skip] {label}: no result")
            continue
        rows[(str(length), mode)] = {
            "group": "",
            "point": str(length),
            "mode": mode,
            **_metrics_dict(m),
        }
    ordered = sorted(
        rows.values(),
        key=lambda r: (MODES.index(r["mode"]), int(r["point"])),
    )

    path = os.path.join(args.out, "results.xlsx")
    ldc.write_aggregate(
        path,
        [
            ldc.MetricSheet(
                SHEET, PREFIX, [ldc.MetricGroup(AXIS_TITLE, ordered)]
            )
        ],
    )
    table = "\n".join(
        [
            f"# {PREFIX} DRAM traffic vs. context length",
            "",
            f"Code: {prov['short']}{' (dirty)' if prov['dirty'] else ''}; "
            + (
                "--fast (layer probes "
                f"{tuple(args.probe_layers)}, extrapolated)"
                if args.fast
                else "full model"
            ),
            "",
            "## Activation and KV-cache share of total DRAM traffic",
            "",
            *share_table(ordered),
            "",
            "## DRAM traffic by class",
            "",
            *bytes_table(ordered),
            "",
        ]
    )
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write(table)
    print("\n" + table)
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
