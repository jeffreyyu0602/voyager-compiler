#!/usr/bin/env python
"""Local pre-push CI for the voyager codegen commands.

Runs every actively-used ``test_codegen.py`` invocation (the ones reached
through the accelerator's ``codegen.mk``) on the *non-bufferized* path,
without dumping tensors, and drops each run's artifacts into a date+time
folder under a user-supplied output location.  If a prior run exists in that
location, the freshly produced ``model.txt`` of each command is compared
against the previous run's and any mismatch (or compile failure) is reported.

Usage (from the repo root, with the conda env active)::

    python test/run_ci.py <output_dir>

Exits non-zero if any command fails to compile or any ``model.txt`` differs
from the previous run, so it can gate a pre-push hook.
"""

import argparse
import difflib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Repo root = parent of this file's directory (test/run_ci.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CODEGEN = REPO_ROOT / "test" / "test_codegen.py"

# Number of unified-diff lines shown in the report for a mismatch.
DIFF_EXCERPT_LINES = 60

# ---------------------------------------------------------------------------
# Command table.
#
# Each Command is expanded at runtime into a test_codegen.py argv:
#     <python> test_codegen.py <model> <SCHEME_ARGS[scheme]>
#         --hardware_unrolling <unrolling> <extra>
#         --model_output_dir <run_dir>/<label>
# The shared per-scheme quantization/compile flags live once in SCHEME_ARGS
# (no --dump_tensors, no --bufferize). To add coverage, add a Command (and a
# new scheme to SCHEME_ARGS if needed).
# ---------------------------------------------------------------------------

# Shared quantization/compile flags per scheme.
SCHEME_ARGS = {
    "E4M3": "--activation fp8_e4m3 --weight fp8_e4m3 --bf16 --transform_layout",
    "P8_1": "--activation posit8_1 --weight posit8_1 --bf16 --transform_layout",
    "INT8": (
        "--activation int8,qs=per_tensor_symmetric "
        "--weight int8,qs=per_tensor_symmetric --bias int24 --bf16 "
        "--calibration_steps 3 --transform_layout"
    ),
    "MXINT8": (
        "--activation int8,qs=microscaling,bs=16 "
        "--weight int8,qs=microscaling,bs=16 --force_scale_power_of_two "
        "--bf16 --transform_layout"
    ),
    "MXNF4": (
        "--activation nf4_6,qs=microscaling,bs=64,scale=fp8_e5m3 "
        "--weight nf4_6,qs=microscaling,bs=64,scale=fp8_e5m3 --bf16 "
        "--residual fp8_e4m3 --quantize_fc --transform_layout "
        "--cache_size 1048576 --num_banks 8 --conv2d_im2col"
    ),
}

# Reused per-command extra-flag groups.
_SINGLE = "--compile_single_layer"
_LLM = "--context_length 1024 --compile_single_layer --quantize_attention_mask"
_LLM_MP = _LLM + " --enable_mixed_precision"
_LLM_SPMM = _LLM_MP + " --outlier_pct 0.01"


@dataclass(frozen=True)
class Command:
    """One codegen invocation, expanded to argv via SCHEME_ARGS."""

    model: str  # test_codegen.py positional argument
    scheme: str  # key into SCHEME_ARGS
    unrolling: str  # --hardware_unrolling value, e.g. "16,16"
    network: str = ""  # output/label name (defaults to model)
    extra: str = ""  # any per-command extra flags


COMMANDS = [
    # -- E4M3 --
    Command("resnet18", "E4M3", "16,16"),
    Command("resnet18", "E4M3", "32,64"),
    Command("resnet18", "E4M3", "4,8"),
    Command("resnet50", "E4M3", "32,64"),
    Command("mobilebert", "E4M3", "16,16", "mobilebert_encoder", _SINGLE),
    Command("mobilebert", "E4M3", "4,8", "mobilebert_encoder", _SINGLE),
    # -- P8_1 --
    Command("resnet18", "P8_1", "16,16"),
    Command("mobilebert", "P8_1", "16,16", "mobilebert_encoder", _SINGLE),
    # -- INT8 --
    Command("resnet18", "INT8", "16,16"),
    Command("mobilebert", "INT8", "16,16", "mobilebert_encoder", _SINGLE),
    # -- MXINT8 --
    Command("resnet18", "MXINT8", "16,16"),
    Command("mobilebert", "MXINT8", "16,16", "mobilebert_encoder", _SINGLE),
    Command("mobilenet_v2", "MXINT8", "16,16"),
    # -- MXNF4 (llama) --
    Command("llm_prefill", "MXNF4", "64,64", "llama_prefill", _LLM),
    Command("llm_prefill", "MXNF4", "64,64", "llama_prefill_mp", _LLM_MP),
    Command("llm_prefill", "MXNF4", "64,64", "llama_prefill_spmm", _LLM_SPMM),
    Command("llm_decode", "MXNF4", "64,64", "llama_decode", _LLM),
    Command("llm_kivi", "MXNF4", "64,64", "llama_decode_kivi", _LLM),
    # -- MXNF4 (vision / bert) --
    Command("resnet18", "MXNF4", "64,64"),
    Command("resnet50", "MXNF4", "64,64"),
    Command("vit", "MXNF4", "64,64"),
    Command("bert", "MXNF4", "64,64"),
]

# Timestamp folder format; lexicographic sort == chronological sort.
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _label(command):
    """``<network>/<scheme>/<unrolling>`` — unique per command (unrolling
    disambiguates commands sharing a network/scheme, e.g. resnet18 E4M3)."""
    network = command.network or command.model
    unroll = command.unrolling.replace(",", "x")
    return f"{network}/{command.scheme}/{unroll}"


assert len({_label(c) for c in COMMANDS}) == len(COMMANDS), (
    "COMMANDS have duplicate <network>/<scheme>/<unrolling> labels; give "
    "colliding commands distinct 'network' names"
)


def _matches(label, pattern):
    """Does ``pattern`` select ``label`` (``network/scheme/unrolling``)?

    Matching is anchored at ``/`` segment boundaries, not raw substring: a
    single token matches a whole segment by prefix (so ``bert`` selects
    ``bert`` but not ``mobilebert_encoder``, while ``mobilebert`` still
    selects ``mobilebert_encoder`` and ``resnet`` selects both resnets); a
    ``a/b`` token matches as an anchored path prefix.
    """
    segs = label.lower().split("/")
    parts = pattern.lower().strip("/").split("/")
    if len(parts) == 1:
        return any(s.startswith(parts[0]) for s in segs)
    return len(parts) <= len(segs) and all(
        segs[i].startswith(parts[i]) for i in range(len(parts))
    )


def _build(command, run_dir):
    """Expand a Command into ``(label, dest, argv)`` for this run.

    The argv runs this repo's test_codegen.py with --model_output_dir pointed
    into the timestamped run folder; --dump_tensors / --bufferize are never
    added.
    """
    label = _label(command)
    dest = run_dir / label

    argv = [sys.executable, str(TEST_CODEGEN), command.model]
    argv += shlex.split(SCHEME_ARGS[command.scheme])
    argv += ["--hardware_unrolling", command.unrolling]
    argv += shlex.split(command.extra)
    argv += ["--model_output_dir", str(dest)]
    return label, dest, argv


def _run_one(label, dest, argv):
    """Run one command; capture combined output to ``dest/run.log``.

    Returns a status string: ``ok`` / ``error`` (nonzero exit) /
    ``no_output`` (exited 0 but no model.txt).
    """
    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "run.log"
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(shlex.quote(a) for a in argv) + "\n\n")
        log.flush()
        proc = subprocess.run(
            argv, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT
        )

    if proc.returncode != 0:
        return "error"
    if not (dest / "model.txt").exists():
        return "no_output"
    return "ok"


def _find_previous_run(out_dir, current):
    """Newest timestamp dir under ``out_dir`` that isn't ``current``."""
    runs = sorted(
        p.name
        for p in out_dir.iterdir()
        if p.is_dir() and TS_RE.match(p.name) and p.name != current
    )
    return out_dir / runs[-1] if runs else None


def _compare(label, status, run_dir, prev_run):
    """Classify one label's model.txt vs the previous run.

    Returns ``(verdict, diff_excerpt)``.  ``diff_excerpt`` is non-empty only
    for a MISMATCH.
    """
    if status != "ok":
        return "FAILED", ""

    cur = (run_dir / label / "model.txt").read_text()

    if prev_run is None:
        return "NEW", ""
    prev_path = prev_run / label / "model.txt"
    if not prev_path.exists():
        return "NEW", ""

    prev = prev_path.read_text()
    if cur == prev:
        return "MATCH", ""

    diff = difflib.unified_diff(
        prev.splitlines(),
        cur.splitlines(),
        fromfile=f"{prev_run.name}/{label}/model.txt",
        tofile=f"{run_dir.name}/{label}/model.txt",
        lineterm="",
    )
    excerpt = list(diff)[:DIFF_EXCERPT_LINES]
    return "MISMATCH", "\n".join(excerpt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Location for dated CI artifacts (created if missing).",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SUBSTR",
        help="Run only commands whose <network>/<scheme>/<unrolling> label "
        "contains SUBSTR (case-insensitive; repeatable).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List command labels and exit.",
    )
    args = parser.parse_args()

    commands = COMMANDS
    if args.only:
        commands = [
            c
            for c in COMMANDS
            if any(_matches(_label(c), p) for p in args.only)
        ]

    if args.list:
        for command in commands:
            _, _, argv = _build(command, Path("<run_dir>"))
            print(" ".join(shlex.quote(a) for a in argv))
        return 0

    if not args.output_dir:
        parser.error("output_dir is required (unless --list)")
    if not commands:
        parser.error(f"--only {args.only} matched no commands")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(TS_FORMAT)
    run_dir = out_dir / timestamp
    run_dir.mkdir()

    prev_run = _find_previous_run(out_dir, timestamp)

    print(f"voyager codegen CI -> {run_dir}")
    if prev_run is not None:
        print(f"comparing model.txt against previous run: {prev_run.name}")
    else:
        print("no previous run found; this run is the baseline")
    print(f"running {len(commands)} commands\n")

    results = []  # (label, status, verdict, diff_excerpt)
    for idx, command in enumerate(commands, 1):
        label, dest, argv = _build(command, run_dir)
        print(f"[{idx}/{len(commands)}] {label} ... ", end="", flush=True)
        status = _run_one(label, dest, argv)
        verdict, excerpt = _compare(label, status, run_dir, prev_run)
        results.append((label, status, verdict, excerpt))
        suffix = "" if status == "ok" else f" ({status})"
        print(f"{verdict}{suffix}")

    report = _build_report(results, run_dir, prev_run)
    (run_dir / "report.txt").write_text(report)
    print("\n" + report)

    # Gate: fail on any compile failure or model.txt mismatch.
    bad = [r for r in results if r[2] in ("FAILED", "MISMATCH")]
    return 1 if bad else 0


def _build_report(results, run_dir, prev_run):
    """Render the human-readable summary written to report.txt + stdout."""
    counts = {}
    for _, _, verdict, _ in results:
        counts[verdict] = counts.get(verdict, 0) + 1

    lines = []
    lines.append("=" * 70)
    lines.append(f"voyager codegen CI report  ({run_dir.name})")
    if prev_run is not None:
        lines.append(f"compared against: {prev_run.name}")
    else:
        lines.append("compared against: (none - baseline run)")
    lines.append("=" * 70)

    order = ["MATCH", "NEW", "MISMATCH", "FAILED", "MISSING"]
    summary = "  ".join(f"{v}={counts[v]}" for v in order if v in counts)
    lines.append(f"totals: {summary}")
    lines.append("")

    # Detail any non-clean verdicts.
    for label, status, verdict, excerpt in results:
        if verdict in ("MATCH", "NEW"):
            continue
        detail = f"  - {verdict}: {label}"
        if status != "ok":
            detail += f" [{status}; see {label}/run.log]"
        lines.append(detail)
        if excerpt:
            for dl in excerpt.splitlines():
                lines.append(f"      {dl}")

    # Labels present in the previous run but absent now.
    if prev_run is not None:
        current_labels = {r[0] for r in results}
        for p in sorted(prev_run.rglob("model.txt")):
            label = str(p.parent.relative_to(prev_run))
            if label not in current_labels:
                lines.append(f"  - MISSING: {label} (in {prev_run.name})")

    bad = [r for r in results if r[2] in ("FAILED", "MISMATCH")]
    lines.append("")
    lines.append("RESULT: FAIL" if bad else "RESULT: PASS")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
