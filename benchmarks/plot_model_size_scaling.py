"""Plot the model-size sweep against parameter count, not the model's name.

``plot_results_mpl`` draws the sweep the way the workbook does: five bars over
the labels 1B / 3B / 7B / 8B / 13B, evenly spaced, which says nothing about how
far apart those designs actually are.  This plots the same two metrics --
latency and DRAM traffic -- against each model's parameter count instead, as a
line chart with prefill stacked over decode.

The parameter count is derived analytically from the HF config; ``--verify``
compares it against the ``num_params`` the sweep recorded, and they must agree.

    python plot_model_size_scaling.py --verify
    python plot_model_size_scaling.py --out tmp
"""

import argparse
import functools
import os

import numpy as np
import plot_results_mpl as style
from openpyxl import load_workbook
from transformers import AutoConfig

SHEET = "model_size"
# Mirrors benchmarks/runner.py's MODEL_SIZES: the sheet stores only the label.
MODEL_IDS = {
    "1B": "meta-llama/Llama-3.2-1B",
    "3B": "meta-llama/Llama-3.2-3B",
    "7B": "meta-llama/Llama-2-7b-hf",
    "8B": "meta-llama/Llama-3.1-8B",
    "13B": "meta-llama/Llama-2-13b-hf",
}

BILLION = 1e9
# Marker diameter in points; a label above a marker starts past its radius.
MARKER_PT = 9


def _dims(cfg):
    """``(layers, hidden, ffn, q_dim, kv_dim, vocab, tied)`` from an HF config.
    ``head_dim`` is explicit on the Llama-3.2 configs and implied elsewhere."""
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    return (
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads * head_dim,
        cfg.num_key_value_heads * head_dim,
        cfg.vocab_size,
        bool(getattr(cfg, "tie_word_embeddings", False)),
    )


def count_params(cfg):
    """The model's parameters, counted off its config.  Tied embeddings are
    counted once -- ``torch``'s ``parameters()`` yields the shared tensor once,
    so the sweep's ``num_params`` counts it once too."""
    layers, h, ffn, q_dim, kv_dim, vocab, tied = _dims(cfg)
    attn = h * q_dim + 2 * (h * kv_dim) + q_dim * h
    mlp = 3 * (h * ffn)
    norms = 2 * h  # input_layernorm + post_attention_layernorm
    embed = vocab * h
    head = 0 if tied else vocab * h
    return layers * (attn + mlp + norms) + embed + head + h


def sweep_rows(workbook):
    """The model_size sheet's rows, in the label order of ``MODEL_IDS``."""
    ws = load_workbook(workbook)[SHEET]
    rows = style._rows(ws)
    order = list(MODEL_IDS)
    return sorted(rows, key=lambda r: order.index(r["point"]))


def model_table(rows):
    """Per design point: its derived parameter count beside the sweep's own
    ``num_params``.  One entry per label, both modes sharing a model."""
    table = {}
    for label, model_id in MODEL_IDS.items():
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
        reported = next(
            (r["num_params"] for r in rows if r["point"] == label), None
        )
        table[label] = {
            "model_id": model_id,
            "params": count_params(cfg),
            "reported_params": reported,
        }
    return table


def verify(table):
    """Print the derived parameter count against the sweep's, and say whether
    every model agrees."""
    print(f"{'model':10} {'derived params':>16} {'sweep params':>16}  match")
    ok = True
    for label, e in table.items():
        agree = e["params"] == e["reported_params"]
        ok = ok and agree
        print(
            f"{label:10} {e['params']:>16,} {e['reported_params']:>16,}"
            f"  {'yes' if agree else 'NO'}"
        )
    print("\nall parameter counts agree" if ok else "\nMISMATCH")
    return ok


def _annotate(ax, point, text, rise, color):
    """``text`` centered ``rise`` points above ``point`` (below, when
    negative), in the series' color with a white halo."""
    ax.annotate(
        text,
        point,
        textcoords="offset points",
        xytext=(0, rise),
        ha="center",
        va="bottom" if rise >= 0 else "top",
        fontsize=style.VALUE_PT,
        fontweight="bold",
        color=color,
        path_effects=style.HALO,
    )


def _draw_axis(fig, ax, points, xlabel, title, logx):
    """Draw a dual-axis line chart onto ``ax`` (latency left, DRAM right) over
    numeric ``points`` -- ``(x, latency_s, dram_gb, label)``.  Every marker is
    labeled with its value, and each latency marker named for its model.
    Returns ``(handles, names)`` for a shared legend; the caller owns the
    figure, so this neither legends nor saves.  ``fig`` is unused: it is the
    ``draw_pair`` callback signature."""
    x = np.array([p[0] for p in points], dtype=float)
    seconds = np.array([p[1] for p in points], dtype=float)
    gigabytes = np.array([p[2] for p in points], dtype=float)
    names = [p[3] for p in points]

    style._grid(ax)
    lat = ax.plot(
        x,
        seconds,
        color=style.COLORS[0],
        linewidth=2.5,
        marker="o",
        markersize=MARKER_PT,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=3,
    )[0]
    ax.set_ylabel("Latency (s)", fontsize=style.HEADER_PT, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=style.HEADER_PT, fontweight="bold")
    if logx:
        ax.set_xscale("log")
    style._limits(ax, list(seconds), False, headroom=style.LATENCY_HEADROOM)

    right = ax.twinx()
    dram = right.plot(
        x,
        gigabytes,
        color=style.COLORS[1],
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=MARKER_PT,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=4,
    )[0]
    right.set_ylabel("DRAM (GB)", fontsize=style.HEADER_PT, fontweight="bold")
    style._limits(right, list(gigabytes), False, headroom=style.DRAM_HEADROOM)

    for axis in (ax, right):
        axis.tick_params(axis="both", labelsize=style.HEADER_PT)
        for lbl in axis.get_yticklabels() + axis.get_xticklabels():
            lbl.set_fontweight("bold")

    # Value labels sit outside the pair of lines -- latency below its marker,
    # DRAM above -- so neither series' markers run through the other's digits
    # (the DRAM line runs above the latency line by the two axes' headroom).
    # Under each latency value goes the model's name: the axis gives the
    # size, but which model sits there is the reason to read the chart.
    clear = style.LABEL_PAD + MARKER_PT / 2
    for xi, sec, gb, name in zip(x, seconds, gigabytes, names):
        _annotate(ax, (xi, sec), style._fmt(sec), -clear, style.COLORS[0])
        _annotate(
            ax,
            (xi, sec),
            name,
            -(clear + style.LINE_HEIGHT * style.VALUE_PT),
            style.COLORS[0],
        )
        _annotate(right, (xi, gb), style._fmt(gb), clear, style.COLORS[1])

    ax.set_title(title, fontsize=style.HEADER_PT, fontweight="bold")
    return [lat, dram], ["Latency", "DRAM Traffic"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workbook", default=style.DEFAULT_WORKBOOK)
    p.add_argument("--out", default=style.DEFAULT_OUT)
    p.add_argument(
        "--verify",
        action="store_true",
        help="Check the derived parameter counts against the sweep's.",
    )
    p.add_argument(
        "--logx", action="store_true", help="Log-scale the size axis."
    )
    return p.parse_args()


def _points(rows, table, mode):
    """The ``(params_B, latency_s, dram_gb, label)`` points for ``mode``, x =
    each model's parameter count in billions."""
    return sorted(
        (
            table[r["point"]]["params"] / BILLION,
            r["total_latency"] / style.CYCLES_PER_SECOND,
            r["dram_total"] / style.BYTES_PER_GB,
            r["point"],
        )
        for r in rows
        if r["mode"] == mode
    )


def main():
    args = parse_args()
    rows = sweep_rows(args.workbook)
    table = model_table(rows)
    if args.verify:
        verify(table)
        print()

    os.makedirs(args.out, exist_ok=True)

    # Parameter scaling: prefill over decode, both vs #params, one shared
    # x-axis.
    written = style.draw_pair(
        [
            functools.partial(
                _draw_axis,
                points=_points(rows, table, mode),
                xlabel="Parameters (B)",
                title=mode.capitalize(),
                logx=args.logx,
            )
            for mode in style.MODES
        ],
        "Latency and DRAM Traffic vs. Parameters",
        style.FIGSIZE,
        args.out,
        "params_scaling",
    )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
