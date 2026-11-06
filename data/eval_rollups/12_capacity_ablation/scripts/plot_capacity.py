"""Two-panel NeurIPS figure for capacity ablation.

Panel (a): Exp-A — ASR vs words, one line per channel, shaded min/max band.
Panel (b): Exp-B — ASR vs duration, mean over 3 anchor lengths per channel.

Writes figure_capacity.pdf + figure_capacity.png.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent

CHANNELS = ["clean", "opus_192k", "opus_128k", "opus_64k",
            "aac_128k", "mp3_192k"]
CHANNEL_LABELS = {"clean": "Clean", "opus_192k": "Opus 192k",
                  "opus_128k": "Opus 128k", "opus_64k": "Opus 64k",
                  "aac_128k": "AAC 128k", "mp3_192k": "MP3 192k"}
CHANNEL_COLORS = {"clean": "#1f77b4", "opus_192k": "#2ca02c",
                  "opus_128k": "#ff7f0e", "opus_64k": "#d62728",
                  "aac_128k": "#9467bd", "mp3_192k": "#8c564b"}


def _crossover(xs, ys, target_y):
    for i in range(len(xs) - 1):
        if (ys[i] - target_y) * (ys[i + 1] - target_y) <= 0 and ys[i] != ys[i + 1]:
            t = (target_y - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def _panel_expA(ax, agg):
    by_ch = defaultdict(list)
    for r in agg:
        by_ch[r["channel"]].append(r)
    for ch in CHANNELS:
        rows = sorted(by_ch[ch], key=lambda r: r["words"])
        if not rows:
            continue
        xs = np.array([r["words"] for r in rows], dtype=float)
        ys = np.array([r["mean_success"] * 100 for r in rows])
        lo = np.array([r["min_success"] * 100 for r in rows])
        hi = np.array([r["max_success"] * 100 for r in rows])
        ax.plot(xs, ys, "-o", color=CHANNEL_COLORS[ch],
                label=CHANNEL_LABELS[ch], markersize=4, linewidth=1.5)
        ax.fill_between(xs, lo, hi, color=CHANNEL_COLORS[ch], alpha=0.15)
    for ch_key in ("opus_64k", "opus_128k"):
        rows = sorted(by_ch[ch_key], key=lambda r: r["words"])
        xs = [r["words"] for r in rows]
        ys = [r["mean_success"] * 100 for r in rows]
        x50 = _crossover(xs, ys, 50.0)
        if x50 is not None:
            ax.axvline(x50, color=CHANNEL_COLORS[ch_key],
                       linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(x50, 55, f"{x50:.0f}w", rotation=90,
                    color=CHANNEL_COLORS[ch_key], fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Target words (log)")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(a) Words vs ASR  (carrier $\\leq$25 s)")
    ax.grid(True, alpha=0.3)


def _panel_expB(ax, agg):
    groups = defaultdict(list)
    for r in agg:
        groups[(r["channel"], r["duration_s"])].append(r)
    by_ch_dur = {}
    for (ch, dur), rs in groups.items():
        by_ch_dur[(ch, dur)] = {
            "mean": np.mean([r["mean_success"] for r in rs]) * 100,
            "min": np.min([r["mean_success"] for r in rs]) * 100,
            "max": np.max([r["mean_success"] for r in rs]) * 100,
        }
    for ch in CHANNELS:
        durs = sorted({d for (c, d) in by_ch_dur if c == ch})
        if not durs:
            continue
        xs = np.array(durs, dtype=float)
        ys = np.array([by_ch_dur[(ch, d)]["mean"] for d in durs])
        lo = np.array([by_ch_dur[(ch, d)]["min"] for d in durs])
        hi = np.array([by_ch_dur[(ch, d)]["max"] for d in durs])
        ax.plot(xs, ys, "-o", color=CHANNEL_COLORS[ch],
                label=CHANNEL_LABELS[ch], markersize=4, linewidth=1.5)
        ax.fill_between(xs, lo, hi, color=CHANNEL_COLORS[ch], alpha=0.15)
    ax.set_xlabel("Carrier duration (s)")
    ax.set_ylabel("ASR (%, mean over 3 anchors)")
    ax.set_ylim(-5, 105)
    ax.set_title("(b) Duration vs ASR")
    ax.grid(True, alpha=0.3)


def main():
    expA_path = ABL_ROOT / "rollup_expA.json"
    expB_path = ABL_ROOT / "rollup_expB.json"
    expA = json.loads(expA_path.read_text())["aggregate"] if expA_path.exists() else []
    expB = json.loads(expB_path.read_text())["aggregate"] if expB_path.exists() else []
    if not expA and not expB:
        print("no rollups found; run build_rollups.py first")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 3.2))
    _panel_expA(axA, expA)
    _panel_expB(axB, expB)
    axA.legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ABL_ROOT / f"figure_capacity.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote figure_capacity.{pdf,png}")


if __name__ == "__main__":
    main()
