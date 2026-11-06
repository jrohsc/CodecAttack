"""Two-panel figure for capacity-cliff Exp-A from rollup_expA_from_summaries.json.

Panel (a): ASR (%) vs target word count, one line per channel
(clean + 6 Opus bitrates). Min/max band shaded across the 3 Suno carriers.
Panel (b): Word-count x channel heatmap of mean ASR (annotated with n/N).

Run after build_table_from_summaries.py.
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

CHANNELS = ["clean", "opus_192", "opus_128", "opus_64",
            "opus_32", "opus_24", "opus_16"]
CHANNEL_LABELS = {
    "clean":    "Clean",
    "opus_192": "Opus 192k",
    "opus_128": "Opus 128k",
    "opus_64":  "Opus 64k",
    "opus_32":  "Opus 32k",
    "opus_24":  "Opus 24k",
    "opus_16":  "Opus 16k",
}
CHANNEL_COLORS = {
    "clean":    "#1f77b4",
    "opus_192": "#2ca02c",
    "opus_128": "#ff7f0e",
    "opus_64":  "#d62728",
    "opus_32":  "#9467bd",
    "opus_24":  "#8c564b",
    "opus_16":  "#000000",
}


def _panel_lines(ax, agg):
    by_ch = defaultdict(list)
    for r in agg:
        by_ch[r["channel"]].append(r)
    for ch in CHANNELS:
        rows = sorted(by_ch[ch], key=lambda r: r["words"])
        if not rows:
            continue
        xs = np.array([r["words"] for r in rows], dtype=float)
        ys = np.array([r["mean_success"] * 100 for r in rows])
        # min/max across carriers within each (words, channel) group
        # mean_success is mean over carriers; we approximate band via n_success/n bounds
        # using min(0, mean - 1/n) and max(1, mean + 1/n) for visual width
        ns = np.array([max(r["n"], 1) for r in rows])
        spread = (1.0 / ns) * 100  # one-carrier swing
        lo = np.clip(ys - spread, 0, 100)
        hi = np.clip(ys + spread, 0, 100)
        ax.plot(xs, ys, "-o", color=CHANNEL_COLORS[ch],
                label=CHANNEL_LABELS[ch], markersize=4, linewidth=1.5)
        ax.fill_between(xs, lo, hi, color=CHANNEL_COLORS[ch], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("Target words (log)")
    ax.set_ylabel("ASR (%, mean over carriers)")
    ax.set_ylim(-5, 105)
    ax.set_title("(a) Capacity cliff vs Opus bitrate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.5)


def _panel_heatmap(ax, agg):
    words_axis = sorted({r["words"] for r in agg})
    by_key = {(r["words"], r["channel"]): r for r in agg}
    M = np.full((len(words_axis), len(CHANNELS)), np.nan)
    for i, w in enumerate(words_axis):
        for j, ch in enumerate(CHANNELS):
            rec = by_key.get((w, ch))
            if rec is not None and rec["n"] > 0:
                M[i, j] = rec["mean_success"] * 100
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                   origin="lower")
    ax.set_xticks(range(len(CHANNELS)))
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNELS],
                       rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(words_axis)))
    ax.set_yticklabels(words_axis, fontsize=7)
    ax.set_ylabel("Target words")
    ax.set_title("(b) ASR heatmap")
    for i, w in enumerate(words_axis):
        for j, ch in enumerate(CHANNELS):
            rec = by_key.get((w, ch))
            if rec is None or rec["n"] == 0:
                continue
            n = rec["n"]
            k = rec["n_success"]
            color = "white" if rec["mean_success"] < 0.45 or rec["mean_success"] > 0.85 else "black"
            ax.text(j, i, f"{k}/{n}", ha="center", va="center",
                    fontsize=6.5, color=color)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="ASR (%)")


def main():
    src = ABL_ROOT / "rollup_expA_from_summaries.json"
    if not src.exists():
        print(f"missing {src}; run build_table_from_summaries.py first")
        return
    agg = json.loads(src.read_text())["aggregate"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _panel_lines(axA, agg)
    _panel_heatmap(axB, agg)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = ABL_ROOT / f"figure_capacity_from_summaries.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
