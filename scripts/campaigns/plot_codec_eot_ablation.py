"""Bar chart for the codec-EoT ablation (S3a, Qwen2-Audio, eps=1.0, n=40).

Mirrors tab:non_robust: ASR (%) per channel, codec-robust EoT vs no-EoT.
Writes paper-neurips/figures/figure_codec_eot_ablation.{pdf,png}.

Numbers are taken directly from the table (the 'with-EoT' value is recovered
as without-EoT minus the parenthetical drop, sign-corrected).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "paper-neurips" / "figures"

CHANNELS = [
    ("Clean",       100.0,  97.5),
    ("Opus 16k",      0.0,  17.5),
    ("Opus 24k",      0.0,  40.0),
    ("Opus 32k",      0.0,  60.0),
    ("Opus 64k",     67.5,  87.5),
    ("MP3 64k",      72.5,  90.0),
    ("AAC-LC 64k",   15.0,  47.5),
]

C_EOT      = "#1f77b4"
C_NO_EOT   = "#d62728"


def main() -> None:
    labels   = [c[0] for c in CHANNELS]
    no_eot   = np.array([c[1] for c in CHANNELS])
    with_eot = np.array([c[2] for c in CHANNELS])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(labels))
    w = 0.38

    b1 = ax.bar(x - w / 2, with_eot, w, label="Codec-robust EoT",
                color=C_EOT, edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, no_eot,   w, label="No EoT",
                color=C_NO_EOT, edgecolor="black", linewidth=0.4)

    for xi, (h_eot, h_no) in enumerate(zip(with_eot, no_eot)):
        drop = h_no - h_eot
        sign = "+" if drop >= 0 else "$-$"
        ax.annotate(f"{sign}{abs(drop):.1f}",
                    xy=(xi + w / 2, h_no),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6, color="black")

    # Light separator between Opus block and the rest.
    ax.axvline(4.5, color="grey", linestyle=":", linewidth=0.6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 109)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title("Codec-EoT ablation  (S3a, Qwen2-Audio, $\\epsilon{=}1.0$, $n{=}40$)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"figure_codec_eot_ablation.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
