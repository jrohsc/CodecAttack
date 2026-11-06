"""Regenerate paper-neurips/tables/tab_watermark.tex from results_defense_watermark/.

Each ASR / WM-destroyed cell is rendered as $p_{\\pm h}$ (point estimate +
95% Wilson half-width, pp). The Adv Det column is the mean detection score
on adversarial audio (continuous, no per-trial scores in the summary, so
no CI is reported here).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # codec_attack/
RESULTS = ROOT / "0_all_combined" / "results_defense_watermark"
OUT = ROOT / "0_all_combined" / "paper-neurips" / "tables" / "tab_watermark.tex"

ROWS = [
    ("Qwen2-Audio",      "s1_qwen2_audio_eps1.0"),
    ("Qwen2.5-Omni",     "s1_qwen25_omni_eps1.0"),
    ("Audio Flamingo 3", "s1_audio_flamingo_eps1.0"),
]

Z_95 = 1.959963984540054


def wilson_half(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + Z_95 * Z_95 / n
    half = Z_95 * math.sqrt(p * (1.0 - p) / n + Z_95 * Z_95 / (4.0 * n * n)) / denom
    center = (p + Z_95 * Z_95 / (2.0 * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * (hi - lo) / 2.0


def cell(k: int, n: int) -> str:
    return f"${100.0 * k / n:.1f}_{{\\pm {wilson_half(k, n):.0f}}}$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    n_seen = set()
    body_lines = []
    for label, run_dir in ROWS:
        d = json.loads((RESULTS / run_dir / "summary.json").read_text())
        n = int(d["n_total"])
        n_seen.add(n)
        asr = cell(int(d["attack_success"]), n)
        wm  = cell(int(d["watermark_destroyed"]), n)
        adv = f"{d['avg_adv_detection_score']:.3f}"
        body_lines.append(f"{label} & {asr} & {wm} & {adv} \\\\")

    n_str = str(next(iter(n_seen))) if len(n_seen) == 1 else "varies"

    tex = (
        r"\begin{wraptable}{r}{0.48\textwidth}" "\n"
        r"\centering" "\n"
        r"% \vspace{-5pt}" "\n"
        r"\caption{AudioSeal watermark survival under \texttt{CodecAttack} on S1 "
        r"(English banking carriers, $\epsilon{=}1.0$, $n{=}" + n_str + r"$ pairs per model). "
        r"ASR and WM Dest.\ cells are $p_{\pm h}$ with $h$ the 95\% Wilson half-width (pp); "
        r"Adv Det.\ is the mean AudioSeal score on adversarial audio (clean baseline "
        r"$\approx$0.795).}" "\n"
        r"\vspace{5pt}" "\n"
        r"\label{tab:defense_watermark}" "\n"
        r"\resizebox{0.5\textwidth}{!}{" "\n"
        r"\begin{tabular}{lrrr}" "\n"
        r"\toprule" "\n"
        r"\textbf{Model} & \textbf{ASR} & \textbf{WM Dest.} & \textbf{Adv Det.} \\" "\n"
        r"\midrule" "\n"
        + "\n".join(body_lines) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}}" "\n"
        r"\vspace{-5pt}" "\n"
        r"\end{wraptable}" "\n"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
