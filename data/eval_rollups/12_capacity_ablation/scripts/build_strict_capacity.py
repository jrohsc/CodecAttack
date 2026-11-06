"""Paper-metric rebuild of capacity-cliff Exp-A from per-cell summaries.

Uses the paper's primary success rule (sections/experiments.tex Setup):
  success = (target appears as substring of normalized output)  OR  (WER <= 0.3)
Drops the buggy hardcoded "human-authored" substring branch in
build_table_from_summaries.py.

Outputs:
  - rollup_expA_strict.json        (rows + aggregate)
  - figure_capacity_strict.{pdf,png}  (line plot + heatmap)
  - table_capacity_expA_strict.tex
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent
EXP_A_ROOT = ABL_ROOT / "expA_words/qwen2_audio_eps1.0_multibitrate"
TARGETS = json.loads((ABL_ROOT / "targets_capacity.json").read_text())
WORDS_BY_ID = {t["id"]: t["words"] for t in TARGETS["targets"]}

OPUS_BITRATES = [192, 128, 64, 32, 24, 16]
COLUMNS = ["clean"] + [f"opus_{b}" for b in OPUS_BITRATES]
COLUMN_LABELS = {
    "clean": "Clean",
    "opus_192": "Opus192",
    "opus_128": "Opus128",
    "opus_64":  "Opus64",
    "opus_32":  "Opus32",
    "opus_24":  "Opus24",
    "opus_16":  "Opus16",
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


def _normalize(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _paper_success(target: str, output: str, wer: float | None) -> bool:
    """Paper's primary metric: target-substring(normalized) OR WER <= 0.3."""
    nt = _normalize(target)
    no = _normalize(output)
    if nt and nt in no:
        return True
    if wer is not None and wer <= 0.3:
        return True
    return False


def _walk():
    rows = []
    for sm in sorted(EXP_A_ROOT.rglob("*_summary.json")):
        carrier = sm.parent.parent.name
        target_id = sm.parent.name
        if target_id not in WORDS_BY_ID:
            continue
        words = WORDS_BY_ID[target_id]
        s = json.loads(sm.read_text())
        target_text = s.get("target_text", "")
        final_output = s.get("final_output", "")

        # Clean: substring(target, output) OR top-level success flag.
        # The clean decode has no per-cell WER recorded, so we use
        # substring + the optimization-side success boolean.
        clean_succ = _paper_success(target_text, final_output, None) or \
                     bool(s.get("success"))
        rows.append({
            "carrier": carrier, "words": words,
            "channel": "clean",
            "success": int(clean_succ),
            "wer": -1.0,
            "snr_db": -1.0,
            "final_loss": s.get("final_loss", -1.0),
        })
        # Per-bitrate: paper's substring-or-WER<=0.3 rule.
        for b in OPUS_BITRATES:
            rec = (s.get("opus_robustness") or {}).get(str(b))
            if rec is None:
                continue
            rows.append({
                "carrier": carrier, "words": words,
                "channel": f"opus_{b}",
                "success": int(_paper_success(
                    target_text, rec.get("output", ""), rec.get("wer", 1.0))),
                "wer": rec.get("wer", -1.0),
                "snr_db": rec.get("snr_db", -1.0),
                "final_loss": s.get("final_loss", -1.0),
            })
    return rows


def _aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["words"], r["channel"])].append(r)
    return {
        key: {
            "n": len(rs),
            "n_success": sum(r["success"] for r in rs),
            "mean_success": statistics.mean(r["success"] for r in rs),
            "mean_final_loss": statistics.mean(
                r["final_loss"] for r in rs if r["final_loss"] >= 0
            ) if any(r["final_loss"] >= 0 for r in rs) else float("nan"),
        }
        for key, rs in grouped.items()
    }


def _cell(rec):
    if rec is None or rec["n"] == 0:
        return "--"
    return f"{rec['n_success']}/{rec['n']}"


def _build_tex(agg, words_axis):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Adversarial channel capacity vs.\ target word count "
        r"and Opus bitrate. Each cell is the number of AI-generated "
        r"carriers (jazz / classic / calm; $n{=}3$) for which the "
        r"attack output meets the paper's primary success rule "
        r"(\S\ref{sec:experimental_setup}). Setup: Qwen2-Audio, "
        r"$\epsilon{=}1.0$, 1000 PGD steps, multi-bitrate Opus EoT, "
        r"25\,s carriers. \textbf{Loss} is mean clean-side PGD loss.}",
        r"\label{tab:capacity_strict}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{r" + "c" * len(COLUMNS) + r"c}",
        r"\toprule",
        r"\textbf{Words} & " +
        " & ".join(r"\textbf{" + COLUMN_LABELS[c] + "}" for c in COLUMNS) +
        r" & \textbf{Loss} \\",
        r"\midrule",
    ]
    for w in words_axis:
        cells = [f"{w}"]
        loss = agg.get((w, "clean"), {}).get("mean_final_loss", float("nan"))
        for ch in COLUMNS:
            cells.append(_cell(agg.get((w, ch))))
        cells.append(f"{loss:.3f}" if loss == loss else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _panel_lines(ax, agg):
    by_ch = defaultdict(list)
    for (w, ch), rec in agg.items():
        by_ch[ch].append((w, rec))
    for ch in COLUMNS:
        rows = sorted(by_ch[ch])
        if not rows:
            continue
        xs = np.array([w for w, _ in rows], dtype=float)
        ys = np.array([rec["mean_success"] * 100 for _, rec in rows])
        ax.plot(xs, ys, "-o", color=CHANNEL_COLORS[ch],
                label=COLUMN_LABELS[ch], markersize=4, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Target words (log)")
    ax.set_ylabel("ASR (%, mean over carriers)")
    ax.set_ylim(-5, 105)
    ax.set_title("(a) Capacity vs.\\ Opus bitrate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.5)


def _panel_heatmap(ax, agg):
    words_axis = sorted({w for (w, _) in agg})
    M = np.full((len(words_axis), len(COLUMNS)), np.nan)
    for i, w in enumerate(words_axis):
        for j, ch in enumerate(COLUMNS):
            rec = agg.get((w, ch))
            if rec and rec["n"] > 0:
                M[i, j] = rec["mean_success"] * 100
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                   origin="lower")
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([COLUMN_LABELS[c] for c in COLUMNS],
                       rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(words_axis)))
    ax.set_yticklabels(words_axis, fontsize=7)
    ax.set_ylabel("Target words")
    ax.set_title("Per-cell success counts (paper metric, $n{=}3$ carriers)")
    for i, w in enumerate(words_axis):
        for j, ch in enumerate(COLUMNS):
            rec = agg.get((w, ch))
            if rec is None or rec["n"] == 0:
                continue
            color = ("white" if rec["mean_success"] < 0.45
                     or rec["mean_success"] > 0.85 else "black")
            ax.text(j, i, f"{rec['n_success']}/{rec['n']}",
                    ha="center", va="center", fontsize=6.5, color=color)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="ASR (%)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=None,
                    help="Truncate to word counts <= this (drops partial rows)")
    args = ap.parse_args()

    rows = _walk()
    agg = _aggregate(rows)
    words_axis = sorted({t["words"] for t in TARGETS["targets"]})
    if args.max_words is not None:
        words_axis = [w for w in words_axis if w <= args.max_words]
        agg = {(w, ch): rec for (w, ch), rec in agg.items()
               if w in set(words_axis)}
        rows = [r for r in rows if r["words"] in set(words_axis)]

    out_json = ABL_ROOT / "rollup_expA_strict.json"
    out_json.write_text(json.dumps({
        "rows": rows,
        "aggregate": [
            {"words": w, "channel": ch, **rec}
            for (w, ch), rec in sorted(agg.items())
        ],
    }, indent=2))
    print(f"wrote {out_json} ({len(rows)} rows)")

    tex = _build_tex(agg, words_axis)
    out_tex = ABL_ROOT / "table_capacity_expA_strict.tex"
    out_tex.write_text(tex)
    print(f"wrote {out_tex}")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.4))
    _panel_heatmap(ax, agg)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = ABL_ROOT / f"figure_capacity_strict.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")

    # Print the strict aggregate so we can see the cliff pattern
    print("\nStrict aggregate (n_success/n per cell):")
    print(f"  {'words':>5} | " + " ".join(f"{COLUMN_LABELS[c]:>8}" for c in COLUMNS))
    for w in words_axis:
        cells = []
        for ch in COLUMNS:
            rec = agg.get((w, ch))
            cells.append(_cell(rec))
        print(f"  {w:>5} | " + " ".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
