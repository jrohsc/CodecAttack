"""Emit table_capacity_v2.tex — NeurIPS booktabs table from rollup_expA.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent

COLUMN_ORDER = ["clean", "opus_192k", "opus_128k", "opus_64k",
                "aac_128k", "mp3_192k"]
COLUMN_LABELS = {"clean": "Clean", "opus_192k": "Opus192",
                 "opus_128k": "Opus128", "opus_64k": "Opus64",
                 "aac_128k": "AAC128", "mp3_192k": "MP3\\_192"}


def _cell(m: float, wer: float) -> str:
    if m >= 0.67:
        return r"\cmark"
    if m >= 0.33:
        return r"{\scriptsize " + f"{wer:.2f}" + "}"
    return r"\xmark"


def _build_table(agg: list[dict]) -> str:
    by_key = {(r["words"], r["channel"]): r for r in agg}
    word_counts = sorted({r["words"] for r in agg})
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Adversarial capacity cliff under multi-bitrate EOT "
        r"(Qwen2-Audio, $\epsilon{=}1.0$, 1000 steps, S3a classifier prompt, "
        r"3 Suno carriers). Cell: \cmark\ ($\geq$67\% ASR across carriers), "
        r"WER (33--66\%), \xmark\ ($<$33\%). Loss is mean final PGD loss on "
        r"clean channel.}",
        r"\label{tab:capacity_v2}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{r" + "c" * len(COLUMN_ORDER) + "c}",
        r"\toprule",
        r"\textbf{Words} & " +
        " & ".join(r"\textbf{" + COLUMN_LABELS[c] + "}" for c in COLUMN_ORDER) +
        r" & \textbf{Loss} \\",
        r"\midrule",
    ]
    for w in word_counts:
        cells = [f"{w:>3d}"]
        loss = 0.0
        for ch in COLUMN_ORDER:
            rec = by_key.get((w, ch))
            if rec is None:
                cells.append("--")
                continue
            cells.append(_cell(rec["mean_success"], rec["mean_wer"]))
            if ch == "clean":
                loss = rec.get("mean_final_loss", 0.0)
        cells.append(f"{loss:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    rollup = json.loads((ABL_ROOT / "rollup_expA.json").read_text())
    tex = _build_table(rollup["aggregate"])
    out = ABL_ROOT / "table_capacity_v2.tex"
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
