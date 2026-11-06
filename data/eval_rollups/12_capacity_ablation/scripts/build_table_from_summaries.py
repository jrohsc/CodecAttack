"""Emit table_capacity_expA_from_summaries.tex from per-cell *_summary.json.

Until the standalone codec eval pass (eval_channels.py -> *_codecs.json) lands,
this builds the headline capacity table from the multi-bitrate Opus sweep that
each cell already records under summary.json["opus_robustness"].

Channels reported: Clean (top-level success on the optimization-side decode)
and Opus at 192 / 128 / 64 / 32 / 24 / 16 kbps.

Cell semantics: success rate across the 3 Suno carriers (jazz, classic, calm).
"Success" follows the S3a rule: target substring ("human-authored", case
insensitive) is present in the channel-decoded output OR fractional WER <= 0.3.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent
EXP_A_ROOT = ABL_ROOT / "expA_words/qwen2_audio_eps1.0_multibitrate"
TARGETS = json.loads((ABL_ROOT / "targets_capacity.json").read_text())
SUBSTR = TARGETS["substring_match_key"].lower()
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


def _is_success_clean(summary: dict) -> bool:
    out = (summary.get("final_output") or "").lower()
    if SUBSTR in out:
        return True
    return bool(summary.get("success"))


def _is_success_opus(rec: dict) -> bool:
    if rec.get("matches_target"):
        return True
    out = (rec.get("output") or "").lower()
    if SUBSTR in out:
        return True
    return rec.get("wer", 1.0) <= 0.3


def _walk() -> list[dict]:
    rows: list[dict] = []
    for sm in sorted(EXP_A_ROOT.rglob("*_summary.json")):
        carrier = sm.parent.parent.name
        target_id = sm.parent.name
        if target_id not in WORDS_BY_ID:
            continue
        words = WORDS_BY_ID[target_id]
        s = json.loads(sm.read_text())
        rows.append({
            "carrier": carrier, "target_id": target_id, "words": words,
            "channel": "clean",
            "success": int(_is_success_clean(s)),
            "wer": -1.0,  # not recorded for the clean optimization-side decode
            "snr_db": -1.0,
            "final_loss": s.get("final_loss", -1.0),
        })
        for b in OPUS_BITRATES:
            rec = (s.get("opus_robustness") or {}).get(str(b))
            if rec is None:
                continue
            rows.append({
                "carrier": carrier, "target_id": target_id, "words": words,
                "channel": f"opus_{b}",
                "success": int(_is_success_opus(rec)),
                "wer": rec.get("wer", -1.0),
                "snr_db": rec.get("snr_db", -1.0),
                "final_loss": s.get("final_loss", -1.0),
            })
    return rows


def _aggregate(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["words"], r["channel"])].append(r)
    agg = {}
    for key, rs in grouped.items():
        succ = [r["success"] for r in rs]
        wers = [r["wer"] for r in rs if r["wer"] >= 0]
        snrs = [r["snr_db"] for r in rs if r["snr_db"] >= -50]
        losses = [r["final_loss"] for r in rs if r["final_loss"] >= 0]
        agg[key] = {
            "n": len(rs),
            "n_success": sum(succ),
            "mean_success": statistics.mean(succ),
            "mean_wer": statistics.mean(wers) if wers else float("nan"),
            "mean_snr_db": statistics.mean(snrs) if snrs else float("nan"),
            "mean_final_loss": statistics.mean(losses) if losses else float("nan"),
        }
    return agg


def _cell(rec: dict | None) -> str:
    if rec is None or rec["n"] == 0:
        return "--"
    return f"{rec['n_success']}/{rec['n']}"


def _build_tex(agg: dict, words_axis: list[int]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Capacity-cliff ablation, Exp-A (target word count). "
        r"Multi-bitrate-EOT latent attack on Qwen2-Audio "
        r"($\epsilon{=}1.0$, 1000 steps, S3a classifier prompt, "
        r"3 Suno carriers: jazz / classic / calm). "
        r"Each cell reports the number of carriers (out of those evaluated) "
        r"for which the attack succeeds; success = ``human-authored'' substring "
        r"in the decoded output OR fractional WER $\leq$ 0.3. Clean column uses the top-level "
        r"optimization-side decode; Opus columns use the in-attack opus\_robustness "
        r"sweep on the attacked waveform. \textbf{Loss} is the mean final PGD loss "
        r"on clean. Rows with ``$\dagger$'' have only 2/3 carriers attacked so far.}",
        r"\label{tab:capacity_expA}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{r" + "c" * len(COLUMNS) + r"c}",
        r"\toprule",
        r"\textbf{Words} & " +
        " & ".join(r"\textbf{" + COLUMN_LABELS[c] + "}" for c in COLUMNS) +
        r" & \textbf{Loss} \\",
        r"\midrule",
    ]
    for w in words_axis:
        n_carriers_clean = agg.get((w, "clean"), {}).get("n", 0)
        partial = "$^{\\dagger}$" if 0 < n_carriers_clean < 3 else ""
        cells = [f"{w}{partial}"]
        loss = agg.get((w, "clean"), {}).get("mean_final_loss", float("nan"))
        for ch in COLUMNS:
            cells.append(_cell(agg.get((w, ch))))
        cells.append(f"{loss:.3f}" if loss == loss else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    rows = _walk()
    agg = _aggregate(rows)
    words_axis = sorted({t["words"] for t in TARGETS["targets"]})

    out_json = ABL_ROOT / "rollup_expA_from_summaries.json"
    out_json.write_text(json.dumps({
        "rows": rows,
        "aggregate": [
            {"words": w, "channel": ch, **rec}
            for (w, ch), rec in sorted(agg.items())
        ],
    }, indent=2))
    print(f"wrote {out_json} ({len(rows)} rows)")

    tex = _build_tex(agg, words_axis)
    out_tex = ABL_ROOT / "table_capacity_expA_from_summaries.tex"
    out_tex.write_text(tex)
    print(f"wrote {out_tex}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
