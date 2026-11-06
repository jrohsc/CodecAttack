"""Phase C — generate the waveform_vs_latent_s3a_qwen2.md comparison table.

Reads the latent rollup (canonical, produced by audio_agent_bench) and the
waveform quick-eval JSON (substring-match + stored model outputs), derives
exact_match / WER on the waveform side from the stored outputs, and emits a
side-by-side markdown table with three sections: headline substring-match
%, avg WER, exact-match %.

Waveform channel tags are normalised to latent's `opus192k`/`mp3_320k`
convention. Channels not present in a bundle are rendered as `—`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCEN = "03a_ai_detection_bypass"
MODEL_SHORT = {
    "qwen2_audio": "qwen2",
    "qwen25_omni": "omni",
    "audio_flamingo": "af3",
}

CHANNEL_ORDER = [
    "clean",
    "opus192k", "opus128k", "opus64k", "opus32k", "opus24k", "opus16k",
    "aac256k", "aac192k", "aac128k",
    "mp3_320k", "mp3_192k",
    "g722_64k",
]


def _normalise_wave_channel(tag: str) -> str:
    if tag == "clean":
        return "clean"
    m = re.match(r"^(opus|aac)(\d+)$", tag)
    if m:
        return f"{m.group(1)}{m.group(2)}k"
    m = re.match(r"^(mp3)_(\d+)$", tag)
    if m:
        return f"mp3_{m.group(2)}k"
    return tag


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _wer(ref: str, hyp: str) -> float:
    ref_t = ref.split()
    hyp_t = hyp.split()
    if not ref_t:
        return 1.0 if hyp_t else 0.0
    dp = list(range(len(hyp_t) + 1))
    for i, r in enumerate(ref_t, 1):
        prev, dp[0] = dp[0], i
        for j, h in enumerate(hyp_t, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r != h))
            prev = cur
    return dp[-1] / len(ref_t)


def _load_latent_by_channel(rollup_path: Path) -> dict[str, dict]:
    d = json.loads(rollup_path.read_text())
    return {row["channel_tag"]: row for row in d["by_channel"]}


def _aggregate_quick(qe_path: Path) -> tuple[dict[str, dict], float]:
    d = json.loads(qe_path.read_text())
    rows = d["rows"]
    snrs = [r["snr_clean_db"] for r in rows if r.get("snr_clean_db") is not None]
    mean_snr = sum(snrs) / len(snrs) if snrs else float("nan")

    per_ch: dict[str, list[dict]] = {}
    for r in rows:
        target = _norm_text(r["target"])
        for raw_tag, cell in r["per_channel"].items():
            norm = _normalise_wave_channel(raw_tag)
            out = _norm_text(cell.get("out", ""))
            sub = bool(target) and target in out
            exact = bool(target) and target == out
            wer = _wer(target, out)
            per_ch.setdefault(norm, []).append(
                {"sub": sub, "exact": exact, "wer": wer}
            )

    agg: dict[str, dict] = {}
    for ch, vs in per_ch.items():
        n = len(vs)
        agg[ch] = {
            "channel_tag": ch,
            "n": n,
            "success_rate_pct": 100.0 * sum(v["sub"] for v in vs) / n,
            "exact_match_pct": 100.0 * sum(v["exact"] for v in vs) / n,
            "avg_wer": sum(v["wer"] for v in vs) / n,
        }
    return agg, mean_snr


def _fmt(val):
    if val is None:
        return "—"
    return f"{val:.1f}"


def _fmt3(val):
    if val is None:
        return "—"
    return f"{val:.3f}"


def _row(label: str, snr, by_ch: dict[str, dict], field: str, fn=_fmt) -> str:
    cells = [fn(by_ch.get(ch, {}).get(field)) for ch in CHANNEL_ORDER]
    snr_cell = f"{snr:.2f}" if isinstance(snr, (int, float)) else str(snr)
    return "| " + " | ".join([label, snr_cell, *cells]) + " |"


def _header() -> list[str]:
    return [
        "| Attack | Mean SNR (dB) | " + " | ".join(CHANNEL_ORDER) + " |",
        "|---|---|" + "---|" * len(CHANNEL_ORDER),
    ]


def _model_title(model: str) -> str:
    return {"qwen2_audio": "Qwen2-Audio",
            "qwen25_omni": "Qwen2.5-Omni",
            "audio_flamingo": "Audio-Flamingo-3"}.get(model, model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2_audio",
                   choices=list(MODEL_SHORT.keys()))
    args = p.parse_args()

    latent_rollup = ROOT / f"results/{SCEN}/{args.model}/eps_0.5/rollup.json"
    candidates = [
        ROOT / f"results/{SCEN}/waveform_comparison/calibration/{args.model}/snr_sweep.json",
        ROOT / f"results/{SCEN}/waveform_comparison/calibration/snr_sweep.json",
    ]
    calib_json = next((c for c in candidates if c.is_file()), None)
    if calib_json is None:
        raise SystemExit(f"missing calibration file: tried {candidates}")
    out_md = ROOT / f"results/{SCEN}/waveform_comparison/waveform_vs_latent_s3a_{MODEL_SHORT[args.model]}.md"

    calib = json.loads(calib_json.read_text())
    eps_star = calib["recommended_eps"]
    wave_dir = ROOT / f"results/{SCEN}/waveform_comparison/{args.model}/eps_{eps_star}"
    qe = wave_dir / "quick_eval_per_channel.json"
    if not qe.is_file():
        raise SystemExit(f"missing quick-eval file: {qe}")

    latent = _load_latent_by_channel(latent_rollup)
    wave, wave_snr_mean = _aggregate_quick(qe)
    latent_snr_mean = calib["latent_reference_snr_db"]
    n_wave = next(iter(wave.values()))["n"] if wave else 0
    n_latent = next(iter(latent.values()))["n"] if latent else 0

    md: list[str] = []
    md.append(f"# Waveform vs Latent on {SCEN} ({_model_title(args.model)})\n")
    md.append(
        f"- Latent: eps=0.5, n={n_latent} pairs, mean SNR "
        f"{latent_snr_mean:.2f} dB (from `{latent_rollup.relative_to(ROOT)}`).\n"
        f"- Waveform: eps={eps_star}, n={n_wave} pairs, mean SNR "
        f"{wave_snr_mean:.2f} dB (computed from `{qe.relative_to(ROOT)}`).\n"
        f"- Waveform WER / exact_match derived from stored model outputs; "
        f"waveform bundle has no `g722_64k` channel.\n"
    )

    md.append("## Headline: target_substring_match %\n")
    md.extend(_header())
    md.append(_row("Latent eps=0.5", latent_snr_mean, latent, "success_rate_pct"))
    md.append(_row(f"Waveform eps={eps_star}", wave_snr_mean, wave, "success_rate_pct"))

    md.append("\n## Secondary: avg WER\n")
    md.extend(_header())
    md.append(_row("Latent eps=0.5", latent_snr_mean, latent, "avg_wer", _fmt3))
    md.append(_row(f"Waveform eps={eps_star}", wave_snr_mean, wave, "avg_wer", _fmt3))

    md.append("\n## Secondary: exact_match %\n")
    md.extend(_header())
    md.append(_row("Latent eps=0.5", latent_snr_mean, latent, "exact_match_pct"))
    md.append(_row(f"Waveform eps={eps_star}", wave_snr_mean, wave, "exact_match_pct"))

    md.append("\n## Sources\n")
    md.append(f"- Latent rollup: `{latent_rollup.relative_to(ROOT)}`")
    md.append(f"- Waveform quick eval: `{qe.relative_to(ROOT)}`")
    md.append(f"- Calibration: `{calib_json.relative_to(ROOT)}`")

    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
