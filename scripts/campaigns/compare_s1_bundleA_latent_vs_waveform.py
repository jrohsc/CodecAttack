"""
S1 Bundle A — fair latent-vs-waveform comparison across codecs.

Aggregates per-carrier robustness for Opus (from *_summary.json) and
MP3/AAC (from *_codecs.json sidecars produced by eval_mp3_aac_bundle.py),
intersects on carriers completed in both bundles, and emits:
  - Vertical per-codec tables (Bitrate | Latent | Waveform) printed to stdout
  - A markdown file with the same tables plus exact-match, WER, SNR rollups.

Fairness choices (see feedback_latent_vs_waveform_framing.md):
  - Same S1 carriers (intersection of completed pairs)
  - Same prompt, same targets
  - SNR-matched perturbation budget (latent eps=1.0 ≈ waveform eps=0.075 →
    both sit near 5.77 dB clean SNR on S1 speech carriers)

Inputs (I/O only, no model inference here):
  - `results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0/audio/
       *_summary.json`           (opus_robustness rows)
  - `results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0/audio/
       *_codecs.json`            (mp3_robustness + aac_robustness sidecars)
  - `results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/
       eps_0.075/audio/*_summary.json` + *_codecs.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 0_all_combined/
SCEN = "01_finance_voice_agent"
DEFAULT_LATENT_DIR = ROOT / f"results_codec_robust/{SCEN}/qwen2_audio/eps_1.0/audio"
DEFAULT_WAVE_DIR = ROOT / f"results_codec_robust_waveform/{SCEN}/qwen2_audio"

OPUS_BITRATES = ["16", "24", "32", "64", "128", "192"]
MP3_BITRATES = ["64", "96", "128", "192"]
AAC_BITRATES = ["64", "96", "128", "192"]

CODEC_SPECS = [
    ("opus", "Opus", OPUS_BITRATES),
    ("mp3",  "MP3",  MP3_BITRATES),
    ("aac",  "AAC",  AAC_BITRATES),
]


def _carrier_stem(clip: str) -> str | None:
    m = re.match(r"^music_(.+?)__target_", clip)
    return m.group(1) if m else None


def _load_json_map(glob_path: str, key_field: str = "clip") -> dict[str, dict]:
    """Map carrier_stem -> dict read from each matching json."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(glob_path)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        clip = d.get(key_field, "")
        stem = _carrier_stem(clip)
        if stem:
            out[stem] = d
    return out


def _pick_wave_eps_dir(wave_root: Path, eps_arg: float | None) -> Path:
    if eps_arg is not None:
        cand = wave_root / f"eps_{eps_arg:g}"
        if not cand.is_dir():
            raise SystemExit(f"no waveform bundle at {cand}")
        return cand / "audio"
    eps_dirs = sorted(p for p in wave_root.iterdir() if p.is_dir() and p.name.startswith("eps_"))
    if not eps_dirs:
        raise SystemExit(f"no eps_* subdirs under {wave_root}")
    target = next((p for p in eps_dirs if p.name == "eps_0.075"), eps_dirs[-1])
    return target / "audio"


def _get_codec_rows(entry: dict, codec: str) -> dict:
    key = f"{codec}_robustness"
    return entry.get(key) or {}


def _aggregate(summaries: dict[str, dict], codec_sidecars: dict[str, dict],
               carriers: list[str], codec: str, bitrates: list[str]) -> dict[str, dict]:
    """Per-bitrate aggregate for a given codec over the carrier subset.

    For opus we read from the *_summary.json; for mp3/aac from the *_codecs.json.
    """
    per_br: dict[str, dict] = {
        bw: {"n": 0, "sub": 0, "exact": 0, "wer_sum": 0.0, "snr_sum": 0.0}
        for bw in bitrates
    }
    for c in carriers:
        if codec == "opus":
            d = summaries.get(c)
        else:
            d = codec_sidecars.get(c)
        if d is None:
            continue
        rows = _get_codec_rows(d, codec)
        for bw, info in rows.items():
            if bw not in per_br or not isinstance(info, dict):
                continue
            per_br[bw]["n"] += 1
            per_br[bw]["sub"] += int(bool(info.get("matches_target", False)))
            per_br[bw]["exact"] += int(bool(info.get("exact_match", False)))
            wer = info.get("wer", 0.0)
            per_br[bw]["wer_sum"] += float(wer if wer is not None else 0.0)
            per_br[bw]["snr_sum"] += float(info.get("snr_db", 0.0) or 0.0)
    agg = {}
    for bw, row in per_br.items():
        n = row["n"]
        agg[bw] = {
            "n": n,
            "success_rate_pct": 100.0 * row["sub"] / n if n else 0.0,
            "exact_match_pct": 100.0 * row["exact"] / n if n else 0.0,
            "avg_wer": row["wer_sum"] / n if n else 0.0,
            "avg_snr_db": row["snr_sum"] / n if n else 0.0,
        }
    return agg


def _fmt_cell(agg: dict, bw: str, field: str, fmt: str) -> str:
    return fmt.format(agg[bw][field]) if agg[bw]["n"] else "—"


def _ascii_table(codec_label: str, bitrates: list[str], latent_agg: dict,
                 wave_agg: dict, field: str, fmt: str = "{:.1f}") -> str:
    """Render ┌─┐-style vertical table: Bitrate | Latent | Waveform."""
    rows = []
    for bw in bitrates:
        tag = f"{codec_label.lower()}{bw}k"
        l = _fmt_cell(latent_agg, bw, field, fmt)
        w = _fmt_cell(wave_agg, bw, field, fmt)
        rows.append((tag, l, w))
    w_bw = max(len("Bitrate"), *(len(r[0]) for r in rows))
    w_la = max(len("Latent"),  *(len(r[1]) for r in rows))
    w_wv = max(len("Waveform"), *(len(r[2]) for r in rows))

    def line(ch_l: str, ch_m: str, ch_r: str, ch_f: str = "─") -> str:
        return f"{ch_l}{ch_f * (w_bw + 2)}{ch_m}{ch_f * (w_la + 2)}{ch_m}{ch_f * (w_wv + 2)}{ch_r}"

    def row(a: str, b: str, c: str) -> str:
        return f"│ {a.ljust(w_bw)} │ {b.ljust(w_la)} │ {c.ljust(w_wv)} │"

    lines = []
    lines.append(line("┌", "┬", "┐"))
    lines.append(row("Bitrate", "Latent", "Waveform"))
    lines.append(line("├", "┼", "┤"))
    for i, (a, b, c) in enumerate(rows):
        lines.append(row(a, b, c))
        if i < len(rows) - 1:
            lines.append(line("├", "┼", "┤"))
    lines.append(line("└", "┴", "┘"))
    return "\n".join(lines)


def _md_table(codec_label: str, bitrates: list[str], latent_agg: dict,
              wave_agg: dict, field: str, fmt: str = "{:.1f}") -> list[str]:
    out = [f"| Bitrate | Latent | Waveform |",
           f"|---|---|---|"]
    for bw in bitrates:
        tag = f"{codec_label.lower()}{bw}k"
        l = _fmt_cell(latent_agg, bw, field, fmt)
        w = _fmt_cell(wave_agg, bw, field, fmt)
        out.append(f"| {tag} | {l} | {w} |")
    return out


def _bundle_state(audio_dir: Path, label: str) -> tuple[dict, dict]:
    summaries = _load_json_map(str(audio_dir / "*_summary.json"))
    codecs = _load_json_map(str(audio_dir / "*_codecs.json"))
    print(f"  {label}: {len(summaries)} summaries, {len(codecs)} codec sidecars")
    return summaries, codecs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR,
                   help="Path to latent bundle's audio/ dir.")
    p.add_argument("--wave-root", type=Path, default=DEFAULT_WAVE_DIR,
                   help="Path to waveform bundle parent (contains eps_* dirs).")
    p.add_argument("--wave-eps", type=float, default=None,
                   help="Which waveform eps subdir to compare (default 0.075).")
    p.add_argument("--latent-snr-db", type=float, default=5.77)
    p.add_argument("--wave-snr-db", type=float, default=5.8)
    p.add_argument("--target-model", default="qwen2_audio")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    wave_audio_dir = _pick_wave_eps_dir(args.wave_root, args.wave_eps)
    wave_eps_tag = wave_audio_dir.parent.name  # "eps_0.075"

    print(f"scenario: {SCEN}  target_model: {args.target_model}")
    latent_summaries, latent_codecs = _bundle_state(args.latent_dir, "latent")
    wave_summaries, wave_codecs = _bundle_state(wave_audio_dir, "waveform")

    common = sorted(set(latent_summaries.keys()) & set(wave_summaries.keys()))
    print(f"intersection (fair comparison set): {len(common)} carriers")
    if not common:
        raise SystemExit("no common carriers")

    # Render each codec's vertical table
    print()
    agg_by_codec: dict[str, tuple[dict, dict]] = {}
    for codec, label, bitrates in CODEC_SPECS:
        lat_agg = _aggregate(latent_summaries, latent_codecs, common, codec, bitrates)
        wav_agg = _aggregate(wave_summaries, wave_codecs, common, codec, bitrates)
        agg_by_codec[codec] = (lat_agg, wav_agg)
        print(f"{label} — Target substring match %")
        print()
        print(_ascii_table(label, bitrates, lat_agg, wav_agg, "success_rate_pct"))
        print()

    # Markdown
    out_md = args.out or (ROOT / f"results_codec_robust_waveform/{SCEN}/"
                          f"latent_vs_waveform_bundleA_s1.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append(f"# S1 Bundle A — Latent vs Waveform across codecs\n")
    md.append(
        f"- Scenario: `{SCEN}`, target model `{args.target_model}`.\n"
        f"- **Fair comparison set**: {len(common)} carriers present in both bundles.\n"
        f"- Latent: eps=1.0, clean SNR {args.latent_snr_db:.2f} dB (measured), "
        f"multi-bitrate Opus EoT {{16,24,32,64,128}} kbps in EnCodec latent space.\n"
        f"- Waveform: {wave_eps_tag}, clean SNR ~{args.wave_snr_db:.2f} dB.\n"
        f"- Opus rows come from each attacker's `test_opus_robustness`; "
        f"MP3/AAC rows come from a post-hoc inference-only eval "
        f"(`scripts/eval_mp3_aac_bundle.py`), no retraining.\n"
    )

    for codec, label, bitrates in CODEC_SPECS:
        lat_agg, wav_agg = agg_by_codec[codec]
        md.append(f"\n## {label} — target_substring_match %\n")
        md.extend(_md_table(label, bitrates, lat_agg, wav_agg, "success_rate_pct"))
        md.append(f"\n### {label} — exact_match %\n")
        md.extend(_md_table(label, bitrates, lat_agg, wav_agg, "exact_match_pct"))
        md.append(f"\n### {label} — avg WER\n")
        md.extend(_md_table(label, bitrates, lat_agg, wav_agg, "avg_wer", "{:.3f}"))
        md.append(f"\n### {label} — post-codec avg SNR (dB)\n")
        md.extend(_md_table(label, bitrates, lat_agg, wav_agg, "avg_snr_db"))

    md.append("\n## Sources\n")
    md.append(f"- Latent bundle: `{args.latent_dir.relative_to(ROOT)}/*_summary.json` + `*_codecs.json`")
    md.append(f"- Waveform bundle: `{wave_audio_dir.relative_to(ROOT)}/*_summary.json` + `*_codecs.json`")

    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
