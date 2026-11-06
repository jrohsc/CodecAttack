"""
Paired-bootstrap 95% CI on the latent - waveform success-rate gap, per
(codec, bitrate) cell. Mirrors the per-clip strict-substring metric used
by build_main_results_tables.py and build_s1_codec_latex.py so the deltas
are directly comparable to tab_latent_vs_waveform.tex.

Method: for each cell (codec, bitrate), build a paired vector of length
n_common where each entry is (latent_hit_for_this_carrier,
waveform_hit_for_this_carrier). Resample carriers with replacement
B times; for each resample compute the delta in success rate
(latent_pct - waveform_pct). Report the 2.5 / 97.5 percentiles as the
95% CI on the delta.

Pairing matters: latent and waveform are evaluated on different adv wavs
*derived from the same carrier*, so per-carrier outcomes are not
independent across the two attacks. The paired bootstrap accounts for
the within-carrier correlation that an unpaired Wilson on each rate
would ignore.

Usage:
    python scripts/bootstrap_latent_vs_waveform_deltas.py
    python scripts/bootstrap_latent_vs_waveform_deltas.py --b 20000
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCEN = "01_finance_voice_agent"
DEFAULT_LATENT_DIR = ROOT / f"results_codec_robust/{SCEN}/qwen2_audio/eps_1.0/audio"
DEFAULT_WAVE_DIR = ROOT / f"results_codec_robust_waveform/{SCEN}/qwen2_audio/eps_0.075/audio"

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


def _load_json_map(audio_dir: Path, suffix: str) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(str(audio_dir / f"*{suffix}"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        stem = _carrier_stem(d.get("clip", ""))
        if stem:
            out[stem] = d
    return out


def _hit(d: dict | None, codec: str, bw: str) -> int | None:
    """Return 1 if strict substring match, 0 if not, None if cell missing."""
    if d is None:
        return None
    key = f"{codec}_robustness"
    row = (d.get(key) or {}).get(bw)
    if not isinstance(row, dict):
        return None
    target = (d.get("target_text") or "").strip().lower()
    out = (row.get("output") or "").strip().lower()
    if not target:
        return None
    return 1 if target in out else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    p.add_argument("--wave-dir",   type=Path, default=DEFAULT_WAVE_DIR)
    p.add_argument("--b",          type=int, default=10000, help="Bootstrap resamples (default 10000)")
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    lat_summ = _load_json_map(args.latent_dir, "_summary.json")
    lat_codec = _load_json_map(args.latent_dir, "_codecs.json")
    wav_summ = _load_json_map(args.wave_dir, "_summary.json")
    wav_codec = _load_json_map(args.wave_dir, "_codecs.json")

    common = sorted(set(lat_summ.keys()) & set(wav_summ.keys()))
    if not common:
        raise SystemExit("no common carriers between latent and waveform bundles")
    print(f"# n_common = {len(common)} carriers")
    print(f"# bootstrap B = {args.b}")
    print()

    print(f"{'codec':6s} {'bw':6s} {'lat_pct':>8s} {'wav_pct':>8s} {'delta':>8s} {'ci_lo':>8s} {'ci_hi':>8s} {'sig':>5s}")
    print("-" * 65)

    rows_for_md = []
    for codec, label, bitrates in CODEC_SPECS:
        for bw in bitrates:
            lat_src = lat_summ if codec == "opus" else lat_codec
            wav_src = wav_summ if codec == "opus" else wav_codec
            lat_hits, wav_hits = [], []
            for c in common:
                lh = _hit(lat_src.get(c), codec, bw)
                wh = _hit(wav_src.get(c), codec, bw)
                if lh is None or wh is None:
                    continue
                lat_hits.append(lh)
                wav_hits.append(wh)
            n = len(lat_hits)
            if n == 0:
                continue
            lat_arr = np.array(lat_hits)
            wav_arr = np.array(wav_hits)
            lat_pct = 100.0 * lat_arr.mean()
            wav_pct = 100.0 * wav_arr.mean()
            delta = lat_pct - wav_pct
            # Paired bootstrap
            idx = rng.integers(0, n, size=(args.b, n))
            lat_boot = lat_arr[idx].mean(axis=1)
            wav_boot = wav_arr[idx].mean(axis=1)
            delta_boot = 100.0 * (lat_boot - wav_boot)
            lo, hi = np.percentile(delta_boot, [2.5, 97.5])
            sig = "***" if (lo > 0 or hi < 0) else "n.s."
            print(f"{label:6s} {bw+'k':>6s} {lat_pct:>7.1f}% {wav_pct:>7.1f}% "
                  f"{delta:>+7.1f} [{lo:>+5.0f},{hi:>+5.0f}] {sig:>5s}")
            rows_for_md.append({
                "codec": label, "bw_kbps": int(bw), "n": n,
                "latent_pct": round(lat_pct, 1),
                "waveform_pct": round(wav_pct, 1),
                "delta_pp": round(delta, 1),
                "ci_lo_pp": round(lo, 1),
                "ci_hi_pp": round(hi, 1),
                "significant_at_95pct": bool(lo > 0 or hi < 0),
            })

    out_path = ROOT / "paper-neurips/tables/latent_minus_waveform_deltas.json"
    out_path.write_text(json.dumps({
        "scenario": SCEN, "model": "qwen2_audio",
        "latent_bundle": str(args.latent_dir.relative_to(ROOT)),
        "waveform_bundle": str(args.wave_dir.relative_to(ROOT)),
        "n_common": len(common),
        "bootstrap_B": args.b,
        "metric": "strict_substring_target_in_output",
        "rows": rows_for_md,
    }, indent=2))
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
