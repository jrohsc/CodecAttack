#!/usr/bin/env python3
"""Measure realized carrier-referenced SNR of an existing attacked-WAV bundle.

Every arm is referenced to the *same* carrier: the 24 kHz, peak-normalized
signal returned by ``load_music_by_name``, serialized to PCM_16 and reloaded.
This is the signal the latent attack perturbs, so referencing the waveform arms
to their own 16 kHz-bandlimited copy instead would compare two different
denominators. CPU only -- no victim model is loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODEC = ROOT.parent
sys.path.insert(0, str(CODEC))

from music_carrier import load_music_by_name  # noqa: E402
from reviewer_controls.common import above_cutoff_energy_fraction, snr_db  # noqa: E402
from reviewer_controls.run_attack import (  # noqa: E402
    duration_s, load_config, partition_pairs, stable_id,
)


def carrier_reference(carrier_path: Path, cache_dir: Path, example_id: str):
    """PCM_16 round-tripped common reference, cached per example."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_path = cache_dir / f"{example_id}_carrier_ref.wav"
    if not ref_path.is_file():
        wav = load_music_by_name(str(carrier_path), duration=duration_s(carrier_path))
        sf.write(ref_path, wav.squeeze().float().numpy(), 24000, subtype="PCM_16")
    data, sr = sf.read(ref_path, dtype="float32")
    return data, sr, ref_path


def find_attacked(audio_dir: Path, scenario: str, pair_index: int, carrier: str):
    stem = Path(carrier).stem
    if scenario == "s3a":
        hits = sorted(audio_dir.rglob(f"*pair{pair_index:03d}_attacked.wav"))
    else:
        hits = sorted(audio_dir.rglob(f"*{stem}*attacked.wav"))
    return hits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--scenario", choices=["s1", "s3a"], required=True)
    p.add_argument("--split", choices=["development", "evaluation", "all"], default="all")
    p.add_argument("--method", required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot", action="store_true",
                   help="Restrict to the documented pilot subset so the reused "
                        "arms cover exactly the examples the new arms ran.")
    args = p.parse_args()

    cfg = load_config(args.config)
    sc = cfg["scenarios"][args.scenario]
    audio_dir = args.bundle / "audio" if (args.bundle / "audio").is_dir() else args.bundle
    splits = ["development", "evaluation"] if args.split == "all" else [args.split]

    rows, missing = [], []
    cache = args.output.parent / "carrier_refs"
    for split in splits:
        for pair_index, pair in partition_pairs(cfg, args.scenario, split, pilot=args.pilot):
            example_id = stable_id(args.scenario, pair_index, pair["carrier"], pair["target"])
            hits = find_attacked(audio_dir, args.scenario, pair_index, pair["carrier"])
            if len(hits) != 1:
                missing.append({"example_id": example_id, "pair_index": pair_index,
                                "carrier": pair["carrier"], "split": split,
                                "status": "missing" if not hits else "ambiguous",
                                "candidates": [str(h) for h in hits]})
                continue
            adv, sr = sf.read(hits[0], dtype="float32")
            if adv.ndim > 1:
                adv = adv.mean(axis=1)
            carrier_path = ROOT / sc["carrier_root"] / pair["carrier"]
            ref, ref_sr, ref_path = carrier_reference(carrier_path, cache, example_id)
            if ref_sr != sr:
                missing.append({"example_id": example_id, "status": "sample_rate_mismatch",
                                "bundle_sr": int(sr), "reference_sr": int(ref_sr)})
                continue
            n = min(len(ref), len(adv))
            if abs(len(ref) - len(adv)) > 1:
                missing.append({"example_id": example_id, "status": "length_mismatch",
                                "bundle_samples": len(adv), "reference_samples": len(ref)})
                continue
            delta = adv[:n].astype(np.float64) - ref[:n].astype(np.float64)
            rows.append({
                "example_id": example_id, "scenario": args.scenario, "split": split,
                "pair_index": pair_index, "method": args.method, "carrier": pair["carrier"],
                "source_carrier_path": str(carrier_path), "target": pair["target"],
                "prompt": sc["prompt"], "adv_path": str(hits[0]),
                "carrier_ref_path": str(ref_path), "sample_rate": int(sr), "samples": int(n),
                "snr_db": snr_db(ref[:n], adv[:n]),
                "post_above_4khz_fraction": above_cutoff_energy_fraction(delta, sr, 4000.0),
                "delta_peak": float(np.max(np.abs(delta))),
                "delta_rms": float(np.sqrt(np.mean(delta ** 2))),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    snrs = np.asarray([r["snr_db"] for r in rows], dtype=float)
    summary = {
        "bundle": str(args.bundle), "method": args.method, "scenario": args.scenario,
        "pilot_subset": bool(args.pilot),
        "n_measured": len(rows), "n_unavailable": len(missing),
        "unavailable": missing,
        "snr_db_mean": float(snrs.mean()) if len(snrs) else None,
        "snr_db_median": float(np.median(snrs)) if len(snrs) else None,
        "snr_db_min": float(snrs.min()) if len(snrs) else None,
        "snr_db_max": float(snrs.max()) if len(snrs) else None,
        "snr_db_sd": float(snrs.std(ddof=1)) if len(snrs) > 1 else None,
        "above_4khz_fraction_mean": float(np.mean([r["post_above_4khz_fraction"] for r in rows])) if rows else None,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
