#!/usr/bin/env python3
"""Audit and select targeted S1 4 kHz SNR repairs without reading ASR outcomes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from reviewer_controls.common import above_cutoff_energy_fraction, snr_db

ROOT = Path(__file__).resolve().parents[1] / "reviewer_control_outputs"
D = ROOT / "s1_4khz_full50_seed1"
BASE = ROOT / "s1_baseline_full40_seed1"
LATENT = BASE / "manifests/armA_latent.jsonl"
CANON = D / "frozen/bandlimited_4khz.evaluation.canonical.jsonl"


def read(path):
    return {r["example_id"]: r for r in map(json.loads, open(path))}


def measured():
    latent, current = read(LATENT), read(CANON)
    assert len(latent) == len(current) == 40 and latent.keys() == current.keys()
    out = []
    for key in sorted(current):
        a, d = latent[key], current[key]
        original, sr0 = sf.read(a["carrier_ref_path"], dtype="float64")
        d_ref, sr1 = sf.read(d["carrier_ref_path"], dtype="float64")
        attacked, sr2 = sf.read(d["adv_path"], dtype="float64")
        attacker_ref, sr3 = sf.read(d["attacker_ref_path"], dtype="float64")
        assert (sr0, sr1, sr2, sr3) == (24000,) * 4
        assert len(original) == len(d_ref) == len(attacked) == len(attacker_ref) == 240000
        assert np.array_equal(original, d_ref), key
        latent_path = Path(a["adv_path"])
        if not latent_path.is_absolute():
            latent_path = ROOT.parent / latent_path
        snr_a = snr_db(original, sf.read(latent_path, dtype="float64")[0])
        snr_d = snr_db(original, attacked)
        delta = attacked - original
        pure_delta = attacked - attacker_ref
        out.append({
            "example_id": key, "latent_snr_db": snr_a, "waveform_snr_db": snr_d,
            "signed_difference_db": snr_d - snr_a,
            "perturbation_l2": float(np.linalg.norm(delta)),
            "perturbation_rms": float(np.sqrt(np.mean(delta ** 2))),
            "perturbation_peak": float(np.max(np.abs(delta))),
            "clipping_rate": float(np.mean(np.abs(attacked) >= 32767 / 32768)),
            "carrier_referenced_above_4khz_fraction": above_cutoff_energy_fraction(delta, 24000, 4000),
            "attack_only_above_4khz_fraction": above_cutoff_energy_fraction(pure_delta, 24000, 4000),
            "epsilon": d["epsilon"], "range_projection_bound": d["range_projection_bound"],
            "waveform_path": d["adv_path"],
        })
    return out


def history(ids, latent_snr):
    result = {key: [] for key in ids}
    for path in sorted(D.glob("s1/evaluation/bandlimited_4khz/*/manifest.jsonl")):
        for row in map(json.loads, open(path)):
            key = row["example_id"]
            if key in result:
                result[key].append({"run": path.parent.name, "epsilon": row["epsilon"],
                                    "residual_db": row["snr_db"] - latent_snr[key],
                                    "range_projection_bound": row.get("range_projection_bound"),
                                    "elapsed_s": row.get("elapsed_s")})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-repair", action="store_true")
    args = p.parse_args()
    records = measured()
    frozen = D / "frozen"
    with open(frozen / "bandlimited_4khz.evaluation.audio_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    bad = [r for r in records if abs(r["signed_difference_db"]) > 1]
    print(json.dumps({"n": len(records), "within_1db": len(records)-len(bad),
                      "mean_residual_db": float(np.mean([r["signed_difference_db"] for r in records])),
                      "max_abs_residual_db": max(abs(r["signed_difference_db"]) for r in records),
                      "failures": [{"id": r["example_id"], "residual_db": r["signed_difference_db"]} for r in bad]}, indent=2))
    if args.write_repair:
        expected = {1, 3, 5, 14, 22}
        assert {int(r["example_id"].split("_")[1][1:]) for r in bad} == expected
        eps = {1: 0.247, 3: 0.335, 5: 0.245, 14: 0.290, 22: 0.240}
        mapping = {r["example_id"]: eps[int(r["example_id"].split("_")[1][1:])] for r in bad}
        hist = history(mapping, {r["example_id"]: r["latent_snr_db"] for r in records})
        payload = {"status": "frozen", "mode": "distortion_only_bracket_interpolation",
                   "rule": "choose an untried epsilon between recorded over- and under-distortion points; no ASR text or success used",
                   "epsilon_map": mapping, "prior_attempts": hist,
                   "additional_optimizations_planned": len(mapping)}
        (frozen / "epsilon_bandlimited_4khz.evaluation.r4.json").write_text(json.dumps(payload, indent=2))
        (frozen / "bandlimited_4khz.evaluation.bad_r4.txt").write_text("".join(k+"\n" for k in sorted(mapping)))


if __name__ == "__main__": main()
