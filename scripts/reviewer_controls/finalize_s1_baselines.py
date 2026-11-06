#!/usr/bin/env python3
"""Validate the complete paired evaluation and write per-example audio records."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from reviewer_controls.common import EVAL_CHANNELS, above_cutoff_energy_fraction, snr_db, target_match
from reviewer_controls.finish_s1_4khz import BASE, D, read

ROOT = BASE.parent.parent
SPLIT = ROOT / "reviewer_control_outputs/S1_SPLIT_MANIFEST.csv"
FILES = [BASE / "eval/arms_ABCD.jsonl", BASE / "eval/armC.jsonl", BASE / "eval/armD.jsonl"]
METHODS = ("latent_codecattack", "original_waveform_published", "tuned_waveform", "bandlimited_4khz")


def path_of(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def main():
    expected = {r["example_id"] for r in csv.DictReader(open(SPLIT)) if r["split"] == "evaluation"}
    assert len(expected) == 40
    records = [json.loads(line) for p in FILES for line in open(p) if line.strip()]
    grouped = {method: {} for method in METHODS}
    for r in records:
        method, key = r["method"], r["example_id"]
        assert method in grouped and key in expected and key not in grouped[method], (method, key)
        assert r["split"] == "evaluation" and set(r["outcomes"]) == set(EVAL_CHANNELS)
        for channel in EVAL_CHANNELS:
            result = r["outcomes"][channel]
            assert result["match"] == target_match(r["target"], result["output"]), (method,key,channel)
        grouped[method][key] = r
    assert all(set(g) == expected for g in grouped.values()), {m:len(g) for m,g in grouped.items()}
    (BASE / "evaluation_records.jsonl").write_text(
        "".join(json.dumps(grouped[m][key], ensure_ascii=False)+"\n"
                for m in METHODS for key in sorted(expected)))
    manifests = {
        METHODS[0]: read(BASE / "manifests/armA_latent.jsonl"),
        METHODS[1]: read(BASE / "manifests/armB_original_waveform.jsonl"),
        METHODS[2]: read(ROOT / "reviewer_control_outputs/s1_tuned_full40_seed1/frozen/tuned_waveform.evaluation.canonical.jsonl"),
        METHODS[3]: read(D / "frozen/bandlimited_4khz.evaluation.r4_selected.jsonl"),
    }
    for method in METHODS:
        assert set(manifests[method]) == expected
        for key, r in grouped[method].items():
            m = manifests[method][key]
            assert r["adv_path"] == m["adv_path"] and r["prompt"] == m["prompt"] and r["target"] == m["target"]
            assert r["sample_rate"] == 24000 and r["samples"] == 240000
    rows = []
    for key in sorted(expected):
        latent = grouped[METHODS[0]][key]
        original, sr = sf.read(path_of(latent["carrier_ref_path"]), dtype="float64")
        assert sr == 24000 and len(original) == 240000
        latent_audio, sr = sf.read(path_of(latent["adv_path"]), dtype="float64")
        latent_snr = snr_db(original, latent_audio)
        for method in METHODS:
            r = grouped[method][key]
            adv, sr = sf.read(path_of(r["adv_path"]), dtype="float64")
            assert sr == 24000 and len(adv) == len(original)
            delta = adv - original
            actual_snr = snr_db(original, adv)
            assert abs(actual_snr-float(r["snr_db"])) < 1e-6, (method,key)
            pure_leak = None
            if method == "bandlimited_4khz":
                own_ref, sr = sf.read(path_of(r["attacker_ref_path"]), dtype="float64")
                assert sr == 24000 and len(own_ref) == len(adv)
                pure_leak = above_cutoff_energy_fraction(adv-own_ref, sr, 4000)
            item = {"example_id":key,"method":method,"carrier":r["carrier"],"target":r["target"],
                    "latent_snr_db":latent_snr,"snr_db":actual_snr,"signed_difference_db":actual_snr-latent_snr,
                    "perturbation_l2":float(np.linalg.norm(delta)),
                    "perturbation_rms":float(np.sqrt(np.mean(delta*delta))),
                    "perturbation_peak":float(np.max(np.abs(delta))),
                    "clipping_rate":float(np.mean(np.abs(adv) >= 32767/32768)),
                    "carrier_referenced_above_4khz_fraction":above_cutoff_energy_fraction(delta,sr,4000),
                    "attack_only_above_4khz_fraction":pure_leak,
                    "waveform_path":str(path_of(r["adv_path"])),
                    "waveform_sha256":hashlib.sha256(path_of(r["adv_path"]).read_bytes()).hexdigest()}
            item.update({f"success_{c}":bool(r["outcomes"][c]["match"]) for c in EVAL_CHANNELS})
            rows.append(item)
    out = BASE / "per_example_records.csv"
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    report_json, report_md = BASE / "baseline_report.json", BASE / "baseline_report.md"
    subprocess.run([sys.executable,"-m","reviewer_controls.build_report","--records",*(str(p) for p in FILES),
                    "--reference-method","latent_codecattack","--output-json",str(report_json),
                    "--output-markdown",str(report_md)],check=True)
    status=json.loads((D / "frozen/bandlimited_4khz.evaluation.r4_selection.json").read_text())
    label="fully matched" if status["fully_matched"] else f"unmatched diagnostic: {status['n_within_1db']}/40 within 1 dB"
    with open(report_md,"a") as f:f.write(f"\nDistortion gate: **{label}**. Every method includes all 40 held-out examples.\n")
    print(f"validated 160 complete records; {label}; wrote {out}")


if __name__ == "__main__":main()
