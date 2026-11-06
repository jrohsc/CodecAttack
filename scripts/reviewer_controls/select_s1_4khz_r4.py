#!/usr/bin/env python3
"""Select five repair WAVs by saved/reloaded SNR alone; preserve the prior canonical set."""
import json
from pathlib import Path

import soundfile as sf

from reviewer_controls.common import snr_db
from reviewer_controls.finish_s1_4khz import BASE, CANON, D, read


def main():
    attack_root = D / "s1/evaluation/bandlimited_4khz"
    repair = attack_root / "eps0.166_ar0.02_steps1000_seed1_snr_correction_r4/manifest.jsonl"
    if not repair.exists(): raise SystemExit("round 4 not complete")
    current, proposed = read(CANON), read(repair)
    reference = read(BASE / "manifests/armA_latent.jsonl")
    expected = set((D / "frozen/bandlimited_4khz.evaluation.bad_r4.txt").read_text().splitlines())
    assert len(current) == 40 and set(proposed) == expected and len(expected) == 5
    attempts = {key: [("prior", current[key])] for key in expected}
    for manifest in sorted(attack_root.glob("*/manifest.jsonl")):
        for row in map(json.loads, open(manifest)):
            if row["example_id"] in expected:
                attempts[row["example_id"]].append((manifest.parent.name, row))
    selection = []
    for key in sorted(expected):
        original, sr = sf.read(reference[key]["carrier_ref_path"], dtype="float64")
        assert sr == 24000
        latent_path = Path(reference[key]["adv_path"])
        if not latent_path.is_absolute(): latent_path = BASE.parent.parent / latent_path
        latent, sr = sf.read(latent_path, dtype="float64")
        target = snr_db(original, latent)
        candidate = []
        for name, row in attempts[key]:
            wav, sr = sf.read(row["adv_path"], dtype="float64")
            assert sr == 24000 and len(wav) == len(original)
            got = snr_db(original, wav)
            candidate.append((abs(got-target), name, got, row))
        best = min(candidate, key=lambda z: (z[0], z[1] != "prior"))
        current[key] = best[3]
        selection.append({"example_id": key, "latent_snr_db": target,
                          "attempts": [{"run":name,"snr_db":got,"epsilon":row["epsilon"],
                                        "residual_db":got-target} for _,name,got,row in candidate],
                          "selected_run": best[1], "selected_snr_db": best[2],
                          "selected_residual_db": best[2]-target})
    frozen = D / "frozen"
    out = frozen / "bandlimited_4khz.evaluation.r4_selected.jsonl"
    out.write_text("".join(json.dumps(current[k])+"\n" for k in sorted(current)))
    status = {"selection_rule": "minimum absolute carrier-referenced SNR residual on saved/reloaded WAV across recorded distortion calibrations, only for the five previously failing IDs; no ASR outcomes read",
              "n": 40, "n_repaired": len(selection), "repairs": selection,
              "n_within_1db": None}
    all_ref = {k: r["snr_db"] for k, r in reference.items()}
    status["n_within_1db"] = sum(abs(row["snr_db"] - all_ref[k]) <= 1 for k,row in current.items())
    status["fully_matched"] = status["n_within_1db"] == 40
    (frozen / "bandlimited_4khz.evaluation.r4_selection.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
