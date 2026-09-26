#!/usr/bin/env python3
"""Combine validated seed 1, 7, and 42 alternate-ensemble summaries."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
BASE = "benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600"
SEEDS = (1, 7, 42)
CHANNELS = ("clean", "opus64", "opus128", "mp3_64", "mp3_128", "aac_64", "aac_128")
MODELS = (
    ("omni7b_MEMBER", "Qwen2.5-Omni-7B", "Member"),
    ("af3_MEMBER", "Audio Flamingo 3", "Member"),
    ("phi4_MEMBER", "Phi-4-multimodal", "Member"),
    ("qwen2_HELDOUT", "Qwen2-Audio", "Held out"),
    ("kimi_HELDOUT_100tok", "Kimi-Audio-7B-Instruct", "Held out"),
    ("omni3b_HELDOUT", "Qwen2.5-Omni-3B", "Held out"),
)


def main():
    runs = {}
    for seed in SEEDS:
        p = ROOT / f"{BASE}_seed{seed}_SUMMARY.json"
        run = json.loads(p.read_text())
        assert run["n"] == 224 and run["seed"] == seed
        assert set(run["models"]) == {model[0] for model in MODELS}
        runs[str(seed)] = run
    output = {"seeds": SEEDS, "n_per_seed": 224,
              "metric": "case-insensitive exact target substring",
              "seed_note": "seed 1 did not explicitly seed per-clip Opus draws; seeds 7 and 42 did",
              "channels": CHANNELS, "models": {},
              "mean_waveform_snr_db_by_seed": {
                  seed: run["mean_waveform_snr_db"] for seed, run in runs.items()}}
    lines = ["# Alternate three-model Opus-EoT ensemble: seeds 1, 7, 42", "",
             "224 targets per seed. Exact target-substring match (%), mean ± sample SD.",
             "Seed 1 did not explicitly seed per-clip Opus draws; seeds 7 and 42 did.", "",
             "| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label, role in MODELS:
        output["models"][key] = {}
        values = []
        for ch in CHANNELS:
            per_seed = [runs[str(seed)]["models"][key][ch] for seed in SEEDS]
            entry = {"percentages": per_seed,
                     "mean_pct": round(statistics.mean(per_seed), 1),
                     "sample_sd_pct": round(statistics.stdev(per_seed), 1)}
            output["models"][key][ch] = entry
            values.append(f"{entry['mean_pct']:.1f} ± {entry['sample_sd_pct']:.1f}")
        lines.append("| " + " | ".join([role, label] + values) + " |")
    (ROOT / f"{BASE}_RUNS_1_7_42_SUMMARY.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / f"{BASE}_RUNS_1_7_42_SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
