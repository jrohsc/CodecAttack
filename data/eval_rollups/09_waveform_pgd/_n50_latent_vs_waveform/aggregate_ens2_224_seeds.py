#!/usr/bin/env python3
"""Validate and summarize the saved 224-target Omni-7B + AF3 seed runs."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results"
CHANNELS = ("clean", "opus64", "opus128", "mp3_64", "mp3_128", "aac_64", "aac_128")
MODELS = {
    "omni7b_MEMBER_allcodecs": "qwen25_omni",
    "af3_MEMBER_allcodecs": "audio_flamingo",
    "qwen2_HELDOUT_allcodecs": "qwen2_audio",
    "omni3b_HELDOUT_allcodecs": "qwen25_omni_3b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-only", action="store_true")
    args = parser.parse_args()
    models = {name: model for name, model in MODELS.items()
              if not args.heldout_only or "HELDOUT" in name}
    output = {"members": ["qwen25_omni", "audio_flamingo"], "attack_steps": 300,
              "epsilon": 10.0, "training_codec": "Opus multi-bitrate",
              "metric": "case-insensitive exact target substring", "seeds": {}}
    for seed in (1, 7, 42):
        bundle = ROOT / f"benchmark_ens2_omni_af3_opuseot_transcribe_eps10.0_jazz_1_seed{seed}"
        cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        assert cfg["ensemble_models"] == output["members"]
        assert cfg["attack_steps"] == output["attack_steps"] and cfg["seed"] == seed
        assert cfg["eps"] == 10.0 and cfg["rand_init"] is True
        assert cfg["ensemble_channel_mode"] == "multi_bitrate"
        wavs = {path.stem for path in (bundle / "audio").glob("*.wav")}
        assert len(wavs) == 224
        output["seeds"][str(seed)] = {}
        for out_name, model in models.items():
            path = bundle / out_name / "summary.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data["results"]
            assert len(rows) == 224 and {row["clip"] for row in rows} == wavs
            assert all(row["target_model"] == model and row["target_text"] for row in rows)
            counts = {ch: sum(row[ch]["exact_match"] for row in rows) for ch in CHANNELS}
            output["seeds"][str(seed)][model] = {
                "role": "member" if "MEMBER" in out_name else "held_out",
                "n": 224, "counts": counts,
                "percentages": {ch: round(100 * count / 224, 1)
                                for ch, count in counts.items()},
            }
    suffix = "HELDOUT_SEEDS_AGGREGATED" if args.heldout_only else "SEEDS_AGGREGATED"
    path = ROOT / f"benchmark_ens2_omni_af3_opuseot_transcribe_eps10.0_jazz_1_{suffix}.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
