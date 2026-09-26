#!/usr/bin/env python3
"""Validate and summarize a 224-target alternate ensemble seed."""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
BASE = "benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600"
CHANNELS = ("clean", "opus64", "opus128", "mp3_64", "mp3_128", "aac_64", "aac_128")
MODELS = (
    ("omni7b_MEMBER", "Qwen2.5-Omni-7B", "Member", "qwen25_omni"),
    ("af3_MEMBER", "Audio Flamingo 3", "Member", "audio_flamingo"),
    ("phi4_MEMBER", "Phi-4-multimodal", "Member", "phi4"),
    ("qwen2_HELDOUT", "Qwen2-Audio", "Held out", "qwen2_audio"),
    ("kimi_HELDOUT_100tok", "Kimi-Audio-7B-Instruct", "Held out", "kimi_audio"),
    ("omni3b_HELDOUT", "Qwen2.5-Omni-3B", "Held out", "qwen25_omni_3b"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 7, 42), default=1)
    args = parser.parse_args()
    prefix = f"{BASE}_seed{args.seed}"
    snrs = []
    for shard in range(4):
        bundle = ROOT / f"{prefix}_shard{shard}of4"
        cfg = json.loads((bundle / "config.json").read_text())
        assert cfg["ensemble_models"] == ["qwen25_omni", "audio_flamingo", "phi4"]
        assert cfg["target_model"] == "qwen25_omni"
        assert cfg["ensemble_channel_mode"] == "multi_bitrate"
        assert cfg["ensemble_best_after_warmup"] is True
        assert cfg["seed"] == args.seed and cfg["rand_init"] is False
        if args.seed != 1:
            assert cfg["seed_codec_per_clip"] is True
        records = list((bundle / "qwen25_omni").glob("*/record.json"))
        assert len(records) == 56, (bundle, len(records))
        snrs.extend(float(json.loads(p.read_text())["direct"]["snr_db"])
                    for p in records)
    assert len(snrs) == 224
    output = {"prefix": prefix, "n": 224, "seed": args.seed,
              "members": [row[3] for row in MODELS[:3]],
              "held_out": [row[3] for row in MODELS[3:]],
              "channels": CHANNELS, "mean_waveform_snr_db": round(statistics.mean(snrs), 3),
              "models": {}}
    lines = [f"# Alternate 3-model Opus-EoT ensemble (seed {args.seed})", "",
             "224 robot-action target strings; epsilon=10, 600 steps, zero initialization.",
             f"Mean waveform SNR: {output['mean_waveform_snr_db']:.2f} dB.", "",
             "| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label, role, model in MODELS:
        p = ROOT / f"{prefix}_{key}_AGGREGATED.json"
        data = json.loads(p.read_text())
        assert data["n"] == 224 and data["heldout_model"] == model
        output["models"][key] = data["percentages"]
        lines.append("| " + " | ".join([role, label] +
                    [f"{data['percentages'][ch]:.1f}" for ch in CHANNELS]) + " |")
    (ROOT / f"{prefix}_SUMMARY.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / f"{prefix}_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(ROOT / f"{prefix}_SUMMARY.md")


if __name__ == "__main__":
    main()
