#!/usr/bin/env python3
"""Validate and summarize the four shards of the two-model waveform run."""

import json
from pathlib import Path

PREFIX = "benchmark_ens2_waveform_transcribe_eps0.075_jazz_1"
CHANNELS = ("clean", "opus64", "opus128", "mp3_64", "mp3_128", "aac_64", "aac_128")
MODELS = {
    "qwen2_MEMBER": "qwen2_audio",
    "omni7b_MEMBER": "qwen25_omni",
    "af3_HELDOUT": "audio_flamingo",
    "omni3b_HELDOUT": "qwen25_omni_3b",
}


def main():
    root = Path("results")
    all_clips = set()
    counts = {model: {c: {"exact": 0, "lenient": 0, "wer_lt_05": 0}
                      for c in CHANNELS} for model in MODELS}
    snrs = []
    shards = []
    for shard in range(4):
        bundle = root / f"{PREFIX}_shard{shard}of4"
        config = json.loads((bundle / "config.json").read_text())
        expected = {
            "ensemble_domain": "waveform",
            "ensemble_models": ["qwen2_audio", "qwen25_omni"],
            "ensemble_channel_mode": "none",
            "ensemble_best_after_warmup": True,
            "ensemble_warmup": 50,
            "attack_steps": 300,
            "eps": 0.075,
            "alpha": 0.015,
        }
        for key, value in expected.items():
            assert config.get(key) == value, (bundle, key, config.get(key), value)
        wavs = {path.stem for path in (bundle / "audio").glob("*.wav")}
        assert len(wavs) == 56, f"{bundle}: {len(wavs)}/56 WAVs"
        assert not wavs & all_clips, f"{bundle}: duplicate clip"
        all_clips.update(wavs)
        primary = {p.parent.name: json.loads(p.read_text())
                   for p in (bundle / "qwen2_audio").glob("*/record.json")}
        assert set(primary) == wavs, f"{bundle}: incomplete attack records"
        snrs.extend(float(r["direct"]["snr_db"]) for r in primary.values())
        for out_name, victim in MODELS.items():
            records = {p.parent.name: json.loads(p.read_text())
                       for p in (bundle / out_name).glob("*/record.json")}
            assert set(records) == wavs, f"{bundle}/{out_name}: {len(records)}/56 evals"
            for clip, rec in records.items():
                assert rec.get("clip") == clip and rec.get("target_text"), (bundle, clip)
                assert rec.get("target_model") == victim, (bundle, clip, victim)
                for channel in CHANNELS:
                    value = rec.get(channel)
                    assert isinstance(value, dict), (bundle, clip, channel)
                    assert isinstance(value.get("exact_match"), bool), (bundle, clip, channel)
                    assert isinstance(value.get("matches_target"), bool), (bundle, clip, channel)
                    counts[out_name][channel]["exact"] += value["exact_match"]
                    counts[out_name][channel]["lenient"] += value["matches_target"]
                    counts[out_name][channel]["wer_lt_05"] += (
                        value.get("wer") is not None and value["wer"] < 0.5)
        shards.append({"bundle": str(bundle), "n": len(wavs)})
    assert len(all_clips) == 224 and len(snrs) == 224
    percentages = {
        model: {c: {f"{metric}_pct": round(100 * count / 224, 1)
                     for metric, count in values.items()}
                for c, values in channels.items()}
        for model, channels in counts.items()
    }
    output = root / f"{PREFIX}_AGGREGATED.json"
    result = {"n": 224, "attack": "two-model waveform, no codec EoT",
              "models": MODELS, "channels": CHANNELS,
              "mean_attack_snr_db": round(sum(snrs) / len(snrs), 2),
              "counts": counts, "percentages": percentages, "shards": shards}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print("mean attack SNR:", result["mean_attack_snr_db"], "dB")
    for model, channels in counts.items():
        print(model, " ".join(
            f"{c}:{v['exact']}/224 ({percentages[model][c]['exact_pct']}%)"
            for c, v in channels.items()))


if __name__ == "__main__":
    main()
