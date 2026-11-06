#!/usr/bin/env python3
"""Validate and aggregate the four shards of a three-model EoT attack."""

import argparse
import json
from pathlib import Path


CHANNELS = ("clean", "opus64", "opus128", "mp3_64", "mp3_128", "aac_64", "aac_128")
MODELS = {
    "qwen2_MEMBER": "qwen2_audio",
    "omni7b_MEMBER": "qwen25_omni",
    "af3_MEMBER": "audio_flamingo",
    "omni3b_HELDOUT": "qwen25_omni_3b",
}
PREFIXES = {
    "opus": "benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1",
    "mp3": "benchmark_ensemble3_mp3eot_transcribe_eps10.0_jazz_1",
}


def aggregate(results_dir: Path, mode: str, prefix: str | None = None,
              expected_steps: int | None = None,
              require_best_after_warmup: bool = False,
              expected_seed: int | None = None,
              expected_init: str | None = None,
              require_seeded_codec: bool = False) -> dict:
    prefix = prefix or PREFIXES[mode]
    counts = {model: {channel: {"exact": 0, "lenient": 0} for channel in CHANNELS}
              for model in MODELS}
    all_clips = set()
    shards = []
    for shard_id in range(4):
        bundle = results_dir / f"{prefix}_shard{shard_id}of4"
        config = json.loads((bundle / "config.json").read_text())
        expected_mode = "multi_bitrate" if mode == "opus" else "mp3_multibitrate"
        assert config.get("ensemble_channel_mode") == expected_mode, bundle
        assert config.get("ensemble_models") == ["qwen2_audio", "qwen25_omni", "audio_flamingo"], bundle
        if expected_steps is not None:
            assert config.get("attack_steps") == expected_steps, bundle
        if require_best_after_warmup:
            assert config.get("ensemble_best_after_warmup") is True, bundle
        if expected_seed is not None:
            assert config.get("seed") == expected_seed, bundle
            assert config.get("eps") == 10.0 and config.get("prompt_mode") == "transcribe", bundle
        if expected_init is not None:
            assert config.get("rand_init") is (expected_init == "random"), bundle
        if require_seeded_codec:
            assert config.get("seed_codec_per_clip") is True, bundle
        wavs = {p.stem for p in (bundle / "audio").glob("*.wav")}
        assert len(wavs) == 56, f"{bundle}: {len(wavs)}/56 attacked wavs"
        assert not wavs & all_clips, f"{bundle}: duplicate clips"
        all_clips.update(wavs)
        for out_name, model_name in MODELS.items():
            records = {p.parent.name: json.loads(p.read_text())
                       for p in (bundle / out_name).glob("*/record.json")}
            assert set(records) == wavs, f"{bundle}/{out_name}: {len(records)}/56 evaluations"
            for clip, rec in records.items():
                assert rec.get("clip") == clip and rec.get("target_text"), (bundle, clip)
                assert rec.get("target_model") == model_name, (bundle, clip, model_name)
                for channel in CHANNELS:
                    value = rec.get(channel)
                    assert isinstance(value, dict) and isinstance(value.get("exact_match"), bool), (bundle, clip, channel)
                    assert isinstance(value.get("matches_target"), bool), (bundle, clip, channel)
                    counts[out_name][channel]["exact"] += value["exact_match"]
                    counts[out_name][channel]["lenient"] += value["matches_target"]
        shards.append({"bundle": str(bundle), "n": len(wavs)})
    assert len(all_clips) == 224
    percentages = {
        model: {
            channel: {
                "exact_pct": round(100 * values["exact"] / 224, 1),
                "lenient_pct": round(100 * values["lenient"] / 224, 1),
            }
            for channel, values in channels.items()
        }
        for model, channels in counts.items()
    }
    return {"mode": mode, "prefix": prefix, "n": 224,
            "channels": list(CHANNELS), "models": MODELS,
            "counts": counts, "percentages": percentages, "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=PREFIXES)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--require-best-after-warmup", action="store_true")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-init", choices=("zero", "random"))
    parser.add_argument("--require-seeded-codec", action="store_true")
    args = parser.parse_args()
    prefix = args.prefix or PREFIXES[args.mode]
    result = aggregate(args.results_dir, args.mode, prefix, args.expected_steps,
                       args.require_best_after_warmup, args.expected_seed,
                       args.expected_init, args.require_seeded_codec)
    output = args.results_dir / f"{prefix}_AGGREGATED.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    for model, channels in result["counts"].items():
        print(model, " ".join(
            f"{name}:{values['exact']}/224 ({result['percentages'][model][name]['exact_pct']}%)"
            for name, values in channels.items()))


if __name__ == "__main__":
    main()
