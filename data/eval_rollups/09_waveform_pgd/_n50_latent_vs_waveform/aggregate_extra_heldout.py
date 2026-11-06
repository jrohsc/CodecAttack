#!/usr/bin/env python3
"""Aggregate additional held-out victims across the 224-target ensemble shards."""

import argparse
import json
from pathlib import Path

CHANNELS = ("clean", "opus64", "opus128", "mp3_64", "mp3_128", "aac_64", "aac_128")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-init", choices=("zero", "random"))
    parser.add_argument("--require-seeded-codec", action="store_true")
    args = parser.parse_args()

    counts = {channel: 0 for channel in CHANNELS}
    clips = set()
    for shard in range(4):
        bundle = args.results_dir / f"{args.prefix}_shard{shard}of4"
        if args.expected_seed is not None:
            cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
            assert cfg["seed"] == args.expected_seed
            assert cfg["attack_steps"] == 600 and cfg["eps"] == 10.0
            if args.expected_init is not None:
                assert cfg["rand_init"] is (args.expected_init == "random")
            if args.require_seeded_codec:
                assert cfg.get("seed_codec_per_clip") is True
        wavs = {path.stem for path in (bundle / "audio").glob("*.wav")}
        assert len(wavs) == 56, f"{bundle}: {len(wavs)}/56 wavs"
        summary_path = bundle / args.out_name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = summary["results"]
        assert len(rows) == 56, f"{summary_path}: {len(rows)}/56 rows"
        row_clips = {row["clip"] for row in rows}
        assert row_clips == wavs, f"{summary_path}: waveform set mismatch"
        assert not row_clips & clips, f"{summary_path}: duplicate clips"
        clips.update(row_clips)
        for row in rows:
            assert row["target_model"] == args.model and row["target_text"]
            for channel in CHANNELS:
                assert isinstance(row[channel]["exact_match"], bool)
                counts[channel] += row[channel]["exact_match"]
    assert len(clips) == 224
    payload = {
        "attack_prefix": args.prefix,
        "heldout_model": args.model,
        "metric": "strict target substring, as recorded by ensemble_heldout_eval.py",
        "n": 224,
        "counts": counts,
        "percentages": {channel: round(100 * count / 224, 1)
                        for channel, count in counts.items()},
    }
    output = args.results_dir / f"{args.prefix}_{args.out_name}_AGGREGATED.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(payload["percentages"])


if __name__ == "__main__":
    main()
