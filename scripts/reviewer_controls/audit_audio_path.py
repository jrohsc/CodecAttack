#!/usr/bin/env python3
"""Check Opus timing and PCM round-trip behavior without a victim model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reviewer_controls.common import codec_roundtrip, write_pcm16_pair


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=24000)
    args = p.parse_args()
    sr = args.sample_rate
    x = np.zeros(2 * sr, dtype=np.float32)
    x[sr // 2] = 0.8
    alignment = {}
    for bitrate in (16, 24, 32, 64, 128):
        y = codec_roundtrip(x, sr, "opus", bitrate)
        alignment[str(bitrate)] = {
            "input_samples": len(x), "output_samples": len(y),
            "peak_lag_samples": int(np.argmax(np.abs(y)) - sr // 2),
            "peak_lag_ms": float((np.argmax(np.abs(y)) - sr // 2) * 1000 / sr),
        }
    rng = np.random.default_rng(1)
    ref = rng.uniform(-0.9, 0.9, sr).astype(np.float32)
    adv = np.clip(ref + 0.001 * rng.standard_normal(sr), -1, 1).astype(np.float32)
    pcm = write_pcm16_pair(args.output.parent / "pcm_probe", "probe", ref, adv, sr)
    payload = {"opus_alignment": alignment, "pcm_roundtrip": pcm,
               "conclusion": "no compensation required" if all(v["peak_lag_samples"] == 0 for v in alignment.values()) else "investigate nonzero lag"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
