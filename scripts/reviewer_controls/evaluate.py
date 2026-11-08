#!/usr/bin/env python3
"""Evaluate delivered PCM files on the exact reviewer-requested metric."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from config import (  # noqa: E402
    MODEL_PATHS, TARGET_SAMPLE_RATE, WHISPER_INJECT_ROOT,
)

# The victim-model package lives in the whisper-inject fork, not in
# codec_attack/models (which only holds base.py). It must precede codec_attack
# on sys.path or `models` resolves to the stub package and the import fails.
# This mirrors scripts/eval_opus_bundle.py, the production evaluator.
sys.path.insert(0, WHISPER_INJECT_ROOT)

from models.qwen2_audio import Qwen2AudioModel  # noqa: E402
from reviewer_controls.common import (  # noqa: E402
    EVAL_CHANNELS, aggregate_rows, append_jsonl, codec_roundtrip, target_match,
)


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--channels", nargs="+", default=list(EVAL_CHANNELS),
                   help="Restrict scoring to these channels. Use the training "
                        "codecs only when scoring the development split, so a "
                        "held-out codec cannot reach a selection decision.")
    p.add_argument("--model", choices=["qwen2_audio", "qwen25_omni", "audio_flamingo", "phi4"], default="qwen2_audio")
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    if not rows:
        raise SystemExit("no manifest rows")
    seen = {}
    for r in rows:
        key = (r["method"], r["example_id"], str(r.get("epsilon")),
               str(r.get("alpha")), str(r.get("seed")))
        if key in seen:
            raise SystemExit(f"duplicate record for {key}; refusing to score it twice")
        seen[key] = True
        for field in ("adv_path", "target", "prompt"):
            if not r.get(field):
                raise SystemExit(f"{r.get('example_id')} is missing '{field}'")
        if not Path(r["adv_path"]).is_file():
            raise SystemExit(f"missing delivered audio: {r['adv_path']}")
    done = set()
    if args.resume and args.output.is_file():
        prior = [json.loads(x) for x in args.output.read_text().splitlines() if x.strip()]
        if any(r.get("evaluation_model", "qwen2_audio") != args.model for r in prior):
            raise ValueError("Cannot resume evaluations from a different model")
        done = {r["record_id"] for r in prior}

    model = None
    for row in rows:
        record_id = ":".join((row["method"], row["example_id"], str(row.get("epsilon")), str(row.get("alpha")), str(row.get("seed"))))
        if record_id in done:
            continue
        if model is None:
            from model_registry import load_model
            model = load_model(args.model, device=args.device, dtype=torch.bfloat16)
            model.model.requires_grad_(False)
        delivered, sr = sf.read(row["adv_path"], dtype="float32")
        if delivered.ndim > 1:
            delivered = delivered.mean(axis=1)
        outcomes = {}
        t0 = time.time()
        for channel in args.channels:
            if channel == "clean":
                transformed = delivered
            else:
                codec, bitrate = channel.split("_", 1) if "_" in channel else ("opus", channel[4:])
                transformed = codec_roundtrip(delivered, sr, codec, int(bitrate))
            audio = torch.from_numpy(np.ascontiguousarray(transformed)).unsqueeze(0)
            if sr != TARGET_SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, TARGET_SAMPLE_RATE)
            with torch.no_grad():
                output = str(model.generate(audio.to(args.device), prompt=row["prompt"]))
            outcomes[channel] = {"match": target_match(row["target"], output), "output": output}
        result = {**row, "evaluation_model": args.model, "record_id": record_id, "outcomes": outcomes, "evaluation_input": "reloaded delivered PCM_16", "eval_elapsed_s": time.time() - t0}
        append_jsonl(args.output, result)
        print(record_id, " ".join(f"{c}={'Y' if outcomes[c]['match'] else 'n'}" for c in args.channels), flush=True)

    all_rows = [json.loads(x) for x in args.output.read_text().splitlines() if x.strip()]
    summary_path = args.summary or args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(aggregate_rows(all_rows), indent=2))
    print(f"wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
