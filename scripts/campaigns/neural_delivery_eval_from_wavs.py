"""Stage B of the split neural-codec delivery eval: score pre-round-tripped wavs.

Reads <bundle>/neural_roundtrip/<clip>__<codec>.wav (written by
precompute_neural_roundtrip.py) plus the original attacked wav, runs the target
model on each, and writes neural_delivery_eval_<model>.json in the SAME schema as
neural_codec_delivery_eval.py -- so downstream scorers cannot tell which path
produced a given bundle.

Needs no codec package at all, which is the point: it runs in the model's own env
(e.g. flamingo3, which has no `dac`).

Usage:
  python scripts/neural_delivery_eval_from_wavs.py \
      --bundle-dir results_codec_robust/03b_copyright_bypass/audio_flamingo/eps_1.0_multibitrate \
      --target-model audio_flamingo --prompt-text "$(cat .../system_prompt.txt)"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from neural_codec_delivery_eval import (  # noqa: E402
    BANK_PROMPT, TARGET_SAMPLE_RATE, _target_text, eval_one, load_target_model,
)


def _load_16k(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != TARGET_SAMPLE_RATE:
        t = torchaudio.functional.resample(
            torch.from_numpy(a).unsqueeze(0), sr, TARGET_SAMPLE_RATE)
        a = t.squeeze(0).numpy().astype(np.float32)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--target-model", required=True,
                   choices=["qwen2_audio", "qwen25_omni", "audio_flamingo"])
    p.add_argument("--codecs", default="dac,encodec")
    p.add_argument("--prompt-text", default=BANK_PROMPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    wavs = sorted((args.bundle_dir / "audio").glob("*_attacked.wav"))
    if args.limit:
        wavs = wavs[:args.limit]
    rt_dir = args.bundle_dir / "neural_roundtrip"
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]

    missing = [f"{w.stem.replace('_attacked','')}__{c}.wav" for w in wavs for c in codecs
               if not (rt_dir / f"{w.stem.replace('_attacked','')}__{c}.wav").is_file()]
    if missing:
        print(f"[ncd-wav] ERROR: {len(missing)} round-tripped wav(s) missing from {rt_dir}\n"
              f"          e.g. {missing[:3]}\n"
              f"          run precompute_neural_roundtrip.py first", flush=True)
        sys.exit(3)

    print(f"[ncd-wav] bundle={args.bundle_dir}  model={args.target_model}  "
          f"{len(wavs)} wavs  codecs={codecs}", flush=True)
    model = load_target_model(args.target_model, args.device, torch.bfloat16)

    results = []
    for i, w in enumerate(wavs, 1):
        clip = w.stem.replace("_attacked", "")
        tt = _target_text(w)
        row = {"clip": clip, "target_text": tt}
        row["clean"] = eval_one(model, _load_16k(w), tt, args.prompt_text, args.device)
        for c in codecs:
            row[c] = eval_one(model, _load_16k(rt_dir / f"{clip}__{c}.wav"),
                              tt, args.prompt_text, args.device)
        results.append(row)
        chans = [k for k in ("clean", *codecs) if k in row]
        print(f"[{i}/{len(wavs)}] " + " ".join(f"{c}={row[c]['matches_target']}" for c in chans),
              flush=True)

    out = args.bundle_dir / f"neural_delivery_eval_{args.target_model}.json"
    json.dump({"bundle_dir": str(args.bundle_dir), "target_model": args.target_model,
               "codecs": codecs, "roundtrip_source": str(rt_dir), "results": results},
              open(out, "w"), indent=2)
    roll = {c: sum(r[c]["matches_target"] for r in results if c in r)
            for c in ("clean", *codecs) if any(c in r for r in results)}
    print(f"[ncd-wav] wrote {out}\n[ncd-wav] ASR out of {len(results)}: {roll}", flush=True)


if __name__ == "__main__":
    main()
