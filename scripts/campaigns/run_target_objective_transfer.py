"""Inference-only transfer evaluation for the target-objective ablation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from aac_channel import apply_aac_compression, apply_mp3_compression
from demo_ota import apply_opus_compression
from model_registry import load_model
from run_grm_inject import PAPER_PROMPT


CHANNELS = ("direct", "opus32", "opus64", "opus128", "mp3_64", "aac_64")


def read_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, 24000,
        ).squeeze().numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def transform(audio: np.ndarray, channel: str) -> np.ndarray:
    if channel == "direct":
        return audio
    if channel.startswith("opus"):
        return apply_opus_compression(audio, 24000, int(channel[4:]))
    if channel == "mp3_64":
        return apply_mp3_compression(audio, 24000, 64)
    if channel == "aac_64":
        return apply_aac_compression(audio, 24000, 64)
    raise ValueError(channel)


def generate(model, audio: np.ndarray, device: str) -> str:
    wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0).to(device)
    wav = torchaudio.functional.resample(wav, 24000, 16000)
    with torch.no_grad():
        return str(model.generate(wav, prompt=PAPER_PROMPT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True,
                        choices=("audio_flamingo", "qwen25_omni"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-idle-minutes", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.target_model, device=args.device, dtype=torch.bfloat16)
    clean_path = args.audio_dir / "clean.wav"
    while not clean_path.exists():
        time.sleep(args.poll_seconds)
    cache_path = args.out / "clean_cache.state"
    clean_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if len(clean_cache) != len(CHANNELS):
        clean_audio = read_audio(clean_path)
        for channel in CHANNELS:
            if channel not in clean_cache:
                clean_cache[channel] = {
                    "output": generate(model, transform(clean_audio, channel), args.device)
                }
                cache_path.write_text(json.dumps(clean_cache, indent=1))

    idle_since = time.time()
    while True:
        progressed = False
        for src in sorted(args.source.glob("*.json")):
            dst = args.out / src.name
            if dst.exists():
                continue
            meta = json.loads(src.read_text())
            mode = meta["target_mode"]
            attack_path = args.audio_dir / f"{meta['key']}__{mode}__attack.wav"
            if not attack_path.exists():
                continue
            audio = read_audio(attack_path)
            attacked = {
                channel: {"output": generate(model, transform(audio, channel), args.device)}
                for channel in CHANNELS
            }
            record = {
                **{k: meta[k] for k in (
                    "key", "behavior", "target_mode", "target_text",
                    "target_words", "carrier", "prompt",
                    "harmful_request_present_at_inference",
                )},
                "model": args.target_model,
                "source_model": "qwen2_audio",
                "transfer_only": True,
                "clean": clean_cache,
                "grm": attacked,
            }
            tmp = dst.with_suffix(".partial")
            tmp.write_text(json.dumps(record, indent=1))
            tmp.rename(dst)
            progressed = True
            idle_since = time.time()
            count = len(list(args.out.glob("*.json")))
            print(f"[transfer:{args.target_model}] {count}/{args.expected} "
                  f"{meta['key']} {mode}", flush=True)

        count = len(list(args.out.glob("*.json")))
        if count >= args.expected:
            print(f"[transfer:{args.target_model}] complete {count}/{args.expected}",
                  flush=True)
            return
        if not progressed:
            if time.time() - idle_since > args.max_idle_minutes * 60:
                raise SystemExit(f"timed out at {count}/{args.expected}")
            print(f"[transfer:{args.target_model}] waiting at {count}/{args.expected}",
                  flush=True)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
