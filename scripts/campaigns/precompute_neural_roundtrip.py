"""Stage A of the split neural-codec delivery eval: write the round-tripped wavs.

Why this exists: the flamingo3 env (Audio Flamingo 3, transformers 5.0.0.dev0) has no
`dac` package, and installing the descript-audio-codec stack into it would drag in a
large dependency chain next to a pinned dev transformers. So instead of moving the
model to the codec, we move the audio: this script runs the DAC / EnCodec round-trip
in an env that HAS the codecs and writes the resulting waveforms to disk; then
`neural_delivery_eval_from_wavs.py` scores them in the model's own env.

Round-trip maths is imported verbatim from neural_codec_delivery_eval.py, so the
single-env path and the split path produce bit-identical audio.

Output: <bundle>/neural_roundtrip/<clip>__<codec>.wav at 16 kHz.
Resumable: existing wavs are skipped.

Usage:
  python scripts/precompute_neural_roundtrip.py \
      --bundle-dir results_codec_robust/03b_copyright_bypass/audio_flamingo/eps_1.0_multibitrate \
      --codecs dac,encodec
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from neural_codec_delivery_eval import (  # noqa: E402
    TARGET_SAMPLE_RATE, dac_roundtrip, encodec_roundtrip,
)


def _load_16k(path: Path) -> np.ndarray:
    adv, sr = sf.read(str(path), dtype="float32")
    if adv.ndim > 1:
        adv = adv.mean(1)
    if sr != TARGET_SAMPLE_RATE:
        t = torchaudio.functional.resample(
            torch.from_numpy(adv).unsqueeze(0), sr, TARGET_SAMPLE_RATE)
        adv = t.squeeze(0).numpy().astype(np.float32)
    return adv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--codecs", default="dac,encodec")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--all-wavs", action="store_true",
                   help="Use every WAV in bundle/audio, including ensemble bundles without _attacked suffix")
    args = p.parse_args()

    pattern = "*.wav" if args.all_wavs else "*_attacked.wav"
    wavs = sorted((args.bundle_dir / "audio").glob(pattern))
    if args.limit:
        wavs = wavs[:args.limit]
    out_dir = args.bundle_dir / "neural_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    print(f"[rt] bundle={args.bundle_dir}  {len(wavs)} wavs  codecs={codecs}", flush=True)

    todo = [(w, c) for w in wavs for c in codecs
            if not (out_dir / f"{w.stem.replace('_attacked','')}__{c}.wav").is_file()]
    if not todo:
        print(f"[rt] all {len(wavs)*len(codecs)} round-trips already on disk", flush=True)
        return
    print(f"[rt] {len(todo)} round-trip(s) to compute", flush=True)

    dac = enc = None
    if any(c == "dac" for _, c in todo):
        from attacks.latent_codec_dac import DACWrapper
        dac = DACWrapper(device=args.device); print("[rt] DAC loaded", flush=True)
    if any(c == "encodec" for _, c in todo):
        from attacks.latent_codec import EnCodecWrapper
        enc = EnCodecWrapper(device=args.device); print("[rt] EnCodec loaded", flush=True)

    cache: dict[Path, np.ndarray] = {}
    for i, (w, c) in enumerate(todo, 1):
        if w not in cache:
            cache = {w: _load_16k(w)}          # one clip at a time, keep RAM flat
        adv = cache[w]
        out = dac_roundtrip(dac, adv) if c == "dac" else encodec_roundtrip(enc, adv)
        dst = out_dir / f"{w.stem.replace('_attacked','')}__{c}.wav"
        sf.write(str(dst), out, TARGET_SAMPLE_RATE)
        print(f"[{i}/{len(todo)}] {c:8s} -> {dst.name}", flush=True)

    n = len(list(out_dir.glob("*.wav")))
    print(f"[rt] done: {n} round-tripped wavs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
