"""Measure clean-channel carrier SNR for any attack bundle, latent or waveform.

Signal = the carrier as the attack actually saw it (reconstructed with the same
loader the driver used), noise = attacked - carrier. Both arms must be measured
with this one script so the latent-vs-waveform comparison is SNR-matched on a
single convention.

Usage:
    python 0_all_combined/scripts/measure_bundle_snr.py \
        --bundle 0_all_combined/results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed1 \
        --loader load_music
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODEC = ROOT.parent
sys.path.insert(0, str(CODEC))


def _carrier_load_music(path, sr, duration):
    from music_carrier import load_music
    return load_music(str(path), target_sr=sr, duration=duration,
                      normalize=True).squeeze().numpy()


def _carrier_soundfile(path, sr, duration):
    import torch, torchaudio
    wav, src = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.FloatTensor(wav).unsqueeze(0)
    if src != sr:
        t = torchaudio.functional.resample(t, src, sr)
    n = int(duration * sr)
    if t.shape[-1] > n:
        t = t[..., :n]
    elif t.shape[-1] < n:
        t = torch.nn.functional.pad(t, (0, n - t.shape[-1]))
    return t.squeeze().numpy()


LOADERS = {"load_music": _carrier_load_music, "soundfile": _carrier_soundfile}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--loader", default="load_music", choices=sorted(LOADERS))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    load = LOADERS[a.loader]
    wavs = sorted(glob.glob(os.path.join(a.bundle, "**", "*_attacked.wav"),
                            recursive=True))
    snrs = []
    for w in wavs:
        js = w.replace("_attacked.wav", "_summary.json")
        if not os.path.isfile(js):
            continue
        j = json.load(open(js))
        adv, sr = sf.read(w)
        if adv.ndim > 1:
            adv = adv.mean(axis=1)
        car = load(j["wav_path"], sr, j["carrier_duration_s"])
        n = min(len(adv), len(car))
        adv, car = adv[:n], car[:n]
        d = adv - car
        snrs.append(10 * np.log10((np.mean(car ** 2) + 1e-12) /
                                  (np.mean(d ** 2) + 1e-12)))
    if not snrs:
        print(json.dumps({"bundle": a.bundle, "n": 0, "error": "no scored clips"}))
        return
    s = np.array(snrs)
    out = {
        "bundle": a.bundle,
        "loader": a.loader,
        "n": int(len(s)),
        "snr_mean": round(float(s.mean()), 3),
        "snr_sd": round(float(s.std(ddof=1)) if len(s) > 1 else 0.0, 3),
        "snr_min": round(float(s.min()), 3),
        "snr_max": round(float(s.max()), 3),
    }
    print(json.dumps(out, indent=None))


if __name__ == "__main__":
    main()
