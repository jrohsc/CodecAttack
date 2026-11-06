"""Fraction of adversarial-perturbation energy below a set of cutoffs.

Backs the band-limited control's caption claim ("99.5% of its perturbation
energy below 400 Hz ... CodecAttack itself places roughly 42% there"). Written
so the same number can be quoted for any cutoff arm (400 / 800 / 1500 Hz) and
for the unconstrained arms it is compared against.

Two sources of delta, in priority order:

  1. `*_delta16k.npy` -- the raw 16 kHz perturbation the attacker last held.
     Only written when a projection mode is active. Preferred: it is the exact
     tensor that was constrained, with no 24 kHz save + resample in between.
  2. `*_attacked.wav` minus the reloaded carrier, resampled to 16 kHz. Needed
     for the latent and unconstrained-waveform arms, which write no npy.

Method (2) is NOT interchangeable with (1) -- a 24k->16k resample has its own
anti-alias rolloff that inflates the low-band fraction. Never compare an npy
number against a wav-reconstructed one; the script tags every row with the
source it used so mixed comparisons are visible.

Usage:
    python 0_all_combined/scripts/measure_delta_band_energy.py \
        --bundle <bundle_dir> [--cutoffs 400 800 1500] [--loader soundfile]
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

SR16 = 16000


def _carrier_load_music(path, sr, duration):
    from music_carrier import load_music
    return load_music(str(path), target_sr=sr, duration=duration,
                      normalize=True).squeeze().numpy()


def _carrier_soundfile(path, sr, duration):
    import torch
    import torchaudio
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


def band_fracs(delta: np.ndarray, cutoffs, sr=SR16) -> dict:
    """Fraction of |rFFT|^2 at or below each cutoff. Parseval, so this is the
    true energy share, not a filtered-signal RMS ratio."""
    spec = np.abs(np.fft.rfft(delta.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(delta), 1.0 / sr)
    total = spec.sum()
    if total <= 0:
        return {str(c): float("nan") for c in cutoffs}
    return {str(c): float(spec[freqs <= c].sum() / total) for c in cutoffs}


def delta_from_wav(wav_path: str, loader) -> np.ndarray | None:
    """adv - carrier at 16 kHz. Both are brought to 16 kHz BEFORE subtracting so
    the difference is not contaminated by mismatched resampler phase."""
    import torch
    import torchaudio
    js = wav_path.replace("_attacked.wav", "_summary.json")
    if not os.path.isfile(js):
        return None
    j = json.load(open(js))
    adv, sr = sf.read(wav_path)
    if adv.ndim > 1:
        adv = adv.mean(axis=1)
    car = loader(j["wav_path"], sr, j["carrier_duration_s"])
    n = min(len(adv), len(car))
    d = (adv[:n] - car[:n]).astype(np.float32)
    if sr != SR16:
        d = torchaudio.functional.resample(
            torch.from_numpy(d).unsqueeze(0), sr, SR16).squeeze().numpy()
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--cutoffs", nargs="+", type=float,
                   default=[400, 800, 1500])
    p.add_argument("--loader", default="soundfile", choices=sorted(LOADERS))
    p.add_argument("--force-wav", action="store_true",
                   help="Ignore *_delta16k.npy and reconstruct from wavs, so a "
                        "projected arm can be compared like-for-like against "
                        "an arm that has no npy.")
    a = p.parse_args()

    npys = sorted(glob.glob(os.path.join(a.bundle, "**", "*_delta16k.npy"),
                            recursive=True))
    source = "delta16k_npy"
    deltas = []
    if npys and not a.force_wav:
        deltas = [np.load(f) for f in npys]
    else:
        source = "wav_minus_carrier"
        loader = LOADERS[a.loader]
        for w in sorted(glob.glob(os.path.join(a.bundle, "**",
                                               "*_attacked.wav"),
                                  recursive=True)):
            d = delta_from_wav(w, loader)
            if d is not None:
                deltas.append(d)

    if not deltas:
        print(json.dumps({"bundle": a.bundle, "n": 0,
                          "error": "no perturbations found"}))
        return

    rows = [band_fracs(d, a.cutoffs) for d in deltas]
    out = {
        "bundle": a.bundle,
        "source": source,
        "loader": a.loader if source == "wav_minus_carrier" else None,
        "n": len(deltas),
        "linf_max": round(float(max(np.abs(d).max() for d in deltas)), 5),
        "rms_mean": round(float(np.mean([np.sqrt((d ** 2).mean())
                                         for d in deltas])), 5),
    }
    for c in a.cutoffs:
        vals = np.array([r[str(c)] for r in rows]) * 100.0
        out[f"pct_energy_le_{int(c)}hz"] = round(float(vals.mean()), 2)
        out[f"pct_energy_le_{int(c)}hz_sd"] = round(
            float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
