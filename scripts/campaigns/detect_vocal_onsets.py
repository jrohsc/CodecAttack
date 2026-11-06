"""Heuristic vocal-onset detection for music carriers.

For lyrical tracks, singing adds mid-band (200-4000 Hz) harmonic energy
above the instrumental floor. We compute a sliding-window ratio of
harmonic energy in the vocal band vs total harmonic energy, and find
the first time that ratio crosses a persistent threshold.

Outputs a suggested offset per file; human should spot-check by listening.
"""
import json
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MUSIC_ROOT = ROOT / "data" / "music"
OUT_JSON = ROOT / "data" / "trim_offsets_suggested.json"

# Which files are labeled lyrical (need vocal-onset detection)
# vs instrumental (just skip intro 5s).
LYRICAL_HINT = "_lyrical"
SR = 22050  # detection only; trim uses 24k
WIN_S = 1.0  # 1-second windows
VOCAL_BAND_HZ = (200, 4000)
THRESH = 0.55
HOLD_S = 2.0  # must stay above threshold this long


def vocal_onset_s(path: Path) -> float:
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    y_h, _ = librosa.effects.hpss(y)  # harmonic only — drums/transients removed

    n_fft = 2048
    hop = int(sr * WIN_S / 4)
    S = np.abs(librosa.stft(y_h, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask = (freqs >= VOCAL_BAND_HZ[0]) & (freqs <= VOCAL_BAND_HZ[1])
    band_energy = S[band_mask].sum(axis=0)
    total_energy = S.sum(axis=0) + 1e-9
    ratio = band_energy / total_energy

    # Smooth over ~1s
    smooth_w = int(sr / hop)
    if smooth_w > 1:
        kernel = np.ones(smooth_w) / smooth_w
        ratio = np.convolve(ratio, kernel, mode="same")

    times = librosa.frames_to_time(np.arange(len(ratio)), sr=sr, hop_length=hop)
    hold_frames = int(HOLD_S * sr / hop)

    for i in range(len(ratio) - hold_frames):
        if np.all(ratio[i : i + hold_frames] > THRESH):
            return float(times[i])
    return 5.0  # fallback


def main() -> None:
    out = {}
    for bucket in ("ai_generated", "copyrighted"):
        for mp3 in sorted((MUSIC_ROOT / bucket).glob("*.mp3")):
            rel = f"{bucket}/{mp3.name}"
            if LYRICAL_HINT in mp3.stem:
                off = vocal_onset_s(mp3)
                kind = "lyrical"
            else:
                off = 5.0
                kind = "instrumental"
            dur = 10.0 if bucket == "copyrighted" else 20.0
            out[rel] = {
                "kind": kind,
                "start_s": round(off, 2),
                "duration_s": dur,
            }
            print(f"{rel:55s}  {kind:12s}  start={off:6.2f}s  dur={dur}s")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
