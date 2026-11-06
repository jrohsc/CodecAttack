#!/usr/bin/env python3
"""
Acoustic analysis: compare original adversarial audio vs physical OTA recordings.
Finds patterns in the MacBook Air speaker → iPhone 16 Pro mic channel.

Analyzes:
  1. Time delay (cross-correlation)
  2. Amplitude/gain change
  3. Frequency response (spectral comparison)
  4. SNR / correlation
  5. Spectral centroid shift
"""

import os
import sys
import json
import subprocess
import numpy as np
import soundfile as sf


def load_wav(path, target_sr=24000):
    """Load wav file at target sample rate."""
    if path.endswith(".m4a"):
        cmd = [
            "ffmpeg", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(target_sr),
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
        return np.frombuffer(result.stdout, dtype=np.float32).copy()
    else:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            import torchaudio
            import torch
            wav_t = torch.from_numpy(wav).unsqueeze(0).float()
            wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
            wav = wav_t.squeeze(0).numpy()
        return wav.astype(np.float32)


def find_delay(original, recorded, sr=24000, max_delay_ms=200):
    """Find time delay via cross-correlation."""
    max_delay_samples = int(max_delay_ms / 1000 * sr)
    # Truncate to shorter length
    min_len = min(len(original), len(recorded))
    orig = original[:min_len]
    rec = recorded[:min_len]

    # Cross-correlation
    corr = np.correlate(rec, orig[:min_len], mode='full')
    # Only look at positive delays (recording is delayed)
    mid = len(orig) - 1
    search_range = corr[mid:mid + max_delay_samples]
    delay_samples = np.argmax(search_range)
    delay_ms = delay_samples / sr * 1000
    peak_corr = search_range[delay_samples]

    # Normalize correlation
    norm = np.sqrt(np.sum(orig**2) * np.sum(rec**2))
    if norm > 0:
        peak_corr_norm = peak_corr / norm
    else:
        peak_corr_norm = 0.0

    return delay_samples, delay_ms, peak_corr_norm


def compute_snr(original, recorded, delay_samples, sr=24000):
    """Compute SNR after aligning signals."""
    # Align
    if delay_samples > 0:
        rec_aligned = recorded[delay_samples:]
    else:
        rec_aligned = recorded

    min_len = min(len(original), len(rec_aligned))
    orig = original[:min_len]
    rec = rec_aligned[:min_len]

    # Scale recorded to match original RMS
    orig_rms = np.sqrt(np.mean(orig**2))
    rec_rms = np.sqrt(np.mean(rec**2))
    if rec_rms > 0:
        gain_db = 20 * np.log10(rec_rms / orig_rms) if orig_rms > 0 else 0
        rec_scaled = rec * (orig_rms / rec_rms)
    else:
        gain_db = -np.inf
        rec_scaled = rec

    # SNR
    noise = rec_scaled - orig
    signal_power = np.mean(orig**2)
    noise_power = np.mean(noise**2)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = np.inf

    # Correlation
    corr = np.corrcoef(orig, rec_scaled)[0, 1]

    return snr_db, gain_db, corr


def spectral_analysis(original, recorded, delay_samples, sr=24000, n_fft=2048):
    """Compare frequency content before/after OTA channel."""
    if delay_samples > 0:
        rec_aligned = recorded[delay_samples:]
    else:
        rec_aligned = recorded

    min_len = min(len(original), len(rec_aligned))
    orig = original[:min_len]
    rec = rec_aligned[:min_len]

    # Compute magnitude spectra
    orig_spec = np.abs(np.fft.rfft(orig * np.hanning(min_len), n=n_fft))
    rec_spec = np.abs(np.fft.rfft(rec * np.hanning(min_len), n=n_fft))

    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)

    # Frequency response (ratio in dB)
    with np.errstate(divide='ignore', invalid='ignore'):
        freq_response_db = 20 * np.log10(rec_spec / (orig_spec + 1e-10))

    # Band analysis
    bands = {
        "sub_bass (20-60Hz)": (20, 60),
        "bass (60-250Hz)": (60, 250),
        "low_mid (250-500Hz)": (250, 500),
        "mid (500-2kHz)": (500, 2000),
        "upper_mid (2-4kHz)": (2000, 4000),
        "presence (4-6kHz)": (4000, 6000),
        "brilliance (6-12kHz)": (6000, 12000),
    }

    band_analysis = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            orig_energy = np.mean(orig_spec[mask]**2)
            rec_energy = np.mean(rec_spec[mask]**2)
            if orig_energy > 0:
                band_gain_db = 10 * np.log10(rec_energy / orig_energy)
            else:
                band_gain_db = 0.0
            band_analysis[name] = round(band_gain_db, 1)

    # Spectral centroid
    orig_centroid = np.sum(freqs * orig_spec**2) / (np.sum(orig_spec**2) + 1e-10)
    rec_centroid = np.sum(freqs * rec_spec**2) / (np.sum(rec_spec**2) + 1e-10)

    return band_analysis, round(orig_centroid, 0), round(rec_centroid, 0)


def main():
    original_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "results_spec_ota/benchmark_qwen2_audio_qa_eps3.0_calm_1_spec_ota_m3_s20_saycan/audio"
    recorded_dir = sys.argv[2] if len(sys.argv) > 2 else \
        "results_spec_ota/benchmark_qwen2_audio_qa_eps3.0_calm_1_spec_ota_m3_s20_saycan/audio_direct_recording_mac_iphone"

    sr = 24000

    # Find matching pairs
    recorded_files = sorted([f for f in os.listdir(recorded_dir) if f.endswith(".m4a")])
    pairs = []
    for rec_file in recorded_files:
        base = os.path.splitext(rec_file)[0]
        orig_file = base + ".wav"
        orig_path = os.path.join(original_dir, orig_file)
        if os.path.exists(orig_path):
            pairs.append((orig_path, os.path.join(recorded_dir, rec_file), base))

    print(f"Found {len(pairs)} matching pairs\n")
    print("=" * 80)

    all_delays = []
    all_gains = []
    all_snrs = []
    all_corrs = []
    all_bands = {}
    all_results = []

    for orig_path, rec_path, name in pairs:
        orig = load_wav(orig_path, sr)
        rec = load_wav(rec_path, sr)

        delay_samples, delay_ms, xcorr = find_delay(orig, rec, sr)
        snr_db, gain_db, corr = compute_snr(orig, rec, delay_samples, sr)
        bands, orig_centroid, rec_centroid = spectral_analysis(orig, rec, delay_samples, sr)

        all_delays.append(delay_ms)
        all_gains.append(gain_db)
        all_snrs.append(snr_db)
        all_corrs.append(corr)

        for band_name, band_gain in bands.items():
            if band_name not in all_bands:
                all_bands[band_name] = []
            all_bands[band_name].append(band_gain)

        result = {
            "name": name,
            "delay_ms": float(round(delay_ms, 2)),
            "gain_db": float(round(gain_db, 2)),
            "snr_db": float(round(snr_db, 2)),
            "correlation": float(round(corr, 4)),
            "xcorr_peak": float(round(xcorr, 4)),
            "orig_centroid_hz": float(orig_centroid),
            "rec_centroid_hz": float(rec_centroid),
            "band_gains_db": {k: float(v) for k, v in bands.items()},
            "orig_duration_s": float(round(len(orig) / sr, 2)),
            "rec_duration_s": float(round(len(rec) / sr, 2)),
        }
        all_results.append(result)

        print(f"\n{name}")
        print(f"  Delay:       {delay_ms:.1f} ms ({delay_samples} samples)")
        print(f"  Gain:        {gain_db:.1f} dB")
        print(f"  SNR:         {snr_db:.1f} dB")
        print(f"  Correlation: {corr:.4f}")
        print(f"  XCorr peak:  {xcorr:.4f}")
        print(f"  Centroid:    {orig_centroid:.0f} Hz → {rec_centroid:.0f} Hz "
              f"({rec_centroid - orig_centroid:+.0f} Hz)")
        print(f"  Duration:    {len(orig)/sr:.2f}s (orig) vs {len(rec)/sr:.2f}s (rec)")
        print(f"  Band gains:")
        for band_name, band_gain in bands.items():
            bar = "+" * max(0, int(band_gain / 2)) if band_gain > 0 else "-" * max(0, int(-band_gain / 2))
            print(f"    {band_name:25s}: {band_gain:+6.1f} dB  {bar}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL PAIRS")
    print("=" * 80)
    print(f"  Delay:       {np.mean(all_delays):.1f} ± {np.std(all_delays):.1f} ms")
    print(f"  Gain:        {np.mean(all_gains):.1f} ± {np.std(all_gains):.1f} dB")
    print(f"  SNR:         {np.mean(all_snrs):.1f} ± {np.std(all_snrs):.1f} dB")
    print(f"  Correlation: {np.mean(all_corrs):.4f} ± {np.std(all_corrs):.4f}")
    print(f"\n  Average frequency band gains (recorded vs original):")
    for band_name, gains in all_bands.items():
        avg = np.mean(gains)
        std = np.std(gains)
        bar = "+" * max(0, int(avg / 2)) if avg > 0 else "-" * max(0, int(-avg / 2))
        print(f"    {band_name:25s}: {avg:+6.1f} ± {std:.1f} dB  {bar}")

    # Determinism check
    print(f"\n  Delay spread (max-min): {max(all_delays) - min(all_delays):.1f} ms")
    print(f"  Gain spread (max-min):  {max(all_gains) - min(all_gains):.1f} dB")
    print(f"  Corr spread (max-min):  {max(all_corrs) - min(all_corrs):.4f}")

    # Save JSON
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return super().default(obj)

    out_path = os.path.join(recorded_dir, "channel_analysis.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "delay_ms": {"mean": round(np.mean(all_delays), 2), "std": round(np.std(all_delays), 2)},
                "gain_db": {"mean": round(np.mean(all_gains), 2), "std": round(np.std(all_gains), 2)},
                "snr_db": {"mean": round(np.mean(all_snrs), 2), "std": round(np.std(all_snrs), 2)},
                "correlation": {"mean": round(np.mean(all_corrs), 4), "std": round(np.std(all_corrs), 4)},
                "band_gains_db": {k: {"mean": float(round(np.mean(v), 1)), "std": float(round(np.std(v), 1))}
                                  for k, v in all_bands.items()},
            },
            "per_file": all_results,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
