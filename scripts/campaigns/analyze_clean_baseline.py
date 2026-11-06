#!/usr/bin/env python3
"""Analyze clean music recording vs original to establish channel baseline."""

import os, sys, subprocess
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MUSIC_FILES, ENCODEC_SAMPLE_RATE

SR = ENCODEC_SAMPLE_RATE  # 24000

def load_m4a(path, sr=SR):
    cmd = ["ffmpeg", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", "1", "-ar", str(sr), "-"]
    result = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()

# Load files
rec = load_m4a("results_spec_ota/clean_mac_iphone_recording_calm-1.m4a")

import torch, torchaudio, librosa
# Load only first 10s to match the attack audio duration
orig, _ = librosa.load(MUSIC_FILES["calm_1"], sr=SR, mono=True, duration=10.0)

print(f"Original: {len(orig)/SR:.2f}s ({len(orig)} samples) at {SR}Hz")
print(f"Recorded: {len(rec)/SR:.2f}s ({len(rec)} samples) at {SR}Hz")
print(f"Orig RMS: {np.sqrt(np.mean(orig**2)):.4f}, max: {np.max(np.abs(orig)):.4f}")
print(f"Rec  RMS: {np.sqrt(np.mean(rec**2)):.4f}, max: {np.max(np.abs(rec)):.4f}")

# Cross-correlation to find delay
max_delay_ms = 500
max_samples = int(max_delay_ms / 1000 * SR)
min_len = min(len(orig), len(rec))
corr = np.correlate(rec[:min_len], orig[:min_len], mode='full')
mid = min_len - 1
search = corr[mid:mid + max_samples]
delay = np.argmax(search)
delay_ms = delay / SR * 1000

# Normalize cross-corr peak
norm = np.sqrt(np.sum(orig[:min_len]**2) * np.sum(rec[:min_len]**2))
xcorr_peak = search[delay] / norm if norm > 0 else 0

print(f"\nDelay: {delay_ms:.1f} ms ({delay} samples)")
print(f"XCorr peak (normalized): {xcorr_peak:.4f}")

# Align and compute metrics
rec_aligned = rec[delay:]
min_len2 = min(len(orig), len(rec_aligned))
o = orig[:min_len2]
r = rec_aligned[:min_len2]

o_rms = np.sqrt(np.mean(o**2))
r_rms = np.sqrt(np.mean(r**2))
gain_db = 20 * np.log10(r_rms / (o_rms + 1e-10))

r_scaled = r * (o_rms / (r_rms + 1e-10))
noise = r_scaled - o
snr = 10 * np.log10(np.mean(o**2) / (np.mean(noise**2) + 1e-10))
correlation = np.corrcoef(o, r_scaled)[0, 1]

print(f"Gain: {gain_db:.1f} dB")
print(f"SNR:  {snr:.1f} dB")
print(f"Pearson correlation: {correlation:.4f}")

# Frequency band analysis
n_fft = 4096
freqs = np.fft.rfftfreq(n_fft, 1.0/SR)
orig_spec = np.abs(np.fft.rfft(o[:n_fft] * np.hanning(n_fft), n=n_fft))
rec_spec = np.abs(np.fft.rfft(r[:n_fft] * np.hanning(n_fft), n=n_fft))

bands = {
    "sub_bass (20-60Hz)": (20, 60),
    "bass (60-250Hz)": (60, 250),
    "low_mid (250-500Hz)": (250, 500),
    "mid (500-2kHz)": (500, 2000),
    "upper_mid (2-4kHz)": (2000, 4000),
    "presence (4-6kHz)": (4000, 6000),
    "brilliance (6-12kHz)": (6000, 12000),
}

print(f"\nFrequency band gains (clean music):")
for name, (lo, hi) in bands.items():
    mask = (freqs >= lo) & (freqs < hi)
    if mask.any():
        orig_e = np.mean(orig_spec[mask]**2)
        rec_e = np.mean(rec_spec[mask]**2)
        bg = 10 * np.log10(rec_e / (orig_e + 1e-20))
        print(f"  {name:25s}: {bg:+6.1f} dB")

print(f"\n{'='*60}")
print(f"COMPARISON: Clean vs Adversarial through OTA channel")
print(f"{'='*60}")
print(f"                        Correlation    SNR       Delay")
print(f"  Clean music:          {correlation:.4f}         {snr:.1f} dB   {delay_ms:.0f} ms")
print(f"  Adversarial (eps=3):  0.0273         -2.9 dB   161 ms (avg)")
print(f"{'='*60}")
if correlation > 0.3:
    print(f"  -> Clean music SURVIVES the channel (corr={correlation:.3f})")
    print(f"  -> The adversarial perturbation is what's being destroyed")
    print(f"  -> Need to make perturbation more robust, not fix the channel")
elif correlation > 0.1:
    print(f"  -> Clean music partially survives (corr={correlation:.3f})")
    print(f"  -> Channel is destructive but not impossible")
else:
    print(f"  -> Clean music also destroyed (corr={correlation:.3f})")
    print(f"  -> Recording setup itself may be the issue (distance? volume?)")
