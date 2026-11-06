#!/usr/bin/env python3
"""
Compare MacBook Air speaker vs Marshall Action III speaker OTA recordings.

For each command recorded from both speakers on iPhone 16 Pro, computes:
  1. Frequency response (transfer function relative to digital original)
  2. Per-band energy changes (sub-bass, bass, mid, upper-mid, high)
  3. Correlation with original (after alignment)
  4. Perturbation survival analysis (energy in perturbation vs carrier)
  5. Evidence of noise cancellation / noise suppression

Key question: Does the iPhone apply different processing for the two speakers?
If noise cancellation is destroying adversarial perturbation, we'd see:
  - Reduced energy in non-speech frequency bands
  - Lower correlation with original perturbation
  - Different frequency shaping patterns
"""

import os
import sys
import json
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# Paths
EXP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_empirical_ota_IR", "1_exp_empirical_nonlinearity"
)
DIGITAL_DIR = os.path.join(EXP_DIR, "audio")
MAC_DIR = os.path.join(EXP_DIR, "audio_mac_iphone_record_ota")
MARSHALL_DIR = os.path.join(EXP_DIR, "audio_marshall_iphone_record")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_mac_vs_marshall")

# Also check experiment 10 for Marshall recordings with different attack config
EXP10_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_empirical_ota_IR", "10_exp_robust_marshall_channel", "eps5.0"
)

TARGET_SR = 16000  # Analysis sample rate


def load_audio(path, target_sr=TARGET_SR):
    """Load audio file (WAV or M4A) and resample to target_sr."""
    if path.endswith('.m4a'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
                 '-acodec', 'pcm_f32le', tmp],
                capture_output=True, check=True,
            )
            sr, data = wavfile.read(tmp)
        finally:
            os.unlink(tmp)
        return data.astype(np.float32), sr
    else:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
        if sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data, target_sr


def align_signals(ref, rec, max_delay_ms=200):
    """Align recorded signal to reference using cross-correlation."""
    max_delay = int(max_delay_ms * TARGET_SR / 1000)
    # Use shorter of the two
    min_len = min(len(ref), len(rec))
    ref_seg = ref[:min_len]
    rec_seg = rec[:min_len]

    corr = signal.correlate(rec_seg, ref_seg[:max_delay * 4], mode='full')
    peak = np.argmax(np.abs(corr))
    delay = peak - len(ref_seg[:max_delay * 4]) + 1

    if delay > 0:
        rec_aligned = rec[delay:]
    elif delay < 0:
        rec_aligned = np.pad(rec, (-delay, 0))
    else:
        rec_aligned = rec

    # Trim to same length
    L = min(len(ref), len(rec_aligned))
    return ref[:L], rec_aligned[:L], delay


def compute_transfer_function(ref, rec, sr=TARGET_SR, nperseg=2048):
    """Compute frequency-domain transfer function H(f) = S_yr / S_rr."""
    f, Pxy = signal.csd(rec, ref, fs=sr, nperseg=nperseg)
    f, Pxx = signal.welch(ref, fs=sr, nperseg=nperseg)
    H = Pxy / (Pxx + 1e-10)
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
    phase = np.angle(H)
    return f, magnitude_db, phase


def band_energy(audio, sr, low, high):
    """Compute energy in a frequency band."""
    f, Pxx = signal.welch(audio, fs=sr, nperseg=2048)
    mask = (f >= low) & (f < high)
    return 10 * np.log10(np.sum(Pxx[mask]) + 1e-20)


def analyze_pair(digital_path, recorded_path, label):
    """Analyze a single digital-recorded pair."""
    dig, _ = load_audio(digital_path)
    rec, _ = load_audio(recorded_path)

    dig_a, rec_a, delay = align_signals(dig, rec)

    # Correlation
    corr = np.corrcoef(dig_a, rec_a)[0, 1]

    # Per-band energy
    bands = {
        'sub_bass': (20, 300),
        'bass': (300, 1000),
        'mid': (1000, 3000),
        'upper_mid': (3000, 6000),
        'high': (6000, 8000),
    }

    band_results = {}
    for name, (lo, hi) in bands.items():
        e_dig = band_energy(dig_a, TARGET_SR, lo, hi)
        e_rec = band_energy(rec_a, TARGET_SR, lo, hi)
        band_results[name] = {
            'digital_dB': float(e_dig),
            'recorded_dB': float(e_rec),
            'delta_dB': float(e_rec - e_dig),
        }

    # Transfer function
    f, mag_db, phase = compute_transfer_function(dig_a, rec_a)

    # SNR (signal = correlated part, noise = residual)
    gain = np.dot(dig_a, rec_a) / (np.dot(dig_a, dig_a) + 1e-10)
    residual = rec_a - gain * dig_a
    snr = 10 * np.log10(np.sum((gain * dig_a) ** 2) / (np.sum(residual ** 2) + 1e-10))

    # Envelope correlation (amplitude modulation similarity)
    from scipy.signal import hilbert
    env_dig = np.abs(hilbert(dig_a))
    env_rec = np.abs(hilbert(rec_a))
    env_corr = np.corrcoef(env_dig, env_rec)[0, 1]

    return {
        'label': label,
        'delay_samples': int(delay),
        'delay_ms': float(delay / TARGET_SR * 1000),
        'correlation': float(corr),
        'envelope_correlation': float(env_corr),
        'snr_dB': float(snr),
        'gain': float(gain),
        'bands': band_results,
        'transfer_f': f,
        'transfer_mag': mag_db,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find overlapping commands
    mac_files = {f.replace('.m4a', ''): os.path.join(MAC_DIR, f)
                 for f in os.listdir(MAC_DIR) if f.endswith('.m4a')}
    marshall_files = {f.replace('.m4a', ''): os.path.join(MARSHALL_DIR, f)
                      for f in os.listdir(MARSHALL_DIR) if f.endswith('.m4a')}
    digital_files = {f.replace('.wav', ''): os.path.join(DIGITAL_DIR, f)
                     for f in os.listdir(DIGITAL_DIR) if f.endswith('.wav')}

    common = sorted(set(mac_files.keys()) & set(marshall_files.keys()) & set(digital_files.keys()))
    print(f"Found {len(common)} commands with recordings from both speakers:")
    for c in common:
        print(f"  {c}")

    mac_results = []
    marshall_results = []

    for cmd in common:
        print(f"\n--- {cmd} ---")
        dig_path = digital_files[cmd]

        print(f"  MacBook Air -> iPhone:")
        mac_r = analyze_pair(dig_path, mac_files[cmd], f"mac_{cmd}")
        mac_results.append(mac_r)
        print(f"    delay={mac_r['delay_ms']:.1f}ms, corr={mac_r['correlation']:.4f}, "
              f"env_corr={mac_r['envelope_correlation']:.4f}, SNR={mac_r['snr_dB']:.1f}dB")

        print(f"  Marshall -> iPhone:")
        mar_r = analyze_pair(dig_path, marshall_files[cmd], f"marshall_{cmd}")
        marshall_results.append(mar_r)
        print(f"    delay={mar_r['delay_ms']:.1f}ms, corr={mar_r['correlation']:.4f}, "
              f"env_corr={mar_r['envelope_correlation']:.4f}, SNR={mar_r['snr_dB']:.1f}dB")

    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: MacBook Air vs Marshall Action III (iPhone 16 Pro recording)")
    print("=" * 70)

    for label, results in [("MacBook Air", mac_results), ("Marshall", marshall_results)]:
        corrs = [r['correlation'] for r in results]
        env_corrs = [r['envelope_correlation'] for r in results]
        snrs = [r['snr_dB'] for r in results]
        delays = [r['delay_ms'] for r in results]

        print(f"\n{label} ({len(results)} samples):")
        print(f"  Correlation:      {np.mean(corrs):.4f} +/- {np.std(corrs):.4f}")
        print(f"  Envelope corr:    {np.mean(env_corrs):.4f} +/- {np.std(env_corrs):.4f}")
        print(f"  SNR:              {np.mean(snrs):.1f} +/- {np.std(snrs):.1f} dB")
        print(f"  Delay:            {np.mean(delays):.1f} +/- {np.std(delays):.1f} ms")

        # Per-band averages
        print(f"  Per-band delta (recorded - digital):")
        for band_name in ['sub_bass', 'bass', 'mid', 'upper_mid', 'high']:
            deltas = [r['bands'][band_name]['delta_dB'] for r in results]
            print(f"    {band_name:12s}: {np.mean(deltas):+.1f} +/- {np.std(deltas):.1f} dB")

    # =========================================================================
    # Plots
    # =========================================================================

    # 1. Transfer function comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for r in mac_results:
        axes[0].plot(r['transfer_f'], r['transfer_mag'], alpha=0.4, color='blue')
    axes[0].set_title('MacBook Air -> iPhone 16 Pro (Transfer Function)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_xlim(20, 8000)
    axes[0].set_ylim(-40, 30)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    for r in marshall_results:
        axes[1].plot(r['transfer_f'], r['transfer_mag'], alpha=0.4, color='red')
    axes[1].set_title('Marshall Action III -> iPhone 16 Pro (Transfer Function)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_xlim(20, 8000)
    axes[1].set_ylim(-40, 30)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'transfer_function_comparison.png'), dpi=150)
    print(f"\nSaved: {OUTPUT_DIR}/transfer_function_comparison.png")

    # 2. Per-band energy delta comparison (bar chart)
    band_names = ['sub_bass', 'bass', 'mid', 'upper_mid', 'high']
    band_labels = ['Sub-bass\n20-300Hz', 'Bass\n300-1kHz', 'Mid\n1-3kHz',
                   'Upper-mid\n3-6kHz', 'High\n6-8kHz']

    mac_deltas = [np.mean([r['bands'][b]['delta_dB'] for r in mac_results]) for b in band_names]
    mar_deltas = [np.mean([r['bands'][b]['delta_dB'] for r in marshall_results]) for b in band_names]
    mac_stds = [np.std([r['bands'][b]['delta_dB'] for r in mac_results]) for b in band_names]
    mar_stds = [np.std([r['bands'][b]['delta_dB'] for r in marshall_results]) for b in band_names]

    x = np.arange(len(band_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mac_deltas, width, yerr=mac_stds, label='MacBook Air', color='steelblue', capsize=5)
    ax.bar(x + width/2, mar_deltas, width, yerr=mar_stds, label='Marshall', color='indianred', capsize=5)
    ax.set_ylabel('Energy Change (dB)')
    ax.set_title('Per-Band Energy Change: Digital -> Physical OTA\n(negative = attenuation, positive = boost)')
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'band_energy_comparison.png'), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/band_energy_comparison.png")

    # 3. Scatter: correlation & SNR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mac_corrs = [r['correlation'] for r in mac_results]
    mar_corrs = [r['correlation'] for r in marshall_results]
    mac_snrs = [r['snr_dB'] for r in mac_results]
    mar_snrs = [r['snr_dB'] for r in marshall_results]

    axes[0].bar(['MacBook Air', 'Marshall'],
                [np.mean(mac_corrs), np.mean(mar_corrs)],
                yerr=[np.std(mac_corrs), np.std(mar_corrs)],
                color=['steelblue', 'indianred'], capsize=10)
    axes[0].set_ylabel('Correlation with Original')
    axes[0].set_title('Waveform Correlation')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(['MacBook Air', 'Marshall'],
                [np.mean(mac_snrs), np.mean(mar_snrs)],
                yerr=[np.std(mac_snrs), np.std(mar_snrs)],
                color=['steelblue', 'indianred'], capsize=10)
    axes[1].set_ylabel('SNR (dB)')
    axes[1].set_title('Signal-to-Noise Ratio')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_snr_comparison.png'), dpi=150)
    print(f"Saved: {OUTPUT_DIR}/correlation_snr_comparison.png")

    # Save JSON summary
    summary = {
        'macbook_air': {
            'n_samples': len(mac_results),
            'mean_correlation': float(np.mean(mac_corrs)),
            'mean_envelope_correlation': float(np.mean([r['envelope_correlation'] for r in mac_results])),
            'mean_snr_dB': float(np.mean(mac_snrs)),
            'mean_delay_ms': float(np.mean(delays)),
            'per_band_delta_dB': {b: float(np.mean([r['bands'][b]['delta_dB'] for r in mac_results]))
                                  for b in band_names},
            'per_command': [{k: v for k, v in r.items() if k not in ('transfer_f', 'transfer_mag')}
                           for r in mac_results],
        },
        'marshall': {
            'n_samples': len(marshall_results),
            'mean_correlation': float(np.mean(mar_corrs)),
            'mean_envelope_correlation': float(np.mean([r['envelope_correlation'] for r in marshall_results])),
            'mean_snr_dB': float(np.mean(mar_snrs)),
            'mean_delay_ms': float(np.mean([r['delay_ms'] for r in marshall_results])),
            'per_band_delta_dB': {b: float(np.mean([r['bands'][b]['delta_dB'] for r in marshall_results]))
                                  for b in band_names},
            'per_command': [{k: v for k, v in r.items() if k not in ('transfer_f', 'transfer_mag')}
                           for r in marshall_results],
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/comparison_summary.json")

    # =========================================================================
    # Noise cancellation diagnostic
    # =========================================================================
    print("\n" + "=" * 70)
    print("NOISE CANCELLATION DIAGNOSTIC")
    print("=" * 70)
    print("\nIf iPhone applies noise cancellation, we'd expect:")
    print("  - Non-speech bands (sub-bass, high) more attenuated than speech bands (mid)")
    print("  - Both speakers show similar attenuation pattern (iPhone processing, not speaker)")
    print("  - Adversarial perturbation (non-speech-like) filtered more than carrier (music)")

    mac_speech = np.mean([mac_deltas[2]])  # mid band
    mac_nonspeech = np.mean([mac_deltas[0], mac_deltas[4]])  # sub-bass + high
    mar_speech = np.mean([mar_deltas[2]])
    mar_nonspeech = np.mean([mar_deltas[0], mar_deltas[4]])

    print(f"\n  MacBook: speech-band delta = {mac_speech:+.1f}dB, non-speech = {mac_nonspeech:+.1f}dB")
    print(f"  Marshall: speech-band delta = {mar_speech:+.1f}dB, non-speech = {mar_nonspeech:+.1f}dB")

    if mac_nonspeech < mac_speech - 5 and mar_nonspeech < mar_speech - 5:
        print("\n  >>> STRONG evidence of noise suppression: non-speech bands heavily attenuated")
    elif mac_nonspeech < mac_speech - 2 or mar_nonspeech < mar_speech - 2:
        print("\n  >>> MODERATE evidence: some differential attenuation of non-speech bands")
    else:
        print("\n  >>> WEAK evidence: attenuation pattern looks like speaker frequency response, not NC")


if __name__ == "__main__":
    main()
