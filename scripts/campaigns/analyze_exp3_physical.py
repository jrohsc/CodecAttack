#!/usr/bin/env python3
"""
Analyze exp3 physical OTA recordings vs original adversarial audio.
Compares spectral content, correlation, energy distribution, and channel transfer function.
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

# Paths
EXP_DIR = "results_empirical_ota_IR/3_exp_physical_channel_fir"
ADV_DIR = os.path.join(EXP_DIR, "audio")
REC_DIR = os.path.join(EXP_DIR, "audio_macbook_iphone_recording")
OUT_DIR = os.path.join(EXP_DIR, "acoustic_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 24000  # EnCodec sample rate


def load_wav(path):
    """Load wav file as numpy array."""
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SR, sr)
        data = resample_poly(data, SR // g, sr // g)
    return data


def load_m4a(path):
    """Load m4a via ffmpeg."""
    cmd = ["ffmpeg", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", "1", "-ar", str(SR), "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def compute_psd(x, nperseg=2048):
    """Compute power spectral density."""
    f, psd = scipy_signal.welch(x, fs=SR, nperseg=nperseg)
    return f, psd


def band_energy(x, bands, nperseg=2048):
    """Compute fraction of energy in each frequency band."""
    f, psd = compute_psd(x, nperseg)
    df = f[1] - f[0]
    total = np.sum(psd) * df
    energies = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        energies[name] = np.sum(psd[mask]) * df / total
    return energies


def align_signals(ref, rec, max_shift_samples=None):
    """Align rec to ref using cross-correlation. Returns aligned rec and delay."""
    if max_shift_samples is None:
        max_shift_samples = min(SR * 2, len(rec) // 2)  # up to 2 seconds

    # Use shorter segments for speed
    seg_len = min(len(ref), len(rec), SR * 5)
    ref_seg = ref[:seg_len]
    rec_seg = rec[:min(seg_len + max_shift_samples, len(rec))]

    corr = scipy_signal.correlate(rec_seg, ref_seg, mode='full')
    mid = len(ref_seg) - 1
    search = corr[mid:mid + max_shift_samples]
    delay = np.argmax(np.abs(search))

    # Trim rec to align
    aligned = rec[delay:delay + len(ref)]
    if len(aligned) < len(ref):
        aligned = np.pad(aligned, (0, len(ref) - len(aligned)))

    return aligned, delay


def main():
    # Find pairs
    adv_files = sorted([f for f in os.listdir(ADV_DIR) if f.endswith('.wav')])
    rec_files = sorted([f for f in os.listdir(REC_DIR) if f.endswith('.m4a')])

    pairs = []
    for af in adv_files:
        key = os.path.splitext(af)[0]
        rf = key + ".m4a"
        if rf in rec_files:
            pairs.append((af, rf, key))

    print(f"Found {len(pairs)} matched pairs")

    bands = {
        "sub-bass (0-150Hz)": (0, 150),
        "bass (150-300Hz)": (150, 300),
        "low-mid (300-800Hz)": (300, 800),
        "mid (800-2kHz)": (800, 2000),
        "upper-mid (2-4kHz)": (2000, 4000),
        "presence (4-8kHz)": (4000, 8000),
        "brilliance (8-12kHz)": (8000, 12000),
    }

    all_results = []
    all_tf_ratios = []

    for af, rf, key in pairs:
        print(f"\n{'='*60}")
        print(f"  {key}")
        print(f"{'='*60}")

        adv = load_wav(os.path.join(ADV_DIR, af))
        rec = load_m4a(os.path.join(REC_DIR, rf))

        print(f"  Adv:  {len(adv)/SR:.2f}s, RMS={np.sqrt(np.mean(adv**2)):.4f}")
        print(f"  Rec:  {len(rec)/SR:.2f}s, RMS={np.sqrt(np.mean(rec**2)):.4f}")

        # Align
        rec_aligned, delay_samples = align_signals(adv, rec)
        delay_ms = delay_samples / SR * 1000
        print(f"  Delay: {delay_ms:.1f} ms ({delay_samples} samples)")

        # Trim to same length
        L = min(len(adv), len(rec_aligned))
        adv_t = adv[:L]
        rec_t = rec_aligned[:L]

        # Correlation
        corr = np.corrcoef(adv_t, rec_t)[0, 1]
        print(f"  Correlation: {corr:.4f}")

        # RMS ratio
        rms_adv = np.sqrt(np.mean(adv_t**2))
        rms_rec = np.sqrt(np.mean(rec_t**2))
        gain_db = 20 * np.log10(rms_rec / (rms_adv + 1e-10))
        print(f"  Gain: {gain_db:.1f} dB")

        # SNR (treating adv as signal, difference as noise)
        diff = rec_t - adv_t * (rms_rec / rms_adv)  # gain-normalized
        snr = 10 * np.log10(np.mean(rec_t**2) / (np.mean(diff**2) + 1e-10))
        print(f"  SNR: {snr:.1f} dB")

        # Band energy
        adv_bands = band_energy(adv_t, bands)
        rec_bands = band_energy(rec_t, bands)
        print(f"\n  Band energy distribution:")
        print(f"  {'Band':<25} {'Adv':>8} {'Rec':>8} {'Ratio':>8}")
        for name in bands:
            a = adv_bands[name]
            r = rec_bands[name]
            ratio = r / (a + 1e-10)
            print(f"  {name:<25} {a*100:>7.1f}% {r*100:>7.1f}% {ratio:>7.2f}x")

        # Transfer function (PSD ratio)
        f_adv, psd_adv = compute_psd(adv_t)
        f_rec, psd_rec = compute_psd(rec_t)
        tf_ratio = psd_rec / (psd_adv + 1e-15)
        tf_db = 10 * np.log10(tf_ratio + 1e-15)
        all_tf_ratios.append(tf_db)

        # Short-time correlation
        hop = SR  # 1-second windows
        n_windows = L // hop
        st_corrs = []
        for i in range(n_windows):
            s = i * hop
            e = s + hop
            if e > L:
                break
            c = np.corrcoef(adv_t[s:e], rec_t[s:e])[0, 1]
            st_corrs.append(c)
        mean_st_corr = np.mean(st_corrs) if st_corrs else 0
        print(f"\n  Short-time correlation: mean={mean_st_corr:.4f}, "
              f"std={np.std(st_corrs):.4f}, min={np.min(st_corrs):.4f}, max={np.max(st_corrs):.4f}")

        # Spectral centroid
        sc_adv = np.sum(f_adv * psd_adv) / (np.sum(psd_adv) + 1e-10)
        sc_rec = np.sum(f_rec * psd_rec) / (np.sum(psd_rec) + 1e-10)
        print(f"  Spectral centroid: adv={sc_adv:.0f} Hz, rec={sc_rec:.0f} Hz (shift={sc_rec-sc_adv:+.0f} Hz)")

        all_results.append({
            "key": key,
            "delay_ms": round(delay_ms, 1),
            "correlation": round(corr, 4),
            "gain_db": round(gain_db, 1),
            "snr_db": round(snr, 1),
            "st_corr_mean": round(mean_st_corr, 4),
            "st_corr_std": round(np.std(st_corrs), 4),
            "spectral_centroid_adv_hz": round(sc_adv, 0),
            "spectral_centroid_rec_hz": round(sc_rec, 0),
            "adv_band_energy": {k: round(v, 4) for k, v in adv_bands.items()},
            "rec_band_energy": {k: round(v, 4) for k, v in rec_bands.items()},
        })

        # Per-file spectrogram comparison
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Adv spectrogram
        axes[0].specgram(adv_t, Fs=SR, NFFT=1024, noverlap=768, cmap='magma',
                         vmin=-80, vmax=-20)
        axes[0].set_title(f'{key} — Adversarial Audio')
        axes[0].set_ylabel('Freq (Hz)')
        axes[0].set_ylim(0, 8000)

        # Rec spectrogram
        axes[1].specgram(rec_t, Fs=SR, NFFT=1024, noverlap=768, cmap='magma',
                         vmin=-80, vmax=-20)
        axes[1].set_title(f'{key} — Physical Recording (MacBook→iPhone)')
        axes[1].set_ylabel('Freq (Hz)')
        axes[1].set_ylim(0, 8000)

        # Transfer function
        axes[2].semilogx(f_adv[1:], tf_db[1:], alpha=0.5, linewidth=0.5)
        # Smoothed
        from scipy.ndimage import uniform_filter1d
        tf_smooth = uniform_filter1d(tf_db, size=20)
        axes[2].semilogx(f_adv[1:], tf_smooth[1:], 'r-', linewidth=2, label='Smoothed TF')
        axes[2].set_title(f'{key} — Channel Transfer Function')
        axes[2].set_xlabel('Freq (Hz)')
        axes[2].set_ylabel('Gain (dB)')
        axes[2].set_xlim(50, 12000)
        axes[2].set_ylim(-40, 20)
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{key}.png'), dpi=150)
        plt.close()

    # ============ SUMMARY PLOTS ============

    # 1. Average transfer function across all pairs
    if all_tf_ratios:
        avg_tf = np.mean(all_tf_ratios, axis=0)
        std_tf = np.std(all_tf_ratios, axis=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        f = np.linspace(0, SR/2, len(avg_tf))

        from scipy.ndimage import uniform_filter1d
        avg_smooth = uniform_filter1d(avg_tf, size=20)
        std_smooth = uniform_filter1d(std_tf, size=20)

        ax.fill_between(f[1:], avg_smooth[1:] - std_smooth[1:],
                        avg_smooth[1:] + std_smooth[1:], alpha=0.2, color='blue')
        ax.semilogx(f[1:], avg_smooth[1:], 'b-', linewidth=2, label='Mean ± std')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Mark key regions
        ax.axvspan(0, 300, alpha=0.1, color='red', label='Speaker rolloff (<300Hz)')
        ax.axvspan(2000, 8000, alpha=0.1, color='green', label='Speaker boost (2-8kHz)')

        ax.set_title('Average Physical Channel Transfer Function (exp3, 8 pairs)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_xlim(50, 12000)
        ax.set_ylim(-40, 20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'average_transfer_function.png'), dpi=150)
        plt.close()

    # 2. Comparison with FIR model
    fir_path = "data/empirical_ota/physical_channel_fir.npy"
    if os.path.exists(fir_path):
        fir_taps = np.load(fir_path)
        w, h = scipy_signal.freqz(fir_taps, worN=len(avg_tf), fs=SR)
        fir_db = 20 * np.log10(np.abs(h) + 1e-10)

        fig, ax = plt.subplots(figsize=(12, 5))
        avg_smooth = uniform_filter1d(avg_tf, size=20)

        ax.semilogx(f[1:], avg_smooth[1:], 'b-', linewidth=2, label='Actual channel (this recording)')
        ax.semilogx(w[1:], fir_db[1:], 'r--', linewidth=2, label='FIR model (training)')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('FIR Model vs Actual Physical Channel')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_xlim(50, 12000)
        ax.set_ylim(-40, 20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'fir_vs_actual_channel.png'), dpi=150)
        plt.close()
        print(f"\n\nFIR vs actual channel comparison saved")

    # 3. Summary bar chart: correlation and gain per file
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    keys = [r["key"].replace("calm_1__", "") for r in all_results]
    corrs = [r["correlation"] for r in all_results]
    gains = [r["gain_db"] for r in all_results]

    colors = ['#ef4444' if c < 0.1 else '#f59e0b' if c < 0.3 else '#22c55e' for c in corrs]
    axes[0].barh(keys, corrs, color=colors)
    axes[0].set_xlabel('Correlation')
    axes[0].set_title('Adv vs Physical Recording Correlation')
    axes[0].axvline(0, color='gray', linestyle='--')
    axes[0].set_xlim(-0.5, 0.5)

    axes[1].barh(keys, gains, color='#3b82f6')
    axes[1].set_xlabel('Gain (dB)')
    axes[1].set_title('Physical Channel Gain')
    axes[1].axvline(0, color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'summary_bars.png'), dpi=150)
    plt.close()

    # 4. Band energy comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    band_names_short = [b.split("(")[0].strip() for b in bands.keys()]
    n_bands = len(band_names_short)
    x = np.arange(n_bands)
    width = 0.35

    avg_adv = [np.mean([r["adv_band_energy"][b] for r in all_results]) * 100 for b in bands]
    avg_rec = [np.mean([r["rec_band_energy"][b] for r in all_results]) * 100 for b in bands]

    ax.bar(x - width/2, avg_adv, width, label='Adversarial', color='#3b82f6')
    ax.bar(x + width/2, avg_rec, width, label='Physical recording', color='#ef4444')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names_short, rotation=30, ha='right')
    ax.set_ylabel('Energy (%)')
    ax.set_title('Band Energy Distribution: Adversarial vs Physical Recording')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'band_energy_comparison.png'), dpi=150)
    plt.close()

    # Save JSON summary
    summary = {
        "n_pairs": len(all_results),
        "avg_correlation": round(np.mean(corrs), 4),
        "avg_gain_db": round(np.mean(gains), 1),
        "avg_delay_ms": round(np.mean([r["delay_ms"] for r in all_results]), 1),
        "avg_snr_db": round(np.mean([r["snr_db"] for r in all_results]), 1),
        "avg_st_corr": round(np.mean([r["st_corr_mean"] for r in all_results]), 4),
        "avg_spectral_centroid_shift_hz": round(
            np.mean([r["spectral_centroid_rec_hz"] - r["spectral_centroid_adv_hz"] for r in all_results]), 0),
        "per_file": all_results,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({len(all_results)} pairs)")
    print(f"{'='*60}")
    print(f"  Avg correlation:    {summary['avg_correlation']}")
    print(f"  Avg gain:           {summary['avg_gain_db']} dB")
    print(f"  Avg delay:          {summary['avg_delay_ms']} ms")
    print(f"  Avg SNR:            {summary['avg_snr_db']} dB")
    print(f"  Avg ST correlation: {summary['avg_st_corr']}")
    print(f"  Spectral centroid shift: {summary['avg_spectral_centroid_shift_hz']:+.0f} Hz")
    print(f"\n  Plots saved to: {OUT_DIR}/")
    print(f"  JSON saved to:  {OUT_DIR}/analysis_summary.json")


if __name__ == "__main__":
    main()
