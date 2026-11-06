"""Compare digital vs physical OTA audio for bp400-700 to understand why speech survives."""
import os, sys, subprocess, tempfile, json
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import correlate

sys.path.insert(0, "${REPO_ROOT}")
sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from config import TARGET_SAMPLE_RATE

BASE = "${REPO_ROOT}/results_empirical_analysis_noise_cancellation/27_exp_narrow_bandpass_marshall"
OUT_DIR = os.path.join(BASE, "bp400_700_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# All bandpass directories with their digital + physical files
BANDS = {
    "bp300-600": {
        "digital": "bp300-600/attack_calm_1_20260408_220657_adversarial.wav",
        "physical": "bp300-600/audio_marshall_iphone_record_bp300-600.m4a",
        "original": "bp300-600/attack_calm_1_20260408_220657_original.wav",
    },
    "bp400-700": {
        "digital": "bp400_700/adversarial_16k.wav",
        "physical": "bp400_700/marshall_recorded.m4a",
        "original": "bp400_700/original_24k.wav",
    },
    "bp500-800": {
        "digital": "bp500_800/attack_calm_1_20260409_025356_adversarial_no-ecc.wav",
        "physical": "bp500_800/audio_marshall_iphone_record_bp500-800_no-ecc.m4a",
        "original": "bp500_800/attack_calm_1_20260409_025356_original_no-ecc.wav",
    },
    "bp800-1200": {
        "digital": None,  # check later
        "physical": "bp800_1200/audio_marshall_iphone_recorded.m4a",
        "original": None,
    },
}


def load_audio(path, target_sr=TARGET_SAMPLE_RATE):
    if path.endswith('.m4a'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(['ffmpeg', '-y', '-i', os.path.join(BASE, path), '-ac', '1',
                            '-ar', str(target_sr), '-acodec', 'pcm_f32le', tmp],
                           capture_output=True, check=True)
            data, sr = sf.read(tmp, dtype='float32')
            return data
        finally:
            os.unlink(tmp)
    else:
        data, sr = sf.read(os.path.join(BASE, path), dtype='float32')
        if sr != target_sr:
            import torchaudio
            t = torch.FloatTensor(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            return t.squeeze().numpy()
        return data


def compute_spectrogram(audio, sr, nperseg=2048, noverlap=1536):
    f, t, Sxx = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
                                    window='hann', scaling='density')
    return f, t, Sxx


def compute_band_energy(audio, sr, bands):
    """Compute energy in frequency bands."""
    nperseg = 2048
    f, t, Sxx = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=1536,
                                    window='hann', scaling='density')
    results = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        results[name] = np.mean(Sxx[mask, :])
    return results


def align_audios(digital, physical, sr):
    """Align physical to digital using cross-correlation."""
    # Use a chunk for alignment
    chunk_len = min(len(digital), len(physical), sr * 5)
    corr = correlate(physical[:chunk_len], digital[:chunk_len], mode='full')
    lag = np.argmax(np.abs(corr)) - (chunk_len - 1)

    if lag > 0:
        physical_aligned = physical[lag:]
    else:
        physical_aligned = np.pad(physical, (-lag, 0))[:len(physical)]

    # Trim to same length
    min_len = min(len(digital), len(physical_aligned))
    return digital[:min_len], physical_aligned[:min_len], lag


def main():
    sr = TARGET_SAMPLE_RATE

    freq_bands = {
        "0-300Hz": (0, 300),
        "300-400Hz": (300, 400),
        "400-500Hz": (400, 500),
        "500-600Hz": (500, 600),
        "600-700Hz": (600, 700),
        "700-800Hz": (700, 800),
        "800-1200Hz": (800, 1200),
        "1200-2000Hz": (1200, 2000),
        "2000-4000Hz": (2000, 4000),
        "4000-8000Hz": (4000, 8000),
    }

    analysis = {}

    for band_name, files in BANDS.items():
        if files["digital"] is None:
            continue
        if not os.path.exists(os.path.join(BASE, files["digital"])):
            print(f"SKIP {band_name}: digital not found")
            continue
        if not os.path.exists(os.path.join(BASE, files["physical"])):
            print(f"SKIP {band_name}: physical not found")
            continue

        print(f"\n{'='*80}")
        print(f"Analyzing: {band_name}")

        digital = load_audio(files["digital"])
        physical = load_audio(files["physical"])

        if files["original"] and os.path.exists(os.path.join(BASE, files["original"])):
            original = load_audio(files["original"])
        else:
            original = None

        print(f"  Digital:  len={len(digital)}, rms={np.sqrt(np.mean(digital**2)):.6f}")
        print(f"  Physical: len={len(physical)}, rms={np.sqrt(np.mean(physical**2)):.6f}")

        # Align
        dig_aligned, phys_aligned, lag = align_audios(digital, physical, sr)
        print(f"  Alignment lag: {lag} samples ({lag/sr*1000:.1f} ms)")

        # Compute perturbation (digital - original)
        if original is not None:
            orig_aligned = original[:len(dig_aligned)]
            if len(orig_aligned) < len(dig_aligned):
                orig_aligned = np.pad(orig_aligned, (0, len(dig_aligned) - len(orig_aligned)))
            perturbation = dig_aligned - orig_aligned[:len(dig_aligned)]
            print(f"  Perturbation rms: {np.sqrt(np.mean(perturbation**2)):.6f}")
        else:
            perturbation = None

        # Residual (physical - digital) = channel distortion
        residual = phys_aligned - dig_aligned
        residual_rms = np.sqrt(np.mean(residual**2))
        digital_rms = np.sqrt(np.mean(dig_aligned**2))
        physical_rms = np.sqrt(np.mean(phys_aligned**2))

        # Correlation
        corr = np.corrcoef(dig_aligned, phys_aligned)[0, 1]
        print(f"  Correlation (digital vs physical): {corr:.4f}")
        print(f"  Residual RMS: {residual_rms:.6f}")
        print(f"  Gain (physical/digital): {physical_rms/digital_rms:.4f}")

        # Band energy analysis
        dig_bands = compute_band_energy(dig_aligned, sr, freq_bands)
        phys_bands = compute_band_energy(phys_aligned, sr, freq_bands)
        if perturbation is not None:
            pert_bands = compute_band_energy(perturbation, sr, freq_bands)
        res_bands = compute_band_energy(residual, sr, freq_bands)

        print(f"\n  {'Band':<15s} | {'Digital':>12s} | {'Physical':>12s} | {'Ratio(P/D)':>12s} | {'Residual':>12s}", end="")
        if perturbation is not None:
            print(f" | {'Perturb':>12s}", end="")
        print()
        print(f"  {'-'*80}")

        band_ratios = {}
        for bname in freq_bands:
            d_val = dig_bands[bname]
            p_val = phys_bands[bname]
            ratio = p_val / d_val if d_val > 0 else 0
            r_val = res_bands[bname]
            band_ratios[bname] = ratio
            print(f"  {bname:<15s} | {d_val:12.2e} | {p_val:12.2e} | {ratio:12.4f} | {r_val:12.2e}", end="")
            if perturbation is not None:
                print(f" | {pert_bands[bname]:12.2e}", end="")
            print()

        # Per-band correlation
        print(f"\n  Per-band correlation (digital vs physical):")
        nperseg = 2048
        f_dig, t_dig, Sxx_dig = signal.spectrogram(dig_aligned, fs=sr, nperseg=nperseg, noverlap=1536)
        f_phys, t_phys, Sxx_phys = signal.spectrogram(phys_aligned, fs=sr, nperseg=nperseg, noverlap=1536)

        min_t = min(Sxx_dig.shape[1], Sxx_phys.shape[1])
        Sxx_dig = Sxx_dig[:, :min_t]
        Sxx_phys = Sxx_phys[:, :min_t]

        band_corrs = {}
        for bname, (lo, hi) in freq_bands.items():
            mask = (f_dig >= lo) & (f_dig < hi)
            if mask.sum() == 0:
                continue
            dig_band_spec = Sxx_dig[mask, :].flatten()
            phys_band_spec = Sxx_phys[mask, :].flatten()
            if np.std(dig_band_spec) > 0 and np.std(phys_band_spec) > 0:
                bc = np.corrcoef(dig_band_spec, phys_band_spec)[0, 1]
            else:
                bc = 0.0
            band_corrs[bname] = bc
            print(f"  {bname:<15s}: {bc:.4f}")

        analysis[band_name] = {
            "lag_samples": int(lag),
            "lag_ms": round(lag / sr * 1000, 1),
            "correlation": round(corr, 4),
            "digital_rms": round(float(digital_rms), 6),
            "physical_rms": round(float(physical_rms), 6),
            "gain": round(float(physical_rms / digital_rms), 4),
            "residual_rms": round(float(residual_rms), 6),
            "perturbation_rms": round(float(np.sqrt(np.mean(perturbation**2))), 6) if perturbation is not None else None,
            "band_energy_digital": {k: float(v) for k, v in dig_bands.items()},
            "band_energy_physical": {k: float(v) for k, v in phys_bands.items()},
            "band_energy_perturbation": {k: float(v) for k, v in pert_bands.items()} if perturbation is not None else None,
            "band_energy_ratio_phys_dig": {k: round(float(v), 4) for k, v in band_ratios.items()},
            "band_correlation": {k: round(float(v), 4) for k, v in band_corrs.items()},
        }

    # ===== Plot comparison =====
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    for idx, (band_name, files) in enumerate(BANDS.items()):
        if band_name not in analysis:
            continue

        digital = load_audio(files["digital"])
        physical = load_audio(files["physical"])
        dig_aligned, phys_aligned, lag = align_audios(digital, physical, sr)

        col = idx if idx < 3 else idx  # adjust for layout
        if idx >= 3:
            break

        # Waveform comparison
        ax = axes[0, idx]
        t_axis = np.arange(len(dig_aligned)) / sr
        ax.plot(t_axis, dig_aligned, alpha=0.5, label='Digital', linewidth=0.3)
        ax.plot(t_axis, phys_aligned, alpha=0.5, label='Physical', linewidth=0.3)
        ax.set_title(f'{band_name} - Waveform')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=8)
        ax.set_ylim(-1.2, 1.2)

        # Spectrogram - digital
        ax = axes[1, idx]
        f, t, Sxx = signal.spectrogram(dig_aligned, fs=sr, nperseg=2048, noverlap=1536)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
        ax.set_ylim(0, 4000)
        ax.set_title(f'{band_name} - Digital Spectrogram')
        ax.set_ylabel('Freq (Hz)')

        # Spectrogram - physical
        ax = axes[2, idx]
        f, t, Sxx = signal.spectrogram(phys_aligned, fs=sr, nperseg=2048, noverlap=1536)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
        ax.set_ylim(0, 4000)
        ax.set_title(f'{band_name} - Physical Spectrogram')
        ax.set_ylabel('Freq (Hz)')
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'digital_vs_physical_comparison.png'), dpi=150)
    print(f"\nSaved comparison plot: {OUT_DIR}/digital_vs_physical_comparison.png")

    # ===== Band energy bar chart =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    band_labels = list(freq_bands.keys())
    x = np.arange(len(band_labels))
    width = 0.35

    for idx, band_name in enumerate(["bp300-600", "bp400-700", "bp500-800"]):
        if band_name not in analysis:
            continue
        ax = axes[idx]
        d_vals = [analysis[band_name]["band_energy_digital"][b] for b in band_labels]
        p_vals = [analysis[band_name]["band_energy_physical"][b] for b in band_labels]

        ax.bar(x - width/2, np.log10(np.array(d_vals) + 1e-12), width, label='Digital', alpha=0.8)
        ax.bar(x + width/2, np.log10(np.array(p_vals) + 1e-12), width, label='Physical', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{band_name} - Band Energy (log10)')
        ax.legend(fontsize=8)
        ax.set_ylabel('log10(energy)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'band_energy_comparison.png'), dpi=150)
    print(f"Saved band energy plot: {OUT_DIR}/band_energy_comparison.png")

    # ===== Ratio comparison across bands =====
    fig, ax = plt.subplots(figsize=(12, 6))

    for band_name in ["bp300-600", "bp400-700", "bp500-800"]:
        if band_name not in analysis:
            continue
        ratios = [analysis[band_name]["band_energy_ratio_phys_dig"][b] for b in band_labels]
        ax.plot(band_labels, ratios, 'o-', label=band_name, linewidth=2, markersize=8)

    ax.set_ylabel('Physical/Digital Energy Ratio')
    ax.set_title('Channel Transfer: Physical/Digital Energy Ratio by Frequency Band')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Unity (no change)')
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'channel_transfer_ratio.png'), dpi=150)
    print(f"Saved channel transfer plot: {OUT_DIR}/channel_transfer_ratio.png")

    # Save analysis JSON
    out_path = os.path.join(OUT_DIR, 'analysis.json')
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis: {out_path}")


if __name__ == "__main__":
    main()
