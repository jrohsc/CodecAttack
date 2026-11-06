#!/usr/bin/env python3
"""
Noise cancellation impact analysis (CPU-only, no GPU needed).

Tests whether noise suppression destroys the adversarial perturbation by:
1. Loading digital adversarial audio (carrier + perturbation)
2. Loading clean carrier (original music without perturbation)
3. Extracting perturbation = adversarial - carrier
4. Applying noise cancellation/suppression to the adversarial audio
5. Extracting residual perturbation after denoising
6. Measuring how much perturbation energy survives

Also applies denoising to physical OTA recordings and compares.

Denoisers tested:
  - noisereduce (spectral gating) — mild and aggressive
  - Wiener filter (using carrier as reference)

If denoising kills most perturbation energy -> iPhone's noise processing
could explain why physical OTA fails.
"""

import os
import sys
import json
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE, MUSIC_FILES

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_noise_cancellation")
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_empirical_ota_IR")

# Test cases: (digital_wav, physical_m4a_mac, physical_m4a_marshall, label)
# Use exp1 which has both Mac and Marshall recordings
EXP1_DIGITAL = "1_exp_empirical_nonlinearity/audio"
EXP1_MAC = "1_exp_empirical_nonlinearity/audio_mac_iphone_record_ota"
EXP1_MARSHALL = "1_exp_empirical_nonlinearity/audio_marshall_iphone_record"

# Also test exp4 (Marshall only, bandpass-constrained attack)
EXP4_DIGITAL = "4_exp_bandpass300_1000_marshall/eps3.0/audio"
EXP4_MARSHALL = "4_exp_bandpass300_1000_marshall/eps3.0/audio_marshall_iphone_recording"

# Also test exp10 (robust Marshall channel, most recent)
EXP10_DIGITAL = "10_exp_robust_marshall_channel/eps5.0/audio"
EXP10_MARSHALL = "10_exp_robust_marshall_channel/eps5.0/audio_marshall_iphone_recording"


def load_audio(path, target_sr=TARGET_SAMPLE_RATE):
    """Load WAV or M4A, return float32 numpy array at target_sr."""
    if path.endswith('.m4a'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
                 '-acodec', 'pcm_f32le', tmp],
                capture_output=True, check=True,
            )
            data, sr = sf.read(tmp, dtype='float32')
            return data
        finally:
            os.unlink(tmp)
    else:
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
        if sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data


def load_carrier(music_name="calm_1", duration_s=None, target_sr=TARGET_SAMPLE_RATE):
    """Load the original clean carrier music."""
    path = MUSIC_FILES[music_name]
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    if duration_s:
        max_samples = int(duration_s * target_sr)
        data = data[:max_samples]
    return data


def denoise_noisereduce(audio, sr, prop_decrease=0.8):
    """Spectral gating denoiser (noisereduce library)."""
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)


def denoise_wiener(audio, carrier, sr):
    """Wiener filter: use carrier as estimate of desired signal."""
    L = min(len(audio), len(carrier))
    audio = audio[:L]
    carrier = carrier[:L]

    nperseg = 2048
    f, Pxx = sig.welch(audio, fs=sr, nperseg=nperseg)
    f, Pcc = sig.welch(carrier, fs=sr, nperseg=nperseg)

    # Wiener gain: G(f) = Pcc / Pxx
    G = Pcc / (Pxx + 1e-10)
    G = np.clip(G, 0, 1)

    # Apply in STFT domain
    f_stft, t_stft, Zxx = sig.stft(audio, fs=sr, nperseg=nperseg)
    # Interpolate G to STFT freq bins
    G_interp = np.interp(f_stft, f, G)
    Zxx_filtered = Zxx * G_interp[:, None]
    _, audio_filtered = sig.istft(Zxx_filtered, fs=sr, nperseg=nperseg)

    return audio_filtered[:L]


def band_energy(audio, sr, low, high):
    """Energy in a frequency band (dB)."""
    f, Pxx = sig.welch(audio, fs=sr, nperseg=2048)
    mask = (f >= low) & (f < high)
    return 10 * np.log10(np.sum(Pxx[mask]) + 1e-20)


def spectral_comparison(original, denoised, sr, label, ax):
    """Plot spectral comparison."""
    f_o, Pxx_o = sig.welch(original, fs=sr, nperseg=2048)
    f_d, Pxx_d = sig.welch(denoised, fs=sr, nperseg=2048)
    ax.semilogy(f_o, Pxx_o, label='Original', alpha=0.8)
    ax.semilogy(f_d, Pxx_d, label='Denoised', alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(label)
    ax.legend()
    ax.set_xlim(20, 8000)
    ax.grid(True, alpha=0.3)


def analyze_denoising_impact(adv_audio, carrier, sr, label):
    """Analyze how denoising affects the adversarial perturbation."""
    L = min(len(adv_audio), len(carrier))
    adv = adv_audio[:L]
    car = carrier[:L]
    perturbation = adv - car

    # Apply different denoisers
    dn_mild = denoise_noisereduce(adv, sr, prop_decrease=0.3)[:L]
    dn_aggressive = denoise_noisereduce(adv, sr, prop_decrease=0.9)[:L]
    dn_wiener = denoise_wiener(adv, car, sr)[:L]

    # Extract remaining perturbation after denoising
    pert_mild = dn_mild - car
    pert_aggressive = dn_aggressive - car
    pert_wiener = dn_wiener - car

    # Measure perturbation survival
    bands = {
        'sub_bass': (20, 300),
        'bass': (300, 1000),
        'mid': (1000, 3000),
        'upper_mid': (3000, 6000),
        'high': (6000, 8000),
        'full': (20, 8000),
    }

    results = {'label': label}
    for bname, (lo, hi) in bands.items():
        e_orig = band_energy(perturbation, sr, lo, hi)
        e_mild = band_energy(pert_mild, sr, lo, hi)
        e_aggr = band_energy(pert_aggressive, sr, lo, hi)
        e_wien = band_energy(pert_wiener, sr, lo, hi)
        results[bname] = {
            'original_perturbation_dB': float(e_orig),
            'after_mild_denoise_dB': float(e_mild),
            'after_aggressive_denoise_dB': float(e_aggr),
            'after_wiener_dB': float(e_wien),
            'mild_reduction_dB': float(e_orig - e_mild),
            'aggressive_reduction_dB': float(e_orig - e_aggr),
            'wiener_reduction_dB': float(e_orig - e_wien),
        }

    # Overall SNR of perturbation
    results['perturbation_snr_dB'] = float(
        10 * np.log10(np.sum(perturbation**2) / (np.sum(car**2) + 1e-20))
    )

    # Waveform correlation: denoised vs original adversarial
    results['corr_mild'] = float(np.corrcoef(adv, dn_mild)[0, 1])
    results['corr_aggressive'] = float(np.corrcoef(adv, dn_aggressive)[0, 1])
    results['corr_wiener'] = float(np.corrcoef(adv, dn_wiener)[0, 1])

    return results, {
        'original': adv,
        'perturbation': perturbation,
        'mild': dn_mild,
        'aggressive': dn_aggressive,
        'wiener': dn_wiener,
        'pert_mild': pert_mild,
        'pert_aggressive': pert_aggressive,
        'pert_wiener': pert_wiener,
    }


def analyze_physical_denoising(physical_audio, sr, label):
    """Apply denoising to physical OTA recording and measure changes."""
    dn_mild = denoise_noisereduce(physical_audio, sr, prop_decrease=0.3)
    dn_aggressive = denoise_noisereduce(physical_audio, sr, prop_decrease=0.9)

    bands = {
        'sub_bass': (20, 300),
        'bass': (300, 1000),
        'mid': (1000, 3000),
        'upper_mid': (3000, 6000),
        'high': (6000, 8000),
        'full': (20, 8000),
    }

    results = {'label': label}
    for bname, (lo, hi) in bands.items():
        e_orig = band_energy(physical_audio, sr, lo, hi)
        e_mild = band_energy(dn_mild, sr, lo, hi)
        e_aggr = band_energy(dn_aggressive, sr, lo, hi)
        results[bname] = {
            'physical_dB': float(e_orig),
            'mild_denoised_dB': float(e_mild),
            'aggressive_denoised_dB': float(e_aggr),
            'mild_change_dB': float(e_mild - e_orig),
            'aggressive_change_dB': float(e_aggr - e_orig),
        }

    return results, dn_mild, dn_aggressive


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "denoised_audio"), exist_ok=True)

    sr = TARGET_SAMPLE_RATE

    # Load carrier
    print("Loading carrier music (calm_1)...")
    carrier = load_carrier("calm_1", target_sr=sr)

    # =========================================================================
    # Part 1: Denoising digital adversarial audio
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Denoising impact on DIGITAL adversarial audio")
    print("(Does noise cancellation destroy the perturbation?)")
    print("=" * 70)

    digital_results = []
    all_audios = {}

    for exp_label, digital_dir in [
        ("exp1", EXP1_DIGITAL),
        ("exp4", EXP4_DIGITAL),
        ("exp10", EXP10_DIGITAL),
    ]:
        full_dir = os.path.join(BASE, digital_dir)
        if not os.path.isdir(full_dir):
            print(f"  Skipping {exp_label}: {full_dir} not found")
            continue

        wavs = sorted([f for f in os.listdir(full_dir) if f.endswith('.wav')])
        for wav_name in wavs[:3]:  # First 3 per experiment
            wav_path = os.path.join(full_dir, wav_name)
            label = f"{exp_label}_{wav_name.replace('.wav', '')}"
            print(f"\n  {label}:")

            adv = load_audio(wav_path, sr)
            result, audios = analyze_denoising_impact(adv, carrier, sr, label)
            digital_results.append(result)
            all_audios[label] = audios

            print(f"    Perturbation SNR: {result['perturbation_snr_dB']:.1f} dB")
            print(f"    Full-band perturbation reduction:")
            fb = result['full']
            print(f"      Mild denoise:   {fb['mild_reduction_dB']:+.1f} dB")
            print(f"      Aggressive:     {fb['aggressive_reduction_dB']:+.1f} dB")
            print(f"      Wiener filter:  {fb['wiener_reduction_dB']:+.1f} dB")

            # Save denoised audio for later GPU eval
            for dn_type in ['mild', 'aggressive', 'wiener']:
                out_path = os.path.join(OUTPUT_DIR, "denoised_audio",
                                        f"{label}_denoised_{dn_type}.wav")
                sf.write(out_path, audios[dn_type], sr)

    # =========================================================================
    # Part 2: Denoising physical OTA recordings
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Denoising impact on PHYSICAL OTA recordings")
    print("(How much does noisereduce change the physical recording?)")
    print("=" * 70)

    physical_results = []

    for speaker, phys_dir_name in [
        ("mac", EXP1_MAC),
        ("marshall_exp1", EXP1_MARSHALL),
        ("marshall_exp4", EXP4_MARSHALL),
        ("marshall_exp10", EXP10_MARSHALL),
    ]:
        full_dir = os.path.join(BASE, phys_dir_name)
        if not os.path.isdir(full_dir):
            print(f"  Skipping {speaker}: {full_dir} not found")
            continue

        m4as = sorted([f for f in os.listdir(full_dir) if f.endswith('.m4a')])
        for m4a_name in m4as[:2]:  # First 2 per speaker/experiment
            m4a_path = os.path.join(full_dir, m4a_name)
            label = f"{speaker}_{m4a_name.replace('.m4a', '')}"
            print(f"\n  {label}:")

            phys = load_audio(m4a_path, sr)
            result, dn_mild, dn_aggressive = analyze_physical_denoising(phys, sr, label)
            physical_results.append(result)

            fb = result['full']
            print(f"    Full-band energy: {fb['physical_dB']:.1f} dB")
            print(f"    After mild denoise:   {fb['mild_change_dB']:+.1f} dB")
            print(f"    After aggressive:     {fb['aggressive_change_dB']:+.1f} dB")

            # Save denoised physical for later GPU eval
            for dn_type, dn_audio in [('mild', dn_mild), ('aggressive', dn_aggressive)]:
                out_path = os.path.join(OUTPUT_DIR, "denoised_audio",
                                        f"physical_{label}_denoised_{dn_type}.wav")
                sf.write(out_path, dn_audio, sr)

    # =========================================================================
    # Part 3: Summary & plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if digital_results:
        print("\nDigital adversarial audio — perturbation reduction by denoising:")
        band_names = ['sub_bass', 'bass', 'mid', 'upper_mid', 'high', 'full']
        band_labels_short = ['<300Hz', '300-1k', '1-3k', '3-6k', '6-8k', 'FULL']

        print(f"\n  {'Band':>10s}  {'Mild':>8s}  {'Aggressive':>10s}  {'Wiener':>8s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
        for bname, blabel in zip(band_names, band_labels_short):
            mild_avg = np.mean([r[bname]['mild_reduction_dB'] for r in digital_results])
            aggr_avg = np.mean([r[bname]['aggressive_reduction_dB'] for r in digital_results])
            wien_avg = np.mean([r[bname]['wiener_reduction_dB'] for r in digital_results])
            print(f"  {blabel:>10s}  {mild_avg:+8.1f}  {aggr_avg:+10.1f}  {wien_avg:+8.1f} dB")

        print("\n  Interpretation:")
        full_aggr = np.mean([r['full']['aggressive_reduction_dB'] for r in digital_results])
        if full_aggr > 5:
            print(f"  >>> Aggressive denoising removes {full_aggr:.1f} dB of perturbation energy")
            print(f"  >>> This is STRONG evidence that noise cancellation could kill the attack")
        elif full_aggr > 2:
            print(f"  >>> Moderate perturbation reduction ({full_aggr:.1f} dB)")
        else:
            print(f"  >>> Minimal perturbation reduction ({full_aggr:.1f} dB) — NC unlikely the cause")

    # Plot: per-band perturbation reduction for digital audio
    if digital_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        band_names_plot = ['sub_bass', 'bass', 'mid', 'upper_mid', 'high']
        x_labels = ['<300', '300-1k', '1-3k', '3-6k', '6-8k']

        for ax, dn_type, title in zip(axes,
            ['mild_reduction_dB', 'aggressive_reduction_dB', 'wiener_reduction_dB'],
            ['Mild Denoise', 'Aggressive Denoise', 'Wiener Filter']):

            vals = [np.mean([r[b][dn_type] for r in digital_results]) for b in band_names_plot]
            stds = [np.std([r[b][dn_type] for r in digital_results]) for b in band_names_plot]
            colors = ['indianred' if v > 0 else 'steelblue' for v in vals]
            ax.bar(range(len(vals)), vals, yerr=stds, color=colors, capsize=5)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(x_labels)
            ax.set_ylabel('Perturbation Reduction (dB)')
            ax.set_title(title)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Perturbation Energy Removed by Denoising\n(positive = energy removed, negative = energy added)',
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'denoising_perturbation_reduction.png'), dpi=150)
        print(f"\nSaved: {OUTPUT_DIR}/denoising_perturbation_reduction.png")

    # Plot: spectral comparison for one example
    if all_audios:
        first_label = list(all_audios.keys())[0]
        audios = all_audios[first_label]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Original perturbation spectrum
        f, Pxx = sig.welch(audios['perturbation'], fs=sr, nperseg=2048)
        axes[0, 0].semilogy(f, Pxx)
        axes[0, 0].set_title(f'Original Perturbation\n({first_label})')
        axes[0, 0].set_xlim(20, 8000)
        axes[0, 0].grid(True, alpha=0.3)

        for ax, pert, title in [
            (axes[0, 1], audios['pert_mild'], 'After Mild Denoise'),
            (axes[1, 0], audios['pert_aggressive'], 'After Aggressive Denoise'),
            (axes[1, 1], audios['pert_wiener'], 'After Wiener Filter'),
        ]:
            f2, Pxx2 = sig.welch(pert, fs=sr, nperseg=2048)
            ax.semilogy(f, Pxx, alpha=0.4, label='Original pert.')
            ax.semilogy(f2, Pxx2, alpha=0.8, label='Remaining pert.')
            ax.set_title(title)
            ax.set_xlim(20, 8000)
            ax.legend()
            ax.grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')

        plt.suptitle('Perturbation Spectrum Before/After Denoising', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'perturbation_spectrum_denoising.png'), dpi=150)
        print(f"Saved: {OUTPUT_DIR}/perturbation_spectrum_denoising.png")

    # Save all results as JSON
    all_json = {
        'digital_denoising': digital_results,
        'physical_denoising': physical_results,
        'denoised_audio_dir': os.path.join(OUTPUT_DIR, 'denoised_audio'),
        'note': 'Denoised audio files saved for GPU eval with Qwen2-Audio',
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(all_json, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/results.json")
    print(f"Denoised audio files saved in: {OUTPUT_DIR}/denoised_audio/")
    print("  -> Run these through Qwen2-Audio on GPU to test if attack survives denoising")


if __name__ == "__main__":
    main()
