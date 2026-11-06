#!/usr/bin/env python3
"""Analyze perturbation survival for bandpass-constrained (300-4000Hz) OTA recordings.

Compares original adversarial wav vs physical OTA recording (MacBook -> iPhone).
Key question: Did the bandpass constraint actually put perturbation energy in
frequencies the speaker can reproduce? Or is it still being destroyed?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import soundfile as sf
import subprocess
import tempfile
import os
import json
from scipy import signal

# ── Paths ──────────────────────────────────────────────────────────────────
EXP_DIR = "${REPO_ROOT}/results_empirical_ota_IR/exp_bandpass300_4000_eps3.0_1000steps"
ADV_DIR = os.path.join(EXP_DIR, "audio")
OTA_DIR = os.path.join(EXP_DIR, "audio_mac_iphone_record_ota")
CARRIER_PATH = "${PROJECT_ROOT}/data/music/calm_1.mp3"
OUT_DIR = os.path.join(EXP_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SR = 16000


def load_audio(path, sr=TARGET_SR):
    """Load audio as mono float64."""
    if path.endswith('.m4a'):
        tmp_path = tempfile.mktemp(suffix='.wav')
        subprocess.run(['ffmpeg', '-y', '-i', path, '-ar', str(sr), '-ac', '1', tmp_path],
                       capture_output=True)
        y, _ = sf.read(tmp_path)
        os.unlink(tmp_path)
    else:
        y, orig_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64)


def align_xcorr(ref, test, sr=TARGET_SR):
    """Align test to ref using cross-correlation."""
    chunk_len = min(len(ref), len(test), sr * 4)
    corr = signal.correlate(ref[:chunk_len], test[:chunk_len], mode='full')
    lag = np.argmax(np.abs(corr)) - chunk_len + 1
    if lag > 0:
        aligned = test[lag:]
    elif lag < 0:
        aligned = np.concatenate([np.zeros(-lag), test])
    else:
        aligned = test
    return aligned, lag


def bandpass_filter(y, sr, low, high, order=5):
    """Apply bandpass filter."""
    nyq = sr / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = signal.butter(order, [low_n, high_n], btype='band')
    return signal.filtfilt(b, a, y)


def compute_spectrum(y, sr):
    """Compute one-sided power spectrum."""
    N = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    power = np.abs(Y)**2 / N
    return freqs, power


# ── Frequency bands for analysis ──────────────────────────────────────────
frequency_bands = [
    (50, 150, '50-150 Hz (sub-bass)'),
    (150, 300, '150-300 Hz (bass)'),
    (300, 600, '300-600 Hz (low-mid)'),
    (600, 1200, '600-1.2k Hz (mid)'),
    (1200, 2500, '1.2-2.5k Hz (upper-mid)'),
    (2500, 4000, '2.5-4k Hz (presence)'),
    (4000, 6000, '4-6k Hz (brilliance)'),
    (6000, 8000, '6-8k Hz (air)'),
]

# ── Load carrier ──────────────────────────────────────────────────────────
print("Loading clean carrier...")
carrier_full = load_audio(CARRIER_PATH, sr=TARGET_SR)

# ── Find matching pairs ───────────────────────────────────────────────────
ota_files = sorted([f for f in os.listdir(OTA_DIR) if f.endswith(('.m4a', '.wav'))])
print(f"Found {len(ota_files)} OTA recordings\n")

all_results = []

for ota_file in ota_files:
    base = ota_file.rsplit('.', 1)[0]  # e.g., calm_1__nav_turn_left
    adv_path = os.path.join(ADV_DIR, base + ".wav")
    ota_path = os.path.join(OTA_DIR, ota_file)

    if not os.path.exists(adv_path):
        print(f"  Skipping {base}: no matching adversarial wav")
        continue

    print(f"Processing {base}...")

    # Load audio
    y_adv = load_audio(adv_path, sr=TARGET_SR)
    y_ota = load_audio(ota_path, sr=TARGET_SR)
    carrier = carrier_full[:len(y_adv)]

    # Extract original perturbation
    min_len = min(len(y_adv), len(carrier))
    delta_orig = y_adv[:min_len] - carrier[:min_len]

    # Align OTA with carrier
    y_ota_aligned, lag = align_xcorr(carrier, y_ota)

    # Trim to common length
    n = min(min_len, len(y_ota_aligned))
    carrier_t = carrier[:n]
    delta_orig_t = delta_orig[:n]
    y_ota_t = y_ota_aligned[:n]

    # Gain-normalize OTA
    gain = np.sqrt(np.sum(carrier_t**2) / (np.sum(y_ota_t**2) + 1e-12))
    y_ota_t = y_ota_t * gain

    # Extract recovered perturbation
    delta_recovered = y_ota_t - carrier_t

    # Overall correlation
    overall_corr = np.corrcoef(delta_orig_t, delta_recovered)[0, 1]

    # Compute spectra of perturbation
    freqs_orig, power_orig = compute_spectrum(delta_orig_t, TARGET_SR)
    freqs_rec, power_rec = compute_spectrum(delta_recovered, TARGET_SR)

    # Per-band analysis
    band_results = []
    for low, high, name in frequency_bands:
        try:
            d_orig_band = bandpass_filter(delta_orig_t, TARGET_SR, low, high)
            d_rec_band = bandpass_filter(delta_recovered, TARGET_SR, low, high)
            corr = np.corrcoef(d_orig_band, d_rec_band)[0, 1]
            orig_energy = np.sum(d_orig_band**2)
            rec_energy = np.sum(d_rec_band**2)
            energy_ratio = rec_energy / (orig_energy + 1e-12)
            orig_rms = float(np.sqrt(np.mean(d_orig_band**2)))
            rec_rms = float(np.sqrt(np.mean(d_rec_band**2)))
        except Exception:
            corr, energy_ratio, orig_rms, rec_rms = 0, 0, 0, 0

        band_results.append({
            'band': name, 'low': low, 'high': high,
            'correlation': float(corr),
            'energy_ratio': float(energy_ratio),
            'orig_rms': orig_rms,
            'rec_rms': rec_rms,
        })

    # Energy distribution: what % of perturbation is in each band?
    total_orig_energy = np.sum(delta_orig_t**2)
    energy_pcts = []
    for low, high, name in frequency_bands:
        try:
            d_band = bandpass_filter(delta_orig_t, TARGET_SR, low, high)
            pct = 100 * np.sum(d_band**2) / (total_orig_energy + 1e-12)
        except Exception:
            pct = 0
        energy_pcts.append(pct)

    # Spectral centroid of perturbation
    spec_centroid_orig = np.sum(freqs_orig * power_orig) / (np.sum(power_orig) + 1e-12)
    spec_centroid_rec = np.sum(freqs_rec * power_rec) / (np.sum(power_rec) + 1e-12)

    # Energy below/within/above bandpass
    bp_low, bp_high = 300, 4000
    below_bp = np.sum(bandpass_filter(delta_orig_t, TARGET_SR, 50, bp_low)**2) / (total_orig_energy + 1e-12) * 100
    within_bp = np.sum(bandpass_filter(delta_orig_t, TARGET_SR, bp_low, bp_high)**2) / (total_orig_energy + 1e-12) * 100
    above_bp = np.sum(bandpass_filter(delta_orig_t, TARGET_SR, bp_high, 7500)**2) / (total_orig_energy + 1e-12) * 100

    result = {
        'sample': base,
        'lag_ms': float(lag / TARGET_SR * 1000),
        'gain_factor': float(gain),
        'overall_correlation': float(overall_corr),
        'spectral_centroid_orig_hz': float(spec_centroid_orig),
        'spectral_centroid_rec_hz': float(spec_centroid_rec),
        'energy_below_300hz_pct': float(below_bp),
        'energy_300_4000hz_pct': float(within_bp),
        'energy_above_4000hz_pct': float(above_bp),
        'orig_perturbation_rms': float(np.sqrt(np.mean(delta_orig_t**2))),
        'recovered_perturbation_rms': float(np.sqrt(np.mean(delta_recovered**2))),
        'bands': band_results,
        'energy_distribution_pct': energy_pcts,
    }
    all_results.append(result)

    print(f"  Lag: {result['lag_ms']:.1f}ms, Gain: {gain:.3f}")
    print(f"  Overall corr(δ_orig, δ_recovered): {overall_corr:.4f}")
    print(f"  Spectral centroid: orig={spec_centroid_orig:.0f}Hz, recovered={spec_centroid_rec:.0f}Hz")
    print(f"  Energy distribution: <300Hz={below_bp:.1f}%, 300-4000Hz={within_bp:.1f}%, >4000Hz={above_bp:.1f}%")
    print(f"  Per-band:")
    for b, pct in zip(band_results, energy_pcts):
        print(f"    {b['band']}: corr={b['correlation']:.4f}, energy_ratio={b['energy_ratio']:.3f}, "
              f"orig_energy={pct:.1f}%")
    print()

# ── Summary ───────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY: Bandpass 300-4000Hz Perturbation Survival After Physical OTA")
print("=" * 70)

mean_corr = np.mean([r['overall_correlation'] for r in all_results])
mean_centroid_orig = np.mean([r['spectral_centroid_orig_hz'] for r in all_results])
mean_centroid_rec = np.mean([r['spectral_centroid_rec_hz'] for r in all_results])
mean_below = np.mean([r['energy_below_300hz_pct'] for r in all_results])
mean_within = np.mean([r['energy_300_4000hz_pct'] for r in all_results])
mean_above = np.mean([r['energy_above_4000hz_pct'] for r in all_results])

print(f"\nMean overall correlation:    {mean_corr:.4f}")
print(f"Mean spectral centroid:     orig={mean_centroid_orig:.0f}Hz, recovered={mean_centroid_rec:.0f}Hz")
print(f"Energy distribution:        <300Hz={mean_below:.1f}%, 300-4000Hz={mean_within:.1f}%, >4000Hz={mean_above:.1f}%")
print()

print("Per-band survival (averaged):")
for i, (_, _, name) in enumerate(frequency_bands):
    band_corrs = [r['bands'][i]['correlation'] for r in all_results]
    band_energies = [r['bands'][i]['energy_ratio'] for r in all_results]
    energy_pcts = [r['energy_distribution_pct'][i] for r in all_results]
    print(f"  {name:>25s}: corr={np.mean(band_corrs):.4f}, energy_ratio={np.mean(band_energies):.3f}, "
          f"orig_energy={np.mean(energy_pcts):.1f}%")

# ── Comparison verdict ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
if mean_within > 80:
    print(f"Bandpass constraint WORKED: {mean_within:.0f}% of perturbation energy is in 300-4000Hz")
    if mean_corr > 0.1:
        print(f"AND perturbation partially survives OTA (corr={mean_corr:.3f})")
        print("→ Physical channel preserves some signal, but attack still fails")
        print("→ May need higher eps or model-side improvements")
    else:
        print(f"BUT perturbation still destroyed by OTA (corr={mean_corr:.3f})")
        print("→ Even in the right frequency range, nonlinear distortion kills the signal")
        print("→ Root cause is NOT just frequency mismatch")
else:
    print(f"Bandpass constraint DID NOT fully work: only {mean_within:.0f}% in 300-4000Hz")
    print(f"Still {mean_below:.0f}% below 300Hz — optimizer may be leaking energy outside band")

# ── Save JSON ─────────────────────────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "bandpass_ota_analysis.json")
with open(json_path, 'w') as f:
    json.dump({
        'summary': {
            'mean_overall_correlation': float(mean_corr),
            'mean_spectral_centroid_orig_hz': float(mean_centroid_orig),
            'mean_spectral_centroid_rec_hz': float(mean_centroid_rec),
            'energy_below_300hz_pct': float(mean_below),
            'energy_300_4000hz_pct': float(mean_within),
            'energy_above_4000hz_pct': float(mean_above),
        },
        'per_sample': all_results,
    }, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

# Use first sample for detailed viz
d_orig = load_audio(os.path.join(ADV_DIR, all_results[0]['sample'] + '.wav'), sr=TARGET_SR)
carrier_viz = carrier_full[:len(d_orig)]
d_orig = d_orig[:len(carrier_viz)] - carrier_viz
d_ota = load_audio(os.path.join(OTA_DIR, ota_files[0]), sr=TARGET_SR)
d_ota_aligned, _ = align_xcorr(carrier_viz, d_ota)
n_viz = min(len(d_orig), len(d_ota_aligned))
carrier_viz = carrier_viz[:n_viz]
d_orig = d_orig[:n_viz]
d_ota_aligned = d_ota_aligned[:n_viz]
gain_viz = np.sqrt(np.sum(carrier_viz**2) / (np.sum(d_ota_aligned**2) + 1e-12))
d_ota_aligned = d_ota_aligned * gain_viz
d_rec = d_ota_aligned - carrier_viz

# Panel 1: Power spectra of original perturbation
ax1 = fig.add_subplot(gs[0, 0])
freqs_o, power_o = compute_spectrum(d_orig, TARGET_SR)
freqs_r, power_r = compute_spectrum(d_rec, TARGET_SR)
# Smooth for visibility
from scipy.ndimage import uniform_filter1d
smooth_o = uniform_filter1d(10*np.log10(power_o + 1e-20), size=50)
smooth_r = uniform_filter1d(10*np.log10(power_r + 1e-20), size=50)
ax1.plot(freqs_o, smooth_o, color='#2ecc71', linewidth=1.5, label='δ_original (crafted)', alpha=0.9)
ax1.plot(freqs_r, smooth_r, color='#e74c3c', linewidth=1.5, label='δ_recovered (from OTA)', alpha=0.9)
ax1.axvspan(300, 4000, alpha=0.1, color='blue', label='Bandpass range (300-4000Hz)')
ax1.axvline(300, color='blue', linestyle='--', alpha=0.4)
ax1.axvline(4000, color='blue', linestyle='--', alpha=0.4)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power (dB)')
ax1.set_title('Perturbation Power Spectrum: Original vs Recovered\n(bandpass 300-4000Hz constraint)',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim(0, 8000)
ax1.grid(True, alpha=0.3)

# Panel 2: Gain (ratio) per frequency
ax2 = fig.add_subplot(gs[0, 1])
gain_spectrum = smooth_r - smooth_o  # dB difference
ax2.plot(freqs_o, gain_spectrum, color='#9b59b6', linewidth=1.5)
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax2.axvspan(300, 4000, alpha=0.1, color='blue', label='Bandpass range')
ax2.axvline(300, color='blue', linestyle='--', alpha=0.4)
ax2.axvline(4000, color='blue', linestyle='--', alpha=0.4)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Gain (dB): recovered - original')
ax2.set_title('Physical Channel Effect on Perturbation\n(negative = energy lost through speaker+air+mic)',
              fontsize=12, fontweight='bold')
ax2.set_xlim(0, 8000)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Per-band energy distribution (pie-like bar chart)
ax3 = fig.add_subplot(gs[1, 0])
band_names = [b[2] for b in frequency_bands]
mean_energy_pcts = [np.mean([r['energy_distribution_pct'][i] for r in all_results])
                    for i in range(len(frequency_bands))]
in_band = [1 if 300 <= frequency_bands[i][0] and frequency_bands[i][1] <= 4000 else 0
           for i in range(len(frequency_bands))]
colors_band = ['#2ecc71' if ib else '#e74c3c' for ib in in_band]
bars3 = ax3.bar(range(len(band_names)), mean_energy_pcts, color=colors_band, edgecolor='white')
ax3.set_xticks(range(len(band_names)))
ax3.set_xticklabels(band_names, rotation=35, ha='right', fontsize=9)
ax3.set_ylabel('% of total perturbation energy')
ax3.set_title('Where is the perturbation energy?\n(green = within bandpass, red = outside)',
              fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, mean_energy_pcts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 4: Per-band correlation (survival)
ax4 = fig.add_subplot(gs[1, 1])
mean_band_corrs = [np.mean([r['bands'][i]['correlation'] for r in all_results])
                   for i in range(len(frequency_bands))]
colors_corr = ['#e74c3c' if c < 0.05 else '#f39c12' if c < 0.2 else '#2ecc71' for c in mean_band_corrs]
bars4 = ax4.bar(range(len(band_names)), mean_band_corrs, color=colors_corr, edgecolor='white')
ax4.set_xticks(range(len(band_names)))
ax4.set_xticklabels(band_names, rotation=35, ha='right', fontsize=9)
ax4.set_ylabel('Correlation')
ax4.set_title('Per-Band Perturbation Survival Through Physical OTA\n(correlation between crafted δ and recovered δ)',
              fontsize=12, fontweight='bold')
ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars4, mean_band_corrs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 5: Energy ratio (how much survives)
ax5 = fig.add_subplot(gs[2, 0])
mean_band_er = [np.mean([r['bands'][i]['energy_ratio'] for r in all_results])
                for i in range(len(frequency_bands))]
colors_er = ['#e74c3c' if e < 0.3 else '#f39c12' if e < 0.8 else '#2ecc71' for e in mean_band_er]
bars5 = ax5.bar(range(len(band_names)), mean_band_er, color=colors_er, edgecolor='white')
ax5.set_xticks(range(len(band_names)))
ax5.set_xticklabels(band_names, rotation=35, ha='right', fontsize=9)
ax5.set_ylabel('Energy ratio (recovered / original)')
ax5.set_title('Per-Band Energy Ratio\n(<1 = speaker+air destroyed energy, >1 = channel added noise)',
              fontsize=12, fontweight='bold')
ax5.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (1.0)')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
for bar, val in zip(bars5, mean_band_er):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 6: Summary
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
summary = (
    f"BANDPASS OTA ANALYSIS\n"
    f"{'='*45}\n\n"
    f"Bandpass constraint: 300-4000 Hz\n"
    f"eps: 3.0, steps: 1000\n\n"
    f"Mean correlation:        {mean_corr:.4f}\n"
    f"Spectral centroid orig:  {mean_centroid_orig:.0f} Hz\n"
    f"Spectral centroid OTA:   {mean_centroid_rec:.0f} Hz\n\n"
    f"Energy distribution:\n"
    f"  < 300 Hz:    {mean_below:.1f}%\n"
    f"  300-4000 Hz: {mean_within:.1f}%\n"
    f"  > 4000 Hz:   {mean_above:.1f}%\n\n"
    f"Physical OTA result: 0/4 (0%)\n"
)
verdict_color = '#e74c3c' if mean_corr < 0.1 else '#f39c12'
ax6.text(0.5, 0.5, summary, ha='center', va='center', fontsize=11,
         family='monospace', transform=ax6.transAxes,
         bbox=dict(boxstyle='round,pad=0.8', facecolor=verdict_color, alpha=0.15,
                   edgecolor=verdict_color))

fig.suptitle(
    "Bandpass 300-4000Hz Perturbation: Does It Survive Physical OTA?\n"
    "MacBook Air speaker → air → iPhone 16 Pro mic",
    fontsize=15, fontweight='bold', y=0.99
)

fig_path = os.path.join(OUT_DIR, "bandpass_ota_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved figure: {fig_path}")
plt.close()
print("Done!")
