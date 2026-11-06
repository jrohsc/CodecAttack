"""
Generate publication-quality figure showing:
  1. Physical channel frequency response (what the speaker passes)
  2. Per-band nonlinearity curves (polynomial distortion)
  3. Linear vs nonlinear R² per band (how much nonlinearity matters)
  4. Perturbation energy vs channel passband overlap

Usage:
    python plot_nonlinearity_analysis.py [--output figs/nonlinearity_analysis.pdf]
"""
import os
import json
import argparse
import numpy as np
import subprocess
from pathlib import Path
from scipy import signal as scipy_signal
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

SR = 24000
PAIRED_DIR = Path(__file__).parent / "soundcloud_recorded" / "directly_played_from_mac_ir_trained_qwen2_audio_calm_1_eps_5.0"
NL_SUMMARY = Path(__file__).parent / "data" / "empirical_ota" / "nonlinearity_summary.json"
IR_PATH = Path(__file__).parent / "data" / "empirical_ota" / "channel_impulse_response.npy"


def load_pair(orig_path, rec_path):
    orig, sr_orig = sf.read(orig_path)
    if orig.ndim > 1:
        orig = orig.mean(axis=1)
    rec_wav = str(rec_path).replace('.m4a', '_tmp.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(rec_path), '-ar', str(SR), '-ac', '1', rec_wav],
        capture_output=True, check=True,
    )
    rec, _ = sf.read(rec_wav)
    if rec.ndim > 1:
        rec = rec.mean(axis=1)
    os.remove(rec_wav)
    if sr_orig != SR:
        from scipy.signal import resample_poly
        orig = resample_poly(orig, SR, sr_orig).astype(np.float32)
    return orig.astype(np.float32), rec.astype(np.float32)


def align_pair(orig, rec):
    seg_len = min(3 * SR, len(orig), len(rec))
    corr = np.correlate(rec[:seg_len + SR], orig[:seg_len], mode='full')
    delay = np.argmax(np.abs(corr)) - seg_len + 1
    if delay > 0:
        rec_aligned = rec[delay:]
    else:
        rec_aligned = np.pad(rec, (-delay, 0))
    min_len = min(len(orig), len(rec_aligned))
    return orig[:min_len], rec_aligned[:min_len]


def bandpass(audio, low, high, sr=SR, order=5):
    nyq = sr / 2
    if low <= 0 and high >= nyq:
        return audio
    elif low <= 0:
        b, a = scipy_signal.butter(order, high / nyq, btype='low')
    elif high >= nyq:
        b, a = scipy_signal.butter(order, low / nyq, btype='high')
    else:
        b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return scipy_signal.filtfilt(b, a, audio).astype(np.float32)


def compute_channel_tf(orig_all, rec_all):
    """Compute frequency-domain transfer function from paired recordings."""
    # Average power spectral density ratio across all pairs
    all_tf = []
    for orig, rec in zip(orig_all, rec_all):
        n_fft = min(len(orig), len(rec))
        freqs = np.fft.rfftfreq(n_fft, d=1.0/SR)
        O = np.fft.rfft(orig, n=n_fft)
        R = np.fft.rfft(rec, n=n_fft)
        # Smoothed PSD ratio
        window = 256
        O_power = np.convolve(np.abs(O)**2, np.ones(window)/window, mode='same')
        R_power = np.convolve(np.abs(R)**2, np.ones(window)/window, mode='same')
        tf = R_power / (O_power + 1e-10)
        all_tf.append(tf)
    avg_tf = np.mean(all_tf, axis=0)
    return freqs, avg_tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="figs/nonlinearity_analysis.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load nonlinearity summary
    with open(NL_SUMMARY) as f:
        nl_data = json.load(f)

    # Load one pair for scatter plots
    orig_dir = PAIRED_DIR / "original"
    rec_dir = PAIRED_DIR / "recorded"
    orig_files = sorted(orig_dir.glob("*.wav"))

    orig_all, rec_all = [], []
    for of in orig_files:
        rec_file = rec_dir / f"{of.stem}.m4a"
        if not rec_file.exists():
            continue
        orig, rec = load_pair(str(of), str(rec_file))
        orig_a, rec_a = align_pair(orig, rec)
        orig_all.append(orig_a)
        rec_all.append(rec_a)

    print(f"Loaded {len(orig_all)} pairs")

    # ─── Figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Channel frequency response ──────────────────────────────
    ax_tf = fig.add_subplot(gs[0, 0:3])
    freqs, avg_tf = compute_channel_tf(orig_all, rec_all)
    tf_db = 10 * np.log10(avg_tf + 1e-10)

    ax_tf.plot(freqs, tf_db, 'k-', linewidth=0.5, alpha=0.3)
    # Smooth version
    from scipy.ndimage import uniform_filter1d
    tf_smooth = uniform_filter1d(tf_db, size=500)
    ax_tf.plot(freqs, tf_smooth, 'b-', linewidth=2, label='Channel response (smoothed)')

    # Shade passband regions
    ax_tf.axvspan(0, 300, alpha=0.15, color='red', label='Speaker rolloff (<300 Hz)')
    ax_tf.axvspan(300, 5000, alpha=0.08, color='green', label='Speaker passband (300-5000 Hz)')
    ax_tf.axvspan(5000, 8000, alpha=0.08, color='orange', label='Resonance peak (5-8 kHz)')
    ax_tf.axhline(y=-6, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_tf.text(11500, -5, '-6 dB', fontsize=8, color='gray')

    ax_tf.set_xlim(20, 12000)
    ax_tf.set_xscale('log')
    ax_tf.set_xlabel('Frequency (Hz)', fontsize=11)
    ax_tf.set_ylabel('Gain (dB)', fontsize=11)
    ax_tf.set_title('(a) Physical OTA Channel Response\n(MacBook Air → iPhone 16 Pro)', fontsize=12)
    ax_tf.legend(fontsize=8, loc='lower left')
    ax_tf.set_ylim(-40, 20)
    ax_tf.grid(True, alpha=0.3)

    # ── Panel B: R² bar chart ────────────────────────────────────────────
    ax_r2 = fig.add_subplot(gs[0, 3])
    bands = []
    r2_lin = []
    r2_poly = []
    for band_name, data in nl_data.items():
        if data.get('skipped'):
            continue
        bands.append(band_name.replace('Hz', ''))
        r2_lin.append(data['linear_r2'])
        r2_poly.append(data['nonlinear_r2'])

    x_pos = np.arange(len(bands))
    width = 0.35
    bars1 = ax_r2.bar(x_pos - width/2, r2_lin, width, label='Linear IR only',
                      color='steelblue', alpha=0.8)
    bars2 = ax_r2.bar(x_pos + width/2, r2_poly, width, label='+ Polynomial NL',
                      color='coral', alpha=0.8)

    # Add delta R² annotations
    for i, (l, p) in enumerate(zip(r2_lin, r2_poly)):
        delta = p - l
        if delta > 0.01:
            ax_r2.annotate(f'+{delta:.2f}',
                          xy=(i + width/2, p), xytext=(0, 5),
                          textcoords='offset points', ha='center', fontsize=7,
                          fontweight='bold', color='darkred')

    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels(bands, rotation=45, ha='right', fontsize=8)
    ax_r2.set_ylabel('R²', fontsize=11)
    ax_r2.set_title('(b) Model Fit per Band\n(Linear vs Nonlinear)', fontsize=12)
    ax_r2.legend(fontsize=8)
    ax_r2.set_ylim(0, 0.7)
    ax_r2.grid(True, axis='y', alpha=0.3)

    # ── Panel C-F: Per-band input/output scatter with polynomial fit ─────
    # Show the 4 significant bands
    significant_bands = [
        ('0-150Hz', 0, 150),
        ('150-300Hz', 150, 300),
        ('300-800Hz', 300, 800),
        ('800-2000Hz', 800, 2000),
    ]

    for idx, (band_name, low, high) in enumerate(significant_bands):
        if band_name not in nl_data or nl_data[band_name].get('skipped'):
            continue

        ax = fig.add_subplot(gs[1, idx])

        data = nl_data[band_name]
        coeffs = np.array(data['coeffs'])

        # Get bandpassed signals from first pair
        x_band = bandpass(orig_all[0], low, high)
        y_band = bandpass(rec_all[0], low, high)

        # Subsample for scatter
        step = max(1, len(x_band) // 3000)
        ax.scatter(x_band[::step], y_band[::step], s=1, alpha=0.15, c='steelblue',
                  rasterized=True)

        # Linear reference
        x_range = np.linspace(x_band.min(), x_band.max(), 300)
        ax.plot(x_range, coeffs[1] * x_range, 'g--', linewidth=1.5,
                alpha=0.6, label='Linear')

        # Polynomial fit
        y_poly = coeffs[0]
        for k in range(1, len(coeffs)):
            if k % 2 == 0:
                y_poly = y_poly + coeffs[k] * np.abs(x_range)**k * np.sign(x_range)
            else:
                y_poly = y_poly + coeffs[k] * x_range**k
        ax.plot(x_range, y_poly, 'r-', linewidth=2, label='Polynomial')

        delta_r2 = data['nonlinear_r2'] - data['linear_r2']
        ax.set_title(f'({chr(99+idx)}) {band_name}\n'
                     f'$\\Delta R^2$ = {delta_r2:.3f}',
                     fontsize=11)
        ax.set_xlabel('Input amplitude', fontsize=9)
        if idx == 0:
            ax.set_ylabel('Output amplitude', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Saved to {args.output}")

    # Also save PNG version
    png_path = args.output.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
