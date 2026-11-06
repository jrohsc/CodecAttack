"""
Extract per-band nonlinearity from paired physical OTA recordings.

Uses the directly-played pairs (MacBook Air -> iPhone 16 Pro) to fit a
Hammerstein model: per-subband polynomial nonlinearity followed by linear IR.

    y(t) = IR * [a1*x(t) + a2*x(t)^2 + a3*x(t)^3]  (per frequency band)

The fitted model captures frequency-dependent speaker distortion that the
linear IR alone cannot explain.

Usage:
    python extract_nonlinearity.py [--poly-order 3] [--n-bands 6] [--plot]
"""
import os
import sys
import json
import argparse
import numpy as np
import subprocess
from pathlib import Path
from scipy import signal as scipy_signal
import soundfile as sf

# ─── Configuration ───────────────────────────────────────────────────────
SR = 24000  # All processing at 24 kHz (EnCodec native rate)
PAIRED_DIR = Path(__file__).parent / "soundcloud_recorded" / "directly_played_from_mac_ir_trained_qwen2_audio_calm_1_eps_5.0"
OUTPUT_DIR = Path(__file__).parent / "data" / "empirical_ota"

# Frequency bands (Hz) matching physical speaker/mic response regions
BAND_EDGES = [
    (0, 150),       # sub-bass — speaker can't reproduce, expect heavy distortion
    (150, 300),     # bass — near speaker rolloff, significant distortion
    (300, 800),     # low-mid — speaker passband starts
    (800, 2000),    # mid — speaker sweet spot
    (2000, 4000),   # upper-mid — still good
    (4000, 8000),   # presence — starts rolling off
    (8000, 12000),  # brilliance/air — high freq residual
]


def load_pair(orig_path: str, rec_path: str) -> tuple:
    """Load an original/recorded pair, convert to 24kHz mono."""
    # Original is WAV at 24kHz
    orig, sr_orig = sf.read(orig_path)
    if orig.ndim > 1:
        orig = orig.mean(axis=1)

    # Recorded is M4A — decode via ffmpeg to raw float32
    rec_wav = rec_path.replace('.m4a', '_tmp.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', rec_path, '-ar', str(SR), '-ac', '1', rec_wav],
        capture_output=True, check=True,
    )
    rec, sr_rec = sf.read(rec_wav)
    if rec.ndim > 1:
        rec = rec.mean(axis=1)
    os.remove(rec_wav)

    # Resample original if needed
    if sr_orig != SR:
        from scipy.signal import resample_poly
        orig = resample_poly(orig, SR, sr_orig).astype(np.float32)

    return orig.astype(np.float32), rec.astype(np.float32)


def align_pair(orig: np.ndarray, rec: np.ndarray) -> tuple:
    """Align recorded to original using cross-correlation. Returns aligned pair."""
    # Use first 3s for correlation
    seg_len = min(3 * SR, len(orig), len(rec))
    corr = np.correlate(rec[:seg_len + SR], orig[:seg_len], mode='full')
    delay = np.argmax(np.abs(corr)) - seg_len + 1

    if delay > 0:
        rec_aligned = rec[delay:]
    else:
        rec_aligned = np.pad(rec, (-delay, 0))

    min_len = min(len(orig), len(rec_aligned))
    return orig[:min_len], rec_aligned[:min_len]


def bandpass(audio: np.ndarray, low: float, high: float, sr: int = SR, order: int = 5) -> np.ndarray:
    """Apply bandpass filter. Handles edge cases for low=0 or high>=Nyquist."""
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


def fit_polynomial_per_band(
    orig_all: list,
    rec_all: list,
    band_edges: list,
    poly_order: int = 3,
) -> dict:
    """
    Fit per-band polynomial nonlinearity from multiple paired recordings.

    For each band, we:
    1. Bandpass filter both original and recorded
    2. Estimate the linear IR for this band (Wiener deconvolution)
    3. Compute linear prediction: y_lin = IR * x_band
    4. Fit polynomial: y_band ~ a1*(IR*x) + a2*(IR*x^2) + a3*(IR*x^3)

    Returns dict with per-band coefficients and diagnostics.
    """
    results = {}

    for (low, high) in band_edges:
        band_name = f"{low}-{high}Hz"
        print(f"\n{'='*60}")
        print(f"Band: {band_name}")

        # Collect filtered signals from all pairs
        all_x_band = []
        all_y_band = []

        for orig, rec in zip(orig_all, rec_all):
            x_band = bandpass(orig, low, high)
            y_band = bandpass(rec, low, high)
            all_x_band.append(x_band)
            all_y_band.append(y_band)

        # Concatenate all pairs for joint fitting
        x_cat = np.concatenate(all_x_band)
        y_cat = np.concatenate(all_y_band)

        # Skip near-silent bands
        x_rms = np.sqrt(np.mean(x_cat**2))
        y_rms = np.sqrt(np.mean(y_cat**2))
        if x_rms < 1e-8 or y_rms < 1e-8:
            print(f"  SKIP: near-silent (x_rms={x_rms:.2e}, y_rms={y_rms:.2e})")
            results[band_name] = {
                'low': low, 'high': high,
                'coeffs': [0.0] * (poly_order + 1),
                'linear_r2': 0.0, 'nonlinear_r2': 0.0,
                'improvement': 0.0,
                'skipped': True,
            }
            continue

        # Step 1: Estimate band-specific IR via Wiener deconvolution
        n_fft = len(x_cat)
        X = np.fft.rfft(x_cat, n=n_fft)
        Y = np.fft.rfft(y_cat, n=n_fft)
        eps_reg = 1e-3 * np.max(np.abs(X)**2)
        H = Y * np.conj(X) / (np.abs(X)**2 + eps_reg)
        ir_band = np.fft.irfft(H, n=n_fft)
        # Truncate IR to 200ms
        ir_len = int(0.2 * SR)
        ir_band = ir_band[:ir_len]

        # Step 2: Linear prediction
        y_lin = scipy_signal.fftconvolve(x_cat, ir_band, mode='full')[:len(x_cat)]
        residual_lin = y_cat - y_lin
        ss_total = np.sum(y_cat**2)
        ss_res_lin = np.sum(residual_lin**2)
        r2_lin = 1 - ss_res_lin / (ss_total + 1e-10)

        print(f"  Linear IR only: R2 = {r2_lin:.4f}, residual RMS = {np.sqrt(np.mean(residual_lin**2)):.6f}")

        # Step 3: Fit polynomial nonlinearity
        # Build design matrix: [1, conv(x, ir), conv(x^2, ir), conv(x^3, ir)]
        design_cols = []
        for k in range(1, poly_order + 1):
            x_pow = np.sign(x_cat) * np.abs(x_cat)**k  # Odd-symmetric for odd powers
            col = scipy_signal.fftconvolve(x_pow, ir_band, mode='full')[:len(x_cat)]
            design_cols.append(col)

        # Add constant (DC offset from speaker/mic)
        design_cols.insert(0, np.ones(len(x_cat)))
        A_mat = np.column_stack(design_cols)

        # Solve via least squares
        # Subsample for speed if very long
        if len(x_cat) > 500000:
            idx = np.random.default_rng(42).choice(len(x_cat), 500000, replace=False)
            idx.sort()
            coeffs, _, _, _ = np.linalg.lstsq(A_mat[idx], y_cat[idx], rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, y_cat, rcond=None)

        # Evaluate full polynomial prediction
        y_poly = A_mat @ coeffs
        residual_poly = y_cat - y_poly
        ss_res_poly = np.sum(residual_poly**2)
        r2_poly = 1 - ss_res_poly / (ss_total + 1e-10)
        improvement = r2_poly - r2_lin

        print(f"  Polynomial (order {poly_order}): R2 = {r2_poly:.4f}")
        print(f"  Improvement over linear: dR2 = {improvement:.4f}")
        print(f"  Coefficients: {['%.6f' % c for c in coeffs]}")

        # Normalized coefficients (relative to a1)
        if abs(coeffs[1]) > 1e-10:
            norm_coeffs = coeffs / coeffs[1]
            print(f"  Normalized (rel to a1): {['%.4f' % c for c in norm_coeffs]}")

        results[band_name] = {
            'low': low, 'high': high,
            'coeffs': coeffs.tolist(),  # [dc, a1, a2, a3, ...]
            'ir_band': ir_band.tolist(),
            'linear_r2': float(r2_lin),
            'nonlinear_r2': float(r2_poly),
            'improvement': float(improvement),
            'x_rms': float(x_rms),
            'y_rms': float(y_rms),
            'residual_rms_linear': float(np.sqrt(np.mean(residual_lin**2))),
            'residual_rms_poly': float(np.sqrt(np.mean(residual_poly**2))),
            'skipped': False,
        }

    return results


def save_model(results: dict, output_path: Path):
    """Save the fitted nonlinearity model as JSON for use in attack/eval."""
    model = {
        'bands': [],
        'description': 'Per-band Hammerstein nonlinearity fitted from MacBook->iPhone OTA pairs',
        'sr': SR,
    }
    for band_name, r in results.items():
        entry = {
            'band_name': band_name,
            'low': r['low'],
            'high': r['high'],
            'coeffs': r['coeffs'],
            'linear_r2': r['linear_r2'],
            'nonlinear_r2': r['nonlinear_r2'],
            'improvement': r['improvement'],
            'skipped': r['skipped'],
        }
        if not r['skipped'] and 'ir_band' in r:
            entry['ir_band'] = r['ir_band']
        model['bands'].append(entry)

    model_path = output_path / 'nonlinearity_model.json'
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)

    # Also save a compact summary
    summary = {}
    for band_name, r in results.items():
        summary[band_name] = {
            'coeffs': r['coeffs'],
            'linear_r2': r['linear_r2'],
            'nonlinear_r2': r['nonlinear_r2'],
            'improvement_r2': r['improvement'],
            'skipped': r['skipped'],
        }
    with open(output_path / 'nonlinearity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nModel saved to {model_path}")
    print(f"Summary saved to {output_path / 'nonlinearity_summary.json'}")


def plot_results(results: dict, orig_all, rec_all, out_dir=OUTPUT_DIR):
    """Visualize the nonlinearity per band."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    active = [(k, v) for k, v in results.items() if not v['skipped']]
    n_bands = len(active)
    fig, axes = plt.subplots(2, n_bands, figsize=(4 * n_bands, 8))
    if n_bands == 1:
        axes = axes.reshape(2, 1)

    for idx, (band_name, r) in enumerate(active):
        low, high = r['low'], r['high']
        coeffs = np.array(r['coeffs'])

        # Plot 1: Input-output scatter (using first pair)
        ax = axes[0, idx]
        x_band = bandpass(orig_all[0], low, high)
        y_band = bandpass(rec_all[0], low, high)
        # Subsample for plotting
        step = max(1, len(x_band) // 5000)
        ax.scatter(x_band[::step], y_band[::step], s=1, alpha=0.3, c='blue')

        # Overlay fitted polynomial (without IR, just shape)
        x_range = np.linspace(x_band.min(), x_band.max(), 200)
        y_fit = coeffs[0]  # DC
        for k in range(1, len(coeffs)):
            y_fit = y_fit + coeffs[k] * np.sign(x_range) * np.abs(x_range)**k
        ax.plot(x_range, y_fit, 'r-', linewidth=2, label='poly fit')
        ax.set_title(f'{band_name}\ndR2={r["improvement"]:.4f}')
        ax.set_xlabel('Input (bandpassed)')
        ax.set_ylabel('Output (bandpassed)')
        ax.legend(fontsize=8)

        # Plot 2: R2 comparison
        ax2 = axes[1, idx]
        bars = ax2.bar(['Linear', 'Polynomial'], [r['linear_r2'], r['nonlinear_r2']],
                       color=['steelblue', 'coral'])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('R2')
        ax2.set_title(f'{band_name}')
        for bar, val in zip(bars, [r['linear_r2'], r['nonlinear_r2']]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / 'nonlinearity_analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract per-band nonlinearity from OTA pairs")
    parser.add_argument("--poly-order", type=int, default=3, help="Polynomial order (default: 3)")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument("--paired-dir", type=str, default=str(PAIRED_DIR),
                        help="Directory with original/ and recorded/ subdirs")
    parser.add_argument("--orig-dir", type=str, default=None,
                        help="Override: directory with original WAV files")
    parser.add_argument("--rec-dir", type=str, default=None,
                        help="Override: directory with recorded M4A files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: data/empirical_ota/)")
    args = parser.parse_args()

    if args.orig_dir and args.rec_dir:
        orig_dir = Path(args.orig_dir)
        rec_dir = Path(args.rec_dir)
    else:
        paired_dir = Path(args.paired_dir)
        orig_dir = paired_dir / "original"
        rec_dir = paired_dir / "recorded"

    # Load all pairs
    orig_files = sorted(orig_dir.glob("*.wav"))
    print(f"Found {len(orig_files)} paired recordings in {orig_dir}")

    orig_all = []
    rec_all = []
    for of in orig_files:
        stem = of.stem
        rec_file = rec_dir / f"{stem}.m4a"
        if not rec_file.exists():
            print(f"  SKIP {stem}: no matching recording")
            continue

        print(f"  Loading {stem}...")
        orig, rec = load_pair(str(of), str(rec_file))
        orig_aligned, rec_aligned = align_pair(orig, rec)
        print(f"    Aligned: {len(orig_aligned)} samples ({len(orig_aligned)/SR:.1f}s)")

        orig_all.append(orig_aligned)
        rec_all.append(rec_aligned)

    if not orig_all:
        print("ERROR: No valid pairs found!")
        sys.exit(1)

    print(f"\nFitting per-band nonlinearity (order={args.poly_order}) from {len(orig_all)} pairs...")

    results = fit_polynomial_per_band(orig_all, rec_all, BAND_EDGES, args.poly_order)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Band':<15} {'Linear R2':>10} {'Poly R2':>10} {'dR2':>10} {'Significant?':>12}")
    print(f"{'='*80}")
    for band_name, r in results.items():
        if r['skipped']:
            print(f"{band_name:<15} {'SKIPPED':>10}")
            continue
        sig = "YES" if r['improvement'] > 0.01 else "no"
        print(f"{band_name:<15} {r['linear_r2']:>10.4f} {r['nonlinear_r2']:>10.4f} {r['improvement']:>10.4f} {sig:>12}")

    # Save
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(results, out_dir)

    if args.plot:
        plot_results(results, orig_all, rec_all, out_dir)


if __name__ == "__main__":
    main()
