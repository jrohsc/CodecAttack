"""Extract Marshall speaker → iPhone impulse response from paired recordings.

Uses Wiener deconvolution on the paired original WAV / recorded M4A files
from experiment 4 (bandpass300_1000_marshall).

Outputs:
  data/empirical_ota_marshall/channel_impulse_response.npy
  data/empirical_ota_marshall/channel_impulse_response.wav
  data/empirical_ota_marshall/residual_noise_*.npy
  data/empirical_ota_marshall/residual_noise_std.npy
"""
import numpy as np
import soundfile as sf
import subprocess
import tempfile
from pathlib import Path
from scipy.signal import correlate, resample_poly, fftconvolve
from scipy.fft import rfft, irfft, rfftfreq
from math import gcd

SR = 24000  # EnCodec native sample rate
IR_LEN_MS = 200
IR_LEN = int(IR_LEN_MS / 1000 * SR)  # 4800 samples

# Source directories
ORIG_DIR = Path("results_empirical_ota_IR/4_exp_bandpass300_1000_marshall/eps3.0/audio")
REC_DIR = Path("results_empirical_ota_IR/4_exp_bandpass300_1000_marshall/eps3.0/audio_marshall_iphone_recording")
OUT_DIR = Path("data/empirical_ota_marshall")


def load_audio(path):
    """Load audio as mono float32 at SR Hz."""
    path = Path(path)
    if path.suffix.lower() in ('.m4a', '.aac', '.mp4'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            subprocess.run([
                'ffmpeg', '-i', str(path),
                '-f', 'wav', '-acodec', 'pcm_f32le',
                '-ac', '1', '-ar', str(SR),
                tmp.name, '-y'
            ], capture_output=True, check=True)
            wav, sr = sf.read(tmp.name)
    else:
        wav, sr = sf.read(str(path))
    if wav.ndim == 2:
        wav = wav.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return wav.astype(np.float64)


def extract_ir(orig, rec):
    """Extract IR via Wiener deconvolution. Returns (ir, delay, correlation)."""
    # Find delay via cross-correlation
    seg_len = min(len(orig), len(rec), SR * 3)
    o_seg = orig[:seg_len]
    r_seg = rec[:seg_len + SR]
    corr = correlate(r_seg, o_seg, mode='full')
    delay = int(np.argmax(np.abs(corr)) - len(o_seg) + 1)

    # Align
    if delay >= 0 and delay + len(orig) <= len(rec):
        aligned = rec[delay:delay + len(orig)]
    elif delay >= 0:
        aligned = rec[delay:]
    else:
        aligned = rec[:len(orig)]

    min_len = min(len(orig), len(aligned))
    o_trim = orig[:min_len]
    a_trim = aligned[:min_len]

    # Correlation
    pearson = float(np.corrcoef(o_trim, a_trim)[0, 1])

    # Wiener deconvolution
    n_fft = min_len
    O = rfft(o_trim, n=n_fft)
    A = rfft(a_trim, n=n_fft)
    eps_reg = 1e-3 * np.max(np.abs(O) ** 2)
    H = A * np.conj(O) / (np.abs(O) ** 2 + eps_reg)
    ir_est = irfft(H, n=n_fft)
    ir_est = ir_est[:IR_LEN]

    return ir_est, delay, pearson, o_trim, a_trim


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orig_files = sorted(ORIG_DIR.glob("*.wav"))
    print(f"Found {len(orig_files)} original WAV files")

    all_irs = []
    all_residuals = []
    results = []

    for of in orig_files:
        stem = of.stem
        rf_m4a = REC_DIR / f"{stem}.m4a"
        if not rf_m4a.exists():
            print(f"  SKIP {stem}: no recording")
            continue

        print(f"  Processing {stem}...")
        orig = load_audio(of)
        rec = load_audio(rf_m4a)

        ir_est, delay, pearson, o_trim, a_trim = extract_ir(orig, rec)
        all_irs.append(ir_est)

        # Compute residual noise: recorded - conv(original, ir)
        predicted = fftconvolve(o_trim, ir_est, mode='full')[:len(o_trim)]
        residual = a_trim - predicted
        all_residuals.append(residual)

        # Residual SNR
        snr = 10 * np.log10(np.mean(a_trim ** 2) / (np.mean(residual ** 2) + 1e-20))

        results.append({
            'name': stem,
            'delay_ms': delay / SR * 1000,
            'correlation': pearson,
            'residual_snr_db': float(snr),
        })

        print(f"    delay={delay / SR * 1000:.1f}ms, corr={pearson:.4f}, residual_snr={snr:.1f}dB")

    if not all_irs:
        print("ERROR: No pairs found!")
        return

    # Average IR across all files
    avg_ir = np.mean(all_irs, axis=0)
    print(f"\nAverage IR: peak at sample {np.argmax(np.abs(avg_ir))}, "
          f"value={avg_ir[np.argmax(np.abs(avg_ir))]:.6f}")

    # Save IR
    np.save(OUT_DIR / "channel_impulse_response.npy", avg_ir)
    sf.write(str(OUT_DIR / "channel_impulse_response.wav"), avg_ir.astype(np.float32), SR)
    print(f"Saved: {OUT_DIR / 'channel_impulse_response.npy'} ({len(avg_ir)} samples, {IR_LEN_MS}ms)")

    # Save individual IRs too (for analysis)
    for i, ir in enumerate(all_irs):
        np.save(OUT_DIR / f"ir_individual_{i:02d}.npy", ir)

    # Save residual noise bank
    for i, res in enumerate(all_residuals):
        np.save(OUT_DIR / f"residual_noise_{i:02d}.npy", res.astype(np.float32))

    # Noise std
    all_res_concat = np.concatenate(all_residuals)
    noise_std = float(np.std(all_res_concat))
    np.save(OUT_DIR / f"residual_noise_std.npy", np.array(noise_std))
    print(f"Noise std: {noise_std:.6f}")
    print(f"Saved {len(all_residuals)} residual noise files")

    # Verify: check phase in attack band
    from scipy.signal import freqz
    w, h = freqz(avg_ir, worN=4096, fs=SR)
    print(f"\n=== Marshall IR frequency response (attack band) ===")
    for freq in [300, 400, 500, 600, 700, 800, 900, 1000]:
        idx = np.argmin(np.abs(w - freq))
        mag = 20 * np.log10(np.abs(h[idx]) + 1e-20)
        phase = np.degrees(np.angle(h[idx]))
        print(f"  {freq}Hz: mag={mag:>7.1f}dB, phase={phase:>7.1f}°")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Files processed: {len(results)}")
    print(f"Mean delay: {np.mean([r['delay_ms'] for r in results]):.1f}ms")
    print(f"Mean correlation: {np.mean([r['correlation'] for r in results]):.4f}")
    print(f"Mean residual SNR: {np.mean([r['residual_snr_db'] for r in results]):.1f}dB")
    print(f"\nOutput directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
