"""Extract averaged MacBook Air -> iPhone IR from all MacBook paired recordings."""
import os, sys, subprocess, tempfile
import numpy as np
import soundfile as sf
from scipy import signal as sig

TARGET_SR = 16000
ENCODEC_SR = 24000
IR_LEN_16K = 4800  # 300ms at 16kHz

BASE = "${REPO_ROOT}/results_empirical_ota_IR"

# (digital_audio_dir, macbook_recording_dir)
MACBOOK_PAIRS = [
    ("1_exp_empirical_nonlinearity/audio",
     "1_exp_empirical_nonlinearity/audio_mac_iphone_record_ota"),
    ("3_exp_physical_channel_fir/audio",
     "3_exp_physical_channel_fir/audio_macbook_iphone_recording"),
    ("8_exp_marshall_combined/eps5.0/audio",
     "8_exp_marshall_combined/eps5.0/audio_macbook_iphone_recording"),
    ("9_exp_ecc_marshall_ir/eps5.0/simplified_rep3/audio",
     "9_exp_ecc_marshall_ir/eps5.0/simplified_rep3/audio_macbook_iphone_recording"),
    ("exp1_1000steps/audio",
     "exp1_1000steps/audio_mac_iphone_recording"),
    ("exp1_10short_eps3.0_1000steps/audio",
     "exp1_10short_eps3.0_1000steps/audio_mac_iphone_record_ota"),
    ("exp_bandpass300_4000_eps3.0_1000steps/audio",
     "exp_bandpass300_4000_eps3.0_1000steps/audio_mac_iphone_record_ota"),
    ("exp_bandpass300_4000_nonlinear_eps3.0_1000steps/audio",
     "exp_bandpass300_4000_nonlinear_eps3.0_1000steps/audio_mac_iphone_record_ota"),
]


def load_audio(path, target_sr=TARGET_SR):
    if path.endswith('.m4a'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
                           '-acodec', 'pcm_f32le', tmp], capture_output=True, check=True)
            data, sr = sf.read(tmp, dtype='float32')
            return data
        finally:
            os.unlink(tmp)
    else:
        data, sr = sf.read(path, dtype='float32')
        if sr != target_sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, sr)
            data = resample_poly(data, target_sr // g, sr // g)
        if data.ndim > 1:
            data = data[:, 0]
        return data


def extract_ir_wiener(digital, physical, ir_len=IR_LEN_16K, reg=1e-3):
    """Extract IR via Wiener deconvolution."""
    min_len = min(len(digital), len(physical))
    d = digital[:min_len].astype(np.float64)
    p = physical[:min_len].astype(np.float64)

    # Align via cross-correlation
    corr = np.correlate(p, d, mode='full')
    delay = np.argmax(np.abs(corr)) - min_len + 1

    if delay > 0:
        d_a = d[:-delay] if delay < len(d) else d
        p_a = p[delay:delay+len(d_a)]
    elif delay < 0:
        p_a = p[:delay] if delay > -len(p) else p
        d_a = d[-delay:-delay+len(p_a)]
    else:
        d_a, p_a = d, p
    n = min(len(d_a), len(p_a))
    d_a, p_a = d_a[:n], p_a[:n]

    n_fft = n + ir_len
    D = np.fft.rfft(d_a, n=n_fft)
    P = np.fft.rfft(p_a, n=n_fft)
    H = (P * np.conj(D)) / (np.abs(D)**2 + reg * np.max(np.abs(D)**2))
    ir = np.fft.irfft(H, n=n_fft)[:ir_len]

    return ir, delay


def extract_noise_residual(digital, physical, ir, delay):
    """Extract noise residual after IR convolution."""
    min_len = min(len(digital), len(physical))
    d = digital[:min_len].astype(np.float64)
    p = physical[:min_len].astype(np.float64)

    # Align
    if delay > 0:
        d_a = d[:-delay] if delay < len(d) else d
        p_a = p[delay:delay+len(d_a)]
    elif delay < 0:
        p_a = p[:delay] if delay > -len(p) else p
        d_a = d[-delay:-delay+len(p_a)]
    else:
        d_a, p_a = d, p
    n = min(len(d_a), len(p_a))
    d_a, p_a = d_a[:n], p_a[:n]

    # Predict physical from digital via IR
    predicted = sig.fftconvolve(d_a, ir, mode='full')[:n]
    residual = p_a - predicted
    return residual.astype(np.float32)


def main():
    os.chdir("${REPO_ROOT}")

    all_irs = []
    all_residuals = []
    labels = []
    polarities = {"positive": 0, "negative": 0}

    print("=" * 70)
    print("  Extracting MacBook Air -> iPhone IR")
    print("=" * 70)

    for dig_dir_rel, rec_dir_rel in MACBOOK_PAIRS:
        dig_dir = os.path.join(BASE, dig_dir_rel)
        rec_dir = os.path.join(BASE, rec_dir_rel)
        if not os.path.isdir(dig_dir) or not os.path.isdir(rec_dir):
            print(f"\n  SKIP (missing dir): {rec_dir_rel}")
            continue

        print(f"\n  {rec_dir_rel}:")

        rec_files = {}
        for f in sorted(os.listdir(rec_dir)):
            if f.endswith('.m4a'):
                base = f.replace('.m4a', '').split('_volume')[0]
                rec_files[base] = f
        dig_files = {f.replace('.wav', ''): f for f in sorted(os.listdir(dig_dir)) if f.endswith('.wav')}

        for base_name in sorted(set(rec_files.keys()) & set(dig_files.keys())):
            dig_path = os.path.join(dig_dir, dig_files[base_name])
            rec_path = os.path.join(rec_dir, rec_files[base_name])

            try:
                digital = load_audio(dig_path)
                physical = load_audio(rec_path)
            except Exception as e:
                print(f"    SKIP (load error): {base_name}: {e}")
                continue

            ir, delay = extract_ir_wiener(digital, physical)
            peak_val = ir[np.argmax(np.abs(ir))]

            if peak_val > 0:
                polarities["positive"] += 1
            else:
                polarities["negative"] += 1

            # Compute prediction quality
            min_len = min(len(digital), len(physical))
            d = digital[:min_len].astype(np.float64)
            p = physical[:min_len].astype(np.float64)
            if delay > 0:
                d_a = d[:-delay] if delay < len(d) else d
                p_a = p[delay:delay+len(d_a)]
            elif delay < 0:
                p_a = p[:delay] if delay > -len(p) else p
                d_a = d[-delay:-delay+len(p_a)]
            else:
                d_a, p_a = d, p
            n = min(len(d_a), len(p_a))
            d_a, p_a = d_a[:n], p_a[:n]
            predicted = sig.fftconvolve(d_a, ir, mode='full')[:n]
            corr = np.corrcoef(predicted, p_a)[0, 1]

            all_irs.append(ir)
            labels.append(f"{dig_dir_rel.split('/')[0]}_{base_name}")
            print(f"    {base_name}: peak={peak_val:+.4f} delay={delay/TARGET_SR*1000:.0f}ms corr={corr:.3f}")

            # Extract noise residual
            residual = extract_noise_residual(digital, physical, ir, delay)
            all_residuals.append(residual)

    print(f"\n{'='*70}")
    print(f"  Total IRs: {len(all_irs)}")
    print(f"  Polarities: {polarities}")

    if len(all_irs) == 0:
        print("ERROR: No IRs extracted!")
        return

    # Check polarity consistency — use majority polarity
    majority_polarity = "negative" if polarities["negative"] >= polarities["positive"] else "positive"
    print(f"  Majority polarity: {majority_polarity}")

    # Optionally filter to majority polarity
    filtered_irs = []
    filtered_labels = []
    for ir, label in zip(all_irs, labels):
        peak_val = ir[np.argmax(np.abs(ir))]
        is_negative = peak_val < 0
        if (majority_polarity == "negative" and is_negative) or \
           (majority_polarity == "positive" and not is_negative):
            filtered_irs.append(ir)
            filtered_labels.append(label)
        else:
            print(f"  EXCLUDED (minority polarity): {label}")

    use_irs = filtered_irs if len(filtered_irs) >= len(all_irs) * 0.5 else all_irs
    use_labels = filtered_labels if len(filtered_irs) >= len(all_irs) * 0.5 else labels
    print(f"  Using {len(use_irs)} IRs after polarity filtering")

    # Pairwise correlation
    if len(use_irs) > 1:
        corrs = []
        for i in range(len(use_irs)):
            for j in range(i+1, len(use_irs)):
                c = np.corrcoef(use_irs[i], use_irs[j])[0, 1]
                corrs.append(c)
        print(f"  Pairwise IR correlation: mean={np.mean(corrs):.3f} std={np.std(corrs):.3f}")

    # Compute average IR
    avg_ir_16k = np.mean(use_irs, axis=0)
    peak_idx = np.argmax(np.abs(avg_ir_16k))
    print(f"\n  Average IR (16kHz):")
    print(f"    Length: {len(avg_ir_16k)} samples = {len(avg_ir_16k)/TARGET_SR*1000:.0f}ms")
    print(f"    Peak: {avg_ir_16k[peak_idx]:+.6f} @ sample {peak_idx}")

    # Resample to 24kHz
    ir_len_24k = int(IR_LEN_16K * ENCODEC_SR / TARGET_SR)  # 7200 samples
    avg_ir_24k = sig.resample(avg_ir_16k, ir_len_24k)
    peak_idx_24k = np.argmax(np.abs(avg_ir_24k))
    print(f"\n  Average IR (24kHz):")
    print(f"    Length: {len(avg_ir_24k)} samples = {len(avg_ir_24k)/ENCODEC_SR*1000:.0f}ms")
    print(f"    Peak: {avg_ir_24k[peak_idx_24k]:+.6f} @ sample {peak_idx_24k}")

    # Save
    out_dir = "data/empirical_ota_macbook"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "channel_impulse_response.npy"), avg_ir_24k)
    np.save(os.path.join(out_dir, "channel_impulse_response_16k.npy"), avg_ir_16k)
    np.save(os.path.join(out_dir, "all_individual_irs_16k.npy"), np.array(use_irs))
    print(f"\n  Saved to {out_dir}/")

    # Save individual IRs
    for i, (ir, label) in enumerate(zip(use_irs, use_labels)):
        np.save(os.path.join(out_dir, f"ir_individual_{i:02d}.npy"), ir)

    # Save noise residuals (trim to min length)
    if all_residuals:
        min_res_len = min(len(r) for r in all_residuals)
        for i, r in enumerate(all_residuals[:20]):  # cap at 20
            # Resample residual to 24kHz
            r_24k = sig.resample(r[:min_res_len], int(min_res_len * ENCODEC_SR / TARGET_SR))
            np.save(os.path.join(out_dir, f"residual_noise_{i:02d}.npy"), r_24k.astype(np.float32))
        print(f"  Saved {min(len(all_residuals), 20)} noise residuals")

    # Save IR as WAV for listening
    sf.write(os.path.join(out_dir, "channel_impulse_response.wav"),
             avg_ir_24k.astype(np.float32), ENCODEC_SR)

    # Compare with Marshall IR
    marshall_ir_path = "data/empirical_ota/channel_impulse_response.npy"
    if os.path.exists(marshall_ir_path):
        marshall_ir = np.load(marshall_ir_path)
        min_len = min(len(marshall_ir), len(avg_ir_24k))
        corr = np.corrcoef(marshall_ir[:min_len], avg_ir_24k[:min_len])[0, 1]
        print(f"\n  Correlation MacBook vs Marshall IR: {corr:.4f}")
        print(f"  Marshall peak: {marshall_ir[np.argmax(np.abs(marshall_ir))]:+.6f}")
        print(f"  MacBook peak:  {avg_ir_24k[peak_idx_24k]:+.6f}")


if __name__ == "__main__":
    main()
