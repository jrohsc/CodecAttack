"""Extract IRs from ALL available paired recordings and analyze consistency.
Key questions:
1. Is the polarity inversion consistent?
2. Are the IRs correlated across recordings?
3. Is there a stable component we can use?
4. Cross-validation: does IR from pair A predict pair B?
"""
import os, sys, subprocess, tempfile
import numpy as np
import soundfile as sf
from scipy import signal
from collections import defaultdict
import json

TARGET_SR = 16000
IR_LEN = 4800  # 300ms at 16kHz

BASE = "${REPO_ROOT}/results_empirical_ota_IR"

# All paired directories: (digital_dir, recording_dir, label)
PAIRS = [
    ("8_exp_marshall_combined/eps5.0/audio",
     "8_exp_marshall_combined/eps5.0/audio_marshall_iphone_recording", "exp8_marshall"),
    ("8_exp_marshall_combined/eps5.0/audio",
     "8_exp_marshall_combined/eps5.0/audio_macbook_iphone_recording", "exp8_macbook"),
    ("7_exp_marshall_ir_timeshift/eps3.0/audio",
     "7_exp_marshall_ir_timeshift/eps3.0/audio_marshall_iphone_record", "exp7_marshall"),
    ("6_exp_marshall_ir_fix/eps3.0/audio",
     "6_exp_marshall_ir_fix/eps3.0/audio_marshall_iphone_recording", "exp6_marshall"),
    ("5_exp_ecc_redundancy_marshall/simplified_rep3/audio",
     "5_exp_ecc_redundancy_marshall/simplified_rep3/audio_marshall_iphone_recording", "exp5_marshall"),
    ("4_exp_bandpass300_1000_marshall/eps3.0/audio",
     "4_exp_bandpass300_1000_marshall/eps3.0/audio_marshall_iphone_recording", "exp4_marshall"),
    ("3_exp_physical_channel_fir/audio",
     "3_exp_physical_channel_fir/marshall_iphone", "exp3_marshall"),
    ("3_exp_physical_channel_fir/audio",
     "3_exp_physical_channel_fir/audio_macbook_iphone_recording", "exp3_macbook"),
    ("1_exp_empirical_nonlinearity/audio",
     "1_exp_empirical_nonlinearity/audio_marshall_iphone_record", "exp1_marshall"),
    ("1_exp_empirical_nonlinearity/audio",
     "1_exp_empirical_nonlinearity/audio_mac_iphone_record_ota", "exp1_macbook"),
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
        return data


def extract_ir_wiener(digital, physical, ir_len=IR_LEN, reg=1e-3):
    """Extract IR via Wiener deconvolution in frequency domain."""
    # First align by cross-correlation
    min_len = min(len(digital), len(physical))
    d = digital[:min_len].astype(np.float64)
    p = physical[:min_len].astype(np.float64)

    corr = np.correlate(p, d, mode='full')
    delay = np.argmax(np.abs(corr)) - min_len + 1

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

    # Wiener deconvolution: H = (P * conj(D)) / (|D|^2 + reg)
    n_fft = n + ir_len
    D = np.fft.rfft(d_a, n=n_fft)
    P = np.fft.rfft(p_a, n=n_fft)
    H = (P * np.conj(D)) / (np.abs(D)**2 + reg * np.max(np.abs(D)**2))
    ir = np.fft.irfft(H, n=n_fft)[:ir_len]

    # Also return alignment info
    peak_idx = np.argmax(np.abs(ir))
    peak_val = ir[peak_idx]
    polarity = "NEGATIVE" if peak_val < 0 else "POSITIVE"

    return ir, delay, peak_idx, peak_val, polarity, np.corrcoef(d_a, p_a)[0, 1]


def predict_with_ir(digital, ir, physical, sr=TARGET_SR):
    """Apply IR to digital and measure correlation with physical."""
    predicted = signal.fftconvolve(digital.astype(np.float64), ir, mode='full')[:len(digital)]
    # RMS normalize
    in_rms = np.sqrt(np.mean(digital**2) + 1e-10)
    pred_rms = np.sqrt(np.mean(predicted**2) + 1e-10)
    predicted = predicted * (in_rms / pred_rms)

    # Align predicted to physical
    min_len = min(len(predicted), len(physical))
    pred = predicted[:min_len]
    phys = physical[:min_len].astype(np.float64)

    corr_full = np.correlate(phys, pred, mode='full')
    delay = np.argmax(np.abs(corr_full)) - min_len + 1

    if delay > 0:
        pr = pred[:-delay] if delay < len(pred) else pred
        ph = phys[delay:delay+len(pr)]
    elif delay < 0:
        ph = phys[:delay] if delay > -len(phys) else phys
        pr = pred[-delay:-delay+len(ph)]
    else:
        pr, ph = pred, phys
    n = min(len(pr), len(ph))
    if n < 100:
        return 0.0
    return np.corrcoef(pr[:n], ph[:n])[0, 1]


def main():
    os.chdir("${REPO_ROOT}")

    all_irs = []  # (label, cmd, ir, delay, peak_idx, peak_val, polarity, raw_corr)
    all_pairs = []  # (label, cmd, digital_path, physical_path)

    print("="*90)
    print("STEP 1: EXTRACT IR FROM EVERY PAIRED RECORDING")
    print("="*90)

    for dig_dir_rel, rec_dir_rel, label in PAIRS:
        dig_dir = os.path.join(BASE, dig_dir_rel)
        rec_dir = os.path.join(BASE, rec_dir_rel)
        if not os.path.isdir(dig_dir) or not os.path.isdir(rec_dir):
            continue

        # Find matching pairs
        rec_files = {f.replace('.m4a', '').replace('.wav', '').split('_volume')[0].split('_70')[0].split('_80')[0].split('_90')[0]: f
                     for f in sorted(os.listdir(rec_dir)) if f.endswith('.m4a')}
        dig_files = {f.replace('.wav', ''): f
                     for f in sorted(os.listdir(dig_dir)) if f.endswith('.wav')}

        for base_name in sorted(set(rec_files.keys()) & set(dig_files.keys())):
            dig_path = os.path.join(dig_dir, dig_files[base_name])
            rec_path = os.path.join(rec_dir, rec_files[base_name])

            digital = load_audio(dig_path)
            physical = load_audio(rec_path)

            ir, delay, peak_idx, peak_val, polarity, raw_corr = extract_ir_wiener(digital, physical)
            cmd = base_name.split('__')[1] if '__' in base_name else base_name

            all_irs.append((label, cmd, ir, delay, peak_idx, peak_val, polarity, raw_corr))
            all_pairs.append((label, cmd, dig_path, rec_path))

            print(f"  [{label}] {cmd:>20}: delay={delay/TARGET_SR*1000:>7.1f}ms, "
                  f"peak={peak_val:+.4f} @{peak_idx} ({polarity}), raw_corr={raw_corr:+.3f}")

    print(f"\nTotal pairs extracted: {len(all_irs)}")

    # Separate Marshall vs MacBook
    marshall_irs = [(l, c, ir, d, pi, pv, pol, rc) for l, c, ir, d, pi, pv, pol, rc in all_irs if 'marshall' in l]
    macbook_irs = [(l, c, ir, d, pi, pv, pol, rc) for l, c, ir, d, pi, pv, pol, rc in all_irs if 'macbook' in l]

    print(f"\nMarshall pairs: {len(marshall_irs)}, MacBook pairs: {len(macbook_irs)}")

    # =========================================================================
    print(f"\n{'='*90}")
    print("STEP 2: POLARITY CONSISTENCY")
    print("="*90)
    for speaker, irs in [("Marshall", marshall_irs), ("MacBook", macbook_irs)]:
        if not irs:
            continue
        n_neg = sum(1 for _, _, _, _, _, pv, _, _ in irs if pv < 0)
        n_pos = len(irs) - n_neg
        print(f"\n  {speaker}: {n_neg} negative peak, {n_pos} positive peak "
              f"({'CONSISTENT' if n_neg == len(irs) or n_pos == len(irs) else 'INCONSISTENT'})")
        for l, c, _, _, pi, pv, pol, rc in irs:
            print(f"    [{l}] {c:>20}: peak={pv:+.6f} @{pi}, raw_corr={rc:+.3f}")

    # =========================================================================
    print(f"\n{'='*90}")
    print("STEP 3: IR-TO-IR CORRELATION (within Marshall)")
    print("="*90)
    if len(marshall_irs) >= 2:
        ir_vecs = np.array([ir for _, _, ir, _, _, _, _, _ in marshall_irs])
        # Correlation matrix
        n = len(ir_vecs)
        corr_matrix = np.corrcoef(ir_vecs)
        # Upper triangle (exclude diagonal)
        upper = []
        for i in range(n):
            for j in range(i+1, n):
                upper.append(corr_matrix[i, j])
        print(f"  Pairwise IR correlations: mean={np.mean(upper):.4f}, "
              f"std={np.std(upper):.4f}, min={np.min(upper):.4f}, max={np.max(upper):.4f}")

        # Group by experiment
        exp_groups = defaultdict(list)
        for idx, (l, c, ir, _, _, _, _, _) in enumerate(marshall_irs):
            exp_groups[l].append(idx)

        within_exp = []
        across_exp = []
        for i in range(n):
            for j in range(i+1, n):
                l_i = marshall_irs[i][0]
                l_j = marshall_irs[j][0]
                if l_i == l_j:
                    within_exp.append(corr_matrix[i, j])
                else:
                    across_exp.append(corr_matrix[i, j])

        if within_exp:
            print(f"  Within-experiment IR corr: mean={np.mean(within_exp):.4f} (n={len(within_exp)})")
        if across_exp:
            print(f"  Across-experiment IR corr: mean={np.mean(across_exp):.4f} (n={len(across_exp)})")

        # Show per-experiment group
        print(f"\n  Per-experiment IR consistency:")
        for exp_label, indices in sorted(exp_groups.items()):
            if len(indices) >= 2:
                sub_corrs = []
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        sub_corrs.append(corr_matrix[indices[i], indices[j]])
                cmds = [marshall_irs[idx][1] for idx in indices]
                print(f"    {exp_label} ({len(indices)} files): "
                      f"mean_corr={np.mean(sub_corrs):.4f}, range=[{np.min(sub_corrs):.4f}, {np.max(sub_corrs):.4f}]")

    # =========================================================================
    print(f"\n{'='*90}")
    print("STEP 4: CROSS-VALIDATION (how well does one IR predict another recording?)")
    print("="*90)

    # For Marshall pairs, use leave-one-out: extract IR from all-but-one, predict the held-out
    if len(marshall_irs) >= 3:
        marshall_pairs = [(l, c, dp, pp) for (l, c, dp, pp), (l2, _, _, _, _, _, _, _)
                         in zip(all_pairs, all_irs) if 'marshall' in l]

        # Average IR from all Marshall pairs
        avg_ir = np.mean([ir for _, _, ir, _, _, _, _, _ in marshall_irs], axis=0)
        print(f"\n  Average IR peak: {avg_ir[np.argmax(np.abs(avg_ir))]:+.6f} "
              f"@ sample {np.argmax(np.abs(avg_ir))}")

        # Test average IR on each pair
        print(f"\n  Average IR prediction accuracy:")
        avg_corrs = []
        for (l, c, dp, pp), (_, _, ir_individual, _, _, _, _, _) in zip(
                [(l, c, dp, pp) for l, c, dp, pp in all_pairs if 'marshall' in l],
                marshall_irs):
            digital = load_audio(dp)
            physical = load_audio(pp)

            # Test with average IR
            avg_pred_corr = predict_with_ir(digital, avg_ir, physical)
            # Test with own IR (sanity check)
            own_pred_corr = predict_with_ir(digital, ir_individual, physical)

            avg_corrs.append(avg_pred_corr)
            print(f"    [{l}] {c:>20}: avg_ir_corr={avg_pred_corr:+.3f}, own_ir_corr={own_pred_corr:+.3f}")

        print(f"\n  Average IR prediction: mean={np.mean(avg_corrs):.3f}")

        # Leave-one-out cross-validation
        print(f"\n  Leave-one-out cross-validation:")
        loo_corrs = []
        for i in range(len(marshall_irs)):
            # Average IR from all except i
            other_irs = [ir for j, (_, _, ir, _, _, _, _, _) in enumerate(marshall_irs) if j != i]
            loo_ir = np.mean(other_irs, axis=0)

            l, c, dp, pp = [(l, c, dp, pp) for l, c, dp, pp in all_pairs if 'marshall' in l][i]
            digital = load_audio(dp)
            physical = load_audio(pp)

            pred_corr = predict_with_ir(digital, loo_ir, physical)
            loo_corrs.append(pred_corr)
            if i < 5 or pred_corr > 0.5:  # Print first 5 and any good ones
                print(f"    [{l}] {c:>20}: loo_corr={pred_corr:+.3f}")

        print(f"\n  LOO prediction: mean={np.mean(loo_corrs):.3f}, "
              f"std={np.std(loo_corrs):.3f}, best={np.max(loo_corrs):.3f}")

    # =========================================================================
    print(f"\n{'='*90}")
    print("STEP 5: COMPARE WITH CURRENT IR USED IN OPTIMIZATION")
    print("="*90)
    current_ir_24k = np.load("data/empirical_ota/channel_impulse_response.npy")
    # Resample to 16kHz
    current_ir = signal.resample(current_ir_24k, int(len(current_ir_24k) * TARGET_SR / 24000))
    # Trim/pad to IR_LEN
    if len(current_ir) < IR_LEN:
        current_ir = np.pad(current_ir, (0, IR_LEN - len(current_ir)))
    else:
        current_ir = current_ir[:IR_LEN]

    print(f"  Current IR: peak={current_ir[np.argmax(np.abs(current_ir))]:+.6f} "
          f"@ sample {np.argmax(np.abs(current_ir))}")
    if len(marshall_irs) > 0:
        avg_ir = np.mean([ir for _, _, ir, _, _, _, _, _ in marshall_irs], axis=0)
        corr_with_avg = np.corrcoef(current_ir, avg_ir)[0, 1]
        print(f"  Correlation (current IR vs average of all extracted): {corr_with_avg:.4f}")

        # Individual correlations
        for l, c, ir, _, _, _, _, _ in marshall_irs[:5]:
            c_ir = np.corrcoef(current_ir, ir)[0, 1]
            print(f"    vs [{l}] {c}: {c_ir:.4f}")

    # =========================================================================
    print(f"\n{'='*90}")
    print("STEP 6: FREQUENCY-DOMAIN IR ANALYSIS")
    print("="*90)
    if len(marshall_irs) >= 2:
        # Check if magnitude response is more stable than phase
        freqs = np.fft.rfftfreq(IR_LEN, d=1.0/TARGET_SR)
        mag_responses = []
        phase_responses = []
        for _, _, ir, _, _, _, _, _ in marshall_irs:
            H = np.fft.rfft(ir, n=IR_LEN)
            mag_responses.append(np.abs(H))
            phase_responses.append(np.angle(H))

        mag_arr = np.array(mag_responses)
        phase_arr = np.array(phase_responses)

        # Per-frequency coefficient of variation for magnitude
        mag_mean = np.mean(mag_arr, axis=0)
        mag_std = np.std(mag_arr, axis=0)
        mag_cv = mag_std / (mag_mean + 1e-10)

        bands = [(0, 300), (300, 1000), (1000, 4000), (4000, 8000)]
        print(f"\n  Magnitude response stability (lower CV = more stable):")
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                continue
            cv = np.mean(mag_cv[mask])
            print(f"    {lo}-{hi}Hz: CV={cv:.3f}")

        # Phase stability (circular std)
        print(f"\n  Phase response stability (lower circ_std = more stable):")
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                continue
            # Circular std of phase
            phases = phase_arr[:, mask]
            mean_vec = np.abs(np.mean(np.exp(1j * phases), axis=0))
            circ_std = np.sqrt(-2 * np.log(np.mean(mean_vec) + 1e-10))
            print(f"    {lo}-{hi}Hz: circ_std={circ_std:.3f} rad")


if __name__ == "__main__":
    main()
