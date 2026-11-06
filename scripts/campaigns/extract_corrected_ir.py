"""Extract the corrected average IR from all Marshall paired recordings and save at 24kHz."""
import os, sys, subprocess, tempfile
import numpy as np
import soundfile as sf
from scipy import signal as sig

TARGET_SR = 16000
ENCODEC_SR = 24000
IR_LEN_16K = 4800  # 300ms at 16kHz

BASE = "${REPO_ROOT}/results_empirical_ota_IR"

MARSHALL_PAIRS = [
    ("8_exp_marshall_combined/eps5.0/audio",
     "8_exp_marshall_combined/eps5.0/audio_marshall_iphone_recording"),
    ("7_exp_marshall_ir_timeshift/eps3.0/audio",
     "7_exp_marshall_ir_timeshift/eps3.0/audio_marshall_iphone_record"),
    ("6_exp_marshall_ir_fix/eps3.0/audio",
     "6_exp_marshall_ir_fix/eps3.0/audio_marshall_iphone_recording"),
    ("5_exp_ecc_redundancy_marshall/simplified_rep3/audio",
     "5_exp_ecc_redundancy_marshall/simplified_rep3/audio_marshall_iphone_recording"),
    ("4_exp_bandpass300_1000_marshall/eps3.0/audio",
     "4_exp_bandpass300_1000_marshall/eps3.0/audio_marshall_iphone_recording"),
    ("3_exp_physical_channel_fir/audio",
     "3_exp_physical_channel_fir/marshall_iphone"),
    ("1_exp_empirical_nonlinearity/audio",
     "1_exp_empirical_nonlinearity/audio_marshall_iphone_record"),
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


def extract_ir_wiener(digital, physical, ir_len=IR_LEN_16K, reg=1e-3):
    """Extract IR via Wiener deconvolution."""
    min_len = min(len(digital), len(physical))
    d = digital[:min_len].astype(np.float64)
    p = physical[:min_len].astype(np.float64)

    # Align
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


def main():
    os.chdir("${REPO_ROOT}")

    all_irs = []
    labels = []

    for dig_dir_rel, rec_dir_rel in MARSHALL_PAIRS:
        dig_dir = os.path.join(BASE, dig_dir_rel)
        rec_dir = os.path.join(BASE, rec_dir_rel)
        if not os.path.isdir(dig_dir) or not os.path.isdir(rec_dir):
            continue

        rec_files = {}
        for f in sorted(os.listdir(rec_dir)):
            if f.endswith('.m4a'):
                base = f.replace('.m4a', '').split('_volume')[0]
                rec_files[base] = f
        dig_files = {f.replace('.wav', ''): f for f in sorted(os.listdir(dig_dir)) if f.endswith('.wav')}

        for base_name in sorted(set(rec_files.keys()) & set(dig_files.keys())):
            dig_path = os.path.join(dig_dir, dig_files[base_name])
            rec_path = os.path.join(rec_dir, rec_files[base_name])

            digital = load_audio(dig_path)
            physical = load_audio(rec_path)

            ir, delay = extract_ir_wiener(digital, physical)
            peak_val = ir[np.argmax(np.abs(ir))]

            # Skip the 2 outliers with positive polarity (exp1 exit_building, turn_left)
            if peak_val > 0:
                print(f"  SKIP (positive polarity): {base_name} peak={peak_val:+.4f}")
                continue

            all_irs.append(ir)
            labels.append(base_name)
            print(f"  OK: {base_name} peak={peak_val:+.4f} delay={delay/TARGET_SR*1000:.0f}ms")

    print(f"\nUsing {len(all_irs)} IRs (excluded positive-polarity outliers)")

    # Compute average IR
    avg_ir_16k = np.mean(all_irs, axis=0)
    peak_idx = np.argmax(np.abs(avg_ir_16k))
    print(f"\nAverage IR (16kHz):")
    print(f"  Length: {len(avg_ir_16k)} samples = {len(avg_ir_16k)/TARGET_SR*1000:.0f}ms")
    print(f"  Peak: {avg_ir_16k[peak_idx]:+.6f} @ sample {peak_idx}")

    # Resample to 24kHz for the attack code
    ir_len_24k = int(IR_LEN_16K * ENCODEC_SR / TARGET_SR)  # 7200 samples
    avg_ir_24k = sig.resample(avg_ir_16k, ir_len_24k)
    peak_idx_24k = np.argmax(np.abs(avg_ir_24k))
    print(f"\nAverage IR (24kHz):")
    print(f"  Length: {len(avg_ir_24k)} samples = {len(avg_ir_24k)/ENCODEC_SR*1000:.0f}ms")
    print(f"  Peak: {avg_ir_24k[peak_idx_24k]:+.6f} @ sample {peak_idx_24k}")

    # Save
    out_dir = "data/empirical_ota"
    # Backup old IR
    old_ir_path = os.path.join(out_dir, "channel_impulse_response.npy")
    if os.path.exists(old_ir_path):
        backup_path = os.path.join(out_dir, "channel_impulse_response_OLD.npy")
        np.save(backup_path, np.load(old_ir_path))
        print(f"\nBacked up old IR to {backup_path}")

    np.save(old_ir_path, avg_ir_24k)
    print(f"Saved corrected IR to {old_ir_path}")

    # Also save 16kHz version and individual IRs for reference
    np.save(os.path.join(out_dir, "channel_impulse_response_16k.npy"), avg_ir_16k)
    np.save(os.path.join(out_dir, "all_individual_irs_16k.npy"), np.array(all_irs))
    print(f"Saved 16kHz IR and {len(all_irs)} individual IRs")

    # Verify: compare new vs old
    old_ir = np.load(os.path.join(out_dir, "channel_impulse_response_OLD.npy"))
    print(f"\nOld IR: {len(old_ir)} samples, peak={old_ir[np.argmax(np.abs(old_ir))]:+.6f} @ {np.argmax(np.abs(old_ir))}")
    print(f"New IR: {len(avg_ir_24k)} samples, peak={avg_ir_24k[peak_idx_24k]:+.6f} @ {peak_idx_24k}")

    # Correlation
    min_len = min(len(old_ir), len(avg_ir_24k))
    corr = np.corrcoef(old_ir[:min_len], avg_ir_24k[:min_len])[0, 1]
    print(f"Correlation old vs new: {corr:.4f}")


if __name__ == "__main__":
    main()
