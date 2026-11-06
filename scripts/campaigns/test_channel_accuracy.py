"""Compare simulated channel output vs actual physical recording.
This tells us if the channel model is accurate."""
import os, sys, subprocess, tempfile
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ENCODEC_SR = 24000
TARGET_SR = 16000

def load_m4a(path, target_sr=TARGET_SR):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
                       '-acodec', 'pcm_f32le', tmp], capture_output=True, check=True)
        data, sr = sf.read(tmp, dtype='float32')
        return data
    finally:
        os.unlink(tmp)


def load_wav(path, target_sr=TARGET_SR):
    data, sr = sf.read(path, dtype='float32')
    if sr != target_sr:
        import torchaudio
        t = torch.FloatTensor(data).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        return t.squeeze().numpy()
    return data


def apply_ir_convolution(audio_np, ir_np):
    """Apply IR convolution exactly as the attack code does (at 24kHz)."""
    # The attack works at 24kHz, so we need to upsample, convolve, then downsample
    # But for comparison, let's just do it at 16kHz after resampling the IR
    result = signal.fftconvolve(audio_np, ir_np, mode='full')[:len(audio_np)]
    # RMS normalize to match input
    in_rms = np.sqrt(np.mean(audio_np**2) + 1e-10)
    out_rms = np.sqrt(np.mean(result**2) + 1e-10)
    result = result * (in_rms / out_rms)
    return result


def align_and_compare(ref, test, sr, label):
    """Align test to ref using cross-correlation, then compare."""
    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]

    # Find delay
    corr = np.correlate(test, ref, mode='full')
    delay = np.argmax(np.abs(corr)) - min_len + 1
    delay_ms = delay / sr * 1000

    # Align
    if delay > 0:
        r = ref[:-delay] if delay < len(ref) else ref
        t = test[delay:delay+len(r)]
    elif delay < 0:
        t = test[:delay] if delay > -len(test) else test
        r = ref[-delay:-delay+len(t)]
    else:
        r, t = ref, test

    n = min(len(r), len(t))
    r, t = r[:n], t[:n]

    pearson = np.corrcoef(r, t)[0, 1] if n > 0 else 0
    mse = np.mean((r - t)**2)
    snr = 10 * np.log10(np.mean(r**2) / (mse + 1e-20))

    print(f"\n  {label}:")
    print(f"    Delay: {delay_ms:.1f}ms, Corr: {pearson:.4f}, SNR: {snr:.1f}dB")
    return pearson, delay


def main():
    os.chdir("${REPO_ROOT}")

    # Load IR at original 24kHz
    ir_24k = np.load("data/empirical_ota/channel_impulse_response.npy")
    # Resample IR to 16kHz for comparison
    ir_16k = signal.resample(ir_24k, int(len(ir_24k) * TARGET_SR / ENCODEC_SR))

    print(f"IR at 24kHz: {len(ir_24k)} samples, peak={ir_24k[np.argmax(np.abs(ir_24k))]:.6f}")
    print(f"IR at 16kHz: {len(ir_16k)} samples, peak={ir_16k[np.argmax(np.abs(ir_16k))]:.6f}")

    for exp_name, dig_path, phys_path in [
        ("Exp10 (speaker_nonlinearity)",
         "results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0_full/audio/calm_1__nav_turn_left.wav",
         "results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0_full/audio_marshall_iphone_recording/calm_1__nav_turn_left.m4a"),
        ("Exp11 (empirical_nonlinearity)",
         "results_empirical_ota_IR/11_exp_data_driven_nonlinearity/eps5.0/audio/calm_1__nav_turn_left.wav",
         "results_empirical_ota_IR/11_exp_data_driven_nonlinearity/eps5.0/audio_marshall_iphone_recording/calm_1__nav_turn_left.m4a"),
    ]:
        print(f"\n{'='*70}")
        print(f"{exp_name}")
        print(f"{'='*70}")

        digital = load_wav(dig_path)
        physical = load_m4a(phys_path)

        # Apply simulated channel to digital audio
        simulated = apply_ir_convolution(digital, ir_16k)

        print(f"  Digital:   {len(digital)} samples, RMS={np.sqrt(np.mean(digital**2)):.4f}")
        print(f"  Physical:  {len(physical)} samples, RMS={np.sqrt(np.mean(physical**2)):.4f}")
        print(f"  Simulated: {len(simulated)} samples, RMS={np.sqrt(np.mean(simulated**2)):.4f}")

        # Compare all pairs
        align_and_compare(digital, physical, TARGET_SR, "Digital vs Physical")
        align_and_compare(simulated, physical, TARGET_SR, "Simulated vs Physical")
        align_and_compare(digital, simulated, TARGET_SR, "Digital vs Simulated")

        # Per-band comparison: simulated vs physical
        bands = [
            ("Sub-bass (0-100Hz)", 0, 100),
            ("Bass (100-300Hz)", 100, 300),
            ("Low-mid (300-1kHz)", 300, 1000),
            ("Mid (1-2kHz)", 1000, 2000),
            ("Upper-mid (2-4kHz)", 2000, 4000),
        ]

        # Align simulated and physical first
        min_len = min(len(simulated), len(physical))
        corr_full = np.correlate(physical[:min_len], simulated[:min_len], mode='full')
        delay = np.argmax(np.abs(corr_full)) - min_len + 1
        if delay > 0:
            sim_a = simulated[:-delay] if delay < len(simulated) else simulated
            phys_a = physical[delay:delay+len(sim_a)]
        elif delay < 0:
            phys_a = physical[:delay] if delay > -len(physical) else physical
            sim_a = simulated[-delay:-delay+len(phys_a)]
        else:
            sim_a, phys_a = simulated, physical
        n = min(len(sim_a), len(phys_a))
        sim_a, phys_a = sim_a[:n], phys_a[:n]

        print(f"\n  Per-band: Simulated vs Physical (aligned, delay={delay/TARGET_SR*1000:.1f}ms)")
        print(f"  {'Band':<25} {'Sim dB':>10} {'Phys dB':>10} {'Diff':>8} {'Corr':>8}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for name, lo, hi in bands:
            try:
                sos = signal.butter(4, [max(lo, 1)/(TARGET_SR/2), min(hi, TARGET_SR//2-1)/(TARGET_SR/2)],
                                   btype='band', output='sos')
                sim_band = signal.sosfilt(sos, sim_a)
                phys_band = signal.sosfilt(sos, phys_a)
                sim_db = 20 * np.log10(np.sqrt(np.mean(sim_band**2)) + 1e-20)
                phys_db = 20 * np.log10(np.sqrt(np.mean(phys_band**2)) + 1e-20)
                if np.std(sim_band) > 1e-10 and np.std(phys_band) > 1e-10:
                    bc = np.corrcoef(sim_band, phys_band)[0, 1]
                else:
                    bc = 0
                print(f"  {name:<25} {sim_db:>10.1f} {phys_db:>10.1f} {phys_db-sim_db:>+8.1f} {bc:>8.3f}")
            except Exception as e:
                print(f"  {name:<25} ERROR: {e}")

    # Also: load clean music to see how much of the perturbation survives
    print(f"\n{'='*70}")
    print(f"PERTURBATION ANALYSIS")
    print(f"{'='*70}")

    # Load clean calm_1 music
    clean_path = None
    for p in ["data/music/calm_1.mp3", "data/music/calm_1.wav"]:
        if os.path.exists(p):
            clean_path = p
            break

    if clean_path:
        clean = load_wav(clean_path)
        for exp_name, dig_path in [
            ("Exp10", "results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0_full/audio/calm_1__nav_turn_left.wav"),
            ("Exp11", "results_empirical_ota_IR/11_exp_data_driven_nonlinearity/eps5.0/audio/calm_1__nav_turn_left.wav"),
        ]:
            digital = load_wav(dig_path)
            min_len = min(len(clean), len(digital))
            perturbation = digital[:min_len] - clean[:min_len]
            print(f"\n  {exp_name}:")
            print(f"    Clean RMS:        {np.sqrt(np.mean(clean[:min_len]**2)):.6f}")
            print(f"    Digital RMS:      {np.sqrt(np.mean(digital[:min_len]**2)):.6f}")
            print(f"    Perturbation RMS: {np.sqrt(np.mean(perturbation**2)):.6f}")
            print(f"    SNR (carrier/perturbation): {10*np.log10(np.mean(clean[:min_len]**2)/(np.mean(perturbation**2)+1e-20)):.1f}dB")

            # Perturbation spectrum
            f, Pxx = signal.welch(perturbation, TARGET_SR, nperseg=2048)
            peak_freq = f[np.argmax(Pxx)]
            print(f"    Perturbation peak frequency: {peak_freq:.0f}Hz")

            # Energy by band
            for name, lo, hi in [("0-300Hz", 0, 300), ("300-1kHz", 300, 1000),
                                  ("1-4kHz", 1000, 4000), ("4-8kHz", 4000, 8000)]:
                mask = (f >= lo) & (f < hi)
                band_energy = np.sum(Pxx[mask]) / (np.sum(Pxx) + 1e-20) * 100
                print(f"    Perturbation energy {name}: {band_energy:.1f}%")


if __name__ == "__main__":
    main()
