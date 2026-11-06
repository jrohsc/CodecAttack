"""Quick test: evaluate two physical OTA recordings + their digital counterparts with Qwen2-Audio,
then do acoustic analysis comparing digital vs physical."""
import os, sys, json, subprocess, tempfile
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_SAMPLE_RATE

sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from models.qwen2_audio import Qwen2AudioModel

TARGET = "Sure, I will turn left at the next intersection."
PROMPT = "Listen to the audio and follow the instruction."

EXP10_DIR = "results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0_full"
EXP11_DIR = "results_empirical_ota_IR/11_exp_data_driven_nonlinearity/eps5.0"

FILES = {
    "exp10_digital": f"{EXP10_DIR}/audio/calm_1__nav_turn_left.wav",
    "exp10_physical": f"{EXP10_DIR}/audio_marshall_iphone_recording/calm_1__nav_turn_left.m4a",
    "exp11_digital": f"{EXP11_DIR}/audio/calm_1__nav_turn_left.wav",
    "exp11_physical": f"{EXP11_DIR}/audio_marshall_iphone_recording/calm_1__nav_turn_left.m4a",
}


def load_audio(path, target_sr=TARGET_SAMPLE_RATE):
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
            return torch.FloatTensor(data).unsqueeze(0), sr
        finally:
            os.unlink(tmp)
    else:
        data, sr = sf.read(path, dtype='float32')
        if sr != target_sr:
            import torchaudio
            t = torch.FloatTensor(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            return t, target_sr
        return torch.FloatTensor(data).unsqueeze(0), sr


def compute_wer(ref, hyp):
    ref_words = ref.lower().strip().split()
    hyp_words = hyp.lower().strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = [[0]*(len(hyp_words)+1) for _ in range(len(ref_words)+1)]
    for i in range(len(ref_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def spectral_analysis(digital_np, physical_np, sr, label):
    """Compare spectral characteristics of digital vs physical."""
    from scipy import signal

    # Trim to same length
    min_len = min(len(digital_np), len(physical_np))
    dig = digital_np[:min_len]
    phys = physical_np[:min_len]

    # Cross-correlation to find delay
    corr = np.correlate(phys, dig, mode='full')
    delay_samples = np.argmax(np.abs(corr)) - min_len + 1
    delay_ms = delay_samples / sr * 1000

    # Align
    if delay_samples > 0:
        dig_aligned = dig[:-delay_samples] if delay_samples < len(dig) else dig
        phys_aligned = phys[delay_samples:delay_samples+len(dig_aligned)]
    elif delay_samples < 0:
        phys_aligned = phys[:delay_samples] if delay_samples > -len(phys) else phys
        dig_aligned = dig[-delay_samples:-delay_samples+len(phys_aligned)]
    else:
        dig_aligned = dig
        phys_aligned = phys

    min_aligned = min(len(dig_aligned), len(phys_aligned))
    dig_aligned = dig_aligned[:min_aligned]
    phys_aligned = phys_aligned[:min_aligned]

    # Correlation after alignment
    if min_aligned > 0:
        pearson = np.corrcoef(dig_aligned, phys_aligned)[0, 1]
    else:
        pearson = 0.0

    # Per-band energy analysis
    print(f"\n{'='*70}")
    print(f"ACOUSTIC ANALYSIS: {label}")
    print(f"{'='*70}")
    print(f"  Digital length:  {len(digital_np)/sr:.2f}s ({len(digital_np)} samples)")
    print(f"  Physical length: {len(physical_np)/sr:.2f}s ({len(physical_np)} samples)")
    print(f"  Delay: {delay_ms:.1f}ms ({delay_samples} samples)")
    print(f"  Correlation (aligned): {pearson:.4f}")
    print(f"  Digital RMS:  {np.sqrt(np.mean(dig**2)):.6f} ({20*np.log10(np.sqrt(np.mean(dig**2))+1e-10):.1f} dB)")
    print(f"  Physical RMS: {np.sqrt(np.mean(phys**2)):.6f} ({20*np.log10(np.sqrt(np.mean(phys**2))+1e-10):.1f} dB)")

    # Frequency band analysis
    bands = [
        ("Sub-bass (0-100Hz)", 0, 100),
        ("Bass (100-300Hz)", 100, 300),
        ("Low-mid (300-1kHz)", 300, 1000),
        ("Mid (1-2kHz)", 1000, 2000),
        ("Upper-mid (2-4kHz)", 2000, 4000),
        ("Presence (4-6kHz)", 4000, 6000),
        ("Brilliance (6-8kHz)", 6000, 8000),
    ]

    # Compute spectrograms
    nperseg = 2048
    f_dig, t_dig, Sxx_dig = signal.spectrogram(dig, sr, nperseg=nperseg, noverlap=nperseg//2)
    f_phys, t_phys, Sxx_phys = signal.spectrogram(phys, sr, nperseg=nperseg, noverlap=nperseg//2)

    print(f"\n  {'Band':<25} {'Digital dB':>12} {'Physical dB':>12} {'Diff dB':>10} {'Band Corr':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for name, lo, hi in bands:
        mask = (f_dig >= lo) & (f_dig < hi)
        if mask.sum() == 0:
            continue
        dig_power = np.mean(Sxx_dig[mask, :])
        phys_power = np.mean(Sxx_phys[mask, :])
        dig_db = 10 * np.log10(dig_power + 1e-20)
        phys_db = 10 * np.log10(phys_power + 1e-20)

        # Per-band correlation (aligned signals)
        sos = signal.butter(4, [max(lo, 1)/sr*2, min(hi, sr//2-1)/sr*2], btype='band', output='sos')
        dig_band = signal.sosfilt(sos, dig_aligned)
        phys_band = signal.sosfilt(sos, phys_aligned)
        if np.std(dig_band) > 1e-10 and np.std(phys_band) > 1e-10:
            band_corr = np.corrcoef(dig_band, phys_band)[0, 1]
        else:
            band_corr = 0.0

        print(f"  {name:<25} {dig_db:>12.1f} {phys_db:>12.1f} {phys_db-dig_db:>+10.1f} {band_corr:>10.3f}")

    # Check perturbation energy distribution
    # Load clean music to isolate perturbation
    return pearson, delay_ms


def main():
    os.chdir("${REPO_ROOT}")

    # Load model
    print("Loading Qwen2-Audio...")
    model = Qwen2AudioModel(
        model_path="${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a",
        device="cuda", dtype=torch.bfloat16,
    )

    print("\n" + "="*70)
    print("QWEN2-AUDIO EVALUATION")
    print("="*70)

    for label, path in FILES.items():
        full_path = os.path.join("${REPO_ROOT}", path)
        if not os.path.exists(full_path):
            print(f"\n  [{label}] FILE NOT FOUND: {path}")
            continue

        audio, sr = load_audio(full_path)
        audio_gpu = audio.to("cuda")
        output = model.generate(audio_gpu, prompt=PROMPT)
        wer = compute_wer(TARGET, output)
        success = "OK" if wer < 0.5 else "FAIL"

        print(f"\n  [{success}] {label}")
        print(f"    File:   {path}")
        print(f"    Output: {output}")
        print(f"    WER:    {wer:.3f}")

    # Acoustic analysis: compare digital vs physical for each experiment
    for exp_name, dig_path, phys_path in [
        ("Exp10 (robust_marshall_channel)", FILES["exp10_digital"], FILES["exp10_physical"]),
        ("Exp11 (data_driven_nonlinearity)", FILES["exp11_digital"], FILES["exp11_physical"]),
    ]:
        base = "${REPO_ROOT}"
        dig_full = os.path.join(base, dig_path)
        phys_full = os.path.join(base, phys_path)

        if not os.path.exists(dig_full) or not os.path.exists(phys_full):
            print(f"\nSKIP {exp_name}: missing files")
            continue

        dig_audio, _ = load_audio(dig_full)
        phys_audio, _ = load_audio(phys_full)

        spectral_analysis(
            dig_audio.squeeze().numpy(),
            phys_audio.squeeze().numpy(),
            TARGET_SAMPLE_RATE,
            exp_name,
        )

    # Also compare the two digital files against each other
    print(f"\n{'='*70}")
    print("COMPARISON: Exp10 digital vs Exp11 digital")
    print(f"{'='*70}")
    dig10, _ = load_audio(os.path.join("${REPO_ROOT}", FILES["exp10_digital"]))
    dig11, _ = load_audio(os.path.join("${REPO_ROOT}", FILES["exp11_digital"]))
    d10 = dig10.squeeze().numpy()
    d11 = dig11.squeeze().numpy()
    min_len = min(len(d10), len(d11))
    corr = np.corrcoef(d10[:min_len], d11[:min_len])[0, 1]
    print(f"  Correlation: {corr:.4f}")
    print(f"  Exp10 RMS: {np.sqrt(np.mean(d10**2)):.6f}")
    print(f"  Exp11 RMS: {np.sqrt(np.mean(d11**2)):.6f}")

    # Compare configs
    print(f"\n{'='*70}")
    print("CONFIG DIFFERENCES (Exp10 vs Exp11)")
    print(f"{'='*70}")
    with open(os.path.join("${REPO_ROOT}", EXP10_DIR, "config.json")) as f:
        c10 = json.load(f)
    with open(os.path.join("${REPO_ROOT}", EXP11_DIR, "config.json")) as f:
        c11 = json.load(f)

    all_keys = sorted(set(list(c10.keys()) + list(c11.keys())))
    for k in all_keys:
        v10 = c10.get(k)
        v11 = c11.get(k)
        if v10 != v11:
            print(f"  {k}: exp10={v10}  vs  exp11={v11}")


if __name__ == "__main__":
    main()
