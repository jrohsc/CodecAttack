"""
Deep acoustic/spectral analysis: WHY did bp400-700 (rep3, 4000 steps, Turn left) detect speech
while all other recordings didn't?

Compare digital vs physical for each recording, looking at:
1. Spectral envelope similarity (mel-band correlation)
2. Temporal modulation (speech has 3-5Hz amplitude modulation)
3. Formant-like structure preservation
4. Perturbation energy distribution
5. Channel distortion patterns
"""
import os, sys, subprocess, tempfile, json
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import correlate, stft
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, "${REPO_ROOT}")
sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from config import TARGET_SAMPLE_RATE

BASE = "${REPO_ROOT}/results_empirical_analysis_noise_cancellation/27_exp_narrow_bandpass_marshall"
OUT_DIR = os.path.join(BASE, "speech_detection_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SR = TARGET_SAMPLE_RATE

# Recordings to compare (all have digital + physical)
RECORDINGS = {
    "bp300-600 (0/10 speech)": {
        "digital": "bp300-600/attack_calm_1_20260408_220657_adversarial.wav",
        "physical": "bp300-600/audio_marshall_iphone_record_bp300-600.m4a",
        "original": "bp300-600/attack_calm_1_20260408_220657_original.wav",
        "speech_detected": False,
        "best_output": "fast piece of music on violin",
    },
    "bp400-700 go_down (0/10)": {
        "digital": "bp400_700_go_down/attack_calm_1_20260409_042509_adversarial.wav",
        "physical": "bp400_700_go_down/marshall_recorded.m4a",
        "original": "bp400_700_go_down/attack_calm_1_20260409_042509_original.wav",
        "speech_detected": False,
        "best_output": "car engine idling/revving",
    },
    "bp400-700 ecc5 (0/10)": {
        "digital": "bp400_700_steps5000_ecc_rep5/attack_calm_1_20260409_041852_adversarial.wav",
        "physical": "bp400_700_steps5000_ecc_rep5/marshall-recorded.m4a",
        "original": "bp400_700_steps5000_ecc_rep5/attack_calm_1_20260409_041852_original.wav",
        "speech_detected": False,
        "best_output": "dog barking / car engine",
    },
    "bp500-800 (2/10 speech)": {
        "digital": "bp500_800/attack_calm_1_20260409_025356_adversarial_no-ecc.wav",
        "physical": "bp500_800/audio_marshall_iphone_record_bp500-800_no-ecc.m4a",
        "original": "bp500_800/attack_calm_1_20260409_025356_original_no-ecc.wav",
        "speech_detected": True,
        "best_output": "two women talking by the door",
    },
}


def load_audio(path):
    full_path = os.path.join(BASE, path)
    if path.endswith('.m4a'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(['ffmpeg', '-y', '-i', full_path, '-ac', '1',
                            '-ar', str(SR), '-acodec', 'pcm_f32le', tmp],
                           capture_output=True, check=True)
            data, sr = sf.read(tmp, dtype='float32')
            return data
        finally:
            os.unlink(tmp)
    else:
        data, sr = sf.read(full_path, dtype='float32')
        if sr != SR:
            import torchaudio
            t = torch.FloatTensor(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SR)
            return t.squeeze().numpy()
        return data


def align_audios(ref, target, sr):
    chunk = min(len(ref), len(target), sr * 5)
    corr = correlate(target[:chunk], ref[:chunk], mode='full')
    lag = np.argmax(np.abs(corr)) - (chunk - 1)
    if lag > 0:
        target_a = target[lag:]
    else:
        target_a = np.pad(target, (-lag, 0))[:len(target)]
    min_len = min(len(ref), len(target_a))
    return ref[:min_len], target_a[:min_len], lag


def compute_mel_spectrogram(audio, sr, n_mels=80, n_fft=2048, hop=512):
    """Compute mel-like spectrogram using scipy."""
    f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx)

    # Create mel filterbank
    mel_low = 0
    mel_high = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bins[i], bins[i + 1]):
            if j < fbank.shape[1]:
                fbank[i, j] = (j - bins[i]) / max(1, bins[i + 1] - bins[i])
        for j in range(bins[i + 1], bins[i + 2]):
            if j < fbank.shape[1]:
                fbank[i, j] = (bins[i + 2] - j) / max(1, bins[i + 2] - bins[i + 1])

    mel_spec = np.dot(fbank, mag)
    mel_spec = np.log10(mel_spec + 1e-10)
    return mel_spec, t


def temporal_modulation_spectrum(audio, sr, frame_len=0.025, hop_len=0.010):
    """Compute amplitude envelope and its modulation spectrum.
    Speech has characteristic 3-5Hz modulation."""
    frame_samples = int(frame_len * sr)
    hop_samples = int(hop_len * sr)

    # Frame energy
    n_frames = (len(audio) - frame_samples) // hop_samples + 1
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        frame = audio[start:start + frame_samples]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    # Modulation spectrum
    mod_spec = np.abs(np.fft.rfft(energy - np.mean(energy)))
    mod_freqs = np.fft.rfftfreq(len(energy), d=hop_len)

    return energy, mod_spec, mod_freqs


def formant_band_temporal_correlation(dig, phys, sr):
    """Compute temporal correlation of energy in formant bands.
    High correlation = formant dynamics preserved through channel."""
    bands = {
        "F1_low (300-500Hz)": (300, 500),
        "F1_high (500-700Hz)": (500, 700),
        "F2_low (700-1000Hz)": (700, 1000),
        "F2_high (1000-1500Hz)": (1000, 1500),
        "F3 (2000-3000Hz)": (2000, 3000),
    }

    hop = 512
    n_fft = 2048
    _, _, Sxx_dig = stft(dig, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Sxx_phys = stft(phys, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)

    min_t = min(Sxx_dig.shape[1], Sxx_phys.shape[1])
    Sxx_dig = np.abs(Sxx_dig[:, :min_t])
    Sxx_phys = np.abs(Sxx_phys[:, :min_t])

    results = {}
    for bname, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        dig_band_energy = Sxx_dig[mask, :].sum(axis=0)
        phys_band_energy = Sxx_phys[mask, :].sum(axis=0)

        if np.std(dig_band_energy) > 0 and np.std(phys_band_energy) > 0:
            corr = np.corrcoef(dig_band_energy, phys_band_energy)[0, 1]
        else:
            corr = 0.0

        # Energy ratio
        ratio = np.mean(phys_band_energy) / max(np.mean(dig_band_energy), 1e-10)

        results[bname] = {"correlation": round(float(corr), 4), "energy_ratio": round(float(ratio), 4)}

    return results


def spectral_flatness(audio, sr, n_fft=2048, hop=512):
    """Spectral flatness — speech is less flat (more peaked) than noise/music."""
    _, _, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx) + 1e-10

    # Per-frame spectral flatness
    geo_mean = np.exp(np.mean(np.log(mag), axis=0))
    arith_mean = np.mean(mag, axis=0)
    flatness = geo_mean / arith_mean

    return np.mean(flatness), np.std(flatness)


def main():
    analysis = {}

    for name, rec in RECORDINGS.items():
        print(f"\n{'='*80}")
        print(f"Analyzing: {name}")
        print(f"  Speech detected: {rec['speech_detected']}")
        print(f"  Best output: {rec['best_output']}")

        digital = load_audio(rec["digital"])
        physical = load_audio(rec["physical"])
        original = load_audio(rec["original"])

        # Align
        dig_a, phys_a, lag = align_audios(digital, physical, SR)
        orig_a = original[:len(dig_a)]
        if len(orig_a) < len(dig_a):
            orig_a = np.pad(orig_a, (0, len(dig_a) - len(orig_a)))

        perturbation = dig_a - orig_a[:len(dig_a)]

        print(f"  Alignment lag: {lag} samples ({lag/SR*1000:.1f}ms)")
        print(f"  Digital RMS: {np.sqrt(np.mean(dig_a**2)):.6f}")
        print(f"  Physical RMS: {np.sqrt(np.mean(phys_a**2)):.6f}")
        print(f"  Perturbation RMS: {np.sqrt(np.mean(perturbation**2)):.6f}")
        print(f"  Original RMS: {np.sqrt(np.mean(orig_a**2)):.6f}")

        # 1. Overall correlation
        overall_corr = np.corrcoef(dig_a, phys_a)[0, 1]
        print(f"\n  1. Overall correlation: {overall_corr:.4f}")

        # 2. Perturbation-to-carrier ratio
        pert_rms = np.sqrt(np.mean(perturbation**2))
        orig_rms = np.sqrt(np.mean(orig_a**2))
        pcr = pert_rms / max(orig_rms, 1e-10)
        pcr_db = 20 * np.log10(pcr)
        print(f"  2. Perturbation/Carrier ratio: {pcr:.4f} ({pcr_db:.1f}dB)")

        # 3. Temporal modulation spectrum (speech = 3-5Hz peak)
        dig_env, dig_mod, mod_freqs = temporal_modulation_spectrum(dig_a, SR)
        phys_env, phys_mod, _ = temporal_modulation_spectrum(phys_a, SR)
        pert_env, pert_mod, _ = temporal_modulation_spectrum(perturbation, SR)

        # Energy in speech modulation band (3-5Hz)
        speech_band = (mod_freqs >= 3) & (mod_freqs <= 5)
        total_band = (mod_freqs >= 0.5) & (mod_freqs <= 20)

        dig_speech_mod = np.mean(dig_mod[speech_band]) / max(np.mean(dig_mod[total_band]), 1e-10)
        phys_speech_mod = np.mean(phys_mod[speech_band]) / max(np.mean(phys_mod[total_band]), 1e-10)
        pert_speech_mod = np.mean(pert_mod[speech_band]) / max(np.mean(pert_mod[total_band]), 1e-10)

        print(f"  3. Speech modulation (3-5Hz relative energy):")
        print(f"     Digital:      {dig_speech_mod:.4f}")
        print(f"     Physical:     {phys_speech_mod:.4f}")
        print(f"     Perturbation: {pert_speech_mod:.4f}")

        # Envelope correlation
        min_env = min(len(dig_env), len(phys_env))
        env_corr = np.corrcoef(dig_env[:min_env], phys_env[:min_env])[0, 1]
        print(f"     Envelope correlation (dig vs phys): {env_corr:.4f}")

        # 4. Formant band temporal correlation
        print(f"  4. Formant band temporal correlation:")
        formant_corrs = formant_band_temporal_correlation(dig_a, phys_a, SR)
        for bname, vals in formant_corrs.items():
            print(f"     {bname:25s}: corr={vals['correlation']:.4f}, ratio={vals['energy_ratio']:.4f}")

        # 5. Spectral flatness (speech = less flat)
        dig_flat_mean, dig_flat_std = spectral_flatness(dig_a, SR)
        phys_flat_mean, phys_flat_std = spectral_flatness(phys_a, SR)
        pert_flat_mean, pert_flat_std = spectral_flatness(perturbation, SR)
        print(f"  5. Spectral flatness (lower = more speech-like):")
        print(f"     Digital:      {dig_flat_mean:.4f} (±{dig_flat_std:.4f})")
        print(f"     Physical:     {phys_flat_mean:.4f} (±{phys_flat_std:.4f})")
        print(f"     Perturbation: {pert_flat_mean:.4f} (±{pert_flat_std:.4f})")

        # 6. Mel spectrogram correlation
        dig_mel, _ = compute_mel_spectrogram(dig_a, SR)
        phys_mel, _ = compute_mel_spectrogram(phys_a, SR)
        min_t = min(dig_mel.shape[1], phys_mel.shape[1])
        mel_corr = np.corrcoef(dig_mel[:, :min_t].flatten(), phys_mel[:, :min_t].flatten())[0, 1]

        # Per-mel-band correlation (focus on 400-700Hz region)
        # mel bins roughly: bin 10-20 covers 200-800Hz
        low_mel_corr = np.corrcoef(dig_mel[10:20, :min_t].flatten(), phys_mel[10:20, :min_t].flatten())[0, 1]
        mid_mel_corr = np.corrcoef(dig_mel[20:35, :min_t].flatten(), phys_mel[20:35, :min_t].flatten())[0, 1]
        high_mel_corr = np.corrcoef(dig_mel[35:60, :min_t].flatten(), phys_mel[35:60, :min_t].flatten())[0, 1]

        print(f"  6. Mel spectrogram correlation:")
        print(f"     Overall:     {mel_corr:.4f}")
        print(f"     Low mel (200-800Hz):  {low_mel_corr:.4f}")
        print(f"     Mid mel (800-3kHz):   {mid_mel_corr:.4f}")
        print(f"     High mel (3k-8kHz):   {high_mel_corr:.4f}")

        # 7. Perturbation spectral centroid (where is perturbation energy centered?)
        f_pert, _, Sxx_pert = signal.spectrogram(perturbation, fs=SR, nperseg=2048, noverlap=1536)
        avg_pert_spec = np.mean(Sxx_pert, axis=1)
        spectral_centroid = np.sum(f_pert * avg_pert_spec) / max(np.sum(avg_pert_spec), 1e-10)

        # Bandwidth (spread around centroid)
        spectral_bw = np.sqrt(np.sum(((f_pert - spectral_centroid) ** 2) * avg_pert_spec) / max(np.sum(avg_pert_spec), 1e-10))

        print(f"  7. Perturbation spectral centroid: {spectral_centroid:.0f}Hz (bandwidth: {spectral_bw:.0f}Hz)")

        # Store
        analysis[name] = {
            "speech_detected": rec["speech_detected"],
            "best_output": rec["best_output"],
            "alignment_lag_ms": round(lag / SR * 1000, 1),
            "digital_rms": round(float(np.sqrt(np.mean(dig_a**2))), 6),
            "physical_rms": round(float(np.sqrt(np.mean(phys_a**2))), 6),
            "perturbation_rms": round(float(pert_rms), 6),
            "original_rms": round(float(orig_rms), 6),
            "perturbation_carrier_ratio": round(float(pcr), 4),
            "perturbation_carrier_ratio_db": round(float(pcr_db), 1),
            "overall_correlation": round(float(overall_corr), 4),
            "envelope_correlation": round(float(env_corr), 4),
            "speech_modulation_3_5Hz": {
                "digital": round(float(dig_speech_mod), 4),
                "physical": round(float(phys_speech_mod), 4),
                "perturbation": round(float(pert_speech_mod), 4),
            },
            "formant_band_temporal_correlation": formant_corrs,
            "spectral_flatness": {
                "digital": round(float(dig_flat_mean), 4),
                "physical": round(float(phys_flat_mean), 4),
                "perturbation": round(float(pert_flat_mean), 4),
            },
            "mel_correlation": {
                "overall": round(float(mel_corr), 4),
                "low_200_800Hz": round(float(low_mel_corr), 4),
                "mid_800_3kHz": round(float(mid_mel_corr), 4),
                "high_3k_8kHz": round(float(high_mel_corr), 4),
            },
            "perturbation_spectral_centroid_hz": round(float(spectral_centroid), 0),
            "perturbation_spectral_bandwidth_hz": round(float(spectral_bw), 0),
        }

    # ===== Comparison plots =====
    names = list(analysis.keys())
    n = len(names)

    # Plot 1: Mel spectrograms (digital vs physical for each)
    fig, axes = plt.subplots(2, n, figsize=(6*n, 8))
    for idx, (name, rec) in enumerate(RECORDINGS.items()):
        digital = load_audio(rec["digital"])
        physical = load_audio(rec["physical"])
        dig_a, phys_a, _ = align_audios(digital, physical, SR)

        dig_mel, t_d = compute_mel_spectrogram(dig_a, SR)
        phys_mel, t_p = compute_mel_spectrogram(phys_a, SR)

        ax = axes[0, idx]
        ax.imshow(dig_mel, aspect='auto', origin='lower', cmap='viridis', vmin=-6, vmax=-1)
        ax.set_title(f'{name}\nDigital', fontsize=9)
        ax.set_ylabel('Mel bin')

        ax = axes[1, idx]
        ax.imshow(phys_mel, aspect='auto', origin='lower', cmap='viridis', vmin=-6, vmax=-1)
        ax.set_title(f'Physical', fontsize=9)
        ax.set_ylabel('Mel bin')
        ax.set_xlabel('Time frame')

        # Mark speech detection
        color = 'green' if rec['speech_detected'] else 'red'
        for row in range(2):
            axes[row, idx].spines['bottom'].set_color(color)
            axes[row, idx].spines['top'].set_color(color)
            axes[row, idx].spines['left'].set_color(color)
            axes[row, idx].spines['right'].set_color(color)
            for spine in axes[row, idx].spines.values():
                spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mel_comparison.png'), dpi=150)
    print(f"\nSaved: {OUT_DIR}/mel_comparison.png")

    # Plot 2: Temporal modulation spectrum comparison
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    for idx, (name, rec) in enumerate(RECORDINGS.items()):
        digital = load_audio(rec["digital"])
        physical = load_audio(rec["physical"])
        dig_a, phys_a, _ = align_audios(digital, physical, SR)
        perturbation = dig_a - load_audio(rec["original"])[:len(dig_a)]

        _, dig_mod, mod_freqs = temporal_modulation_spectrum(dig_a, SR)
        _, phys_mod, _ = temporal_modulation_spectrum(phys_a, SR)
        _, pert_mod, _ = temporal_modulation_spectrum(perturbation, SR)

        ax = axes[idx]
        mask = mod_freqs <= 20
        ax.plot(mod_freqs[mask], dig_mod[mask] / max(dig_mod[mask].max(), 1e-10), label='Digital', alpha=0.8)
        ax.plot(mod_freqs[mask], phys_mod[mask] / max(phys_mod[mask].max(), 1e-10), label='Physical', alpha=0.8)
        ax.axvspan(3, 5, alpha=0.2, color='green', label='Speech band')
        ax.set_title(f'{name}', fontsize=9)
        ax.set_xlabel('Modulation freq (Hz)')
        ax.set_ylabel('Normalized energy')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'modulation_spectrum.png'), dpi=150)
    print(f"Saved: {OUT_DIR}/modulation_spectrum.png")

    # Plot 3: Bar chart comparison of key metrics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    labels = [n.split('(')[0].strip() for n in names]
    colors = ['green' if analysis[n]['speech_detected'] else 'red' for n in names]
    x = np.arange(len(names))

    # Perturbation/carrier ratio
    ax = axes[0, 0]
    vals = [analysis[n]['perturbation_carrier_ratio'] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Perturbation/Carrier Ratio')
    ax.set_ylabel('Ratio')

    # Envelope correlation
    ax = axes[0, 1]
    vals = [analysis[n]['envelope_correlation'] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Amplitude Envelope Correlation\n(Digital vs Physical)')

    # Speech modulation (physical)
    ax = axes[0, 2]
    vals = [analysis[n]['speech_modulation_3_5Hz']['physical'] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Speech Modulation (3-5Hz)\nin Physical Recording')

    # Mel correlation (low bands)
    ax = axes[1, 0]
    vals = [analysis[n]['mel_correlation']['low_200_800Hz'] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Mel Correlation (200-800Hz)\nDigital vs Physical')

    # Spectral flatness (physical)
    ax = axes[1, 1]
    vals = [analysis[n]['spectral_flatness']['physical'] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Spectral Flatness (Physical)\nLower = more speech-like')

    # Perturbation spectral centroid
    ax = axes[1, 2]
    vals = [analysis[n]['perturbation_spectral_centroid_hz'] for n in names]
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Perturbation Spectral Centroid (Hz)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'metrics_comparison.png'), dpi=150)
    print(f"Saved: {OUT_DIR}/metrics_comparison.png")

    # Save JSON
    out_path = os.path.join(OUT_DIR, 'analysis.json')
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n{'='*120}")
    print(f"SUMMARY: Why does speech detection succeed or fail?")
    print(f"{'='*120}")
    print(f"{'Recording':<30s} | {'Speech':>6s} | {'P/C ratio':>9s} | {'Env corr':>8s} | {'3-5Hz mod':>9s} | {'Mel corr':>8s} | {'Flatness':>8s} | {'Centroid':>8s}")
    print(f"{'-'*120}")
    for name in names:
        a = analysis[name]
        s = "YES" if a["speech_detected"] else "NO"
        print(f"{name:<30s} | {s:>6s} | {a['perturbation_carrier_ratio']:>9.4f} | {a['envelope_correlation']:>8.4f} | {a['speech_modulation_3_5Hz']['physical']:>9.4f} | {a['mel_correlation']['low_200_800Hz']:>8.4f} | {a['spectral_flatness']['physical']:>8.4f} | {a['perturbation_spectral_centroid_hz']:>7.0f}Hz")


if __name__ == "__main__":
    main()
