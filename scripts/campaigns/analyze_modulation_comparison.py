"""Compare modulation spectrum of new bp500-800+modulation attack vs previous attacks."""
import os, sys, subprocess, tempfile, json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "${REPO_ROOT}")
from config import TARGET_SAMPLE_RATE

BASE = "${REPO_ROOT}/results_empirical_analysis_noise_cancellation/27_exp_narrow_bandpass_marshall"
OUT_DIR = os.path.join(BASE, "speech_detection_analysis")
SR = TARGET_SAMPLE_RATE

# Digital adversarial files to compare
ATTACKS = [
    {"label": "bp400-700 (SPEECH 6/10)", "speech": True,
     "adv": os.path.join(BASE, "find_bp_400_700_working/bp400-700_eps3.0_steps_4000_ecc_3reps.m4a"),
     "fmt": "m4a"},
    {"label": "bp500-800 (speech 2/10)", "speech": True,
     "adv": os.path.join(BASE, "bp500_800/attack_calm_1_20260409_025356_adversarial_no-ecc.wav"),
     "orig": os.path.join(BASE, "bp500_800/attack_calm_1_20260409_025356_original_no-ecc.wav"),
     "fmt": "wav"},
    {"label": "bp400-700 go_down (0/10)", "speech": False,
     "adv": os.path.join(BASE, "bp400_700_go_down/attack_calm_1_20260409_042509_adversarial.wav"),
     "orig": os.path.join(BASE, "bp400_700_go_down/attack_calm_1_20260409_042509_original.wav"),
     "fmt": "wav"},
    {"label": "bp400-700 ecc5 (0/10)", "speech": False,
     "adv": os.path.join(BASE, "bp400_700_steps5000_ecc_rep5 (need recording)/attack_calm_1_20260409_041852_adversarial.wav"),
     "orig": os.path.join(BASE, "bp400_700_steps5000_ecc_rep5 (need recording)/attack_calm_1_20260409_041852_original.wav"),
     "fmt": "wav"},
    {"label": "bp500-800+MOD (NEW)", "speech": None,
     "adv": os.path.join(BASE, "bp500_800_modulation/attack_calm_1_20260409_052040_adversarial.wav"),
     "orig": os.path.join(BASE, "bp500_800_modulation/attack_calm_1_20260409_052040_original.wav"),
     "fmt": "wav"},
]

import torch
import torchaudio

def load_audio(path, fmt="wav"):
    if fmt == "m4a":
        with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as fi:
            tmp_in = fi.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fo:
            tmp_out = fo.name
        try:
            os.unlink(tmp_in)
            os.symlink(path, tmp_in)
            subprocess.run(['ffmpeg', '-y', '-i', tmp_in, '-ac', '1', '-ar', str(SR),
                            '-acodec', 'pcm_f32le', tmp_out], capture_output=True, check=True)
            data, sr = sf.read(tmp_out, dtype='float32')
            return data
        finally:
            for f in [tmp_in, tmp_out]:
                if os.path.exists(f): os.unlink(f)
    else:
        data, sr = sf.read(path, dtype='float32')
        if sr != SR:
            t = torch.FloatTensor(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SR)
            return t.squeeze().numpy()
        return data


def compute_modulation(audio, sr):
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = (len(audio) - frame_len) // hop_len + 1
    env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        env[i] = np.sqrt(np.mean(audio[start:start+frame_len]**2))
    mod_spec = np.abs(np.fft.rfft(env - np.mean(env)))
    mod_freqs = np.fft.rfftfreq(len(env), d=0.010)
    return env, mod_spec, mod_freqs


def compute_band_energy(audio, sr):
    from scipy.signal import stft
    f, t, Zxx = stft(audio, fs=sr, nperseg=2048, noverlap=1536)
    power = np.abs(Zxx)**2
    bands = {'300-400': (300,400), '400-500': (400,500), '500-600': (500,600),
             '600-700': (600,700), '700-800': (700,800), '800-1000': (800,1000)}
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        result[name] = float(np.mean(power[mask, :]))
    return result


def main():
    results = []

    for atk in ATTACKS:
        if not os.path.exists(atk["adv"]):
            print(f"SKIP {atk['label']}: {atk['adv']} not found")
            continue

        print(f"\nAnalyzing: {atk['label']}")
        adv = load_audio(atk["adv"], atk["fmt"])

        # Compute perturbation if original available
        if "orig" in atk and atk["orig"] and os.path.exists(atk["orig"]):
            orig = load_audio(atk["orig"], "wav")
            min_len = min(len(adv), len(orig))
            perturbation = adv[:min_len] - orig[:min_len]
        else:
            perturbation = adv  # m4a recorded, use full signal

        # Modulation spectrum
        _, adv_mod, mod_freqs = compute_modulation(adv, SR)
        _, pert_mod, _ = compute_modulation(perturbation, SR)

        # Key modulation metrics
        speech_mask = (mod_freqs >= 3) & (mod_freqs <= 5)
        total_mask = (mod_freqs >= 0.5) & (mod_freqs <= 20)

        adv_35hz = float(np.mean(adv_mod[speech_mask]) / max(np.mean(adv_mod[total_mask]), 1e-10))
        pert_35hz = float(np.mean(pert_mod[speech_mask]) / max(np.mean(pert_mod[total_mask]), 1e-10))

        # Band energy
        adv_bands = compute_band_energy(adv, SR)
        pert_bands = compute_band_energy(perturbation, SR)

        # Spectral centroid
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(perturbation, fs=SR, nperseg=2048, noverlap=1536)
        avg_spec = np.mean(np.abs(Zxx)**2, axis=1)
        centroid = float(np.sum(f * avg_spec) / max(np.sum(avg_spec), 1e-10))

        rms = float(np.sqrt(np.mean(adv**2)))
        pert_rms = float(np.sqrt(np.mean(perturbation**2)))

        print(f"  3-5Hz modulation (adv): {adv_35hz:.4f}")
        print(f"  3-5Hz modulation (pert): {pert_35hz:.4f}")
        print(f"  Spectral centroid: {centroid:.0f}Hz")
        print(f"  RMS: {rms:.6f}, Pert RMS: {pert_rms:.6f}")
        print(f"  600-700Hz energy: {adv_bands.get('600-700', 0):.2e}")
        print(f"  700-800Hz energy: {adv_bands.get('700-800', 0):.2e}")

        results.append({
            "label": atk["label"],
            "speech_detected_physical": atk["speech"],
            "modulation_3_5Hz_adv": round(adv_35hz, 4),
            "modulation_3_5Hz_pert": round(pert_35hz, 4),
            "spectral_centroid_hz": round(centroid, 0),
            "rms": round(rms, 6),
            "perturbation_rms": round(pert_rms, 6),
            "band_energy_adv": adv_bands,
            "band_energy_pert": pert_bands,
        })

    # Print comparison table
    print(f"\n{'='*110}")
    print(f"{'Attack':<35s} | {'Phys OTA':>8s} | {'3-5Hz(adv)':>10s} | {'3-5Hz(pert)':>11s} | {'Centroid':>8s} | {'600-700Hz':>10s} | {'700-800Hz':>10s}")
    print(f"{'-'*110}")
    for r in results:
        sp = "YES" if r["speech_detected_physical"] == True else ("NEW" if r["speech_detected_physical"] is None else "NO")
        print(f"{r['label']:<35s} | {sp:>8s} | {r['modulation_3_5Hz_adv']:>10.4f} | {r['modulation_3_5Hz_pert']:>11.4f} | {r['spectral_centroid_hz']:>7.0f}Hz | {r['band_energy_adv'].get('600-700', 0):>10.2e} | {r['band_energy_adv'].get('700-800', 0):>10.2e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Modulation spectra
    ax = axes[0]
    for atk in ATTACKS:
        if not os.path.exists(atk["adv"]): continue
        adv = load_audio(atk["adv"], atk["fmt"])
        _, mod, mf = compute_modulation(adv, SR)
        mask = mf <= 15
        color = 'green' if atk["speech"] == True else ('blue' if atk["speech"] is None else 'red')
        lw = 2.5 if atk["speech"] is None else 1.5
        ax.plot(mf[mask], mod[mask]/max(mod[mask].max(), 1e-10), label=atk["label"], color=color, linewidth=lw)
    ax.axvspan(3, 5, alpha=0.15, color='green')
    ax.set_xlabel('Modulation freq (Hz)')
    ax.set_ylabel('Normalized energy')
    ax.set_title('Temporal Modulation Spectrum (digital)')
    ax.legend(fontsize=7)

    # 3-5Hz bar chart
    ax = axes[1]
    labels = [r["label"][:20] for r in results]
    vals = [r["modulation_3_5Hz_adv"] for r in results]
    colors = ['green' if r["speech_detected_physical"]==True else ('blue' if r["speech_detected_physical"] is None else 'red') for r in results]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_title('3-5Hz Modulation Energy')
    ax.set_ylabel('Relative energy')

    # Band energy profiles
    ax = axes[2]
    band_names = ['300-400', '400-500', '500-600', '600-700', '700-800', '800-1000']
    for r in results:
        vals = [r["band_energy_adv"].get(b, 0) for b in band_names]
        color = 'green' if r["speech_detected_physical"]==True else ('blue' if r["speech_detected_physical"] is None else 'red')
        lw = 2.5 if r["speech_detected_physical"] is None else 1.5
        ax.plot(band_names, [np.log10(v+1e-12) for v in vals], 'o-', label=r["label"][:20], color=color, linewidth=lw)
    ax.set_title('Band Energy Profile (digital)')
    ax.set_ylabel('log10(energy)')
    ax.legend(fontsize=7)
    plt.xticks(rotation=30, ha='right')

    plt.suptitle('New Modulation Attack (blue) vs Previous Attacks', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'modulation_attack_comparison.png'), dpi=150)
    print(f"\nSaved: {OUT_DIR}/modulation_attack_comparison.png")

    # Save
    with open(os.path.join(OUT_DIR, 'modulation_attack_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_DIR}/modulation_attack_comparison.json")


if __name__ == "__main__":
    main()
