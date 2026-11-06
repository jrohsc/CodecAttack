"""
Compare ALL physical OTA recordings: speech-detected vs non-speech.
Find the acoustic signature that predicts speech detection.

Speech group: 녹음159 (bp400-700), 녹음155 (bp500-800), 녹음157 (bp500-800 low)
Non-speech group: everything else
"""
import os, sys, subprocess, tempfile, json
import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import stft

sys.path.insert(0, "${REPO_ROOT}")
sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from config import TARGET_SAMPLE_RATE

BASE = "${REPO_ROOT}/results_empirical_analysis_noise_cancellation/27_exp_narrow_bandpass_marshall"
FIND_DIR = os.path.join(BASE, "find_bp_400_700_working")
OUT_DIR = os.path.join(BASE, "speech_detection_analysis")
os.makedirs(OUT_DIR, exist_ok=True)
SR = TARGET_SAMPLE_RATE

# Use original file locations (avoids Korean filename encoding issues)
RECORDINGS = [
    {"path": os.path.join(BASE, "bp300-600/audio_marshall_iphone_record_bp300-600.m4a"),
     "label": "bp300-600", "speech": False, "output": "fast piece of music on guitar"},
    {"path": os.path.join(BASE, "bp500_800/audio_marshall_iphone_record_bp500-800_no-ecc.m4a"),
     "label": "bp500-800", "speech": True, "output": "instruction is to play an instrument"},
    {"path": os.path.join(BASE, "bp300-600/audio_marshall_iphone_record_bp300-600_low_volume.m4a"),
     "label": "bp300-600 low", "speech": False, "output": "fast piece of music on guitar"},
    {"path": os.path.join(BASE, "bp500_800/audio_marshall_iphone_record_bp500-800_low_volume_no-ecc.m4a"),
     "label": "bp500-800 low", "speech": True, "output": "woman speaks in background"},
    {"path": os.path.join(BASE, "bp800_1200/audio_marshall_iphone_recorded.m4a"),
     "label": "bp800-1200", "speech": False, "output": "music in F major"},
    {"path": os.path.join(BASE, "find_bp_400_700_working/bp400-700_eps3.0_steps_4000_ecc_3reps.m4a"),
     "label": "bp400-700 *", "speech": True, "output": "man speaking 'Order them yourself'"},
    {"path": os.path.join(BASE, "bp500_800_v2/marshall_recorded.m4a"),
     "label": "bp500-800 v2", "speech": False, "output": "dog barking"},
    {"path": os.path.join(BASE, "bp300-600,800-1200/marshall_recorded.m4a"),
     "label": "bp300-600+800-1200", "speech": False, "output": "telephone bell ringing"},
    {"path": os.path.join(BASE, "bp400_600_no_ecc/marshall_reocrded.m4a"),
     "label": "bp400-600 no-ecc", "speech": False, "output": "classical music violin"},
    {"path": os.path.join(BASE, "bp400_700_go_down/marshall_recorded.m4a"),
     "label": "bp400-700 go_down", "speech": False, "output": "car engine revving"},
    {"path": os.path.join(BASE, "bp400_600/marshall_recorded.m4a"),
     "label": "bp400-600", "speech": False, "output": "things and mechanisms"},
    {"path": os.path.join(BASE, "bp400_700_steps5000_ecc_rep5/marshall-recorded.m4a"),
     "label": "bp400-700 ecc5", "speech": False, "output": "car engine starting up"},
]


def load_m4a(path):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_out = f.name
    # Symlink to avoid unicode path issues with ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as f:
        tmp_in = f.name
    try:
        os.unlink(tmp_in)
        os.symlink(path, tmp_in)
        subprocess.run(['ffmpeg', '-y', '-i', tmp_in, '-ac', '1', '-ar', str(SR),
                        '-acodec', 'pcm_f32le', tmp_out], capture_output=True, check=True)
        data, sr = sf.read(tmp_out, dtype='float32')
        return data
    finally:
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)
        if os.path.exists(tmp_in):
            os.unlink(tmp_in)


def compute_features(audio, sr):
    """Compute comprehensive acoustic features."""
    features = {}

    # Basic stats
    features['rms'] = float(np.sqrt(np.mean(audio**2)))
    features['peak'] = float(np.max(np.abs(audio)))
    features['crest_factor'] = features['peak'] / max(features['rms'], 1e-10)

    # Spectrogram
    n_fft = 2048
    hop = 512
    f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx)
    power = mag ** 2

    # Band energies (fine-grained)
    bands = {
        '0-200': (0, 200), '200-300': (200, 300), '300-400': (300, 400),
        '400-500': (400, 500), '500-600': (500, 600), '600-700': (600, 700),
        '700-800': (700, 800), '800-1000': (800, 1000), '1000-1500': (1000, 1500),
        '1500-2000': (1500, 2000), '2000-3000': (2000, 3000), '3000-4000': (3000, 4000),
        '4000-8000': (4000, 8000),
    }

    band_energies = {}
    for bname, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        band_energies[bname] = float(np.mean(power[mask, :]))
    features['band_energy'] = band_energies

    # Spectral centroid
    avg_spec = np.mean(power, axis=1)
    features['spectral_centroid'] = float(np.sum(f * avg_spec) / max(np.sum(avg_spec), 1e-10))

    # Spectral bandwidth
    centroid = features['spectral_centroid']
    features['spectral_bandwidth'] = float(np.sqrt(
        np.sum(((f - centroid)**2) * avg_spec) / max(np.sum(avg_spec), 1e-10)))

    # Spectral flatness (lower = more tonal/speech-like)
    geo_mean = np.exp(np.mean(np.log(mag + 1e-10), axis=0))
    arith_mean = np.mean(mag, axis=0)
    flatness = geo_mean / (arith_mean + 1e-10)
    features['spectral_flatness_mean'] = float(np.mean(flatness))
    features['spectral_flatness_std'] = float(np.std(flatness))

    # Spectral rolloff (freq below which 85% of energy)
    cumsum = np.cumsum(avg_spec)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    features['spectral_rolloff_85'] = float(f[min(rolloff_idx, len(f)-1)])

    # Temporal modulation spectrum
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = (len(audio) - frame_len) // hop_len + 1
    energy_envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start:start + frame_len]
        energy_envelope[i] = np.sqrt(np.mean(frame**2))

    mod_spec = np.abs(np.fft.rfft(energy_envelope - np.mean(energy_envelope)))
    mod_freqs = np.fft.rfftfreq(len(energy_envelope), d=0.010)

    # Modulation energy in key bands
    for mod_name, (mod_lo, mod_hi) in [('1-3Hz', (1, 3)), ('3-5Hz', (3, 5)),
                                         ('5-8Hz', (5, 8)), ('8-15Hz', (8, 15))]:
        mask = (mod_freqs >= mod_lo) & (mod_freqs < mod_hi)
        total_mask = (mod_freqs >= 0.5) & (mod_freqs <= 20)
        features[f'modulation_{mod_name}'] = float(
            np.mean(mod_spec[mask]) / max(np.mean(mod_spec[total_mask]), 1e-10))

    # Peak modulation frequency
    valid = (mod_freqs >= 1) & (mod_freqs <= 15)
    peak_idx = np.argmax(mod_spec[valid])
    features['peak_modulation_freq'] = float(mod_freqs[valid][peak_idx])

    # Envelope dynamics (speech has high variance)
    features['envelope_mean'] = float(np.mean(energy_envelope))
    features['envelope_std'] = float(np.std(energy_envelope))
    features['envelope_cv'] = features['envelope_std'] / max(features['envelope_mean'], 1e-10)

    # Zero crossing rate (speech has moderate ZCR)
    zcr = np.mean(np.abs(np.diff(np.sign(audio))) / 2)
    features['zero_crossing_rate'] = float(zcr)

    # Harmonic-to-noise ratio proxy: autocorrelation peak
    chunk = audio[:sr]  # first second
    autocorr = np.correlate(chunk, chunk, mode='full')
    autocorr = autocorr[len(chunk):]
    autocorr = autocorr / autocorr[0]
    # Find first peak after initial decay (pitch period)
    min_lag = int(sr / 500)  # 500Hz max
    max_lag = int(sr / 80)   # 80Hz min
    if max_lag < len(autocorr):
        segment = autocorr[min_lag:max_lag]
        features['autocorr_peak'] = float(np.max(segment))
    else:
        features['autocorr_peak'] = 0.0

    # Formant band energy ratios
    e_400_700 = sum(band_energies[b] for b in ['400-500', '500-600', '600-700'])
    e_total = sum(band_energies.values())
    features['energy_ratio_400_700'] = e_400_700 / max(e_total, 1e-10)

    e_below_400 = sum(band_energies[b] for b in ['0-200', '200-300', '300-400'])
    e_above_1000 = sum(band_energies[b] for b in ['1000-1500', '1500-2000', '2000-3000', '3000-4000', '4000-8000'])
    features['energy_ratio_low_vs_high'] = e_below_400 / max(e_above_1000, 1e-10)

    # F1 region (400-700Hz) temporal variance — speech has dynamic formants
    f1_mask = (f >= 400) & (f < 700)
    f1_energy_over_time = np.mean(power[f1_mask, :], axis=0)
    features['f1_temporal_variance'] = float(np.std(f1_energy_over_time) / max(np.mean(f1_energy_over_time), 1e-10))

    # F1/F2 energy ratio over time — speech has varying ratio
    f2_mask = (f >= 800) & (f < 1500)
    f2_energy_over_time = np.mean(power[f2_mask, :], axis=0)
    ratio_over_time = f1_energy_over_time / (f2_energy_over_time + 1e-10)
    features['f1_f2_ratio_mean'] = float(np.mean(ratio_over_time))
    features['f1_f2_ratio_std'] = float(np.std(ratio_over_time))

    return features


def main():
    all_features = []

    for rec in RECORDINGS:
        path = rec["path"]
        print(f"Processing: {rec['label']} ({'SPEECH' if rec['speech'] else 'no speech'})")

        audio = load_m4a(path)
        feats = compute_features(audio, SR)
        feats['label'] = rec['label']
        feats['speech'] = rec['speech']
        feats['output'] = rec['output']
        all_features.append(feats)

    # Separate groups
    speech_feats = [f for f in all_features if f['speech']]
    nospeech_feats = [f for f in all_features if not f['speech']]

    # Print comparison table
    key_metrics = [
        'rms', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness_mean',
        'spectral_rolloff_85', 'modulation_1-3Hz', 'modulation_3-5Hz', 'modulation_5-8Hz',
        'peak_modulation_freq', 'envelope_cv', 'zero_crossing_rate', 'autocorr_peak',
        'energy_ratio_400_700', 'f1_temporal_variance', 'f1_f2_ratio_mean', 'f1_f2_ratio_std',
        'crest_factor',
    ]

    print(f"\n{'='*140}")
    print(f"{'Metric':<30s}", end="")
    for f in all_features:
        marker = "★" if f['speech'] else " "
        print(f" | {marker}{f['label'][:12]:>12s}", end="")
    print(f" | {'Speech avg':>10s} | {'Other avg':>10s} | {'Ratio':>6s}")
    print(f"{'-'*140}")

    discriminative_metrics = []

    for metric in key_metrics:
        print(f"{metric:<30s}", end="")
        s_vals = []
        n_vals = []
        for f in all_features:
            val = f[metric]
            marker = "★" if f['speech'] else " "
            print(f" | {marker}{val:>12.4f}", end="")
            if f['speech']:
                s_vals.append(val)
            else:
                n_vals.append(val)

        s_mean = np.mean(s_vals)
        n_mean = np.mean(n_vals)
        ratio = s_mean / max(n_mean, 1e-10)
        print(f" | {s_mean:>10.4f} | {n_mean:>10.4f} | {ratio:>6.2f}x")

        # Track how discriminative this metric is
        # Use separation: |speech_mean - nonspeech_mean| / pooled_std
        pooled_std = np.std(s_vals + n_vals)
        if pooled_std > 0:
            separation = abs(s_mean - n_mean) / pooled_std
        else:
            separation = 0
        discriminative_metrics.append((metric, separation, s_mean, n_mean, ratio))

    # Sort by discriminative power
    discriminative_metrics.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*80}")
    print("TOP DISCRIMINATIVE METRICS (speech vs non-speech):")
    print(f"{'='*80}")
    for metric, sep, s_mean, n_mean, ratio in discriminative_metrics[:10]:
        direction = "HIGHER in speech" if s_mean > n_mean else "LOWER in speech"
        print(f"  {sep:.3f}  {metric:<30s}  speech={s_mean:.4f}  other={n_mean:.4f}  ({direction})")

    # Band energy comparison
    print(f"\n{'='*80}")
    print("BAND ENERGY COMPARISON (speech vs non-speech averages):")
    print(f"{'='*80}")
    band_names = list(all_features[0]['band_energy'].keys())
    print(f"{'Band':<15s} | {'Speech avg':>12s} | {'Other avg':>12s} | {'Ratio':>8s}")
    print(f"{'-'*55}")
    for bname in band_names:
        s_vals = [f['band_energy'][bname] for f in speech_feats]
        n_vals = [f['band_energy'][bname] for f in nospeech_feats]
        s_mean = np.mean(s_vals)
        n_mean = np.mean(n_vals)
        ratio = s_mean / max(n_mean, 1e-10)
        marker = " <<<" if ratio > 1.3 or ratio < 0.7 else ""
        print(f"{bname:<15s} | {s_mean:>12.2e} | {n_mean:>12.2e} | {ratio:>7.2f}x{marker}")

    # ===== PLOTS =====

    # Plot 1: Key metrics bar chart — speech vs non-speech
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    plot_metrics = ['modulation_3-5Hz', 'spectral_centroid', 'envelope_cv',
                    'f1_temporal_variance', 'spectral_flatness_mean', 'autocorr_peak',
                    'energy_ratio_400_700', 'zero_crossing_rate', 'spectral_rolloff_85']

    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx // 3, idx % 3]
        vals = [f[metric] for f in all_features]
        colors = ['green' if f['speech'] else 'red' for f in all_features]
        labels = [f['label'][:15] for f in all_features]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric, fontsize=10)
        # Add mean lines
        s_mean = np.mean([f[metric] for f in speech_feats])
        n_mean = np.mean([f[metric] for f in nospeech_feats])
        ax.axhline(y=s_mean, color='green', linestyle='--', alpha=0.5, label=f'speech avg={s_mean:.3f}')
        ax.axhline(y=n_mean, color='red', linestyle='--', alpha=0.5, label=f'other avg={n_mean:.3f}')
        ax.legend(fontsize=7)

    plt.suptitle('Speech-detected (green) vs Non-speech (red) recordings', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'speech_vs_nonspeech_metrics.png'), dpi=150)
    print(f"\nSaved: {OUT_DIR}/speech_vs_nonspeech_metrics.png")

    # Plot 2: Band energy profiles
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(band_names))
    for f in all_features:
        vals = [np.log10(f['band_energy'][b] + 1e-12) for b in band_names]
        style = '-o' if f['speech'] else '--x'
        color = 'green' if f['speech'] else 'red'
        alpha = 0.9 if f['speech'] else 0.3
        lw = 2 if f['speech'] else 1
        ax.plot(x, vals, style, label=f['label'], color=color, alpha=alpha, linewidth=lw, markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_ylabel('log10(energy)')
    ax.set_title('Band Energy Profiles: Speech (green) vs Non-speech (red)')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'band_energy_profiles.png'), dpi=150)
    print(f"Saved: {OUT_DIR}/band_energy_profiles.png")

    # Plot 3: Modulation spectrum for each
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for idx, rec in enumerate(RECORDINGS):
        ax = axes[idx // 4, idx % 4]
        path = rec["path"]
        audio = load_m4a(path)

        frame_len = int(0.025 * SR)
        hop_len = int(0.010 * SR)
        n_frames = (len(audio) - frame_len) // hop_len + 1
        env = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_len
            env[i] = np.sqrt(np.mean(audio[start:start+frame_len]**2))

        mod_spec = np.abs(np.fft.rfft(env - np.mean(env)))
        mod_freqs = np.fft.rfftfreq(len(env), d=0.010)
        mask = mod_freqs <= 20

        color = 'green' if rec['speech'] else 'red'
        ax.plot(mod_freqs[mask], mod_spec[mask] / max(mod_spec[mask].max(), 1e-10), color=color)
        ax.axvspan(3, 5, alpha=0.15, color='blue')
        ax.set_title(f"{rec['label']}\n{'SPEECH' if rec['speech'] else 'no speech'}", fontsize=8,
                     color=color, fontweight='bold')
        ax.set_xlabel('Mod freq (Hz)', fontsize=7)
        ax.set_xlim(0, 15)

    plt.suptitle('Temporal Modulation Spectra (blue band = 3-5Hz speech)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'modulation_spectra_all.png'), dpi=150)
    print(f"Saved: {OUT_DIR}/modulation_spectra_all.png")

    # Save analysis
    out_path = os.path.join(OUT_DIR, 'speech_vs_nonspeech_analysis.json')
    # Clean features for JSON
    for f in all_features:
        for k, v in f.items():
            if isinstance(v, np.floating):
                f[k] = float(v)
    with open(out_path, 'w') as ff:
        json.dump({
            "recordings": all_features,
            "discriminative_metrics": [
                {"metric": m, "separation": round(s, 4), "speech_mean": round(sm, 6),
                 "nonspeech_mean": round(nm, 6), "ratio": round(r, 4)}
                for m, s, sm, nm, r in discriminative_metrics
            ],
        }, ff, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
