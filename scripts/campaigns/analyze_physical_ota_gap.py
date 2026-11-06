"""
Analyze sim-to-real gap: compare generated adversarial audio vs physical OTA recording.

For each paired file:
1. Spectral comparison (what frequencies survive)
2. Perturbation survival per band (how much adversarial energy remains)
3. Waveform correlation (time-domain similarity)
4. Compare simulated OTA vs physical OTA (where does the simulator diverge?)

Usage:
    python analyze_physical_ota_gap.py \
        --adv-dir results_empirical_ota_IR/1_exp_empirical_nonlinearity/audio \
        --rec-dir results_empirical_ota_IR/1_exp_empirical_nonlinearity/audio_mac_iphone_record_ota \
        --config results_empirical_ota_IR/1_exp_empirical_nonlinearity/config.json \
        --output results_empirical_ota_IR/1_exp_empirical_nonlinearity/ota_gap_analysis
"""
import os, sys, json, argparse, subprocess, tempfile
import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal
import soundfile as sf
import librosa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ENCODEC_SAMPLE_RATE, MUSIC_FILES

SR = ENCODEC_SAMPLE_RATE  # 24000

BANDS = [
    (20, 150, 'sub-bass'),
    (150, 300, 'bass'),
    (300, 800, 'low-mid'),
    (800, 2000, 'mid'),
    (2000, 4000, 'upper-mid'),
    (4000, 8000, 'presence'),
    (8000, 12000, 'brilliance'),
]


def load_m4a(path, target_sr=SR):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
             '-acodec', 'pcm_f32le', tmp],
            capture_output=True, check=True,
        )
        data, _ = sf.read(tmp, dtype='float32')
        return data
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def bandpass(audio, low, high, sr=SR, order=5):
    nyq = sr / 2
    if low <= 0:
        b, a = scipy_signal.butter(order, high / nyq, btype='low')
    elif high >= nyq:
        b, a = scipy_signal.butter(order, low / nyq, btype='high')
    else:
        b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return scipy_signal.filtfilt(b, a, audio).astype(np.float32)


def align_signals(ref, rec, sr=SR):
    """Align recorded to reference using cross-correlation."""
    seg_len = min(3 * sr, len(ref), len(rec))
    corr = np.correlate(rec[:seg_len + sr], ref[:seg_len], mode='full')
    delay = np.argmax(np.abs(corr)) - seg_len + 1
    if delay > 0:
        rec_aligned = rec[delay:]
    else:
        rec_aligned = np.pad(rec, (-delay, 0))
    min_len = min(len(ref), len(rec_aligned))
    return ref[:min_len], rec_aligned[:min_len], delay


def simulate_ota(adv_audio, sr=SR):
    """Run our simulated OTA channel for comparison."""
    # Load IR
    ir_path = os.path.join(os.path.dirname(__file__), "data", "empirical_ota", "channel_impulse_response.npy")
    ir = np.load(ir_path).astype(np.float32)
    sim = scipy_signal.fftconvolve(adv_audio, ir, mode='full')[:len(adv_audio)]

    # Apply empirical nonlinearity
    import torch
    from realistic_channel import EmpiricalNonlinearity
    nl = EmpiricalNonlinearity()
    nl.train(False)
    t = torch.FloatTensor(sim).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        sim = nl(t).squeeze().numpy()

    # Add noise (representative sample)
    noise_dir = os.path.join(os.path.dirname(__file__), "data", "empirical_ota")
    noise_files = sorted([f for f in os.listdir(noise_dir) if f.startswith("residual_noise") and f.endswith(".npy")])
    if noise_files:
        noise = np.load(os.path.join(noise_dir, noise_files[0])).astype(np.float32)
        if len(noise) < len(sim):
            noise = np.tile(noise, (len(sim) // len(noise)) + 1)[:len(sim)]
        else:
            noise = noise[:len(sim)]
        sig_power = np.mean(sim ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            scale = np.sqrt(sig_power / (noise_power * 10 ** (25 / 10)))
            sim = sim + noise * scale

    return sim.astype(np.float32)


def analyze_pair(adv, rec_phys, carrier, cmd_name, output_dir):
    """Full analysis of one adversarial/recorded pair."""
    # Align
    adv_a, rec_a, delay = align_signals(adv, rec_phys)
    print(f"  Delay: {delay} samples ({delay/SR*1000:.1f} ms)")

    # Simulated OTA for comparison — apply to adv_a directly (same time base, no alignment needed)
    sim_a = simulate_ota(adv_a)
    min_len = min(len(adv_a), len(rec_a), len(sim_a))
    adv_a, rec_a, sim_a = adv_a[:min_len], rec_a[:min_len], sim_a[:min_len]

    # Carrier (trim to same length)
    carrier_t = carrier[:min_len] if len(carrier) >= min_len else np.pad(carrier, (0, min_len - len(carrier)))

    # Extract perturbation
    perturbation = adv_a - carrier_t

    # Overall correlation
    corr_sim = np.corrcoef(adv_a, sim_a)[0, 1]
    corr_phys = np.corrcoef(adv_a, rec_a)[0, 1]
    corr_sim_phys = np.corrcoef(sim_a, rec_a)[0, 1]

    # Gain
    gain_sim = np.dot(sim_a, adv_a) / (np.dot(adv_a, adv_a) + 1e-10)
    gain_phys = np.dot(rec_a, adv_a) / (np.dot(adv_a, adv_a) + 1e-10)

    print(f"  Correlation — sim: {corr_sim:.4f}, phys: {corr_phys:.4f}, sim_vs_phys: {corr_sim_phys:.4f}")
    print(f"  Gain — sim: {20*np.log10(abs(gain_sim)+1e-10):.1f} dB, phys: {20*np.log10(abs(gain_phys)+1e-10):.1f} dB")

    result = {
        'command': cmd_name,
        'delay_samples': int(delay),
        'delay_ms': float(delay / SR * 1000),
        'corr_adv_sim': float(corr_sim),
        'corr_adv_phys': float(corr_phys),
        'corr_sim_phys': float(corr_sim_phys),
        'gain_sim_db': float(20 * np.log10(abs(gain_sim) + 1e-10)),
        'gain_phys_db': float(20 * np.log10(abs(gain_phys) + 1e-10)),
        'bands': {},
    }

    # ── Per-band analysis ──
    print(f"  {'Band':<12} {'Perturb RMS':>11} {'Sim RMS':>11} {'Phys RMS':>11} {'Survival(sim)':>14} {'Survival(phys)':>15} {'Sim-Phys corr':>14}")
    for low, high, name in BANDS:
        p_band = bandpass(perturbation, low, high)
        adv_band = bandpass(adv_a, low, high)
        sim_band = bandpass(sim_a, low, high)
        rec_band = bandpass(rec_a, low, high)
        carrier_band = bandpass(carrier_t, low, high)

        p_rms = np.sqrt(np.mean(p_band ** 2))
        adv_rms = np.sqrt(np.mean(adv_band ** 2))
        sim_rms = np.sqrt(np.mean(sim_band ** 2))
        rec_rms = np.sqrt(np.mean(rec_band ** 2))

        # Perturbation survival: how much of the perturbation band exists in output?
        # Compare (output - carrier_band) to perturbation_band
        sim_perturb = sim_band - bandpass(sim_a, low, high) * 0  # full sim signal in this band
        rec_perturb = rec_band

        # Correlation between adv band and output band
        if adv_rms > 1e-10 and sim_rms > 1e-10:
            corr_sim_band = np.corrcoef(adv_band, sim_band)[0, 1]
        else:
            corr_sim_band = 0.0
        if adv_rms > 1e-10 and rec_rms > 1e-10:
            corr_phys_band = np.corrcoef(adv_band, rec_band)[0, 1]
        else:
            corr_phys_band = 0.0
        if sim_rms > 1e-10 and rec_rms > 1e-10:
            corr_sp_band = np.corrcoef(sim_band, rec_band)[0, 1]
        else:
            corr_sp_band = 0.0

        # Survival ratio (RMS of output band / RMS of adv band)
        surv_sim = sim_rms / (adv_rms + 1e-10)
        surv_phys = rec_rms / (adv_rms + 1e-10)

        print(f"  {name:<12} {p_rms:>11.6f} {sim_rms:>11.6f} {rec_rms:>11.6f} {surv_sim:>13.3f}x {surv_phys:>14.3f}x {corr_sp_band:>14.4f}")

        result['bands'][name] = {
            'low': low, 'high': high,
            'perturbation_rms': float(p_rms),
            'adv_rms': float(adv_rms),
            'sim_rms': float(sim_rms),
            'phys_rms': float(rec_rms),
            'survival_sim': float(surv_sim),
            'survival_phys': float(surv_phys),
            'corr_adv_sim': float(corr_sim_band),
            'corr_adv_phys': float(corr_phys_band),
            'corr_sim_phys': float(corr_sp_band),
        }

    # ── Spectral comparison figure ──
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel 1: PSD comparison (adv vs sim vs phys)
    ax1 = fig.add_subplot(gs[0, :])
    for sig, label, color in [
        (adv_a, 'Adversarial (generated)', 'blue'),
        (sim_a, 'Simulated OTA', 'green'),
        (rec_a, 'Physical OTA (recorded)', 'red'),
        (carrier_t, 'Clean carrier', 'gray'),
    ]:
        freqs_psd, psd = scipy_signal.welch(sig, fs=SR, nperseg=4096)
        psd_db = 10 * np.log10(psd + 1e-15)
        ax1.plot(freqs_psd, psd_db, label=label, color=color, alpha=0.8, linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlim(20, 12000)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (dB)')
    ax1.set_title(f'Power Spectral Density — {cmd_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Per-band survival ratio
    ax2 = fig.add_subplot(gs[1, 0])
    band_names = [name for _, _, name in BANDS]
    surv_sim_vals = [result['bands'][n]['survival_sim'] for n in band_names]
    surv_phys_vals = [result['bands'][n]['survival_phys'] for n in band_names]
    x = np.arange(len(band_names))
    ax2.bar(x - 0.2, surv_sim_vals, 0.35, label='Simulated OTA', color='green', alpha=0.7)
    ax2.bar(x + 0.2, surv_phys_vals, 0.35, label='Physical OTA', color='red', alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names, rotation=45, ha='right')
    ax2.set_ylabel('RMS ratio (output / input)')
    ax2.set_title('Signal Survival per Band')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    # Panel 3: Per-band sim-vs-phys correlation
    ax3 = fig.add_subplot(gs[1, 1])
    corr_sp_vals = [result['bands'][n]['corr_sim_phys'] for n in band_names]
    corr_ap_vals = [result['bands'][n]['corr_adv_phys'] for n in band_names]
    ax3.bar(x - 0.2, corr_ap_vals, 0.35, label='Adv vs Physical', color='orange', alpha=0.7)
    ax3.bar(x + 0.2, corr_sp_vals, 0.35, label='Sim vs Physical', color='purple', alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(band_names, rotation=45, ha='right')
    ax3.set_ylabel('Pearson correlation')
    ax3.set_title('Correlation per Band')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(-0.5, 1.0)

    # Panel 4: Transfer function comparison (sim vs phys)
    ax4 = fig.add_subplot(gs[2, 0])
    n_fft = min(len(adv_a), 2**16)
    freqs_tf = np.fft.rfftfreq(n_fft, d=1.0/SR)
    A = np.fft.rfft(adv_a[:n_fft])
    S = np.fft.rfft(sim_a[:n_fft])
    P = np.fft.rfft(rec_a[:n_fft])
    # Smoothed TF
    win = 128
    A_pow = np.convolve(np.abs(A)**2, np.ones(win)/win, mode='same')
    S_pow = np.convolve(np.abs(S)**2, np.ones(win)/win, mode='same')
    P_pow = np.convolve(np.abs(P)**2, np.ones(win)/win, mode='same')
    tf_sim = 10 * np.log10(S_pow / (A_pow + 1e-15) + 1e-15)
    tf_phys = 10 * np.log10(P_pow / (A_pow + 1e-15) + 1e-15)
    ax4.plot(freqs_tf, tf_sim, 'g-', alpha=0.7, linewidth=1.5, label='Simulated TF')
    ax4.plot(freqs_tf, tf_phys, 'r-', alpha=0.7, linewidth=1.5, label='Physical TF')
    ax4.fill_between(freqs_tf, tf_sim, tf_phys, alpha=0.15, color='orange', label='Sim-Real gap')
    ax4.set_xscale('log')
    ax4.set_xlim(20, 12000)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Gain (dB)')
    ax4.set_title('Transfer Function: Simulated vs Physical')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-50, 20)

    # Panel 5: Time-domain waveform snippet (first 0.5s)
    ax5 = fig.add_subplot(gs[2, 1])
    t_show = int(0.5 * SR)
    t_axis = np.arange(t_show) / SR * 1000
    ax5.plot(t_axis, adv_a[:t_show], 'b-', alpha=0.5, linewidth=0.5, label='Adversarial')
    ax5.plot(t_axis, sim_a[:t_show], 'g-', alpha=0.5, linewidth=0.5, label='Sim OTA')
    ax5.plot(t_axis, rec_a[:t_show], 'r-', alpha=0.5, linewidth=0.5, label='Phys OTA')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Amplitude')
    ax5.set_title('Waveform (first 500ms)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f'Sim-to-Real OTA Gap Analysis: {cmd_name}', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, f'{cmd_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-dir", required=True)
    parser.add_argument("--rec-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    # Load carrier
    music_name = config['music_files'][0]
    carrier, _ = librosa.load(MUSIC_FILES[music_name], sr=SR, mono=True)

    # Find matching pairs
    rec_files = {Path(f).stem: f for f in os.listdir(args.rec_dir) if f.endswith('.m4a')}
    adv_files = {Path(f).stem: f for f in os.listdir(args.adv_dir) if f.endswith('.wav')}

    common = sorted(set(rec_files.keys()) & set(adv_files.keys()))
    print(f"Found {len(common)} matched pairs\n")

    all_results = []
    for cmd_name in common:
        print(f"\n{'='*70}")
        print(f"Analyzing: {cmd_name}")

        adv, _ = sf.read(os.path.join(args.adv_dir, adv_files[cmd_name]), dtype='float32')
        rec = load_m4a(os.path.join(args.rec_dir, rec_files[cmd_name]), target_sr=SR)

        result = analyze_pair(adv, rec, carrier, cmd_name, args.output)
        all_results.append(result)

    # ── Summary across all pairs ──
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*70}")

    # Average per-band stats
    print(f"\n{'Band':<12} {'Surv(sim)':>10} {'Surv(phys)':>11} {'Gap':>10} {'Corr(s-p)':>10}")
    for _, _, name in BANDS:
        ss = np.mean([r['bands'][name]['survival_sim'] for r in all_results])
        sp = np.mean([r['bands'][name]['survival_phys'] for r in all_results])
        csp = np.mean([r['bands'][name]['corr_sim_phys'] for r in all_results])
        gap = ss - sp
        print(f"  {name:<12} {ss:>9.3f}x {sp:>10.3f}x {gap:>+9.3f} {csp:>10.4f}")

    avg_corr_sim = np.mean([r['corr_adv_sim'] for r in all_results])
    avg_corr_phys = np.mean([r['corr_adv_phys'] for r in all_results])
    avg_corr_sp = np.mean([r['corr_sim_phys'] for r in all_results])
    print(f"\nOverall correlation — adv-sim: {avg_corr_sim:.4f}, adv-phys: {avg_corr_phys:.4f}, sim-phys: {avg_corr_sp:.4f}")

    # Save
    with open(os.path.join(args.output, 'analysis_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
