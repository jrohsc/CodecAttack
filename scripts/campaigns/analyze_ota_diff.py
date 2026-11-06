"""Compare digital adversarial audio vs physical OTA recording."""
import numpy as np
import soundfile as sf
import subprocess, tempfile, os, sys
from scipy import signal
from scipy.signal import stft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MUSIC_DIR

digital_path = 'results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0/audio/calm_1__nav_turn_left.wav'
rec_path = 'results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0/audio_marshall_iphone_recording/calm_1__nav_turn_left.m4a'

# --- Load digital ---
dig_data, dig_sr = sf.read(digital_path, dtype='float32')
print(f"Digital: sr={dig_sr}, len={len(dig_data)}, dur={len(dig_data)/dig_sr:.2f}s, "
      f"range=[{dig_data.min():.4f}, {dig_data.max():.4f}], rms={np.sqrt(np.mean(dig_data**2)):.5f}")

# --- Load recorded (m4a -> wav) ---
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    tmp = f.name
subprocess.run(['ffmpeg', '-y', '-i', rec_path, '-ac', '1', '-ar', str(dig_sr),
                '-acodec', 'pcm_f32le', tmp], capture_output=True, check=True)
rec_data, rec_sr = sf.read(tmp, dtype='float32')
os.unlink(tmp)
print(f"Recorded: sr={rec_sr}, len={len(rec_data)}, dur={len(rec_data)/rec_sr:.2f}s, "
      f"range=[{rec_data.min():.4f}, {rec_data.max():.4f}], rms={np.sqrt(np.mean(rec_data**2)):.5f}")

# --- Align via cross-correlation ---
min_len = min(len(dig_data), len(rec_data))
corr = signal.correlate(rec_data[:min_len], dig_data[:min_len], mode='full')
lag = np.argmax(np.abs(corr)) - (min_len - 1)
print(f"\nAlignment: lag={lag} samples = {lag/dig_sr*1000:.1f}ms")

if lag > 0:
    rec_aligned = rec_data[lag:]
    dig_aligned = dig_data[:len(rec_aligned)]
else:
    dig_aligned = dig_data[-lag:]
    rec_aligned = rec_data[:len(dig_aligned)]

common_len = min(len(dig_aligned), len(rec_aligned))
dig_aligned = dig_aligned[:common_len]
rec_aligned = rec_aligned[:common_len]

scale = np.sqrt(np.mean(dig_aligned**2)) / (np.sqrt(np.mean(rec_aligned**2)) + 1e-10)
rec_scaled = rec_aligned * scale
print(f"Scale factor (RMS match): {scale:.4f}")

residual = rec_scaled - dig_aligned
correlation = np.corrcoef(dig_aligned, rec_scaled)[0, 1]
snr = 10 * np.log10(np.mean(dig_aligned**2) / (np.mean(residual**2) + 1e-20))
print(f"Global correlation: {correlation:.4f}")
print(f"Residual SNR: {snr:.1f} dB")

# --- Per-band spectral analysis ---
nperseg = 1024
f_dig, t_dig, Zdig = stft(dig_aligned, fs=dig_sr, nperseg=nperseg)
f_rec, t_rec, Zrec = stft(rec_scaled, fs=dig_sr, nperseg=nperseg)
min_t = min(Zdig.shape[1], Zrec.shape[1])
Zdig = Zdig[:, :min_t]
Zrec = Zrec[:, :min_t]

mag_dig = np.abs(Zdig)
mag_rec = np.abs(Zrec)
avg_dig = mag_dig.mean(axis=1)
avg_rec = mag_rec.mean(axis=1)
transfer = avg_rec / (avg_dig + 1e-10)
transfer_db = 20 * np.log10(transfer + 1e-10)

bands = [
    ("Sub-bass (0-100Hz)", 0, 100),
    ("Bass (100-300Hz)", 100, 300),
    ("Low-mid (300-1kHz)", 300, 1000),
    ("Mid (1-3kHz)", 1000, 3000),
    ("High-mid (3-6kHz)", 3000, 6000),
    ("High (6-12kHz)", 6000, 12000),
]

print(f"\n{'Band':<25} {'Dig RMS':>10} {'Rec RMS':>10} {'Gain(dB)':>10} {'Correlation':>12}")
print("-" * 70)
for name, lo, hi in bands:
    idx = np.where((f_dig >= lo) & (f_dig < hi))[0]
    if len(idx) == 0:
        continue
    d_band = mag_dig[idx].mean()
    r_band = mag_rec[idx].mean()
    gain = 20 * np.log10(r_band / (d_band + 1e-10) + 1e-10)
    band_dig = Zdig[idx, :].sum(axis=0)
    band_rec = Zrec[idx, :].sum(axis=0)
    bc = np.corrcoef(np.abs(band_dig), np.abs(band_rec))[0, 1]
    print(f"{name:<25} {d_band:>10.5f} {r_band:>10.5f} {gain:>+10.1f} {bc:>12.3f}")

# --- Perturbation analysis ---
clean_path = os.path.join(MUSIC_DIR, 'calm_1.mp3')
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    tmp2 = f.name
subprocess.run(['ffmpeg', '-y', '-i', clean_path, '-ac', '1', '-ar', str(dig_sr),
                '-acodec', 'pcm_f32le', '-t', str(len(dig_data)/dig_sr), tmp2],
               capture_output=True, check=True)
clean_data, _ = sf.read(tmp2, dtype='float32')
os.unlink(tmp2)

clean_len = min(len(clean_data), len(dig_data))
perturbation = dig_data[:clean_len] - clean_data[:clean_len]
pert_rms = np.sqrt(np.mean(perturbation**2))
music_rms = np.sqrt(np.mean(clean_data[:clean_len]**2))
print(f"\nPerturbation RMS: {pert_rms:.5f}")
print(f"Music RMS: {music_rms:.5f}")
print(f"Perturbation/Music ratio: {pert_rms/music_rms:.3f} ({20*np.log10(pert_rms/(music_rms+1e-10)):.1f} dB)")

_, _, Zpert = stft(perturbation[:clean_len], fs=dig_sr, nperseg=nperseg)
_, _, Zclean = stft(clean_data[:clean_len], fs=dig_sr, nperseg=nperseg)
min_tp = min(Zpert.shape[1], Zclean.shape[1])
Zpert = Zpert[:, :min_tp]
Zclean = Zclean[:, :min_tp]

print(f"\n{'Band':<25} {'Pert Energy':>12} {'Music Energy':>12} {'Pert/Music(dB)':>15} {'ChGain(dB)':>12}")
print("-" * 80)
for name, lo, hi in bands:
    idx = np.where((f_dig >= lo) & (f_dig < hi))[0]
    if len(idx) == 0:
        continue
    p_energy = np.abs(Zpert[idx]).mean()
    m_energy = np.abs(Zclean[idx]).mean()
    ratio_db = 20 * np.log10(p_energy / (m_energy + 1e-10) + 1e-10)
    ch_gain = transfer_db[idx].mean()
    print(f"{name:<25} {p_energy:>12.5f} {m_energy:>12.5f} {ratio_db:>+15.1f} {ch_gain:>+12.1f}")

# --- Phase coherence ---
print(f"\n--- Phase coherence (1.0=perfect, 0.0=random) ---")
for name, lo, hi in bands:
    idx = np.where((f_dig >= lo) & (f_dig < hi))[0]
    if len(idx) == 0:
        continue
    phase_dig = np.angle(Zdig[idx])
    phase_rec = np.angle(Zrec[idx])
    phase_diff = phase_rec - phase_dig
    coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
    print(f"{name:<25} {coherence:.4f}")

# --- Plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(14, 16))

t = np.arange(common_len) / dig_sr
axes[0].plot(t, dig_aligned, alpha=0.6, label='Digital', linewidth=0.3)
axes[0].plot(t, rec_scaled, alpha=0.6, label='Recorded (scaled)', linewidth=0.3)
axes[0].set_title(f'Waveform (corr={correlation:.4f}, lag={lag/dig_sr*1000:.1f}ms)')
axes[0].set_xlabel('Time (s)')
axes[0].legend()

axes[1].plot(f_dig, transfer_db)
axes[1].set_title('Channel Transfer Function (Recorded/Digital)')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Gain (dB)')
axes[1].set_xlim([0, 12000])
axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

from scipy.signal import spectrogram
f_s, t_s, Sdig = spectrogram(dig_aligned, fs=dig_sr, nperseg=512, noverlap=384)
_, _, Srec = spectrogram(rec_scaled[:len(dig_aligned)], fs=dig_sr, nperseg=512, noverlap=384)
min_ts = min(Sdig.shape[1], Srec.shape[1])
diff_spec = 10*np.log10(Srec[:, :min_ts]+1e-10) - 10*np.log10(Sdig[:, :min_ts]+1e-10)
im = axes[2].pcolormesh(t_s[:min_ts], f_s, diff_spec, cmap='RdBu_r', vmin=-20, vmax=20, shading='gouraud')
axes[2].set_title('Spectral Difference (Recorded - Digital) dB')
axes[2].set_ylabel('Freq (Hz)')
axes[2].set_ylim([0, 12000])
plt.colorbar(im, ax=axes[2], label='dB')

# Perturbation spectrum vs channel
pert_spec = 20*np.log10(np.abs(Zpert).mean(axis=1)+1e-10)
music_spec = 20*np.log10(np.abs(Zclean).mean(axis=1)+1e-10)
axes[3].plot(f_dig[:len(pert_spec)], pert_spec[:len(f_dig)], label='Perturbation', color='red')
axes[3].plot(f_dig[:len(music_spec)], music_spec[:len(f_dig)], label='Music', color='blue', alpha=0.5)
axes[3].plot(f_dig, transfer_db, label='Channel gain', color='green', alpha=0.7)
axes[3].set_title('Perturbation vs Music Spectrum + Channel')
axes[3].set_xlabel('Frequency (Hz)')
axes[3].set_ylabel('dB')
axes[3].set_xlim([0, 12000])
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
out = 'results_empirical_ota_IR/10_exp_robust_marshall_channel/eps5.0/digital_vs_recorded_analysis.png'
plt.savefig(out, dpi=150)
print(f"\nPlot saved to: {out}")
