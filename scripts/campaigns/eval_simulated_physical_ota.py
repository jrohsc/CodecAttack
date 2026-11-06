"""
Evaluate adversarial audio with simulated physical OTA channel.

Applies the full chain that models real MacBook -> air -> iPhone:
  1. Bandpass filter on perturbation -- speaker passband
  2. Empirical IR convolution -- measured room response
  3. Speaker nonlinearity (soft clipping) -- tanh distortion
  4. Background noise -- measured ambient

Can run on any experiment's audio/ folder after the fact.

Usage:
    python eval_simulated_physical_ota.py \
        --exp-dir results_empirical_ota_IR/exp_bandpass300_4000_nonlinear_eps3.0_1000steps
"""
import os, sys, json, argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE

sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from models.qwen2_audio import Qwen2AudioModel

from realistic_channel import SpeakerNonlinearity, EmpiricalNonlinearity


def load_carrier(music_name, sr=ENCODEC_SAMPLE_RATE):
    """Load clean carrier music."""
    from config import MUSIC_FILES
    import librosa
    path = MUSIC_FILES[music_name]
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def bandpass_filter(audio, sr, low_hz, high_hz, order=5):
    """Apply bandpass filter to audio."""
    nyq = sr / 2
    b, a = scipy_signal.butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return scipy_signal.filtfilt(b, a, audio).astype(np.float32)


def apply_empirical_ir(audio, sr=ENCODEC_SAMPLE_RATE):
    """Convolve with measured IR."""
    ir_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "empirical_ota", "channel_impulse_response.npy"
    )
    if not os.path.isfile(ir_path):
        ir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "soundcloud_recorded", "analysis", "channel_impulse_response.npy"
        )
    ir = np.load(ir_path).astype(np.float32)
    convolved = scipy_signal.fftconvolve(audio, ir, mode='full')[:len(audio)]
    return convolved.astype(np.float32)


def apply_speaker_nonlinearity(audio, drive=2.0, mix=0.3):
    """Apply soft clipping (tanh-based speaker nonlinearity)."""
    t = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    speaker = SpeakerNonlinearity(drive=drive, mix=mix)
    speaker.train()  # enable randomized drive
    with torch.no_grad():
        out = speaker(t, severity=1.0)
    return out.squeeze().numpy()


def apply_empirical_nonlinearity(audio, model_path=None):
    """Apply data-driven per-band nonlinearity from extract_nonlinearity.py."""
    t = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    nl = EmpiricalNonlinearity(model_path=model_path)
    nl.train(False)  # deterministic (no jitter)
    with torch.no_grad():
        out = nl(t, severity=1.0)
    return out.squeeze().numpy()


def add_background_noise(audio, sr=ENCODEC_SAMPLE_RATE, snr_db=25.0):
    """Add measured background noise."""
    noise_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "empirical_ota"
    )
    noise_files = sorted([f for f in os.listdir(noise_dir)
                          if f.startswith("residual_noise") and f.endswith(".npy")])
    if not noise_files:
        return audio
    noise = np.load(os.path.join(noise_dir, noise_files[0])).astype(np.float32)
    if len(noise) < len(audio):
        repeats = (len(audio) // len(noise)) + 1
        noise = np.tile(noise, repeats)[:len(audio)]
    else:
        noise = noise[:len(audio)]
    sig_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
        audio = audio + noise * scale
    return audio


def simulate_physical_ota(adv_audio, carrier_audio, sr=ENCODEC_SAMPLE_RATE,
                          bp_low=300, bp_high=4000,
                          speaker_drive=2.0, speaker_mix=0.3,
                          noise_snr_db=25.0, skip_ir=False,
                          use_empirical_nl=False, nl_model_path=None):
    """
    Full simulated physical OTA channel:
      1. Bandpass perturbation to speaker passband
      2. Empirical IR (room response) — optional
      3. Speaker nonlinearity — empirical per-band or tanh soft clip
      4. Background noise
    """
    min_len = min(len(adv_audio), len(carrier_audio))
    adv = adv_audio[:min_len]
    carrier = carrier_audio[:min_len]
    perturbation = adv - carrier
    perturbation_bp = bandpass_filter(perturbation, sr, bp_low, bp_high)
    audio = carrier + perturbation_bp

    if not skip_ir:
        audio = apply_empirical_ir(audio, sr)
    if use_empirical_nl:
        audio = apply_empirical_nonlinearity(audio, model_path=nl_model_path)
    else:
        audio = apply_speaker_nonlinearity(audio, drive=speaker_drive, mix=speaker_mix)
    audio = add_background_noise(audio, sr, snr_db=noise_snr_db)

    return audio


# Frequency bands for per-band evaluation
FREQUENCY_BANDS = [
    (50, 150, 'sub-bass'),
    (150, 300, 'bass'),
    (300, 600, 'low-mid'),
    (600, 1200, 'mid'),
    (1200, 2500, 'upper-mid'),
    (2500, 4000, 'presence'),
    (4000, 6000, 'brilliance'),
    (6000, 8000, 'air'),
]


def analyze_perturbation_bands(adv_audio, carrier_audio, sr=ENCODEC_SAMPLE_RATE):
    """Compute energy distribution of perturbation across frequency bands."""
    min_len = min(len(adv_audio), len(carrier_audio))
    perturbation = adv_audio[:min_len] - carrier_audio[:min_len]
    total_energy = np.sum(perturbation ** 2)
    if total_energy < 1e-12:
        return {}

    band_info = {}
    for low, high, name in FREQUENCY_BANDS:
        try:
            p_band = bandpass_filter(perturbation, sr, low, high)
            energy = np.sum(p_band ** 2)
            band_info[name] = {
                "energy_pct": float(100 * energy / total_energy),
                "rms": float(np.sqrt(np.mean(p_band ** 2))),
            }
        except Exception:
            band_info[name] = {"energy_pct": 0.0, "rms": 0.0}
    return band_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True,
                        help="Experiment directory with audio/ and config.json")
    parser.add_argument("--bp-low", type=float, default=300.0)
    parser.add_argument("--bp-high", type=float, default=4000.0)
    parser.add_argument("--speaker-drive", type=float, default=2.0)
    parser.add_argument("--speaker-mix", type=float, default=0.3)
    parser.add_argument("--noise-snr-db", type=float, default=25.0)
    parser.add_argument("--no-ir", action="store_true",
                        help="Skip IR convolution (only bandpass + nonlinearity + noise)")
    parser.add_argument("--empirical-nonlinearity", action="store_true",
                        help="Use data-driven per-band nonlinearity instead of tanh")
    parser.add_argument("--nl-model-path", type=str, default=None,
                        help="Path to nonlinearity_model.json")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    audio_dir = os.path.join(exp_dir, "audio")
    config_path = os.path.join(exp_dir, "config.json")

    with open(config_path) as f:
        config = json.load(f)

    targets = config["targets"]
    commands = config["commands"]
    prompt = config.get("prompt_text", "Listen to the audio and follow the instruction.")

    # Load model
    model = Qwen2AudioModel(
        model_path="${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a",
        device="cuda", dtype=torch.bfloat16,
    )

    # Find audio files
    wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    print(f"Evaluating {len(wav_files)} files with simulated physical OTA")
    print(f"  Bandpass: {args.bp_low}-{args.bp_high} Hz")
    print(f"  Speaker nonlinearity: drive={args.speaker_drive}, mix={args.speaker_mix}")
    print(f"  Noise SNR: {args.noise_snr_db} dB\n")

    # Cache carriers
    carrier_cache = {}

    results = []
    for wav_file in wav_files:
        base = wav_file.replace(".wav", "")
        parts = base.split("__")
        if len(parts) < 2:
            continue
        music_name = parts[0]
        cmd_key = parts[1]

        target = targets.get(cmd_key, "")
        if not target:
            continue

        # Load carrier
        if music_name not in carrier_cache:
            carrier_cache[music_name] = load_carrier(music_name, sr=ENCODEC_SAMPLE_RATE)
        carrier = carrier_cache[music_name]

        # Load adversarial audio
        adv_audio, sr = sf.read(os.path.join(audio_dir, wav_file))
        if sr != ENCODEC_SAMPLE_RATE:
            import librosa
            adv_audio = librosa.resample(adv_audio, orig_sr=sr, target_sr=ENCODEC_SAMPLE_RATE)

        # Analyze perturbation frequency bands
        band_info = analyze_perturbation_bands(adv_audio, carrier, sr=ENCODEC_SAMPLE_RATE)

        # Apply simulated physical OTA
        ota_audio = simulate_physical_ota(
            adv_audio, carrier, sr=ENCODEC_SAMPLE_RATE,
            bp_low=args.bp_low, bp_high=args.bp_high,
            speaker_drive=args.speaker_drive, speaker_mix=args.speaker_mix,
            noise_snr_db=args.noise_snr_db,
            skip_ir=args.no_ir,
            use_empirical_nl=args.empirical_nonlinearity,
            nl_model_path=args.nl_model_path,
        )

        # Resample to 16kHz and evaluate
        ota_tensor = torch.FloatTensor(ota_audio).unsqueeze(0)
        ota_16k = torchaudio.functional.resample(
            ota_tensor.to("cuda"), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )

        with torch.no_grad():
            output = model.generate(ota_16k, prompt=prompt)

        success = target.lower() in output.lower()

        # Also evaluate direct (no channel)
        adv_tensor = torch.FloatTensor(adv_audio).unsqueeze(0)
        adv_16k = torchaudio.functional.resample(
            adv_tensor.to("cuda"), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        with torch.no_grad():
            direct_output = model.generate(adv_16k, prompt=prompt)
        direct_success = target.lower() in direct_output.lower()

        results.append({
            "file": wav_file,
            "music": music_name,
            "command": cmd_key,
            "target": target,
            "direct_output": direct_output,
            "direct_success": direct_success,
            "sim_phys_ota_output": output,
            "sim_phys_ota_success": success,
            "perturbation_bands": band_info,
        })

        d_status = "OK" if direct_success else "FAIL"
        s_status = "OK" if success else "FAIL"
        print(f"  [{s_status}] {cmd_key}  (direct={d_status})")
        if not success:
            print(f"    Target: {target[:80]}")
            print(f"    Output: {output[:80]}")

        # Print band distribution
        if band_info:
            in_band = sum(band_info[b]["energy_pct"] for b in ["low-mid", "mid", "upper-mid", "presence"]
                          if b in band_info)
            print(f"    Perturbation energy in 300-4000Hz: {in_band:.1f}%")
        print()

    # Summary
    n = len(results)
    n_direct = sum(1 for r in results if r["direct_success"])
    n_sim_ota = sum(1 for r in results if r["sim_phys_ota_success"])
    print(f"{'='*65}")
    print(f"SIMULATED PHYSICAL OTA RESULTS")
    print(f"  Bandpass {args.bp_low}-{args.bp_high}Hz + Speaker NL (drive={args.speaker_drive})")
    print(f"{'='*65}")
    if n > 0:
        print(f"  Direct:             {n_direct}/{n} ({100*n_direct/n:.1f}%)")
        print(f"  Sim Physical OTA:   {n_sim_ota}/{n} ({100*n_sim_ota/n:.1f}%)")
    print(f"{'='*65}")

    # Per-band summary
    if results and results[0].get("perturbation_bands"):
        print(f"\nPerturbation energy distribution (averaged):")
        for low, high, name in FREQUENCY_BANDS:
            pcts = [r["perturbation_bands"].get(name, {}).get("energy_pct", 0) for r in results]
            in_bp = "  [IN BAND]" if 300 <= low and high <= 4000 else ""
            print(f"  {name:>12s} ({low}-{high}Hz): {np.mean(pcts):.1f}%{in_bp}")

    # Save
    out_path = os.path.join(exp_dir, "sim_physical_ota_eval.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "bp_low": args.bp_low, "bp_high": args.bp_high,
                "speaker_drive": args.speaker_drive, "speaker_mix": args.speaker_mix,
                "noise_snr_db": args.noise_snr_db,
            },
            "direct_success": n_direct, "sim_ota_success": n_sim_ota, "total": n,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
