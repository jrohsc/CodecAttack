"""Quick benchmark for joint latent+waveform attack on multiple commands."""

import os
import sys
import json
import time

import torch
import torchaudio.functional as AF

from config import (
    AGENT_COMMANDS, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    PROMPT_MODES, DEFAULT_PROMPT_MODE, RESULTS_DIR, compute_wer,
    load_saycan_commands,
)
from music_carrier import load_music_by_name
from joint_attack import JointAttacker


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--music", default="jazz_1")
    parser.add_argument("--latent-eps", type=float, default=10.0)
    parser.add_argument("--waveform-eps", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--n-eot-samples", type=int, default=4)
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--bandpass-low-hz", type=float, default=1000.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=4000.0)
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.add_argument("--noise-snr-std", type=float, default=5.0)
    parser.add_argument("--prompt-mode", default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--dataset", default="saycan", choices=["ours", "saycan"])
    parser.add_argument("--n-commands", type=int, default=None,
                        help="Limit number of commands (default: all)")
    parser.add_argument("--rir-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "rirs"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    # Load commands
    if args.dataset == "saycan":
        all_commands = load_saycan_commands()
    else:
        all_commands = AGENT_COMMANDS

    command_keys = list(all_commands.keys())
    if args.n_commands:
        command_keys = command_keys[:args.n_commands]

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(RESULTS_DIR), "results_joint",
        f"benchmark_{args.dataset}_leps{args.latent_eps}_weps{args.waveform_eps}_{args.music}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Joint Attack Benchmark: {len(command_keys)} commands ({args.dataset}), music={args.music}")
    print(f"  latent_eps={args.latent_eps}, waveform_eps={args.waveform_eps}")
    print(f"  steps={args.steps}, EoT={args.n_eot_samples}")
    print(f"  Output: {output_dir}")
    print()

    music_wav = load_music_by_name(args.music)

    attacker = JointAttacker(
        latent_eps=args.latent_eps,
        waveform_eps=args.waveform_eps,
        warmup_ratio=args.warmup_ratio,
        rir_dir=args.rir_dir,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        noise_snr_db=args.noise_snr_db,
        noise_snr_std=args.noise_snr_std,
        n_eot_samples=args.n_eot_samples,
    )

    results = []
    for i, cmd_key in enumerate(command_keys):
        target_text = all_commands[cmd_key][args.prompt_mode]

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(command_keys)}] {cmd_key}: \"{target_text}\"")
        print(f"{'='*70}")

        result = attacker.attack(
            music_wav,
            target_text=target_text,
            steps=args.steps,
            check_every=args.check_every,
            music_name=args.music,
            prompt=prompt_text,
        )

        # Also test Opus robustness on the saved OTA-successful audio
        opus_results = {}
        if result.adversarial_wav is not None:
            for bitrate in [64, 128]:
                try:
                    from run_benchmark import eval_opus
                    opus_out = eval_opus(attacker.base, result, [bitrate], prompt=prompt_text)
                    opus_results[bitrate] = opus_out.get(bitrate, {})
                except Exception as e:
                    opus_results[bitrate] = {"error": str(e)}

        entry = {
            "command": cmd_key,
            "target": target_text,
            "direct_output": result.adversarial_output,
            "ota_output": getattr(result, 'ota_output', ''),
            "direct_success": getattr(result, 'direct_success', False),
            "ota_success": result.success,
            "opus_64_success": opus_results.get(64, {}).get("success", None),
            "opus_128_success": opus_results.get(128, {}).get("success", None),
            "snr_db": result.snr_db,
            "duration_s": result.attack_duration_s,
        }
        results.append(entry)

        status = (f"  Direct={'OK' if entry['direct_success'] else 'FAIL'} | "
                  f"OTA={'OK' if entry['ota_success'] else 'FAIL'} | "
                  f"Opus64={'OK' if entry['opus_64_success'] else 'FAIL'} | "
                  f"Opus128={'OK' if entry['opus_128_success'] else 'FAIL'} | "
                  f"SNR={result.snr_db:.1f}dB | {result.attack_duration_s:.0f}s")
        print(status)

        # Save audio
        import soundfile as sf
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        adv_np = result.adversarial_wav.squeeze().cpu().numpy()
        sf.write(os.path.join(audio_dir, f"{cmd_key}_24k.wav"), adv_np, ENCODEC_SAMPLE_RATE)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    n = len(results)
    direct_ok = sum(1 for r in results if r["direct_success"])
    ota_ok = sum(1 for r in results if r["ota_success"])
    opus64_ok = sum(1 for r in results if r["opus_64_success"])
    opus128_ok = sum(1 for r in results if r["opus_128_success"])
    avg_snr = sum(r["snr_db"] for r in results) / n

    print(f"Direct:   {direct_ok}/{n} ({100*direct_ok/n:.0f}%)")
    print(f"OTA:      {ota_ok}/{n} ({100*ota_ok/n:.0f}%)")
    print(f"Opus 64:  {opus64_ok}/{n} ({100*opus64_ok/n:.0f}%)")
    print(f"Opus 128: {opus128_ok}/{n} ({100*opus128_ok/n:.0f}%)")
    print(f"Avg SNR:  {avg_snr:.1f} dB")

    print(f"\n{'Command':<20} {'Direct':>8} {'OTA':>8} {'Opus64':>8} {'Opus128':>8} {'SNR':>8}")
    print("-" * 70)
    for r in results:
        d = "OK" if r["direct_success"] else "FAIL"
        o = "OK" if r["ota_success"] else "FAIL"
        o64 = "OK" if r["opus_64_success"] else "FAIL"
        o128 = "OK" if r["opus_128_success"] else "FAIL"
        print(f"{r['command']:<20} {d:>8} {o:>8} {o64:>8} {o128:>8} {r['snr_db']:>7.1f}dB")

    # Save
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
