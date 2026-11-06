"""
Benchmark for airgap-robust waveform-domain attacks.

Mirrors run_robust_benchmark.py but uses RobustWaveformAttacker instead of
RobustLatentCodecAttacker. Results are saved to results_waveform_airgap/.

For each (music, command) pair:
  1. Run robust waveform attack via RobustWaveformAttacker
  2. Evaluate in 4 modes:
     a) Direct  -- adversarial WAV fed straight to model
     b) Opus 64 kbps  -- compressed via Opus
     c) Opus 128 kbps -- compressed via Opus
     d) OTA simulation -- Opus + empirical IR + noise

Usage:
    python run_robust_waveform_benchmark.py --target-model qwen2_audio \\
        --music jazz_1 --eps 0.01 --dataset saycan

    python run_robust_waveform_benchmark.py --quick --target-model qwen2_audio \\
        --music jazz_1 --eps 0.01
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

from config import (
    MUSIC_FILES, DEFAULT_MUSIC, AGENT_COMMANDS, RESULTS_DIR, PROJECT_ROOT,
    WAVEFORM_EPS, WAVEFORM_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
    OPUS_EVAL_BITRATES, PROMPT_MODES, DEFAULT_PROMPT_MODE, TARGET_MODEL,
    DATASET_CHOICES, load_saycan_commands,
    compute_wer, save_results_summary, ENCODEC_SAMPLE_RATE,
)
from music_carrier import load_music_by_name, resolve_music_path
from robust_waveform_attack import RobustWaveformAttacker
from demo_ota import simulate_ota_channel, simulate_empirical_ota_channel

from run_benchmark import (
    _music_short_name, CATEGORY_PREFIXES, get_category,
    filter_commands, load_completed, eval_opus,
    _save_summary, _print_summary,
)


def eval_ota(
    attacker: RobustWaveformAttacker,
    result,
    prompt: str = None,
    opus_bitrate: int = 128,
    snr_db: float = 25.0,
    low_hz: float = 200.0,
    high_hz: float = 12000.0,
    gain_db: float = -3.0,
) -> dict:
    """
    Evaluate adversarial audio after full SoundCloud + airgap channel.
    Same eval pipeline as run_robust_benchmark.py for fair comparison.
    """
    import torchaudio
    import numpy as np

    adv_audio = result.adversarial_wav.squeeze().cpu().numpy()

    try:
        from demo_ota import apply_opus_compression
        audio_after_codec = apply_opus_compression(adv_audio, ENCODEC_SAMPLE_RATE, opus_bitrate)
        if len(audio_after_codec) > len(adv_audio):
            audio_after_codec = audio_after_codec[:len(adv_audio)]
        elif len(audio_after_codec) < len(adv_audio):
            audio_after_codec = np.pad(audio_after_codec, (0, len(adv_audio) - len(audio_after_codec)))

        ota_audio = simulate_empirical_ota_channel(
            audio_after_codec,
            sr=ENCODEC_SAMPLE_RATE,
            noise_scale=1.0,
        )
        channel_type = "soundcloud+airgap"
    except FileNotFoundError:
        ota_audio = simulate_ota_channel(
            adv_audio,
            sr=ENCODEC_SAMPLE_RATE,
            opus_bitrate=opus_bitrate,
            snr_db=snr_db,
            bandpass=True,
            low_hz=low_hz,
            high_hz=high_hz,
            gain_db=gain_db,
        )
        channel_type = "synthetic"

    ota_tensor = torch.FloatTensor(ota_audio).unsqueeze(0)
    from config import TARGET_SAMPLE_RATE
    ota_16k = torchaudio.functional.resample(
        ota_tensor.to(attacker.device), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
    )

    with torch.no_grad():
        output = attacker.target_model.generate(ota_16k, prompt=prompt)

    matches = result.target_text.lower() in output.lower()

    min_len = min(len(adv_audio), len(ota_audio))
    adv_t = torch.FloatTensor(adv_audio[:min_len])
    ota_t = torch.FloatTensor(ota_audio[:min_len])
    snr = attacker._compute_snr(adv_t, ota_t)

    return {
        "success": matches,
        "output": output,
        "snr_db": snr,
        "channel_type": channel_type,
        "channel_config": {
            "opus_bitrate": opus_bitrate,
            "snr_db": snr_db,
            "low_hz": low_hz,
            "high_hz": high_hz,
            "gain_db": gain_db,
        },
    }


def run_benchmark(args):
    """Run the robust waveform benchmark."""
    # Determine music files
    if args.music:
        music_load_keys = [m.strip() for m in args.music.split(",")]
        for m in music_load_keys:
            if resolve_music_path(m) is None:
                available = ", ".join(MUSIC_FILES.keys())
                raise ValueError(f"Unknown music '{m}'. Available: {available}")
        music_names = [_music_short_name(m) for m in music_load_keys]
    else:
        music_load_keys = list(MUSIC_FILES.keys())
        music_names = list(MUSIC_FILES.keys())

    # Determine commands
    commands = filter_commands(args.category, mode=args.prompt_mode, dataset=args.dataset)
    if args.commands:
        commands = {k: v for k, v in commands.items() if k in args.commands}

    opus_bitrates = [int(b) for b in args.opus_bitrates.split(",")]

    prompt_mode = args.prompt_mode
    if prompt_mode not in PROMPT_MODES:
        available = ", ".join(PROMPT_MODES.keys())
        raise ValueError(f"Unknown prompt mode '{prompt_mode}'. Available: {available}")
    prompt_text = PROMPT_MODES[prompt_mode]["prompt"]

    print(f"ROBUST WAVEFORM Benchmark")
    print(f"{'=' * 70}")
    print(f"{len(music_names)} music x {len(commands)} commands "
          f"= {len(music_names) * len(commands)} experiments")
    print(f"Dataset: {args.dataset} ({len(commands)} commands)")
    print(f"Prompt mode: {prompt_mode} -- \"{prompt_text}\"")
    print(f"Opus eval bitrates: {opus_bitrates} kbps")
    print(f"Attack steps: {args.steps}, eps: {args.eps}, alpha: {args.alpha}")
    channel_desc = "disabled" if args.no_channel else \
        f"ota (EoT x{args.n_eot_samples}, severity={args.channel_severity}, " \
        f"curriculum={args.channel_curriculum})"
    print(f"Channel: {channel_desc}")
    print(f"Warmup ratio: {args.warmup_ratio}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.resume:
        output_dir = args.resume
    else:
        music_tag = "_".join(music_names) if len(music_names) <= 3 else f"{len(music_names)}music"
        dataset_tag = f"_{args.dataset}" if args.dataset != "ours" else ""
        attack_mode = "untargeted" if args.untargeted else prompt_mode
        if args.no_channel:
            ch_tag = "noCh"
        else:
            ch_tag = f"ota_eot{args.n_eot_samples}"
        dir_name = (f"robust_waveform_{args.target_model}_{attack_mode}_"
                    f"eps{args.eps}_{music_tag}_{ch_tag}{dataset_tag}")
        waveform_airgap_dir = os.path.join(PROJECT_ROOT, "results_waveform_airgap")
        output_dir = os.path.join(waveform_airgap_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Subdirectories
    audio_dir = os.path.join(output_dir, "audio")
    model_subdir = args.target_model
    model_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load completed
    completed = load_completed(output_dir, model_subdir=model_subdir)
    if completed:
        print(f"Skipping {len(completed)} already-completed experiments")

    # Initialize attacker
    print("\nInitializing RobustWaveformAttacker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attacker = RobustWaveformAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        device=device,
        verbose=not args.quiet,
        no_channel=args.no_channel,
        warmup_ratio=args.warmup_ratio,
        n_eot_samples=args.n_eot_samples,
        channel_severity=args.channel_severity,
        channel_curriculum=args.channel_curriculum,
        empirical_data_dir=getattr(args, 'empirical_data_dir', None),
    )

    # Save experiment config
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        config_data = {
            "attack_type": "robust_waveform",
            "target_model": args.target_model,
            "dataset": args.dataset,
            "music_files": music_names,
            "commands": {k: v["command"] for k, v in commands.items()},
            "targets": {k: v["target"] for k, v in commands.items()},
            "opus_bitrates": opus_bitrates,
            "attack_steps": args.steps,
            "eps": args.eps,
            "alpha": args.alpha,
            "perceptual_weight": args.perceptual_weight,
            "prompt_mode": prompt_mode,
            "prompt_text": prompt_text,
            "untargeted": args.untargeted,
            "robust": True,
            "n_eot_samples": args.n_eot_samples,
            "channel_severity": args.channel_severity,
            "channel_curriculum": args.channel_curriculum,
            "warmup_ratio": args.warmup_ratio,
            "no_channel": args.no_channel,
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    # Run experiments
    all_results = []

    # Load existing results
    summary_path = os.path.join(model_dir, "summary.json")
    if os.path.isdir(model_dir):
        for entry in sorted(os.listdir(model_dir)):
            rec_path = os.path.join(model_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        all_results.append(json.load(f))
                except (json.JSONDecodeError, KeyError):
                    pass

    total = len(music_names) * len(commands)
    done = len(completed)

    for mi, music_name in enumerate(music_names):
        music_wav = load_music_by_name(music_load_keys[mi])

        for ci, (cmd_name, cmd_info) in enumerate(commands.items()):
            cmd_text = cmd_info["command"]
            target_text = cmd_info["target"]

            if (music_name, cmd_name) in completed:
                continue

            done += 1
            print(f"\n{'=' * 70}")
            print(f"[{done}/{total}] music={music_name}, cmd={cmd_name}")
            print(f"Command: \"{cmd_text}\"")
            print(f"Target:  \"{target_text}\"")
            print(f"{'=' * 70}")

            start_time = time.time()

            # Run robust waveform attack
            result = attacker.attack(
                music_wav,
                target_text=target_text,
                steps=args.steps,
                music_name=music_name,
                prompt=prompt_text,
                untargeted=args.untargeted,
            )

            # Opus evaluation (reuse from waveform attacker)
            opus_results = attacker.test_opus_robustness(
                result, bitrates=opus_bitrates, prompt=prompt_text
            )

            # OTA evaluation
            ota_result = eval_ota(
                attacker, result, prompt=prompt_text,
                opus_bitrate=args.ota_opus_bitrate,
                snr_db=args.ota_snr_db,
            )

            duration = time.time() - start_time

            # OTA output from attack (differentiable channel eval)
            attack_ota_output = getattr(result, 'ota_output', None)
            attack_ota_success = False
            if attack_ota_output and not args.untargeted:
                attack_ota_success = target_text.lower() in attack_ota_output.lower()

            # Build result record
            effective_target = result.target_text if args.untargeted else target_text
            record = {
                "music": music_name,
                "command_name": cmd_name,
                "command_text": cmd_text,
                "target_text": effective_target,
                "category": get_category(cmd_name),
                "attack_type": "robust_waveform",
                "direct": {
                    "success": target_text.lower() in result.adversarial_output.lower() if not args.untargeted else result.success,
                    "output": result.adversarial_output,
                    "snr_db": result.snr_db,
                    "latent_snr_db": 0.0,
                    "final_loss": result.final_loss,
                },
                "ota": {
                    "success": ota_result["success"],
                    "output": ota_result["output"],
                    "snr_db": ota_result["snr_db"],
                    "channel_type": ota_result.get("channel_type", "unknown"),
                },
                "opus": {},
                "prompt_mode": prompt_mode,
                "prompt_text": prompt_text,
                "untargeted": args.untargeted,
                "steps": result.steps_taken,
                "duration_s": duration,
                "original_output": result.original_output,
                "robust_config": {
                    "n_eot_samples": args.n_eot_samples,
                    "warmup_ratio": args.warmup_ratio,
                    "no_channel": args.no_channel,
                    "attack_ota_success": attack_ota_success,
                },
            }

            for bw, opus_r in opus_results.items():
                if args.untargeted:
                    opus_wer_vs_orig = compute_wer(result.original_output, opus_r["output"])
                    opus_success = opus_wer_vs_orig > 0.5
                else:
                    opus_success = opus_r["matches_target"]
                opus_entry = {
                    "success": opus_success,
                    "output": opus_r["output"],
                    "snr_db": opus_r["snr_db"],
                }
                if args.untargeted:
                    opus_entry["wer_vs_original"] = opus_wer_vs_orig
                if prompt_mode == "transcribe" and effective_target:
                    opus_entry["wer"] = compute_wer(effective_target, opus_r["output"])
                record["opus"][str(bw)] = opus_entry

            if args.untargeted:
                record["direct"]["wer_vs_original"] = compute_wer(
                    result.original_output, result.adversarial_output
                )
                ota_wer = compute_wer(result.original_output, ota_result["output"])
                record["ota"]["success"] = ota_wer > 0.5
                record["ota"]["wer_vs_original"] = ota_wer

            if prompt_mode == "transcribe" and effective_target:
                record["direct"]["wer"] = compute_wer(effective_target, result.adversarial_output)
                record["ota"]["wer"] = compute_wer(effective_target, ota_result["output"])

            all_results.append(record)

            # Save adversarial WAV
            exp_key = f"{music_name}__{cmd_name}"
            attacker.save_adversarial_wav(result, audio_dir, exp_key)

            # Save record.json
            exp_record_dir = os.path.join(model_dir, exp_key)
            os.makedirs(exp_record_dir, exist_ok=True)
            rec_path = os.path.join(exp_record_dir, "record.json")
            with open(rec_path, "w") as f:
                json.dump(record, f, indent=2)

            # Incremental summary
            _save_summary(summary_path, all_results, music_names, commands, opus_bitrates)
            save_results_summary(output_dir, all_results, total=total)

            # Quick progress line
            d_ok = "OK" if record["direct"]["success"] else "FAIL"
            ota_ok = "OK" if ota_result["success"] else "FAIL"
            opus_status = " | ".join(
                f"opus{bw}={'OK' if opus_r['matches_target'] else 'FAIL'}"
                for bw, opus_r in opus_results.items()
            )
            print(f"  Result: direct={d_ok} | {opus_status} | ota={ota_ok} | "
                  f"SNR={result.snr_db:.1f}dB | {duration:.1f}s")

    # Final summary
    _save_summary(summary_path, all_results, music_names, commands, opus_bitrates)
    _print_summary(all_results, music_names, commands, opus_bitrates)

    # OTA summary
    ota_results = [r for r in all_results if "ota" in r]
    if ota_results:
        ota_ok = sum(1 for r in ota_results if r["ota"]["success"])
        n = len(ota_results)
        print(f"\nOTA success rate: {ota_ok}/{n} ({100*ota_ok/n:.1f}%)")

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Robust waveform benchmark: waveform attacks with OTA channel hardening"
    )

    # Scope
    parser.add_argument("--music", type=str, default=None,
                        help=f"Comma-separated music carriers. Options: {', '.join(MUSIC_FILES.keys())}")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--commands", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    # Attack params
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=WAVEFORM_EPS)
    parser.add_argument("--alpha", type=float, default=WAVEFORM_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)

    # Evaluation
    parser.add_argument("--opus-bitrates", type=str,
                        default=",".join(str(b) for b in OPUS_EVAL_BITRATES))

    # Prompt mode
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)

    # Target model
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)

    # Dataset
    parser.add_argument("--dataset", type=str, default="ours", choices=DATASET_CHOICES)

    # Attack mode
    parser.add_argument("--untargeted", action="store_true")

    # Robust-specific
    parser.add_argument("--warmup-ratio", type=float, default=0.5)
    parser.add_argument("--no-channel", action="store_true")
    parser.add_argument("--n-eot-samples", type=int, default=2)
    parser.add_argument("--channel-severity", type=float, default=1.0)
    parser.add_argument("--channel-curriculum", action="store_true")
    parser.add_argument("--empirical-data-dir", type=str, default=None)

    # OTA eval
    parser.add_argument("--ota-opus-bitrate", type=int, default=128)
    parser.add_argument("--ota-snr-db", type=float, default=25.0)

    # Convenience
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.quick:
        if not args.music:
            args.music = DEFAULT_MUSIC
        args.steps = 150
        if args.dataset == "saycan":
            cmd_source = load_saycan_commands()
        else:
            cmd_source = AGENT_COMMANDS
        first_two = list(cmd_source.keys())[:2]
        args.commands = args.commands or first_two
        args.quiet = True

    run_benchmark(args)


if __name__ == "__main__":
    main()
