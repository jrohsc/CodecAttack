"""
Waveform-domain baseline benchmark: direct waveform perturbation attacks
across music carriers and physical agent commands.

This is the baseline comparison for latent-space attacks. It demonstrates
that waveform-domain perturbations — the standard adversarial audio approach —
do NOT survive Opus compression, even when they succeed on direct evaluation.

For each (music, command) pair:
  1. Run waveform-domain attack via WaveformAttacker
  2. Evaluate in 3 modes:
     a) Direct  - adversarial WAV fed straight to target model
     b) Opus 64 kbps  - compressed via Opus then evaluated
     c) Opus 128 kbps - compressed via Opus then evaluated

Usage:
    python run_waveform_baseline.py                         # Full run
    python run_waveform_baseline.py --music jazz_1          # Single music carrier
    python run_waveform_baseline.py --quick --eps 0.01      # Quick test
    python run_waveform_baseline.py --eps 0.005             # Sweep eps
    python run_waveform_baseline.py --resume results/waveform_baseline_...  # Resume
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
    compute_wer,
)
from music_carrier import load_music_by_name, resolve_music_path
from waveform_attack import WaveformAttacker
from run_benchmark import (
    filter_commands, load_completed, get_category,
    _save_summary, _print_summary, _music_short_name,
)


def run_benchmark(args):
    """Run the waveform baseline benchmark."""
    # Determine music files
    if args.music:
        music_load_keys = [m.strip() for m in args.music.split(",")]
        for m in music_load_keys:
            if resolve_music_path(m) is None:
                available = ", ".join(MUSIC_FILES.keys())
                raise ValueError(f"Unknown music '{m}'. Not a known name ({available}) and not a valid file path.")
        music_names = [_music_short_name(m) for m in music_load_keys]
    else:
        music_load_keys = list(MUSIC_FILES.keys())
        music_names = list(MUSIC_FILES.keys())

    # Determine commands
    commands = filter_commands(args.category, mode=args.prompt_mode, dataset=args.dataset)
    if args.commands:
        commands = {k: v for k, v in commands.items() if k in args.commands}

    opus_bitrates = [int(b) for b in args.opus_bitrates.split(",")]

    # Resolve prompt mode
    prompt_mode = args.prompt_mode
    if prompt_mode not in PROMPT_MODES:
        available = ", ".join(PROMPT_MODES.keys())
        raise ValueError(f"Unknown prompt mode '{prompt_mode}'. Available: {available}")
    prompt_text = PROMPT_MODES[prompt_mode]["prompt"]

    print(f"WAVEFORM BASELINE Benchmark")
    print(f"{'=' * 70}")
    print(f"{len(music_names)} music files x {len(commands)} commands "
          f"= {len(music_names) * len(commands)} experiments")
    print(f"Dataset: {args.dataset} ({len(commands)} commands)")
    print(f"Attack mode: {'UNTARGETED (disrupt)' if args.untargeted else 'targeted'}")
    print(f"Prompt mode: {prompt_mode} — \"{prompt_text}\"")
    print(f"Opus evaluation bitrates: {opus_bitrates} kbps")
    print(f"Attack steps: {args.steps}, eps: {args.eps}, alpha: {args.alpha}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.resume:
        output_dir = args.resume
    else:
        music_tag = "_".join(music_names) if len(music_names) <= 3 else f"{len(music_names)}music"
        dataset_tag = f"_{args.dataset}" if args.dataset != "ours" else ""
        attack_mode = "untargeted" if args.untargeted else prompt_mode
        waveform_results_dir = os.path.join(PROJECT_ROOT, "results_waveform")
        dir_name = f"waveform_baseline_{args.target_model}_{attack_mode}_eps{args.eps}_{music_tag}{dataset_tag}"
        output_dir = os.path.join(waveform_results_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Subdirectories
    audio_dir = os.path.join(output_dir, "audio")
    model_subdir = args.target_model
    model_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load completed pairs
    completed = load_completed(output_dir, model_subdir=model_subdir)
    if completed:
        print(f"Skipping {len(completed)} already-completed experiments")

    # Initialize attacker
    print("\nInitializing WaveformAttacker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attacker = WaveformAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        device=device,
        verbose=not args.quiet,
    )

    # Save experiment config
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        config_data = {
            "attack_type": "waveform_baseline",
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

            # Run attack
            result = attacker.attack(
                music_wav,
                target_text=target_text,
                steps=args.steps,
                music_name=music_name,
                prompt=prompt_text,
                untargeted=args.untargeted,
            )

            # Opus evaluation
            opus_results = attacker.test_opus_robustness(result, bitrates=opus_bitrates, prompt=prompt_text)

            duration = time.time() - start_time

            # Build result record (same format as run_benchmark.py)
            effective_target = result.target_text if args.untargeted else target_text
            record = {
                "music": music_name,
                "command_name": cmd_name,
                "command_text": cmd_text,
                "target_text": effective_target,
                "category": get_category(cmd_name),
                "attack_type": "waveform_baseline",
                "direct": {
                    "success": result.success,
                    "output": result.adversarial_output,
                    "snr_db": result.snr_db,
                    "latent_snr_db": 0.0,
                    "final_loss": result.final_loss,
                },
                "opus": {},
                "prompt_mode": prompt_mode,
                "prompt_text": prompt_text,
                "untargeted": args.untargeted,
                "steps": result.steps_taken,
                "duration_s": duration,
                "original_output": result.original_output,
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

            # Compute WER
            if args.untargeted:
                record["direct"]["wer_vs_original"] = compute_wer(
                    result.original_output, result.adversarial_output
                )
            if prompt_mode == "transcribe" and effective_target:
                record["direct"]["wer"] = compute_wer(effective_target, result.adversarial_output)

            all_results.append(record)

            # Save audio
            exp_key = f"{music_name}__{cmd_name}"
            attacker.save_adversarial_wav(result, audio_dir, exp_key)

            # Save record.json
            exp_record_dir = os.path.join(model_dir, exp_key)
            os.makedirs(exp_record_dir, exist_ok=True)
            rec_path = os.path.join(exp_record_dir, "record.json")
            with open(rec_path, "w") as f:
                json.dump(record, f, indent=2)

            # Incrementally save summary
            _save_summary(summary_path, all_results, music_names, commands, opus_bitrates)

            # Quick progress line
            d_ok = "OK" if result.success else "FAIL"
            opus_status = " | ".join(
                f"opus{bw}={'OK' if opus_r['matches_target'] else 'FAIL'}"
                for bw, opus_r in opus_results.items()
            )
            print(f"  Result: direct={d_ok} | {opus_status} | SNR={result.snr_db:.1f}dB | {duration:.1f}s")

    # Final summary
    _save_summary(summary_path, all_results, music_names, commands, opus_bitrates)
    _print_summary(all_results, music_names, commands, opus_bitrates)

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Waveform-domain baseline: direct waveform perturbation attacks"
    )

    # Scope
    parser.add_argument("--music", type=str, default=None,
                        help=f"Comma-separated music carriers. Options: {', '.join(MUSIC_FILES.keys())}")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter commands by category")
    parser.add_argument("--commands", nargs="+", default=None,
                        help="Explicit command names to run")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Shared output dir (for parallel runs)")

    # Attack params
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=WAVEFORM_EPS)
    parser.add_argument("--alpha", type=float, default=WAVEFORM_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)

    # Evaluation
    parser.add_argument("--opus-bitrates", type=str,
                        default=",".join(str(b) for b in OPUS_EVAL_BITRATES),
                        help="Comma-separated Opus bitrates in kbps (default: 64,128)")

    # Prompt mode
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE,
                        help=f"Prompt mode: {', '.join(PROMPT_MODES.keys())} (default: {DEFAULT_PROMPT_MODE})")

    # Target model
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL,
                        help=f"Target model to attack (default: {TARGET_MODEL})")

    # Dataset
    parser.add_argument("--dataset", type=str, default="ours",
                        choices=DATASET_CHOICES,
                        help="Command dataset: 'ours' or 'saycan'")

    # Attack mode
    parser.add_argument("--untargeted", action="store_true",
                        help="Untargeted attack: disrupt model output instead of forcing specific text")

    # Convenience
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 music, 2 commands, 150 steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous benchmark output dir to resume")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce per-step logging")

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
