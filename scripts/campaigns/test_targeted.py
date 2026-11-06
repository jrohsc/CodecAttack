"""
Phase A: Targeted transcription attacks using music carriers.

Experiments:
1. Load music from /data/music/ as carrier
2. Encode to EnCodec latent space
3. Run latent attack targeting specific text outputs
4. Evaluate model output, SNR, and compression robustness

Usage:
    python test_targeted.py                          # Run all music + all targets
    python test_targeted.py --music jazz_1           # Single music carrier
    python test_targeted.py --target simple_factual  # Single target text
    python test_targeted.py --quick                  # Quick smoke test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

from config import (
    MUSIC_FILES, DEFAULT_MUSIC, TARGETED_TEXTS, RESULTS_DIR,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
)
from music_carrier import load_music_by_name
from latent_attack import LatentCodecAttacker


def run_smoke_test(attacker: LatentCodecAttacker):
    """
    Quick smoke test: verify pipeline works end-to-end without attack.

    1. Load music, encode, decode, transcribe
    2. Run 10 attack steps, verify loss decreases and gradients flow
    """
    print("\n" + "=" * 60)
    print("SMOKE TEST: Pipeline Verification")
    print("=" * 60)

    # Load music
    music_wav = load_music_by_name(DEFAULT_MUSIC)
    print(f"Loaded music: {DEFAULT_MUSIC}, shape: {music_wav.shape}")

    # Encode -> decode -> transcribe (no attack)
    z = attacker.encode_music(music_wav)
    with torch.no_grad():
        import torchaudio
        from config import ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        decoded = attacker.codec.decode_from_continuous(z)
        decoded_16k = torchaudio.functional.resample(
            decoded.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        output = attacker.target_model.generate(decoded_16k)
    print(f"Model output on clean music: {output}")

    # Quick gradient flow test (10 steps)
    print("\nGradient flow test (10 steps)...")
    result = attacker.attack(
        music_wav,
        target_text="The answer is 42",
        steps=10,
        check_every=0,
        music_name="smoke_test",
    )
    print(f"Loss decreased: {result.history['loss'][0]:.4f} -> {result.history['loss'][-1]:.4f}")
    print(f"Delta norm: {result.latent_perturbation.norm().item():.4f}")

    if result.history['loss'][-1] < result.history['loss'][0]:
        print("PASS: Loss is decreasing, gradients are flowing correctly")
    else:
        print("WARNING: Loss did not decrease. Check gradient flow.")

    return result


def run_single_experiment(
    attacker: LatentCodecAttacker,
    music_name: str,
    target_name: str,
    target_text: str,
    output_dir: str,
    steps: int = ATTACK_STEPS,
):
    """Run a single targeted attack experiment."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: music={music_name}, target={target_name}")
    print(f"Target text: '{target_text}'")
    print(f"{'=' * 60}")

    # Load music
    music_wav = load_music_by_name(music_name)
    print(f"Music shape: {music_wav.shape}")

    # Run attack
    result = attacker.attack(
        music_wav,
        target_text=target_text,
        steps=steps,
        music_name=music_name,
    )

    # Test robustness
    print("\n--- EnCodec Robustness ---")
    attacker.test_encodec_robustness(result)

    print("\n--- Opus Robustness ---")
    attacker.test_opus_robustness(result)

    # Save results
    exp_dir = os.path.join(output_dir, f"{music_name}_{target_name}")
    attacker.save_result(result, output_dir=exp_dir, prefix="targeted")

    return result


def run_all_experiments(args):
    """Run targeted attack experiments across music files and targets."""
    # Determine which music files to use
    if args.music:
        music_names = [args.music]
    else:
        music_names = list(MUSIC_FILES.keys())

    # Determine which targets to use
    if args.target:
        targets = {args.target: TARGETED_TEXTS[args.target]}
    else:
        targets = TARGETED_TEXTS

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, f"targeted_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize attacker
    print("Initializing attacker...")
    attacker = LatentCodecAttacker(
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run smoke test first
    if not args.no_smoke_test:
        run_smoke_test(attacker)

    # Run experiments
    all_results = []
    for music_name in music_names:
        for target_name, target_text in targets.items():
            result = run_single_experiment(
                attacker,
                music_name=music_name,
                target_name=target_name,
                target_text=target_text,
                output_dir=output_dir,
                steps=args.steps,
            )
            all_results.append({
                "music": music_name,
                "target_name": target_name,
                "target_text": target_text,
                **result.to_json(),
            })

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Music':<25} {'Target':<20} {'Success':<10} {'SNR (dB)':<10} {'Loss':<10}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['music']:<25} {r['target_name']:<20} "
            f"{'YES' if r['success'] else 'NO':<10} "
            f"{r['snr_db']:<10.2f} {r['final_loss']:<10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Phase A: Targeted transcription attacks")
    parser.add_argument("--music", type=str, default=None,
                        help=f"Music carrier name. Options: {', '.join(MUSIC_FILES.keys())}")
    parser.add_argument("--target", type=str, default=None,
                        help=f"Target name. Options: {', '.join(TARGETED_TEXTS.keys())}")
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS,
                        help=f"Attack steps (default: {ATTACK_STEPS})")
    parser.add_argument("--eps", type=float, default=LATENT_EPS,
                        help=f"Latent epsilon (default: {LATENT_EPS})")
    parser.add_argument("--alpha", type=float, default=LATENT_ALPHA,
                        help=f"Learning rate (default: {LATENT_ALPHA})")
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT,
                        help=f"Perceptual loss weight (default: {PERCEPTUAL_WEIGHT})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 music, 1 target, 150 steps")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Skip smoke test")
    args = parser.parse_args()

    if args.quick:
        args.music = args.music or DEFAULT_MUSIC
        args.target = args.target or "simple_factual"
        args.steps = 150
        args.no_smoke_test = True

    run_all_experiments(args)


if __name__ == "__main__":
    main()
