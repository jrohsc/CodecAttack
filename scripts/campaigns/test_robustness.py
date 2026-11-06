"""
Cross-codec robustness testing for latent-space adversarial attacks.

Tests how adversarial audio survives through:
1. EnCodec re-encoding at multiple bitrates (1.5, 3, 6, 12 kbps)
2. Opus compression at multiple bitrates (16, 32, 64, 128 kbps)

Metrics per codec/bitrate:
- matches_target: Does model output contain the target text?
- output: Full model output text
- snr_db: Audio quality after compression

Usage:
    python test_robustness.py --audio results/targeted_*/jazz_1_simple_factual/targeted_jazz_1_*_adversarial.wav \\
                              --target "The answer is 42"
    python test_robustness.py --run-attack --music jazz_1 --target-name simple_factual
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import torchaudio

from config import (
    MUSIC_FILES, DEFAULT_MUSIC, TARGETED_TEXTS, RESULTS_DIR,
    ENCODEC_TEST_BITRATES, OPUS_TEST_BITRATES,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
)
from music_carrier import load_music_by_name
from latent_attack import LatentCodecAttacker


def test_robustness_from_audio(
    attacker: LatentCodecAttacker,
    audio_path: str,
    target_text: str,
    output_dir: str,
):
    """
    Test robustness of a pre-generated adversarial audio file.

    Args:
        attacker: Initialized LatentCodecAttacker
        audio_path: Path to adversarial WAV file
        target_text: Target text to check against
        output_dir: Directory for results
    """
    print(f"\nLoading adversarial audio: {audio_path}")
    audio, sr = torchaudio.load(audio_path)
    if sr != ENCODEC_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, ENCODEC_SAMPLE_RATE)
    audio = audio.to(attacker.device)

    # First test without compression
    print("\n--- Baseline (no compression) ---")
    with torch.no_grad():
        audio_16k = torchaudio.functional.resample(audio, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        baseline_output = attacker.target_model.generate(audio_16k)
    print(f"Output: {baseline_output}")
    baseline_match = target_text.lower() in baseline_output.lower()
    print(f"Matches target: {baseline_match}")

    results = {
        "audio_path": audio_path,
        "target_text": target_text,
        "baseline": {
            "output": baseline_output,
            "matches_target": baseline_match,
        },
        "encodec": {},
        "opus": {},
    }

    # EnCodec robustness
    print("\n--- EnCodec Re-encoding Robustness ---")
    original_bw = attacker.codec._bandwidth
    for bw in ENCODEC_TEST_BITRATES:
        attacker.codec.set_bandwidth(bw)
        with torch.no_grad():
            if audio.dim() == 2:
                audio_3d = audio.unsqueeze(0)
            else:
                audio_3d = audio
            frames = attacker.codec.model.encode(audio_3d)
            reencoded = attacker.codec.model.decode(frames)
            reencoded_16k = torchaudio.functional.resample(
                reencoded.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            output = attacker.target_model.generate(reencoded_16k)

        matches = target_text.lower() in output.lower()
        snr = LatentCodecAttacker._compute_snr(
            audio.squeeze(), reencoded.squeeze().detach()
        )

        results["encodec"][str(bw)] = {
            "output": output,
            "matches_target": matches,
            "snr_db": snr,
        }
        print(f"  {bw:5.1f} kbps | Match: {matches:<5} | SNR: {snr:6.2f} dB | Output: {output[:80]}")

    attacker.codec.set_bandwidth(original_bw)

    # Opus robustness
    print("\n--- Opus Compression Robustness ---")
    import subprocess
    import tempfile

    for bw in OPUS_TEST_BITRATES:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "input.wav")
            opus_path = os.path.join(tmpdir, "compressed.opus")
            out_path = os.path.join(tmpdir, "output.wav")

            import soundfile as sf
            adv_np = audio.cpu().squeeze().numpy()
            sf.write(wav_path, adv_np, ENCODEC_SAMPLE_RATE)

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path, "-c:a", "opus",
                     "-strict", "-2",
                     "-b:a", f"{bw}k", opus_path],
                    capture_output=True, check=True
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", opus_path, "-ar", str(ENCODEC_SAMPLE_RATE), out_path],
                    capture_output=True, check=True
                )

                decoded_np, decode_sr = sf.read(out_path)
                decoded_audio = torch.FloatTensor(decoded_np).unsqueeze(0)
                if decode_sr != ENCODEC_SAMPLE_RATE:
                    decoded_audio = torchaudio.functional.resample(decoded_audio, decode_sr, ENCODEC_SAMPLE_RATE)

                with torch.no_grad():
                    decoded_16k = torchaudio.functional.resample(
                        decoded_audio.to(attacker.device), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output = attacker.target_model.generate(decoded_16k)

                min_len = min(audio.squeeze().shape[-1], decoded_audio.squeeze().shape[-1])
                snr = LatentCodecAttacker._compute_snr(
                    audio.squeeze().cpu()[..., :min_len],
                    decoded_audio.squeeze()[..., :min_len]
                )

                matches = target_text.lower() in output.lower()
                results["opus"][str(bw)] = {
                    "output": output,
                    "matches_target": matches,
                    "snr_db": snr,
                }
                print(f"  {bw:4d} kbps | Match: {matches:<5} | SNR: {snr:6.2f} dB | Output: {output[:80]}")

            except subprocess.CalledProcessError as e:
                print(f"  {bw:4d} kbps | ERROR: ffmpeg failed ({e})")
                results["opus"][str(bw)] = {"output": "", "matches_target": False, "snr_db": 0.0, "error": str(e)}

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"robustness_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"Target: '{target_text}'")
    print(f"Baseline match: {baseline_match}")
    print()

    print("EnCodec:")
    for bw in ENCODEC_TEST_BITRATES:
        r = results["encodec"][str(bw)]
        status = "SURVIVES" if r["matches_target"] else "FAILS"
        print(f"  {bw:5.1f} kbps -> {status} (SNR: {r['snr_db']:.2f} dB)")

    print("Opus:")
    for bw in OPUS_TEST_BITRATES:
        r = results["opus"][str(bw)]
        if "error" in r:
            print(f"  {bw:4d}  kbps -> ERROR")
        else:
            status = "SURVIVES" if r["matches_target"] else "FAILS"
            print(f"  {bw:4d}  kbps -> {status} (SNR: {r['snr_db']:.2f} dB)")

    return results


def run_attack_and_test(args):
    """Run attack first, then test robustness."""
    print("Initializing attacker...")
    attacker = LatentCodecAttacker(
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run attack
    music_wav = load_music_by_name(args.music)
    target_text = TARGETED_TEXTS[args.target_name]

    result = attacker.attack(
        music_wav,
        target_text=target_text,
        steps=args.steps,
        music_name=args.music,
    )

    # Save attack result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, f"robustness_{args.music}_{args.target_name}_{timestamp}")
    attacker.save_result(result, output_dir=output_dir)

    # Test robustness
    adv_path = os.path.join(output_dir, os.listdir(output_dir)[0])
    # Find the adversarial wav file
    for f in os.listdir(output_dir):
        if f.endswith("_adversarial.wav"):
            adv_path = os.path.join(output_dir, f)
            break

    test_robustness_from_audio(attacker, adv_path, target_text, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Cross-codec robustness testing")

    # Mode 1: Test existing adversarial audio
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to pre-generated adversarial WAV file")
    parser.add_argument("--target", type=str, default=None,
                        help="Target text to check against")

    # Mode 2: Run attack then test
    parser.add_argument("--run-attack", action="store_true",
                        help="Run attack first, then test robustness")
    parser.add_argument("--music", type=str, default=DEFAULT_MUSIC,
                        help=f"Music carrier name (default: {DEFAULT_MUSIC})")
    parser.add_argument("--target-name", type=str, default="simple_factual",
                        help=f"Target name from config (default: simple_factual)")

    # Attack params
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=LATENT_EPS)
    parser.add_argument("--alpha", type=float, default=LATENT_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)

    args = parser.parse_args()

    if args.audio:
        # Mode 1: Test existing audio
        if not args.target:
            print("ERROR: --target is required when using --audio")
            sys.exit(1)

        attacker = LatentCodecAttacker(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        output_dir = os.path.join(RESULTS_DIR, f"robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        test_robustness_from_audio(attacker, args.audio, args.target, output_dir)

    elif args.run_attack:
        # Mode 2: Run attack + test
        run_attack_and_test(args)

    else:
        parser.print_help()
        print("\nUse --audio <path> --target <text> to test existing audio")
        print("Use --run-attack --music <name> --target-name <name> to run attack + test")


if __name__ == "__main__":
    main()
