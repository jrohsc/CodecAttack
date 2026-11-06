"""
Watermark defense experiments: evaluate AudioSeal as a defense against
latent-space adversarial attacks on Audio LLMs.

Experiments:
  (A) codec_roundtrip  — Does watermark survive clean EnCodec encode→decode?
  (B) attack_detection — Does adversarial perturbation destroy the watermark?
  (C) opus_robustness  — Does watermark survive Opus compression?
  (D) threshold_analysis — ROC curve for detection threshold

Usage:
    python run_watermark_experiments.py --experiment codec_roundtrip
    python run_watermark_experiments.py --experiment attack_detection --music jazz_1 --num_commands 5
    python run_watermark_experiments.py --experiment all
"""

import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
from typing import Dict, List, Optional

import torch
import numpy as np

from config import (
    MUSIC_FILES, ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    ENCODEC_TEST_BITRATES, OPUS_TEST_BITRATES,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
    WAVEFORM_EPS, WAVEFORM_ALPHA,
    AGENT_COMMANDS, RESULTS_DIR, TARGET_MODEL,
    WATERMARK_NBITS, WATERMARK_DETECTION_THRESHOLD, WATERMARK_THRESHOLDS_SWEEP,
    DATASET_CHOICES, load_saycan_commands,
)
from music_carrier import load_music_by_name
from watermark_defense import AudioSealDefense, WatermarkResult

RESULTS_WATERMARK_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "results_watermark")


def save_results(results: dict, experiment_name: str, output_dir: str = None):
    """Save experiment results to JSON."""
    if output_dir is None:
        output_dir = os.path.join(RESULTS_WATERMARK_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


# ============================================================================
# (A) Codec round-trip: watermark survival through clean EnCodec
# ============================================================================

def experiment_codec_roundtrip(
    music_names: List[str],
    bandwidths: List[float] = None,
    device: str = "cuda",
):
    """Test if watermark survives clean EnCodec encode→decode at various bandwidths."""
    if bandwidths is None:
        bandwidths = ENCODEC_TEST_BITRATES

    print("=" * 60)
    print("Experiment A: Codec Round-Trip Watermark Survival")
    print("=" * 60)

    defense = AudioSealDefense(
        nbits=WATERMARK_NBITS,
        detection_threshold=WATERMARK_DETECTION_THRESHOLD,
        device=device,
    )

    # Load EnCodec
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                     "external", "whisper-inject-v2-fork"))
    from attacks.latent_codec import EnCodecWrapper

    results = {"experiment": "codec_roundtrip", "bandwidths": bandwidths, "results": []}

    for music_name in music_names:
        print(f"\n--- Music: {music_name} ---")
        music_wav = load_music_by_name(music_name)  # [1, 1, T] at 24kHz

        # Watermark the audio
        watermarked_24k, message = defense.watermark_for_codec(music_wav.squeeze(0).to(device))
        # watermarked_24k: [1, 1, T_24k]

        # Verify watermark before codec (sanity check)
        pre_codec = defense.verify_after_codec(watermarked_24k, message)
        print(f"  Pre-codec: score={pre_codec.detection_score:.4f}, "
              f"msg_acc={pre_codec.message_accuracy:.4f}, detected={pre_codec.is_watermarked}")

        for bw in bandwidths:
            codec = EnCodecWrapper(bandwidth=bw, device=device)

            # Encode → decode
            with torch.no_grad():
                wm_for_codec = watermarked_24k.to(device)
                if wm_for_codec.dim() == 2:
                    wm_for_codec = wm_for_codec.unsqueeze(0)
                z = codec.encode_to_continuous(wm_for_codec)
                decoded = codec.decode_from_continuous(z)

            # Verify watermark after codec
            post_codec = defense.verify_after_codec(decoded, message)

            entry = {
                "music": music_name,
                "bandwidth_kbps": bw,
                "pre_codec": pre_codec.to_json(),
                "post_codec": post_codec.to_json(),
            }
            results["results"].append(entry)

            print(f"  BW={bw:5.1f} kbps: score={post_codec.detection_score:.4f}, "
                  f"msg_acc={post_codec.message_accuracy:.4f}, detected={post_codec.is_watermarked}")

            del codec
            torch.cuda.empty_cache()

    # Summary
    total = len(results["results"])
    survived = sum(1 for r in results["results"] if r["post_codec"]["is_watermarked"])
    results["summary"] = {
        "total_tests": total,
        "watermark_survived": survived,
        "survival_rate": survived / total if total > 0 else 0,
    }
    print(f"\nSummary: {survived}/{total} watermarks survived codec round-trip "
          f"({results['summary']['survival_rate']:.1%})")

    save_results(results, "codec_roundtrip")
    return results


# ============================================================================
# (B) Attack detection: does adversarial perturbation destroy watermark?
# ============================================================================

def experiment_attack_detection(
    music_names: List[str],
    num_commands: int = 5,
    eps: float = LATENT_EPS,
    steps: int = ATTACK_STEPS,
    device: str = "cuda",
    commands: Dict = None,
    target_model: str = TARGET_MODEL,
):
    """Test if adversarial attacks destroy the watermark."""
    if commands is None:
        commands = AGENT_COMMANDS

    print("=" * 60)
    print("Experiment B: Attack Detection via Watermark")
    print(f"  Target model: {target_model}")
    print(f"  Dataset: {len(commands)} commands, using {num_commands}")
    print("=" * 60)

    defense = AudioSealDefense(
        nbits=WATERMARK_NBITS,
        detection_threshold=WATERMARK_DETECTION_THRESHOLD,
        device=device,
    )

    from latent_attack import LatentCodecAttacker

    attacker = LatentCodecAttacker(
        target_model=target_model,
        encodec_bandwidth=ENCODEC_BANDWIDTH,
        eps=eps,
        alpha=LATENT_ALPHA,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        device=device,
        verbose=True,
    )

    # Select a subset of commands
    cmd_items = list(commands.items())[:num_commands]

    results = {
        "experiment": "attack_detection",
        "eps": eps,
        "steps": steps,
        "results": [],
    }

    for music_name in music_names:
        print(f"\n{'='*40}")
        print(f"Music: {music_name}")
        print(f"{'='*40}")

        music_wav = load_music_by_name(music_name)  # [1, 1, T] at 24kHz

        # Step 1: Watermark the clean audio
        watermarked_24k, message = defense.watermark_for_codec(
            music_wav.squeeze(0).to(device)
        )
        # [1, 1, T]

        # Verify clean watermarked audio
        clean_result = defense.verify_after_codec(watermarked_24k, message)
        print(f"Clean watermarked: score={clean_result.detection_score:.4f}, "
              f"detected={clean_result.is_watermarked}")

        # Also verify after clean codec round-trip (baseline)
        with torch.no_grad():
            z_clean = attacker.codec.encode_to_continuous(watermarked_24k.to(device))
            decoded_clean = attacker.codec.decode_from_continuous(z_clean)
        clean_codec_result = defense.verify_after_codec(decoded_clean, message)
        print(f"Clean after codec: score={clean_codec_result.detection_score:.4f}, "
              f"detected={clean_codec_result.is_watermarked}")

        for cmd_name, cmd_info in cmd_items:
            target_text = cmd_info.get("transcribe", cmd_info.get("command", ""))
            print(f"\n  Command: {cmd_name} -> '{target_text}'")

            # Step 2: Attack the watermarked audio
            attack_result = attacker.attack(
                watermarked_24k,  # Attack the watermarked version
                target_text=target_text,
                steps=steps,
                music_name=music_name,
            )

            # Step 3: Check watermark on adversarial audio
            adv_wav = attack_result.adversarial_wav.to(device)
            if adv_wav.dim() == 2:
                adv_wav = adv_wav.unsqueeze(0)
            adv_result = defense.verify_after_codec(adv_wav, message)

            entry = {
                "music": music_name,
                "command": cmd_name,
                "target_text": target_text,
                "attack_success": attack_result.success,
                "attack_output": attack_result.adversarial_output,
                "clean_watermark": clean_codec_result.to_json(),
                "adversarial_watermark": adv_result.to_json(),
                "watermark_destroyed": not adv_result.is_watermarked,
                "detection_score_drop": clean_codec_result.detection_score - adv_result.detection_score,
            }
            results["results"].append(entry)

            status = "DESTROYED" if not adv_result.is_watermarked else "SURVIVED"
            print(f"    Attack success: {attack_result.success}")
            print(f"    Watermark: {status} (score={adv_result.detection_score:.4f})")
            print(f"    Score drop: {entry['detection_score_drop']:.4f}")

    # Summary
    total = len(results["results"])
    successful_attacks = [r for r in results["results"] if r["attack_success"]]
    detected_attacks = [r for r in successful_attacks if r["watermark_destroyed"]]

    results["summary"] = {
        "total_pairs": total,
        "successful_attacks": len(successful_attacks),
        "attack_success_rate": len(successful_attacks) / total if total > 0 else 0,
        "watermark_destroyed_on_successful": len(detected_attacks),
        "defense_rate": len(detected_attacks) / len(successful_attacks) if successful_attacks else 0,
        "avg_clean_score": np.mean([r["clean_watermark"]["detection_score"] for r in results["results"]]),
        "avg_adversarial_score": np.mean([r["adversarial_watermark"]["detection_score"] for r in results["results"]]),
    }

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total pairs: {total}")
    print(f"  Successful attacks: {len(successful_attacks)}/{total}")
    print(f"  Defense rate (watermark destroyed on successful attacks): "
          f"{results['summary']['defense_rate']:.1%}")
    print(f"  Avg clean score: {results['summary']['avg_clean_score']:.4f}")
    print(f"  Avg adversarial score: {results['summary']['avg_adversarial_score']:.4f}")

    music_tag = "_".join(sorted(set(r["music"] for r in results["results"])))
    subdir = f"attack_detection/eps{eps}_{music_tag}"
    if target_model != TARGET_MODEL:
        subdir = f"attack_detection/{target_model}/eps{eps}_{music_tag}"
    save_results(results, subdir)
    return results


# ============================================================================
# (C) Opus robustness: watermark survival through Opus compression
# ============================================================================

def experiment_opus_robustness(
    music_names: List[str],
    opus_bitrates: List[int] = None,
    device: str = "cuda",
):
    """Test if watermark survives Opus compression at various bitrates."""
    if opus_bitrates is None:
        opus_bitrates = OPUS_TEST_BITRATES

    print("=" * 60)
    print("Experiment C: Opus Robustness")
    print("=" * 60)

    defense = AudioSealDefense(
        nbits=WATERMARK_NBITS,
        detection_threshold=WATERMARK_DETECTION_THRESHOLD,
        device=device,
    )

    # Find ffmpeg
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("ERROR: ffmpeg not found. Cannot run Opus robustness test.")
        return None

    results = {"experiment": "opus_robustness", "bitrates": opus_bitrates, "results": []}

    for music_name in music_names:
        print(f"\n--- Music: {music_name} ---")
        music_wav = load_music_by_name(music_name)  # [1, 1, T] at 24kHz

        # Watermark
        watermarked_24k, message = defense.watermark_for_codec(
            music_wav.squeeze(0).to(device)
        )

        # Pre-Opus baseline
        pre_opus = defense.verify_after_codec(watermarked_24k, message)
        print(f"  Pre-Opus: score={pre_opus.detection_score:.4f}, detected={pre_opus.is_watermarked}")

        for bw in opus_bitrates:
            # Opus encode/decode via ffmpeg
            import soundfile as sf

            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = os.path.join(tmpdir, "input.wav")
                opus_path = os.path.join(tmpdir, "compressed.opus")
                out_path = os.path.join(tmpdir, "output.wav")

                # Save watermarked audio
                wm_np = watermarked_24k.squeeze().cpu().numpy()
                sf.write(wav_path, wm_np, ENCODEC_SAMPLE_RATE)

                # Encode to Opus
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", wav_path, "-c:a", "opus",
                     "-strict", "-2", "-b:a", f"{bw}k", opus_path],
                    capture_output=True, check=True,
                )

                # Decode from Opus
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", opus_path, "-ar", str(ENCODEC_SAMPLE_RATE), out_path],
                    capture_output=True, check=True,
                )

                decoded_np, sr = sf.read(out_path)
                decoded_audio = torch.FloatTensor(decoded_np).unsqueeze(0).unsqueeze(0).to(device)

            # Verify watermark
            post_opus = defense.verify_after_codec(decoded_audio, message)

            entry = {
                "music": music_name,
                "opus_bitrate_kbps": bw,
                "pre_opus": pre_opus.to_json(),
                "post_opus": post_opus.to_json(),
            }
            results["results"].append(entry)

            print(f"  Opus {bw:4d} kbps: score={post_opus.detection_score:.4f}, "
                  f"msg_acc={post_opus.message_accuracy:.4f}, detected={post_opus.is_watermarked}")

    # Summary
    total = len(results["results"])
    survived = sum(1 for r in results["results"] if r["post_opus"]["is_watermarked"])
    results["summary"] = {
        "total_tests": total,
        "watermark_survived": survived,
        "survival_rate": survived / total if total > 0 else 0,
    }
    print(f"\nSummary: {survived}/{total} watermarks survived Opus compression "
          f"({results['summary']['survival_rate']:.1%})")

    save_results(results, "opus_robustness")
    return results


# ============================================================================
# (D) Threshold analysis: ROC curve data
# ============================================================================

def experiment_threshold_analysis(
    music_names: List[str],
    num_commands: int = 3,
    eps: float = LATENT_EPS,
    steps: int = ATTACK_STEPS,
    thresholds: List[float] = None,
    device: str = "cuda",
    commands: Dict = None,
    target_model: str = TARGET_MODEL,
):
    """Sweep detection thresholds to produce ROC curve data."""
    if commands is None:
        commands = AGENT_COMMANDS
    if thresholds is None:
        thresholds = WATERMARK_THRESHOLDS_SWEEP

    print("=" * 60)
    print("Experiment D: Threshold Analysis (ROC Data)")
    print(f"  Target model: {target_model}")
    print(f"  Dataset: {len(commands)} commands, using {num_commands}")
    print("=" * 60)

    defense = AudioSealDefense(
        nbits=WATERMARK_NBITS,
        detection_threshold=WATERMARK_DETECTION_THRESHOLD,
        device=device,
    )

    from latent_attack import LatentCodecAttacker

    attacker = LatentCodecAttacker(
        target_model=target_model,
        encodec_bandwidth=ENCODEC_BANDWIDTH,
        eps=eps,
        alpha=LATENT_ALPHA,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        device=device,
        verbose=False,
    )

    cmd_items = list(commands.items())[:num_commands]

    # Collect raw detection scores for clean and adversarial audio
    clean_scores = []
    adversarial_scores = []

    for music_name in music_names:
        print(f"\nMusic: {music_name}")
        music_wav = load_music_by_name(music_name)

        # Watermark
        watermarked_24k, message = defense.watermark_for_codec(
            music_wav.squeeze(0).to(device)
        )

        # Clean codec round-trip score
        with torch.no_grad():
            z_clean = attacker.codec.encode_to_continuous(watermarked_24k.to(device))
            decoded_clean = attacker.codec.decode_from_continuous(z_clean)
        clean_result = defense.verify_after_codec(decoded_clean, message)
        clean_scores.append(clean_result.detection_score)
        print(f"  Clean codec score: {clean_result.detection_score:.4f}")

        for cmd_name, cmd_info in cmd_items:
            target_text = cmd_info.get("transcribe", cmd_info.get("command", ""))
            print(f"  Attacking with: {cmd_name}...", end=" ", flush=True)

            attack_result = attacker.attack(
                watermarked_24k,
                target_text=target_text,
                steps=steps,
                music_name=music_name,
            )

            adv_wav = attack_result.adversarial_wav.to(device)
            if adv_wav.dim() == 2:
                adv_wav = adv_wav.unsqueeze(0)
            adv_result = defense.verify_after_codec(adv_wav, message)
            adversarial_scores.append(adv_result.detection_score)
            print(f"score={adv_result.detection_score:.4f}, attack_ok={attack_result.success}")

    # Compute ROC data
    roc_data = []
    for thresh in thresholds:
        tpr = sum(1 for s in clean_scores if s >= thresh) / len(clean_scores) if clean_scores else 0
        fpr = sum(1 for s in adversarial_scores if s >= thresh) / len(adversarial_scores) if adversarial_scores else 0
        roc_data.append({
            "threshold": thresh,
            "tpr": tpr,  # True positive rate: clean audio correctly detected as watermarked
            "fpr": fpr,  # False positive rate: adversarial audio incorrectly detected as watermarked
        })
        print(f"  Threshold {thresh:.1f}: TPR={tpr:.3f}, FPR={fpr:.3f}")

    # Find optimal threshold (maximize TPR - FPR)
    best = max(roc_data, key=lambda r: r["tpr"] - r["fpr"])

    results = {
        "experiment": "threshold_analysis",
        "clean_scores": clean_scores,
        "adversarial_scores": adversarial_scores,
        "roc_data": roc_data,
        "optimal_threshold": best["threshold"],
        "optimal_tpr": best["tpr"],
        "optimal_fpr": best["fpr"],
    }

    print(f"\nOptimal threshold: {best['threshold']:.1f} (TPR={best['tpr']:.3f}, FPR={best['fpr']:.3f})")

    save_results(results, "threshold_analysis")
    return results


# ============================================================================
# (E) Waveform baseline attack detection: does waveform perturbation destroy watermark?
# ============================================================================

def experiment_waveform_attack_detection(
    music_names: List[str],
    num_commands: int = 5,
    eps: float = WAVEFORM_EPS,
    steps: int = ATTACK_STEPS,
    device: str = "cuda",
    commands: Dict = None,
    target_model: str = TARGET_MODEL,
):
    """Test if waveform-domain attacks destroy the watermark (baseline comparison)."""
    if commands is None:
        commands = AGENT_COMMANDS

    print("=" * 60)
    print("Experiment E: Waveform Baseline Attack Detection via Watermark")
    print(f"  Target model: {target_model}")
    print(f"  eps={eps} (waveform L-inf), steps={steps}")
    print(f"  Dataset: {len(commands)} commands, using {num_commands}")
    print("=" * 60)

    defense = AudioSealDefense(
        nbits=WATERMARK_NBITS,
        detection_threshold=WATERMARK_DETECTION_THRESHOLD,
        device=device,
    )

    from waveform_attack import WaveformAttacker

    attacker = WaveformAttacker(
        target_model=target_model,
        eps=eps,
        alpha=WAVEFORM_ALPHA,
        perceptual_weight=PERCEPTUAL_WEIGHT,
        device=device,
        verbose=True,
    )

    cmd_items = list(commands.items())[:num_commands]

    results = {
        "experiment": "waveform_attack_detection",
        "attack_type": "waveform_baseline",
        "eps": eps,
        "steps": steps,
        "results": [],
    }

    for music_name in music_names:
        print(f"\n{'='*40}")
        print(f"Music: {music_name}")
        print(f"{'='*40}")

        music_wav = load_music_by_name(music_name)  # [1, 1, T] at 24kHz

        # Step 1: Watermark the clean audio
        watermarked_24k, message = defense.watermark_for_codec(
            music_wav.squeeze(0).to(device)
        )

        # Verify clean watermarked audio
        clean_result = defense.verify_after_codec(watermarked_24k, message)
        print(f"Clean watermarked: score={clean_result.detection_score:.4f}, "
              f"detected={clean_result.is_watermarked}")

        for cmd_name, cmd_info in cmd_items:
            target_text = cmd_info.get("transcribe", cmd_info.get("command", ""))
            print(f"\n  Command: {cmd_name} -> '{target_text}'")

            # Step 2: Waveform attack on the watermarked audio
            attack_result = attacker.attack(
                watermarked_24k,  # [1, 1, T] at 24kHz
                target_text=target_text,
                steps=steps,
                music_name=music_name,
            )

            # Step 3: Check watermark on adversarial audio
            # WaveformAttacker returns adversarial_wav at 24kHz
            adv_wav = attack_result.adversarial_wav.to(device)
            if adv_wav.dim() == 2:
                adv_wav = adv_wav.unsqueeze(0)
            if adv_wav.dim() == 3 and adv_wav.shape[1] != 1:
                # might be [1, T] -> unsqueeze channel
                pass
            adv_result = defense.verify_after_codec(adv_wav, message)

            entry = {
                "music": music_name,
                "command": cmd_name,
                "target_text": target_text,
                "attack_success": attack_result.success,
                "attack_output": attack_result.adversarial_output,
                "snr_db": attack_result.snr_db,
                "clean_watermark": clean_result.to_json(),
                "adversarial_watermark": adv_result.to_json(),
                "watermark_destroyed": not adv_result.is_watermarked,
                "detection_score_drop": clean_result.detection_score - adv_result.detection_score,
            }
            results["results"].append(entry)

            status = "DESTROYED" if not adv_result.is_watermarked else "SURVIVED"
            print(f"    Attack success: {attack_result.success}")
            print(f"    SNR: {attack_result.snr_db:.2f} dB")
            print(f"    Watermark: {status} (score={adv_result.detection_score:.4f})")
            print(f"    Score drop: {entry['detection_score_drop']:.4f}")

    # Summary
    total = len(results["results"])
    successful_attacks = [r for r in results["results"] if r["attack_success"]]
    destroyed_on_successful = [r for r in successful_attacks if r["watermark_destroyed"]]
    destroyed_all = [r for r in results["results"] if r["watermark_destroyed"]]

    results["summary"] = {
        "total_pairs": total,
        "successful_attacks": len(successful_attacks),
        "attack_success_rate": len(successful_attacks) / total if total > 0 else 0,
        "watermark_destroyed_on_successful": len(destroyed_on_successful),
        "watermark_destroyed_total": len(destroyed_all),
        "defense_bypass_rate": len(destroyed_on_successful) / len(successful_attacks) if successful_attacks else 0,
        "watermark_survival_rate": 1.0 - (len(destroyed_all) / total if total > 0 else 0),
        "avg_clean_score": np.mean([r["clean_watermark"]["detection_score"] for r in results["results"]]),
        "avg_adversarial_score": np.mean([r["adversarial_watermark"]["detection_score"] for r in results["results"]]),
        "avg_snr_db": np.mean([r["snr_db"] for r in results["results"]]),
    }

    print(f"\n{'='*60}")
    print("Waveform Baseline Watermark Summary:")
    print(f"  Total pairs: {total}")
    print(f"  Successful attacks: {len(successful_attacks)}/{total} ({results['summary']['attack_success_rate']:.1%})")
    print(f"  Watermark destroyed (on successful): {len(destroyed_on_successful)}/{len(successful_attacks)}")
    print(f"  Watermark survival rate: {results['summary']['watermark_survival_rate']:.1%}")
    print(f"  Avg clean score: {results['summary']['avg_clean_score']:.4f}")
    print(f"  Avg adversarial score: {results['summary']['avg_adversarial_score']:.4f}")
    print(f"  Avg SNR: {results['summary']['avg_snr_db']:.2f} dB")

    # Include eps and music in output path to avoid overwriting across conditions
    music_tag = "_".join(sorted(set(r["music"] for r in results["results"])))
    save_results(results, f"waveform_attack_detection/eps{eps}_{music_tag}")
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AudioSeal watermark defense experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["codec_roundtrip", "attack_detection",
                                 "waveform_attack_detection",
                                 "opus_robustness", "threshold_analysis", "all"],
                        help="Which experiment to run")
    parser.add_argument("--music", type=str, nargs="*", default=None,
                        help="Music carriers to use (default: all)")
    parser.add_argument("--dataset", type=str, default="ours",
                        choices=DATASET_CHOICES,
                        help="Command dataset: ours (200 cmds) | saycan (81 cmds)")
    parser.add_argument("--num_commands", type=int, default=None,
                        help="Number of commands (default: all commands in dataset)")
    parser.add_argument("--eps", type=float, default=None,
                        help="Perturbation epsilon (default: LATENT_EPS for latent, WAVEFORM_EPS for waveform)")
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS,
                        help="Attack optimization steps")
    parser.add_argument("--model", type=str, default=TARGET_MODEL,
                        help=f"Target model for attack experiments (default: {TARGET_MODEL})")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Resolve music list
    if args.music:
        music_names = args.music
    else:
        music_names = list(MUSIC_FILES.keys())

    # Resolve dataset
    if args.dataset == "saycan":
        commands = load_saycan_commands()
    else:
        commands = AGENT_COMMANDS

    # Resolve num_commands: default to all if not specified
    num_commands = args.num_commands if args.num_commands is not None else len(commands)

    # Resolve eps default based on experiment type
    eps = args.eps if args.eps is not None else LATENT_EPS

    target_model = args.model

    # Use dataset-aware results directory to avoid overwriting
    if args.dataset != "ours":
        global RESULTS_WATERMARK_DIR
        RESULTS_WATERMARK_DIR = os.path.join(
            os.path.dirname(RESULTS_DIR), f"results_watermark_{args.dataset}"
        )

    print(f"Target model: {target_model}")
    print(f"Dataset: {args.dataset} ({len(commands)} total, using {num_commands})")
    print(f"Music carriers: {music_names}")
    print(f"Device: {args.device}")

    if args.experiment in ("codec_roundtrip", "all"):
        experiment_codec_roundtrip(music_names, device=args.device)

    if args.experiment in ("opus_robustness", "all"):
        experiment_opus_robustness(music_names, device=args.device)

    if args.experiment in ("attack_detection", "all"):
        experiment_attack_detection(
            music_names,
            num_commands=num_commands,
            eps=eps,
            steps=args.steps,
            device=args.device,
            commands=commands,
            target_model=target_model,
        )

    if args.experiment in ("threshold_analysis", "all"):
        experiment_threshold_analysis(
            music_names,
            num_commands=num_commands,
            eps=eps,
            steps=args.steps,
            device=args.device,
            commands=commands,
            target_model=target_model,
        )

    if args.experiment in ("waveform_attack_detection", "all"):
        wf_eps = args.eps if args.eps is not None else WAVEFORM_EPS
        experiment_waveform_attack_detection(
            music_names,
            num_commands=num_commands,
            eps=wf_eps,
            steps=args.steps,
            device=args.device,
            commands=commands,
            target_model=target_model,
        )


if __name__ == "__main__":
    main()
