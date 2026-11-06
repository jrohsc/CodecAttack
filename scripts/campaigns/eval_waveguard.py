"""
WaveGuard defense evaluation for adversarial audio.

Applies signal-level transforms (WaveGuard) to adversarial audio and measures
whether model output changes, indicating the perturbation was destroyed.

Usage:
    python eval_waveguard.py \
        --benchmark results_qa/benchmark_qwen2_audio_eps0.3_calm_1_saycan \
        --model qwen2_audio \
        --transforms quantize_8bit downup_8k lowpass_4k mel_reconstruct gaussian_0.005 \
        --threshold 0.5 \
        --include-clean \
        --resume
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torchaudio
import soundfile as sf

from config import (
    WHISPER_INJECT_ROOT, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    PROMPT_MODES, DEFAULT_PROMPT_MODE, MUSIC_FILES,
    WAVEGUARD_DETECTION_THRESHOLD, WAVEGUARD_THRESHOLDS_SWEEP,
    compute_wer,
)
from eval_models import (
    load_model, find_audio_files, load_attack_records, resample_and_cache,
)
from waveguard_defense import WaveGuardDefense, WaveGuardResult
from music_carrier import load_music_by_name

sys.path.insert(0, WHISPER_INJECT_ROOT)


def generate_output(model, model_name: str, wav_path: str, model_sr: int,
                    prompt_text: str, device: str = "cuda") -> str:
    """Generate model output for an audio file."""
    if model_name == "kimi_audio":
        return model.generate_from_path(wav_path, prompt=prompt_text)

    wav_np, sr = sf.read(wav_path)
    wav_tensor = torch.FloatTensor(wav_np).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.generate(wav_tensor, prompt=prompt_text)


def generate_output_from_tensor(model, model_name: str, audio: torch.Tensor,
                                model_sr: int, prompt_text: str,
                                device: str = "cuda") -> str:
    """Generate model output from an audio tensor [1, T]."""
    audio = audio.to(device)
    with torch.no_grad():
        return model.generate(audio.unsqueeze(0) if audio.dim() == 1 else audio,
                              prompt=prompt_text)


def save_transformed_audio(audio: torch.Tensor, path: str, sample_rate: int):
    """Save audio tensor to WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio_np = audio.squeeze().cpu().numpy()
    sf.write(path, audio_np, sample_rate)


def evaluate_waveguard(
    model_name: str,
    benchmark_dir: str,
    audio_files: Dict[str, str],
    attack_records: Dict[str, dict],
    prompt_text: str,
    prompt_mode: str,
    transform_names: List[str],
    threshold: float = 0.5,
    include_clean: bool = False,
    resume: bool = True,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    threshold_sweep: Optional[List[float]] = None,
):
    """Run WaveGuard evaluation on adversarial (and optionally clean) audio."""

    waveguard_dir = os.path.join(benchmark_dir, "waveguard")
    os.makedirs(waveguard_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name} ...")
    model, model_sr = load_model(model_name, device=device)
    print(f"Model loaded. Sample rate: {model_sr}")

    # Cache dir for resampled audio
    cache_dir = os.path.join(benchmark_dir, "audio", f"{model_sr // 1000}k")

    # ── Evaluate adversarial audio ──
    adv_results = _evaluate_set(
        label="adversarial",
        model=model,
        model_name=model_name,
        model_sr=model_sr,
        audio_files=audio_files,
        attack_records=attack_records,
        prompt_text=prompt_text,
        prompt_mode=prompt_mode,
        transform_names=transform_names,
        threshold=threshold,
        output_dir=os.path.join(waveguard_dir, "adversarial"),
        cache_dir=cache_dir,
        resume=resume,
        device=device,
        max_samples=max_samples,
    )

    # ── Evaluate clean music (false positive rate) ──
    clean_results = []
    if include_clean:
        clean_audio_files = {}
        clean_records = {}
        for name, path in MUSIC_FILES.items():
            if not os.path.isfile(path):
                continue
            clean_audio_files[name] = path
            clean_records[name] = {"music": name, "target_text": ""}

        clean_cache_dir = os.path.join(waveguard_dir, "clean_audio", f"{model_sr // 1000}k")
        clean_results = _evaluate_set(
            label="clean",
            model=model,
            model_name=model_name,
            model_sr=model_sr,
            audio_files=clean_audio_files,
            attack_records=clean_records,
            prompt_text=prompt_text,
            prompt_mode=prompt_mode,
            transform_names=transform_names,
            threshold=threshold,
            output_dir=os.path.join(waveguard_dir, "clean"),
            cache_dir=clean_cache_dir,
            resume=resume,
            device=device,
            max_samples=None,
            is_clean=True,
        )

    # ── Summary ──
    summary = _compute_summary(
        adv_results, clean_results, transform_names, threshold, threshold_sweep
    )
    summary["model"] = model_name
    summary["prompt_mode"] = prompt_mode
    summary["transforms"] = transform_names
    summary["threshold"] = threshold

    summary_path = os.path.join(waveguard_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    _print_summary(summary, transform_names)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_set(
    label: str,
    model,
    model_name: str,
    model_sr: int,
    audio_files: Dict[str, str],
    attack_records: Dict[str, dict],
    prompt_text: str,
    prompt_mode: str,
    transform_names: List[str],
    threshold: float,
    output_dir: str,
    cache_dir: str,
    resume: bool,
    device: str,
    max_samples: Optional[int],
    is_clean: bool = False,
) -> List[dict]:
    """Evaluate a set of audio files through WaveGuard transforms."""

    os.makedirs(output_dir, exist_ok=True)

    # Check what's already done
    completed = set()
    if resume:
        for entry in os.listdir(output_dir) if os.path.isdir(output_dir) else []:
            rec_path = os.path.join(output_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                completed.add(entry)

    # Determine what to evaluate
    todo_keys = sorted(audio_files.keys())
    if max_samples:
        todo_keys = todo_keys[:max_samples]

    new_keys = [k for k in todo_keys if k not in completed]
    if not new_keys and completed:
        print(f"\n  [{label}] All {len(todo_keys)} already evaluated, loading cached results")
    elif new_keys:
        print(f"\n  [{label}] {len(new_keys)} to evaluate ({len(completed)} cached)")

    all_results = []

    # Load cached results
    for exp_key in todo_keys:
        if exp_key in completed:
            rec_path = os.path.join(output_dir, exp_key, "record.json")
            try:
                with open(rec_path) as f:
                    all_results.append(json.load(f))
            except (json.JSONDecodeError, KeyError):
                new_keys.append(exp_key)  # re-evaluate if corrupt
            continue

    # Evaluate new
    for i, exp_key in enumerate(new_keys):
        wav_path = audio_files[exp_key]
        attack_rec = attack_records.get(exp_key, {})
        target_text = attack_rec.get("target_text", attack_rec.get("command_text", ""))

        print(f"    [{i+1}/{len(new_keys)}] {exp_key} ...", end=" ", flush=True)
        start = time.time()

        # Load audio at model's sample rate for model inference
        if model_sr != ENCODEC_SAMPLE_RATE and not is_clean:
            resampled_path = resample_and_cache(wav_path, cache_dir, model_sr)
        elif is_clean:
            # Clean audio might be MP3, need to load and resample
            resampled_path = _prepare_clean_audio(wav_path, cache_dir, model_sr)
        else:
            resampled_path = wav_path

        # Get model output on original (untransformed) audio
        try:
            output_original = generate_output(
                model, model_name, resampled_path, model_sr, prompt_text, device
            )
        except Exception as e:
            print(f"ERROR on original: {e}")
            continue

        # Load audio at EnCodec rate for transforms (transforms operate at native rate)
        wav_np_native, sr_native = sf.read(wav_path)
        if is_clean:
            import librosa
            wav_np_native, sr_native = librosa.load(wav_path, sr=ENCODEC_SAMPLE_RATE, mono=True)
        audio_native = torch.FloatTensor(wav_np_native).unsqueeze(0)  # [1, T]

        # Apply each transform
        per_transform = {}
        for tname in transform_names:
            result = WaveGuardDefense.apply_transform(
                audio_native, tname, sample_rate=sr_native if not is_clean else ENCODEC_SAMPLE_RATE
            )
            transformed = result.transformed_audio

            # Resample transformed audio to model's sample rate if needed
            if model_sr != (sr_native if not is_clean else ENCODEC_SAMPLE_RATE):
                transformed_at_model_sr = torchaudio.functional.resample(
                    transformed,
                    sr_native if not is_clean else ENCODEC_SAMPLE_RATE,
                    model_sr,
                )
            else:
                transformed_at_model_sr = transformed

            # Get model output on transformed audio
            try:
                output_transformed = generate_output_from_tensor(
                    model, model_name, transformed_at_model_sr,
                    model_sr, prompt_text, device
                )
            except Exception as e:
                output_transformed = f"ERROR: {e}"

            # Compute WER between original and transformed outputs
            wer = compute_wer(output_original, output_transformed)

            # Check neutralization: target text no longer in output
            neutralized = False
            if target_text and not is_clean:
                neutralized = target_text.lower() not in output_transformed.lower()

            per_transform[tname] = {
                "wer": wer,
                "flagged": wer > threshold,
                "snr_db": result.snr_db,
                "output_original": output_original,
                "output_transformed": output_transformed,
                "neutralized": neutralized,
            }

        # Aggregate detection decision
        num_flagged = sum(1 for v in per_transform.values() if v["flagged"])
        detected = num_flagged > len(transform_names) / 2

        duration = time.time() - start

        record = {
            "exp_key": exp_key,
            "is_clean": is_clean,
            "target_text": target_text,
            "output_original": output_original,
            "detected": detected,
            "num_flagged": num_flagged,
            "num_transforms": len(transform_names),
            "per_transform": per_transform,
            "duration_s": duration,
        }

        # Save per-experiment record
        exp_dir = os.path.join(output_dir, exp_key)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "record.json"), "w") as f:
            json.dump(record, f, indent=2)

        all_results.append(record)

        status = "DETECTED" if detected else "missed"
        print(f"{status} ({num_flagged}/{len(transform_names)} flagged) ({duration:.1f}s)")

    return all_results


def _prepare_clean_audio(audio_path: str, cache_dir: str, target_sr: int) -> str:
    """Load clean audio (possibly MP3), resample, and cache as WAV."""
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(audio_path))[0] + ".wav"
    cached_path = os.path.join(cache_dir, basename)

    if os.path.isfile(cached_path):
        return cached_path

    import librosa
    wav_np, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    sf.write(cached_path, wav_np, target_sr)
    return cached_path


def _compute_summary(
    adv_results: List[dict],
    clean_results: List[dict],
    transform_names: List[str],
    threshold: float,
    threshold_sweep: Optional[List[float]] = None,
) -> dict:
    """Compute summary statistics."""
    summary = {}

    # ── Adversarial stats ──
    n_adv = len(adv_results)
    if n_adv > 0:
        n_detected = sum(1 for r in adv_results if r["detected"])
        summary["adversarial"] = {
            "total": n_adv,
            "detected": n_detected,
            "detection_rate": n_detected / n_adv,
        }

        # Per-transform stats
        per_transform_stats = {}
        for tname in transform_names:
            flagged = 0
            neutralized = 0
            wers = []
            snrs = []
            for r in adv_results:
                info = r.get("per_transform", {}).get(tname, {})
                if info.get("flagged"):
                    flagged += 1
                if info.get("neutralized"):
                    neutralized += 1
                if "wer" in info:
                    wers.append(info["wer"])
                if "snr_db" in info and info["snr_db"] != float("inf"):
                    snrs.append(info["snr_db"])

            per_transform_stats[tname] = {
                "detection_rate": flagged / n_adv,
                "neutralization_rate": neutralized / n_adv,
                "avg_wer": sum(wers) / len(wers) if wers else 0,
                "avg_snr_db": sum(snrs) / len(snrs) if snrs else None,
                "flagged": flagged,
                "neutralized": neutralized,
            }

        summary["per_transform"] = per_transform_stats

    # ── Clean stats (false positive rate) ──
    n_clean = len(clean_results)
    if n_clean > 0:
        n_false_pos = sum(1 for r in clean_results if r["detected"])
        summary["clean"] = {
            "total": n_clean,
            "false_positives": n_false_pos,
            "false_positive_rate": n_false_pos / n_clean,
        }

        clean_per_transform = {}
        for tname in transform_names:
            flagged = 0
            wers = []
            for r in clean_results:
                info = r.get("per_transform", {}).get(tname, {})
                if info.get("flagged"):
                    flagged += 1
                if "wer" in info:
                    wers.append(info["wer"])
            clean_per_transform[tname] = {
                "false_positive_rate": flagged / n_clean,
                "avg_wer": sum(wers) / len(wers) if wers else 0,
                "flagged": flagged,
            }
        summary["clean_per_transform"] = clean_per_transform

    # ── Threshold sweep ──
    if threshold_sweep and n_adv > 0:
        sweep_results = []
        for t in threshold_sweep:
            # Re-compute detection at each threshold
            detected_adv = 0
            for r in adv_results:
                n_flag = sum(
                    1 for info in r.get("per_transform", {}).values()
                    if info.get("wer", 0) > t
                )
                if n_flag > len(transform_names) / 2:
                    detected_adv += 1

            detected_clean = 0
            for r in clean_results:
                n_flag = sum(
                    1 for info in r.get("per_transform", {}).values()
                    if info.get("wer", 0) > t
                )
                if n_flag > len(transform_names) / 2:
                    detected_clean += 1

            sweep_results.append({
                "threshold": t,
                "detection_rate": detected_adv / n_adv,
                "false_positive_rate": detected_clean / n_clean if n_clean > 0 else None,
            })
        summary["threshold_sweep"] = sweep_results

    return summary


def _print_summary(summary: dict, transform_names: List[str]):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("WAVEGUARD DEFENSE EVALUATION SUMMARY")
    print("=" * 80)

    adv = summary.get("adversarial", {})
    if adv:
        print(f"\nAdversarial: {adv['detected']}/{adv['total']} detected "
              f"({100*adv['detection_rate']:.1f}%)")

    clean = summary.get("clean", {})
    if clean:
        print(f"Clean (FP):  {clean['false_positives']}/{clean['total']} false positives "
              f"({100*clean['false_positive_rate']:.1f}%)")

    # Per-transform table
    pt = summary.get("per_transform", {})
    if pt:
        print(f"\n{'Transform':<20} {'Detect':>8} {'Neutral':>8} {'Avg WER':>8} {'Avg SNR':>8}")
        print("-" * 56)
        for tname in transform_names:
            stats = pt.get(tname, {})
            detect = f"{100*stats.get('detection_rate', 0):.0f}%"
            neutral = f"{100*stats.get('neutralization_rate', 0):.0f}%"
            wer = f"{stats.get('avg_wer', 0):.3f}"
            snr = f"{stats.get('avg_snr_db', 0):.1f}" if stats.get("avg_snr_db") else "---"
            print(f"{tname:<20} {detect:>8} {neutral:>8} {wer:>8} {snr:>8}")

    # Clean per-transform
    cpt = summary.get("clean_per_transform", {})
    if cpt:
        print(f"\n{'Transform':<20} {'FP Rate':>8} {'Avg WER':>8}")
        print("-" * 38)
        for tname in transform_names:
            stats = cpt.get(tname, {})
            fpr = f"{100*stats.get('false_positive_rate', 0):.0f}%"
            wer = f"{stats.get('avg_wer', 0):.3f}"
            print(f"{tname:<20} {fpr:>8} {wer:>8}")

    # Threshold sweep
    sweep = summary.get("threshold_sweep", [])
    if sweep:
        print(f"\n{'Threshold':>10} {'Detection':>10} {'FPR':>10}")
        print("-" * 32)
        for s in sweep:
            det = f"{100*s['detection_rate']:.1f}%"
            fpr = f"{100*s['false_positive_rate']:.1f}%" if s['false_positive_rate'] is not None else "---"
            print(f"{s['threshold']:>10.2f} {det:>10} {fpr:>10}")

    print()


def main():
    parser = argparse.ArgumentParser(description="WaveGuard defense evaluation")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark results directory")
    parser.add_argument("--model", type=str, default="qwen2_audio",
                        help="Model to evaluate (default: qwen2_audio)")
    parser.add_argument("--transforms", nargs="+", default=None,
                        help=f"Transforms to apply. Available: {list(WaveGuardDefense.DEFAULT_TRANSFORMS.keys())}")
    parser.add_argument("--threshold", type=float, default=WAVEGUARD_DETECTION_THRESHOLD,
                        help=f"WER threshold for detection (default: {WAVEGUARD_DETECTION_THRESHOLD})")
    parser.add_argument("--include-clean", action="store_true",
                        help="Also evaluate clean music for false positive rate")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-evaluated experiments")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max adversarial samples to evaluate")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Run threshold sweep for ROC-style analysis")
    parser.add_argument("--prompt-mode", type=str, default=None,
                        help="Override prompt mode (reads from config.json by default)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")

    args = parser.parse_args()

    benchmark_dir = args.benchmark
    if not os.path.isdir(benchmark_dir):
        print(f"Error: benchmark directory not found: {benchmark_dir}")
        sys.exit(1)

    # Read prompt mode from config.json if not overridden
    prompt_mode = args.prompt_mode
    prompt_text = None
    config_path = os.path.join(benchmark_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if prompt_mode is None:
            prompt_mode = config.get("prompt_mode", DEFAULT_PROMPT_MODE)
        prompt_text = config.get("prompt_text")

    if prompt_mode is None:
        prompt_mode = DEFAULT_PROMPT_MODE
    if prompt_text is None:
        prompt_text = PROMPT_MODES[prompt_mode]["prompt"]

    # Determine transforms
    transform_names = args.transforms
    if transform_names is None:
        transform_names = list(WaveGuardDefense.DEFAULT_TRANSFORMS.keys())

    # Validate transforms
    for t in transform_names:
        if t not in WaveGuardDefense.DEFAULT_TRANSFORMS:
            print(f"Error: unknown transform '{t}'. Available: {list(WaveGuardDefense.DEFAULT_TRANSFORMS.keys())}")
            sys.exit(1)

    threshold_sweep = WAVEGUARD_THRESHOLDS_SWEEP if args.threshold_sweep else None

    print(f"Benchmark: {benchmark_dir}")
    print(f"Model: {args.model}")
    print(f"Prompt mode: {prompt_mode} — \"{prompt_text}\"")
    print(f"Transforms: {transform_names}")
    print(f"Detection threshold: {args.threshold}")
    if args.include_clean:
        print(f"Including clean music for FP evaluation")

    # Load attack records and audio files
    attack_records = load_attack_records(benchmark_dir)
    audio_files = find_audio_files(benchmark_dir)

    if not audio_files:
        print("Error: no adversarial audio files found.")
        print(f"Expected at: {os.path.join(benchmark_dir, 'audio', '*.wav')}")
        sys.exit(1)

    print(f"Found {len(audio_files)} adversarial audio files")
    print(f"Found {len(attack_records)} attack records")

    evaluate_waveguard(
        model_name=args.model,
        benchmark_dir=benchmark_dir,
        audio_files=audio_files,
        attack_records=attack_records,
        prompt_text=prompt_text,
        prompt_mode=prompt_mode,
        transform_names=transform_names,
        threshold=args.threshold,
        include_clean=args.include_clean,
        resume=args.resume,
        device=args.device,
        max_samples=args.max_samples,
        threshold_sweep=threshold_sweep,
    )


if __name__ == "__main__":
    main()
