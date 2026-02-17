"""
Cross-model evaluation of adversarial audio.

Loads adversarial WAVs from a benchmark's audio/ directory and evaluates
them on one or more models. Saves per-model responses alongside the
attack model's results.

Usage:
    python eval_models.py --benchmark results_qa/benchmark_eps0.4_jazz_1 --models kimi_audio
    python eval_models.py --benchmark results_qa/benchmark_eps0.4_jazz_1 --models kimi_audio audio_flamingo
    python eval_models.py --benchmark results_qa/benchmark_eps0.4_jazz_1 --models kimi_audio --resume
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torchaudio

from config import (
    WHISPER_INJECT_ROOT, MODEL_PATHS, EVAL_MODELS,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    PROMPT_MODES, DEFAULT_PROMPT_MODE,
    compute_wer, ComplianceJudge,
)

sys.path.insert(0, WHISPER_INJECT_ROOT)


def load_model(model_name: str, device: str = "cuda"):
    """Load an eval model by name. Returns (model, sample_rate)."""
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        model = Qwen2AudioModel(
            model_path=MODEL_PATHS["qwen2_audio"],
            device=device,
            dtype=torch.bfloat16,
        )
        return model, TARGET_SAMPLE_RATE

    elif model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        model = KimiAudioModel(
            model_path=MODEL_PATHS.get("kimi_audio"),
            device=device,
        )
        return model, model.SAMPLE_RATE

    elif model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        model = AudioFlamingoModel(
            model_path=MODEL_PATHS.get("audio_flamingo"),
            device=device,
            dtype=torch.bfloat16,
        )
        return model, model.SAMPLE_RATE

    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {EVAL_MODELS}")


def resample_and_cache(wav_path: str, cache_dir: str, target_sr: int) -> str:
    """Resample a WAV file and cache the result. Returns path to resampled file."""
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, os.path.basename(wav_path))

    if os.path.isfile(cached_path):
        return cached_path

    import soundfile as sf
    wav_np, sr = sf.read(wav_path)
    wav_tensor = torch.FloatTensor(wav_np).unsqueeze(0)  # [1, T]

    if sr != target_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, target_sr)

    wav_resampled = wav_tensor.squeeze(0).numpy()
    sf.write(cached_path, wav_resampled, target_sr)
    return cached_path


def load_attack_records(benchmark_dir: str) -> Dict[str, dict]:
    """Load attack records from the qwen2_audio subdir (or old flat layout).

    Returns dict: {exp_key: record}
    """
    records = {}

    # Try new layout: qwen2_audio/{exp_key}/record.json
    model_dir = os.path.join(benchmark_dir, "qwen2_audio")
    if os.path.isdir(model_dir):
        for entry in sorted(os.listdir(model_dir)):
            rec_path = os.path.join(model_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        records[entry] = json.load(f)
                except (json.JSONDecodeError, KeyError):
                    pass

    # Fallback: old flat layout {exp_key}/record.json
    if not records:
        for entry in sorted(os.listdir(benchmark_dir)):
            rec_path = os.path.join(benchmark_dir, entry, "record.json")
            if os.path.isfile(rec_path):
                try:
                    with open(rec_path) as f:
                        records[entry] = json.load(f)
                except (json.JSONDecodeError, KeyError):
                    pass

    return records


def find_audio_files(benchmark_dir: str) -> Dict[str, str]:
    """Find adversarial WAV files. Returns dict: {exp_key: wav_path}.

    Checks audio/ subdir first (new layout), then old per-experiment dirs.
    """
    audio_files = {}

    # New layout: audio/{exp_key}.wav
    audio_dir = os.path.join(benchmark_dir, "audio")
    if os.path.isdir(audio_dir):
        for fname in sorted(os.listdir(audio_dir)):
            if fname.endswith(".wav") and not fname.startswith("."):
                exp_key = os.path.splitext(fname)[0]
                audio_files[exp_key] = os.path.join(audio_dir, fname)

    # Fallback: old layout {exp_key}/bench_*_adversarial.wav
    if not audio_files:
        for entry in sorted(os.listdir(benchmark_dir)):
            entry_dir = os.path.join(benchmark_dir, entry)
            if not os.path.isdir(entry_dir):
                continue
            for fname in os.listdir(entry_dir):
                if fname.endswith("_adversarial.wav"):
                    audio_files[entry] = os.path.join(entry_dir, fname)
                    break

    return audio_files


def load_completed_eval(model_dir: str) -> set:
    """Load already-evaluated exp_keys for a model."""
    completed = set()
    if not os.path.isdir(model_dir):
        return completed
    for entry in os.listdir(model_dir):
        rec_path = os.path.join(model_dir, entry, "record.json")
        if os.path.isfile(rec_path):
            completed.add(entry)
    return completed


def evaluate_model(
    model_name: str,
    benchmark_dir: str,
    audio_files: Dict[str, str],
    attack_records: Dict[str, dict],
    prompt_text: str,
    prompt_mode: str,
    device: str = "cuda",
    resume: bool = True,
    judge: Optional["ComplianceJudge"] = None,
):
    """Evaluate all adversarial audio files on a single model."""
    model_dir = os.path.join(benchmark_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Check what's already done
    completed = load_completed_eval(model_dir) if resume else set()
    todo = {k: v for k, v in audio_files.items() if k not in completed}

    if not todo:
        print(f"  {model_name}: all {len(audio_files)} experiments already evaluated")
        return

    print(f"  {model_name}: {len(todo)} to evaluate ({len(completed)} already done)")

    # Load model
    model, model_sr = load_model(model_name, device=device)

    # Cache dir for resampled audio
    cache_dir = os.path.join(benchmark_dir, "audio", f"{model_sr // 1000}k")

    all_results = []
    for i, (exp_key, wav_path) in enumerate(sorted(todo.items())):
        attack_rec = attack_records.get(exp_key, {})
        # New records have separate target_text; old records used command_text as target
        target_text = attack_rec.get("target_text", attack_rec.get("command_text", ""))
        music = attack_rec.get("music", "")
        cmd_name = attack_rec.get("command_name", "")
        category = attack_rec.get("category", "")

        print(f"    [{i+1}/{len(todo)}] {exp_key} ...", end=" ", flush=True)
        start = time.time()

        # Get audio at correct sample rate
        if model_sr != ENCODEC_SAMPLE_RATE:
            resampled_path = resample_and_cache(wav_path, cache_dir, model_sr)
        else:
            resampled_path = wav_path

        # Generate response
        try:
            if model_name == "kimi_audio":
                output = model.generate_from_path(resampled_path, prompt=prompt_text)
            else:
                import soundfile as sf
                wav_np, sr = sf.read(resampled_path)
                wav_tensor = torch.FloatTensor(wav_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model.generate(wav_tensor, prompt=prompt_text)
        except Exception as e:
            output = f"ERROR: {e}"
            print(f"ERROR: {e}")

        duration = time.time() - start
        success = target_text.lower() in output.lower() if target_text else False

        command_text = attack_rec.get("command_text", "")
        record = {
            "music": music,
            "command_name": cmd_name,
            "command_text": command_text,
            "target_text": target_text,
            "category": category,
            "model": model_name,
            "prompt_mode": prompt_mode,
            "prompt_text": prompt_text,
            "output": output,
            "success": success,
            "duration_s": duration,
        }

        # Compute WER for transcription mode
        if prompt_mode == "transcribe" and target_text:
            record["wer"] = compute_wer(target_text, output)

        # LLM compliance judge for QA mode
        if prompt_mode == "qa" and judge is not None and command_text:
            verdict = judge.judge(command_text, output)
            record["compliance"] = verdict["compliant"]
            record["compliance_raw"] = verdict["raw_output"]

        # Save per-experiment record
        exp_record_dir = os.path.join(model_dir, exp_key)
        os.makedirs(exp_record_dir, exist_ok=True)
        with open(os.path.join(exp_record_dir, "record.json"), "w") as f:
            json.dump(record, f, indent=2)

        all_results.append(record)
        status = "OK" if success else "FAIL"
        print(f"{status} ({duration:.1f}s) — {output[:80]}")

    # Save model summary
    total = len(all_results)
    if total > 0:
        successes = sum(1 for r in all_results if r["success"])
        summary = {
            "model": model_name,
            "total": total,
            "success_rate": successes / total,
            "successes": successes,
            "results": all_results,
        }

        # Add average WER for transcription mode
        wer_values = [r["wer"] for r in all_results if "wer" in r]
        if wer_values:
            avg_wer = sum(wer_values) / len(wer_values)
            summary["avg_wer"] = avg_wer

        # Add compliance rate for QA mode
        compliance_values = [r["compliance"] for r in all_results if "compliance" in r]
        if compliance_values:
            compliance_rate = sum(compliance_values) / len(compliance_values)
            summary["compliance_rate"] = compliance_rate

        with open(os.path.join(model_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        msg = f"  {model_name}: {successes}/{total} ({100*successes/total:.0f}%) exact match"
        if wer_values:
            msg += f" | avg WER: {avg_wer:.3f}"
        if compliance_values:
            compliant = sum(compliance_values)
            msg += f" | compliance: {compliant}/{len(compliance_values)} ({100*compliance_rate:.0f}%)"
        print(msg)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_success(record: dict) -> bool:
    """Check success from either attack record (direct.success) or eval record (success)."""
    if "success" in record:
        return record["success"]
    return record.get("direct", {}).get("success", False)


def print_comparison(benchmark_dir: str, model_names: List[str]):
    """Print cross-model comparison table."""
    # Load all results per model
    model_results = {}
    for mname in model_names:
        model_dir = os.path.join(benchmark_dir, mname)
        results = {}
        if os.path.isdir(model_dir):
            for entry in sorted(os.listdir(model_dir)):
                rec_path = os.path.join(model_dir, entry, "record.json")
                if os.path.isfile(rec_path):
                    try:
                        with open(rec_path) as f:
                            results[entry] = json.load(f)
                    except (json.JSONDecodeError, KeyError):
                        pass
        model_results[mname] = results

    # All exp_keys across models
    all_keys = set()
    for results in model_results.values():
        all_keys.update(results.keys())

    if not all_keys:
        print("No results to compare.")
        return

    print("\n" + "=" * 90)
    print("CROSS-MODEL COMPARISON")
    print("=" * 90)

    # Overall success rates
    header = f"{'Model':<20} {'Exact Match':>12}  {'Rate':>6}  {'Avg WER':>8}  {'Compliance':>12}"
    print(header)
    print("-" * 75)
    for mname in model_names:
        results = model_results[mname]
        total = len(results)
        if total == 0:
            print(f"{mname:<20}  {'(no results)':>12}")
            continue
        successes = sum(1 for r in results.values() if r.get("success"))
        wer_vals = [r["wer"] for r in results.values() if "wer" in r]
        wer_str = f"{sum(wer_vals)/len(wer_vals):.3f}" if wer_vals else "---"
        comp_vals = [r["compliance"] for r in results.values() if "compliance" in r]
        if comp_vals:
            comp_ok = sum(comp_vals)
            comp_str = f"{comp_ok}/{len(comp_vals)} ({100*comp_ok/len(comp_vals):.0f}%)"
        else:
            comp_str = "---"
        print(f"{mname:<20}  {successes:>4}/{total:<4}     {100*successes/total:5.1f}%  {wer_str:>8}  {comp_str:>12}")

    # Per-category comparison
    categories = set()
    for results in model_results.values():
        for r in results.values():
            cat = r.get("category", "other")
            if cat:
                categories.add(cat)

    if categories:
        print(f"\n{'':─<90}")
        print("PER-CATEGORY SUCCESS RATES")
        print(f"{'':─<90}")
        header = f"{'Category':<20}"
        for mname in model_names:
            header += f" {mname:>15}"
        print(header)
        print("-" * (20 + 16 * len(model_names)))

        for cat in sorted(categories):
            line = f"{cat:<20}"
            for mname in model_names:
                cat_results = [r for r in model_results[mname].values() if r.get("category") == cat]
                if not cat_results:
                    line += f" {'---':>15}"
                else:
                    ok = sum(1 for r in cat_results if r.get("success"))
                    n = len(cat_results)
                    line += f" {ok:>3}/{n:<3}({100*ok/n:4.0f}%)"
                    line += " " * max(0, 15 - len(f"{ok:>3}/{n:<3}({100*ok/n:4.0f}%)"))
            print(line)

    print()


def main():
    parser = argparse.ArgumentParser(description="Cross-model evaluation of adversarial audio")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to benchmark results directory")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to evaluate. Available: {', '.join(EVAL_MODELS)}")
    parser.add_argument("--prompt-mode", type=str, default=None,
                        help="Override prompt mode (reads from config.json by default)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-evaluate all (don't skip completed)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only print comparison table (no evaluation)")

    args = parser.parse_args()

    benchmark_dir = args.benchmark
    if not os.path.isdir(benchmark_dir):
        print(f"Error: benchmark directory not found: {benchmark_dir}")
        sys.exit(1)

    # Determine models to evaluate
    model_names = args.models
    if model_names is None:
        # Default: all non-attack models
        model_names = [m for m in EVAL_MODELS if m != "qwen2_audio"]

    # Read prompt mode from config.json if not overridden
    prompt_mode = args.prompt_mode
    prompt_text = None
    config_path = os.path.join(benchmark_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if prompt_mode is None:
            prompt_mode = config.get("prompt_mode", DEFAULT_PROMPT_MODE)
        if prompt_text is None:
            prompt_text = config.get("prompt_text")

    if prompt_mode is None:
        prompt_mode = DEFAULT_PROMPT_MODE
    if prompt_text is None:
        prompt_text = PROMPT_MODES[prompt_mode]["prompt"]

    print(f"Benchmark: {benchmark_dir}")
    print(f"Prompt mode: {prompt_mode} — \"{prompt_text}\"")
    print(f"Models: {', '.join(model_names)}")

    if args.compare_only:
        # Include qwen2_audio in comparison if it has results
        all_models = ["qwen2_audio"] + [m for m in model_names if m != "qwen2_audio"]
        print_comparison(benchmark_dir, all_models)
        return

    # Load attack records and audio files
    attack_records = load_attack_records(benchmark_dir)
    audio_files = find_audio_files(benchmark_dir)

    if not audio_files:
        print("Error: no adversarial audio files found.")
        print(f"Expected at: {os.path.join(benchmark_dir, 'audio', '*.wav')}")
        sys.exit(1)

    print(f"Found {len(audio_files)} adversarial audio files")
    print(f"Found {len(attack_records)} attack records")

    # Load compliance judge for QA mode
    judge = None
    if prompt_mode == "qa":
        print("\nLoading compliance judge (Llama-3.2-3B-Instruct)...")
        judge = ComplianceJudge()
        print("Compliance judge loaded.")

    # Evaluate each model sequentially (one model at a time to avoid OOM)
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")
        evaluate_model(
            model_name=model_name,
            benchmark_dir=benchmark_dir,
            audio_files=audio_files,
            attack_records=attack_records,
            prompt_text=prompt_text,
            prompt_mode=prompt_mode,
            resume=not args.no_resume,
            judge=judge,
        )

    # Print comparison
    all_models = ["qwen2_audio"] + [m for m in model_names if m != "qwen2_audio"]
    print_comparison(benchmark_dir, all_models)


if __name__ == "__main__":
    main()
