"""
Standalone evaluator for pre-crafted black-box adversarial audio.

Loads a single target model and evaluates all audio files in a directory.
Designed to run in the model's conda env (e.g., flamingo3 for AF3).

Usage:
    conda run -n codec-attack python eval_blackbox.py \
        --model qwen2_audio \
        --audio-dir results_blackbox/benchmark_blend_.../audio \
        --output-dir results_blackbox/benchmark_blend_.../qwen2_audio \
        --prompt-mode transcribe

    conda run -n flamingo3 python eval_blackbox.py \
        --model audio_flamingo \
        --audio-dir results_blackbox/benchmark_blend_.../audio \
        --output-dir results_blackbox/benchmark_blend_.../audio_flamingo \
        --prompt-mode transcribe
"""

import argparse
import json
import os
import sys
import glob
import time

import torch
import torchaudio

from config import (
    WHISPER_INJECT_ROOT, MODEL_PATHS, PROMPT_MODES,
    TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE,
    DEFAULT_PROMPT_MODE, OPUS_EVAL_BITRATES,
    compute_wer,
)

sys.path.insert(0, WHISPER_INJECT_ROOT)


def load_model(model_name, device="cuda", dtype=torch.bfloat16):
    """Load target model."""
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        return Qwen2AudioModel(model_path=MODEL_PATHS[model_name], device=device, dtype=dtype)
    elif model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        return AudioFlamingoModel(model_path=MODEL_PATHS[model_name], device=device, dtype=dtype)
    elif model_name == "qwen25_omni":
        from models.qwen25_omni import Qwen25OmniModel
        return Qwen25OmniModel(model_path=MODEL_PATHS[model_name], device=device, dtype=dtype)
    elif model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        return KimiAudioModel(model_path=MODEL_PATHS[model_name], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def eval_audio(model, audio_path, prompt, device="cuda"):
    """Evaluate single audio file."""
    import soundfile as sf

    wav_np, sr = sf.read(audio_path)
    audio = torch.FloatTensor(wav_np).unsqueeze(0)
    if sr != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SAMPLE_RATE)
    audio = audio.to(device)

    with torch.no_grad():
        return model.generate(audio, prompt=prompt)


def parse_filename(filename):
    """Parse experiment key from filename.

    Expected format: {music}__{cmd_name}__alpha{X}.wav
    Returns (music, cmd_name, alpha) or None.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("__")
    if len(parts) != 3:
        return None
    music = parts[0]
    cmd_name = parts[1]
    alpha_str = parts[2]
    if not alpha_str.startswith("alpha"):
        return None
    try:
        alpha = float(alpha_str[5:])
    except ValueError:
        return None
    return music, cmd_name, alpha


def run_eval(args):
    """Evaluate all audio files in audio_dir."""
    audio_dir = args.audio_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    prompt_mode = args.prompt_mode
    prompt_text = PROMPT_MODES[prompt_mode]["prompt"]
    opus_bitrates = [int(b) for b in args.opus_bitrates.split(",")]

    # Find all direct audio files (not in opus subdirs)
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"Found {len(wav_files)} audio files in {audio_dir}")

    # Load target text mapping from config.json (in parent dir)
    parent_dir = os.path.dirname(audio_dir)
    config_path = os.path.join(parent_dir, "config.json")
    targets = {}
    commands = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        targets = cfg.get("targets", {})
        commands = cfg.get("commands", {})

    # Load completed
    completed = set()
    for entry in os.listdir(output_dir) if os.path.isdir(output_dir) else []:
        rec_path = os.path.join(output_dir, entry, "record.json")
        if os.path.isfile(rec_path):
            try:
                with open(rec_path) as f:
                    rec = json.load(f)
                completed.add((rec["music"], rec["command_name"], rec["alpha"]))
            except (json.JSONDecodeError, KeyError):
                pass

    if completed:
        print(f"Skipping {len(completed)} already-completed")

    # Quick mode subset
    if args.quick:
        wav_files = wav_files[:4]

    # Load model
    print(f"Loading model: {args.model}...")
    model = load_model(args.model)

    all_results = []

    # Load existing results
    for entry in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
        rec_path = os.path.join(output_dir, entry, "record.json")
        if os.path.isfile(rec_path):
            try:
                with open(rec_path) as f:
                    all_results.append(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass

    for i, wav_path in enumerate(wav_files):
        filename = os.path.basename(wav_path)
        parsed = parse_filename(filename)
        if parsed is None:
            print(f"Skipping unrecognized file: {filename}")
            continue

        music, cmd_name, alpha = parsed

        if (music, cmd_name, alpha) in completed:
            continue

        target_text = targets.get(cmd_name, "")
        cmd_text = commands.get(cmd_name, "")

        print(f"\n[{i+1}/{len(wav_files)}] {filename}")
        print(f"  Target: \"{target_text[:80]}\"")

        start_time = time.time()

        # Direct eval
        direct_output = eval_audio(model, wav_path, prompt_text)
        direct_success = target_text.lower() in direct_output.lower()

        # Opus eval
        opus_results = {}
        for bw in opus_bitrates:
            opus_dir = os.path.join(audio_dir, f"opus{bw}")
            opus_wav = os.path.join(opus_dir, filename)
            if os.path.isfile(opus_wav):
                opus_output = eval_audio(model, opus_wav, prompt_text)
                opus_success = target_text.lower() in opus_output.lower()
                entry = {"success": opus_success, "output": opus_output}
                if prompt_mode == "transcribe" and target_text:
                    entry["wer"] = compute_wer(target_text, opus_output)
                opus_results[str(bw)] = entry

        duration = time.time() - start_time

        record = {
            "music": music,
            "command_name": cmd_name,
            "command_text": cmd_text,
            "target_text": target_text,
            "alpha": alpha,
            "direct": {
                "success": direct_success,
                "output": direct_output,
            },
            "opus": opus_results,
            "prompt_mode": prompt_mode,
            "prompt_text": prompt_text,
            "duration_s": duration,
        }

        if prompt_mode == "transcribe" and target_text:
            record["direct"]["wer"] = compute_wer(target_text, direct_output)

        all_results.append(record)

        # Save record
        exp_key = os.path.splitext(filename)[0]
        exp_dir = os.path.join(output_dir, exp_key)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "record.json"), "w") as f:
            json.dump(record, f, indent=2)

        d_ok = "OK" if direct_success else "FAIL"
        opus_status = " | ".join(
            f"opus{bw}={'OK' if opus_results.get(str(bw), {}).get('success') else 'FAIL'}"
            for bw in opus_bitrates
        )
        print(f"  {d_ok} | {opus_status} | {duration:.1f}s")
        print(f"  Output: {direct_output[:100]}")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    n = len(all_results)
    if n > 0:
        direct_ok = sum(1 for r in all_results if r["direct"]["success"])
        stats = {
            "total_experiments": n,
            "direct_success_rate": direct_ok / n,
            "direct_successes": direct_ok,
        }
        for bw in opus_bitrates:
            bw_key = str(bw)
            ok = sum(1 for r in all_results
                     if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            stats[f"opus_{bw}_success_rate"] = ok / n
            stats[f"opus_{bw}_successes"] = ok

        # Per-alpha
        by_alpha = {}
        seen_alphas = sorted(set(r.get("alpha", 0) for r in all_results))
        for a in seen_alphas:
            a_res = [r for r in all_results if r.get("alpha") == a]
            a_n = len(a_res)
            a_ok = sum(1 for r in a_res if r["direct"]["success"])
            entry = {"total": a_n, "direct_ok": a_ok, "direct_rate": a_ok / a_n}
            for bw in opus_bitrates:
                bw_key = str(bw)
                ok = sum(1 for r in a_res
                         if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
                entry[f"opus_{bw}_ok"] = ok
                entry[f"opus_{bw}_rate"] = ok / a_n
            by_alpha[str(a)] = entry
        stats["by_alpha"] = by_alpha
    else:
        stats = {}

    with open(summary_path, "w") as f:
        json.dump({"stats": stats, "results": all_results}, f, indent=2)

    # Print summary
    if n > 0:
        print(f"\n{'=' * 60}")
        print(f"EVAL SUMMARY — {args.model}")
        print(f"{'=' * 60}")
        print(f"Total: {n}")
        direct_ok = sum(1 for r in all_results if r["direct"]["success"])
        print(f"Direct: {direct_ok}/{n} ({100*direct_ok/n:.1f}%)")
        for bw in opus_bitrates:
            bw_key = str(bw)
            ok = sum(1 for r in all_results
                     if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            print(f"Opus {bw}: {ok}/{n} ({100*ok/n:.1f}%)")

        print(f"\nPer-alpha:")
        for a in sorted(set(r.get("alpha", 0) for r in all_results)):
            a_res = [r for r in all_results if r.get("alpha") == a]
            a_n = len(a_res)
            a_ok = sum(1 for r in a_res if r["direct"]["success"])
            print(f"  alpha={a:.2f}: {a_ok}/{a_n} ({100*a_ok/a_n:.1f}%)")

    print(f"\nResults: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate black-box adversarial audio on a target model")
    parser.add_argument("--model", type=str, required=True,
                        help="Target model (qwen2_audio, audio_flamingo, qwen25_omni, kimi_audio)")
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Directory containing adversarial WAV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write evaluation results")
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--opus-bitrates", type=str,
                        default=",".join(str(b) for b in OPUS_EVAL_BITRATES))
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
