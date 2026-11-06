"""
Black-box latent blending benchmark.

For each (music, command, alpha) triple:
  1. Generate TTS of command
  2. Blend TTS + music latents in EnCodec space (no target model)
  3. Save adversarial audio (direct + Opus 64/128 variants)
  4. Evaluate on target model (direct + Opus)

Usage:
    # Quick test (2 commands, single alpha)
    python run_blackbox_benchmark.py --quick --alpha 0.3

    # Full sweep
    python run_blackbox_benchmark.py --alpha 0.1,0.3,0.5,0.7,1.0 --music empire_state_of_mind

    # Single alpha, specific model eval
    python run_blackbox_benchmark.py --alpha 0.5 --target-model qwen2_audio --music empire_state_of_mind
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
import torchaudio

from config import (
    MUSIC_FILES, DEFAULT_MUSIC, AGENT_COMMANDS,
    OPUS_EVAL_BITRATES, PROMPT_MODES, DEFAULT_PROMPT_MODE,
    TARGET_MODEL, TARGET_SAMPLE_RATE, MODEL_PATHS,
    ENCODEC_SAMPLE_RATE, WHISPER_INJECT_ROOT,
    DATASET_CHOICES, load_saycan_commands, compute_wer, save_results_summary,
)
from music_carrier import load_music_by_name, resolve_music_path
from blackbox_blend import LatentBlendAttacker, BlendResult

# Reuse helpers from existing benchmark
from run_benchmark import (
    filter_commands, get_category, CATEGORY_PREFIXES,
    _music_short_name, _save_summary, _print_summary,
)

sys.path.insert(0, WHISPER_INJECT_ROOT)


def load_target_model(model_name, device="cuda", dtype=torch.bfloat16):
    """Load a target model for evaluation."""
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        return Qwen2AudioModel(
            model_path=MODEL_PATHS[model_name], device=device, dtype=dtype,
        )
    elif model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        return AudioFlamingoModel(
            model_path=MODEL_PATHS[model_name], device=device, dtype=dtype,
        )
    elif model_name == "qwen25_omni":
        from models.qwen25_omni import Qwen25OmniModel
        return Qwen25OmniModel(
            model_path=MODEL_PATHS[model_name], device=device, dtype=dtype,
        )
    elif model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        return KimiAudioModel(
            model_path=MODEL_PATHS[model_name], device=device, dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def eval_audio_on_model(model, audio_path, prompt, device="cuda"):
    """Evaluate a single audio file on a loaded model.

    Returns the model's text output.
    """
    import soundfile as sf

    wav_np, sr = sf.read(audio_path)
    audio = torch.FloatTensor(wav_np).unsqueeze(0)  # [1, T]
    if sr != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SAMPLE_RATE)
    audio = audio.to(device)

    with torch.no_grad():
        output = model.generate(audio, prompt=prompt)

    return output


def opus_compress(wav_path, bitrate, out_path):
    """Compress a WAV file through Opus and decode back."""
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        env_bin = os.path.dirname(sys.executable)
        candidate = os.path.join(env_bin, "ffmpeg")
        if os.path.isfile(candidate):
            ffmpeg_bin = candidate
    if ffmpeg_bin is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        opus_path = os.path.join(tmpdir, "compressed.opus")
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", wav_path, "-c:a", "opus",
             "-strict", "-2", "-b:a", f"{bitrate}k", opus_path],
            capture_output=True, check=True,
        )
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", opus_path, "-ar", str(ENCODEC_SAMPLE_RATE), out_path],
            capture_output=True, check=True,
        )
    return out_path


def run_benchmark(args):
    """Run the black-box blending benchmark."""
    # Parse alpha values
    alphas = [float(a) for a in args.alpha.split(",")]

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

    # Commands
    commands = filter_commands(args.category, mode=args.prompt_mode, dataset=args.dataset)
    if args.commands:
        commands = {k: v for k, v in commands.items() if k in args.commands}

    opus_bitrates = [int(b) for b in args.opus_bitrates.split(",")]

    # Prompt
    prompt_mode = args.prompt_mode
    prompt_text = PROMPT_MODES[prompt_mode]["prompt"]

    total = len(music_names) * len(commands) * len(alphas)
    print(f"Black-box blend benchmark")
    print(f"  {len(music_names)} music x {len(commands)} commands x {len(alphas)} alphas = {total} experiments")
    print(f"  Blend mode: {args.blend_mode}")
    print(f"  Alphas: {alphas}")
    print(f"  Prompt mode: {prompt_mode} — \"{prompt_text}\"")
    print(f"  Target model: {args.target_model}")
    print(f"  Opus eval bitrates: {opus_bitrates}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        alpha_tag = "_".join(f"{a}" for a in alphas)
        music_tag = "_".join(music_names) if len(music_names) <= 3 else f"{len(music_names)}music"
        dir_name = f"benchmark_blackbox_{args.blend_mode}_{prompt_mode}_{music_tag}"
        output_dir = os.path.join("results_blackbox", dir_name)
    os.makedirs(output_dir, exist_ok=True)

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Per-model results dir
    model_subdir = args.target_model
    model_dir = os.path.join(output_dir, model_subdir)
    os.makedirs(model_dir, exist_ok=True)

    # Load completed experiments
    completed = set()
    for entry in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
        rec_path = os.path.join(model_dir, entry, "record.json")
        if os.path.isfile(rec_path):
            try:
                with open(rec_path) as f:
                    rec = json.load(f)
                completed.add((rec["music"], rec["command_name"], rec["alpha"]))
            except (json.JSONDecodeError, KeyError):
                pass

    if completed:
        print(f"Skipping {len(completed)} already-completed experiments")

    # Initialize blender (EnCodec only, no target model)
    print("\nInitializing EnCodec blender...")
    blender = LatentBlendAttacker(
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=not args.quiet,
    )

    # Initialize target model for evaluation
    print(f"Loading target model: {args.target_model}...")
    target_model = load_target_model(args.target_model)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "attack_type": "blackbox_blend",
                "target_model": args.target_model,
                "blend_mode": args.blend_mode,
                "alphas": alphas,
                "dataset": args.dataset,
                "music_files": music_names,
                "commands": {k: v["command"] for k, v in commands.items()},
                "targets": {k: v["target"] for k, v in commands.items()},
                "opus_bitrates": opus_bitrates,
                "prompt_mode": prompt_mode,
                "prompt_text": prompt_text,
            }, f, indent=2)

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

    done = len(completed)

    for mi, music_name in enumerate(music_names):
        music_wav = load_music_by_name(music_load_keys[mi])

        for alpha in alphas:
            # TTS cache: generate once per command, reuse across alphas would need
            # restructuring. Generate per command within the alpha loop.
            for ci, (cmd_name, cmd_info) in enumerate(commands.items()):
                cmd_text = cmd_info["command"]
                target_text = cmd_info["target"]

                if (music_name, cmd_name, alpha) in completed:
                    continue

                done += 1
                print(f"\n{'=' * 70}")
                print(f"[{done}/{total}] music={music_name}, cmd={cmd_name}, alpha={alpha}")
                print(f"  Target: \"{target_text[:80]}\"")
                print(f"{'=' * 70}")

                start_time = time.time()

                # Blend (no model needed)
                result = blender.blend(
                    music_wav,
                    target_text=target_text,
                    alpha=alpha,
                    blend_mode=args.blend_mode,
                    music_name=music_name,
                )

                # Save adversarial audio
                exp_key = f"{music_name}__{cmd_name}__alpha{alpha}"
                wav_path = blender.save_adversarial_wav(result, audio_dir, exp_key)

                # Evaluate direct
                direct_output = eval_audio_on_model(
                    target_model, wav_path, prompt_text,
                )
                direct_success = target_text.lower() in direct_output.lower()

                # Evaluate Opus variants
                opus_results = {}
                for bw in opus_bitrates:
                    opus_dir = os.path.join(audio_dir, f"opus{bw}")
                    os.makedirs(opus_dir, exist_ok=True)
                    opus_wav_path = os.path.join(opus_dir, f"{exp_key}.wav")

                    if not os.path.isfile(opus_wav_path):
                        opus_compress(wav_path, bw, opus_wav_path)

                    if os.path.isfile(opus_wav_path):
                        opus_output = eval_audio_on_model(
                            target_model, opus_wav_path, prompt_text,
                        )
                        opus_success = target_text.lower() in opus_output.lower()
                        opus_results[str(bw)] = {
                            "success": opus_success,
                            "output": opus_output,
                        }
                        # WER for transcription mode
                        if prompt_mode == "transcribe" and target_text:
                            opus_results[str(bw)]["wer"] = compute_wer(target_text, opus_output)

                duration = time.time() - start_time

                # Build record
                record = {
                    "music": music_name,
                    "command_name": cmd_name,
                    "command_text": cmd_text,
                    "target_text": target_text,
                    "category": get_category(cmd_name),
                    "alpha": alpha,
                    "blend_mode": args.blend_mode,
                    "direct": {
                        "success": direct_success,
                        "output": direct_output,
                        "snr_db": result.snr_db,
                        "latent_snr_db": result.latent_snr_db,
                    },
                    "opus": opus_results,
                    "prompt_mode": prompt_mode,
                    "prompt_text": prompt_text,
                    "duration_s": duration,
                }

                if prompt_mode == "transcribe" and target_text:
                    record["direct"]["wer"] = compute_wer(target_text, direct_output)

                all_results.append(record)

                # Save per-experiment record
                exp_record_dir = os.path.join(model_dir, exp_key)
                os.makedirs(exp_record_dir, exist_ok=True)
                with open(os.path.join(exp_record_dir, "record.json"), "w") as f:
                    json.dump(record, f, indent=2)

                # Incremental summary
                _save_blackbox_summary(summary_path, all_results, alphas, opus_bitrates)
                save_results_summary(output_dir, all_results, total=total)

                # Progress
                d_ok = "OK" if direct_success else "FAIL"
                opus_status = " | ".join(
                    f"opus{bw}={'OK' if opus_results.get(str(bw), {}).get('success') else 'FAIL'}"
                    for bw in opus_bitrates
                )
                print(f"  Result: direct={d_ok} | {opus_status} | "
                      f"SNR={result.snr_db:.1f}dB | {duration:.1f}s")
                print(f"  Output: {direct_output[:100]}")

    # Final summary
    _save_blackbox_summary(summary_path, all_results, alphas, opus_bitrates)
    _print_blackbox_summary(all_results, alphas, opus_bitrates)

    print(f"\nResults saved to: {output_dir}")


def _save_blackbox_summary(path, results, alphas, opus_bitrates):
    """Save summary with per-alpha stats."""
    n = len(results)
    if n == 0:
        stats = {}
    else:
        # Overall stats
        direct_ok = sum(1 for r in results if r["direct"]["success"])
        stats = {
            "total_experiments": n,
            "direct_success_rate": direct_ok / n,
            "direct_successes": direct_ok,
        }
        for bw in opus_bitrates:
            bw_key = str(bw)
            ok = sum(1 for r in results
                     if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            stats[f"opus_{bw}_success_rate"] = ok / n
            stats[f"opus_{bw}_successes"] = ok

        # Per-alpha stats
        by_alpha = {}
        for alpha in alphas:
            a_results = [r for r in results if r.get("alpha") == alpha]
            if not a_results:
                continue
            a_n = len(a_results)
            a_direct = sum(1 for r in a_results if r["direct"]["success"])
            entry = {
                "total": a_n,
                "direct_ok": a_direct,
                "direct_rate": a_direct / a_n,
            }
            for bw in opus_bitrates:
                bw_key = str(bw)
                ok = sum(1 for r in a_results
                         if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
                entry[f"opus_{bw}_ok"] = ok
                entry[f"opus_{bw}_rate"] = ok / a_n
            by_alpha[str(alpha)] = entry
        stats["by_alpha"] = by_alpha

        # Per-category stats
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "direct_ok": 0}
                for bw in opus_bitrates:
                    categories[cat][f"opus_{bw}_ok"] = 0
            categories[cat]["total"] += 1
            if r["direct"]["success"]:
                categories[cat]["direct_ok"] += 1
            for bw in opus_bitrates:
                bw_key = str(bw)
                if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"]:
                    categories[cat][f"opus_{bw}_ok"] += 1
        stats["by_category"] = categories

    summary = {"stats": stats, "results": results}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _print_blackbox_summary(results, alphas, opus_bitrates):
    """Print summary with per-alpha breakdown."""
    if not results:
        print("No results.")
        return

    n = len(results)
    direct_ok = sum(1 for r in results if r["direct"]["success"])

    print("\n" + "=" * 80)
    print("BLACK-BOX BLEND BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total: {n} experiments")
    print(f"Overall direct success: {direct_ok}/{n} ({100*direct_ok/n:.1f}%)")

    for bw in opus_bitrates:
        bw_key = str(bw)
        ok = sum(1 for r in results if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
        print(f"Overall Opus {bw}kbps: {ok}/{n} ({100*ok/n:.1f}%)")

    # Per-alpha table
    print(f"\n{'':─<80}")
    print("PER-ALPHA SUCCESS RATES")
    print(f"{'':─<80}")
    header = f"{'Alpha':<10} {'Direct':>12}"
    for bw in opus_bitrates:
        header += f" {'Opus'+str(bw):>12}"
    header += f" {'Avg SNR':>10}"
    print(header)
    print("-" * 80)

    for alpha in alphas:
        a_results = [r for r in results if r.get("alpha") == alpha]
        if not a_results:
            continue
        a_n = len(a_results)
        d_ok = sum(1 for r in a_results if r["direct"]["success"])
        avg_snr = sum(r["direct"]["snr_db"] for r in a_results) / a_n

        line = f"{alpha:<10.2f} {d_ok:>4}/{a_n:<4} ({100*d_ok/a_n:4.0f}%)"
        for bw in opus_bitrates:
            bw_key = str(bw)
            ok = sum(1 for r in a_results if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            line += f" {ok:>4}/{a_n:<4} ({100*ok/a_n:4.0f}%)"
        line += f" {avg_snr:>8.1f}dB"
        print(line)

    # Per-category table
    print(f"\n{'':─<80}")
    print("PER-CATEGORY SUCCESS RATES")
    print(f"{'':─<80}")
    header = f"{'Category':<22} {'Direct':>10}"
    for bw in opus_bitrates:
        header += f" {'Opus'+str(bw):>10}"
    print(header)
    print("-" * 80)

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        c_results = [r for r in results if r["category"] == cat]
        c_n = len(c_results)
        d_ok = sum(1 for r in c_results if r["direct"]["success"])
        line = f"{cat:<22} {d_ok:>3}/{c_n:<3} ({100*d_ok/c_n:4.0f}%)"
        for bw in opus_bitrates:
            bw_key = str(bw)
            ok = sum(1 for r in c_results if bw_key in r.get("opus", {}) and r["opus"][bw_key]["success"])
            line += f" {ok:>3}/{c_n:<3} ({100*ok/c_n:4.0f}%)"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Black-box latent blending benchmark"
    )

    # Scope
    parser.add_argument("--music", type=str, default=None,
                        help=f"Comma-separated music carriers")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--commands", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    # Blend params
    parser.add_argument("--alpha", type=str, default="0.3",
                        help="Comma-separated alpha values for blending (default: 0.3)")
    parser.add_argument("--blend-mode", type=str, default="interpolate",
                        choices=["interpolate", "additive"],
                        help="Blending strategy (default: interpolate)")

    # Evaluation
    parser.add_argument("--opus-bitrates", type=str,
                        default=",".join(str(b) for b in OPUS_EVAL_BITRATES))
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--dataset", type=str, default="ours",
                        choices=DATASET_CHOICES)

    # Convenience
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 music, 2 commands")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.quick:
        if not args.music:
            args.music = DEFAULT_MUSIC
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
