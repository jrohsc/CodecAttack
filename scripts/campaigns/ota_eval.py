#!/usr/bin/env python3
"""
SERVER SCRIPT — Run this on the GPU server.

Evaluates a recorded WAV file (captured over-the-air) on an Audio LLM.

Usage:
    # Single file
    python ota_eval.py --audio ota_recording.wav --target "Turn left at the next intersection" \
                       --model qwen2_audio

    # Compare original vs OTA recording
    python ota_eval.py --audio ota_recording.wav --original adversarial.wav \
                       --target "Turn left at the next intersection" --model qwen2_audio

    # Test multiple models
    python ota_eval.py --audio ota_recording.wav --target "Turn left" \
                       --model qwen2_audio qwen25_omni

    # Batch: evaluate all recordings in a directory
    python ota_eval.py --audio-dir ota_recordings/ --targets targets.json --model qwen2_audio
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WHISPER_INJECT_ROOT, MODEL_PATHS,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    PROMPT_MODES, DEFAULT_PROMPT_MODE,
)
sys.path.insert(0, WHISPER_INJECT_ROOT)


def load_model(model_name: str, device: str = "cuda"):
    """Load an Audio LLM by name."""
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        model = Qwen2AudioModel(
            model_path=MODEL_PATHS["qwen2_audio"],
            device=device, dtype=torch.bfloat16,
        )
        return model, TARGET_SAMPLE_RATE

    elif model_name == "qwen25_omni":
        from models.qwen25_omni import Qwen25OmniModel
        model = Qwen25OmniModel(
            model_path=MODEL_PATHS["qwen25_omni"],
            device=device, dtype=torch.bfloat16,
        )
        return model, TARGET_SAMPLE_RATE

    elif model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        model = AudioFlamingoModel(
            model_path=MODEL_PATHS["audio_flamingo"],
            device=device, dtype=torch.bfloat16,
        )
        return model, model.SAMPLE_RATE

    elif model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        model = KimiAudioModel(
            model_path=MODEL_PATHS.get("kimi_audio"),
            device=device,
        )
        return model, model.SAMPLE_RATE

    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_wav(model, model_name, wav_path, model_sr, prompt, device="cuda"):
    """Load a WAV and run model inference."""
    audio_np, sr = sf.read(wav_path)

    # Resample to model's expected rate
    if sr != model_sr:
        audio_tensor = torchaudio.functional.resample(
            torch.FloatTensor(audio_np).unsqueeze(0), sr, model_sr
        )
        audio_np = audio_tensor.squeeze(0).numpy()

    if model_name == "kimi_audio":
        # Kimi needs file path at correct SR
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, model_sr)
            output = model.generate_from_path(f.name, prompt=prompt)
            os.unlink(f.name)
    else:
        wav_tensor = torch.FloatTensor(audio_np).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.generate(wav_tensor, prompt=prompt)

    return output


def compute_snr(original_np, recorded_np):
    """Compute SNR between original and recorded audio."""
    min_len = min(len(original_np), len(recorded_np))
    orig = original_np[:min_len]
    rec = recorded_np[:min_len]
    noise = rec - orig
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OTA-recorded audio on Audio LLMs")

    parser.add_argument("--audio", type=str, help="Path to OTA-recorded WAV file")
    parser.add_argument("--audio-dir", type=str, help="Directory of OTA-recorded WAV files")
    parser.add_argument("--original", type=str, default=None,
                        help="Path to original adversarial WAV (for comparison)")
    parser.add_argument("--target", type=str, help="Target text to check")
    parser.add_argument("--targets", type=str, default=None,
                        help="JSON file mapping filenames to target texts")
    parser.add_argument("--model", type=str, nargs="+", default=["qwen2_audio"],
                        choices=["qwen2_audio", "qwen25_omni", "audio_flamingo", "kimi_audio"],
                        help="Model(s) to evaluate")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE,
                        help="Prompt mode (transcribe/qa)")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()

    prompt = args.prompt or PROMPT_MODES[args.prompt_mode]["prompt"]

    # Collect WAV files to evaluate
    wav_files = []
    if args.audio:
        wav_files.append(args.audio)
    elif args.audio_dir:
        for f in sorted(os.listdir(args.audio_dir)):
            if f.endswith(".wav"):
                wav_files.append(os.path.join(args.audio_dir, f))
    else:
        parser.print_help()
        return

    # Load targets map if provided
    targets_map = {}
    if args.targets:
        with open(args.targets) as f:
            targets_map = json.load(f)

    all_results = []

    for model_name in args.model:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"Prompt: \"{prompt}\"")
        print(f"{'='*70}")

        model, model_sr = load_model(model_name)

        # Evaluate original (digital baseline) if provided
        if args.original and len(wav_files) == 1:
            print(f"\n--- Digital Baseline (no OTA) ---")
            bl_output = evaluate_wav(model, model_name, args.original, model_sr, prompt)
            target = args.target or targets_map.get(os.path.basename(args.original), "")
            bl_match = target.lower() in bl_output.lower() if target else None
            print(f"Output: {bl_output}")
            if target:
                print(f"Target: {target}")
                print(f"Match:  {'YES' if bl_match else 'NO'}")

        # Evaluate OTA recordings
        successes = 0
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            target = args.target or targets_map.get(fname, "")

            print(f"\n--- {fname} ---")
            start = time.time()
            output = evaluate_wav(model, model_name, wav_path, model_sr, prompt)
            elapsed = time.time() - start

            match = target.lower() in output.lower() if target else None
            if match:
                successes += 1

            print(f"Output: {output}")
            if target:
                print(f"Target: {target}")
                print(f"Match:  {'YES' if match else 'NO'}")
            print(f"Time:   {elapsed:.1f}s")

            # SNR if original provided
            if args.original:
                orig_np, orig_sr = sf.read(args.original)
                rec_np, rec_sr = sf.read(wav_path)
                # Resample both to same rate for comparison
                if orig_sr != rec_sr:
                    orig_t = torchaudio.functional.resample(
                        torch.FloatTensor(orig_np).unsqueeze(0), orig_sr, rec_sr
                    )
                    orig_np = orig_t.squeeze(0).numpy()
                snr = compute_snr(orig_np, rec_np)
                print(f"SNR:    {snr:.1f} dB")

            result = {
                "file": wav_path,
                "model": model_name,
                "output": output,
                "target": target,
                "match": match,
                "duration_s": elapsed,
            }
            all_results.append(result)

        if len(wav_files) > 1 and args.target:
            n = len(wav_files)
            print(f"\n--- Summary: {model_name} ---")
            print(f"OTA Success: {successes}/{n} ({100*successes/n:.1f}%)")

        # Free GPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved: {args.save}")


if __name__ == "__main__":
    main()
