"""
Quick verification: load each adversarial WAV and run Qwen2-Audio on it.
This confirms that the model actually outputs the target text.

Usage:
    python verify_results.py
    python verify_results.py --result_dir results/humanoid_calm_1_20260206_054757
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import torch
import soundfile as sf
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'whisper-inject-v2-fork'))

from config import MODEL_PATHS, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE, RESULTS_DIR


def verify_single(model, adv_wav_path, metrics_path):
    """Run Qwen2-Audio on a single adversarial WAV and compare to saved metrics."""
    # Load metrics
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Load adversarial audio
    audio, sr = sf.read(adv_wav_path)
    audio_t = torch.FloatTensor(audio).unsqueeze(0)  # [1, T]

    # Resample to 16kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        audio_t = torchaudio.functional.resample(audio_t, sr, TARGET_SAMPLE_RATE)

    audio_t = audio_t.to(model.device)

    # Run model
    with torch.no_grad():
        output = model.generate(audio_t)

    # Compare
    target = metrics['target_text']
    saved_output = metrics['adversarial_output']
    match = target.lower() in output.lower()

    return {
        'target_text': target,
        'saved_output': saved_output,
        'live_output': output,
        'match': match,
        'music_name': metrics.get('music_name', ''),
        'snr_db': metrics.get('snr_db', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='Verify adversarial attack results')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Specific result directory to verify (default: all)')
    args = parser.parse_args()

    # Load model
    print("Loading Qwen2-Audio model...")
    from models.qwen2_audio import Qwen2AudioModel
    model = Qwen2AudioModel(
        model_path=MODEL_PATHS['qwen2_audio'],
        device='cuda',
        dtype=torch.bfloat16,
    )
    print("Model loaded.\n")

    # Find result directories
    results_root = Path(RESULTS_DIR)
    if args.result_dir:
        dirs = [Path(args.result_dir)]
    else:
        dirs = sorted(results_root.rglob('*_metrics.json'))
        dirs = [p.parent for p in dirs]
        dirs = list(dict.fromkeys(dirs))  # deduplicate preserving order

    print(f"{'='*80}")
    print(f"{'Music':<22s} {'Target':<28s} {'Match':>6s}  Live Output")
    print(f"{'='*80}")

    total, passed = 0, 0
    for d in dirs:
        metrics_files = sorted(d.glob('*_metrics.json'))
        for mf in metrics_files:
            adv_wav = str(mf).replace('_metrics.json', '_adversarial.wav')
            if not os.path.exists(adv_wav):
                print(f"  [SKIP] No adversarial WAV for {mf.name}")
                continue

            result = verify_single(model, adv_wav, str(mf))
            total += 1
            if result['match']:
                passed += 1

            status = 'YES' if result['match'] else 'NO'
            print(f"{result['music_name']:<22s} {result['target_text'][:28]:<28s} "
                  f"{status:>6s}  {result['live_output'][:60]}")

    print(f"\n{'='*80}")
    print(f"Verified: {passed}/{total} attacks confirmed successful")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
