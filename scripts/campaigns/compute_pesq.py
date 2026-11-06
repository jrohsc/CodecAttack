"""
Compute PESQ (Perceptual Evaluation of Speech Quality) scores for all
adversarial audio files across results_qa, results_transcribe, and results_waveform.

PESQ-WB (wideband, 16kHz) measures perceptual quality degradation.
Range: -0.5 to 4.5 (higher = better quality, 4.5 = transparent).

Usage:
    python compute_pesq.py                          # All results dirs
    python compute_pesq.py --dirs results_qa        # Specific dir
    python compute_pesq.py --dirs results_waveform  # Waveform baseline only
    python compute_pesq.py --summary-only           # Just print existing results
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import soundfile as sf
from pesq import pesq

# PESQ wideband operates at 16kHz
PESQ_SR = 16000
ENCODEC_SR = 24000
NUM_WORKERS = min(cpu_count(), 12)  # cap at 12 to avoid memory pressure


def resample_to_16k(audio, orig_sr):
    """Resample audio to 16kHz for PESQ using scipy."""
    if orig_sr == PESQ_SR:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, PESQ_SR)
    up = PESQ_SR // g
    down = orig_sr // g
    return resample_poly(audio, up, down)


def load_reference_music(music_name, duration=10.0):
    """Load reference music at 24kHz, return as numpy array."""
    import librosa
    music_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "music")

    # Try direct name match
    for ext in [".mp3", ".wav", ".flac"]:
        path = os.path.join(music_dir, music_name + ext)
        if os.path.isfile(path):
            wav, _ = librosa.load(path, sr=ENCODEC_SR, mono=True, duration=duration)
            target_len = int(duration * ENCODEC_SR)
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            elif len(wav) > target_len:
                wav = wav[:target_len]
            return wav

    return None


def compute_pesq_for_pair(ref_24k, adv_path):
    """Compute PESQ-WB between reference and adversarial audio."""
    adv_24k, sr = sf.read(adv_path)
    if adv_24k.ndim > 1:
        adv_24k = adv_24k[:, 0]

    # Match lengths
    min_len = min(len(ref_24k), len(adv_24k))
    ref = ref_24k[:min_len]
    adv = adv_24k[:min_len]

    # Resample to 16kHz
    ref_16k = resample_to_16k(ref, sr)
    adv_16k = resample_to_16k(adv, sr)

    # Match lengths after resampling
    min_len_16k = min(len(ref_16k), len(adv_16k))
    ref_16k = ref_16k[:min_len_16k]
    adv_16k = adv_16k[:min_len_16k]

    try:
        score = pesq(PESQ_SR, ref_16k, adv_16k, 'wb')
        return score
    except Exception as e:
        print(f"  PESQ error for {adv_path}: {e}")
        return None


def _worker_pesq(args):
    """Multiprocessing worker: compute PESQ for one (ref_path_info, adv_path) pair."""
    music_name, adv_path = args
    ref_24k = _load_ref_cached(music_name)
    if ref_24k is None:
        return (os.path.basename(adv_path), None)
    score = compute_pesq_for_pair(ref_24k, adv_path)
    return (os.path.basename(adv_path), score)


# Module-level cache for multiprocessing workers (each process gets its own copy)
_ref_cache_worker = {}

def _load_ref_cached(music_name):
    if music_name not in _ref_cache_worker:
        _ref_cache_worker[music_name] = load_reference_music(music_name)
    return _ref_cache_worker[music_name]


def extract_music_name(wav_filename):
    """Extract music name from wav filename like 'calm_1__saycan_000.wav'."""
    base = os.path.splitext(wav_filename)[0]
    # Split on double underscore: music_name__command_name
    parts = base.split("__")
    if len(parts) >= 2:
        return parts[0]
    return base


def process_benchmark_dir(benchmark_dir, ref_cache, pool=None):
    """Process a single benchmark directory (latent or waveform)."""
    audio_dir = os.path.join(benchmark_dir, "audio")
    if not os.path.isdir(audio_dir):
        return {}

    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_files:
        return {}

    # Build work items: (music_name, adv_path)
    work_items = []
    for wf in wav_files:
        music_name = extract_music_name(os.path.basename(wf))
        work_items.append((music_name, wf))

    # Run in parallel
    if pool is not None:
        raw_results = pool.map(_worker_pesq, work_items)
    else:
        raw_results = [_worker_pesq(item) for item in work_items]

    results = {}
    for key, score in raw_results:
        if score is not None:
            results[key] = score

    return results


def update_records_with_pesq(benchmark_dir, pesq_scores):
    """Write PESQ scores into existing record.json files."""
    updated = 0
    # Find all model subdirs containing record.json
    for model_dir in os.listdir(benchmark_dir):
        model_path = os.path.join(benchmark_dir, model_dir)
        if not os.path.isdir(model_path) or model_dir in ("audio", "audio_16k"):
            continue
        for exp_dir in os.listdir(model_path):
            rec_path = os.path.join(model_path, exp_dir, "record.json")
            if not os.path.isfile(rec_path):
                continue
            # Match to wav file
            wav_key = exp_dir + ".wav"
            if wav_key in pesq_scores:
                with open(rec_path) as f:
                    record = json.load(f)
                record["pesq_wb"] = round(pesq_scores[wav_key], 3)
                with open(rec_path, "w") as f:
                    json.dump(record, f, indent=2)
                updated += 1
    return updated


def process_results_dir(results_dir, ref_cache, pool=None):
    """Process all benchmark subdirs within a results directory."""
    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    dir_name = os.path.basename(results_dir)
    print(f"\n{'='*70}")
    print(f"Processing: {dir_name}")
    print(f"{'='*70}")

    all_scores = {}

    for bench_name in sorted(os.listdir(results_dir)):
        bench_path = os.path.join(results_dir, bench_name)
        if not os.path.isdir(bench_path):
            continue

        print(f"\n  {bench_name}...")
        scores = process_benchmark_dir(bench_path, ref_cache, pool=pool)
        if not scores:
            print(f"    No audio files found")
            continue

        vals = list(scores.values())
        print(f"    {len(vals)} files, PESQ-WB: mean={np.mean(vals):.3f}, "
              f"std={np.std(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")

        # Update record.json files
        n_updated = update_records_with_pesq(bench_path, scores)
        if n_updated:
            print(f"    Updated {n_updated} record.json files")

        all_scores[bench_name] = {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3),
            "median": round(float(np.median(vals)), 3),
        }

    # Save summary
    if all_scores:
        summary_path = os.path.join(results_dir, "pesq_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    return all_scores


def print_summary_table(all_results):
    """Print a formatted summary table."""
    print(f"\n{'='*90}")
    print(f"{'PESQ-WB SUMMARY':^90}")
    print(f"{'='*90}")
    print(f"{'Directory':<60} {'N':>5} {'Mean':>7} {'Std':>6} {'Min':>6} {'Max':>6}")
    print(f"{'-'*90}")

    for results_dir, benchmarks in sorted(all_results.items()):
        print(f"\n  {results_dir}:")
        for bench, stats in sorted(benchmarks.items()):
            short = bench[:58]
            print(f"    {short:<56} {stats['n']:>5} {stats['mean']:>7.3f} {stats['std']:>6.3f} "
                  f"{stats['min']:>6.3f} {stats['max']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compute PESQ-WB scores for adversarial audio")
    parser.add_argument("--dirs", nargs="*", default=None,
                        help="Result directories to process (default: all)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print existing PESQ summaries")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))

    # Default: all results dirs
    if args.dirs:
        results_dirs = [os.path.join(project_root, d) for d in args.dirs]
    else:
        results_dirs = [
            os.path.join(project_root, d)
            for d in ["results_qa", "results_transcribe", "results_waveform"]
            if os.path.isdir(os.path.join(project_root, d))
        ]

    if args.summary_only:
        all_results = {}
        for rd in results_dirs:
            summary_path = os.path.join(rd, "pesq_summary.json")
            if os.path.isfile(summary_path):
                with open(summary_path) as f:
                    all_results[os.path.basename(rd)] = json.load(f)
        if all_results:
            print_summary_table(all_results)
        else:
            print("No existing PESQ summaries found. Run without --summary-only first.")
        return

    ref_cache = {}
    all_results = {}

    print(f"Using {NUM_WORKERS} worker processes")
    with Pool(NUM_WORKERS) as pool:
        for rd in results_dirs:
            scores = process_results_dir(rd, ref_cache, pool=pool)
            if scores:
                all_results[os.path.basename(rd)] = scores

    if all_results:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
