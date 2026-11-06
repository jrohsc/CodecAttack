#!/usr/bin/env python3
"""Compute average WER across all benchmark results (latent QA + waveform baseline)."""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Reuse existing WER computation
sys.path.insert(0, str(Path(__file__).parent))
from config import compute_wer

PROJECT_DIR = Path(__file__).parent
MODELS = ("qwen2_audio", "audio_flamingo", "qwen25_omni", "kimi_audio")

WER_THRESHOLD = 0.3
KEYS = ["direct", "opus_64", "opus_128"]
SNR_KEYS = ["snr_db", "opus_64_snr", "opus_128_snr"]
ALL_KEYS = KEYS + ["snr_db", "latent_snr_db", "opus_64_snr", "opus_128_snr"]


def _new_bucket():
    return {k: [] for k in ALL_KEYS}


def collect_wer_stats(results_dir, prefix):
    """Walk all record.json files and compute WER per experiment.

    Args:
        results_dir: Path to results directory (e.g. results_qa/ or results_waveform/)
        prefix: Directory name prefix (e.g. "benchmark_" or "waveform_baseline_")

    Returns:
        model -> eps -> music -> {direct: [...], opus_64: [...], ...}
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}

    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(_new_bucket)))

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith(prefix):
            continue

        # Find model subdir
        model_name = None
        for subdir in exp_dir.iterdir():
            if subdir.is_dir() and subdir.name in MODELS:
                model_name = subdir.name
                break
        if model_name is None:
            continue

        # Parse eps and music from dir name
        # Remove prefix, then find model name in remainder
        remainder = exp_dir.name[len(prefix):]
        # For waveform: {model}_{mode}_eps{X}_{music}_saycan
        # For latent:   {model}_eps{X}_{music}_saycan
        after_model = remainder[len(model_name) + 1:]  # skip "model_"
        # waveform has mode before eps, latent doesn't
        parts = after_model.split("_")
        # Find the eps part
        eps_idx = next(i for i, p in enumerate(parts) if p.startswith("eps"))
        eps_str = parts[eps_idx].replace("eps", "")
        # Music is everything between eps and "saycan"
        music = "_".join(parts[eps_idx + 1:-1])

        model_dir = exp_dir / model_name
        if not model_dir.is_dir():
            continue

        for cmd_dir in sorted(model_dir.iterdir()):
            if not cmd_dir.is_dir():
                continue
            record_path = cmd_dir / "record.json"
            if not record_path.exists():
                continue

            with open(record_path) as f:
                rec = json.load(f)

            target = rec.get("target_text", "")
            if not target:
                continue

            # Direct
            direct_data = rec.get("direct", {})
            direct_out = direct_data.get("output", "")
            if direct_out:
                wer = compute_wer(target, direct_out)
                stats[model_name][eps_str][music]["direct"].append(wer)
                if "snr_db" in direct_data:
                    stats[model_name][eps_str][music]["snr_db"].append(direct_data["snr_db"])
                if "latent_snr_db" in direct_data:
                    stats[model_name][eps_str][music]["latent_snr_db"].append(direct_data["latent_snr_db"])

            # Opus
            for bw in ("64", "128"):
                opus_data = rec.get("opus", {}).get(bw, {})
                opus_out = opus_data.get("output", "")
                if opus_out:
                    wer = compute_wer(target, opus_out)
                    stats[model_name][eps_str][music][f"opus_{bw}"].append(wer)
                    if "snr_db" in opus_data:
                        stats[model_name][eps_str][music][f"opus_{bw}_snr"].append(opus_data["snr_db"])

    return stats


def _success_rate(wer_list):
    """Fraction of samples with WER < threshold."""
    if not wer_list:
        return float('nan')
    return sum(1 for w in wer_list if w < WER_THRESHOLD) / len(wer_list)


def _avg(lst):
    return sum(lst) / len(lst) if lst else float('nan')


def _fmt_snr(lst):
    """Format average SNR, or '-' if empty."""
    if not lst:
        return "   -"
    return f"{_avg(lst):>4.1f}"


def print_table(stats, title=""):
    """Print results as a formatted table."""
    if title:
        print(f"\n{'#' * 80}")
        print(f"# {title}")
        print(f"{'#' * 80}")

    W = 150
    print("=" * W)
    print(f"{'':>57}"
          f" {'--- Direct ---':>18}"
          f"  {'--- Opus 64 ---':>18}"
          f"  {'--- Opus 128 ---':>18}"
          f"  {'-- SNR (dB) --':>18}")
    print(f"{'Model':<20} {'Eps':<6} {'Music':<25} {'N':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'Wave':>5} {'Latent':>6}")
    print("=" * W)

    model_agg = defaultdict(_new_bucket)
    model_eps_agg = defaultdict(_new_bucket)

    for model in sorted(stats):
        for eps in sorted(stats[model], key=float):
            for music in sorted(stats[model][eps]):
                d = stats[model][eps][music]
                n = len(d["direct"])

                print(f"{model:<20} {eps:<6} {music:<25} {n:>4}"
                      f"  {_avg(d['direct']):>6.3f} {_success_rate(d['direct'])*100:>5.1f} {_fmt_snr(d['snr_db'])}"
                      f"  {_avg(d['opus_64']):>6.3f} {_success_rate(d['opus_64'])*100:>5.1f} {_fmt_snr(d['opus_64_snr'])}"
                      f"  {_avg(d['opus_128']):>6.3f} {_success_rate(d['opus_128'])*100:>5.1f} {_fmt_snr(d['opus_128_snr'])}"
                      f"  {_fmt_snr(d['snr_db']):>5} {_fmt_snr(d['latent_snr_db']):>6}")

                for k in ALL_KEYS:
                    model_agg[model][k].extend(d[k])
                    model_eps_agg[(model, eps)][k].extend(d[k])

        print("-" * W)

    # Per-model, per-eps summary
    print(f"\n{'=' * 120}")
    print(f"SUMMARY: Average WER, Success Rate (WER < {WER_THRESHOLD}), and SNR per Model x Epsilon")
    print("=" * 120)
    print(f"{'Model':<20} {'Eps':<6} {'N':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'Wave':>5} {'Latent':>6}")
    print(f"{'':>32}"
          f"  {'--- Direct ---':>16}"
          f"  {'--- Opus 64 ---':>16}"
          f"  {'--- Opus 128 ---':>16}"
          f"  {'-- SNR (dB) --':>12}")
    print("-" * 120)

    for (model, eps) in sorted(model_eps_agg):
        d = model_eps_agg[(model, eps)]
        n = len(d["direct"])
        print(f"{model:<20} {eps:<6} {n:>4}"
              f"  {_avg(d['direct']):>6.3f} {_success_rate(d['direct'])*100:>5.1f} {_fmt_snr(d['snr_db'])}"
              f"  {_avg(d['opus_64']):>6.3f} {_success_rate(d['opus_64'])*100:>5.1f} {_fmt_snr(d['opus_64_snr'])}"
              f"  {_avg(d['opus_128']):>6.3f} {_success_rate(d['opus_128'])*100:>5.1f} {_fmt_snr(d['opus_128_snr'])}"
              f"  {_fmt_snr(d['snr_db']):>5} {_fmt_snr(d['latent_snr_db']):>6}")

    # Per-model summary
    print(f"\n{'=' * 110}")
    print(f"SUMMARY: Average WER, Success Rate (WER < {WER_THRESHOLD}), and SNR per Model (all eps, all music)")
    print("=" * 110)
    print(f"{'Model':<20} {'N':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'WER':>6} {'SR%':>5} {'SNR':>4}"
          f"  {'Wave':>5} {'Latent':>6}")
    print(f"{'':>26}"
          f"  {'--- Direct ---':>16}"
          f"  {'--- Opus 64 ---':>16}"
          f"  {'--- Opus 128 ---':>16}"
          f"  {'-- SNR (dB) --':>12}")
    print("-" * 110)

    for model in sorted(model_agg):
        d = model_agg[model]
        n = len(d["direct"])
        print(f"{model:<20} {n:>4}"
              f"  {_avg(d['direct']):>6.3f} {_success_rate(d['direct'])*100:>5.1f} {_fmt_snr(d['snr_db'])}"
              f"  {_avg(d['opus_64']):>6.3f} {_success_rate(d['opus_64'])*100:>5.1f} {_fmt_snr(d['opus_64_snr'])}"
              f"  {_avg(d['opus_128']):>6.3f} {_success_rate(d['opus_128'])*100:>5.1f} {_fmt_snr(d['opus_128_snr'])}"
              f"  {_fmt_snr(d['snr_db']):>5} {_fmt_snr(d['latent_snr_db']):>6}")


if __name__ == "__main__":
    # Latent-space QA results
    qa_stats = collect_wer_stats(PROJECT_DIR / "results_qa", "benchmark_")
    if qa_stats:
        print_table(qa_stats, title="LATENT-SPACE ATTACK (QA mode)")

    # Waveform baseline results
    wf_stats = collect_wer_stats(PROJECT_DIR / "results_waveform", "waveform_baseline_")
    if wf_stats:
        print_table(wf_stats, title="WAVEFORM BASELINE (transcribe mode)")

    if not qa_stats and not wf_stats:
        print("No results found.")
        sys.exit(1)
