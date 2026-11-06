#!/usr/bin/env python3
"""Scan all results directories and print a summary table of attack results."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("${REPO_ROOT}")
RESULTS_DIRS = [
    BASE_DIR / "results_qa",
    BASE_DIR / "results_transcribe",
    BASE_DIR / "results_waveform",
]


def parse_benchmark_name(name):
    """Parse benchmark dir name like benchmark_qwen2_audio_eps0.5_empire_state_of_mind."""
    eps_match = re.search(r"eps([\d.]+)", name)
    eps = float(eps_match.group(1)) if eps_match else None
    prefix = name.split("_eps")[0].replace("benchmark_", "")
    after_eps = re.split(r"eps[\d.]+_", name)[1] if eps_match else ""
    return prefix, eps, after_eps


def get_category(command_name):
    parts = command_name.split("_")
    if len(parts) >= 2:
        return parts[0]
    return command_name


def load_records(benchmark_path):
    results = {}
    for item in sorted(benchmark_path.iterdir()):
        if item.is_dir() and item.name not in ("audio", "audio/16k"):
            model_name = item.name
            records = []
            for cmd_dir in sorted(item.iterdir()):
                if cmd_dir.is_dir():
                    record_file = cmd_dir / "record.json"
                    if record_file.exists():
                        try:
                            with open(record_file) as f:
                                rec = json.load(f)
                            rec["_command_name"] = rec.get("command_name", cmd_dir.name)
                            records.append(rec)
                        except (json.JSONDecodeError, KeyError):
                            pass
            if records:
                results[model_name] = records
    return results


def compute_stats(records):
    total = len(records)
    if total == 0:
        return None

    direct_success = sum(1 for r in records if r.get("direct", {}).get("success", False))
    opus64_success = sum(1 for r in records if r.get("opus", {}).get("64", {}).get("success", False))
    opus128_success = sum(1 for r in records if r.get("opus", {}).get("128", {}).get("success", False))

    has_opus64 = any(r.get("opus", {}).get("64") for r in records)
    has_opus128 = any(r.get("opus", {}).get("128") for r in records)

    snr_vals = [r["direct"]["snr_db"] for r in records if "direct" in r and "snr_db" in r.get("direct", {})]
    latent_snr_vals = [r["direct"]["latent_snr_db"] for r in records if "direct" in r and "latent_snr_db" in r.get("direct", {})]
    loss_vals = [r["direct"]["final_loss"] for r in records if "direct" in r and "final_loss" in r.get("direct", {})]

    return {
        "total": total,
        "direct_success": direct_success,
        "direct_rate": direct_success / total * 100,
        "opus64_success": opus64_success if has_opus64 else None,
        "opus64_rate": (opus64_success / total * 100) if has_opus64 else None,
        "opus128_success": opus128_success if has_opus128 else None,
        "opus128_rate": (opus128_success / total * 100) if has_opus128 else None,
        "avg_snr": sum(snr_vals) / len(snr_vals) if snr_vals else None,
        "avg_latent_snr": sum(latent_snr_vals) / len(latent_snr_vals) if latent_snr_vals else None,
        "avg_loss": sum(loss_vals) / len(loss_vals) if loss_vals else None,
    }


def fmt_pct(val, count=None, total=None):
    if val is None:
        return "  N/A  "
    if count is not None and total is not None:
        return f"{val:5.1f}% ({count}/{total})"
    return f"{val:5.1f}%"


def fmt_float(val):
    if val is None:
        return "  N/A"
    return f"{val:6.1f}"


def print_separator(width=145):
    print("=" * width)


def print_thin_separator(width=145):
    print("-" * width)


def main(show_category=True):
    all_benchmarks = []

    for results_dir in RESULTS_DIRS:
        if not results_dir.exists():
            continue
        mode = results_dir.name.replace("results_", "")

        for bench_dir in sorted(results_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            if not (bench_dir.name.startswith("benchmark_") or bench_dir.name.startswith("waveform_baseline_")):
                continue

            config_file = bench_dir / "config.json"
            if not config_file.exists():
                continue

            with open(config_file) as f:
                config = json.load(f)

            target_model, eps, music = parse_benchmark_name(bench_dir.name)
            model_records = load_records(bench_dir)

            all_benchmarks.append({
                "mode": mode,
                "target_model": target_model,
                "eps": eps,
                "music": music,
                "dir": bench_dir.name,
                "model_records": model_records,
                "config": config,
            })

    if not all_benchmarks:
        print("No benchmark results found.")
        return

    # ==================== OVERALL SUMMARY TABLE ====================
    print()
    print_separator()
    print(f"{'ATTACK RESULTS SUMMARY':^145}")
    print_separator()

    header = f"{'Mode':<12} {'Target Model':<17} {'Eps':<5} {'Music':<25} {'Eval Model':<17} {'N':>4} {'Direct':>17} {'Opus64':>17} {'Opus128':>17} {'AvgSNR':>7} {'LatSNR':>7} {'AvgLoss':>8}"
    print(header)
    print_thin_separator()

    for bench in all_benchmarks:
        first_model = True
        for model_name, records in sorted(bench["model_records"].items()):
            stats = compute_stats(records)
            if stats is None:
                continue

            if first_model:
                mode_str = bench["mode"]
                target_str = bench["target_model"]
                eps_str = f"{bench['eps']}"
                music_str = bench["music"][:24]
                first_model = False
            else:
                mode_str = ""
                target_str = ""
                eps_str = ""
                music_str = ""

            direct_str = fmt_pct(stats["direct_rate"], stats["direct_success"], stats["total"])
            opus64_str = fmt_pct(stats["opus64_rate"], stats["opus64_success"], stats["total"]) if stats["opus64_rate"] is not None else "     N/A      "
            opus128_str = fmt_pct(stats["opus128_rate"], stats["opus128_success"], stats["total"]) if stats["opus128_rate"] is not None else "     N/A      "

            print(f"{mode_str:<12} {target_str:<17} {eps_str:<5} {music_str:<25} {model_name:<17} {stats['total']:>4} {direct_str:>17} {opus64_str:>17} {opus128_str:>17} {fmt_float(stats['avg_snr']):>7} {fmt_float(stats['avg_latent_snr']):>7} {fmt_float(stats['avg_loss']):>8}")

        if not first_model:
            print_thin_separator()

    # ==================== PER-CATEGORY BREAKDOWN ====================
    if not show_category:
        print()
        return

    print()
    print()
    print_separator(120)
    print(f"{'PER-CATEGORY BREAKDOWN':^120}")
    print_separator(120)

    for bench in all_benchmarks:
        for model_name, records in sorted(bench["model_records"].items()):
            cat_records = defaultdict(list)
            for rec in records:
                cat = rec.get("category", None)
                if cat is None:
                    cat = get_category(rec.get("_command_name", "unknown"))
                cat_records[cat].append(rec)

            if len(cat_records) <= 1:
                continue

            print()
            title = f"{bench['mode']} | target={bench['target_model']} | eps={bench['eps']} | music={bench['music']} | eval={model_name}"
            print(f"  {title}")
            print(f"  {'Category':<20} {'N':>4} {'Direct':>17} {'Opus64':>17} {'Opus128':>17} {'AvgSNR':>7} {'LatSNR':>7}")
            print(f"  {'-'*95}")

            for cat in sorted(cat_records.keys()):
                recs = cat_records[cat]
                stats = compute_stats(recs)
                if stats is None:
                    continue
                direct_str = fmt_pct(stats["direct_rate"], stats["direct_success"], stats["total"])
                opus64_str = fmt_pct(stats["opus64_rate"], stats["opus64_success"], stats["total"]) if stats["opus64_rate"] is not None else "     N/A      "
                opus128_str = fmt_pct(stats["opus128_rate"], stats["opus128_success"], stats["total"]) if stats["opus128_rate"] is not None else "     N/A      "
                print(f"  {cat:<20} {stats['total']:>4} {direct_str:>17} {opus64_str:>17} {opus128_str:>17} {fmt_float(stats['avg_snr']):>7} {fmt_float(stats['avg_latent_snr']):>7}")

            stats = compute_stats(records)
            print(f"  {'-'*95}")
            direct_str = fmt_pct(stats["direct_rate"], stats["direct_success"], stats["total"])
            opus64_str = fmt_pct(stats["opus64_rate"], stats["opus64_success"], stats["total"]) if stats["opus64_rate"] is not None else "     N/A      "
            opus128_str = fmt_pct(stats["opus128_rate"], stats["opus128_success"], stats["total"]) if stats["opus128_rate"] is not None else "     N/A      "
            print(f"  {'TOTAL':<20} {stats['total']:>4} {direct_str:>17} {opus64_str:>17} {opus128_str:>17} {fmt_float(stats['avg_snr']):>7} {fmt_float(stats['avg_latent_snr']):>7}")

    print()


if __name__ == "__main__":
    import sys
    show_category = "--category" in sys.argv or "-c" in sys.argv
    main(show_category=show_category)
