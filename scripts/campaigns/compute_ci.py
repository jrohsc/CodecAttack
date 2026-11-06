"""
Compute 95% Wilson confidence intervals on all ASR results.

Wilson CI is preferred over normal approximation for proportions,
especially with extreme success rates (near 0% or 100%).

Usage:
    python compute_ci.py                    # All results dirs
    python compute_ci.py --dirs results_qa  # Specific dir
"""

import argparse
import json
import math
import os
from collections import defaultdict


def wilson_ci(successes, total, z=1.96):
    """Compute Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds as percentages.
    """
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    lower = max(0.0, center - spread) * 100
    upper = min(1.0, center + spread) * 100
    return (round(lower, 1), round(upper, 1))


def load_all_records(benchmark_dir):
    """Load all record.json files from a benchmark directory."""
    records = []
    for root, dirs, files in os.walk(benchmark_dir):
        if "record.json" in files:
            rec_path = os.path.join(root, "record.json")
            try:
                with open(rec_path) as f:
                    rec = json.load(f)
                records.append(rec)
            except (json.JSONDecodeError, KeyError):
                pass
    return records


def compute_ci_for_records(records):
    """Compute CIs for a list of records."""
    n = len(records)
    if n == 0:
        return None

    direct_succ = sum(1 for r in records if r.get("direct", {}).get("success", False))

    opus64_records = [r for r in records if "64" in r.get("opus", {})]
    opus64_succ = sum(1 for r in opus64_records if r["opus"]["64"].get("success", False))
    n_opus64 = len(opus64_records)

    opus128_records = [r for r in records if "128" in r.get("opus", {})]
    opus128_succ = sum(1 for r in opus128_records if r["opus"]["128"].get("success", False))
    n_opus128 = len(opus128_records)

    # SNR stats
    snrs = [r["direct"]["snr_db"] for r in records if "snr_db" in r.get("direct", {})]
    avg_snr = sum(snrs) / len(snrs) if snrs else None

    result = {
        "n": n,
        "direct": {
            "successes": direct_succ,
            "rate_pct": round(direct_succ / n * 100, 1),
            "ci_95": wilson_ci(direct_succ, n),
        },
    }

    if n_opus64 > 0:
        result["opus64"] = {
            "n": n_opus64,
            "successes": opus64_succ,
            "rate_pct": round(opus64_succ / n_opus64 * 100, 1),
            "ci_95": wilson_ci(opus64_succ, n_opus64),
        }

    if n_opus128 > 0:
        result["opus128"] = {
            "n": n_opus128,
            "successes": opus128_succ,
            "rate_pct": round(opus128_succ / n_opus128 * 100, 1),
            "ci_95": wilson_ci(opus128_succ, n_opus128),
        }

    if avg_snr is not None:
        result["avg_snr_db"] = round(avg_snr, 1)

    return result


def process_results_dir(results_dir):
    """Process all benchmark subdirectories."""
    if not os.path.isdir(results_dir):
        print(f"Not found: {results_dir}")
        return {}

    dir_name = os.path.basename(results_dir)
    print(f"\n{'=' * 80}")
    print(f"  {dir_name}")
    print(f"{'=' * 80}")

    all_ci = {}

    for bench_name in sorted(os.listdir(results_dir)):
        bench_path = os.path.join(results_dir, bench_name)
        if not os.path.isdir(bench_path):
            continue

        records = load_all_records(bench_path)
        if not records:
            continue

        ci = compute_ci_for_records(records)
        if ci is None:
            continue

        all_ci[bench_name] = ci

        # Print
        d = ci["direct"]
        line = f"  {bench_name:<60} N={ci['n']:>3}  Direct: {d['rate_pct']:5.1f}% [{d['ci_95'][0]:.1f}, {d['ci_95'][1]:.1f}]"
        if "opus64" in ci:
            o = ci["opus64"]
            line += f"  Opus64: {o['rate_pct']:5.1f}% [{o['ci_95'][0]:.1f}, {o['ci_95'][1]:.1f}]"
        if "opus128" in ci:
            o = ci["opus128"]
            line += f"  Opus128: {o['rate_pct']:5.1f}% [{o['ci_95'][0]:.1f}, {o['ci_95'][1]:.1f}]"
        print(line)

    # Save
    if all_ci:
        out_path = os.path.join(results_dir, "ci_summary.json")
        with open(out_path, "w") as f:
            json.dump(all_ci, f, indent=2)
        print(f"\n  Saved to {out_path}")

    return all_ci


def main():
    parser = argparse.ArgumentParser(description="Compute 95% Wilson CIs on ASR results")
    parser.add_argument("--dirs", nargs="*", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))

    if args.dirs:
        results_dirs = [os.path.join(project_root, d) for d in args.dirs]
    else:
        results_dirs = [
            os.path.join(project_root, d)
            for d in ["results_transcribe", "results_qa", "results_waveform"]
            if os.path.isdir(os.path.join(project_root, d))
        ]

    all_results = {}
    for rd in results_dirs:
        ci = process_results_dir(rd)
        if ci:
            all_results[os.path.basename(rd)] = ci

    # Print grand summary
    if all_results:
        print(f"\n{'=' * 80}")
        print(f"  GRAND SUMMARY — 95% Wilson Confidence Intervals")
        print(f"{'=' * 80}")
        for dir_name, benchmarks in sorted(all_results.items()):
            print(f"\n  [{dir_name}]")
            for bench, ci in sorted(benchmarks.items()):
                d = ci["direct"]
                parts = [f"Direct: {d['rate_pct']:.1f}% [{d['ci_95'][0]:.1f}, {d['ci_95'][1]:.1f}]"]
                if "opus64" in ci:
                    o = ci["opus64"]
                    parts.append(f"Opus64: {o['rate_pct']:.1f}% [{o['ci_95'][0]:.1f}, {o['ci_95'][1]:.1f}]")
                snr_str = f"SNR={ci['avg_snr_db']:.1f}dB" if "avg_snr_db" in ci else ""
                print(f"    {bench:<55} N={ci['n']:>3}  {' | '.join(parts)}  {snr_str}")


if __name__ == "__main__":
    main()
