#!/usr/bin/env python3
"""Build the four-arm comparison table from scored per-example records.

Reports, per arm and delivery channel: ASR with its numerator and denominator,
conditional survival with its own denominator (the clean-successful subset,
which differs between arms), and compression-induced new successes.  Survival is
never approximated by compressed ASR / clean ASR.  Distortion is reported as the
realized carrier-referenced SNR and, against a named reference arm, the full
distribution of per-example SNR differences -- over every evaluated example, not
only the successful ones.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANNELS = ("clean", "opus16", "opus24", "opus32", "opus64", "opus128", "mp3_64", "aac_64")
HELDOUT = ("mp3_64", "aac_64")
ARM_LABELS = {
    "latent_codecattack": "A. Latent CodecAttack",
    "original_waveform_published": "B. Original waveform",
    "tuned_waveform": "C. Tuned unrestricted waveform",
    "bandlimited_4khz": "D. Tuned <=4 kHz waveform",
}


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    return rows


def fmt(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, nargs="+", required=True,
                   help="evaluate.py output JSONL files (one or more arms).")
    p.add_argument("--reference-method", default="latent_codecattack",
                   help="Arm whose per-example SNR the others are matched to.")
    p.add_argument("--snr-tolerance-db", type=float, default=1.0)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-markdown", type=Path, required=True)
    args = p.parse_args()

    rows = load(args.records)
    if not rows:
        raise SystemExit("no records")

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    ref_snr = {r["example_id"]: float(r["snr_db"])
               for r in by_method.get(args.reference_method, [])}

    report = {
        "reference_method": args.reference_method,
        "snr_tolerance_db": args.snr_tolerance_db,
        "channels": list(CHANNELS),
        "heldout_channels": list(HELDOUT),
        "target_match_metric": "complete target as a contiguous substring after "
                               "lowercasing and whitespace normalization; punctuation retained",
        "methods": {},
    }

    for method in sorted(by_method):
        mr = by_method[method]
        seen, uniq = set(), []
        for r in mr:                      # one row per example per arm
            if r["example_id"] in seen:
                continue
            seen.add(r["example_id"])
            uniq.append(r)
        mr = uniq

        snr = np.asarray([float(r["snr_db"]) for r in mr], dtype=float)
        diffs = np.asarray([float(r["snr_db"]) - ref_snr[r["example_id"]]
                            for r in mr if r["example_id"] in ref_snr], dtype=float)

        channels = {}
        for ch in CHANNELS:
            valid = [r for r in mr if ch in r.get("outcomes", {})]
            if not valid:
                continue
            hits = sum(int(bool(r["outcomes"][ch]["match"])) for r in valid)
            clean_ok = [r for r in valid if bool(r["outcomes"]["clean"]["match"])]
            survived = [r for r in clean_ok if bool(r["outcomes"][ch]["match"])]
            new = [r for r in valid
                   if not bool(r["outcomes"]["clean"]["match"])
                   and bool(r["outcomes"][ch]["match"])]
            channels[ch] = {
                "asr_numerator": hits,
                "asr_denominator": len(valid),
                "asr_percent": 100.0 * hits / len(valid),
                "conditional_survival_numerator": len(survived),
                "conditional_survival_denominator": len(clean_ok),
                "conditional_survival_percent": (100.0 * len(survived) / len(clean_ok))
                                                if clean_ok else None,
                "compression_induced_new_successes": (0 if ch == "clean" else len(new)),
            }

        report["methods"][method] = {
            "arm_label": ARM_LABELS.get(method, method),
            "n_examples": len(mr),
            "example_ids": sorted(seen),
            "seeds": sorted({str(r.get("seed")) for r in mr}),
            "steps": sorted({str(r.get("steps")) for r in mr}),
            "alpha_over_epsilon": sorted({str(r.get("alpha_over_epsilon")) for r in mr}),
            "missing_or_failed_cases": [],
            "snr_db": {
                "mean": float(snr.mean()), "median": float(np.median(snr)),
                "sd": float(snr.std(ddof=1)) if len(snr) > 1 else None,
                "min": float(snr.min()), "max": float(snr.max()),
            },
            "snr_difference_vs_reference_db": ({
                "n_paired": int(len(diffs)),
                "n_unpaired": int(len(mr) - len(diffs)),
                "mean": float(diffs.mean()), "median": float(np.median(diffs)),
                "min": float(diffs.min()), "max": float(diffs.max()),
                "max_absolute": float(np.abs(diffs).max()),
                "n_within_tolerance": int((np.abs(diffs) <= args.snr_tolerance_db).sum()),
                "all_within_tolerance": bool(np.all(np.abs(diffs) <= args.snr_tolerance_db)),
                "aggregate_within_tolerance": bool(abs(float(diffs.mean())) <= args.snr_tolerance_db),
                "per_example": [
                    {"example_id": r["example_id"],
                     "snr_db": float(r["snr_db"]),
                     "reference_snr_db": ref_snr[r["example_id"]],
                     "difference_db": float(r["snr_db"]) - ref_snr[r["example_id"]]}
                    for r in mr if r["example_id"] in ref_snr
                ],
            } if len(diffs) else None),
            "perturbation_energy_above_4khz": {
                "delivered_vs_carrier_mean": float(np.mean(
                    [r["post_above_4khz_fraction"] for r in mr
                     if r.get("post_above_4khz_fraction") is not None])),
                "optimizer_domain_16k_mean": (
                    float(np.mean([r["delta16k_above_4khz_fraction"] for r in mr
                                   if r.get("delta16k_above_4khz_fraction") is not None]))
                    if any(r.get("delta16k_above_4khz_fraction") is not None for r in mr) else None),
                "perturbation_post_serialization_mean": (
                    float(np.mean([r["perturbation_above_4khz_fraction_post_serialization"]
                                   for r in mr
                                   if r.get("perturbation_above_4khz_fraction_post_serialization") is not None]))
                    if any(r.get("perturbation_above_4khz_fraction_post_serialization") is not None for r in mr) else None),
            },
            "channels": channels,
        }

    # A method's conditional-survival denominator is intentionally its own
    # clean-success set. Also provide the stricter paired view requested by the
    # reviewer: examples that are clean-successful for *every* reported arm.
    method_rows = {
        method: {r["example_id"]: r for r in rows_}
        for method, rows_ in by_method.items()
    }
    shared_ids = set.intersection(*(set(rows_) for rows_ in method_rows.values()))
    common_clean = sorted(
        example_id for example_id in shared_ids
        if all(bool(rows_[example_id].get("outcomes", {}).get("clean", {}).get("match"))
               for rows_ in method_rows.values())
    )
    report["common_clean_success_subset"] = {
        "definition": "examples present in every arm and clean-successful in every arm",
        "denominator": len(common_clean),
        "example_ids": common_clean,
        "methods": {},
    }
    for method, rows_ in method_rows.items():
        channel_counts = {}
        for ch in CHANNELS:
            if ch == "clean":
                continue
            available = [example_id for example_id in common_clean
                         if ch in rows_[example_id].get("outcomes", {})]
            hits = sum(bool(rows_[example_id]["outcomes"][ch]["match"])
                       for example_id in available)
            channel_counts[ch] = {
                "numerator": int(hits),
                "denominator": len(available),
                "percent": (100.0 * hits / len(available)) if available else None,
            }
        report["common_clean_success_subset"]["methods"][method] = channel_counts

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))

    lines = ["# Reviewer-control comparison", "",
             f"Target match: {report['target_match_metric']}.", "",
             f"SNR reference arm: `{args.reference_method}`; "
             f"tolerance {args.snr_tolerance_db:g} dB.", "",
             "## Distortion", "",
             "| Arm | n | SNR mean | SNR median | SNR SD | dSNR mean | dSNR max\\|.\\| | within tol | >4 kHz (delivered-vs-carrier) | >4 kHz (saved perturbation) | >4 kHz (optimizer) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in report["methods"].items():
        s, diff = d["snr_db"], d["snr_difference_vs_reference_db"]
        band = d["perturbation_energy_above_4khz"]
        lines.append(
            f"| {d['arm_label']} | {d['n_examples']} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['sd'])} | "
            + (f"{fmt(diff['mean'])} | {fmt(diff['max_absolute'])} | "
               f"{diff['n_within_tolerance']}/{diff['n_paired']} | " if diff else "-- | -- | -- | ")
            + f"{band['delivered_vs_carrier_mean']:.2e} | "
            + (f"{band['perturbation_post_serialization_mean']:.2e} |" if band['perturbation_post_serialization_mean'] is not None else "-- |")
            + (f"{band['optimizer_domain_16k_mean']:.2e} |" if band['optimizer_domain_16k_mean'] is not None else "-- |")
        )

    lines += ["", "## Attack success rate (exact target substring)", "",
              "| Arm | " + " | ".join(CHANNELS) + " |",
              "|---" * (len(CHANNELS) + 1) + "|"]
    for m, d in report["methods"].items():
        cells = []
        for ch in CHANNELS:
            c = d["channels"].get(ch)
            cells.append("--" if c is None else f"{c['asr_percent']:.1f} ({c['asr_numerator']}/{c['asr_denominator']})")
        lines.append(f"| {d['arm_label']} | " + " | ".join(cells) + " |")

    lines += ["", "## Conditional survival: successful before AND after / successful before", "",
              "| Arm | " + " | ".join(c for c in CHANNELS if c != "clean") + " |",
              "|---" * len(CHANNELS) + "|"]
    for m, d in report["methods"].items():
        cells = []
        for ch in CHANNELS:
            if ch == "clean":
                continue
            c = d["channels"].get(ch)
            if c is None or c["conditional_survival_percent"] is None:
                cells.append(f"N/A (0 clean successes)" if c is not None else "N/A")
            else:
                cells.append(f"{c['conditional_survival_percent']:.1f} "
                             f"({c['conditional_survival_numerator']}/{c['conditional_survival_denominator']})")
        lines.append(f"| {d['arm_label']} | " + " | ".join(cells) + " |")

    lines += ["", "## Compression-induced new successes (clean failure, compressed success)", "",
              "| Arm | " + " | ".join(c for c in CHANNELS if c != "clean") + " |",
              "|---" * len(CHANNELS) + "|"]
    for m, d in report["methods"].items():
        cells = []
        for ch in CHANNELS:
            if ch == "clean":
                continue
            c = d["channels"].get(ch)
            cells.append("--" if c is None else str(c["compression_induced_new_successes"]))
        lines.append(f"| {d['arm_label']} | " + " | ".join(cells) + " |")

    common = report["common_clean_success_subset"]
    lines += ["", "## Common-clean-success subset", "",
              f"Intersection denominator: {common['denominator']} examples clean-successful in every arm.", "",
              "| Arm | " + " | ".join(c for c in CHANNELS if c != "clean") + " |",
              "|---" * len(CHANNELS) + "|"]
    for m, counts in common["methods"].items():
        cells = []
        for ch in CHANNELS:
            if ch == "clean":
                continue
            c = counts[ch]
            cells.append("N/A (0)" if c["percent"] is None else
                         f"{c['percent']:.1f} ({c['numerator']}/{c['denominator']})")
        label = report["methods"][m]["arm_label"]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    args.output_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
