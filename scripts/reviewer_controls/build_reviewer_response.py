#!/usr/bin/env python3
"""Assemble the final S1 reviewer-control report from completed, audited results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def best_arm(payload: dict, channel: str) -> tuple[str, int, int]:
    choices = []
    for method, arm in payload["methods"].items():
        cell = arm["channels"][channel]
        choices.append((cell["asr_numerator"], method, cell["asr_denominator"]))
    numerator, method, denominator = max(choices)
    return payload["methods"][method]["arm_label"], numerator, denominator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-json", type=Path, required=True)
    p.add_argument("--baseline-markdown", type=Path, required=True)
    p.add_argument("--ensemble-json", type=Path, required=True)
    p.add_argument("--ensemble-markdown", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--job-ledger", type=Path, required=True)
    p.add_argument("--reproduce", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    baseline = read_json(args.baseline_json)
    ensemble = read_json(args.ensemble_json)
    required = {
        "latent_codecattack", "original_waveform_published",
        "tuned_waveform", "bandlimited_4khz",
    }
    if set(baseline["methods"]) != required:
        raise SystemExit(f"four-arm result incomplete: {set(baseline['methods'])}")
    for method, arm in baseline["methods"].items():
        if arm["n_examples"] != 40:
            raise SystemExit(f"{method}: expected 40 examples, got {arm['n_examples']}")
        if arm["missing_or_failed_cases"]:
            raise SystemExit(f"{method}: missing/failed cases present")
    if not ensemble["identical_carrier_target_pairs"]:
        raise SystemExit("ensemble carrier-target sets differ")
    for name, arm in ensemble["arms"].items():
        if arm["n_examples"] != 50 or arm["missing_or_failed"]:
            raise SystemExit(f"{name}: incomplete evaluation")

    split_lines = args.split_manifest.read_text(encoding="utf-8").splitlines()
    n_dev = sum(",development," in f",{line}," for line in split_lines[1:])
    n_eval = sum(",evaluation," in f",{line}," for line in split_lines[1:])
    common_n = baseline["common_clean_success_subset"]["denominator"]
    best_clean = best_arm(baseline, "clean")
    best_opus16 = best_arm(baseline, "opus16")

    lines = [
        "# S1 controlled-comparison reviewer report",
        "",
        "## Scope and completion",
        "",
        f"This report covers the frozen S1 comparison only: {n_dev} development and "
        f"{n_eval} held-out source recordings. Pilot outputs are not pooled with these results. "
        "Every baseline arm has 40/40 held-out examples; every ensemble arm has 50/50 attack "
        "waveforms evaluated on all four victims. There are no missing or failed final cases.",
        "",
        f"Exact matching lowercases and normalizes whitespace, retains punctuation, and requires "
        f"the complete target as a contiguous substring. The common-clean-success intersection "
        f"contains {common_n} examples.",
        "",
        "## Baseline comparison (A/B/C/D)",
        "",
        args.baseline_markdown.read_text(encoding="utf-8").strip(),
        "",
        "The full per-example SNR residuals and channel outcomes are in `baseline_report.json`.",
        "",
        "## Ensemble comparison (N=1/N=2/N=3)",
        "",
        args.ensemble_markdown.read_text(encoding="utf-8").strip(),
        "",
        "Training delivery was clean for steps 1--300 and randomized Opus BPDA at "
        "{16,24,32,64,128} kb/s for steps 301--1000. MP3 and AAC-LC were held out.",
        "",
        "## Reviewer-response mapping",
        "",
        "| Concern | Status | Evidence / limitation |",
        "|---|---|---|",
        "| Controlled A/B/C/D comparison | Resolved for S1, seed 1 | Same 40 held-out IDs, prompt, delivery implementations, exact metric, and fixed waveform per arm/channel. |",
        "| Development/evaluation separation | Resolved | Learning-rate ratio was frozen from development attack outcomes; held-out results were never used for tuning. Epsilon refinement used distortion only. |",
        "| Tuned unrestricted waveform baseline | Resolved | Arm C is complete and was selected for effectiveness under the development distortion regime, not nearest SNR alone. |",
        "| <=4 kHz enforcement | Resolved | Reported both optimizer-domain and saved/reloaded perturbation leakage; PCM quantization is retained rather than rounded away. |",
        "| Correct conditional survival | Resolved | Every cell reports clean-and-compressed / clean; zero-clean denominators are N/A, and new successes are separate. |",
        "| Ensemble-size attribution | Partly resolved | Member sets, pairs, prompts, budgets, declared schedules, and checkpoint rule match, but compute scales with N and Opus bitrate draws use unseeded process-global RNG. |",
        "| Parameterization-only latent/waveform claim | Partly resolved | Tuned C/D remove several legacy differences, but published A restores periodic successes while B/C/D save the last iterate; B also has legacy normalization/schedule differences. |",
        "| Generalization beyond S1 / uncertainty across seeds | Unresolved by design | No extra scenario/model/seed sweep was launched; conclusions are restricted to this controlled S1 campaign. |",
        "",
        "## Concise conclusions",
        "",
        f"- The strongest clean baseline arm is {best_clean[0]} at {best_clean[1]}/{best_clean[2]}; "
        f"the strongest Opus-16 arm is {best_opus16[0]} at {best_opus16[1]}/{best_opus16[2]}. "
        "These are paired S1 outcomes, not broad superiority claims.",
        "- Compression robustness is interpreted using conditional survival counts, not compressed-ASR/clean-ASR ratios; compression-induced successes are reported separately.",
        "- The ensemble ladder is evidence about adding nested members under the same EoT distribution while increasing compute. It is not a compute-matched or exact-randomness-matched estimate of ensemble size alone.",
        "- Unfavorable or null results are retained in the tables; unfinished jobs and pilot outputs are excluded.",
        "",
        "## Audit and reproducibility",
        "",
        "See `PROTOCOL_AUDIT_20260921.md` for the detailed confounds and manuscript statements "
        "requiring correction, `S1_SPLIT_MANIFEST.csv` for stable IDs/source split, "
        "`JOB_LEDGER_20260921.json` for job provenance, and `REPRODUCE_20260921.md` for commands.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
