"""Codec-sidecar eval wrapper for capacity-ablation bundles.

For every leaf bundle (directory containing *_attacked.wav) under
expA_words/ and expB_duration/, runs the shared eval_mp3_aac_bundle.py
script which writes a *_codecs.json per attacked wav.

Usage:
    python scripts/eval_channels.py
    python scripts/eval_channels.py --exp expA
    python scripts/eval_channels.py --exp transcribe
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent
REPO_ROOT = ABL_ROOT.parents[1]
BENCH_SCRIPTS = REPO_ROOT / "0_all_combined/scripts"
SCENARIO_DIR = REPO_ROOT / "0_all_combined/scenarios/03a_ai_detection_bypass"

PY = os.environ.get(
    "CODEC_ATTACK_PY",
    "${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python",
)

TRANSCRIBE_PROMPT = "What does the person say in this audio?"


def _leaf_bundles(root: Path) -> list[Path]:
    return sorted({w.parent for w in root.rglob("*_attacked.wav")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["expA", "expB", "transcribe", "both", "all"],
                    default="all")
    args = ap.parse_args()

    s3a_prompt = (SCENARIO_DIR / "system_prompt.txt").read_text().strip()

    jobs: list[tuple[str, Path, str]] = []
    if args.exp in ("expA", "both", "all"):
        jobs.append(("expA", ABL_ROOT / "expA_words/qwen2_audio_eps1.0_multibitrate",
                     s3a_prompt))
    if args.exp in ("expB", "both", "all"):
        jobs.append(("expB", ABL_ROOT / "expB_duration/qwen2_audio_eps1.0_multibitrate",
                     s3a_prompt))
    if args.exp in ("transcribe", "all"):
        jobs.append(("transcribe",
                     ABL_ROOT / "transcribe_control/qwen2_audio_eps1.0_multibitrate",
                     TRANSCRIBE_PROMPT))

    failures = []
    for label, root, prompt in jobs:
        if not root.exists():
            print(f"[{label}] root missing: {root} (skipping)")
            continue
        bundles = _leaf_bundles(root)
        print(f"[{label}] evaluating {len(bundles)} bundles")
        for i, b in enumerate(bundles, 1):
            print(f"\n  [{label} {i}/{len(bundles)}] {b.relative_to(ABL_ROOT)}")
            rc = subprocess.run([
                PY, "-u", str(BENCH_SCRIPTS / "eval_mp3_aac_bundle.py"),
                "--bundle-dir", str(b),
                "--target-model", "qwen2_audio",
                "--prompt-text", prompt,
            ]).returncode
            if rc != 0:
                failures.append(f"{label}:{b}")
                print(f"  FAILED exit={rc}")

    print(f"\nEval done. failures={len(failures)}")
    for f in failures:
        print(f"  {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
