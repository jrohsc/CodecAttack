"""Exp-A driver: capacity vs. target word count.

3 Suno carriers x 10 target lengths = 30 attack cells.
Each cell shells out to codec_attack/dataset/interview/run_attack.py with
the S3a recipe: eps=1.0, steps=1000, channel_mode=multi_bitrate.

Cells that already have an attacked wav on disk are skipped (resumable).

Usage:
    python scripts/run_expA_words.py                # full 30 cells
    python scripts/run_expA_words.py --dry-run
    python scripts/run_expA_words.py --carriers jazz
    python scripts/run_expA_words.py --targets w015 w040
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent
REPO_ROOT = ABL_ROOT.parents[1]                       # codec_attack/
BENCH_ROOT = REPO_ROOT / "0_all_combined"
SCENARIO_DIR = BENCH_ROOT / "scenarios/03a_ai_detection_bypass"
CARRIER_DIR = BENCH_ROOT / "data/music/ai_generated"

sys.path.insert(0, str(HERE))
import _common  # noqa: E402

PY = os.environ.get(
    "CODEC_ATTACK_PY",
    "${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python",
)

EPS = "1.0"
STEPS = "1000"
ALPHA = "0.2"
OPUS_BITRATES = "16,24,32,64,128,192"
TARGET_MODEL = "qwen2_audio"
CHANNEL_MODE = "multi_bitrate"


def _load_prompt() -> str:
    return (SCENARIO_DIR / "system_prompt.txt").read_text().strip()


def _already_attacked(bundle: Path) -> bool:
    return any(bundle.glob("*_attacked.wav"))


def _carrier_capped_duration(src_duration_s: float) -> float:
    return max(10.0, min(25.0, src_duration_s))


def _run_cell(*, bundle: Path, carrier_path: Path, target_text: str,
              carrier_duration_s: float, prompt: str, dry_run: bool) -> int:
    bundle.mkdir(parents=True, exist_ok=True)
    if _already_attacked(bundle):
        print(f"  SKIP: {bundle.name} already has attacked wav")
        return 0
    cmd = [
        PY, "-u", str(REPO_ROOT / "dataset/interview/run_attack.py"),
        "--music-file", str(carrier_path),
        "--target-clip", "american_f_q1_edgetts",
        "--target-text", target_text,
        "--prompt-text", prompt,
        "--eps", EPS,
        "--alpha", ALPHA,
        "--steps", STEPS,
        "--target-model", TARGET_MODEL,
        "--opus-bitrates", OPUS_BITRATES,
        "--channel-mode", CHANNEL_MODE,
        "--duration", f"{carrier_duration_s:.2f}",
        "--output-dir", str(bundle),
    ]
    if dry_run:
        print("  DRY-RUN:", " ".join(cmd[:8]), "...")
        return 0
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    print(f"  exit={rc} elapsed={time.time() - t0:.1f}s")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--carriers", nargs="+", default=None)
    ap.add_argument("--targets", nargs="+", default=None)
    args = ap.parse_args()

    cfg = _common.load_targets_json(ABL_ROOT / "targets_capacity.json")
    prompt = _load_prompt()
    carriers = cfg["carriers"]
    targets = cfg["targets"]
    if args.carriers:
        carriers = [c for c in carriers if c["id"] in set(args.carriers)]
    if args.targets:
        targets = [t for t in targets if t["id"] in set(args.targets)]

    total = len(carriers) * len(targets)
    print(f"Exp-A: {len(carriers)} carriers x {len(targets)} targets = {total} cells")

    idx = 0
    failures = []
    for carrier in carriers:
        cpath = CARRIER_DIR / carrier["file"]
        cdur = _carrier_capped_duration(carrier["src_duration_s"])
        for target in targets:
            idx += 1
            bundle = _common.bundle_path_expA(
                ABL_ROOT, carrier_id=carrier["id"], target_id=target["id"])
            print(f"\n[{idx}/{total}] carrier={carrier['id']:7s} "
                  f"target={target['id']} ({target['words']}w) dur={cdur:.1f}s")
            rc = _run_cell(
                bundle=bundle, carrier_path=cpath,
                target_text=target["text"], carrier_duration_s=cdur,
                prompt=prompt, dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((carrier["id"], target["id"], rc))

    print(f"\nExp-A done. failures={len(failures)}")
    for f in failures:
        print(f"  {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
