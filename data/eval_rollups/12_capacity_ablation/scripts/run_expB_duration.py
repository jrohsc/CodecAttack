"""Exp-B driver: capacity vs. carrier duration.

3 Suno carriers x 6 durations x 3 anchor targets = 54 attack cells.
Same S3a recipe as Exp-A; only difference is --duration overrides the
25s cap so we can test 30s and 60s carriers.

Usage:
    python scripts/run_expB_duration.py                # full 54 cells
    python scripts/run_expB_duration.py --dry-run
    python scripts/run_expB_duration.py --durations 15 30
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
REPO_ROOT = ABL_ROOT.parents[1]
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

DURATIONS = [5, 10, 15, 20, 30, 60]


def _load_prompt() -> str:
    return (SCENARIO_DIR / "system_prompt.txt").read_text().strip()


def _already_attacked(bundle: Path) -> bool:
    return any(bundle.glob("*_attacked.wav"))


def _run_cell(*, bundle: Path, carrier_path: Path, target_text: str,
              duration_s: int, prompt: str, dry_run: bool) -> int:
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
        "--duration", f"{duration_s:.2f}",
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
    ap.add_argument("--durations", nargs="+", type=int, default=DURATIONS)
    ap.add_argument("--anchors", nargs="+", default=None)
    args = ap.parse_args()

    cfg = _common.load_targets_json(ABL_ROOT / "targets_capacity.json")
    prompt = _load_prompt()
    carriers = cfg["carriers"]
    if args.carriers:
        carriers = [c for c in carriers if c["id"] in set(args.carriers)]
    anchor_ids = args.anchors if args.anchors else cfg["anchors_expB"]
    by_id = {t["id"]: t for t in cfg["targets"]}
    anchors = [by_id[a] for a in anchor_ids]

    total = len(carriers) * len(args.durations) * len(anchors)
    print(f"Exp-B: {len(carriers)} carriers x {len(args.durations)} durations "
          f"x {len(anchors)} anchors = {total} cells")

    idx = 0
    failures = []
    for carrier in carriers:
        cpath = CARRIER_DIR / carrier["file"]
        for dur in args.durations:
            if dur > carrier["src_duration_s"]:
                print(f"  SKIP dur={dur}s > src {carrier['src_duration_s']:.1f}s "
                      f"for {carrier['id']}")
                continue
            for anchor in anchors:
                idx += 1
                bundle = _common.bundle_path_expB(
                    ABL_ROOT, carrier_id=carrier["id"],
                    duration_s=dur, target_id=anchor["id"])
                print(f"\n[{idx}/{total}] carrier={carrier['id']:7s} "
                      f"dur={dur:3d}s anchor={anchor['id']} ({anchor['words']}w)")
                rc = _run_cell(
                    bundle=bundle, carrier_path=cpath,
                    target_text=anchor["text"], duration_s=dur,
                    prompt=prompt, dry_run=args.dry_run,
                )
                if rc != 0:
                    failures.append((carrier["id"], dur, anchor["id"], rc))

    print(f"\nExp-B done. failures={len(failures)}")
    for f in failures:
        print(f"  {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
