"""Exp-A budget sweep: rerun Exp-A at additional eps values.

Mirrors run_expA_words.py with --eps as a parameter and writes outputs to
expA_words/qwen2_audio_eps{X}_multibitrate/<carrier>/<wXXX>/.

Usage:
    python scripts/run_expA_eps_sweep.py --eps 0.5 --carriers jazz
    python scripts/run_expA_eps_sweep.py --eps 1.5 --carriers jazz \\
        --targets w002 w005 w008 w015 w020 w032
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

STEPS = "1000"
ALPHA_DEFAULT_RATIO = 0.2  # alpha = eps * ratio
OPUS_BITRATES = "16,24,32,64,128,192"
TARGET_MODEL = "qwen2_audio"
CHANNEL_MODE = "multi_bitrate"


def _attack_tag(eps: float) -> str:
    return f"qwen2_audio_eps{eps:g}_multibitrate"


def _bundle_path(eps: float, carrier_id: str, target_id: str) -> Path:
    return ABL_ROOT / "expA_words" / _attack_tag(eps) / carrier_id / target_id


def _load_prompt() -> str:
    return (SCENARIO_DIR / "system_prompt.txt").read_text().strip()


def _already_attacked(bundle: Path) -> bool:
    return any(bundle.glob("*_attacked.wav"))


def _carrier_capped_duration(src_duration_s: float) -> float:
    return max(10.0, min(25.0, src_duration_s))


def _run_cell(*, eps: float, alpha: float, bundle: Path, carrier_path: Path,
              target_text: str, carrier_duration_s: float,
              prompt: str, dry_run: bool) -> int:
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
        "--eps", f"{eps}",
        "--alpha", f"{alpha:g}",
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
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=None,
                    help="PGD step (default eps * 0.2)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--carriers", nargs="+", default=None)
    ap.add_argument("--targets", nargs="+", default=None)
    args = ap.parse_args()

    eps = args.eps
    alpha = args.alpha if args.alpha is not None else eps * ALPHA_DEFAULT_RATIO

    cfg = _common.load_targets_json(ABL_ROOT / "targets_capacity.json")
    prompt = _load_prompt()
    carriers = cfg["carriers"]
    targets = cfg["targets"]
    if args.carriers:
        carriers = [c for c in carriers if c["id"] in set(args.carriers)]
    if args.targets:
        targets = [t for t in targets if t["id"] in set(args.targets)]

    total = len(carriers) * len(targets)
    print(f"Exp-A eps={eps} alpha={alpha:g}: "
          f"{len(carriers)} carriers x {len(targets)} targets = {total} cells")

    idx = 0
    failures = []
    for carrier in carriers:
        cpath = CARRIER_DIR / carrier["file"]
        cdur = _carrier_capped_duration(carrier["src_duration_s"])
        for target in targets:
            idx += 1
            bundle = _bundle_path(eps, carrier["id"], target["id"])
            print(f"\n[{idx}/{total}] eps={eps} carrier={carrier['id']:7s} "
                  f"target={target['id']} ({target['words']}w) dur={cdur:.1f}s")
            rc = _run_cell(
                eps=eps, alpha=alpha,
                bundle=bundle, carrier_path=cpath,
                target_text=target["text"], carrier_duration_s=cdur,
                prompt=prompt, dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((carrier["id"], target["id"], rc))

    print(f"\nExp-A eps={eps} done. failures={len(failures)}")
    for f in failures:
        print(f"  {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
