"""Transcribe-prompt control: 1 cell (jazz/w015) with transcribe prompt.

Rules out 'the cliff is a prompt artifact' in the paper. Only the attack
is run here; eval_channels.py --exp transcribe handles the codec sidecars.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABL_ROOT = HERE.parent
REPO_ROOT = ABL_ROOT.parents[1]
BENCH_ROOT = REPO_ROOT / "0_all_combined"
CARRIER_DIR = BENCH_ROOT / "data/music/ai_generated"

sys.path.insert(0, str(HERE))
import _common  # noqa: E402

PY = os.environ.get(
    "CODEC_ATTACK_PY",
    "${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python",
)

TRANSCRIBE_PROMPT = "What does the person say in this audio?"


def main():
    cfg = _common.load_targets_json(ABL_ROOT / "targets_capacity.json")
    target = next(t for t in cfg["targets"] if t["id"] == "w015")
    carrier = next(c for c in cfg["carriers"] if c["id"] == "jazz")
    cpath = CARRIER_DIR / carrier["file"]

    bundle = ABL_ROOT / "transcribe_control" / \
             "qwen2_audio_eps1.0_multibitrate" / "jazz" / "w015"
    bundle.mkdir(parents=True, exist_ok=True)

    if any(bundle.glob("*_attacked.wav")):
        print(f"  SKIP: {bundle.name} already has attacked wav")
        return

    cmd = [
        PY, "-u", str(REPO_ROOT / "dataset/interview/run_attack.py"),
        "--music-file", str(cpath),
        "--target-clip", "american_f_q1_edgetts",
        "--target-text", target["text"],
        "--prompt-text", TRANSCRIBE_PROMPT,
        "--eps", "1.0", "--alpha", "0.2", "--steps", "1000",
        "--target-model", "qwen2_audio",
        "--opus-bitrates", "16,24,32,64,128,192",
        "--channel-mode", "multi_bitrate",
        "--duration", "15.00",
        "--output-dir", str(bundle),
    ]
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    print(f"  exit={rc} elapsed={time.time() - t0:.1f}s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
