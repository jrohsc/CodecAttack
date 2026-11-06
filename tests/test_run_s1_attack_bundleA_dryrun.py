"""Dry-run planning test: Bundle A driver produces the right subprocess
command line and targets the robust_optimization subtree."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 0_all_combined/


def test_bundleA_dry_run_lists_multi_bitrate_command(tmp_path, monkeypatch):
    script = ROOT / "scripts/run_s1_attack_bundleA.py"
    assert script.is_file(), f"driver not found at {script}"

    # Run --dry-run and capture stdout.
    proc = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # Every planned command must carry the new channel mode and bitrate set.
    assert "--channel-mode" in out and "multi_bitrate" in out
    assert "--opus-train-bitrates" in out and "16,24,32,64,128" in out
    # Output dir must land under robust_optimization/bundleA_multibitrate/...
    assert "robust_optimization/bundleA_multibitrate/qwen2_audio/eps_1.0/audio" in out
    # Only Qwen2 at eps=1.0 (per spec).
    assert "eps 1.0" in out or "eps=1.0" in out
