"""Symlink source wavs into scenarios/diagnostics/smoke_latent_vs_waveform/audio/.

Run once before `bash scripts/run_scenario.sh scenarios/diagnostics/smoke_latent_vs_waveform`.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CODEC = HERE.parent  # codec_attack/

PAIRS = {
    "diagnostics/smoke_latent_vs_waveform": [
        (
            CODEC / "1_results_qa/benchmark_qwen2_audio_eps1.0_empire_state_of_mind_saycan/audio/empire_state_of_mind__saycan_000.wav",
            "latent__empire_saycan_000.wav",
        ),
        (
            CODEC / "6_results_waveform/sweep_waveform_eps2.0_empire_state_of_mind/audio/empire_state_of_mind__saycan_000.wav",
            "waveform__empire_saycan_000.wav",
        ),
    ],
}


def main() -> None:
    for scenario, links in PAIRS.items():
        dst_dir = HERE / "scenarios" / scenario / "audio"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src, name in links:
            if not src.is_file():
                print(f"MISSING SOURCE: {src}")
                continue
            dst = dst_dir / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            print(f"linked {dst.relative_to(HERE)} -> {src}")


if __name__ == "__main__":
    main()
