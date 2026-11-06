"""Render the paper's seven delivery channels once for new held-out victims.

The functions and order mirror ensemble_heldout_eval.py. FLOAT WAV preserves
the float32 waveform passed to that evaluator without an extra PCM16 roundtrip.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo_ota import apply_opus_compression  # noqa: E402
from aac_channel import apply_mp3_compression, apply_aac_compression  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT = RESULTS / "ensemble_shared112_delivery"
CHANNELS = {
    "opus64": lambda a, sr: apply_opus_compression(a, sr, bitrate_kbps=64),
    "opus128": lambda a, sr: apply_opus_compression(a, sr, bitrate_kbps=128),
    "mp3_64": lambda a, sr: apply_mp3_compression(a, sr, bitrate_kbps=64),
    "mp3_128": lambda a, sr: apply_mp3_compression(a, sr, bitrate_kbps=128),
    "aac_64": lambda a, sr: apply_aac_compression(a, sr, bitrate_kbps=64),
    "aac_128": lambda a, sr: apply_aac_compression(a, sr, bitrate_kbps=128),
}


def bundle(seed, shard):
    suffix = f"_zeroeot_seed{seed}" if seed != 1 else ""
    name = ("benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1"
            f"_bestpostcodec_steps600{suffix}_shard{shard}of4")
    return RESULTS / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full224", action="store_true",
                    help="prepare all four shards instead of the original shared 112")
    args = ap.parse_args()
    output = RESULTS / ("ensemble_full224_delivery" if args.full224 else "ensemble_shared112_delivery")
    expected = 224 if args.full224 else 112
    files = {}
    for seed in (1, 7, 42):
        files[seed] = {}
        for shard in range(4 if args.full224 else 2):
            b = bundle(seed, shard)
            files[seed].update({p.name: (p, b) for p in (b / "audio").glob("*.wav")})
    common = sorted(set.intersection(*(set(d) for d in files.values())))
    if len(common) != expected:
        raise RuntimeError(f"expected {expected} common targets, found {len(common)}")

    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in (1, 7, 42):
        for filename in common:
            src, b = files[seed][filename]
            target = json.loads(
                (b / "qwen2_audio" / src.stem / "record.json").read_text()
            )["target_text"]
            dest = output / str(seed) / src.stem
            old = OUTPUT / str(seed) / src.stem
            if args.full224 and old.is_dir() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(old, target_is_directory=True)
            else:
                dest.mkdir(parents=True, exist_ok=True)
            manifest.append({"seed": seed, "clip": src.stem,
                             "target": target, "source": str(src)})
            needed = [c for c in ("clean", *CHANNELS)
                      if not (dest / f"{c}.wav").exists()]
            if not needed:
                continue
            audio, sr = sf.read(src, dtype="float32")
            if sr != 24000 or audio.ndim != 1:
                raise ValueError(f"expected 24 kHz mono source: {src}")
            for channel in needed:
                decoded = audio if channel == "clean" else CHANNELS[channel](audio, sr)
                decoded = np.ascontiguousarray(np.asarray(decoded, dtype=np.float32))
                tmp = dest / f"{channel}.tmp.wav"
                sf.write(tmp, decoded, sr, subtype="FLOAT")
                tmp.replace(dest / f"{channel}.wav")
            print(f"seed={seed} {src.stem} wrote={','.join(needed)}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} clip entries to {output}", flush=True)


if __name__ == "__main__":
    main()
