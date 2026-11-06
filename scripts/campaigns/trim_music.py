"""Strip head/tail of music carriers per data/trim_offsets.json.

Lyrical tracks get their intro+outro stripped so attack sees vocal-heavy
middle; instrumental tracks are copied as-is. All outputs are mono 24 kHz
WAV under data/music_trimmed/<bucket>/<name>.wav.
"""
import json
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MUSIC_ROOT = ROOT / "data" / "music"
OUT_ROOT = ROOT / "data" / "music_trimmed"
OFFSETS = ROOT / "data" / "trim_offsets.json"
SR = 24000


def main() -> None:
    cfg = json.loads(OFFSETS.read_text())
    for rel, meta in cfg.items():
        if rel.startswith("__"):
            continue
        src = MUSIC_ROOT / rel
        if not src.exists():
            print(f"[skip] missing: {src}")
            continue
        head = float(meta["head_s"])
        tail = float(meta["tail_s"])

        y, _ = librosa.load(str(src), sr=SR, mono=True)
        total_s = len(y) / SR
        start = int(head * SR)
        end = len(y) - int(tail * SR)
        if end <= start:
            print(f"[skip] {rel}: head+tail >= duration")
            continue
        clip = y[start:end]

        out = (OUT_ROOT / rel).with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), clip, SR)
        kept = len(clip) / SR
        print(
            f"{rel:55s}  {meta['kind']:12s}  "
            f"total={total_s:6.1f}s  kept={kept:6.1f}s  -> {out.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
