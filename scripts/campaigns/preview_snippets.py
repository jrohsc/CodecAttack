"""Export 2-second preview snippets at several offsets per lyrical music file.

Listen to each preview to find the offset where vocals begin, then write
the chosen start_s per file into data/trim_offsets.json.
"""
import json
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MUSIC_ROOT = ROOT / "data" / "music"
PREVIEW_ROOT = ROOT / "data" / "previews"

LYRICAL_HINT = "_lyrical"
OFFSETS = [0, 5, 10, 15, 20, 25, 30, 40]  # seconds
SNIPPET_S = 2.0
SR = 24000


def main() -> None:
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    index = {}
    for bucket in ("ai_generated", "copyrighted"):
        for mp3 in sorted((MUSIC_ROOT / bucket).glob("*.mp3")):
            if LYRICAL_HINT not in mp3.stem:
                continue
            y, _ = librosa.load(str(mp3), sr=SR, mono=True)
            total_s = len(y) / SR
            entries = []
            for off in OFFSETS:
                if off + SNIPPET_S > total_s:
                    continue
                clip = y[int(off * SR) : int((off + SNIPPET_S) * SR)]
                out = PREVIEW_ROOT / f"{mp3.stem}__at_{off:03d}s.wav"
                sf.write(str(out), clip, SR)
                entries.append({"start_s": off, "preview": str(out.relative_to(ROOT))})
            index[f"{bucket}/{mp3.name}"] = {"total_s": round(total_s, 1), "previews": entries}
            print(f"{bucket}/{mp3.name}: {len(entries)} previews")

    (ROOT / "data" / "preview_index.json").write_text(json.dumps(index, indent=2))
    print(f"\nPreviews written to {PREVIEW_ROOT}")
    print(f"Index at {ROOT / 'data' / 'preview_index.json'}")


if __name__ == "__main__":
    main()
