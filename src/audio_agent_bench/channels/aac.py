"""AAC-LC — the dominant music-streaming codec (Spotify, Apple Music, YouTube, SoundCloud).

Round-trip: wav -> AAC-LC in an .m4a container at `bitrate_kbps` -> decode back to wav at
the input sample rate. Uses ffmpeg's native `aac` encoder (no libfdk_aac dependency).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class AACChannel:
    def __init__(self, bitrate_kbps: int = 128) -> None:
        self.bitrate_kbps = bitrate_kbps

    def tag(self) -> str:
        return f"aac{self.bitrate_kbps}k"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".m4a")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav),
             "-acodec", "aac", "-b:a", f"{self.bitrate_kbps}k",
             "-ac", "1", str(tmp)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-ac", "1", str(out)],
            check=True, capture_output=True,
        )
        tmp.unlink(missing_ok=True)
        return out
