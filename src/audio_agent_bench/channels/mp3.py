"""MP3 — legacy music codec still pervasive in user uploads, DAW exports, and archival libraries.

Round-trip: wav -> libmp3lame at `bitrate_kbps` -> decode back to wav at the input sample rate.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class MP3Channel:
    def __init__(self, bitrate_kbps: int = 192) -> None:
        self.bitrate_kbps = bitrate_kbps

    def tag(self) -> str:
        return f"mp3_{self.bitrate_kbps}k"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav),
             "-acodec", "libmp3lame", "-b:a", f"{self.bitrate_kbps}k",
             "-ac", "1", str(tmp)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-ac", "1", str(out)],
            check=True, capture_output=True,
        )
        tmp.unlink(missing_ok=True)
        return out
