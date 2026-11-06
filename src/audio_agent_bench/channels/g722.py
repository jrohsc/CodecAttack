"""G.722 wide-band (16 kHz, 48/56/64 kbps). Enterprise HD-voice condition."""
from __future__ import annotations

import subprocess
from pathlib import Path


class G722Channel:
    def __init__(self, bitrate_kbps: int = 64) -> None:
        self.bitrate_kbps = bitrate_kbps

    def tag(self) -> str:
        return f"g722_{self.bitrate_kbps}k"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".g722")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav), "-ar", "16000", "-ac", "1",
             "-acodec", "g722", str(tmp)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-ar", "16000", "-ac", "1", str(out)],
            check=True, capture_output=True,
        )
        tmp.unlink(missing_ok=True)
        return out
