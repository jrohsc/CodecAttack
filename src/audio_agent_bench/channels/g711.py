"""G.711 μ-law narrow-band (8 kHz, 64 kbps). PSTN-gateway condition.

Round-trip: resample 16k -> 8k -> encode μ-law -> decode μ-law -> resample 8k -> 16k.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class G711MulawChannel:
    def __init__(self, a_law: bool = False) -> None:
        self.a_law = a_law

    def tag(self) -> str:
        return "g711_alaw" if self.a_law else "g711_mulaw"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        codec = "pcm_alaw" if self.a_law else "pcm_mulaw"
        raw = out.with_suffix(".raw")
        # Encode to μ-law at 8kHz
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav), "-ar", "8000", "-ac", "1",
             "-acodec", codec, "-f", codec.split("_")[1], str(raw)],
            check=True, capture_output=True,
        )
        # Decode back to wav at 16 kHz (model-friendly)
        subprocess.run(
            ["ffmpeg", "-y", "-f", codec.split("_")[1], "-ar", "8000",
             "-i", str(raw), "-ar", "16000", "-ac", "1", str(out)],
            check=True, capture_output=True,
        )
        raw.unlink(missing_ok=True)
        return out
