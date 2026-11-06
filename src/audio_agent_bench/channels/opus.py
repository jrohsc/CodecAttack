"""Opus codec eval — wraps the `apply_opus_compression` ffmpeg utility from `demo_ota.py`."""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

# Make parent codec_attack/ importable so we can reuse demo_ota.apply_opus_compression
_PARENT = "${REPO_ROOT}"
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


class OpusChannel:
    def __init__(self, bitrate_kbps: int = 24) -> None:
        self.bitrate_kbps = bitrate_kbps

    def tag(self) -> str:
        return f"opus{self.bitrate_kbps}k"

    def apply(self, in_wav, out_wav) -> Path:
        from demo_ota import apply_opus_compression  # type: ignore[import]

        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        wav_np, sr = sf.read(str(in_wav))
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        opus_np = apply_opus_compression(wav_np, sr, self.bitrate_kbps)
        sf.write(str(out), opus_np, sr)
        return out
