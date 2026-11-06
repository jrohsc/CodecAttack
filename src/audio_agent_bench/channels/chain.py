"""Compose channels in order. Example: opus_24k -> g711 simulates a PSTN bridge."""
from __future__ import annotations

from pathlib import Path

from . import get_channel


class ChainChannel:
    def __init__(self, steps: list[dict]) -> None:
        """steps: list of {"kind": "opus", "params": {"bitrate_kbps": 24}}."""
        self.children = [get_channel(s["kind"], **s.get("params", {})) for s in steps]

    def tag(self) -> str:
        return "chain[" + ">".join(c.tag() for c in self.children) + "]"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        current = Path(in_wav)
        for i, ch in enumerate(self.children):
            step_out = out.with_name(f"{out.stem}__step{i}_{ch.tag()}.wav")
            current = ch.apply(current, step_out)
        # rename final step to out_wav
        current.replace(out)
        return out
