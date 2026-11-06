from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Channel(Protocol):
    def tag(self) -> str: ...
    def apply(self, in_wav: str | Path, out_wav: str | Path) -> Path: ...


def apply_channel(ch: Channel, in_wav: str | Path, out_wav: str | Path) -> Path:
    return ch.apply(in_wav, out_wav)
