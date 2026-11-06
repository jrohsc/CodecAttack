"""Gilbert-Elliott packet-loss simulator on 20 ms RTP frames.

TODO (Phase 3): implement with Opus PLC — encode, drop frames, decode with concealment.
For now a zero-fill fallback is OK as a skeleton.
"""
from __future__ import annotations

from pathlib import Path


class PacketLossChannel:
    def __init__(self, p_loss: float = 0.03, burst: bool = False) -> None:
        self.p_loss = p_loss
        self.burst = burst

    def tag(self) -> str:
        return f"ploss{int(self.p_loss*100)}pct" + ("_burst" if self.burst else "")

    def apply(self, in_wav, out_wav) -> Path:
        raise NotImplementedError("Phase 3: Opus + PLC frame drop")
