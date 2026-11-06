from __future__ import annotations

import shutil
from pathlib import Path


class CleanChannel:
    def tag(self) -> str:
        return "clean"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        if Path(in_wav).resolve() != out.resolve():
            shutil.copy(in_wav, out)
        return out
