"""Base adapter protocol."""
from __future__ import annotations

from typing import Any, Protocol


class Adapter(Protocol):
    def name(self) -> str: ...

    def respond(
        self,
        system_prompt: str,
        audio_path: str,
        history: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> str: ...
