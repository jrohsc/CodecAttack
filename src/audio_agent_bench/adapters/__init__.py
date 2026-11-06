"""Model adapters — thin wrappers over `external/whisper-inject-v2-fork/models/*.py`.

All adapters implement:

    respond(system_prompt: str, audio_path: str, history: list[dict], tools: list[ToolSchema] | None) -> str
    name() -> str
"""
from .base import Adapter

__all__ = ["Adapter", "get_adapter"]

_ADAPTER_CACHE: dict[str, Adapter] = {}


def get_adapter(name: str) -> Adapter:
    if name in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[name]
    if name == "qwen2_audio":
        from .qwen2_audio import Qwen2AudioAdapter
        _ADAPTER_CACHE[name] = Qwen2AudioAdapter()
    elif name == "audio_flamingo":
        from .audio_flamingo import AudioFlamingoAdapter
        _ADAPTER_CACHE[name] = AudioFlamingoAdapter()
    elif name == "qwen25_omni":
        from .qwen25_omni import Qwen25OmniAdapter
        _ADAPTER_CACHE[name] = Qwen25OmniAdapter()
    else:
        raise ValueError(f"unknown adapter: {name}")
    return _ADAPTER_CACHE[name]
