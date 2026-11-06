"""Audio Flamingo 3 adapter. Requires `flamingo3` conda env (transformers>=5.0)."""
from __future__ import annotations

import sys
from typing import Any

_FORK = "${PROJECT_ROOT}/external/whisper-inject-v2-fork"
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)


def _build_prompt(system_prompt: str, history: list[dict[str, Any]] | None) -> str:
    # Do NOT append a "Listen to the audio and respond." suffix — attacker
    # prompts are passed verbatim (no suffix), so adding one here misaligns
    # the eval prompt with the optimization prompt and collapses success.
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    for turn in history or []:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


class AudioFlamingoAdapter:
    def __init__(self, model_path: str | None = None, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def name(self) -> str:
        return "audio_flamingo"

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from models.audio_flamingo import AudioFlamingoModel  # type: ignore[import]
        self._model = AudioFlamingoModel(
            model_path=self.model_path, device=self.device, dtype=torch.bfloat16,
        )

    def respond(
        self,
        system_prompt: str,
        audio_path: str,
        history: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> str:
        self._lazy_load()
        import soundfile as sf
        import torch
        import torchaudio

        wav_np, sr = sf.read(audio_path)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        wav = torch.from_numpy(wav_np).float().unsqueeze(0)
        target_sr = self._model.SAMPLE_RATE
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.to(self._model.device)

        prompt = _build_prompt(system_prompt, history)
        return self._model.generate(wav, prompt=prompt, max_new_tokens=128).strip()
