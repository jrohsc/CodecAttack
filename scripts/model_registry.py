"""
Single source of truth for "name -> model wrapper".

Before this file the same if/elif chain was duplicated in ensemble_attack.py,
gradient_worker.py and gradient_worker_persistent.py. They had already drifted
(only the persistent worker knew about qwen25_omni_3b), and a surrogate that is
missing from one of the three silently never contributes a gradient. Growing
the pool from 2 to 6 makes that failure mode much more likely, so all three now
delegate here.

DIFFERENTIABLE marks which models can act as ensemble SURROGATES. A model that
is inference-only (kimi_audio) can still be a held-out VICTIM.
"""

import os
import sys

# Models whose compute_loss() is differentiable w.r.t. the input waveform,
# i.e. which can serve as ensemble surrogates.
DIFFERENTIABLE = {
    "qwen2_audio", "qwen25_omni", "qwen25_omni_3b", "audio_flamingo", "phi4",
}

# Inference-only: usable as a held-out victim, never as a surrogate.
INFERENCE_ONLY = {"kimi_audio"}

SUPPORTED = DIFFERENTIABLE | INFERENCE_ONLY

# Audio front-end lineage. The whole point of scaling the surrogate pool is
# encoder diversity, so this is recorded explicitly rather than left implicit
# in the model names.
ENCODER_FAMILY = {
    "qwen2_audio": "whisper",
    "qwen25_omni": "whisper",
    "qwen25_omni_3b": "whisper",
    "audio_flamingo": "whisper",
    "kimi_audio": "whisper",
    "phi4": "conformer",   # SpeechLib log-fbank -> Conformer trained from scratch
}

LLM_BACKBONE = {
    "qwen2_audio": "qwen2-7b",
    "qwen25_omni": "qwen2.5-7b",
    "qwen25_omni_3b": "qwen2.5-3b",
    "audio_flamingo": "qwen2.5-7b",
    "kimi_audio": "qwen2.5-7b",
    "phi4": "phi4-mini-3.8b",
}


def ensure_paths():
    """Put scripts/ and codec_wrappers/ on sys.path.

    ORDER MATTERS: codec_wrappers/ must come first so that the `models`
    package resolves to codec_wrappers/models/.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    wrappers = os.path.join(os.path.dirname(here), "codec_wrappers")
    for p in (here, wrappers):           # wrappers inserted last => ends up first
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return wrappers


def load_model(model_name: str, device: str = "cuda", dtype=None):
    """Instantiate a model wrapper by name."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    ensure_paths()

    from config import MODEL_PATHS
    model_path = MODEL_PATHS.get(model_name)

    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        return Qwen2AudioModel(model_path=model_path, device=device, dtype=dtype)
    if model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        return AudioFlamingoModel(model_path=model_path, device=device, dtype=dtype)
    if model_name == "kimi_audio":
        from models.kimi_audio import KimiAudioModel
        return KimiAudioModel(model_path=model_path, device=device, dtype=dtype)
    if model_name in ("qwen25_omni", "qwen25_omni_3b"):
        from models.qwen25_omni import Qwen25OmniModel
        return Qwen25OmniModel(model_path=model_path, device=device, dtype=dtype)
    if model_name == "phi4":
        from models.phi4_mm import Phi4MMModel
        return Phi4MMModel(model_path=model_path, device=device, dtype=dtype)
    raise ValueError(f"Unknown model: {model_name} (supported: {sorted(SUPPORTED)})")


# Models eligible to be scored as held-out victims.
VICTIM_ELIGIBLE = SUPPORTED
