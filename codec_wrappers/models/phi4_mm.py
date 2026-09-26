"""
Phi-4-multimodal-instruct wrapper with a differentiable loss path.

Why this model is in the ensemble: its audio front-end is NOT
Whisper-derived. Phi-4-MM feeds an 80-bin SpeechLib log-filterbank into a
Conformer encoder trained from scratch, on a Phi-4-mini (3.8B) backbone.
Every other model here (Qwen2-Audio, Qwen2.5-Omni, Kimi-Audio,
Audio-Flamingo-3) wraps a Whisper-large-v3 encoder.

Differentiable path mirrors models/qwen2_audio.py: run the real processor
once (cached) to fix the token layout, then substitute a differentiable
reimplementation of the feature extractor for `input_audio_embeds`.

Weights: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
(override with $MODEL_PATH_PHI4_MM to use a local snapshot).

Env: `audio-llm` (transformers 4.47 / peft 0.11). The shipped remote code
calls get_peft_model on the base model at __init__ and modern peft (>=0.14)
raises `Phi4MMModel has no attribute prepare_inputs_for_generation`, so this
wrapper must run out-of-process from the main codec-attack env.
"""

import os
import sys
import torch
from typing import Optional

from models.base import BaseAudioModel

# models/ -> codec_wrappers/, which holds phi4_frontend.py
_FRONTENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTENDS not in sys.path:
    sys.path.insert(0, _FRONTENDS)


class Phi4MMModel(BaseAudioModel):

    SAMPLE_RATE = 16000
    DEFAULT_PROMPT = "What does the person say in this audio?"
    # Phi-4-MM chat format; <|audio_1|> is expanded by the processor into
    # audio_embed_size copies of the audio placeholder id.
    PROMPT_TEMPLATE = "<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
    # The checkpoint's eos_token_id is <|endoftext|> (199999), NOT the turn
    # terminator <|end|> (200020), so generate() runs straight past the end of
    # the assistant turn and hallucinates a follow-on dialogue. Left unfixed
    # that inflates WER far above 1.0 and silently breaks the evaluator's
    # "exact OR wer<=0.5" success criterion for every Phi-4 victim row.
    STOP_TOKEN_IDS = [200020, 199999]
    DEFAULT_MODEL_PATH = os.environ.get(
        "MODEL_PATH_PHI4_MM", "microsoft/Phi-4-multimodal-instruct"
    )

    def __init__(self, model_path: str = None, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, token: Optional[str] = None):
        self._device = device
        self._dtype = dtype
        model_path = model_path or self.DEFAULT_MODEL_PATH

        from transformers import AutoProcessor, AutoModelForCausalLM
        print(f"Loading Phi-4-multimodal from: {model_path}")

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=os.path.isdir(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
            trust_remote_code=True, local_files_only=os.path.isdir(model_path),
            _attn_implementation="eager").eval()
        self.tokenizer = self.processor.tokenizer

        from phi4_frontend import Phi4DifferentiableFrontend
        self.frontend = Phi4DifferentiableFrontend.from_feature_extractor(
            self.processor.audio_processor, device=device)
        print("Phi-4-multimodal loaded successfully!")

    @property
    def sample_rate(self) -> int: return self.SAMPLE_RATE
    @property
    def device(self) -> str: return self._device
    @property
    def dtype(self) -> torch.dtype: return self._dtype

    def generate(self, wav: torch.Tensor, max_tokens: int = 100, prompt: str = None,
                 do_sample: bool = False, temperature: float = 1.0,
                 max_new_tokens: int = None, **kw) -> str:
        prompt = prompt or self.DEFAULT_PROMPT
        if max_new_tokens is not None:
            max_tokens = max_new_tokens
        with torch.no_grad():
            wav_np = wav.detach().squeeze().float().cpu().numpy()
            inputs = self.processor(
                text=self.PROMPT_TEMPLATE.format(prompt=prompt),
                audios=[(wav_np, self.SAMPLE_RATE)], return_tensors="pt",
            ).to(self._device)
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None,
                eos_token_id=self.STOP_TOKEN_IDS)
            out = out[:, inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()

    _loss_cache_key = None
    _loss_cache = None

    def reset_cache(self):
        self._loss_cache_key = None
        self._loss_cache = None

    def _get_cached_inputs(self, wav, target_text, prompt):
        """Token layout depends only on (prompt, target, wav length), not content."""
        key = (prompt, target_text, int(wav.shape[-1]))
        if self._loss_cache_key == key and self._loss_cache is not None:
            return self._loss_cache

        wav_np = wav.detach().squeeze().float().cpu().numpy()
        inputs = self.processor(
            text=self.PROMPT_TEMPLATE.format(prompt=prompt),
            audios=[(wav_np, self.SAMPLE_RATE)], return_tensors="pt")

        input_ids = inputs["input_ids"].to(self._device)
        attn = inputs["attention_mask"].to(self._device)
        audio_embed_sizes = inputs["audio_embed_sizes"].to(self._device)
        ref_frames = int(inputs["input_audio_embeds"].shape[1])

        target_ids = self.tokenizer(target_text, return_tensors="pt",
                                    add_special_tokens=False).input_ids.to(self._device)
        full_ids = torch.cat([input_ids, target_ids], dim=1)
        full_attn = torch.cat([attn, torch.ones_like(target_ids)], dim=1)
        labels = torch.full_like(full_ids, -100)
        labels[0, -target_ids.shape[1]:] = target_ids[0]

        self._loss_cache_key = key
        self._loss_cache = (full_ids, full_attn, audio_embed_sizes, labels, ref_frames)
        return self._loss_cache

    def compute_loss(self, wav: torch.Tensor, target_text: str,
                     prompt: str = None, **kw) -> torch.Tensor:
        prompt = prompt or self.DEFAULT_PROMPT
        full_ids, full_attn, audio_embed_sizes, labels, ref_frames = \
            self._get_cached_inputs(wav, target_text, prompt)

        feats = self.frontend(wav)  # [n_frames, 80], differentiable
        if feats.shape[0] != ref_frames:
            raise RuntimeError(
                f"frame count mismatch: differentiable frontend produced {feats.shape[0]}, "
                f"processor produced {ref_frames}. Audio token layout would be wrong.")
        input_audio_embeds = feats.unsqueeze(0).to(self._dtype)

        out = self.model(
            input_ids=full_ids,
            attention_mask=full_attn,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            input_mode=2,  # speech
            labels=labels,
        )
        return out.loss
