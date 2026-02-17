"""
Qwen2-Audio model wrapper for adversarial attacks.

Differentiable pipeline: WAV -> MEL -> audio encoder -> LM forward -> loss
"""

import os
import torch
import torchaudio
import numpy as np
from typing import Optional

from models.base import BaseAudioModel


class Qwen2AudioModel(BaseAudioModel):
    """Wrapper for Qwen2-Audio with differentiable loss computation."""

    SAMPLE_RATE = 16000
    MEL_LENGTH = 3000
    WAV_LENGTH = 480000  # 30s * 16kHz

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._device = device
        self._dtype = dtype

        local_files_only = os.path.isdir(model_path)

        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=local_files_only, trust_remote_code=True,
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
            local_files_only=local_files_only, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

        self.tokenizer = self.processor.tokenizer

        mel_filters_np = np.array(self.processor.feature_extractor.mel_filters)
        self.mel_filters = torch.from_numpy(mel_filters_np.T).float().to(device)
        self.hann_window = torch.hann_window(400).to(device)

        print(f"Qwen2-Audio loaded on {device}")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram (differentiable, Whisper-style)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self._device, dtype=torch.float32)

        wav_1d = wav.squeeze(0)
        if wav_1d.shape[0] < self.WAV_LENGTH:
            wav_1d = torch.cat([wav_1d, torch.zeros(self.WAV_LENGTH - wav_1d.shape[0], device=wav_1d.device)])
        elif wav_1d.shape[0] > self.WAV_LENGTH:
            wav_1d = wav_1d[:self.WAV_LENGTH]

        stft = torch.stft(wav_1d, n_fft=400, hop_length=160, window=self.hann_window, return_complex=True)
        magnitudes = stft.abs() ** 2
        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec[:, :-1]

        mel = log_spec.unsqueeze(0)
        if mel.shape[2] > self.MEL_LENGTH:
            mel = mel[:, :, :self.MEL_LENGTH]
        return mel

    DEFAULT_PROMPT = "What does the person say in this audio?"

    def generate(self, wav: torch.Tensor, max_tokens: int = 100, prompt: str = None, **kwargs) -> str:
        """Generate text from audio (non-differentiable)."""
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        with torch.no_grad():
            wav_np = wav.detach().squeeze().cpu().numpy()
            conversation = [{"role": "user", "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=[wav_np], return_tensors="pt", padding=True, sampling_rate=self.SAMPLE_RATE)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            gen_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            gen_ids = gen_ids[:, inputs['input_ids'].shape[1]:]
            return self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def compute_loss(self, wav: torch.Tensor, target_text: str, prompt: str = None) -> torch.Tensor:
        """Compute cross-entropy loss for target text (differentiable)."""
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        mel = self.wav_to_mel(wav)
        input_features = mel.to(self._dtype)

        wav_np = wav.detach().squeeze().cpu().numpy()
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "placeholder"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[wav_np], return_tensors="pt", sampling_rate=self.SAMPLE_RATE)

        input_ids = inputs['input_ids'].to(self._device)
        attention_mask = inputs['attention_mask'].to(self._device)
        feature_attention_mask = inputs['feature_attention_mask'].to(self._device)

        target_ids = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self._device)
        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        full_attention_mask = torch.cat([attention_mask, torch.ones_like(target_ids)], dim=1)

        labels = torch.full_like(full_input_ids, -100)
        labels[0, -target_ids.shape[1]:] = target_ids[0]

        outputs = self.model(
            input_ids=full_input_ids, input_features=input_features,
            attention_mask=full_attention_mask, feature_attention_mask=feature_attention_mask,
            labels=labels,
        )
        return outputs.loss
