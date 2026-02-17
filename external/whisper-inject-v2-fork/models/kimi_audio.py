"""
Kimi-Audio-7B-Instruct model wrapper for adversarial attacks.

Supports both:
- generate(): Text generation from audio (eval-only, via KimiAudio API)
- compute_loss(): Differentiable loss for gradient-based attacks

Gradient flow for compute_loss():
  wav -> mel (differentiable, Whisper-style) -> Whisper encoder (gradients enabled)
  -> reshape [1,T,1280] -> [1,T//4,5120] -> model.forward() with fixed discrete tokens
  -> text_logits -> CE loss on target text

Discrete audio tokens are computed once from the original (unperturbed) audio
and cached. Only the continuous Whisper features are recomputed differentiably
on each call.

Requires: kimi-audio conda env (flash_attn).
"""

import os
import sys
import tempfile

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.base import BaseAudioModel

# Add Kimi-Audio code to path for kimia_infer package
_KIMI_CODE = "/datasets/ai/moonshot/Kimi-Audio"
if _KIMI_CODE not in sys.path:
    sys.path.insert(0, _KIMI_CODE)


class KimiAudioModel(BaseAudioModel):
    """
    Wrapper for Kimi-Audio-7B-Instruct with differentiable loss computation.

    Uses fixed discrete audio tokens (from original audio) + differentiable
    continuous Whisper features (recomputed each step) for gradient-based attacks.
    """

    MODEL_PATH = "/datasets/ai/moonshot/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
    SAMPLE_RATE = 16000
    WAV_LENGTH = 480000  # 30s * 16kHz
    N_FFT = 400
    HOP_LENGTH = 160

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None,
    ):
        self._device = device
        self._dtype = dtype

        if model_path is None:
            model_path = self.MODEL_PATH

        print(f"Loading Kimi-Audio model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        # Load full KimiAudio API (loads main model + prompt manager + whisper)
        from kimia_infer.api.kimia import KimiAudio
        self.kimi = KimiAudio(model_path=model_path, load_detokenizer=False)

        # Access components directly
        self.model = self.kimi.alm  # MoonshotKimiaForCausalLM
        self.prompt_manager = self.kimi.prompt_manager
        self.whisper_encoder = self.prompt_manager.whisper_model.speech_encoder
        self.text_tokenizer = self.prompt_manager.text_tokenizer

        # Load mel filterbank (128 bins, matching Whisper Large-V3)
        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import mel_filters
        self.mel_filters_tensor = mel_filters(device, n_mels=128)  # [128, 201]
        self.hann_window = torch.hann_window(self.N_FFT).to(device)

        # Cache for discrete tokens + prompt structure
        self._cached_prompt = None

        print("Kimi-Audio model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def reset_cache(self):
        """Reset cached prompt structure (call when music carrier changes)."""
        self._cached_prompt = None

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable log-mel spectrogram (Whisper-style, 128 bins).

        Matches kimia_infer's log_mel_spectrogram() exactly.

        Args:
            wav: Audio waveform [1, T] or [T] at 16kHz

        Returns:
            Log-mel spectrogram [128, 3000]
        """
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        wav = wav.to(self._device, dtype=torch.float32)

        # Zero-pad to 30s (matches Whisper's pad_or_trim)
        if wav.shape[0] < self.WAV_LENGTH:
            wav = torch.cat([
                wav,
                torch.zeros(self.WAV_LENGTH - wav.shape[0], dtype=wav.dtype, device=wav.device)
            ])
        elif wav.shape[0] > self.WAV_LENGTH:
            wav = wav[:self.WAV_LENGTH]

        # STFT
        stft = torch.stft(
            wav, self.N_FFT, self.HOP_LENGTH,
            window=self.hann_window, return_complex=True
        )
        # Drop last time frame (matches Kimi's log_mel_spectrogram)
        magnitudes = stft[..., :-1].abs() ** 2  # [201, 3000]

        # Mel filterbank
        mel_spec = self.mel_filters_tensor @ magnitudes  # [128, 3000]

        # Whisper-style log normalization (differentiable)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec  # [128, 3000]

    def wav_to_whisper_features(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable Whisper continuous features.

        Bypasses WhisperEncoder.tokenize_waveform() which has @torch.no_grad().
        Instead calls the speech_encoder directly with gradients enabled.

        Args:
            wav: Audio waveform [1, T] or [T] at 16kHz

        Returns:
            Whisper features [1, T//4, 5120] (reshaped from [1, T, 1280])
        """
        if wav.dim() == 2:
            wav_1d = wav.squeeze(0)
        else:
            wav_1d = wav

        # Compute audio length for token_len calculation (before padding)
        L = wav_1d.shape[0]
        token_len = (L - 1) // (self.HOP_LENGTH * 8) + 1

        # Compute mel spectrogram (differentiable)
        mel = self.wav_to_mel(wav_1d)  # [128, 3000]

        # Pass through Whisper encoder WITH gradients
        # (speech_encoder is in eval mode but we don't use @torch.no_grad())
        mel_input = mel.unsqueeze(0).to(self._dtype)  # [1, 128, 3000]
        encoder_output = self.whisper_encoder(
            mel_input, return_dict=True
        ).last_hidden_state  # [1, 1500, 1280]

        # Truncate to actual audio length
        encoder_output = encoder_output[:, :token_len * 4, :]  # [1, T, 1280]

        # Reshape: concatenate every 4 frames → [1, T//4, 5120]
        T = encoder_output.shape[1]
        # Ensure T is divisible by 4
        T_aligned = (T // 4) * 4
        encoder_output = encoder_output[:, :T_aligned, :]
        whisper_features = encoder_output.reshape(
            1, T_aligned // 4, 1280 * 4
        )  # [1, T//4, 5120]

        return whisper_features

    def _get_cached_prompt(self, wav: torch.Tensor, prompt: str):
        """
        Get prompt structure with discrete tokens (cached).

        On first call, saves wav to temp file and uses KimiAPromptManager to
        construct the full prompt. Caches the result for subsequent calls.
        """
        if self._cached_prompt is not None:
            return self._cached_prompt

        import soundfile as sf

        # Save wav to temp file for discrete tokenization
        wav_np = wav.detach().squeeze().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_np, self.SAMPLE_RATE)
            tmp_path = f.name

        try:
            # Build messages (text prompt first, then audio)
            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": tmp_path},
            ]

            # Get prompt with discrete tokens + whisper features
            content = self.prompt_manager.get_prompt(messages, output_type="text")
            self._cached_prompt = content
        finally:
            os.unlink(tmp_path)

        return self._cached_prompt

    DEFAULT_PROMPT = "What action is being requested in this audio?"

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        prompt: str = None,
    ) -> str:
        """Generate text from audio (eval-only, via KimiAudio API)."""
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        import soundfile as sf

        wav_np = wav.detach().squeeze().cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_np, self.SAMPLE_RATE)
            tmp_path = f.name

        try:
            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": tmp_path},
            ]

            _, text = self.kimi.generate(
                messages,
                output_type="text",
                text_temperature=0.0,
                text_top_k=5,
                audio_temperature=0.8,
                audio_top_k=10,
            )
            return text.strip() if text else ""
        finally:
            os.unlink(tmp_path)

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text (differentiable).

        Uses cached discrete tokens from original audio + differentiable
        Whisper continuous features from the (perturbed) waveform.

        Args:
            wav: Audio waveform [1, T] at 16kHz
            target_text: Target text to force the model to output
            prompt: Text prompt (default: DEFAULT_PROMPT)

        Returns:
            Loss tensor (scalar, with grad_fn for backprop)
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        # 1. Get cached prompt structure (discrete tokens + template)
        content = self._get_cached_prompt(wav, prompt)

        # 2. Compute differentiable Whisper features from (perturbed) waveform
        whisper_features = self.wav_to_whisper_features(wav)  # [1, T//4, 5120]

        # 3. Convert prompt to tensors
        audio_ids, text_ids, is_continuous_mask, _, _ = content.to_tensor()
        audio_ids = audio_ids.to(self._device)        # [1, seq_len]
        text_ids = text_ids.to(self._device)           # [1, seq_len]
        is_continuous_mask = is_continuous_mask.to(self._device)  # [1, seq_len]

        # 4. Tokenize target text and append
        target_token_ids = self.text_tokenizer.encode(target_text, bos=False, eos=False)
        target_tokens = torch.tensor(
            [target_token_ids], dtype=torch.long, device=self._device
        )
        n_target = target_tokens.shape[1]

        # For target text positions: audio stream gets blank token, text stream gets target
        blank_token = self.prompt_manager.extra_tokens.kimia_text_blank
        audio_pad = torch.full(
            (1, n_target), blank_token, dtype=torch.long, device=self._device
        )
        mask_pad = torch.zeros(
            (1, n_target), dtype=torch.bool, device=self._device
        )

        full_audio_ids = torch.cat([audio_ids, audio_pad], dim=1)
        full_text_ids = torch.cat([text_ids, target_tokens], dim=1)
        full_mask = torch.cat([is_continuous_mask, mask_pad], dim=1)

        # 5. Create attention mask (all ones)
        seq_len = full_audio_ids.shape[1]
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self._device)

        # 6. Create labels: -100 for prompt, target token IDs for target portion
        #    Labels go on the TEXT stream (text_logits)
        labels_text = torch.full(
            (1, seq_len), -100, dtype=torch.long, device=self._device
        )
        labels_text[0, -n_target:] = target_tokens[0]

        # 7. Forward pass
        outputs = self.model(
            input_ids=full_audio_ids,
            text_input_ids=full_text_ids,
            whisper_input_feature=[whisper_features],
            is_continuous_mask=full_mask,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # 8. Extract text_logits and compute CE loss
        # outputs.logits = (audio_logits, text_logits)
        text_logits = outputs.logits[1]  # [1, seq_len, vocab_size]

        # Shift for next-token prediction
        shift_logits = text_logits[:, :-1, :].contiguous()
        shift_labels = labels_text[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def compute_margin_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        kappa: float = 5.0,
        early_weight: float = 5.0,
    ) -> torch.Tensor:
        """Compute margin loss (C&W style) — falls back to CE loss."""
        return self.compute_loss(wav, target_text)
