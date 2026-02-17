"""
Audio Flamingo 3 model wrapper for adversarial attacks.

Uses model.forward() with differentiable mel features:
1. WAV -> MEL spectrogram (differentiable, Whisper-style normalization)
2. Processor generates input_ids with expanded <sound> tokens
3. model.forward(input_ids, input_features, input_features_mask, labels)
"""

import torch
import numpy as np
from typing import Optional

from models.base import BaseAudioModel


class AudioFlamingoModel(BaseAudioModel):
    """
    Wrapper for NVIDIA Audio Flamingo 3 model.

    Implements the BaseAudioModel interface for use with attack algorithms.
    """

    MODEL_PATH = '/datasets/ai/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7'
    SAMPLE_RATE = 16000
    MEL_LENGTH = 3000  # 30s chunk * 16kHz / 160 hop_length

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        self._device = device
        self._dtype = dtype

        if model_path is None:
            model_path = self.MODEL_PATH

        print(f"Loading Audio Flamingo 3 model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        from transformers import AutoProcessor, AutoModel

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device
        ).eval()

        self.tokenizer = self.processor.tokenizer

        # Whisper-style mel: use processor's mel filterbank
        mel_filters_np = self.processor.feature_extractor.mel_filters  # (201, 128)
        self.mel_filters = torch.from_numpy(mel_filters_np.T).float().to(device)  # (128, 201)
        self.hann_window = torch.hann_window(400).to(device)

        print("Audio Flamingo 3 model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # Target waveform length: 30s * 16kHz = 480000 samples
    WAV_LENGTH = 480000

    def wav_to_mel(self, wav: torch.Tensor, pad_to_length: int = None) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram (differentiable).

        Matches WhisperFeatureExtractor exactly by zero-padding the raw
        waveform to 30s BEFORE computing STFT (not padding mel after).

        Args:
            wav: Audio waveform [1, T] or [T]
            pad_to_length: Pad mel to this length (default: MEL_LENGTH)

        Returns:
            Log-mel spectrogram [1, n_mels, T']
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self._device, dtype=torch.float32)

        # Zero-pad raw waveform to 30s (matches processor behavior exactly)
        wav_1d = wav.squeeze(0)
        if wav_1d.shape[0] < self.WAV_LENGTH:
            pad_zeros = torch.zeros(
                self.WAV_LENGTH - wav_1d.shape[0],
                dtype=wav_1d.dtype, device=wav_1d.device
            )
            wav_1d = torch.cat([wav_1d, pad_zeros])
        elif wav_1d.shape[0] > self.WAV_LENGTH:
            wav_1d = wav_1d[:self.WAV_LENGTH]

        # STFT -> magnitude squared
        stft = torch.stft(
            wav_1d, n_fft=400, hop_length=160,
            window=self.hann_window, return_complex=True
        )
        magnitudes = stft.abs() ** 2  # [n_freqs, T]

        # Apply mel filterbank
        mel_spec = self.mel_filters @ magnitudes  # [128, T]

        # Whisper-style log normalization (differentiable)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Drop last frame (matches processor behavior)
        log_spec = log_spec[:, :-1]

        # Add batch dimension: [1, 128, T]
        mel = log_spec.unsqueeze(0)

        # Truncate if needed (should be exactly MEL_LENGTH=3000 now)
        if pad_to_length is None:
            pad_to_length = self.MEL_LENGTH
        if mel.shape[2] > pad_to_length:
            mel = mel[:, :, :pad_to_length]

        return mel

    DEFAULT_PROMPT = "Answer the question in the audio."

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        prompt: str = None,
    ) -> str:
        """Generate text from audio."""
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        with torch.no_grad():
            wav_np = wav.detach().squeeze().cpu().numpy()

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": wav_np},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = self.processor(
                text=text,
                audio=[wav_np],
                return_tensors="pt",
                sampling_rate=self.SAMPLE_RATE
            )
            inputs = {
                k: v.to(self.model.device, dtype=self._dtype) if v.is_floating_point() else v.to(self.model.device)
                for k, v in inputs.items()
            }

            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
            )

            gen_ids = gen_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                gen_ids,
                skip_special_tokens=True
            )[0]

        return response.strip()

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        prompt: str = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text (differentiable).

        Uses the model's own forward() method with:
        - input_ids from processor (with expanded <sound> tokens)
        - Differentiable mel features as input_features
        - input_features_mask from processor
        - Labels on target text tokens only
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        # 1. Compute differentiable mel features
        mel = self.wav_to_mel(wav, pad_to_length=self.MEL_LENGTH)
        input_features = mel.to(self._dtype)  # [1, 128, 3000]

        # 2. Use processor to get input_ids with expanded <sound> tokens
        wav_np = wav.detach().squeeze().cpu().numpy()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_np},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audio=[wav_np],
            return_tensors="pt", sampling_rate=self.SAMPLE_RATE
        )

        input_ids = inputs['input_ids'].to(self._device)
        attention_mask = inputs['attention_mask'].to(self._device)
        input_features_mask = inputs['input_features_mask'].to(self._device)

        # 3. Append target text tokens
        target_ids = self.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self._device)

        full_input_ids = torch.cat([input_ids, target_ids], dim=1)
        full_attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(target_ids)
        ], dim=1)

        # 4. Create labels: -100 for prompt, actual ids for target
        labels = torch.full_like(full_input_ids, -100)
        labels[0, -target_ids.shape[1]:] = target_ids[0]

        # 5. Forward pass through the FULL model
        outputs = self.model(
            input_ids=full_input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=full_attention_mask,
            labels=labels,
        )

        return outputs.loss

    def compute_margin_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        kappa: float = 5.0,
        early_weight: float = 5.0
    ) -> torch.Tensor:
        """Compute margin loss (C&W style)."""
        return self.compute_loss(wav, target_text)
