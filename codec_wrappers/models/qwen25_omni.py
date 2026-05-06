"""
Qwen2.5-Omni model wrapper for adversarial attacks.

Qwen2.5-Omni has a thinker-talker architecture. For adversarial attacks
we use the thinker (text generation) directly, which has the same
forward interface as Qwen2-Audio (input_ids, input_features, labels, etc).

Audio features are computed differentiably via wav_to_mel() so gradients
flow back to the input audio tensor.
"""

import warnings
import logging

import torch
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor

# Suppress noisy warnings from transformers / torch
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


SAMPLE_RATE = 16000
# Qwen2.5-Omni uses chunk_length=300 -> nb_max_frames=30000
# For 10s audio at hop_length=160: 10*16000/160 = 1000 frames
# Pad to match processor output shape
MEL_LENGTH = 30000


class Qwen25OmniModel:
    """
    Wrapper around Qwen2.5-Omni for adversarial attack use.

    Uses the thinker sub-model for compute_loss() (teacher-forced CE loss)
    and the full model for generate().
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Enable gradient checkpointing to reduce memory during backward pass
        self.model.gradient_checkpointing_enable()

        # The thinker sub-model supports standard forward(input_ids, ..., labels)
        self.thinker = self.model.thinker

        # Free the talker (~2.5 GiB bf16). We only need text generation for
        # adversarial attacks; the audio-output path is never invoked. Without
        # this the 7B model + activations OOMs on a 40 GB A100 at 15 s carrier.
        if hasattr(self.model, "talker") and self.model.talker is not None:
            self.model.talker = None
            self.model.has_talker = False
            torch.cuda.empty_cache()

        # Extract mel filters from the feature extractor for differentiable computation
        fe = self.processor.feature_extractor
        self._mel_filters = torch.from_numpy(fe.mel_filters).float().to(device)
        self._n_fft = fe.n_fft
        self._hop_length = fe.hop_length

    def _prepare_inputs(self, audio: torch.Tensor, prompt: str = None):
        """Prepare model inputs from raw audio tensor."""
        if prompt is None:
            prompt = "What does the person say in this audio?"

        audio_np = audio.detach().cpu().squeeze().numpy()

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_np},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=text,
            audio=[audio_np],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def wav_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Whisper-style log-mel spectrogram.

        Uses the same parameters as the model's feature extractor
        (n_fft=400, hop_length=160, 128 mel bands).

        Args:
            audio: [T] at 16kHz

        Returns:
            [n_mels, T'] mel spectrogram (differentiable)
        """
        window = torch.hann_window(self._n_fft, device=audio.device)
        stft = torch.stft(
            audio, n_fft=self._n_fft, hop_length=self._hop_length,
            window=window, return_complex=True,
        )
        magnitudes = stft.abs() ** 2

        # Use the exact mel filters from the feature extractor
        # mel_filters shape: [n_freqs, n_mels] -> transpose to [n_mels, n_freqs]
        mel_filters = self._mel_filters.T.to(audio.device)
        mel_spec = torch.matmul(mel_filters, magnitudes)

        # Whisper-style log normalization
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def compute_loss(self, audio: torch.Tensor, target_text: str,
                     prompt: str = None) -> torch.Tensor:
        """
        Compute cross-entropy loss for target text given audio input.

        Audio features are computed differentiably via wav_to_mel() so
        gradients flow back to the input audio tensor.

        Args:
            audio: [1, T] at 16kHz (can have gradients)
            target_text: Text the model should output
            prompt: Text prompt

        Returns:
            Scalar loss tensor (differentiable)
        """
        if prompt is None:
            prompt = "What does the person say in this audio?"

        audio_squeezed = audio.squeeze()

        # Compute mel spectrogram differentiably
        input_features = self.wav_to_mel(audio_squeezed)

        # Use processor for text tokenization (with detached audio for template)
        audio_np = audio_squeezed.detach().cpu().numpy()

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_np},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        text_inputs = self.processor(
            text=text,
            audio=[audio_np],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        input_ids = text_inputs["input_ids"].to(self.device)

        # Tokenize target
        target_ids = self.processor.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        full_ids = torch.cat([input_ids, target_ids], dim=1)
        full_attention = torch.ones_like(full_ids)

        labels = torch.full_like(full_ids, -100)
        labels[:, input_ids.shape[1]:] = target_ids

        # Pad/truncate mel features to expected length
        if input_features.shape[-1] < MEL_LENGTH:
            pad = MEL_LENGTH - input_features.shape[-1]
            input_features = torch.nn.functional.pad(input_features, (0, pad))
        else:
            input_features = input_features[..., :MEL_LENGTH]

        # Add batch dim: [1, n_mels, T]
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        # Build feature_attention_mask from processor output
        feature_attention_mask = text_inputs.get("feature_attention_mask")
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(self.device)

        # Get audio_feature_lengths if present
        audio_feature_lengths = text_inputs.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            audio_feature_lengths = audio_feature_lengths.to(self.device)

        # Use thinker sub-model (has standard forward with input_ids, labels)
        outputs = self.thinker(
            input_ids=full_ids,
            attention_mask=full_attention,
            input_features=input_features.to(self.dtype),
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            labels=labels,
        )
        return outputs.loss

    def generate(self, audio: torch.Tensor, prompt: str = None,
                 max_new_tokens: int = 128) -> str:
        """
        Generate text from audio input.

        Args:
            audio: [1, T] or [T] at 16kHz
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        inputs = self._prepare_inputs(audio, prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=max_new_tokens,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()
