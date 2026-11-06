"""
WaveGuard defense: signal-level audio transforms + detection logic.

Based on "WaveGuard: Understanding and Mitigating Audio Adversarial Examples"
(USENIX Security 2021). Applies audio transformations and detects adversarial
examples by checking if model output changes significantly after transformation.

No model dependencies — purely signal processing.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio


# ============================================================================
# Dataclasses
# ============================================================================

@dataclasses.dataclass
class TransformResult:
    """Result of applying a single transform to an audio sample."""
    transform_name: str
    transformed_audio: torch.Tensor  # [1, T] at original sample rate
    snr_db: float  # signal-to-noise ratio of the transformation


@dataclasses.dataclass
class WaveGuardResult:
    """Aggregated WaveGuard detection result for one audio sample."""
    detected: bool  # True if flagged as adversarial
    num_transforms_flagged: int
    num_transforms_total: int
    per_transform: Dict[str, dict]  # transform_name -> {wer, flagged, snr_db, output_original, output_transformed}


# ============================================================================
# Signal-level transforms
# ============================================================================

class WaveGuardDefense:
    """Collection of signal-level audio transformations for WaveGuard defense."""

    # Default transform suite: name -> (method_name, kwargs)
    DEFAULT_TRANSFORMS = {
        "quantize_8bit": ("quantize", {"bits": 8}),
        "downup_8k": ("down_up_sample", {"intermediate_sr": 8000}),
        "downup_4k": ("down_up_sample", {"intermediate_sr": 4000}),
        "lowpass_4k": ("low_pass_filter", {"cutoff_freq": 4000}),
        "lowpass_2k": ("low_pass_filter", {"cutoff_freq": 2000}),
        "mel_reconstruct": ("mel_reconstruct", {"n_mels": 80}),
        "gaussian_0.001": ("add_gaussian_noise", {"sigma": 0.001}),
        "gaussian_0.005": ("add_gaussian_noise", {"sigma": 0.005}),
        "gaussian_0.01": ("add_gaussian_noise", {"sigma": 0.01}),
    }

    @staticmethod
    def quantize(audio: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Reduce bit depth and restore to float.

        Args:
            audio: [1, T] or [T] float tensor in [-1, 1]
            bits: target bit depth

        Returns:
            Quantized audio, same shape as input
        """
        levels = 2 ** bits
        # Clamp to [-1, 1], quantize, then restore
        clamped = audio.clamp(-1.0, 1.0)
        # Map to [0, levels-1]
        scaled = ((clamped + 1.0) / 2.0 * (levels - 1)).round()
        # Map back to [-1, 1]
        return scaled / (levels - 1) * 2.0 - 1.0

    @staticmethod
    def down_up_sample(
        audio: torch.Tensor,
        sample_rate: int = 24000,
        intermediate_sr: int = 8000,
    ) -> torch.Tensor:
        """Downsample then upsample back to original rate.

        Args:
            audio: [1, T] float tensor
            sample_rate: original sample rate
            intermediate_sr: intermediate (lower) sample rate

        Returns:
            Resampled audio at original sample rate, same shape
        """
        down = torchaudio.functional.resample(audio, sample_rate, intermediate_sr)
        up = torchaudio.functional.resample(down, intermediate_sr, sample_rate)
        # Match original length (resampling can drift by a sample or two)
        orig_len = audio.shape[-1]
        if up.shape[-1] > orig_len:
            up = up[..., :orig_len]
        elif up.shape[-1] < orig_len:
            up = torch.nn.functional.pad(up, (0, orig_len - up.shape[-1]))
        return up

    @staticmethod
    def low_pass_filter(
        audio: torch.Tensor,
        sample_rate: int = 24000,
        cutoff_freq: int = 4000,
    ) -> torch.Tensor:
        """Apply low-pass biquad filter.

        Args:
            audio: [1, T] float tensor
            sample_rate: sample rate of the audio
            cutoff_freq: cutoff frequency in Hz

        Returns:
            Filtered audio, same shape
        """
        return torchaudio.functional.lowpass_biquad(audio, sample_rate, cutoff_freq)

    @staticmethod
    def mel_reconstruct(
        audio: torch.Tensor,
        sample_rate: int = 24000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> torch.Tensor:
        """Convert to mel spectrogram and reconstruct via GriffinLim.

        Args:
            audio: [1, T] float tensor
            sample_rate: sample rate
            n_mels: number of mel bins
            n_fft: FFT size
            hop_length: hop length for STFT

        Returns:
            Reconstructed audio, same shape
        """
        orig_len = audio.shape[-1]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).to(audio.device)

        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        ).to(audio.device)

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
        ).to(audio.device)

        # audio [1, T] -> mel [1, n_mels, frames] -> stft [1, n_fft//2+1, frames] -> wav [1, T']
        mel = mel_transform(audio)
        stft_approx = inverse_mel(mel)
        reconstructed = griffin_lim(stft_approx)

        # Match original length
        if reconstructed.shape[-1] > orig_len:
            reconstructed = reconstructed[..., :orig_len]
        elif reconstructed.shape[-1] < orig_len:
            reconstructed = torch.nn.functional.pad(
                reconstructed, (0, orig_len - reconstructed.shape[-1])
            )
        return reconstructed

    @staticmethod
    def add_gaussian_noise(audio: torch.Tensor, sigma: float = 0.005) -> torch.Tensor:
        """Add Gaussian noise.

        Args:
            audio: [1, T] float tensor
            sigma: standard deviation of noise

        Returns:
            Noisy audio, same shape
        """
        noise = torch.randn_like(audio) * sigma
        return (audio + noise).clamp(-1.0, 1.0)

    @classmethod
    def apply_transform(
        cls,
        audio: torch.Tensor,
        transform_name: str,
        sample_rate: int = 24000,
    ) -> TransformResult:
        """Apply a named transform and compute SNR.

        Args:
            audio: [1, T] float tensor
            transform_name: key from DEFAULT_TRANSFORMS
            sample_rate: sample rate of the audio

        Returns:
            TransformResult with transformed audio and SNR
        """
        if transform_name not in cls.DEFAULT_TRANSFORMS:
            raise ValueError(
                f"Unknown transform: {transform_name}. "
                f"Available: {list(cls.DEFAULT_TRANSFORMS.keys())}"
            )

        method_name, kwargs = cls.DEFAULT_TRANSFORMS[transform_name]
        method = getattr(cls, method_name)

        # Add sample_rate to kwargs for transforms that need it
        if method_name in ("down_up_sample", "low_pass_filter", "mel_reconstruct"):
            kwargs = {**kwargs, "sample_rate": sample_rate}

        transformed = method(audio, **kwargs)

        # Compute SNR
        noise = audio - transformed
        signal_power = (audio ** 2).mean()
        noise_power = (noise ** 2).mean()
        if noise_power < 1e-10:
            snr_db = float("inf")
        else:
            snr_db = 10 * math.log10(signal_power.item() / noise_power.item())

        return TransformResult(
            transform_name=transform_name,
            transformed_audio=transformed,
            snr_db=snr_db,
        )

    @classmethod
    def detect(
        cls,
        per_transform_wers: Dict[str, float],
        threshold: float = 0.5,
    ) -> WaveGuardResult:
        """Aggregate per-transform WERs into a detection decision.

        Args:
            per_transform_wers: {transform_name: {wer, snr_db, output_original, output_transformed}}
            threshold: WER threshold above which a transform flags adversarial

        Returns:
            WaveGuardResult with detection decision
        """
        flagged = 0
        per_transform = {}
        for name, info in per_transform_wers.items():
            wer = info["wer"]
            is_flagged = wer > threshold
            if is_flagged:
                flagged += 1
            per_transform[name] = {
                "wer": wer,
                "flagged": is_flagged,
                "snr_db": info.get("snr_db", None),
                "output_original": info.get("output_original", ""),
                "output_transformed": info.get("output_transformed", ""),
            }

        total = len(per_transform_wers)
        # Detected if majority of transforms flag it
        detected = flagged > total / 2 if total > 0 else False

        return WaveGuardResult(
            detected=detected,
            num_transforms_flagged=flagged,
            num_transforms_total=total,
            per_transform=per_transform,
        )
