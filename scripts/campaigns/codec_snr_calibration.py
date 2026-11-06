"""Decoded-waveform SNR calibration for continuous neural audio codecs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torchaudio.functional as AF

from audio_attack_controls import waveform_snr_db


class CalibrationError(RuntimeError):
    """Raised when the requested decoded SNR cannot be bracketed."""


def _as_bct(audio: torch.Tensor) -> torch.Tensor:
    """Normalize mono/batched audio to codec convention ``[B, C, T]``."""
    if audio.ndim == 1:
        return audio[None, None, :]
    if audio.ndim == 2:
        return audio[:, None, :]
    if audio.ndim == 3:
        return audio
    raise ValueError(f"expected [T], [B,T], or [B,C,T] audio; got {audio.shape}")


def _align(reference: torch.Tensor, decoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = _as_bct(reference)
    decoded = _as_bct(decoded)
    if reference.shape[0] != decoded.shape[0]:
        raise ValueError("codec changed the batch size during calibration")
    if reference.shape[1] != decoded.shape[1]:
        if reference.shape[1] > 1 and decoded.shape[1] == 1:
            reference = reference.mean(dim=1, keepdim=True)
        elif decoded.shape[1] > 1 and reference.shape[1] == 1:
            decoded = decoded.mean(dim=1, keepdim=True)
        else:
            raise ValueError("incompatible reference/decoded channel counts")
    length = min(reference.shape[-1], decoded.shape[-1])
    if length < 1:
        raise ValueError("codec produced empty audio")
    return reference[..., :length], decoded[..., :length]


def _direction_for(
    codec_model: Any,
    latent: torch.Tensor,
    perturbation_direction: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]],
) -> torch.Tensor:
    direction: torch.Tensor
    if callable(perturbation_direction):
        direction = perturbation_direction(latent)
    elif perturbation_direction is not None:
        direction = perturbation_direction
    elif hasattr(codec_model, "calibration_direction"):
        candidate = codec_model.calibration_direction
        direction = candidate(latent) if callable(candidate) else candidate
    else:
        # Reproducible fallback for standalone calibration. For paper results,
        # pass each attack's saved final latent delta so direction and only
        # direction remains fixed throughout the epsilon search.
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(0)
        direction = torch.empty_like(latent).uniform_(-1.0, 1.0, generator=generator)

    direction = direction.to(device=latent.device, dtype=latent.dtype)
    if direction.shape != latent.shape:
        raise ValueError(
            f"perturbation direction shape {direction.shape} does not match latent "
            f"shape {latent.shape}"
        )
    norm = direction.abs().amax()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("perturbation direction must be finite and nonzero")
    return direction / norm


def find_calibrated_epsilon(
    codec_model: Any,
    input_audio: torch.Tensor,
    target_snr_db: float,
    init_epsilon: float = 1.0,
    tolerance_db: float = 0.5,
    *,
    input_sample_rate: Optional[int] = None,
    perturbation_direction: Optional[
        Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
    max_iterations: int = 40,
    max_epsilon: float = 1e6,
) -> float:
    """Find a latent L-infinity budget with the requested decoded-audio SNR.

    The codec must provide ``encode_to_continuous`` and
    ``decode_from_continuous`` and should expose ``sample_rate``. The latent
    perturbation is ``epsilon * direction``, where ``direction`` is normalized
    to unit L-infinity norm and held fixed during the search. Supply the saved
    optimized attack delta as ``perturbation_direction`` for the strongest
    experimental control; a seeded direction is used only as a reproducible
    fallback.

    SNR follows the requested definition ``SNR(x, D(z + delta))``. If
    ``input_sample_rate`` differs from the codec's native rate (16/24/44.1 kHz,
    for example), the reference is differentiably resampled before encoding and
    measurement. Binary-search bracketing expands automatically beyond
    ``init_epsilon``.

    Returns:
        Calibrated scalar epsilon.

    Raises:
        CalibrationError: If codec reconstruction at epsilon zero is already
            noisier than the target, or no upper bracket is found.
    """
    if not math.isfinite(float(target_snr_db)):
        raise ValueError("target_snr_db must be finite")
    if init_epsilon <= 0 or tolerance_db <= 0:
        raise ValueError("init_epsilon and tolerance_db must be positive")
    if max_iterations < 1 or max_epsilon <= 0:
        raise ValueError("max_iterations and max_epsilon must be positive")
    for method in ("encode_to_continuous", "decode_from_continuous"):
        if not callable(getattr(codec_model, method, None)):
            raise TypeError(f"codec_model must implement {method}()")

    codec_sr = int(getattr(codec_model, "sample_rate", input_sample_rate or 0))
    if codec_sr <= 0:
        raise ValueError("codec_model.sample_rate or input_sample_rate is required")
    source_sr = int(input_sample_rate or codec_sr)

    try:
        device = torch.device(getattr(codec_model, "device"))
    except (TypeError, RuntimeError):
        device = input_audio.device
    reference = _as_bct(input_audio).to(device=device, dtype=torch.float32)
    if source_sr != codec_sr:
        reference = AF.resample(reference, source_sr, codec_sr)

    with torch.inference_mode():
        latent = codec_model.encode_to_continuous(reference).detach()
        direction = _direction_for(codec_model, latent, perturbation_direction)

        def evaluate(epsilon: float) -> float:
            decoded = codec_model.decode_from_continuous(latent + epsilon * direction)
            ref_aligned, decoded_aligned = _align(reference, decoded.float())
            return float(waveform_snr_db(ref_aligned, decoded_aligned).mean())

        snr_zero = evaluate(0.0)
        if snr_zero < target_snr_db - tolerance_db:
            raise CalibrationError(
                f"target {target_snr_db:.3f} dB is unattainable: codec-only "
                f"reconstruction SNR is {snr_zero:.3f} dB"
            )
        if abs(snr_zero - target_snr_db) <= tolerance_db:
            return 0.0

        lower, upper = 0.0, float(init_epsilon)
        snr_upper = evaluate(upper)
        while snr_upper > target_snr_db and upper < max_epsilon:
            lower = upper
            upper = min(2.0 * upper, max_epsilon)
            snr_upper = evaluate(upper)
        if snr_upper > target_snr_db:
            raise CalibrationError(
                f"could not bracket {target_snr_db:.3f} dB by epsilon "
                f"{max_epsilon:g}; last SNR was {snr_upper:.3f} dB"
            )

        best_epsilon = upper
        best_error = abs(snr_upper - target_snr_db)
        for _ in range(max_iterations):
            middle = (lower + upper) / 2.0
            snr_middle = evaluate(middle)
            error = abs(snr_middle - target_snr_db)
            if error < best_error:
                best_epsilon, best_error = middle, error
            if error <= tolerance_db:
                return float(middle)
            # For a fixed latent ray, decoded SNR is expected to decrease as
            # magnitude grows. Bracketing is retained on every iteration.
            if snr_middle > target_snr_db:
                lower = middle
            else:
                upper = middle
        return float(best_epsilon)


def load_calibrated_epsilon(
    calibration_file: Union[str, Path], codec_name: str
) -> float:
    """Load a codec's aggregate matched-SNR epsilon from calibration JSON."""
    path = Path(calibration_file)
    payload = json.loads(path.read_text())
    try:
        epsilon = payload["codecs"][codec_name.lower()]["epsilon"]
    except (KeyError, TypeError) as error:
        raise KeyError(
            f"no calibrated epsilon for codec {codec_name!r} in {path}"
        ) from error
    epsilon = float(epsilon)
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError(f"invalid calibrated epsilon {epsilon} in {path}")
    return epsilon

