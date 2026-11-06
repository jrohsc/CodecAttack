"""Reusable experimental controls for waveform attacks.

The low-pass projection uses a real FFT of the complete signal.  It is an
orthogonal, differentiable projection: bins at and above ``max_freq`` are
exactly zero in the represented waveform (up to floating-point roundoff).

This module is a standalone, model-agnostic reference implementation. The
band-limited arm that is actually run and reported lives in
``RobustWaveformAttacker`` (``spectral_cutoff_hz``), driven by
``0_all_combined/reviewer_controls/run_attack.py``; note that it keeps bins *at*
the cutoff whereas ``frequency_mask`` here excludes them, matching each one's
documented convention. Both decide membership by exact integer comparison.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch


def waveform_snr_db(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return waveform SNR in dB, reduced over the last (time) dimension."""
    if reference.shape != estimate.shape:
        raise ValueError(
            f"SNR inputs must have identical shapes, got {reference.shape} and "
            f"{estimate.shape}"
        )
    signal = reference.float().square().mean(dim=-1)
    noise = (estimate.float() - reference.float()).square().mean(dim=-1)
    return 10.0 * torch.log10(signal.clamp_min(eps) / noise.clamp_min(eps))


def frequency_mask(
    delta: torch.Tensor,
    sample_rate: int,
    max_freq: float = 4000.0,
) -> torch.Tensor:
    """Differentiably project ``delta`` onto frequencies strictly below a cutoff.

    Args:
        delta: Real waveform with time on the last axis. Any leading dimensions
            (batch and/or channels) are retained.
        sample_rate: Waveform sample rate in Hz.
        max_freq: Exclusive upper edge in Hz. Every FFT bin with frequency
            ``>= max_freq`` is removed.

    Returns:
        A real tensor with the same shape, dtype, and device as ``delta``.

    Raises:
        ValueError: For an empty signal, invalid sample rate, or a cutoff outside
            ``(0, Nyquist]``. A cutoff equal to Nyquist is valid and removes the
            Nyquist bin for even-length signals.
    """
    if delta.ndim == 0 or delta.shape[-1] < 1:
        raise ValueError("delta must have a non-empty time dimension")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    nyquist = sample_rate / 2.0
    if not 0.0 < max_freq <= nyquist:
        raise ValueError(
            f"max_freq must be in (0, {nyquist:g}] Hz for {sample_rate} Hz audio; "
            f"got {max_freq}"
        )

    original_dtype = delta.dtype
    # torch.fft does not support CPU float16/bfloat16 and is numerically more
    # stable in float32. The casts remain differentiable.
    work = delta if delta.dtype in (torch.float32, torch.float64) else delta.float()
    n = work.shape[-1]
    spectrum = torch.fft.rfft(work, dim=-1, norm="ortho")
    # Bin membership is decided by exact integer comparison rather than by
    # comparing torch.fft.rfftfreq (float32) against the cutoff: a bin sitting
    # exactly on the cutoff rounds differently in float32 and float64, which
    # makes an exactly band-limited signal measure as carrying ~2e-5 of its
    # energy out of band. Must stay identical to
    # ``reviewer_controls.common.inband_rfft_mask`` and to
    # ``RobustWaveformAttacker._spectral_delta``, which is the production path.
    k = torch.arange(n // 2 + 1, device=work.device, dtype=torch.float64)
    mask = k * float(sample_rate) < float(max_freq) * float(n)
    projected = torch.fft.irfft(
        spectrum * mask.to(spectrum.dtype),
        n=n,
        dim=-1,
        norm="ortho",
    )
    return projected.to(original_dtype)


def project_to_snr(
    delta: torch.Tensor,
    reference: torch.Tensor,
    target_snr_db: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale each example/channel globally to a requested waveform SNR.

    Global scaling preserves the Fourier support of an already band-limited
    perturbation. Reduction is over time only, so ``[B, C, T]`` inputs are
    calibrated independently per batch item and channel.
    """
    if delta.shape != reference.shape:
        raise ValueError("delta and reference must have the same shape")
    if not math.isfinite(float(target_snr_db)):
        raise ValueError("target_snr_db must be finite")

    signal_rms = reference.float().square().mean(dim=-1, keepdim=True).sqrt()
    wanted_noise_rms = signal_rms * (10.0 ** (-float(target_snr_db) / 20.0))
    actual_noise_rms = delta.float().square().mean(dim=-1, keepdim=True).sqrt()
    if torch.any((actual_noise_rms <= eps) & (wanted_noise_rms > eps)):
        raise ValueError("cannot SNR-scale an all-zero perturbation")
    scale = wanted_noise_rms / actual_noise_rms.clamp_min(eps)
    return delta * scale.to(delta.dtype)


def _call_with_optional_prompt(function: Callable[..., Any], *args: Any,
                               prompt: Optional[str]) -> Any:
    if prompt is None:
        return function(*args)
    try:
        return function(*args, prompt=prompt)
    except TypeError as error:
        if "prompt" not in str(error):
            raise
        return function(*args)


def run_bandlimited_waveform_attack(
    audio_tensor: torch.Tensor,
    target: str,
    model: Any,
    max_freq: float = 4000.0,
    target_snr: float = 10.0,
    *,
    sample_rate: Optional[int] = None,
    steps: int = 1000,
    step_size: float = 1e-2,
    prompt: Optional[str] = None,
    success_fn: Optional[Callable[[str, str], bool]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Run SNR-matched PGD in the waveform subspace below ``max_freq``.

    ``model`` must expose ``compute_loss(audio, target[, prompt=...])`` and
    ``generate(audio[, prompt=...])``. The optimization variable is projected
    through the FFT mask in every forward pass and after every update. It is
    also scaled to the requested SNR after every update, making the comparison
    independent of an arbitrary waveform L-infinity epsilon.

    Sample rate is read from ``model.sample_rate`` when omitted, falling back
    to 16 kHz. The function never clips the final waveform: pointwise clipping
    would reintroduce frequencies above the cutoff. Callers that require a
    bounded PCM range should normalize the clean carrier beforehand.

    Returns:
        ``(adversarial_audio, measured_snr_db, asr)``. For this single-example
        API, ASR is ``1.0`` on an exact case-insensitive target substring match
        and ``0.0`` otherwise. Pass ``success_fn`` for task-specific scoring.
    """
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    if steps < 1 or step_size <= 0:
        raise ValueError("steps and step_size must be positive")
    if not torch.is_floating_point(audio_tensor):
        raise TypeError("audio_tensor must be floating point")
    sr = int(sample_rate or getattr(model, "sample_rate", 16000))

    clean = audio_tensor.detach()
    # A nonzero band-limited start avoids the undefined projection of a zero
    # vector onto a fixed-SNR sphere. Its scale does not affect the budget.
    raw_delta = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    with torch.no_grad():
        raw_delta.copy_(project_to_snr(
            frequency_mask(raw_delta, sr, max_freq), clean, target_snr
        ))
    raw_delta.requires_grad_(True)
    optimizer = torch.optim.Adam([raw_delta], lr=step_size)

    best_loss = float("inf")
    best_delta = raw_delta.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = project_to_snr(
            frequency_mask(raw_delta, sr, max_freq), clean, target_snr
        )
        adversarial = clean + delta
        loss = _call_with_optional_prompt(
            model.compute_loss, adversarial, target, prompt=prompt
        )
        if loss.ndim:
            loss = loss.mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            feasible = project_to_snr(
                frequency_mask(raw_delta, sr, max_freq), clean, target_snr
            )
            raw_delta.copy_(feasible)
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best_delta = feasible.detach().clone()

    with torch.no_grad():
        # Reproject once after selecting the best iterate. This prevents stored
        # optimizer roundoff from becoming part of the reported control.
        final_delta = project_to_snr(
            frequency_mask(best_delta, sr, max_freq), clean, target_snr
        )
        adversarial = clean + final_delta
        measured = float(waveform_snr_db(clean, adversarial).mean())
        output = _call_with_optional_prompt(model.generate, adversarial, prompt=prompt)

    if isinstance(output, (list, tuple)):
        outputs = [str(item) for item in output]
        scorer = success_fn or (lambda wanted, got: wanted.lower() in got.lower())
        asr = sum(float(scorer(target, item)) for item in outputs) / len(outputs)
    else:
        scorer = success_fn or (lambda wanted, got: wanted.lower() in got.lower())
        asr = float(scorer(target, str(output)))
    return adversarial.detach(), measured, asr

