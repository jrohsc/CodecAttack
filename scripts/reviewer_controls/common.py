"""Shared, testable definitions for the reviewer-control experiment.

All reported metrics are computed from reloaded PCM_16 files.  The in-memory
pre-serialization perturbation is retained separately only for leakage checks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


TRAIN_OPUS_KBPS = (16, 24, 32, 64, 128)
EVAL_CHANNELS = ("clean", "opus16", "opus24", "opus32", "opus64", "opus128", "mp3_64", "aac_64")


def normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace; deliberately retain punctuation."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def target_match(target: str, output: str) -> bool:
    return normalize_for_match(target) in normalize_for_match(output)


def snr_db(reference: np.ndarray, delivered: np.ndarray) -> float:
    """Carrier-referenced, sample-aligned power SNR (mean over all samples)."""
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(delivered, dtype=np.float64).reshape(-1)
    if len(a) != len(b):
        raise ValueError(f"SNR inputs must have equal lengths, got {len(a)} and {len(b)}")
    signal = float(np.mean(a * a))
    noise = float(np.mean((b - a) ** 2))
    if noise == 0:
        return float("inf")
    if signal == 0:
        return float("-inf")
    return float(10.0 * np.log10(signal / noise))


def inband_rfft_mask(n: int, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Boolean rFFT mask for bins at or below ``cutoff_hz``, decided exactly.

    ``numpy``/``torch`` disagree on whether bin ``k`` sits at exactly the cutoff
    because ``torch.fft.rfftfreq`` defaults to float32: at 16 kHz over 74880
    samples it reports bin 18720 as 4000.0 Hz while float64 reports
    4000.0000000000005 Hz.  A float comparison therefore classifies that bin
    differently in the attacker and in the measurement, which shows up as a
    spurious ~2e-5 out-of-band energy fraction.  Comparing ``k * sample_rate``
    against ``cutoff_hz * n`` is exact for integer rates, so the optimizer and
    the audit agree bin-for-bin.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    k = np.arange(n // 2 + 1, dtype=np.float64)
    return k * float(sample_rate) <= float(cutoff_hz) * float(n)


def above_cutoff_energy_fraction(delta: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    """Fraction of real-signal perturbation energy above ``cutoff_hz``.

    "Above" is the exact complement of :func:`inband_rfft_mask`, so a signal the
    attacker band-limited to the cutoff measures as zero here up to float
    roundoff.
    """
    x = np.asarray(delta, dtype=np.float64).reshape(-1)
    spectrum = np.fft.rfft(x)
    energy = np.abs(spectrum) ** 2
    total = float(energy.sum())
    if total == 0:
        return 0.0
    inband = inband_rfft_mask(len(x), sample_rate, cutoff_hz)
    return float(energy[~inband].sum() / total)


def write_pcm16_pair(
    output_dir: Path,
    stem: str,
    reference: np.ndarray,
    adversarial: np.ndarray,
    sample_rate: int,
) -> dict:
    """Write and reload both arms, returning delivered-path distortion metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / f"{stem}_carrier_ref.wav"
    adv_path = output_dir / f"{stem}_attacked.wav"
    sf.write(ref_path, np.asarray(reference, dtype=np.float32), sample_rate, subtype="PCM_16")
    sf.write(adv_path, np.asarray(adversarial, dtype=np.float32), sample_rate, subtype="PCM_16")
    ref_pcm, ref_sr = sf.read(ref_path, dtype="float32")
    adv_pcm, adv_sr = sf.read(adv_path, dtype="float32")
    if ref_sr != sample_rate or adv_sr != sample_rate or len(ref_pcm) != len(adv_pcm):
        raise RuntimeError("PCM round-trip changed sample rate or relative length")
    pre_delta = np.asarray(adversarial, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    post_delta = adv_pcm.astype(np.float64) - ref_pcm.astype(np.float64)
    return {
        "carrier_ref_path": str(ref_path),
        "adv_path": str(adv_path),
        "sample_rate": sample_rate,
        "samples": len(adv_pcm),
        "snr_db": snr_db(ref_pcm, adv_pcm),
        "pre_above_4khz_fraction": above_cutoff_energy_fraction(pre_delta, sample_rate, 4000.0),
        "post_above_4khz_fraction": above_cutoff_energy_fraction(post_delta, sample_rate, 4000.0),
        "serialization_added_above_4khz_fraction": (
            above_cutoff_energy_fraction(post_delta, sample_rate, 4000.0)
            - above_cutoff_energy_fraction(pre_delta, sample_rate, 4000.0)
        ),
        "pre_peak": float(np.max(np.abs(adversarial))),
        "post_peak": float(np.max(np.abs(adv_pcm))),
        # Clipping is the one way this pipeline can inject out-of-band energy
        # into an otherwise band-limited perturbation, so count it rather than
        # infer it from the peak: a peak marginally above 1.0 is usually
        # absorbed by PCM_16 rounding without any sample reaching full scale.
        "pre_peak_exceeds_unit": bool(np.max(np.abs(adversarial)) > 1.0),
        "clipped_samples": int(np.sum(np.abs(adv_pcm) >= 32767 / 32768)),
    }


def _production_codec_fn(codec: str):
    """Return the production encode/decode helper for a delivery codec.

    The reviewer controls must be delivered through *exactly* the channel that
    produced the existing paper numbers, otherwise a difference in ffmpeg flags
    would masquerade as a difference between attacks.  These are the same
    functions ``scripts/eval_opus_bundle.py`` and ``scripts/eval_mp3_aac_bundle.py``
    call.  In particular this ffmpeg build has no ``libopus``; the production
    path uses the native ``opus`` encoder with ``-strict -2``.
    """
    import sys
    from pathlib import Path as _Path

    codec_root = str(_Path(__file__).resolve().parents[2])
    if codec_root not in sys.path:
        sys.path.insert(0, codec_root)
    if codec == "opus":
        from demo_ota import apply_opus_compression
        return apply_opus_compression
    if codec == "mp3":
        from aac_channel import apply_mp3_compression
        return apply_mp3_compression
    if codec == "aac":
        from aac_channel import apply_aac_compression
        return apply_aac_compression
    raise ValueError(codec)


def codec_roundtrip(audio: np.ndarray, sample_rate: int, codec: str, bitrate_kbps: int) -> np.ndarray:
    """Encode/decode through the production ffmpeg path at the input rate.

    FFmpeg's Opus decoder honors the container pre-skip, so the decoded signal
    is already sample-aligned with the input; ``audit_audio_path.py`` measures
    this and no delay compensation is applied.  Only the tail is trimmed or
    zero-padded to restore the exact sample count.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    y = _production_codec_fn(codec)(x, sample_rate, bitrate_kbps=int(bitrate_kbps))
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.reshape(-1)
    if len(y) >= len(x):
        return y[: len(x)]
    return np.pad(y, (0, len(x) - len(y)))


def aggregate_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    methods = sorted({r["method"] for r in rows})
    out = {}
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        channels = {}
        for channel in EVAL_CHANNELS:
            valid = [r for r in mr if channel in r.get("outcomes", {})]
            if not valid:
                continue
            paired = [r for r in valid if "clean" in r["outcomes"]]
            clean_success = [r for r in paired if r["outcomes"]["clean"]["match"]]
            both = [r for r in clean_success if r["outcomes"][channel]["match"]]
            new = [r for r in paired if not r["outcomes"]["clean"]["match"] and r["outcomes"][channel]["match"]]
            channels[channel] = {
                "asr_numerator": sum(r["outcomes"][channel]["match"] for r in valid),
                "asr_denominator": len(valid),
                "asr": sum(r["outcomes"][channel]["match"] for r in valid) / len(valid),
                "conditional_survival_numerator": len(both),
                "conditional_survival_denominator": len(clean_success),
                "conditional_survival": (len(both) / len(clean_success)) if clean_success else None,
                "compression_induced_new_successes": (len(new) if channel != "clean" else 0) if paired else None,
                "clean_paired_denominator": len(paired),
            }
        snrs = np.asarray([r["snr_db"] for r in mr], dtype=float)
        out[method] = {
            "n": len(mr),
            "snr_db_mean": float(snrs.mean()) if len(snrs) else None,
            "snr_db_median": float(np.median(snrs)) if len(snrs) else None,
            "above_4khz_fraction_mean": float(np.mean([r["post_above_4khz_fraction"] for r in mr])),
            "channels": channels,
        }
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
