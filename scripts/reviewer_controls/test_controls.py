from __future__ import annotations

import numpy as np
import torch

from reviewer_controls.common import (
    above_cutoff_energy_fraction, aggregate_rows, inband_rfft_mask,
    normalize_for_match, target_match, write_pcm16_pair,
)


def test_exact_metric_keeps_punctuation_and_normalizes_whitespace():
    assert target_match("Do it, now.", "prefix DO\n IT,   NOW. suffix")
    assert not target_match("Do it, now.", "do it now")
    assert normalize_for_match(" A\tB\n") == "a b"


def test_bandlimited_feasible_projection_has_no_out_of_band_energy():
    from robust_waveform_attack import RobustWaveformAttacker
    attacker = RobustWaveformAttacker.__new__(RobustWaveformAttacker)
    attacker.spectral_cutoff_hz = 4000.0
    attacker.eps = 0.05
    torch.manual_seed(1)
    original = torch.empty(1, 16000).uniform_(-0.999, 0.999)
    proposal = torch.randn_like(original)
    projected = attacker._project_spectral_feasible(proposal, original)
    assert float(projected.abs().max()) <= 0.050001
    assert float((original + projected).abs().max()) <= 1.000001
    leakage = above_cutoff_energy_fraction(projected.numpy(), 16000, 4000)
    assert leakage < 1e-10


def test_pcm_metrics_use_reloaded_pair(tmp_path):
    sr = 24000
    t = np.arange(sr) / sr
    ref = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    adv = (ref + 0.02 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    metrics = write_pcm16_pair(tmp_path, "x", ref, adv, sr)
    assert metrics["samples"] == sr
    assert np.isfinite(metrics["snr_db"])
    # PCM_16 quantization noise is white, so it lands partly above the cutoff.
    # It is bounded and small relative to a perturbation of realistic size.
    assert metrics["pre_above_4khz_fraction"] < 1e-9
    assert metrics["post_above_4khz_fraction"] < 1e-4


def test_pcm16_out_of_band_leakage_shrinks_as_perturbation_grows(tmp_path):
    """The serialization floor is absolute, so its relative share must fall."""
    sr = 24000
    t = np.arange(sr) / sr
    ref = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    fractions = []
    for amp in (0.001, 0.01, 0.1):
        adv = (ref + amp * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        fractions.append(write_pcm16_pair(tmp_path, f"a{amp}", ref, adv, sr)["post_above_4khz_fraction"])
    assert fractions[0] > fractions[1] > fractions[2]
    assert fractions[2] < 1e-5


def test_cutoff_bin_membership_is_exact_not_float32_rounded():
    """torch.fft.rfftfreq is float32; a bin sitting exactly on the cutoff must
    not be classified differently by the optimizer and by the audit."""
    n, sr, cut = 74880, 16000, 4000.0
    mask = inband_rfft_mask(n, sr, cut)
    boundary = int(round(cut * n / sr))
    assert mask[boundary] and not mask[boundary + 1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    spectrum = np.fft.rfft(x)
    spectrum[~mask] = 0
    band_limited = np.fft.irfft(spectrum, n=n)
    assert above_cutoff_energy_fraction(band_limited, sr, cut) < 1e-20


def test_survival_uses_clean_success_subset_not_asr_ratio():
    rows = [
        {"method": "m", "snr_db": 10, "post_above_4khz_fraction": 0,
         "outcomes": {"clean": {"match": True}, "opus16": {"match": False}}},
        {"method": "m", "snr_db": 11, "post_above_4khz_fraction": 0,
         "outcomes": {"clean": {"match": False}, "opus16": {"match": True}}},
    ]
    got = aggregate_rows(rows)["m"]["channels"]["opus16"]
    assert got["conditional_survival_numerator"] == 0
    assert got["conditional_survival_denominator"] == 1
    assert got["conditional_survival"] == 0
    assert got["compression_induced_new_successes"] == 1
