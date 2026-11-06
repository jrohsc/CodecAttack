"""Perceptual audio quality metrics.

PESQ/STOI are provided by the parent repo's `compute_pesq.py`; import lazily.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import MetricResult, Scenario
from .._registry import register_metric


@register_metric("snr_db")
def snr_db(output: str, scenario: Scenario, *, clean_wav: str | None = None,
           attacked_wav: str | None = None, **_) -> MetricResult:
    if clean_wav is None or attacked_wav is None:
        return MetricResult(name="snr_db", value=float("nan"),
                            extra={"note": "need clean+attacked paths"})
    import soundfile as sf
    c, _ = sf.read(clean_wav)
    a, _ = sf.read(attacked_wav)
    n = min(len(c), len(a))
    c, a = c[:n], a[:n]
    noise = a - c
    p_sig = float(np.mean(c ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    v = 10.0 * float(np.log10(p_sig / p_noise))
    return MetricResult(name="snr_db", value=v)
