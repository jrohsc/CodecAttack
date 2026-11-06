"""SNR(adversarial, carrier) helper used by the waveform-vs-latent calibration."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import scipy.signal as sp


PathLike = Union[str, Path]


def _load_mono(path: PathLike) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def snr_db(adv_path: PathLike, carrier_path: PathLike) -> float:
    """Signal = carrier, noise = adv - carrier. Returns 10*log10(P_sig/P_noise)."""
    adv, sra = _load_mono(adv_path)
    car, src = _load_mono(carrier_path)
    if src != sra:
        car = sp.resample_poly(car, sra, src).astype(np.float32)
    n = min(len(adv), len(car))
    adv, car = adv[:n], car[:n]
    noise = adv - car
    sig_p = float(np.mean(car ** 2) + 1e-12)
    noise_p = float(np.mean(noise ** 2) + 1e-12)
    return 10.0 * np.log10(sig_p / noise_p)
