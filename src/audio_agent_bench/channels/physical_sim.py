"""Simulated physical OTA channel: IR convolution + timeshift + Gaussian noise + random gain.

Physical OTA in reality is nonlinear and stochastic (see corrected_ir_analysis.md,
physical_ota_acoustic_analysis.md). This channel is an *approximation* for benchmark
reproducibility; real hardware eval stays in `3_results_physical_attack/`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


class PhysicalSimChannel:
    def __init__(
        self,
        ir_path: str | None = None,
        timeshift_ms: float = 60.0,
        noise_snr_db: float = 20.0,
        gain_db_range: tuple[float, float] = (-3.0, 3.0),
        seed: int | None = None,
    ) -> None:
        self.ir_path = ir_path
        self.timeshift_ms = timeshift_ms
        self.noise_snr_db = noise_snr_db
        self.gain_db_range = tuple(gain_db_range)
        self.seed = seed

    def tag(self) -> str:
        return f"physim_ts{int(self.timeshift_ms)}_snr{int(self.noise_snr_db)}"

    def apply(self, in_wav, out_wav) -> Path:
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)

        x, sr = sf.read(str(in_wav))
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32)

        rng = np.random.default_rng(self.seed)

        # 1) IR convolution (skip if no IR provided; real IRs are in
        #    3_results_physical_attack/ and vary per experiment).
        y = x
        if self.ir_path:
            ir_path = Path(self.ir_path)
            if ir_path.suffix == ".npy":
                ir = np.load(ir_path).astype(np.float32)
            else:
                ir, _ = sf.read(str(ir_path))
                ir = np.asarray(ir, dtype=np.float32)
                if ir.ndim > 1:
                    ir = ir.mean(axis=1)
            if ir.size:
                y = fftconvolve(y, ir, mode="full")[: x.size + ir.size - 1]
                # Normalize back to original peak to avoid clipping.
                peak = np.max(np.abs(y)) + 1e-9
                y = y / peak * (np.max(np.abs(x)) + 1e-9)

        # 2) Timeshift: integer-sample delay (pad with zeros, truncate tail).
        shift_samples = int(round(self.timeshift_ms * 1e-3 * sr))
        if shift_samples > 0:
            y = np.concatenate([np.zeros(shift_samples, dtype=np.float32), y])
        elif shift_samples < 0:
            y = y[-shift_samples:]
        y = y[: x.size]  # keep length stable

        # 3) Additive white Gaussian noise at target SNR.
        if np.isfinite(self.noise_snr_db):
            sig_power = float(np.mean(y**2) + 1e-12)
            noise_power = sig_power / (10 ** (self.noise_snr_db / 10))
            noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
            y = y + noise

        # 4) Random gain in dB.
        g_lo, g_hi = self.gain_db_range
        if g_hi > g_lo:
            gain_db = float(rng.uniform(g_lo, g_hi))
            y = y * (10 ** (gain_db / 20.0))

        # Clip to [-1, 1] to stay inside PCM range.
        y = np.clip(y, -1.0, 1.0)

        sf.write(str(out), y, sr)
        return out
