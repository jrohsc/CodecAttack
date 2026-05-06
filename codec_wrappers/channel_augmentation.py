"""
Differentiable OTA channel simulator for channel-robust adversarial attacks.

Provides randomized, differentiable augmentation layers that simulate the
over-the-air (OTA) pipeline: speaker → air → microphone.

Components (all operate at 24kHz, EnCodec native rate):
  1. DifferentiableBandpass  — FIR bandpass via F.conv1d
  2. DifferentiableIRConvolution — Convolve with empirical impulse response
  3. AdditiveColoredNoise — Empirical noise bank or shaped Gaussian fallback
  4. DynamicRangeCompressor — Soft-knee compressor (sigmoid approximation)
  5. CodecSimulator — Simple quantization noise (legacy)
  5b. DifferentiableCodecProxy — EnCodec encode→decode STE proxy (BPDA)
  6. OTAChannelAugmentation — Chains all components with randomized params

All components use torch ops for gradient flow. Non-differentiable parts
(noise sampling) are treated as constants so gradients flow through the
clean path (straight-through).
"""

import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBandpass(nn.Module):
    """
    FIR bandpass filter applied via F.conv1d.

    Pre-computes a bank of FIR filters at init time with randomized cutoffs,
    then randomly selects from the bank during forward passes. This avoids
    calling scipy.signal.firwin on every forward pass (expensive CPU call
    inside a GPU optimization loop).
    """

    FILTER_BANK_SIZE = 32  # Number of pre-computed random filters

    def __init__(
        self,
        sample_rate: int = 24000,
        low_hz: float = 800.0,
        high_hz: float = 7000.0,
        num_taps: int = 101,
        randomize_range: float = 0.3,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.num_taps = num_taps
        self.randomize_range = randomize_range

        # Pre-compute default filter
        taps = self._design_filter(low_hz, high_hz)
        self.register_buffer("default_taps", taps)

        # Pre-compute a bank of randomized filters to avoid scipy calls
        # during the optimization loop
        if randomize_range > 0:
            bank = [taps]  # include default
            import random
            for _ in range(self.FILTER_BANK_SIZE - 1):
                low_jitter = 1.0 + (random.random() - 0.5) * 2 * randomize_range
                high_jitter = 1.0 + (random.random() - 0.5) * 2 * randomize_range
                low = max(low_hz * low_jitter, 50.0)
                high = min(high_hz * high_jitter, sample_rate / 2 - 100)
                if low >= high:
                    low, high = low_hz, high_hz
                bank.append(self._design_filter(low, high))
            self.register_buffer("filter_bank", torch.stack(bank))  # [N, num_taps]

    def _design_filter(self, low_hz: float, high_hz: float) -> torch.Tensor:
        """Design a bandpass FIR filter using scipy."""
        from scipy.signal import firwin

        nyquist = self.sample_rate / 2.0
        low = max(low_hz / nyquist, 0.001)
        high = min(high_hz / nyquist, 0.999)
        taps = firwin(self.num_taps, [low, high], pass_zero=False)
        return torch.FloatTensor(taps)

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Apply bandpass filter.

        Args:
            x: Audio tensor [B, 1, T] or [1, T]
            severity: 0=no filtering, 1=full filtering

        Returns:
            Filtered audio, same shape as input
        """
        if severity < 0.01:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        # Pick filter: random from pre-computed bank (training) or default (eval)
        if self.training and self.randomize_range > 0:
            idx = torch.randint(self.filter_bank.shape[0], (1,)).item()
            taps = self.filter_bank[idx]
        else:
            taps = self.default_taps

        # Reshape for conv1d: [out_channels, in_channels, kernel_size]
        kernel = taps.view(1, 1, -1).to(x.device, x.dtype)
        pad = self.num_taps // 2

        filtered = F.conv1d(x, kernel, padding=pad)

        # Blend with original based on severity
        if severity < 1.0:
            filtered = severity * filtered + (1.0 - severity) * x

        if squeeze:
            filtered = filtered.squeeze(0)

        return filtered


class DifferentiableIRConvolution(nn.Module):
    """
    Convolve with an impulse response (IR) to simulate room/speaker effects.

    If an empirical IR is available (from recorded pairs analysis), uses that.
    Otherwise falls back to a simple synthetic exponential decay IR.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        ir_length_ms: float = 50.0,
        randomize_scale: float = 0.1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.ir_length = int(sample_rate * ir_length_ms / 1000.0)
        self.randomize_scale = randomize_scale

        # Default: synthetic exponential decay IR
        t = torch.arange(self.ir_length, dtype=torch.float32)
        decay = torch.exp(-t / (self.ir_length / 4.0))
        decay[0] = 1.0  # Direct path
        decay = decay / decay.sum()  # Normalize
        self.register_buffer("ir", decay)
        self._empirical = False

    def load_empirical_ir(self, ir_path: str):
        """Load empirical IR from a .npy file."""
        ir_np = np.load(ir_path)
        ir_tensor = torch.FloatTensor(ir_np)
        # Normalize
        ir_tensor = ir_tensor / (ir_tensor.abs().sum() + 1e-8)
        self.ir = ir_tensor.to(self.ir.device)
        self.ir_length = len(ir_tensor)
        self._empirical = True

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Apply IR convolution.

        Args:
            x: [B, 1, T] or [1, T]
            severity: 0=bypass, 1=full convolution
        """
        if severity < 0.01:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        ir = self.ir.clone()

        # Randomize IR slightly
        if self.training and self.randomize_scale > 0:
            noise = torch.randn_like(ir) * self.randomize_scale
            ir = ir + noise
            ir = ir / (ir.abs().sum() + 1e-8)

        kernel = ir.view(1, 1, -1).to(x.device, x.dtype)
        pad = self.ir_length // 2

        convolved = F.conv1d(x, kernel, padding=pad)
        # Trim to match input length
        convolved = convolved[..., :x.shape[-1]]

        if severity < 1.0:
            convolved = severity * convolved + (1.0 - severity) * x

        if squeeze:
            convolved = convolved.squeeze(0)

        return convolved


class AdditiveColoredNoise(nn.Module):
    """
    Add colored noise matching empirical OTA noise characteristics.

    If a noise bank (from recorded pairs) is available, samples from it.
    Otherwise generates colored Gaussian noise with a 1/f spectral shape
    (pink noise), which approximates typical ambient noise.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        snr_db_mean: float = 20.0,
        snr_db_std: float = 5.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.snr_db_mean = snr_db_mean
        self.snr_db_std = snr_db_std
        self._noise_bank = None

    def load_noise_bank(self, noise_dir: str):
        """
        Load empirical noise samples from a directory of .npy or .wav files.
        """
        import glob
        noise_files = sorted(
            glob.glob(os.path.join(noise_dir, "residual_noise_[0-9]*.npy")) +
            glob.glob(os.path.join(noise_dir, "noise_*.npy")) +
            glob.glob(os.path.join(noise_dir, "noise_*.wav"))
        )
        if not noise_files:
            return

        bank = []
        for fp in noise_files:
            if fp.endswith(".npy"):
                n = np.load(fp)
            else:
                import soundfile as sf
                n, _ = sf.read(fp)
            bank.append(torch.FloatTensor(n))

        if bank:
            self._noise_bank = bank

    def _generate_pink_noise(self, length: int, device: torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
        """Generate pink (1/f) noise via FFT shaping."""
        white = torch.randn(length, device=device, dtype=dtype)
        fft = torch.fft.rfft(white)
        freqs = torch.arange(1, fft.shape[0] + 1, device=device, dtype=dtype)
        # 1/f shaping (pink noise)
        fft = fft / torch.sqrt(freqs)
        pink = torch.fft.irfft(fft, n=length)
        # Normalize to unit variance
        pink = pink / (pink.std() + 1e-8)
        return pink

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Add noise to signal. Noise is detached (no gradients through noise).

        Args:
            x: [B, 1, T] or [1, T]
            severity: scales the noise power (0=no noise, 1=full noise at snr_db_mean)
        """
        if severity < 0.01:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        B, C, T = x.shape

        # Randomize SNR
        if self.training:
            snr_db = self.snr_db_mean + torch.randn(1).item() * self.snr_db_std
        else:
            snr_db = self.snr_db_mean

        # Adjust SNR with severity (higher severity = more noise = lower SNR)
        # At severity=0: infinite SNR (no noise)
        # At severity=1: snr_db as specified
        effective_snr = snr_db / max(severity, 0.01)

        # Generate or sample noise
        if self._noise_bank is not None and len(self._noise_bank) > 0:
            idx = torch.randint(len(self._noise_bank), (1,)).item()
            noise = self._noise_bank[idx]
            # Repeat/trim to match length
            if len(noise) < T:
                reps = (T // len(noise)) + 1
                noise = noise.repeat(reps)
            noise = noise[:T].to(x.device, x.dtype)
            noise = noise / (noise.std() + 1e-8)
            noise = noise.view(1, 1, T).expand(B, C, T)
        else:
            noise = self._generate_pink_noise(T, x.device, x.dtype)
            noise = noise.view(1, 1, T).expand(B, C, T)

        # Scale noise to target SNR
        signal_power = torch.mean(x.detach() ** 2)
        noise_power_target = signal_power / (10 ** (effective_snr / 10) + 1e-10)
        noise = noise * torch.sqrt(noise_power_target + 1e-10)

        # Detach noise — gradients flow through x only (straight-through)
        result = x + noise.detach()

        if squeeze:
            result = result.squeeze(0)

        return result


class DynamicRangeCompressor(nn.Module):
    """
    Differentiable soft-knee compressor simulating speaker nonlinearity.

    Uses tanh-based soft clipping which is naturally differentiable.
    """

    def __init__(
        self,
        threshold_db: float = -10.0,
        ratio: float = 4.0,
        knee_width_db: float = 6.0,
    ):
        super().__init__()
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_width_db = knee_width_db

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Apply soft compression.

        Args:
            x: [B, 1, T] or any shape
            severity: 0=bypass, 1=full compression
        """
        if severity < 0.01:
            return x

        # Soft clipping via tanh — simple and fully differentiable
        # Scale factor controls how aggressive the clipping is
        threshold_linear = 10 ** (self.threshold_db / 20.0)
        # At severity=1, apply full compression; at 0, pass through
        scale = 1.0 / (threshold_linear * max(severity, 0.01))

        compressed = torch.tanh(x * scale) * threshold_linear

        # Blend with original
        if severity < 1.0:
            compressed = severity * compressed + (1.0 - severity) * x

        return compressed


class CodecSimulator(nn.Module):
    """
    Legacy codec simulator that adds simple quantization noise.
    Kept for backward compatibility. Use DifferentiableCodecProxy instead.
    """

    def __init__(self, codec_wrapper=None, add_quant_noise: bool = True):
        super().__init__()
        self.codec = codec_wrapper
        self.add_quant_noise = add_quant_noise

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        if severity < 0.01:
            return x
        if self.add_quant_noise:
            noise_scale = 0.01 * severity
            quant_noise = torch.randn_like(x) * noise_scale * x.abs().mean().detach()
            return x + quant_noise.detach()
        return x


class DifferentiableCodecProxy(nn.Module):
    """
    Differentiable codec proxy using EnCodec encode→decode with Straight-Through
    Estimator (STE / BPDA).

    Forward pass: runs the full non-differentiable EnCodec encode→decode cycle
    at a randomly sampled bandwidth (Expectation over Transformation).
    Backward pass: treats the codec as identity — gradients flow through x.

    This approximates the effect of lossy codecs (Opus, AAC) used by platforms
    like SoundCloud, enabling adversarial optimization through the codec channel.
    """

    DEFAULT_BANDWIDTHS = [1.5, 3.0, 6.0]

    def __init__(
        self,
        encodec_model,
        proxy_bandwidths: list = None,
    ):
        """
        Args:
            encodec_model: The EncodecModel instance (shared with LatentCodecAttacker.codec.model).
            proxy_bandwidths: List of bandwidths (kbps) to randomly sample from
                             each forward pass. Default: [1.5, 3.0, 6.0].
        """
        super().__init__()
        self.encodec_model = encodec_model
        self.proxy_bandwidths = proxy_bandwidths or self.DEFAULT_BANDWIDTHS

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Apply codec proxy with STE.

        Args:
            x: Audio tensor [B, 1, T] at 24kHz
            severity: 0=bypass, 1=full codec degradation

        Returns:
            Codec-degraded audio with STE gradient path through x.
        """
        if severity < 0.01:
            return x

        # Randomly sample bandwidth (EoT)
        bw = self.proxy_bandwidths[torch.randint(len(self.proxy_bandwidths), (1,)).item()]

        # Full non-differentiable codec cycle
        with torch.no_grad():
            self.encodec_model.set_target_bandwidth(bw)
            frames = self.encodec_model.encode(x)
            degraded = self.encodec_model.decode(frames)
            # Trim/pad to match input length
            if degraded.shape[-1] > x.shape[-1]:
                degraded = degraded[..., :x.shape[-1]]
            elif degraded.shape[-1] < x.shape[-1]:
                degraded = F.pad(degraded, (0, x.shape[-1] - degraded.shape[-1]))

        # STE: forward uses degraded, backward treats as identity through x
        result = x + (degraded - x).detach()

        # Blend with original based on severity
        if severity < 1.0:
            result = severity * result + (1.0 - severity) * x

        return result


class DifferentiableOpusProxy(nn.Module):
    """
    Differentiable Opus codec proxy using FFmpeg encode→decode with a
    Straight-Through Estimator (STE / BPDA).

    Forward pass: routes [B, 1, T] audio at `sample_rate` through ffmpeg Opus
    at a bitrate randomly sampled from `bitrates_kbps` (Expectation over
    Transformation). Backward pass: treats the codec as identity — gradient
    flows unchanged through x.

    Used by `robust_latent_attack.py` channel_mode="multi_bitrate" to harden
    latent-space perturbations against the eval-time Opus channel grid.
    """

    DEFAULT_BITRATES = [16, 24, 32, 64, 128]

    def __init__(
        self,
        sample_rate: int = 24000,
        bitrates_kbps: list = None,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.bitrates_kbps = [int(b) for b in (bitrates_kbps or self.DEFAULT_BITRATES)]

    @staticmethod
    def _encode_decode_opus(x_np, sr: int, bitrate_kbps: int):
        """Run one ffmpeg Opus encode→decode cycle. Returns float32 np.ndarray.

        Raises RuntimeError if ffmpeg is not available (instead of silently
        returning the input unchanged, which would make the attack optimize
        against a no-op channel).
        """
        import shutil, os, sys
        # Import locally so tests that skip on no-ffmpeg still import the module.
        from demo_ota import apply_opus_compression

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            env_bin = os.path.dirname(sys.executable)
            if not os.path.isfile(os.path.join(env_bin, "ffmpeg")):
                raise RuntimeError(
                    "ffmpeg not found; DifferentiableOpusProxy cannot encode. "
                    "Install ffmpeg in the active conda env or add it to PATH."
                )

        return apply_opus_compression(x_np, sr, bitrate_kbps=bitrate_kbps)

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] audio at self.sample_rate.
            severity: 0 = bypass (passthrough); 1 = apply codec.
        Returns:
            STE-wrapped codec output: forward is lossy, backward is identity.
        """
        if severity < 0.01:
            return x

        bitrate = self.bitrates_kbps[
            torch.randint(len(self.bitrates_kbps), (1,)).item()
        ]

        orig_shape = x.shape
        orig_device = x.device
        orig_dtype = x.dtype

        with torch.no_grad():
            # Process per-item in the batch (ffmpeg is file-based; batch=1 is typical here)
            outs = []
            x_cpu = x.detach().to(dtype=torch.float32, device="cpu")
            for b in range(x_cpu.shape[0]):
                mono = x_cpu[b, 0].numpy()
                degraded = self._encode_decode_opus(mono, self.sample_rate, bitrate)
                # Normalize dtype early: sf.read inside apply_opus_compression can
                # return float64 depending on the source; cast before trim/pad so
                # downstream dtype stays consistent.
                import numpy as np
                degraded = np.asarray(degraded, dtype=np.float32)
                # Trim/pad to input length
                T = mono.shape[0]
                if degraded.shape[0] > T:
                    degraded = degraded[:T]
                elif degraded.shape[0] < T:
                    pad = T - degraded.shape[0]
                    degraded = torch.nn.functional.pad(
                        torch.from_numpy(degraded), (0, pad)
                    ).numpy()
                outs.append(torch.from_numpy(degraded))
            y = torch.stack(outs).unsqueeze(1)  # [B, 1, T]
            y = y.to(device=orig_device, dtype=orig_dtype)
            if y.shape != orig_shape:
                raise RuntimeError(
                    f"DifferentiableOpusProxy shape mismatch: {y.shape} vs {orig_shape}"
                )

        # Straight-Through: forward = y (lossy), backward = identity through x.
        return y.detach() + (x - x.detach())


class DifferentiableSpectralGating(nn.Module):
    """
    Differentiable spectral gating denoiser (proxy for iPhone VPIO noise suppression).

    Simulates noise cancellation by estimating a noise floor from the signal
    and applying a soft suppression mask in the STFT domain. Fully differentiable
    via PyTorch STFT/iSTFT.

    Pipeline:
        1. STFT the input
        2. Estimate noise floor (mean magnitude of lowest-energy frames)
        3. Compute soft suppression mask: sigmoid((mag - threshold * noise_floor) / smoothness)
        4. Apply mask to complex STFT
        5. iSTFT back to waveform

    The mask is soft (sigmoid) so gradients flow smoothly. The noise floor
    estimation uses detached statistics so it doesn't create gradient loops.

    Parameters:
        prop_decrease: How aggressively to suppress (0=no suppression, 1=full).
            Maps to iPhone VPIO strength. 0.3=mild, 0.6=moderate, 0.9=aggressive.
        n_fft: STFT window size.
        hop_length: STFT hop size.
        noise_percentile: Fraction of lowest-energy frames used to estimate noise floor.
        threshold: Multiplier on noise floor for the suppression gate.
        smoothness: Controls sigmoid sharpness (lower=harder gate, higher=softer).
    """

    def __init__(
        self,
        prop_decrease: float = 0.5,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_percentile: float = 0.1,
        threshold: float = 1.5,
        smoothness: float = 0.5,
        randomize: bool = True,
    ):
        super().__init__()
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_percentile = noise_percentile
        self.threshold = threshold
        self.smoothness = smoothness
        self.randomize = randomize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral gating denoiser.

        Args:
            x: Audio tensor [B, 1, T] at any sample rate

        Returns:
            Denoised audio, same shape as input.
        """
        squeeze_batch = False
        squeeze_chan = False
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
            squeeze_chan = True
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        B, C, T = x.shape
        # Process each batch/channel independently
        results = []
        for b in range(B):
            for c in range(C):
                results.append(self._denoise_single(x[b, c]))
        result = torch.stack(results).view(B, C, T)

        if squeeze_batch:
            result = result.squeeze(0)
        if squeeze_chan:
            result = result.squeeze(0)
        return result

    def _denoise_single(self, wav: torch.Tensor) -> torch.Tensor:
        """Denoise a single 1D waveform."""
        T = wav.shape[-1]

        # STFT
        window = torch.hann_window(self.n_fft, device=wav.device, dtype=wav.dtype)
        stft = torch.stft(wav, self.n_fft, self.hop_length, window=window,
                          return_complex=True)
        # stft shape: [n_freq, n_frames]
        mag = stft.abs()

        # Estimate noise floor from lowest-energy frames (detached)
        with torch.no_grad():
            frame_energy = mag.sum(dim=0)  # [n_frames]
            n_noise_frames = max(1, int(self.noise_percentile * frame_energy.shape[0]))
            _, noise_indices = frame_energy.topk(n_noise_frames, largest=False)
            noise_floor = mag[:, noise_indices].mean(dim=1, keepdim=True)  # [n_freq, 1]

        # Randomize prop_decrease during training for EoT
        prop = self.prop_decrease
        if self.training and self.randomize:
            # Uniform in [prop * 0.5, min(prop * 1.5, 1.0)]
            lo = prop * 0.5
            hi = min(prop * 1.5, 1.0)
            prop = lo + (hi - lo) * torch.rand(1, device=wav.device).item()

        # Soft suppression mask via sigmoid
        # When mag >> noise_floor * threshold: mask ≈ 1 (keep)
        # When mag << noise_floor * threshold: mask ≈ 0 (suppress)
        gate_threshold = noise_floor * self.threshold
        mask = torch.sigmoid((mag - gate_threshold) / (self.smoothness * noise_floor.clamp(min=1e-8)))

        # Blend: mask = 1 means keep original, mask = 0 means suppress
        # prop_decrease controls how much suppression to apply
        effective_mask = 1.0 - prop * (1.0 - mask)

        # Apply mask to complex STFT
        stft_masked = stft * effective_mask

        # iSTFT
        denoised = torch.istft(stft_masked, self.n_fft, self.hop_length,
                               window=window, length=T)
        return denoised


class BPDASpectralDenoiser(nn.Module):
    """
    BPDA denoiser: runs real noisereduce in forward, differentiable proxy in backward.

    Forward pass: actual noisereduce spectral gating (non-differentiable, accurate).
    Backward pass: DifferentiableSpectralGating proxy (differentiable, approximate).

    This closes the gap between the soft sigmoid proxy and real spectral subtraction.
    The optimizer "sees" the real denoising effect but gets usable gradients.
    """

    def __init__(
        self,
        prop_decrease: float = 0.9,
        sample_rate: int = 24000,
        randomize: bool = True,
        prop_range: tuple = (0.5, 1.0),
        n_passes: int = 1,
    ):
        super().__init__()
        self.prop_decrease = prop_decrease
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.prop_range = prop_range
        self.n_passes = n_passes
        # Differentiable proxy for backward pass
        self._proxy = DifferentiableSpectralGating(
            prop_decrease=prop_decrease,
            n_fft=1024,
            hop_length=256,
            smoothness=0.2,   # harder gate than default (0.5) to better match noisereduce
            threshold=2.0,    # higher threshold to match noisereduce aggressiveness
            randomize=False,  # we handle randomization ourselves
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Randomize prop during training
        prop = self.prop_decrease
        if self.training and self.randomize:
            lo, hi = self.prop_range
            prop = lo + (hi - lo) * torch.rand(1, device=x.device).item()

        # Update proxy to match current prop
        self._proxy.prop_decrease = prop

        if not x.requires_grad:
            # No gradient needed — just run noisereduce directly
            out = x
            for _ in range(self.n_passes):
                out = self._run_noisereduce(out, prop)
            return out

        # BPDA: forward = real noisereduce, backward = proxy gradient
        # 1. Run proxy (differentiable) — we'll use its gradient
        proxy_out = self._proxy(x)

        # 2. Run real noisereduce N times (non-differentiable)
        with torch.no_grad():
            real_out = x
            for _ in range(self.n_passes):
                real_out = self._run_noisereduce(real_out, prop)

        # 3. Straight-through: use real output but proxy gradient
        # result = real_out + (proxy_out - proxy_out.detach())
        # This gives: forward value = real_out, backward gradient = d(proxy_out)/dx
        return real_out + (proxy_out - proxy_out.detach())

    def _run_noisereduce(self, x: torch.Tensor, prop: float) -> torch.Tensor:
        """Run actual noisereduce on tensor, return tensor."""
        import noisereduce as nr

        device = x.device
        dtype = x.dtype
        orig_shape = x.shape

        # Handle [B, 1, T] or [1, T] or [T]
        if x.dim() == 3:
            wav_np = x[0, 0].detach().cpu().float().numpy()
        elif x.dim() == 2:
            wav_np = x[0].detach().cpu().float().numpy()
        else:
            wav_np = x.detach().cpu().float().numpy()

        denoised_np = nr.reduce_noise(
            y=wav_np,
            sr=self.sample_rate,
            prop_decrease=prop,
            stationary=False,
            n_fft=1024,
        )

        result = torch.from_numpy(denoised_np).to(dtype=dtype, device=device)
        return result.view(orig_shape)


class OTAChannelAugmentation(nn.Module):
    """
    Full OTA channel simulator chaining all differentiable components.

    Pipeline:
        1. Codec simulation (optional)
        2. Bandpass filter (speaker + mic frequency response)
        3. IR convolution (room acoustics)
        4. Dynamic range compression (speaker nonlinearity)
        5. Additive colored noise (ambient noise)

    Each forward pass randomizes parameters for Expectation over
    Transformation (EoT) training.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        # Bandpass
        bandpass_low_hz: float = 800.0,
        bandpass_high_hz: float = 7000.0,
        # Noise
        snr_db_mean: float = 20.0,
        snr_db_std: float = 5.0,
        # IR
        ir_length_ms: float = 50.0,
        # Compressor
        compressor_threshold_db: float = -10.0,
        # Component toggles
        enable_bandpass: bool = True,
        enable_ir: bool = True,
        enable_noise: bool = True,
        enable_compressor: bool = True,
        enable_codec: bool = False,
        # Codec
        codec_wrapper=None,
    ):
        super().__init__()

        self.enable_bandpass = enable_bandpass
        self.enable_ir = enable_ir
        self.enable_noise = enable_noise
        self.enable_compressor = enable_compressor
        self.enable_codec = enable_codec

        self.bandpass = DifferentiableBandpass(
            sample_rate=sample_rate,
            low_hz=bandpass_low_hz,
            high_hz=bandpass_high_hz,
        )
        self.ir_conv = DifferentiableIRConvolution(
            sample_rate=sample_rate,
            ir_length_ms=ir_length_ms,
        )
        self.noise = AdditiveColoredNoise(
            sample_rate=sample_rate,
            snr_db_mean=snr_db_mean,
            snr_db_std=snr_db_std,
        )
        self.compressor = DynamicRangeCompressor(
            threshold_db=compressor_threshold_db,
        )
        # Use DifferentiableCodecProxy (STE) when an EncodecModel is provided,
        # otherwise fall back to simple CodecSimulator (quantization noise)
        self.codec_proxy = None
        if codec_wrapper is not None:
            self.codec_proxy = DifferentiableCodecProxy(
                encodec_model=codec_wrapper,
                proxy_bandwidths=[1.5, 3.0, 6.0, 12.0],
            )
        self.codec_sim = CodecSimulator(
            codec_wrapper=codec_wrapper,
            add_quant_noise=True,
        )

    def load_empirical_data(self, data_dir: str):
        """
        Load empirical IR and noise bank from a directory.

        Expected files:
          - impulse_response.npy or ir.npy
          - noise_*.npy or noise_*.wav files
        """
        # Try loading IR
        for ir_name in ["channel_impulse_response.npy", "impulse_response.npy", "ir.npy", "estimated_ir.npy"]:
            ir_path = os.path.join(data_dir, ir_name)
            if os.path.isfile(ir_path):
                self.ir_conv.load_empirical_ir(ir_path)
                break

        # Try loading noise bank
        self.noise.load_noise_bank(data_dir)

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """
        Apply full OTA channel augmentation (differentiable).

        Args:
            x: Audio tensor [B, 1, T] at 24kHz
            severity: 0.0 = mild (near pass-through), 1.0 = full empirical channel

        Returns:
            Augmented audio tensor, same shape as input
        """
        # Set training mode for randomization
        was_training = self.training
        self.train(True)

        if self.enable_codec:
            if self.codec_proxy is not None:
                x = self.codec_proxy(x, severity=severity)
            else:
                x = self.codec_sim(x, severity=severity)

        if self.enable_bandpass:
            x = self.bandpass(x, severity=severity)

        if self.enable_ir:
            x = self.ir_conv(x, severity=severity)

        if self.enable_compressor:
            x = self.compressor(x, severity=severity)

        if self.enable_noise:
            x = self.noise(x, severity=severity)

        self.train(was_training)
        return x
