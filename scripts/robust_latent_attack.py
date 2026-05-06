"""
Airgap-robust latent-space adversarial attacks on Audio LLMs.

Extends LatentCodecAttacker to optimize perturbations that survive the
over-the-air channel (speaker → air → microphone). The channel is modeled
as a differentiable convolution with an empirical impulse response (IR)
extracted from real MacBook Air → iPhone 12 Pro recordings.

Two-stage optimization:
  Stage 1 (warmup): vanilla attack — optimize delta directly against model
  Stage 2 (harden): optimize delta through IR convolution against model

Gradient flow (Stage 2):
    delta → z_adv = z_orig + delta
    → EnCodec.decoder(z_adv) → audio_24kHz
    → F.conv1d(audio, IR_kernel) → ir_audio_24kHz  [differentiable!]
    → resample(24k → 16k) → ir_audio_16kHz
    → model.compute_loss(ir_audio_16kHz, target_text)
    → loss.backward() → gradients flow through IR conv → decoder → delta

Usage:
    python robust_latent_attack.py --music jazz_1 --target "Turn left" --eps 1.0
"""

import os
import sys
import json
import time
import glob as glob_module
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from config import (
    CODECATTACK_LIB_ROOT, MODEL_PATHS, TARGET_MODEL,
    ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    MIMI_SAMPLE_RATE, MIMI_NUM_CODEBOOKS,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, ENCODEC_TEST_BITRATES, OPUS_TEST_BITRATES,
    RESULTS_DIR, compute_wer,
)
from music_carrier import mel_distance, multi_scale_mel_distance
from latent_attack import LatentCodecAttacker, AttackResult
from channel_augmentation import (
    DifferentiableCodecProxy, OTAChannelAugmentation, AdditiveColoredNoise,
    DifferentiableSpectralGating, BPDASpectralDenoiser,
)
from realistic_channel import SpeakerNonlinearity, EmpiricalNonlinearity, PhysicalChannelFilter
from aac_channel import AACCodecSTE

# WER threshold for untargeted disruption success
UNTARGETED_WER_THRESHOLD = 0.5

# Path to empirical IR and data directory
_EMPIRICAL_IR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soundcloud_recorded", "analysis", "channel_impulse_response.npy"
)
_EMPIRICAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soundcloud_recorded", "analysis"
)

sys.path.insert(0, CODECATTACK_LIB_ROOT)


class DifferentiableBandPassFilter(torch.nn.Module):
    """
    Differentiable FIR band-pass filter (Yakura & Sakuma, IJCAI 2019).

    Constrains adversarial perturbation energy to frequencies that survive
    the physical speaker→air→mic channel. Frequencies outside the passband
    are destroyed by the channel, so perturbation energy there is wasted.

    Applied to the waveform AFTER EnCodec decoding, BEFORE channel simulation.
    Fully differentiable via F.conv1d.
    """

    def __init__(
        self,
        low_hz: float = 1000.0,
        high_hz: float = 4000.0,
        sample_rate: int = ENCODEC_SAMPLE_RATE,
        num_taps: int = 257,
        device: str = "cuda",
        bands: list = None,
    ):
        super().__init__()
        from scipy.signal import firwin
        nyq = sample_rate / 2.0
        if bands is not None:
            # Multi-band: bands is list of (low, high) tuples
            # e.g. [(300, 600), (800, 1200)]
            cutoffs = []
            for lo, hi in bands:
                cutoffs.extend([lo / nyq, hi / nyq])
            taps = firwin(num_taps, cutoffs, pass_zero=False)
            self.bands = bands
        else:
            # Single band (legacy)
            taps = firwin(num_taps, [low_hz / nyq, high_hz / nyq], pass_zero=False)
            self.bands = [(low_hz, high_hz)]
        # F.conv1d does cross-correlation, so flip for convolution
        taps_t = torch.from_numpy(taps).float().flip(0)
        self.register_buffer("kernel", taps_t.unsqueeze(0).unsqueeze(0))
        self.pad = num_taps // 2
        self.low_hz = low_hz
        self.high_hz = high_hz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Band-pass filter audio. Input: [B, 1, T]. Output: same shape."""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        kernel = self.kernel.to(x.device, x.dtype)
        y = F.conv1d(x, kernel, padding=self.pad)
        y = y[..., :x.shape[-1]]
        if squeeze:
            y = y.squeeze(0)
        return y


class DifferentiableIRConv(torch.nn.Module):
    """
    Differentiable convolution with room impulse responses.

    Supports two modes:
      1. Single IR: loads one .npy file (legacy, our measured IR)
      2. Multi-IR (Yakura-style): loads a directory of .wav/.npy RIR files.
         Each forward pass randomly samples one IR for EoT diversity.

    Gradients flow through the convolution. IRs are frozen.
    """

    def __init__(self, ir_path: str = None, ir_dir: str = None,
                 sample_rate: int = ENCODEC_SAMPLE_RATE,
                 max_ir_length: int = None,
                 device: str = "cuda"):
        super().__init__()
        self._ir_bank = []
        self.ir_len = 0
        self._sample_rate = sample_rate

        if ir_dir and os.path.isdir(ir_dir):
            # Load directory of RIR files (Yakura-style diverse RIRs)
            import glob
            import soundfile as sf_io
            rir_files = sorted(
                glob.glob(os.path.join(ir_dir, "**", "*.wav"), recursive=True)
                + glob.glob(os.path.join(ir_dir, "**", "*.npy"), recursive=True)
            )
            max_len = 0
            for f in rir_files:
                try:
                    if f.endswith('.npy'):
                        ir = np.load(f).astype(np.float64)
                    else:
                        # Use soundfile (17x faster than librosa for same-SR files)
                        ir, sr = sf_io.read(f, dtype='float64')
                        if ir.ndim > 1:
                            ir = ir[:, 0]  # mono
                        if sr != sample_rate:
                            import librosa
                            ir = librosa.resample(ir, orig_sr=sr, target_sr=sample_rate)
                    # Normalize IR energy
                    ir = ir / (np.max(np.abs(ir)) + 1e-10)
                    if max_ir_length and len(ir) > max_ir_length:
                        ir = ir[:max_ir_length]
                    max_len = max(max_len, len(ir))
                    self._ir_bank.append(ir)
                except Exception:
                    continue
            if self._ir_bank:
                # Pad all to same length and store as tensor bank
                padded = []
                for ir in self._ir_bank:
                    padded.append(np.pad(ir, (0, max_len - len(ir))))
                bank_np = np.stack(padded)  # [N_irs, max_len]
                # Flip for convolution
                bank_t = torch.from_numpy(bank_np).float().flip(1)
                # Shape: [N_irs, 1, 1, max_len]
                self.register_buffer("ir_bank", bank_t.unsqueeze(1).unsqueeze(1))
                self.ir_len = max_len
                self._n_irs = len(self._ir_bank)
            else:
                raise ValueError(f"No valid RIR files found in {ir_dir}")
        elif ir_path and os.path.isfile(ir_path):
            # Single IR (legacy mode)
            ir_np = np.load(ir_path)
            ir_tensor = torch.from_numpy(ir_np).float().flip(0)
            self.register_buffer("ir_bank", ir_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            self.ir_len = len(ir_np)
            self._n_irs = 1
        else:
            raise FileNotFoundError(f"No IR source: ir_path={ir_path}, ir_dir={ir_dir}")

    def forward(self, x: torch.Tensor, ir_idx: int = None) -> torch.Tensor:
        """
        Apply IR convolution with random RIR selection.

        Args:
            x: Audio tensor [B, 1, T] at 24kHz
            ir_idx: If given, use this specific RIR index (overrides random/eval selection)

        Returns:
            IR-convolved audio [B, 1, T], gain-normalized to match input RMS
        """
        if ir_idx is not None:
            idx = ir_idx
        elif self.training and self._n_irs > 1:
            idx = torch.randint(self._n_irs, (1,)).item()
        else:
            idx = 0
        kernel = self.ir_bank[idx]  # [1, 1, ir_len]

        pad = self.ir_len - 1
        y = F.conv1d(F.pad(x, (pad, 0)), kernel.to(x.device, x.dtype))
        y = y[..., :x.shape[-1]]

        # Gain-normalize: match RMS of input so only spectral shape changes
        orig_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-10)
        y_rms = torch.sqrt(torch.mean(y ** 2, dim=-1, keepdim=True) + 1e-10)
        y = y * (orig_rms / y_rms)

        return y

    def select_diverse_subset(self, n: int = 20) -> list:
        """
        Select a diverse subset of RIR indices by energy decay (proxy for RT60).

        Estimates each RIR's decay rate from its energy envelope, then picks
        n indices evenly spaced across the decay-rate distribution.

        Returns:
            List of RIR indices (ints)
        """
        if self._n_irs <= n:
            return list(range(self._n_irs))

        # Estimate decay rate for each RIR from energy envelope
        # ir_bank is [N, 1, 1, ir_len] (already flipped for conv, so flip back)
        bank = self.ir_bank.squeeze(1).squeeze(1).flip(1)  # [N, ir_len]
        decay_rates = []
        for i in range(self._n_irs):
            ir = bank[i].float()
            # Compute energy in 10ms windows
            win = int(0.01 * self._sample_rate)
            energy = ir.unfold(0, win, win).pow(2).mean(dim=1)
            # Decay rate = ratio of energy in last quarter vs first quarter
            q = len(energy) // 4
            if q > 0:
                early = energy[:q].mean().item() + 1e-12
                late = energy[-q:].mean().item() + 1e-12
                decay_rates.append(late / early)  # higher = more reverberant
            else:
                decay_rates.append(0.0)

        # Sort by decay rate and pick evenly spaced indices
        sorted_indices = sorted(range(self._n_irs), key=lambda i: decay_rates[i])
        step = len(sorted_indices) / n
        selected = [sorted_indices[int(i * step)] for i in range(n)]
        return selected


# Default path to empirical transfer function
_EMPIRICAL_TF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soundcloud_recorded", "ir_trained_qwen2_audio_calm_1_eps_5.0", "real_channel_tf.npy"
)
_EMPIRICAL_TF_FREQS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soundcloud_recorded", "ir_trained_qwen2_audio_calm_1_eps_5.0", "real_channel_tf_freqs.npy"
)


class DifferentiableFreqResponse(torch.nn.Module):
    """
    Differentiable frequency response augmentation using empirical FIR filter.

    Loads the measured channel transfer function (MacBook speaker -> air -> iPhone mic)
    and applies it via F.conv1d. Fully differentiable — real gradients flow through.

    Maintains a bank of jittered filters for EoT-style randomization during training:
    each forward pass picks a random filter variant so the optimizer learns robustness
    to frequency response variations, not just the mean response.
    """

    def __init__(
        self,
        tf_path: str = _EMPIRICAL_TF_PATH,
        tf_freqs_path: str = _EMPIRICAL_TF_FREQS_PATH,
        sample_rate: int = ENCODEC_SAMPLE_RATE,
        num_taps: int = 257,
        n_variants: int = 16,
        jitter: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()
        from scipy.signal import firwin2

        tf_power = np.load(tf_path)
        tf_freqs = np.load(tf_freqs_path)
        tf_amplitude = np.sqrt(np.maximum(tf_power, 1e-10))

        nyq = sample_rate / 2.0
        norm_freqs = tf_freqs / nyq
        valid = (norm_freqs >= 0) & (norm_freqs <= 1.0)
        norm_freqs = norm_freqs[valid]
        tf_amplitude = tf_amplitude[valid]

        # Ensure endpoints
        if norm_freqs[0] > 0:
            norm_freqs = np.concatenate([[0], norm_freqs])
            tf_amplitude = np.concatenate([[tf_amplitude[0]], tf_amplitude])
        if norm_freqs[-1] < 1.0:
            norm_freqs = np.concatenate([norm_freqs, [1.0]])
            tf_amplitude = np.concatenate([tf_amplitude, [tf_amplitude[-1]]])

        # Clamp extreme values
        tf_amplitude = np.clip(tf_amplitude, 10**(-30/20), 10**(20/20))

        # Design default filter
        default_taps = firwin2(num_taps, norm_freqs, tf_amplitude)
        self.register_buffer("default_taps",
                             torch.FloatTensor(default_taps).view(1, 1, -1))

        # Build jittered filter bank for EoT
        bank = [torch.FloatTensor(default_taps)]
        for _ in range(n_variants - 1):
            jitter_factors = 1.0 + (np.random.rand(len(tf_amplitude)) - 0.5) * 2 * jitter
            jittered_amp = tf_amplitude * jitter_factors
            jittered_amp = np.clip(jittered_amp, 10**(-30/20), 10**(20/20))
            taps = firwin2(num_taps, norm_freqs, jittered_amp)
            bank.append(torch.FloatTensor(taps))
        # Shape: (n_variants, 1, 1, num_taps)
        self.register_buffer("filter_bank",
                             torch.stack(bank).unsqueeze(1).unsqueeze(1))
        self.num_taps = num_taps
        self.pad = num_taps // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency response.

        Args:
            x: Audio tensor [B, 1, T] or [1, T] at 24kHz

        Returns:
            Frequency-shaped audio, RMS-normalized to match input.
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        # Pick random filter during training, default during eval
        if self.training and self.filter_bank.shape[0] > 1:
            idx = torch.randint(self.filter_bank.shape[0], (1,)).item()
            kernel = self.filter_bank[idx]
        else:
            kernel = self.default_taps

        kernel = kernel.to(x.device, x.dtype)
        filtered = F.conv1d(x, kernel, padding=self.pad)
        filtered = filtered[..., :x.shape[-1]]

        # RMS-normalize: preserve overall energy, only change spectral shape
        orig_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-10)
        filt_rms = torch.sqrt(torch.mean(filtered ** 2, dim=-1, keepdim=True) + 1e-10)
        filtered = filtered * (orig_rms / filt_rms)

        if squeeze:
            filtered = filtered.squeeze(0)
        return filtered


class RobustLatentCodecAttacker(LatentCodecAttacker):
    """
    Airgap-robust adversarial attack using differentiable channel simulation.

    Two-stage optimization:
      Stage 1: vanilla attack (no channel) to find a good initial perturbation
      Stage 2: fine-tune through differentiable channel for robustness

    Channel modes:
      - "ir": IR convolution only (linear channel model)
      - "codec": EnCodec STE proxy only (codec compression)
      - "full": codec proxy + IR convolution
      - "ota": Full OTA channel (codec + bandpass + IR + compressor + noise)
                with EoT averaging over multiple stochastic passes
      - "yakura_ota": Yakura-style OTA (bandpass perturbation + diverse RIRs
                + Gaussian noise). Requires --rir-dir with 600+ RIRs and
                --bandpass-low-hz/--bandpass-high-hz (typically 1000/4000).
      - "soundcloud": SoundCloud digital pipeline via AAC STE
                (resample + AAC + encoder padding, all with STE gradients)
      - "cyclic_ir": Cyclic gradient accumulation over diverse RIRs.
                Each step cycles through a batch of RIRs deterministically,
                accumulates gradients from each, then takes one averaged PGD step.
                Ensures all RIRs are seen equally during optimization.
      - "spec_ota": SpecAugment-based OTA robustness (inspired by "Attacker's
                Noise", 2025). Instead of simulating the physical channel with
                RIRs, applies SpecAugment (random mel frequency band masking)
                + temporal shift + additive noise during optimization. This
                forces perturbation to distribute across many frequency bands,
                achieving robustness to arbitrary frequency-selective fading
                without needing an accurate channel model.
                Our latent-space optimization provides codec robustness on
                top — something waveform-domain attacks cannot achieve.
    """

    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        encodec_bandwidth: float = ENCODEC_BANDWIDTH,
        eps: float = LATENT_EPS,
        alpha: float = LATENT_ALPHA,
        perceptual_weight: float = PERCEPTUAL_WEIGHT,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        # Robust-specific
        ir_path: str = _EMPIRICAL_IR_PATH,
        empirical_data_dir: str = _EMPIRICAL_DATA_DIR,
        warmup_ratio: float = 0.5,
        no_channel: bool = False,
        channel_only: bool = False,
        # Channel mode: "ir", "codec", "multi_bitrate", "full", "ota", "soundcloud",
        # "spec_ota", "yakura_ota", "empirical_ota", "diverse_ir", "cyclic_ir"
        channel_mode: str = "ir",
        aac_bitrates: list = None,
        proxy_bandwidths: list = None,
        n_eot_codec: int = 1,
        n_eot_samples: int = 4,
        channel_severity: float = 1.0,
        channel_curriculum: bool = False,
        freq_shaping: bool = False,
        # Time-shift augmentation (robustness to encoder padding)
        time_shift_ms: float = 0.0,
        # Frequency response augmentation (robustness to speaker/mic reshaping)
        freq_augment: bool = False,
        freq_augment_tf_path: str = _EMPIRICAL_TF_PATH,
        freq_augment_tf_freqs_path: str = _EMPIRICAL_TF_FREQS_PATH,
        freq_augment_jitter: float = 0.2,
        freq_augment_n_variants: int = 16,
        # Band-pass filter on perturbation (Yakura IJCAI 2019 technique)
        bandpass_low_hz: float = 0.0,
        bandpass_high_hz: float = 0.0,
        bandpass_bands: list = None,  # Multi-band: [(300,600),(800,1200)]
        # Diverse RIR directory (Yakura-style, replaces single IR file)
        rir_dir: str = None,
        max_ir_length: int = None,
        # SpecAugment OTA params (spec_ota mode)
        spec_augment_n_mask: int = 10,
        spec_augment_mask_size: int = 50,
        spec_augment_noise_eps: float = 0.02,
        # Gradient accumulation (simulates EoT via accumulated stochastic grads)
        grad_accum_steps: int = 1,
        # Measured channel response for frequency-constrained attack
        channel_response_path: str = None,
        # Speaker nonlinearity (soft clipping) during optimization
        speaker_nonlinearity: bool = False,
        speaker_drive: float = 2.0,
        speaker_mix: float = 0.3,
        # Skip IR convolution (use only nonlinearity + noise)
        skip_ir: bool = False,
        # Data-driven per-band nonlinearity (from extract_nonlinearity.py)
        empirical_nonlinearity: bool = False,
        empirical_nonlinearity_path: str = None,
        # Physical channel FIR filter (from measured PSD ratios)
        physical_channel_fir: bool = False,
        physical_channel_fir_path: str = None,
        # Robust FIR: FIR + per-band jitter + time shift + SpecAugment
        robust_fir: bool = False,
        robust_fir_band_jitter_db: float = 3.0,
        robust_fir_gain_jitter_db: float = 3.0,
        robust_fir_phase_jitter: float = 0.0,  # radians, 0=disabled
        # Spectral gating denoiser (proxy for iPhone VPIO noise suppression)
        spectral_denoise: bool = False,
        spectral_denoise_strength: float = 0.5,
        # BPDA denoiser: real noisereduce forward, proxy backward
        bpda_denoise: bool = False,
        bpda_denoise_strength: float = 0.9,
        bpda_denoise_passes: int = 1,
        # Music-shaped perturbation (denoiser evasion)
        spectral_match_weight: float = 0.0,
        modulation_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            target_model=target_model,
            encodec_bandwidth=encodec_bandwidth,
            eps=eps,
            alpha=alpha,
            perceptual_weight=perceptual_weight,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )

        self.warmup_ratio = warmup_ratio
        self.no_channel = no_channel
        self.channel_only = channel_only
        self.channel_mode = channel_mode
        self.grad_accum_steps = grad_accum_steps
        self._skip_ir = skip_ir
        if skip_ir:
            self.log("IR convolution SKIPPED (--no-ir flag)")
        # Random gain augmentation (simulates OTA volume loss)
        self._gain_range_db = kwargs.get('gain_range_db', None)
        if self._gain_range_db is not None:
            self.log(f"Random gain augmentation: {self._gain_range_db[0]:.0f} to "
                     f"{self._gain_range_db[1]:.0f} dB")
        # Time-shift augmentation: max shift in samples at 24kHz
        self._max_shift_samples = int(time_shift_ms / 1000.0 * ENCODEC_SAMPLE_RATE)
        if self._max_shift_samples > 0:
            self.log(f"Time-shift augmentation: 0-{time_shift_ms:.0f}ms "
                     f"(0-{self._max_shift_samples} samples at {ENCODEC_SAMPLE_RATE}Hz)")

        # MI-FGSM + DI transfer-mode (opt-in for cross-model transfer pilot).
        # Default "none" preserves the existing Adam-based optimizer path.
        # "mi_di"   = sign(momentum) step (classic MI-FGSM)
        # "mi_norm" = L1-mean-normalized momentum step (Adam-like scaling),
        #             default decay=0.9 to forget noisy EOT gradients faster.
        self.transfer_mode = kwargs.get('transfer_mode', 'none')
        _default_decay = 0.9 if self.transfer_mode == 'mi_norm' else 1.0
        _mi_decay_arg = kwargs.get('mi_decay', None)
        self._mi_decay = float(_mi_decay_arg) if _mi_decay_arg is not None else _default_decay
        self._mi_alpha = kwargs.get('mi_alpha', None)  # None => 2*eps/steps in attack()
        self._di_shift_samples = int(kwargs.get('di_shift_samples', 100))
        if self.transfer_mode in ('mi_di', 'mi_norm'):
            step_kind = "sign" if self.transfer_mode == 'mi_di' else "L1-norm"
            self.log(f"Transfer mode: MI-FGSM ({step_kind}) + DI "
                     f"(decay={self._mi_decay}, "
                     f"DI shift=±{self._di_shift_samples} samples @24kHz)")

        # Noise module for yakura_ota mode
        self.yakura_noise = None
        self._yakura_noise_snr_db = kwargs.get('noise_snr_db', 20.0)
        self._yakura_noise_snr_std = kwargs.get('noise_snr_std', 5.0)
        self._yakura_n_eot = kwargs.get('yakura_n_eot', n_eot_samples)

        # Band-pass filter for perturbation (Yakura IJCAI 2019 technique)
        self.bandpass = None
        if bandpass_bands is not None and len(bandpass_bands) > 0:
            self.bandpass = DifferentiableBandPassFilter(
                bands=bandpass_bands,
                sample_rate=ENCODEC_SAMPLE_RATE,
                device=device,
            ).to(device)
            bands_str = " + ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in bandpass_bands)
            self.log(f"Multi-band filter: {bands_str} Hz "
                     f"(constraining perturbation to channel passband)")
        elif bandpass_low_hz > 0 and bandpass_high_hz > bandpass_low_hz:
            self.bandpass = DifferentiableBandPassFilter(
                low_hz=bandpass_low_hz,
                high_hz=bandpass_high_hz,
                sample_rate=ENCODEC_SAMPLE_RATE,
                device=device,
            ).to(device)
            self.log(f"Band-pass filter: {bandpass_low_hz:.0f}-{bandpass_high_hz:.0f} Hz "
                     f"(constraining perturbation to channel passband)")

        # Store rir_dir for IR initialization
        self._rir_dir = rir_dir
        self._max_ir_length = max_ir_length
        self._diverse_ir_indices = None  # Set during init for diverse_ir mode
        self._diverse_ir_n = kwargs.get('diverse_ir_n', 20)
        self._worst_case_loss = kwargs.get('worst_case_loss', True)

        # Frequency response augmentation
        self.freq_response = None
        if freq_augment:
            if os.path.isfile(freq_augment_tf_path) and os.path.isfile(freq_augment_tf_freqs_path):
                self.freq_response = DifferentiableFreqResponse(
                    tf_path=freq_augment_tf_path,
                    tf_freqs_path=freq_augment_tf_freqs_path,
                    sample_rate=ENCODEC_SAMPLE_RATE,
                    n_variants=freq_augment_n_variants,
                    jitter=freq_augment_jitter,
                    device=device,
                ).to(device)
                self.freq_response.train()
                self.log(f"Freq response augmentation: ON "
                         f"({freq_augment_n_variants} variants, jitter={freq_augment_jitter})")
            else:
                self.log(f"WARNING: Freq response TF not found at {freq_augment_tf_path}, "
                         f"augmentation DISABLED")

        self.n_eot_codec = n_eot_codec
        self.n_eot_samples = n_eot_samples
        self.channel_severity = channel_severity
        self.channel_curriculum = channel_curriculum

        # Initialize codec proxy for "codec" and "full" modes
        self.codec_proxy = None
        if channel_mode in ("codec", "full") and not no_channel:
            if not self._uses_mimi:
                self.codec_proxy = DifferentiableCodecProxy(
                    encodec_model=self.codec.model,
                    proxy_bandwidths=proxy_bandwidths,
                ).to(device)
                bw_str = ", ".join(str(b) for b in self.codec_proxy.proxy_bandwidths)
                self.log(f"Codec proxy initialized (bandwidths: [{bw_str}] kbps, "
                         f"EoT samples: {n_eot_codec})")
            else:
                self.log("WARNING: Codec proxy not supported with Mimi codec, "
                         "falling back to IR-only")
                self.channel_mode = "ir"

        # Initialize Opus proxy for "multi_bitrate" mode (Bundle A — S1 robust-optimization).
        self.opus_proxy = None
        self._multi_bitrate_kbps = kwargs.get(
            "multi_bitrate_kbps", [16, 24, 32, 64, 128]
        )
        if channel_mode == "multi_bitrate" and not no_channel:
            self._init_multi_bitrate_proxy(sample_rate=ENCODEC_SAMPLE_RATE)

        # Initialize full OTA channel for "ota" mode
        self.ota_channel = None
        if channel_mode == "ota" and not no_channel:
            if empirical_data_dir is None:
                empirical_data_dir = _EMPIRICAL_DATA_DIR
            has_empirical = os.path.isdir(empirical_data_dir)
            # When using empirical IR, disable bandpass and compressor:
            # the IR already captures speaker+air+mic frequency response
            # and the residual noise captures nonlinear distortion.
            # Only add codec proxy (SoundCloud compression) + IR + noise.
            self.ota_channel = OTAChannelAugmentation(
                sample_rate=ENCODEC_SAMPLE_RATE,
                enable_codec=not self._uses_mimi,
                codec_wrapper=self.codec.model if not self._uses_mimi else None,
                enable_bandpass=not has_empirical,
                enable_compressor=not has_empirical,
                enable_ir=True,
                enable_noise=True,
            ).to(device)
            if has_empirical:
                self.ota_channel.load_empirical_data(empirical_data_dir)
                self.log(f"OTA channel loaded empirical data from: {empirical_data_dir}")
                self.log(f"  Bandpass/compressor DISABLED (empirical IR already includes these)")
                if self.ota_channel.ir_conv._empirical:
                    self.log(f"  IR: empirical ({self.ota_channel.ir_conv.ir_length} samples)")
                if self.ota_channel.noise._noise_bank:
                    self.log(f"  Noise bank: {len(self.ota_channel.noise._noise_bank)} samples")
            else:
                self.log(f"WARNING: Empirical data dir not found: {empirical_data_dir}, "
                         "using synthetic channel (bandpass + compressor enabled)")
            self.log(f"OTA channel: EoT samples={n_eot_samples}, "
                     f"severity={channel_severity}, curriculum={channel_curriculum}")

        # Initialize SoundCloud STE channel for "soundcloud" mode
        self.sc_ste = None
        if channel_mode == "soundcloud" and not no_channel:
            sc_bitrates = aac_bitrates if aac_bitrates else [128, 160, 192]
            self.sc_ste = AACCodecSTE(
                sample_rate=ENCODEC_SAMPLE_RATE,
                aac_bitrates=sc_bitrates,
                soundcloud_mode=True,
                soundcloud_sr=44100,
            ).to(device)
            br_str = ", ".join(str(b) for b in sc_bitrates)
            self.log(f"SoundCloud STE channel (bitrates: [{br_str}] kbps)")

        # Initialize spec_ota mode: SpecAugment + time shift + additive noise
        # No RIRs, no bandpass, no channel model — just mel-domain augmentation
        self._spec_augment_enabled = False
        self._spec_augment_n_mask = spec_augment_n_mask
        self._spec_augment_mask_size = spec_augment_mask_size
        self._spec_augment_noise_eps = spec_augment_noise_eps
        if channel_mode == "spec_ota" and not no_channel:
            self._spec_augment_enabled = True
            # Use additive noise module for waveform-domain noise
            self.spec_ota_noise = AdditiveColoredNoise(
                sample_rate=ENCODEC_SAMPLE_RATE,
                snr_db_mean=self._yakura_noise_snr_db,
                snr_db_std=self._yakura_noise_snr_std,
            ).to(device)
            self.spec_ota_noise.train()
            self.log(f"SpecAugment OTA channel initialized:")
            self.log(f"  SpecAugment: n_mask={spec_augment_n_mask}, "
                     f"mask_size={spec_augment_mask_size}")
            self.log(f"  Additive noise: eps={spec_augment_noise_eps}")
            if self._max_shift_samples > 0:
                self.log(f"  Time shift: 0-{time_shift_ms:.0f}ms")
            else:
                self.log(f"  Time shift: DISABLED (set --time-shift-ms for robustness)")

        # Measured channel response filter (can be used with any mode)
        self._channel_filter = None
        if channel_response_path and os.path.exists(channel_response_path):
            self._init_channel_filter(channel_response_path, device)

        # Initialize empirical_ota mode: measured IR + real noise bank
        self._empirical_ota_ir = None
        self._empirical_ota_noise_bank = None
        self._empirical_ota_n_eot = kwargs.get('empirical_ota_n_eot', n_eot_samples)
        if channel_mode == "empirical_ota" and not no_channel:
            emp_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "empirical_ota"
            )
            # Allow override via empirical_data_dir
            if empirical_data_dir and os.path.isdir(empirical_data_dir):
                emp_dir = empirical_data_dir

            # Load measured IR (single averaged or bank of individual IRs)
            ir_bank_dir = os.path.join(emp_dir, "ir_bank")
            ir_file = os.path.join(emp_dir, "channel_impulse_response.npy")
            use_ir_bank = kwargs.get('ir_bank', False) and os.path.isdir(ir_bank_dir)
            if use_ir_bank:
                self._empirical_ota_ir = DifferentiableIRConv(
                    ir_dir=ir_bank_dir, device=device,
                ).to(device)
            elif os.path.isfile(ir_file):
                self._empirical_ota_ir = DifferentiableIRConv(
                    ir_path=ir_file, device=device,
                ).to(device)
            else:
                raise FileNotFoundError(
                    f"Empirical IR not found: {ir_file}\n"
                    "Copy from soundcloud_recorded/analysis/ to data/empirical_ota/")
            self._empirical_ota_ir.train()

            # Load noise bank (residual_noise_*.npy)
            noise_files = sorted(glob_module.glob(
                os.path.join(emp_dir, "residual_noise_[0-9]*.npy")))
            if noise_files:
                noise_list = []
                for nf in noise_files:
                    n = np.load(nf).astype(np.float32)
                    noise_list.append(torch.from_numpy(n))
                # Pad to same length
                max_nlen = max(n.shape[0] for n in noise_list)
                padded = [F.pad(n, (0, max_nlen - n.shape[0])) for n in noise_list]
                noise_bank = torch.stack(padded)  # [N_noise, T]
                self._empirical_ota_noise_bank = noise_bank.to(device)
                self.log(f"Empirical OTA channel initialized:")
                self.log(f"  IR: {ir_file} ({self._empirical_ota_ir.ir_len} samples, "
                         f"{self._empirical_ota_ir.ir_len / ENCODEC_SAMPLE_RATE * 1000:.0f}ms)")
                self.log(f"  Noise bank: {len(noise_files)} real recordings")
                self.log(f"  EoT samples: {self._empirical_ota_n_eot}")
            else:
                self.log(f"WARNING: No noise files found in {emp_dir}, using noise-free mode")
                self.log(f"Empirical OTA channel initialized (IR only):")
                self.log(f"  IR: {ir_file}")

            # Per-band gain/phase jitter for robustness to channel variance
            self._ota_band_jitter_db = kwargs.get('ota_band_jitter_db', 0.0)
            self._ota_phase_jitter = kwargs.get('ota_phase_jitter', 0.0)
            if self._ota_band_jitter_db > 0 or self._ota_phase_jitter > 0:
                self.log(f"  Band jitter: ±{self._ota_band_jitter_db}dB, "
                         f"phase jitter: ±{self._ota_phase_jitter:.2f}rad")

        # Speaker nonlinearity (soft clipping) for empirical_ota
        self._speaker_nonlinearity = None
        if speaker_nonlinearity and channel_mode == "empirical_ota" and not no_channel:
            self._speaker_nonlinearity = SpeakerNonlinearity(
                drive=speaker_drive, mix=speaker_mix,
            ).to(device)
            self._speaker_nonlinearity.train()
            self.log(f"Speaker nonlinearity enabled: drive={speaker_drive}, mix={speaker_mix}")

        # Data-driven per-band nonlinearity (from extract_nonlinearity.py)
        self._empirical_nonlinearity = None
        if empirical_nonlinearity and channel_mode == "empirical_ota" and not no_channel:
            self._empirical_nonlinearity = EmpiricalNonlinearity(
                model_path=empirical_nonlinearity_path,
                sample_rate=ENCODEC_SAMPLE_RATE,
            ).to(device)
            self._empirical_nonlinearity.train()
            self.log("Empirical per-band nonlinearity enabled (data-driven)")

        # Physical channel FIR (measured PSD-based transfer function)
        self._physical_channel_fir = None
        self._robust_fir = robust_fir
        if (physical_channel_fir or robust_fir) and channel_mode == "empirical_ota" and not no_channel:
            fir_jitter = 0.3 if robust_fir else 0.1
            self._physical_channel_fir = PhysicalChannelFilter(
                fir_path=physical_channel_fir_path,
                sample_rate=ENCODEC_SAMPLE_RATE,
                jitter=fir_jitter,
                robust=robust_fir,
                band_jitter_db=robust_fir_band_jitter_db,
                gain_jitter_db=robust_fir_gain_jitter_db,
                phase_jitter=robust_fir_phase_jitter,
            ).to(device)
            self._physical_channel_fir.train()
            if robust_fir:
                # Enable SpecAugment + time shift for robust FIR mode
                self._spec_augment_enabled = True
                if time_shift_ms == 0.0:
                    self._time_shift_samples = int(0.1 * ENCODEC_SAMPLE_RATE)  # 100ms default
                phase_str = f" + phase (±{robust_fir_phase_jitter:.1f}rad)" if robust_fir_phase_jitter > 0 else ""
                self.log(f"Robust FIR mode: PSD FIR + band jitter (±{robust_fir_band_jitter_db}dB) "
                         f"+ gain (±{robust_fir_gain_jitter_db}dB){phase_str} + SpecAugment + time shift")
            else:
                self.log("Physical channel FIR filter enabled (measured PSD ratios)")

        # Differentiable spectral gating denoiser (proxy for iPhone VPIO)
        self._spectral_denoiser = None
        if bpda_denoise and channel_mode == "empirical_ota" and not no_channel:
            self._spectral_denoiser = BPDASpectralDenoiser(
                prop_decrease=bpda_denoise_strength,
                sample_rate=ENCODEC_SAMPLE_RATE,
                randomize=True,
                prop_range=(max(0.3, bpda_denoise_strength - 0.3),
                            min(1.0, bpda_denoise_strength + 0.1)),
                n_passes=bpda_denoise_passes,
            ).to(device)
            self._spectral_denoiser.train()
            passes_str = f", {bpda_denoise_passes}x passes" if bpda_denoise_passes > 1 else ""
            self.log(f"BPDA denoiser enabled: strength={bpda_denoise_strength:.2f}{passes_str} "
                     f"(real noisereduce forward, proxy backward)")
        elif spectral_denoise and channel_mode == "empirical_ota" and not no_channel:
            self._spectral_denoiser = DifferentiableSpectralGating(
                prop_decrease=spectral_denoise_strength,
                n_fft=1024,
                hop_length=256,
                randomize=True,
            ).to(device)
            self._spectral_denoiser.train()
            self.log(f"Spectral gating denoiser enabled: strength={spectral_denoise_strength:.2f} "
                     f"(proxy for iPhone VPIO noise suppression)")

        # Music-shaped perturbation: spectral matching loss
        self._spectral_match_weight = spectral_match_weight
        self._carrier_spectrum = None  # precomputed in attack()
        if spectral_match_weight > 0:
            self.log(f"Spectral matching loss enabled: weight={spectral_match_weight}")

        # 3-5Hz amplitude modulation loss (speech-like temporal dynamics)
        self._modulation_weight = modulation_weight
        if modulation_weight > 0:
            self.log(f"Modulation loss enabled: weight={modulation_weight} "
                     f"(encourages 3-5Hz speech-like amplitude modulation)")

        # Initialize yakura_ota mode: diverse RIRs + Gaussian noise
        if channel_mode == "yakura_ota" and not no_channel:
            if not (self._rir_dir and os.path.isdir(self._rir_dir)):
                raise ValueError(
                    "yakura_ota mode requires --rir-dir with diverse RIR files.\n"
                    "Generate with: python generate_rirs.py --output-dir data/rirs"
                )
            self.ir_conv = DifferentiableIRConv(
                ir_dir=self._rir_dir,
                sample_rate=ENCODEC_SAMPLE_RATE,
                max_ir_length=self._max_ir_length,
                device=device,
            ).to(device)
            self.ir_conv.train()  # Enable random RIR selection
            self.yakura_noise = AdditiveColoredNoise(
                sample_rate=ENCODEC_SAMPLE_RATE,
                snr_db_mean=self._yakura_noise_snr_db,
                snr_db_std=self._yakura_noise_snr_std,
            ).to(device)
            self.yakura_noise.train()
            self.log(f"Yakura OTA channel initialized:")
            self.log(f"  RIRs: {self.ir_conv._n_irs} from {self._rir_dir} "
                     f"(max {self.ir_conv.ir_len / ENCODEC_SAMPLE_RATE * 1000:.0f}ms)")
            self.log(f"  Noise: Gaussian, SNR={self._yakura_noise_snr_db}±{self._yakura_noise_snr_std} dB")
            self.log(f"  EoT samples: {self._yakura_n_eot}")
            if self.bandpass is None:
                self.log(f"  WARNING: No bandpass filter configured! Set --bandpass-low-hz/--bandpass-high-hz "
                         f"(recommended: 1000/4000)")

        # Load IR(s) for "ir", "full", and "diverse_ir" modes
        self.ir_conv = None if channel_mode != "yakura_ota" else self.ir_conv
        needs_ir = (channel_mode in ("ir", "full", "diverse_ir", "cyclic_ir")) and not no_channel
        if needs_ir:
            if self._rir_dir and os.path.isdir(self._rir_dir):
                # Diverse RIR mode (Yakura-style)
                self.ir_conv = DifferentiableIRConv(
                    ir_dir=self._rir_dir,
                    sample_rate=ENCODEC_SAMPLE_RATE,
                    max_ir_length=self._max_ir_length,
                    device=device,
                ).to(device)
                self.ir_conv.train()  # Enable random RIR selection
                self.log(f"Loaded {self.ir_conv._n_irs} diverse RIRs from: {self._rir_dir} "
                         f"(max IR length: {self.ir_conv.ir_len} samples, "
                         f"{self.ir_conv.ir_len / ENCODEC_SAMPLE_RATE * 1000:.0f} ms)")

                # For diverse_ir mode: select a fixed diverse subset
                if channel_mode == "diverse_ir":
                    self._diverse_ir_indices = self.ir_conv.select_diverse_subset(
                        n=self._diverse_ir_n)
                    loss_type = "worst-case (max)" if self._worst_case_loss else "average (mean)"
                    self.log(f"Diverse IR mode: {len(self._diverse_ir_indices)} RIRs selected "
                             f"by decay diversity, loss={loss_type}")
                elif channel_mode == "cyclic_ir":
                    self._diverse_ir_indices = self.ir_conv.select_diverse_subset(
                        n=self._diverse_ir_n)
                    # Batch size per step (how many RIRs to accumulate gradients from)
                    self._cyclic_batch_size = min(self._diverse_ir_n, len(self._diverse_ir_indices))
                    self.log(f"Cyclic IR mode: {len(self._diverse_ir_indices)} RIRs selected, "
                             f"batch_size={self._cyclic_batch_size} per step, "
                             f"gradient accumulation")
            elif os.path.isfile(ir_path):
                # Single IR (legacy mode)
                self.ir_conv = DifferentiableIRConv(
                    ir_path=ir_path, device=device,
                ).to(device)
                self.log(f"Loaded single empirical IR from: {ir_path} "
                         f"({self.ir_conv.ir_len} samples, "
                         f"{self.ir_conv.ir_len / ENCODEC_SAMPLE_RATE * 1000:.0f} ms)")
            else:
                raise FileNotFoundError(
                    f"No IR source found: ir_path={ir_path}, rir_dir={self._rir_dir}\n"
                    "Provide --ir-path or --rir-dir."
                )

        if no_channel:
            self.log("Channel DISABLED (--no-channel)")
        else:
            self.log(f"Channel mode: {self.channel_mode}")

        self.log(f"Warmup ratio: {warmup_ratio} (vanilla warmup for first "
                 f"{int(warmup_ratio * 100)}% of steps)")

    def _get_severity(self, step: int, total_steps: int) -> float:
        """Get channel severity, optionally with curriculum warmup."""
        if not self.channel_curriculum:
            return self.channel_severity
        # Linear ramp from 0.1 to channel_severity over the hardening phase
        warmup_steps = int(total_steps * self.warmup_ratio)
        harden_steps = total_steps - warmup_steps
        if harden_steps <= 0:
            return self.channel_severity
        progress = min((step - warmup_steps) / harden_steps, 1.0)
        return 0.1 + progress * (self.channel_severity - 0.1)

    def _apply_band_jitter(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random per-band gain and phase jitter (differentiable).

        Simulates the stochastic variance in a physical OTA channel that
        a single linear IR cannot capture. Each call applies different
        random gain/phase perturbations per frequency band.
        """
        band_jitter_db = getattr(self, '_ota_band_jitter_db', 0.0)
        phase_jitter = getattr(self, '_ota_phase_jitter', 0.0)
        if band_jitter_db <= 0 and phase_jitter <= 0:
            return audio

        # Work in frequency domain
        x = audio.squeeze()  # [T]
        X = torch.fft.rfft(x)
        freqs = torch.fft.rfftfreq(x.shape[-1], 1.0 / ENCODEC_SAMPLE_RATE).to(audio.device)

        # Define bands matching the Marshall acoustic analysis
        bands = [(0, 100), (100, 300), (300, 1000), (1000, 2000),
                 (2000, 4000), (4000, 6000), (6000, 8000)]

        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                continue
            # Random gain jitter
            if band_jitter_db > 0:
                gain_db = (torch.rand(1, device=audio.device) * 2 - 1) * band_jitter_db
                gain = 10.0 ** (gain_db / 20.0)
                X[mask] = X[mask] * gain
            # Random phase jitter
            if phase_jitter > 0:
                phase = (torch.rand(1, device=audio.device) * 2 - 1) * phase_jitter
                X[mask] = X[mask] * torch.exp(1j * phase)

        result = torch.fft.irfft(X, n=x.shape[-1])
        return result.unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    def _apply_time_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random time shift by prepending silence (differentiable).

        Simulates encoder padding that platforms like SoundCloud add (~46ms).
        Truncates from the end to maintain the original length.
        """
        if self._max_shift_samples <= 0:
            return audio
        shift = torch.randint(0, self._max_shift_samples + 1, (1,)).item()
        if shift == 0:
            return audio
        pad = torch.zeros_like(audio[..., :shift])
        return torch.cat([pad, audio[..., :-shift]], dim=-1)

    def _apply_random_gain(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random gain attenuation (differentiable).

        Simulates the volume loss of speaker→air→mic channel.
        Our cross-pair analysis shows OTA gain is -12 to -20 dB.
        """
        if self._gain_range_db is None:
            return audio
        lo, hi = self._gain_range_db
        gain_db = lo + (hi - lo) * torch.rand(1, device=audio.device).item()
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear

    def _apply_freq_response(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply empirical frequency response augmentation (differentiable).

        Simulates the spectral reshaping of the speaker→air→mic channel.
        Uses a random filter variant from the jittered bank during training.
        """
        if self.freq_response is None:
            return audio
        return self.freq_response(audio)

    def _init_channel_filter(self, npz_path: str, device: str):
        """
        Build a differentiable FIR filter from a measured channel response.

        The filter simulates the speaker→air→mic frequency response so the
        optimizer learns to place perturbation energy in surviving bands.
        """
        import numpy as np
        from scipy.signal import firwin2

        data = np.load(npz_path)
        freqs = data['freqs']        # Hz
        H_db = data['H_db']          # dB
        sr = int(data['sr'])

        # Normalize: shift so the peak is 0 dB (we only care about relative shape)
        H_db_norm = H_db - np.max(H_db)

        # Clip to avoid extreme attenuation (floor at -30 dB)
        H_db_norm = np.clip(H_db_norm, -30, 0)

        # Convert to linear magnitude
        H_linear = 10 ** (H_db_norm / 20.0)

        # Normalize frequency axis for firwin2 (0 to 1, Nyquist = 1)
        nyquist = sr / 2.0
        freq_norm = freqs / nyquist
        # Ensure endpoints
        freq_norm = np.clip(freq_norm, 0, 1)
        if freq_norm[0] != 0:
            freq_norm = np.insert(freq_norm, 0, 0)
            H_linear = np.insert(H_linear, 0, H_linear[0])
        if freq_norm[-1] != 1:
            freq_norm = np.append(freq_norm, 1)
            H_linear = np.append(H_linear, H_linear[-1])

        # Design FIR filter (order 255 for good freq resolution)
        n_taps = 255
        taps = firwin2(n_taps, freq_norm, H_linear)

        # Store as conv kernel [1, 1, n_taps]
        kernel = torch.from_numpy(taps).float().unsqueeze(0).unsqueeze(0).to(device)
        self._channel_filter = kernel
        self._channel_filter_pad = n_taps // 2
        self.log(f"Measured channel filter loaded from {npz_path}")
        self.log(f"  FIR taps: {n_taps}, attenuation floor: -30 dB")

    def _apply_channel_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply measured channel frequency response filter (differentiable)."""
        if self._channel_filter is None:
            return audio
        # audio: [1, 1, T] or [1, T]
        needs_unsqueeze = audio.dim() == 2
        if needs_unsqueeze:
            audio = audio.unsqueeze(0)
        kernel = self._channel_filter.to(audio.dtype)
        filtered = F.conv1d(
            F.pad(audio, (self._channel_filter_pad, self._channel_filter_pad)),
            kernel)
        filtered = filtered[..., :audio.shape[-1]]
        # RMS normalize to preserve energy
        orig_rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True) + 1e-10)
        filt_rms = torch.sqrt(torch.mean(filtered ** 2, dim=-1, keepdim=True) + 1e-10)
        filtered = filtered * (orig_rms / filt_rms)
        if needs_unsqueeze:
            filtered = filtered.squeeze(0)
        return filtered

    def _apply_channel(self, audio_24k: torch.Tensor,
                       step: int = 0, total_steps: int = 0) -> torch.Tensor:
        """
        Apply channel augmentation based on channel_mode.

        Pipeline: time shift -> freq response -> channel-specific augmentation.
        All differentiable — real gradients flow through.

        For "ota" mode, averages n_eot_samples stochastic passes through the
        full OTA channel (bandpass + IR + compressor + noise). Each pass has
        different random noise samples, filter selections, and SNR jitter,
        giving Expectation-over-Transformation (EoT) gradients.

        For codec/full modes with n_eot_codec > 1, averages multiple
        codec proxy passes (EoT over codec bandwidth sampling).
        """
        # Apply band-pass filter to perturbation only (Yakura technique):
        # forces adversarial energy into the channel's passband
        if self.bandpass is not None and hasattr(self, '_original_audio_24k'):
            perturbation = audio_24k - self._original_audio_24k
            perturbation_filtered = self.bandpass(perturbation)
            audio_24k = self._original_audio_24k + perturbation_filtered

        # Apply time shift first (differentiable) — skip for soundcloud mode
        # since SoundCloud STE already simulates encoder padding
        if self.channel_mode != "soundcloud":
            audio_24k = self._apply_time_shift(audio_24k)

        # Apply freq response augmentation (differentiable)
        # This teaches the perturbation to survive speaker/mic spectral reshaping
        audio_24k = self._apply_freq_response(audio_24k)

        # Apply random gain (differentiable)
        # Simulates OTA volume loss (-12 to -20 dB measured)
        audio_24k = self._apply_random_gain(audio_24k)

        if self.channel_mode == "soundcloud":
            severity = self._get_severity(step, total_steps)
            return self.sc_ste(audio_24k, severity=severity)
        elif self.channel_mode == "ota":
            severity = self._get_severity(step, total_steps)
            if self.n_eot_samples > 1:
                acc = torch.zeros_like(audio_24k)
                for _ in range(self.n_eot_samples):
                    acc = acc + self.ota_channel(audio_24k, severity=severity)
                return acc / self.n_eot_samples
            return self.ota_channel(audio_24k, severity=severity)
        elif self.channel_mode == "codec":
            if self.n_eot_codec > 1:
                # Average multiple codec proxy passes for EoT
                acc = torch.zeros_like(audio_24k)
                for _ in range(self.n_eot_codec):
                    acc = acc + self.codec_proxy(audio_24k)
                return acc / self.n_eot_codec
            return self.codec_proxy(audio_24k)
        elif self.channel_mode == "multi_bitrate":
            return self._apply_channel_multi_bitrate(
                audio_24k, step=step, total_steps=total_steps
            )
        elif self.channel_mode == "full":
            # Codec proxy then IR convolution
            if self.n_eot_codec > 1:
                acc = torch.zeros_like(audio_24k)
                for _ in range(self.n_eot_codec):
                    acc = acc + self.codec_proxy(audio_24k)
                audio_coded = acc / self.n_eot_codec
            else:
                audio_coded = self.codec_proxy(audio_24k)
            return self.ir_conv(audio_coded)
        elif self.channel_mode == "spec_ota":
            # SpecAugment OTA: translation + additive noise in waveform domain.
            # SpecAugment itself is applied on mel features inside compute_loss()
            # (the _spec_augment_enabled flag signals the attack loop to pass
            #  spec_augment=True to compute_loss on channel steps).

            # 1. Circular translation (Attacker's Noise, Sadasivan et al. 2025)
            #    A_translation(x) = x[i:] ⊕ x[:i], i ~ U[0, L)
            L = audio_24k.shape[-1]
            i = torch.randint(0, L, (1,)).item()
            audio_24k = torch.cat([audio_24k[..., i:], audio_24k[..., :i]], dim=-1)

            # 2. Measured channel frequency response filter
            audio_24k = self._apply_channel_filter(audio_24k)

            # 3. Additive uniform noise
            if self._spec_augment_noise_eps > 0:
                noise = torch.empty_like(audio_24k).uniform_(
                    -self._spec_augment_noise_eps, self._spec_augment_noise_eps)
                audio_24k = audio_24k + noise.detach()

            # 4. SpecAugment is applied later on mel features in compute_loss()
            return audio_24k
        elif self.channel_mode == "yakura_ota":
            # Yakura-style OTA: diverse RIR convolution + Gaussian noise
            # Bandpass on perturbation is already applied above
            severity = self._get_severity(step, total_steps)
            if self._yakura_n_eot > 1:
                acc = torch.zeros_like(audio_24k)
                for _ in range(self._yakura_n_eot):
                    aug = self.ir_conv(audio_24k)
                    aug = self.yakura_noise(aug, severity=severity)
                    acc = acc + aug
                return acc / self._yakura_n_eot
            aug = self.ir_conv(audio_24k)
            return self.yakura_noise(aug, severity=severity)
        elif self.channel_mode == "empirical_ota":
            # Empirical OTA: measured IR convolution + speaker nonlinearity + real noise
            n_eot = self._empirical_ota_n_eot
            acc = torch.zeros_like(audio_24k)
            for _ in range(n_eot):
                aug = audio_24k
                # 0. Circular time shift (robust_fir mode)
                if self._robust_fir and self._time_shift_samples > 0:
                    L = aug.shape[-1]
                    i = torch.randint(0, self._time_shift_samples, (1,)).item()
                    aug = torch.cat([aug[..., i:], aug[..., :i]], dim=-1)
                # 1. Channel frequency shaping (differentiable)
                if self._physical_channel_fir is not None:
                    # Use measured PSD-based FIR (replaces both IR and NL)
                    aug = self._physical_channel_fir(aug)
                elif self._skip_ir:
                    pass
                else:
                    aug = self._empirical_ota_ir(aug)
                    # 1.5. Per-band gain/phase jitter (simulates channel variance)
                    aug = self._apply_band_jitter(aug)
                    # 2. Speaker nonlinearity (differentiable)
                    if self._empirical_nonlinearity is not None:
                        aug = self._empirical_nonlinearity(aug)
                    elif self._speaker_nonlinearity is not None:
                        aug = self._speaker_nonlinearity(aug)
                # 3. Add real noise sample (scaled to match measured SNR)
                if self._empirical_ota_noise_bank is not None:
                    noise_idx = torch.randint(
                        self._empirical_ota_noise_bank.shape[0], (1,)).item()
                    noise = self._empirical_ota_noise_bank[noise_idx]
                    # Trim or tile noise to match audio length
                    T = aug.shape[-1]
                    if noise.shape[0] >= T:
                        noise = noise[:T]
                    else:
                        repeats = (T // noise.shape[0]) + 1
                        noise = noise.repeat(repeats)[:T]
                    # Reshape to match audio: [B, 1, T]
                    noise = noise.view(1, 1, -1).to(aug.dtype)
                    aug = aug + noise.detach()
                # 4. Additive uniform noise (robust_fir mode)
                if self._robust_fir:
                    noise_eps = 0.02
                    noise = torch.empty_like(aug).uniform_(-noise_eps, noise_eps)
                    aug = aug + noise.detach()
                # 5. Spectral gating denoiser (proxy for iPhone VPIO)
                if self._spectral_denoiser is not None:
                    aug = self._spectral_denoiser(aug)
                acc = acc + aug
            return acc / n_eot
        elif self.channel_mode == "diverse_ir":
            # Diverse IR mode: apply ALL selected RIRs, return worst-case
            # The caller handles worst-case loss selection in the training loop.
            # Here we just apply a random RIR from the diverse subset for the
            # single-output path. The multi-RIR logic is in the attack loop.
            idx = self._diverse_ir_indices[
                torch.randint(len(self._diverse_ir_indices), (1,)).item()
            ]
            return self.ir_conv(audio_24k, ir_idx=idx)
        else:
            # "ir" mode (legacy default)
            return self.ir_conv(audio_24k)

    def _init_multi_bitrate_proxy(self, sample_rate: int = None):
        """Instantiate the DifferentiableOpusProxy for channel_mode=multi_bitrate."""
        if sample_rate is None:
            sample_rate = ENCODEC_SAMPLE_RATE
        from channel_augmentation import DifferentiableOpusProxy
        self.opus_proxy = DifferentiableOpusProxy(
            sample_rate=sample_rate,
            bitrates_kbps=self._multi_bitrate_kbps,
        )
        self.log(
            f"Multi-bitrate Opus EOT enabled: bitrates_kbps={self._multi_bitrate_kbps}"
        )

    def _apply_channel_multi_bitrate(self, audio_24k, step: int, total_steps: int):
        """Route audio through the Opus proxy (random bitrate per call)."""
        severity = self._get_severity(step, total_steps)
        return self.opus_proxy(audio_24k, severity=severity)

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps: int = ATTACK_STEPS,
        check_every: int = CHECK_EVERY,
        music_name: str = "",
        use_multi_scale_loss: bool = False,
        prompt: str = None,
        untargeted: bool = False,
    ) -> AttackResult:
        """
        Run airgap-robust latent-space adversarial attack.

        Stage 1 (warmup): optimize delta directly (no IR)
        Stage 2 (harden): optimize delta through IR convolution
        """
        # If no channel, fall back to vanilla attack
        if self.no_channel:
            return super().attack(
                music_wav, target_text, steps=steps, check_every=check_every,
                music_name=music_name, use_multi_scale_loss=use_multi_scale_loss,
                prompt=prompt, untargeted=untargeted,
            )

        start_time = time.time()
        music_wav = music_wav.to(self.device)

        if hasattr(self.target_model, 'reset_cache'):
            self.target_model.reset_cache()

        # Encode to latent space
        z_original = self.encode_music(music_wav)

        # Cache original audio for perceptual loss, baseline output, and band-pass
        with torch.no_grad():
            original_audio_24k = self.codec.decode_from_continuous(z_original)
            # Store for band-pass filter (to extract perturbation = adv - original)
            self._original_audio_24k = original_audio_24k.detach()
            # Precompute carrier spectrum for spectral matching loss
            if self._spectral_match_weight > 0:
                carrier_stft = torch.stft(
                    original_audio_24k.squeeze(), n_fft=1024, hop_length=256,
                    window=torch.hann_window(1024, device=original_audio_24k.device),
                    return_complex=True)
                carrier_mag = carrier_stft.abs().mean(dim=-1)  # avg over time
                self._carrier_spectrum = (carrier_mag / (carrier_mag.sum() + 1e-8)).detach()
            if self._uses_mimi:
                original_audio_for_model = original_audio_24k.squeeze(0)
            else:
                original_audio_for_model = torchaudio.functional.resample(
                    original_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
            original_output = self.target_model.generate(
                original_audio_for_model, prompt=prompt
            )

        self.log(f"Original output: {original_output}")

        if untargeted:
            target_text = original_output
            self.log("UNTARGETED mode: maximizing loss on original output")
        self.log(f"Target: {target_text}")
        if prompt:
            self.log(f"Prompt: {prompt}")

        # Initialize perturbation
        delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        # MI-FGSM momentum buffer (used when transfer_mode in {"mi_di","mi_norm"}).
        # Resolved here so we can pick a sensible alpha from steps if not provided.
        _is_mi_mode = self.transfer_mode in ("mi_di", "mi_norm")
        mi_momentum = (torch.zeros_like(delta).detach()
                       if _is_mi_mode else None)
        mi_alpha_resolved = (self._mi_alpha
                             if self._mi_alpha is not None
                             else 2.0 * self.eps / max(steps, 1))
        if _is_mi_mode:
            self.log(f"MI-FGSM step size (resolved): {mi_alpha_resolved:.6f}")

        # Stage boundaries
        warmup_steps = int(steps * self.warmup_ratio)

        # Tracking
        history = {
            "loss": [], "behavior_loss": [], "perceptual_loss": [],
            "outputs": [], "steps": [], "stage": [],
        }
        best_loss = float('inf')
        best_delta = delta.data.clone()
        best_ir_delta = None  # Checkpoint delta when IR target is achieved
        best_ir_step = None
        success = False

        self.log(f"\nStarting airgap-robust attack: eps={self.eps}, "
                 f"alpha={self.alpha}, steps={steps}")
        stage2_label = {
            "ir": "IR hardening",
            "codec": "codec proxy hardening",
            "multi_bitrate": f"multi-bitrate Opus EOT ({self._multi_bitrate_kbps} kbps)",
            "full": "codec proxy + IR hardening",
            "ota": f"full OTA channel (EoT x{self.n_eot_samples})",
            "yakura_ota": f"Yakura OTA (RIR+noise, EoT x{self._yakura_n_eot})",
            "spec_ota": f"SpecAugment OTA (n_mask={self._spec_augment_n_mask}, "
                        f"mask={self._spec_augment_mask_size}, noise={self._spec_augment_noise_eps})",
            "cyclic_ir": f"Cyclic IR (grad accum x{getattr(self, '_cyclic_batch_size', '?')})",
            "empirical_ota": f"Empirical OTA (measured IR+noise, EoT x{self._empirical_ota_n_eot})",
        }.get(self.channel_mode, "channel hardening")
        if self.channel_only:
            self.log(f"Channel-only mode: ALL {steps} steps through {stage2_label}")
        else:
            self.log(f"Stage 1 (steps 1-{warmup_steps}): vanilla warmup")
            self.log(f"Stage 2 (steps {warmup_steps + 1}-{steps}): alternating direct/{stage2_label}")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        if self.grad_accum_steps > 1:
            effective_steps = steps // self.grad_accum_steps
            self.log(f"Gradient accumulation: {self.grad_accum_steps} micro-steps per update "
                     f"({steps} total steps = {effective_steps} effective updates)")
        self.log("-" * 60)

        # Enable gradients through codec decoder
        if self._uses_mimi:
            self.codec.set_decode_train_mode(True)
        else:
            self.codec.model.decoder.train()

        _ga = self.grad_accum_steps  # shorthand
        for step in range(steps):
            if _ga <= 1 or step % _ga == 0:
                optimizer.zero_grad()

            # Channel-only mode: every step goes through channel simulation
            # Default mode: warmup (vanilla) then alternating direct/channel
            if self.channel_only:
                use_channel = True
            elif step < warmup_steps:
                use_channel = False
            else:
                use_channel = (step % 2 == 1)

            # Forward: decode latents to audio
            z_adv = z_original + delta
            audio_adv_24k = self.codec.decode_from_continuous(z_adv)

            # Apply channel on channel steps
            if use_channel:
                audio_for_loss = self._apply_channel(
                    audio_adv_24k, step=step, total_steps=steps)
            else:
                audio_for_loss = audio_adv_24k

            # DI-FGSM (input diversity) augmentation for transfer attack:
            # apply a random circular temporal shift on audio_for_loss so the
            # model sees a slightly perturbed input each step. Differentiable
            # because torch.roll preserves gradient flow through the rolled tensor.
            if _is_mi_mode and self._di_shift_samples > 0:
                shift = int(torch.randint(
                    -self._di_shift_samples, self._di_shift_samples + 1, (1,)).item())
                if shift != 0:
                    audio_for_loss = torch.roll(audio_for_loss, shifts=shift, dims=-1)

            # Compute behavior loss
            # For cyclic_ir mode: gradient accumulation over batch of RIRs
            if (use_channel and self.channel_mode == "cyclic_ir"
                    and self._diverse_ir_indices is not None):
                # Cyclic gradient accumulation: each step processes a batch
                # of RIRs, cycling through all of them deterministically.
                n_irs = len(self._diverse_ir_indices)
                batch_size = self._cyclic_batch_size
                # Determine which RIRs to use this step (cyclic)
                start_idx = (step * batch_size) % n_irs
                batch_indices = []
                for i in range(batch_size):
                    batch_indices.append(
                        self._diverse_ir_indices[(start_idx + i) % n_irs])

                # Accumulate gradients from each RIR
                accum_b_loss = 0.0
                p_loss_val = 0.0
                for k, ir_idx in enumerate(batch_indices):
                    # Fresh forward pass each time (re-decode to get fresh graph)
                    z_adv_k = z_original + delta
                    audio_k = self.codec.decode_from_continuous(z_adv_k)
                    audio_ir = self.ir_conv(audio_k, ir_idx=ir_idx)
                    if self._uses_mimi:
                        ll = self.codec.encode_to_continuous(audio_ir)
                        b_loss_k = self.target_model.compute_loss_from_latents(
                            ll, target_text)
                    else:
                        a16 = torchaudio.functional.resample(
                            audio_ir.squeeze(0), ENCODEC_SAMPLE_RATE,
                            TARGET_SAMPLE_RATE)
                        b_loss_k = self.target_model.compute_loss(
                            a16, target_text, prompt=prompt)
                    # Scale loss by 1/batch_size so gradients average out
                    scaled_loss = b_loss_k / batch_size
                    if untargeted:
                        scaled_loss = -scaled_loss
                    # Add perceptual loss (only once, from first RIR's audio)
                    if k == 0 and self.perceptual_weight > 0:
                        if use_multi_scale_loss:
                            p_loss = multi_scale_mel_distance(
                                audio_k.squeeze(0),
                                original_audio_24k.squeeze(0).detach(),
                                sample_rate=ENCODEC_SAMPLE_RATE,
                            )
                        else:
                            p_loss = mel_distance(
                                audio_k.squeeze(0),
                                original_audio_24k.squeeze(0).detach(),
                                sample_rate=ENCODEC_SAMPLE_RATE,
                            )
                        scaled_loss = scaled_loss + self.perceptual_weight * p_loss
                        p_loss_val = p_loss.item()
                    # Add modulation loss (only once, from first RIR's audio)
                    if k == 0 and self._modulation_weight > 0:
                        pert_k = audio_k.squeeze() - original_audio_24k.squeeze().detach()
                        frame_len_k = int(0.025 * ENCODEC_SAMPLE_RATE)
                        hop_len_k = int(0.010 * ENCODEC_SAMPLE_RATE)
                        frames_k = pert_k.unfold(-1, frame_len_k, hop_len_k)
                        env_k = torch.sqrt(torch.mean(frames_k ** 2, dim=-1) + 1e-10)
                        mod_fft_k = torch.fft.rfft(env_k - env_k.mean())
                        mod_pow_k = mod_fft_k.abs()
                        n_fr_k = env_k.shape[0]
                        mf_k = torch.fft.rfftfreq(n_fr_k, d=0.010).to(mod_pow_k.device)
                        sp_mask = (mf_k >= 3.0) & (mf_k <= 5.0)
                        tot_mask = (mf_k >= 0.5) & (mf_k <= 20.0)
                        mod_loss_k = -(mod_pow_k[sp_mask].mean() / (mod_pow_k[tot_mask].mean() + 1e-10))
                        scaled_loss = scaled_loss + self._modulation_weight * mod_loss_k
                    scaled_loss.backward()
                    accum_b_loss += b_loss_k.item()
                    # Free memory between RIRs
                    del z_adv_k, audio_k, audio_ir, b_loss_k, scaled_loss
                    if not self._uses_mimi:
                        del a16
                    torch.cuda.empty_cache()

                # Take PGD step with accumulated gradients
                optimizer.step()
                with torch.no_grad():
                    delta.data = torch.clamp(delta.data, -self.eps, self.eps)

                # Track average behavior loss across the batch
                b_loss_val = accum_b_loss / batch_size
                loss_val = b_loss_val + p_loss_val * self.perceptual_weight

                history["loss"].append(loss_val)
                history["behavior_loss"].append(b_loss_val)
                history["perceptual_loss"].append(p_loss_val)
                history["stage"].append("channel")

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_delta = delta.data.clone()

                # Progress logging
                if (step + 1) % 10 == 0:
                    stage = "Cyclic-IR"
                    self.log(
                        f"Step {step + 1:4d}/{steps} [{stage}] | "
                        f"Loss: {loss_val:.4f} | "
                        f"Behavior: {b_loss_val:.4f} | "
                        f"Perceptual: {p_loss_val:.4f}"
                    )
                if (step + 1) % 50 == 0:
                    # Quick eval
                    with torch.no_grad():
                        z_eval = z_original + delta
                        a_eval = self.codec.decode_from_continuous(z_eval)
                        a16_eval = torchaudio.functional.resample(
                            a_eval.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                        direct_out = self.target_model.generate(
                            a16_eval, prompt=prompt)
                        # Also eval through a random RIR
                        rand_ir = self._diverse_ir_indices[
                            step % len(self._diverse_ir_indices)]
                        a_ir = self.ir_conv(a_eval, ir_idx=rand_ir)
                        a16_ir = torchaudio.functional.resample(
                            a_ir.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                        ir_out = self.target_model.generate(
                            a16_ir, prompt=prompt)
                    d_ok = target_text.lower() in direct_out.lower()
                    i_ok = target_text.lower() in ir_out.lower()
                    self.log(f"  -> Direct: {'OK' if d_ok else 'FAIL'} | {direct_out[:80]}")
                    self.log(f"  -> IR:     {'OK' if i_ok else 'FAIL'} | {ir_out[:80]}")

                continue  # Skip the normal backward/step below

            # For diverse_ir mode on channel steps: compute loss for EACH RIR
            # in the diverse subset and take the worst case (max loss).
            if (use_channel and self.channel_mode == "diverse_ir"
                    and self._diverse_ir_indices is not None):
                # Worst-case loss across diverse RIR subset.
                # To avoid OOM: compute loss for each RIR sequentially,
                # find the worst-case, then do a single backward on that.
                # First pass: find worst-case index (no grad)
                with torch.no_grad():
                    worst_loss_val = -1.0
                    worst_ir_idx = self._diverse_ir_indices[0]
                    for ir_idx in self._diverse_ir_indices:
                        audio_ir = self.ir_conv(audio_adv_24k.detach(), ir_idx=ir_idx)
                        if self._uses_mimi:
                            ll = self.codec.encode_to_continuous(audio_ir)
                            lv = self.target_model.compute_loss_from_latents(
                                ll, target_text).item()
                        else:
                            a16 = torchaudio.functional.resample(
                                audio_ir.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                            lv = self.target_model.compute_loss(
                                a16, target_text, prompt=prompt).item()
                        if self._worst_case_loss:
                            if lv > worst_loss_val:
                                worst_loss_val = lv
                                worst_ir_idx = ir_idx
                        else:
                            # For average mode, accumulate
                            worst_loss_val += lv
                    torch.cuda.empty_cache()

                # Second pass: compute loss WITH gradients on the worst-case RIR only
                if self._worst_case_loss:
                    audio_ir = self.ir_conv(audio_adv_24k, ir_idx=worst_ir_idx)
                    if self._uses_mimi:
                        ll = self.codec.encode_to_continuous(audio_ir)
                        behavior_loss = self.target_model.compute_loss_from_latents(
                            ll, target_text)
                    else:
                        a16 = torchaudio.functional.resample(
                            audio_ir.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                        behavior_loss = self.target_model.compute_loss(
                            a16, target_text, prompt=prompt)
                else:
                    # Average mode: use random RIR from subset (gradient on one)
                    rand_idx = self._diverse_ir_indices[
                        torch.randint(len(self._diverse_ir_indices), (1,)).item()]
                    audio_ir = self.ir_conv(audio_adv_24k, ir_idx=rand_idx)
                    if self._uses_mimi:
                        ll = self.codec.encode_to_continuous(audio_ir)
                        behavior_loss = self.target_model.compute_loss_from_latents(
                            ll, target_text)
                    else:
                        a16 = torchaudio.functional.resample(
                            audio_ir.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                        behavior_loss = self.target_model.compute_loss(
                            a16, target_text, prompt=prompt)
            elif self._uses_mimi:
                loss_latents = self.codec.encode_to_continuous(audio_for_loss)
                behavior_loss = self.target_model.compute_loss_from_latents(
                    loss_latents, target_text
                )
            else:
                audio_16k = torchaudio.functional.resample(
                    audio_for_loss.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                # For spec_ota mode: apply SpecAugment on mel features during
                # channel steps. This forces the perturbation to distribute
                # across many frequency bands for OTA robustness.
                use_spec_aug = (use_channel and self._spec_augment_enabled)
                if self._spec_augment_enabled:
                    behavior_loss = self.target_model.compute_loss(
                        audio_16k, target_text, prompt=prompt,
                        spec_augment=use_spec_aug,
                        spec_augment_n_mask=self._spec_augment_n_mask,
                        spec_augment_mask_size=self._spec_augment_mask_size,
                    )
                else:
                    behavior_loss = self.target_model.compute_loss(
                        audio_16k, target_text, prompt=prompt,
                    )

            total_loss = behavior_loss
            if untargeted:
                total_loss = -total_loss

            # Perceptual loss (on pre-IR audio to preserve quality)
            p_loss_val = 0.0
            if self.perceptual_weight > 0:
                if use_multi_scale_loss:
                    p_loss = multi_scale_mel_distance(
                        audio_adv_24k.squeeze(0),
                        original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE,
                    )
                else:
                    p_loss = mel_distance(
                        audio_adv_24k.squeeze(0),
                        original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE,
                    )
                total_loss = total_loss + self.perceptual_weight * p_loss
                p_loss_val = p_loss.item()

            # Spectral matching loss: penalize perturbation spectral divergence from carrier
            sm_loss_val = 0.0
            if self._spectral_match_weight > 0 and self._carrier_spectrum is not None:
                perturbation = audio_adv_24k.squeeze() - self._original_audio_24k.squeeze()
                pert_stft = torch.stft(
                    perturbation, n_fft=1024, hop_length=256,
                    window=torch.hann_window(1024, device=perturbation.device),
                    return_complex=True)
                pert_mag = pert_stft.abs().mean(dim=-1)  # avg over time
                pert_spectrum = pert_mag / (pert_mag.sum() + 1e-8)
                # KL divergence: carrier_spectrum as target distribution
                sm_loss = F.kl_div(
                    (pert_spectrum + 1e-8).log(),
                    self._carrier_spectrum,
                    reduction='sum')
                total_loss = total_loss + self._spectral_match_weight * sm_loss
                sm_loss_val = sm_loss.item()

            # Modulation loss: encourage 3-5Hz amplitude modulation (speech-like dynamics)
            mod_loss_val = 0.0
            if self._modulation_weight > 0:
                perturbation = audio_adv_24k.squeeze() - self._original_audio_24k.squeeze()
                # Compute frame-level energy envelope (differentiable)
                frame_len = int(0.025 * ENCODEC_SAMPLE_RATE)  # 25ms frames
                hop_len = int(0.010 * ENCODEC_SAMPLE_RATE)    # 10ms hop
                n_frames = (perturbation.shape[-1] - frame_len) // hop_len + 1
                # Unfold into frames
                frames = perturbation.unfold(-1, frame_len, hop_len)  # [n_frames, frame_len]
                envelope = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-10)  # [n_frames]
                # FFT of envelope to get modulation spectrum
                envelope_centered = envelope - envelope.mean()
                mod_fft = torch.fft.rfft(envelope_centered)
                mod_power = mod_fft.abs()
                mod_freqs = torch.fft.rfftfreq(n_frames, d=0.010).to(mod_power.device)
                # Energy in 3-5Hz band (speech syllable rate)
                speech_mask = (mod_freqs >= 3.0) & (mod_freqs <= 5.0)
                total_mask = (mod_freqs >= 0.5) & (mod_freqs <= 20.0)
                speech_energy = mod_power[speech_mask].mean()
                total_energy = mod_power[total_mask].mean() + 1e-10
                # Negative because we want to MAXIMIZE 3-5Hz modulation
                mod_loss = -speech_energy / total_energy
                total_loss = total_loss + self._modulation_weight * mod_loss
                mod_loss_val = mod_loss.item()

            # Gradient accumulation: scale loss and only step every _ga steps
            if _ga > 1:
                (total_loss / _ga).backward()
                if (step + 1) % _ga == 0:
                    if _is_mi_mode:
                        with torch.no_grad():
                            g = delta.grad
                            g_norm = g / (g.abs().mean() + 1e-12)
                            mi_momentum.mul_(self._mi_decay).add_(g_norm)
                            if self.transfer_mode == "mi_di":
                                # Classic MI-FGSM: sign(momentum) step.
                                step_dir = mi_momentum.sign()
                            else:  # "mi_norm": Adam-like coordinate scaling
                                step_dir = mi_momentum / (mi_momentum.abs().mean() + 1e-12)
                            delta.data = delta.data - mi_alpha_resolved * step_dir
                            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                            delta.grad.zero_()
                    else:
                        optimizer.step()
                        with torch.no_grad():
                            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            else:
                total_loss.backward()
                if _is_mi_mode:
                    with torch.no_grad():
                        g = delta.grad
                        g_norm = g / (g.abs().mean() + 1e-12)
                        mi_momentum.mul_(self._mi_decay).add_(g_norm)
                        if self.transfer_mode == "mi_di":
                            # Classic MI-FGSM: sign(momentum) step.
                            step_dir = mi_momentum.sign()
                        else:  # "mi_norm": Adam-like coordinate scaling
                            step_dir = mi_momentum / (mi_momentum.abs().mean() + 1e-12)
                        delta.data = delta.data - mi_alpha_resolved * step_dir
                        delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                        delta.grad.zero_()
                else:
                    optimizer.step()
                    with torch.no_grad():
                        delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Track
            b_loss_val = behavior_loss.item()
            loss_val = total_loss.item()

            history["loss"].append(loss_val)
            history["behavior_loss"].append(b_loss_val)
            history["perceptual_loss"].append(p_loss_val)
            history["stage"].append("channel" if use_channel else "direct")

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            # Progress logging
            if (step + 1) % 10 == 0:
                stage_labels = {"ir": "S2-IR", "codec": "S2-codec", "multi_bitrate": "S2-MultiBR", "full": "S2-full", "ota": "S2-OTA", "yakura_ota": "S2-Yakura", "spec_ota": "S2-SpecOTA", "empirical_ota": "S2-EmpOTA"}
                if step < warmup_steps and not self.channel_only:
                    stage = "S1-vanilla"
                elif use_channel:
                    stage = stage_labels.get(self.channel_mode, "S2-ch")
                else:
                    stage = "S2-direct"
                msg = (f"Step {step + 1:4d}/{steps} [{stage}] | "
                       f"Loss: {loss_val:.4f} | "
                       f"Behavior: {b_loss_val:.4f} | "
                       f"Perceptual: {p_loss_val:.4f}")
                if sm_loss_val > 0:
                    msg += f" | SpecMatch: {sm_loss_val:.4f}"
                self.log(msg)

            # Periodic eval: test BOTH direct and IR paths
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta
                    audio_test_24k = self.codec.decode_from_continuous(z_test)

                    # Direct eval
                    if self._uses_mimi:
                        audio_direct = audio_test_24k.squeeze(0)
                    else:
                        audio_direct = torchaudio.functional.resample(
                            audio_test_24k.squeeze(0),
                            ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
                        )
                    output_direct = self.target_model.generate(
                        audio_direct, prompt=prompt
                    )

                    # Channel eval (IR, codec proxy, OTA, or both)
                    audio_ir_24k = self._apply_channel(
                        audio_test_24k, step=step, total_steps=steps)
                    if self._uses_mimi:
                        audio_ir_model = audio_ir_24k.squeeze(0)
                    else:
                        audio_ir_model = torchaudio.functional.resample(
                            audio_ir_24k.squeeze(0),
                            ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
                        )
                    output_ir = self.target_model.generate(
                        audio_ir_model, prompt=prompt
                    )

                history["outputs"].append({
                    "direct": output_direct,
                    "ir": output_ir,
                })
                history["steps"].append(step + 1)

                direct_ok = target_text.lower() in output_direct.lower()
                ir_ok = target_text.lower() in output_ir.lower()
                self.log(f"  -> Direct: {'OK' if direct_ok else 'FAIL'} | "
                         f"{output_direct[:80]}")
                self.log(f"  -> IR:     {'OK' if ir_ok else 'FAIL'} | "
                         f"{output_ir[:80]}")

                if not untargeted and ir_ok:
                    self.log(f"  -> IR TARGET ACHIEVED at step {step + 1}!")
                    success = True
                    best_ir_delta = delta.data.clone()
                    best_ir_step = step + 1

        # Restore eval mode and original bandwidth
        if self._uses_mimi:
            self.codec.set_decode_train_mode(False)
        else:
            self.codec.model.decoder.eval()
            # Restore original bandwidth (codec proxy may have changed it)
            if self.codec_proxy is not None:
                self.codec.model.set_target_bandwidth(self.codec._bandwidth)

        # Use best IR delta if we ever achieved IR target, otherwise use last delta
        if best_ir_delta is not None:
            self.log(f"\nUsing best IR delta from step {best_ir_step}")
            final_delta = best_ir_delta
        else:
            final_delta = delta.data

        # Final evaluation
        with torch.no_grad():
            z_final = z_original + final_delta
            audio_final_24k = self.codec.decode_from_continuous(z_final)
            if self._uses_mimi:
                audio_final_for_model = audio_final_24k.squeeze(0)
            else:
                audio_final_for_model = torchaudio.functional.resample(
                    audio_final_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
            final_output = self.target_model.generate(
                audio_final_for_model, prompt=prompt
            )

            # Also eval through channel (IR, codec proxy, OTA, or both)
            audio_ir_final = self._apply_channel(
                audio_final_24k, step=steps - 1, total_steps=steps)
            if self._uses_mimi:
                audio_ir_for_model = audio_ir_final.squeeze(0)
            else:
                audio_ir_for_model = torchaudio.functional.resample(
                    audio_ir_final.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
            ir_output = self.target_model.generate(
                audio_ir_for_model, prompt=prompt
            )

        # Success = target achieved through IR (the harder condition)
        success = False
        if untargeted:
            final_wer = compute_wer(target_text, ir_output)
            if final_wer > UNTARGETED_WER_THRESHOLD:
                success = True
            self.log(f"Final IR WER: {final_wer:.3f}")
        else:
            if target_text.lower() in ir_output.lower():
                success = True

        # SNR
        snr_db = self._compute_snr(
            original_audio_24k.squeeze().detach(),
            audio_final_24k.squeeze().detach(),
        )
        latent_snr_db = self._compute_snr(z_original.detach(), z_final.detach())

        duration = time.time() - start_time

        self.log("\n" + "=" * 60)
        self.log("Airgap-Robust Attack Complete")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Latent SNR: {latent_snr_db:.2f} dB")
        self.log(f"Direct Output: {final_output[:100]}")
        self.log(f"IR Output: {ir_output[:100]}")
        self.log(f"IR Success: {success}")
        self.log(f"Duration: {duration:.1f}s")
        self.log("=" * 60)

        result = AttackResult(
            original_wav=original_audio_24k.detach().cpu().squeeze(0),
            adversarial_wav=audio_final_24k.detach().cpu().squeeze(0),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=z_final.detach().cpu(),
            latent_perturbation=final_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            latent_snr_db=latent_snr_db,
            history=history,
            music_name=music_name,
            attack_duration_s=duration,
        )
        # Store IR output on the result object for benchmark to access
        result.ir_output = ir_output
        return result


def main():
    """CLI for single robust attack."""
    import argparse
    from music_carrier import load_music_by_name
    from config import MUSIC_FILES, DEFAULT_MUSIC, PROMPT_MODES, DEFAULT_PROMPT_MODE

    parser = argparse.ArgumentParser(
        description="Airgap-robust latent-space attack on Audio LLMs"
    )
    parser.add_argument("--music", type=str, default=DEFAULT_MUSIC)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=LATENT_EPS)
    parser.add_argument("--alpha", type=float, default=LATENT_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.5)
    parser.add_argument("--no-channel", action="store_true")
    parser.add_argument("--channel-only", action="store_true",
                        help="Use channel simulation on ALL steps (no warmup, no alternation)")
    parser.add_argument("--ir-path", type=str, default=_EMPIRICAL_IR_PATH)
    parser.add_argument("--channel-mode", type=str, default="ir",
                        choices=["ir", "codec", "multi_bitrate", "full", "ota", "yakura_ota", "diverse_ir", "cyclic_ir", "spec_ota", "empirical_ota"],
                        help="Channel mode: ir, codec, multi_bitrate, full, ota, yakura_ota, diverse_ir, cyclic_ir, spec_ota, empirical_ota")
    parser.add_argument("--diverse-ir-n", type=int, default=20,
                        help="Number of diverse RIRs to select for diverse_ir mode")
    parser.add_argument("--no-worst-case", action="store_true",
                        help="Use average loss instead of worst-case for diverse_ir mode")
    parser.add_argument("--proxy-bandwidths", type=str, default="1.5,3.0,6.0",
                        help="Comma-separated bandwidths for codec proxy (kbps)")
    parser.add_argument("--n-eot-codec", type=int, default=1,
                        help="Number of EoT codec proxy passes per step")
    # Legacy flags (kept for backward compat with run_robust.sh)
    parser.add_argument("--n-eot-samples", type=int, default=1)
    parser.add_argument("--channel-severity", type=float, default=1.0)
    parser.add_argument("--channel-curriculum", action="store_true")
    parser.add_argument("--freq-shaping", action="store_true")
    parser.add_argument("--empirical-data-dir", type=str, default=None)
    parser.add_argument("--time-shift-ms", type=float, default=0.0,
                        help="Max random time shift in ms (simulates encoder padding)")
    parser.add_argument("--freq-augment", action="store_true",
                        help="Enable freq response augmentation (empirical FIR from measured channel)")
    parser.add_argument("--gain-range-db", type=str, default=None,
                        help="Random gain range in dB, e.g. '-20,-10' (simulates OTA volume loss)")
    parser.add_argument("--freq-augment-jitter", type=float, default=0.2,
                        help="Jitter factor for freq response filter bank")
    parser.add_argument("--bandpass-low", type=float, default=0.0,
                        help="Band-pass filter low cutoff in Hz (0=disabled). "
                             "Yakura IJCAI'19 used 1000 Hz.")
    parser.add_argument("--bandpass-high", type=float, default=0.0,
                        help="Band-pass filter high cutoff in Hz (0=disabled). "
                             "Yakura IJCAI'19 used 4000 Hz.")
    parser.add_argument("--rir-dir", type=str, default=None,
                        help="Directory of diverse RIR wav/npy files for EoT. "
                             "Overrides --ir-path. Yakura used 615 real RIRs.")
    parser.add_argument("--max-ir-length", type=int, default=None,
                        help="Max IR length in samples (truncate long RIRs)")
    # SpecAugment OTA parameters
    parser.add_argument("--spec-augment-n-mask", type=int, default=10,
                        help="Number of mel frequency bands to mask per step (spec_ota mode)")
    parser.add_argument("--spec-augment-mask-size", type=int, default=50,
                        help="Max width of each masked mel band (spec_ota mode)")
    parser.add_argument("--spec-augment-noise-eps", type=float, default=0.02,
                        help="Uniform noise amplitude for spec_ota mode")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (simulates EoT via stochastic grad averaging)")
    parser.add_argument("--channel-response", type=str, default=None,
                        help="Path to measured channel response .npz for frequency-constrained attack")
    parser.add_argument("--speaker-nonlinearity", action="store_true",
                        help="Enable speaker soft clipping during optimization (empirical_ota mode)")
    parser.add_argument("--speaker-drive", type=float, default=2.0,
                        help="Speaker nonlinearity drive factor (higher = more distortion)")
    parser.add_argument("--speaker-mix", type=float, default=0.3,
                        help="Speaker nonlinearity wet/dry mix (0=bypass, 1=full distortion)")
    parser.add_argument("--no-ir", action="store_true",
                        help="Skip IR convolution in empirical_ota mode (use only nonlinearity + noise)")
    parser.add_argument("--empirical-nonlinearity", action="store_true",
                        help="Use data-driven per-band nonlinearity (from extract_nonlinearity.py)")
    parser.add_argument("--empirical-nonlinearity-path", type=str, default=None,
                        help="Path to nonlinearity_model.json (default: data/empirical_ota/)")
    parser.add_argument("--physical-channel-fir", action="store_true",
                        help="Use measured PSD-based FIR filter (replaces IR+NL with direct TF)")
    parser.add_argument("--physical-channel-fir-path", type=str, default=None,
                        help="Path to physical_channel_fir.npy")
    parser.add_argument("--robust-fir", action="store_true",
                        help="Robust FIR mode: PSD FIR + per-band jitter + gain jitter + SpecAugment + time shift")
    parser.add_argument("--robust-fir-band-jitter-db", type=float, default=3.0,
                        help="Per-band random gain jitter in dB for robust FIR mode")
    parser.add_argument("--robust-fir-gain-jitter-db", type=float, default=3.0,
                        help="Global gain jitter in dB for robust FIR mode")
    parser.add_argument("--robust-fir-phase-jitter", type=float, default=0.0,
                        help="Phase jitter in radians for robust FIR mode (pi=full random, 0=disabled)")
    parser.add_argument("--spectral-denoise", action="store_true",
                        help="Enable differentiable spectral gating denoiser (proxy for iPhone VPIO)")
    parser.add_argument("--spectral-denoise-strength", type=float, default=0.5,
                        help="Denoiser suppression strength (0=none, 0.3=mild, 0.5=moderate, 0.9=aggressive)")
    parser.add_argument("--bpda-denoise", action="store_true",
                        help="Enable BPDA denoiser (real noisereduce forward, proxy backward)")
    parser.add_argument("--bpda-denoise-strength", type=float, default=0.9,
                        help="BPDA denoiser prop_decrease (0.7=strong, 0.9=aggressive)")
    parser.add_argument("--bpda-denoise-passes", type=int, default=1,
                        help="Number of noisereduce passes in BPDA forward (2=double-pass)")
    parser.add_argument("--spectral-match-weight", type=float, default=0.0,
                        help="Weight for spectral matching loss (music-shaped perturbation)")
    parser.add_argument("--modulation-weight", type=float, default=0.0,
                        help="Weight for 3-5Hz amplitude modulation loss (encourages speech-like dynamics)")

    args = parser.parse_args()
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]
    music_wav = load_music_by_name(args.music)

    proxy_bws = [float(b) for b in args.proxy_bandwidths.split(",")]
    attacker = RobustLatentCodecAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        warmup_ratio=args.warmup_ratio,
        no_channel=args.no_channel,
        channel_only=args.channel_only,
        ir_path=args.ir_path,
        channel_mode=args.channel_mode,
        proxy_bandwidths=proxy_bws,
        n_eot_codec=args.n_eot_codec,
        n_eot_samples=args.n_eot_samples,
        channel_severity=args.channel_severity,
        channel_curriculum=args.channel_curriculum,
        empirical_data_dir=args.empirical_data_dir or _EMPIRICAL_DATA_DIR,
        time_shift_ms=args.time_shift_ms,
        freq_augment=args.freq_augment,
        freq_augment_jitter=args.freq_augment_jitter,
        gain_range_db=[float(x) for x in args.gain_range_db.split(",")] if args.gain_range_db else None,
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
        rir_dir=args.rir_dir,
        max_ir_length=args.max_ir_length,
        diverse_ir_n=args.diverse_ir_n,
        worst_case_loss=not args.no_worst_case,
        spec_augment_n_mask=args.spec_augment_n_mask,
        spec_augment_mask_size=args.spec_augment_mask_size,
        spec_augment_noise_eps=args.spec_augment_noise_eps,
        grad_accum_steps=args.grad_accum,
        channel_response_path=args.channel_response,
        speaker_nonlinearity=args.speaker_nonlinearity,
        speaker_drive=args.speaker_drive,
        speaker_mix=args.speaker_mix,
        skip_ir=args.no_ir,
        empirical_nonlinearity=args.empirical_nonlinearity,
        empirical_nonlinearity_path=args.empirical_nonlinearity_path,
        physical_channel_fir=args.physical_channel_fir,
        physical_channel_fir_path=args.physical_channel_fir_path,
        robust_fir=args.robust_fir,
        robust_fir_band_jitter_db=args.robust_fir_band_jitter_db,
        robust_fir_gain_jitter_db=args.robust_fir_gain_jitter_db,
        robust_fir_phase_jitter=args.robust_fir_phase_jitter,
        spectral_denoise=args.spectral_denoise,
        spectral_denoise_strength=args.spectral_denoise_strength,
        bpda_denoise=args.bpda_denoise,
        bpda_denoise_strength=args.bpda_denoise_strength,
        bpda_denoise_passes=args.bpda_denoise_passes,
        spectral_match_weight=args.spectral_match_weight,
        modulation_weight=args.modulation_weight,
    )

    result = attacker.attack(
        music_wav,
        target_text=args.target,
        steps=args.steps,
        music_name=args.music,
        prompt=prompt_text,
    )

    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "robust_single")
    attacker.save_result(result, output_dir)

    print(f"\nIR Success: {result.success}")
    print(f"Direct Output: {result.adversarial_output}")
    print(f"IR Output: {getattr(result, 'ir_output', 'N/A')}")
    print(f"SNR: {result.snr_db:.2f} dB")


if __name__ == "__main__":
    main()
