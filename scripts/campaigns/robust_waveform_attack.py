"""
Airgap-robust waveform-domain adversarial attacks on Audio LLMs.

Extends WaveformAttacker to optimize perturbations that survive the
over-the-air channel (speaker -> air -> microphone). Uses the same
differentiable OTA channel augmentation as RobustLatentCodecAttacker,
but operates directly in the 16kHz waveform domain.

Two-stage optimization:
  Stage 1 (warmup): vanilla waveform attack (no channel)
  Stage 2 (harden): optimize through differentiable OTA channel

The OTA channel operates at 24kHz (EnCodec native rate), so we upsample
the 16kHz perturbation to 24kHz for the channel, then downsample back.

Usage:
    python robust_waveform_attack.py --music jazz_1 --target "Turn left" --eps 0.01
"""

import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from config import (
    WHISPER_INJECT_ROOT, TARGET_MODEL,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    WAVEFORM_EPS, WAVEFORM_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, compute_wer,
)
from waveform_attack import WaveformAttacker
from latent_attack import AttackResult, UNTARGETED_WER_THRESHOLD
from music_carrier import mel_distance, multi_scale_mel_distance
from channel_augmentation import OTAChannelAugmentation, DifferentiableOpusProxy

_EMPIRICAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "soundcloud_recorded", "analysis"
)

sys.path.insert(0, WHISPER_INJECT_ROOT)


class RobustWaveformAttacker(WaveformAttacker):
    """
    Airgap-robust waveform-domain adversarial attack.

    Two-stage optimization:
      Stage 1: vanilla waveform attack (no channel) for warmup
      Stage 2: optimize through differentiable OTA channel for robustness

    The OTA channel operates at 24kHz, so perturbations at 16kHz are
    upsampled -> channel -> downsampled for the loss computation.
    """

    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        eps: float = WAVEFORM_EPS,
        alpha: float = WAVEFORM_ALPHA,
        perceptual_weight: float = PERCEPTUAL_WEIGHT,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        # Robust-specific
        empirical_data_dir: str = _EMPIRICAL_DATA_DIR,
        warmup_ratio: float = 0.5,
        no_channel: bool = False,
        n_eot_samples: int = 4,
        channel_severity: float = 1.0,
        channel_curriculum: bool = False,
        channel_mode: str = "ota",  # "ota" or "multi_bitrate"
        multi_bitrate_kbps: list = None,
        lowpass_hz: float = 0.0,  # >0 = project delta into [0, lowpass_hz] each step
        spectral_cutoff_hz: float = 0.0,  # >0 = optimize through an exact rFFT subspace
        alternate_after_warmup: bool = False,
    ):
        super().__init__(
            target_model=target_model,
            eps=eps,
            alpha=alpha,
            perceptual_weight=perceptual_weight,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )

        self.warmup_ratio = warmup_ratio
        self.no_channel = no_channel
        self.n_eot_samples = n_eot_samples
        self.channel_severity = channel_severity
        self.channel_curriculum = channel_curriculum
        self.channel_mode = channel_mode
        # Historical waveform runs used the channel on every post-warmup step.
        # Keep that behavior as the default, while allowing reviewer controls
        # to match RobustLatentCodecAttacker's direct/channel alternation.
        self.alternate_after_warmup = bool(alternate_after_warmup)
        self._multi_bitrate_kbps = list(multi_bitrate_kbps or [16, 24, 32, 64, 128])

        self.ota_channel = None
        self.opus_proxy = None
        if not no_channel and channel_mode == "multi_bitrate":
            self.opus_proxy = DifferentiableOpusProxy(
                sample_rate=ENCODEC_SAMPLE_RATE,
                bitrates_kbps=self._multi_bitrate_kbps,
            ).to(device)
            self.log(
                f"Multi-bitrate Opus EOT enabled: bitrates_kbps={self._multi_bitrate_kbps} "
                f"@ {ENCODEC_SAMPLE_RATE} Hz"
            )
        elif not no_channel:
            if empirical_data_dir is None:
                empirical_data_dir = _EMPIRICAL_DATA_DIR
            has_empirical = os.path.isdir(empirical_data_dir)
            self.ota_channel = OTAChannelAugmentation(
                sample_rate=ENCODEC_SAMPLE_RATE,
                enable_codec=False,  # No EnCodec proxy for waveform attacks
                codec_wrapper=None,
                enable_bandpass=not has_empirical,
                enable_compressor=not has_empirical,
                enable_ir=True,
                enable_noise=True,
            ).to(device)
            if has_empirical:
                self.ota_channel.load_empirical_data(empirical_data_dir)
                self.log(f"OTA channel loaded empirical data from: {empirical_data_dir}")
                self.log(f"  Bandpass/compressor DISABLED (empirical IR includes these)")
                if self.ota_channel.ir_conv._empirical:
                    self.log(f"  IR: empirical ({self.ota_channel.ir_conv.ir_length} samples)")
                if self.ota_channel.noise._noise_bank:
                    self.log(f"  Noise bank: {len(self.ota_channel.noise._noise_bank)} samples")
            else:
                self.log(f"WARNING: Empirical data not found: {empirical_data_dir}, "
                         "using synthetic channel")
            self.log(f"OTA channel: EoT samples={n_eot_samples}, "
                     f"severity={channel_severity}, curriculum={channel_curriculum}")

        if no_channel:
            self.log("Channel DISABLED (--no-channel)")
        self.log(f"Warmup ratio: {warmup_ratio}")

        # Optional sub-band PGD: project delta into [0, lowpass_hz] after every
        # step. Used as the falsification control for the spectral-confinement
        # mechanistic story (forces waveform attack into the Bark 0-3 region
        # where the latent attack also concentrates).
        self.lowpass_hz = float(lowpass_hz or 0.0)
        self.spectral_cutoff_hz = float(spectral_cutoff_hz or 0.0)
        if self.lowpass_hz > 0.0 and self.spectral_cutoff_hz > 0.0:
            raise ValueError(
                "lowpass_hz and spectral_cutoff_hz are mutually exclusive"
            )
        self._lowpass_sos = None
        if self.lowpass_hz > 0.0:
            from scipy.signal import butter
            nyq = TARGET_SAMPLE_RATE / 2.0
            self._lowpass_sos = butter(
                4, self.lowpass_hz / nyq, btype="low", output="sos"
            )
            self.log(f"Lowpass-projected PGD: delta clamped to <= {self.lowpass_hz:.0f} Hz "
                     f"(4-pole Butterworth) at {TARGET_SAMPLE_RATE} Hz sample rate")
        elif self.spectral_cutoff_hz > 0.0:
            nyq = TARGET_SAMPLE_RATE / 2.0
            if self.spectral_cutoff_hz >= nyq:
                raise ValueError(
                    f"spectral_cutoff_hz must be below Nyquist ({nyq:g} Hz)"
                )
            self.log(
                "FFT-parameterized PGD: optimizing an exact real-Fourier "
                f"subspace <= {self.spectral_cutoff_hz:.0f} Hz at "
                f"{TARGET_SAMPLE_RATE} Hz"
            )

    def _project_lowpass(self, delta_data: torch.Tensor) -> torch.Tensor:
        """Project delta into the configured low-pass band. Runs scipy SOS
        filtfilt offline (CPU-side) and re-uploads the result. Called only
        on .data, so the autograd graph is not touched."""
        if self._lowpass_sos is None:
            return delta_data
        from scipy.signal import sosfilt
        device = delta_data.device
        dtype = delta_data.dtype
        arr = delta_data.detach().cpu().to(torch.float32).numpy()
        flat = arr.reshape(-1, arr.shape[-1])
        out = np.empty_like(flat)
        for i in range(flat.shape[0]):
            out[i] = sosfilt(self._lowpass_sos, flat[i])
        out = out.reshape(arr.shape)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    def _spectral_delta(self, delta_param: torch.Tensor) -> torch.Tensor:
        """Map the optimization variable into an exact low-frequency subspace.

        Unlike the legacy causal Butterworth update, this is an orthogonal,
        idempotent Fourier projection and remains in the autograd graph. Thus
        optimization happens *through* the frequency-domain parameterization
        instead of repeatedly filtering and attenuating the saved waveform.
        """
        if self.spectral_cutoff_hz <= 0.0:
            return delta_param
        original_dtype = delta_param.dtype
        work = delta_param.to(torch.float32)
        n = work.shape[-1]
        spectrum = torch.fft.rfft(work, dim=-1, norm="ortho")
        # Decide bin membership exactly. torch.fft.rfftfreq returns float32, so
        # a bin sitting exactly on the cutoff rounds differently there than in
        # the float64 audit, which would report a nonexistent ~2e-5 of
        # out-of-band energy. k * fs <= cutoff * n is exact for integer rates.
        k = torch.arange(n // 2 + 1, device=work.device, dtype=torch.float64)
        keep = k * float(TARGET_SAMPLE_RATE) <= float(self.spectral_cutoff_hz) * float(n)
        mask = keep.to(spectrum.dtype)
        projected = torch.fft.irfft(
            spectrum * mask,
            n=n,
            dim=-1,
            norm="ortho",
        )
        return projected.to(original_dtype)

    def _project_spectral_feasible(
        self,
        delta_param: torch.Tensor,
        original_16k: torch.Tensor,
    ) -> torch.Tensor:
        """Project to the spectral subspace and waveform feasibility set.

        A single non-negative scale is applied to the band-limited direction.
        This preserves exact Fourier support while satisfying both the
        L-infinity perturbation budget and the valid-audio range. In
        particular, there is no pointwise clamp after the spectral projection:
        pointwise clipping would create out-of-band harmonics.
        """
        projected = self._spectral_delta(delta_param)
        peak = projected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(self.eps / peak, max=1.0)

        # Largest scalar that keeps original + scalar * projected in [-1, 1].
        inf = torch.full_like(projected, float("inf"))
        headroom = torch.where(
            projected > 0,
            (1.0 - original_16k) / projected.clamp_min(1e-12),
            torch.where(
                projected < 0,
                (-1.0 - original_16k) / projected.clamp_max(-1e-12),
                inf,
            ),
        )
        range_scale = headroom.amin(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
        # Record which of the two constraints actually bound. A realized peak
        # below eps does NOT by itself mean the range constraint bound: the
        # epsilon scale is clamped at 1.0 (projection onto the ball only shrinks,
        # never grows), so an iterate that has not yet reached the budget also
        # lands below eps. The range constraint bound only when it was the
        # smaller of the two scales.
        self._last_eps_scale = float(scale.min())
        self._last_range_scale = float(range_scale.min())
        self._last_range_bound = bool(float(range_scale.min()) < float(scale.min()))
        scale = torch.minimum(scale, range_scale)
        return projected * scale

    def _get_severity(self, step: int, total_steps: int) -> float:
        if not self.channel_curriculum:
            return self.channel_severity
        warmup_steps = int(total_steps * self.warmup_ratio)
        harden_steps = total_steps - warmup_steps
        if harden_steps <= 0:
            return self.channel_severity
        progress = min((step - warmup_steps) / harden_steps, 1.0)
        return 0.1 + progress * (self.channel_severity - 0.1)

    def _apply_channel(self, audio_16k: torch.Tensor,
                       step: int = 0, total_steps: int = 0) -> torch.Tensor:
        """
        Apply the active training-time channel to 16kHz audio.

        Both modes upsample 16k -> 24k, run through the channel, downsample
        24k -> 16k, and pad/truncate back to the input length.

        - channel_mode="ota":           OTA channel (IR + noise).
        - channel_mode="multi_bitrate": DifferentiableOpusProxy at a random
                                        Opus bitrate per call (STE backward).
        """
        orig_dim = audio_16k.dim()
        audio_24k = torchaudio.functional.resample(
            audio_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE
        )
        # Normalize to [B, 1, T] for the channel modules.
        if audio_24k.dim() == 1:
            audio_24k = audio_24k.unsqueeze(0).unsqueeze(0)
        elif audio_24k.dim() == 2:
            audio_24k = audio_24k.unsqueeze(0)

        severity = self._get_severity(step, total_steps)

        if self.channel_mode == "multi_bitrate":
            # EoT across Opus bitrates: a single pass samples a random bitrate;
            # averaging multiple samples improves gradient stability.
            if self.n_eot_samples > 1:
                acc = torch.zeros_like(audio_24k)
                for _ in range(self.n_eot_samples):
                    acc = acc + self.opus_proxy(audio_24k, severity=severity)
                audio_ch_24k = acc / self.n_eot_samples
            else:
                audio_ch_24k = self.opus_proxy(audio_24k, severity=severity)
        else:
            if self.n_eot_samples > 1:
                acc = torch.zeros_like(audio_24k)
                for _ in range(self.n_eot_samples):
                    acc = acc + self.ota_channel(audio_24k, severity=severity)
                audio_ch_24k = acc / self.n_eot_samples
            else:
                audio_ch_24k = self.ota_channel(audio_24k, severity=severity)

        # [B, 1, T_24] -> resample -> [B, 1, T_16]
        audio_ch_16k = torchaudio.functional.resample(
            audio_ch_24k, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        # Restore the caller's rank so the attack loop's arithmetic holds.
        while audio_ch_16k.dim() > orig_dim:
            audio_ch_16k = audio_ch_16k.squeeze(0)

        T = audio_16k.shape[-1]
        if audio_ch_16k.shape[-1] > T:
            audio_ch_16k = audio_ch_16k[..., :T]
        elif audio_ch_16k.shape[-1] < T:
            audio_ch_16k = F.pad(audio_ch_16k, (0, T - audio_ch_16k.shape[-1]))

        return audio_ch_16k

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
        Run airgap-robust waveform-domain adversarial attack.

        Stage 1 (warmup): optimize delta directly (no channel)
        Stage 2 (harden): optimize delta through OTA channel
        """
        # If no channel, fall back to vanilla waveform attack
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

        # Resample from 24kHz to 16kHz
        music_16k = torchaudio.functional.resample(
            music_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )  # [1, T_16k]

        # Get original model output
        with torch.no_grad():
            original_output = self.target_model.generate(music_16k, prompt=prompt)

        self.log(f"Original output: {original_output}")

        if untargeted:
            target_text = original_output
            self.log("UNTARGETED mode: maximizing loss on original output")
        self.log(f"Target: {target_text}")
        if prompt:
            self.log(f"Prompt: {prompt}")

        # Initialize perturbation. Default: zeros (deterministic, matches the
        # headline bundles). When self.rand_init is set (multi-seed study), draw
        # a random start uniformly from the L-inf eps ball so that different RNG
        # seeds explore different initializations (standard PGD multi-seed).
        original_16k = music_16k.detach()
        if getattr(self, "rand_init", False):
            delta = torch.empty_like(original_16k).uniform_(-self.eps, self.eps)
            if self.spectral_cutoff_hz > 0.0:
                delta = self._project_spectral_feasible(delta, original_16k)
            delta.requires_grad_(True)
        else:
            delta = torch.zeros_like(original_16k, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        # Stage boundaries
        warmup_steps = int(steps * self.warmup_ratio)

        # Tracking
        history = {
            "loss": [], "behavior_loss": [], "perceptual_loss": [],
            "outputs": [], "steps": [], "stage": [],
        }
        best_loss = float('inf')
        best_delta = delta.data.clone()
        success = False

        self.log(f"\nStarting robust waveform attack: eps={self.eps}, "
                 f"alpha={self.alpha}, steps={steps}, channel_mode={self.channel_mode}")
        self.log(f"Stage 1 (steps 1-{warmup_steps}): vanilla warmup")
        if self.channel_mode == "multi_bitrate":
            stage2_label = (f"multi-bitrate Opus EoT "
                            f"({self._multi_bitrate_kbps} kbps, x{self.n_eot_samples})")
        else:
            stage2_label = f"OTA channel (EoT x{self.n_eot_samples})"
        self.log(f"Stage 2 (steps {warmup_steps + 1}-{steps}): {stage2_label}")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        self.log(f"Waveform shape: {original_16k.shape} at {TARGET_SAMPLE_RATE}Hz")
        self.log("-" * 60)

        for step in range(steps):
            optimizer.zero_grad()

            use_channel = step >= warmup_steps
            if use_channel and self.alternate_after_warmup:
                # Match RobustLatentCodecAttacker exactly: odd zero-based steps
                # are codec steps, even steps are direct updates.
                use_channel = (step % 2 == 1)

            # Apply perturbation. In spectral mode, ``delta`` is the optimizer
            # variable and the realized waveform perturbation is its exact
            # Fourier-subspace image.
            realized_delta = self._spectral_delta(delta)
            audio_adv_16k = original_16k + realized_delta

            # Stage 2: apply OTA channel
            if use_channel:
                audio_for_loss = self._apply_channel(
                    audio_adv_16k, step=step, total_steps=steps)
            else:
                audio_for_loss = audio_adv_16k

            # Behavior loss
            behavior_loss = self.target_model.compute_loss(
                audio_for_loss, target_text, prompt=prompt
            )

            total_loss = behavior_loss
            if untargeted:
                total_loss = -total_loss

            # Perceptual loss (on pre-channel audio to preserve quality)
            p_loss_val = 0.0
            if self.perceptual_weight > 0:
                if use_multi_scale_loss:
                    p_loss = multi_scale_mel_distance(
                        audio_adv_16k, original_16k.detach(),
                        sample_rate=TARGET_SAMPLE_RATE,
                    )
                else:
                    p_loss = mel_distance(
                        audio_adv_16k, original_16k.detach(),
                        sample_rate=TARGET_SAMPLE_RATE,
                    )
                total_loss = total_loss + self.perceptual_weight * p_loss
                p_loss_val = p_loss.item()

            total_loss.backward()
            optimizer.step()

            # Project to L-inf epsilon ball + valid range. Lowpass is applied
            # LAST so the saved perturbation is exactly band-limited; the
            # post-clamp L-inf may drift slightly above eps in HF-empty
            # configurations, which is the desired tradeoff for the
            # spectral-confinement control experiment.
            with torch.no_grad():
                if self.spectral_cutoff_hz > 0.0:
                    delta.data = self._project_spectral_feasible(
                        delta.data, original_16k
                    )
                else:
                    delta.data = torch.clamp(
                        delta.data, -self.eps, self.eps
                    )
                    delta.data = (
                        torch.clamp(
                            original_16k + delta.data, -1.0, 1.0
                        )
                        - original_16k
                    )
                    if self._lowpass_sos is not None:
                        delta.data = self._project_lowpass(delta.data)

            # Track
            b_loss_val = behavior_loss.item()
            loss_val = total_loss.item()

            history["loss"].append(loss_val)
            history["behavior_loss"].append(b_loss_val)
            history["perceptual_loss"].append(p_loss_val)
            history["stage"].append("ota" if use_channel else "vanilla")

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = self._spectral_delta(delta.data).detach().clone()

            # Progress logging
            if (step + 1) % 10 == 0:
                stage = "S2-OTA" if use_channel else "S1-vanilla"
                self.log(
                    f"Step {step + 1:4d}/{steps} [{stage}] | "
                    f"Loss: {loss_val:.4f} | "
                    f"Behavior: {b_loss_val:.4f} | "
                    f"Perceptual: {p_loss_val:.4f}"
                )

            # Periodic eval: test BOTH direct and OTA paths
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    audio_test_16k = (
                        original_16k + self._spectral_delta(delta)
                    )

                    # Direct eval
                    output_direct = self.target_model.generate(
                        audio_test_16k, prompt=prompt
                    )

                    # OTA eval
                    audio_ota_16k = self._apply_channel(
                        audio_test_16k, step=step, total_steps=steps)
                    output_ota = self.target_model.generate(
                        audio_ota_16k, prompt=prompt
                    )

                history["outputs"].append({
                    "direct": output_direct,
                    "ota": output_ota,
                })
                history["steps"].append(step + 1)

                direct_ok = target_text.lower() in output_direct.lower()
                ota_ok = target_text.lower() in output_ota.lower()
                self.log(f"  -> Direct: {'OK' if direct_ok else 'FAIL'} | "
                         f"{output_direct[:80]}")
                self.log(f"  -> OTA:    {'OK' if ota_ok else 'FAIL'} | "
                         f"{output_ota[:80]}")

                if not untargeted and ota_ok:
                    self.log(f"  -> OTA TARGET ACHIEVED at step {step + 1}!")
                    success = True

        # Final evaluation with last delta
        with torch.no_grad():
            final_delta = self._spectral_delta(delta.data)
            audio_final_16k = original_16k + final_delta

            # Direct eval
            final_output = self.target_model.generate(
                audio_final_16k, prompt=prompt
            )

            # OTA eval
            audio_ota_final_16k = self._apply_channel(
                audio_final_16k, step=steps - 1, total_steps=steps)
            ota_output = self.target_model.generate(
                audio_ota_final_16k, prompt=prompt
            )

        # Success = target achieved through OTA (the harder condition)
        success = False
        if untargeted:
            final_wer = compute_wer(target_text, ota_output)
            if final_wer > UNTARGETED_WER_THRESHOLD:
                success = True
            self.log(f"Final OTA WER: {final_wer:.3f}")
        else:
            if target_text.lower() in ota_output.lower():
                success = True

        # SNR
        snr_db = self._compute_snr(
            original_16k.squeeze().detach(),
            audio_final_16k.squeeze().detach(),
        )

        # Expose the 16 kHz perturbation and carrier for downstream spectral
        # verification. The 24 kHz save resamples and requantizes, so band
        # energy measured on the saved file cannot separate the optimizer's own
        # Fourier support from serialization leakage; measuring both is the
        # point of the >=4 kHz control, so this is always retained.
        self._last_delta_16k = final_delta.detach().cpu().clone()
        self._last_original_16k = original_16k.detach().cpu().clone()

        # Upsample to 24kHz for saving
        audio_final_24k = torchaudio.functional.resample(
            audio_final_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE
        )
        original_24k = torchaudio.functional.resample(
            original_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE
        )

        duration = time.time() - start_time

        self.log("\n" + "=" * 60)
        self.log("Airgap-Robust Waveform Attack Complete")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Direct Output: {final_output[:100]}")
        self.log(f"OTA Output: {ota_output[:100]}")
        self.log(f"OTA Success: {success}")
        self.log(f"Duration: {duration:.1f}s")
        self.log("=" * 60)

        result = AttackResult(
            original_wav=original_24k.detach().cpu(),
            adversarial_wav=audio_final_24k.detach().cpu(),
            original_latents=None,
            adversarial_latents=None,
            latent_perturbation=None,
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            latent_snr_db=0.0,
            history=history,
            music_name=music_name,
            attack_duration_s=duration,
        )
        # Store OTA output for benchmark to access
        result.ota_output = ota_output
        return result


def main():
    """CLI for single robust waveform attack."""
    import argparse
    from music_carrier import load_music_by_name
    from config import MUSIC_FILES, DEFAULT_MUSIC, PROMPT_MODES, DEFAULT_PROMPT_MODE

    parser = argparse.ArgumentParser(
        description="Airgap-robust waveform-domain attack on Audio LLMs"
    )
    parser.add_argument("--music", type=str, default=DEFAULT_MUSIC)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=WAVEFORM_EPS)
    parser.add_argument("--alpha", type=float, default=WAVEFORM_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.5)
    parser.add_argument("--no-channel", action="store_true")
    parser.add_argument("--n-eot-samples", type=int, default=4)
    parser.add_argument("--channel-severity", type=float, default=1.0)
    parser.add_argument("--channel-curriculum", action="store_true")
    parser.add_argument("--empirical-data-dir", type=str, default=None)

    args = parser.parse_args()
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]
    music_wav = load_music_by_name(args.music)

    attacker = RobustWaveformAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        warmup_ratio=args.warmup_ratio,
        no_channel=args.no_channel,
        n_eot_samples=args.n_eot_samples,
        channel_severity=args.channel_severity,
        channel_curriculum=args.channel_curriculum,
        empirical_data_dir=args.empirical_data_dir or _EMPIRICAL_DATA_DIR,
    )

    result = attacker.attack(
        music_wav,
        target_text=args.target,
        steps=args.steps,
        music_name=args.music,
        prompt=prompt_text,
    )

    from config import RESULTS_DIR
    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "robust_waveform_single")
    attacker.save_adversarial_wav(result, output_dir, "robust_waveform_test")

    print(f"\nOTA Success: {result.success}")
    print(f"Direct Output: {result.adversarial_output}")
    print(f"OTA Output: {getattr(result, 'ota_output', 'N/A')}")
    print(f"SNR: {result.snr_db:.2f} dB")


if __name__ == "__main__":
    main()
