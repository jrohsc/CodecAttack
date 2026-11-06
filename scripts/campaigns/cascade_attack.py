"""
Cascade attack: latent-space PGD (codec robustness) + waveform PGD (OTA robustness).

Phase 1: Latent-space PGD on EnCodec latents
    → produces adversarial audio that survives codec compression (Opus, AAC)

Phase 2: Waveform-space PGD with OTA augmentation (band-pass + RIR + noise + EoT)
    → refines the Phase 1 output to also survive the physical OTA channel
    → small waveform eps to avoid undoing the codec robustness from Phase 1

Gradient flow:
    Phase 1: delta_latent → EnCodec.decoder → audio_24k → resample → model.loss
    Phase 2: delta_waveform → audio_24k + delta → OTA channel → resample → model.loss

Usage:
    # Quick test
    python cascade_attack.py --music jazz_1 --target "Turn left at the next intersection"

    # Full config
    python cascade_attack.py --music jazz_1 --target "Turn left" \
        --latent-eps 10.0 --latent-steps 300 \
        --waveform-eps 0.05 --waveform-steps 200 \
        --n-eot-samples 4 --rir-dir data/rirs
"""

import os
import sys
import time
import argparse
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as AF
import numpy as np

from config import (
    WHISPER_INJECT_ROOT, TARGET_MODEL, MUSIC_FILES, DEFAULT_MUSIC,
    ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, OPUS_TEST_BITRATES,
    PROMPT_MODES, DEFAULT_PROMPT_MODE,
    RESULTS_DIR, compute_wer,
)
from latent_attack import LatentCodecAttacker, AttackResult
from robust_latent_attack import (
    RobustLatentCodecAttacker,
    DifferentiableBandPassFilter,
    DifferentiableIRConv,
)
from channel_augmentation import AdditiveColoredNoise
from music_carrier import load_music_by_name, mel_distance, multi_scale_mel_distance

sys.path.insert(0, WHISPER_INJECT_ROOT)

# Defaults for Phase 2 (waveform refinement)
WAVEFORM_EPS_DEFAULT = 0.05
WAVEFORM_ALPHA_DEFAULT = 1e-4
WAVEFORM_STEPS_DEFAULT = 200


class CascadeAttacker:
    """
    Two-phase cascade: latent PGD → waveform PGD with OTA augmentation.

    Phase 1 uses LatentCodecAttacker (or RobustLatentCodecAttacker) to produce
    codec-robust adversarial audio.

    Phase 2 takes that audio as the starting point and applies waveform-level
    PGD with band-pass filtering, diverse RIR convolution, and Gaussian noise
    to add OTA robustness on top.
    """

    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        encodec_bandwidth: float = ENCODEC_BANDWIDTH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        # Phase 1 (latent)
        latent_eps: float = LATENT_EPS,
        latent_alpha: float = LATENT_ALPHA,
        latent_steps: int = ATTACK_STEPS,
        latent_perceptual_weight: float = PERCEPTUAL_WEIGHT,
        # Phase 2 (waveform OTA)
        waveform_eps: float = WAVEFORM_EPS_DEFAULT,
        waveform_alpha: float = WAVEFORM_ALPHA_DEFAULT,
        waveform_steps: int = WAVEFORM_STEPS_DEFAULT,
        waveform_perceptual_weight: float = 0.0,
        # OTA channel config
        rir_dir: str = None,
        bandpass_low_hz: float = 1000.0,
        bandpass_high_hz: float = 4000.0,
        noise_snr_db: float = 20.0,
        noise_snr_std: float = 5.0,
        n_eot_samples: int = 4,
        warmup_ratio: float = 0.3,
    ):
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        # Phase 1 config
        self.latent_steps = latent_steps
        self.latent_eps = latent_eps

        # Phase 2 config
        self.waveform_eps = waveform_eps
        self.waveform_alpha = waveform_alpha
        self.waveform_steps = waveform_steps
        self.waveform_perceptual_weight = waveform_perceptual_weight
        self.n_eot_samples = n_eot_samples
        self.warmup_ratio = warmup_ratio

        # Phase 1 attacker (latent-space, no channel hardening needed)
        self.log("Initializing Phase 1: Latent-space attacker...")
        self.latent_attacker = LatentCodecAttacker(
            target_model=target_model,
            encodec_bandwidth=encodec_bandwidth,
            eps=latent_eps,
            alpha=latent_alpha,
            perceptual_weight=latent_perceptual_weight,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )

        # Phase 2 OTA components (reuse from robust_latent_attack)
        self.log("Initializing Phase 2: OTA channel components...")

        # Band-pass filter
        self.bandpass = None
        if bandpass_low_hz > 0 and bandpass_high_hz > bandpass_low_hz:
            self.bandpass = DifferentiableBandPassFilter(
                low_hz=bandpass_low_hz,
                high_hz=bandpass_high_hz,
                sample_rate=ENCODEC_SAMPLE_RATE,
                device=device,
            ).to(device)
            self.log(f"  Band-pass: {bandpass_low_hz:.0f}-{bandpass_high_hz:.0f} Hz")

        # Diverse RIR convolution
        self.ir_conv = None
        if rir_dir and os.path.isdir(rir_dir):
            self.ir_conv = DifferentiableIRConv(
                ir_dir=rir_dir,
                sample_rate=ENCODEC_SAMPLE_RATE,
                device=device,
            )
            self.log(f"  RIRs: {rir_dir} ({self.ir_conv._n_irs} files)")
        else:
            self.log(f"  WARNING: No RIR directory provided or found. OTA sim will be limited.")

        # Gaussian noise
        self.noise = AdditiveColoredNoise(
            sample_rate=ENCODEC_SAMPLE_RATE,
            snr_db_mean=noise_snr_db,
            snr_db_std=noise_snr_std,
        ).to(device)
        self._noise_snr_db = noise_snr_db
        self._noise_snr_std = noise_snr_std
        self.log(f"  Noise: SNR={noise_snr_db:.0f}±{noise_snr_std:.0f} dB")
        self.log(f"  EoT samples: {n_eot_samples}")

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _apply_ota_channel(self, audio_24k: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """Apply OTA channel: RIR + noise, EoT averaged."""
        self.noise.train()  # enable SNR jittering
        acc = torch.zeros_like(audio_24k)
        for _ in range(self.n_eot_samples):
            x = audio_24k
            # RIR convolution (random selection)
            if self.ir_conv is not None:
                x = self.ir_conv(x)
            # Additive noise (SNR jittering handled internally)
            x = self.noise(x, severity=severity)
            acc = acc + x
        return acc / self.n_eot_samples

    def _compute_snr(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        noise = adversarial - original
        signal_power = (original ** 2).mean()
        noise_power = (noise ** 2).mean()
        if noise_power < 1e-12:
            return 100.0
        return 10 * torch.log10(signal_power / noise_power).item()

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps_phase1: int = None,
        steps_phase2: int = None,
        check_every: int = CHECK_EVERY,
        music_name: str = "",
        prompt: str = None,
    ) -> AttackResult:
        """
        Run cascade attack.

        Phase 1: Latent PGD for codec robustness
        Phase 2: Waveform PGD with OTA augmentation for physical robustness
        """
        steps_phase1 = steps_phase1 or self.latent_steps
        steps_phase2 = steps_phase2 or self.waveform_steps
        start_time = time.time()

        # ================================================================
        # Phase 1: Latent-space attack (codec robustness)
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("PHASE 1: Latent-space PGD (codec robustness)")
        self.log(f"  eps={self.latent_eps}, steps={steps_phase1}")
        self.log("=" * 60)

        phase1_result = self.latent_attacker.attack(
            music_wav,
            target_text=target_text,
            steps=steps_phase1,
            check_every=check_every,
            music_name=music_name,
            prompt=prompt,
        )

        self.log(f"\nPhase 1 complete: success={phase1_result.success}, "
                 f"SNR={phase1_result.snr_db:.1f}dB")
        self.log(f"  Output: {phase1_result.adversarial_output[:100]}")

        # ================================================================
        # Phase 2: Waveform PGD with OTA augmentation
        # ================================================================
        self.log("\n" + "=" * 60)
        self.log("PHASE 2: Waveform PGD with OTA augmentation")
        self.log(f"  eps={self.waveform_eps}, steps={steps_phase2}, "
                 f"EoT x{self.n_eot_samples}")
        self.log("=" * 60)

        # Use Phase 1 adversarial audio as the starting point (24kHz)
        adv_audio_24k = phase1_result.adversarial_wav.to(self.device)  # [1, T] at 24kHz
        if adv_audio_24k.dim() == 2:
            adv_audio_24k = adv_audio_24k.unsqueeze(0)  # [1, 1, T]

        original_24k = adv_audio_24k.detach().clone()

        # Initialize waveform perturbation (in 24kHz space to match OTA channel)
        delta = torch.zeros_like(original_24k, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.waveform_alpha)

        # Warmup: optimize without channel first
        warmup_steps = int(steps_phase2 * self.warmup_ratio)

        target_model = self.latent_attacker.target_model
        best_loss = float('inf')
        best_delta = delta.data.clone()
        p2_success = False

        self.log(f"  Warmup: steps 1-{warmup_steps} (no OTA channel)")
        self.log(f"  Harden: steps {warmup_steps + 1}-{steps_phase2} (with OTA channel)")
        self.log("-" * 60)

        for step in range(steps_phase2):
            optimizer.zero_grad()

            use_channel = step >= warmup_steps

            # Apply perturbation
            audio_adv = original_24k + delta

            # Band-pass filter the perturbation (Yakura technique)
            if self.bandpass is not None:
                perturbation = audio_adv - original_24k
                perturbation_filtered = self.bandpass(perturbation)
                audio_adv = original_24k + perturbation_filtered

            # Apply OTA channel in hardening phase
            if use_channel:
                audio_for_loss = self._apply_ota_channel(audio_adv)
            else:
                audio_for_loss = audio_adv

            # Resample to 16kHz for model
            audio_16k = AF.resample(
                audio_for_loss.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # Behavior loss
            behavior_loss = target_model.compute_loss(audio_16k, target_text, prompt=prompt)
            total_loss = behavior_loss

            # Optional perceptual loss (on pre-channel audio)
            p_loss_val = 0.0
            if self.waveform_perceptual_weight > 0:
                audio_pre_ch_16k = AF.resample(
                    audio_adv.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                original_16k = AF.resample(
                    original_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                p_loss = mel_distance(audio_pre_ch_16k, original_16k.detach(),
                                      sample_rate=TARGET_SAMPLE_RATE)
                total_loss = total_loss + self.waveform_perceptual_weight * p_loss
                p_loss_val = p_loss.item()

            total_loss.backward()
            optimizer.step()

            # Project to L-inf ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.waveform_eps, self.waveform_eps)
                delta.data = torch.clamp(original_24k + delta.data, -1.0, 1.0) - original_24k

            loss_val = total_loss.item()
            b_loss_val = behavior_loss.item()

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            # Progress logging
            if (step + 1) % 10 == 0:
                stage = "S2-OTA" if use_channel else "S2-warm"
                self.log(f"  Step {step + 1:4d}/{steps_phase2} [{stage}] | "
                         f"Loss: {loss_val:.4f} | Behavior: {b_loss_val:.4f} | "
                         f"Perceptual: {p_loss_val:.4f}")

            # Periodic eval
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    audio_test = original_24k + delta
                    if self.bandpass is not None:
                        pert = audio_test - original_24k
                        audio_test = original_24k + self.bandpass(pert)

                    # Direct eval
                    test_16k = AF.resample(
                        audio_test.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output_direct = target_model.generate(test_16k, prompt=prompt)

                    # OTA eval
                    test_ota = self._apply_ota_channel(audio_test)
                    test_ota_16k = AF.resample(
                        test_ota.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output_ota = target_model.generate(test_ota_16k, prompt=prompt)

                direct_ok = target_text.lower() in output_direct.lower()
                ota_ok = target_text.lower() in output_ota.lower()
                self.log(f"    Direct: {'OK' if direct_ok else 'FAIL'} | {output_direct[:80]}")
                self.log(f"    OTA:    {'OK' if ota_ok else 'FAIL'} | {output_ota[:80]}")

                if ota_ok:
                    self.log(f"    OTA TARGET ACHIEVED at step {step + 1}!")
                    p2_success = True

        # ================================================================
        # Final evaluation
        # ================================================================
        with torch.no_grad():
            audio_final = original_24k + best_delta
            if self.bandpass is not None:
                pert = audio_final - original_24k
                audio_final = original_24k + self.bandpass(pert)

            # Direct
            final_16k = AF.resample(
                audio_final.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            output_direct = target_model.generate(final_16k, prompt=prompt)

            # OTA
            final_ota = self._apply_ota_channel(audio_final)
            final_ota_16k = AF.resample(
                final_ota.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            output_ota = target_model.generate(final_ota_16k, prompt=prompt)

        # Compute SNR relative to original clean music
        original_music_24k = music_wav.to(self.device).squeeze()
        final_squeezed = audio_final.squeeze().detach()
        # Match lengths
        min_len = min(original_music_24k.shape[-1], final_squeezed.shape[-1])
        snr_db = self._compute_snr(
            original_music_24k[..., :min_len], final_squeezed[..., :min_len]
        )

        duration = time.time() - start_time

        direct_ok = target_text.lower() in output_direct.lower()
        ota_ok = target_text.lower() in output_ota.lower()

        self.log("\n" + "=" * 60)
        self.log("CASCADE ATTACK COMPLETE")
        self.log(f"  Phase 1 (latent):   success={phase1_result.success}")
        self.log(f"  Phase 2 (waveform): direct={'OK' if direct_ok else 'FAIL'}, "
                 f"OTA={'OK' if ota_ok else 'FAIL'}")
        self.log(f"  Direct output: {output_direct[:100]}")
        self.log(f"  OTA output:    {output_ota[:100]}")
        self.log(f"  Total SNR: {snr_db:.1f} dB")
        self.log(f"  Duration: {duration:.1f}s")
        self.log("=" * 60)

        # Build result
        result = AttackResult(
            original_wav=music_wav.cpu(),
            adversarial_wav=audio_final.squeeze(0).detach().cpu(),
            original_latents=phase1_result.original_latents,
            adversarial_latents=phase1_result.adversarial_latents,
            latent_perturbation=phase1_result.latent_perturbation,
            original_output=phase1_result.original_output,
            adversarial_output=output_direct,
            target_text=target_text,
            success=ota_ok,
            final_loss=best_loss,
            steps_taken=steps_phase1 + steps_phase2,
            snr_db=snr_db,
            latent_snr_db=phase1_result.latent_snr_db,
            history={
                "phase1": phase1_result.history,
                "phase2_best_loss": best_loss,
            },
            music_name=music_name,
            attack_duration_s=duration,
        )
        result.ota_output = output_ota
        result.phase1_result = phase1_result
        return result


def main():
    """CLI for cascade attack."""
    parser = argparse.ArgumentParser(description="Cascade attack: latent PGD + waveform OTA PGD")

    # Music / target
    parser.add_argument("--music", type=str, default=DEFAULT_MUSIC)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)

    # Phase 1 (latent)
    parser.add_argument("--latent-eps", type=float, default=LATENT_EPS)
    parser.add_argument("--latent-alpha", type=float, default=LATENT_ALPHA)
    parser.add_argument("--latent-steps", type=int, default=300)
    parser.add_argument("--latent-perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)

    # Phase 2 (waveform OTA)
    parser.add_argument("--waveform-eps", type=float, default=WAVEFORM_EPS_DEFAULT)
    parser.add_argument("--waveform-alpha", type=float, default=WAVEFORM_ALPHA_DEFAULT)
    parser.add_argument("--waveform-steps", type=int, default=WAVEFORM_STEPS_DEFAULT)
    parser.add_argument("--waveform-perceptual-weight", type=float, default=0.0)

    # OTA channel
    parser.add_argument("--rir-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "rirs"))
    parser.add_argument("--bandpass-low-hz", type=float, default=1000.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=4000.0)
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.add_argument("--noise-snr-std", type=float, default=5.0)
    parser.add_argument("--n-eot-samples", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.3)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--check-every", type=int, default=50)

    args = parser.parse_args()
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    # Generate RIRs if needed
    rir_dir = args.rir_dir
    if not os.path.isdir(rir_dir):
        print(f"RIR directory not found: {rir_dir}")
        print("Run: python generate_rirs.py --output-dir data/rirs --n-rirs 615")
        print("Or: bash run_yakura_ota.sh  (generates RIRs automatically)")
        sys.exit(1)

    music_wav = load_music_by_name(args.music)

    attacker = CascadeAttacker(
        target_model=args.target_model,
        latent_eps=args.latent_eps,
        latent_alpha=args.latent_alpha,
        latent_steps=args.latent_steps,
        latent_perceptual_weight=args.latent_perceptual_weight,
        waveform_eps=args.waveform_eps,
        waveform_alpha=args.waveform_alpha,
        waveform_steps=args.waveform_steps,
        waveform_perceptual_weight=args.waveform_perceptual_weight,
        rir_dir=rir_dir,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        noise_snr_db=args.noise_snr_db,
        noise_snr_std=args.noise_snr_std,
        n_eot_samples=args.n_eot_samples,
        warmup_ratio=args.warmup_ratio,
    )

    result = attacker.attack(
        music_wav,
        target_text=args.target,
        steps_phase1=args.latent_steps,
        steps_phase2=args.waveform_steps,
        check_every=args.check_every,
        music_name=args.music,
        prompt=prompt_text,
    )

    # Save results
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(RESULTS_DIR), "results_cascade",
        f"cascade_eps{args.latent_eps}_weps{args.waveform_eps}_{args.music}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save audio
    import soundfile as sf
    adv_np = result.adversarial_wav.squeeze().cpu().numpy()
    sf.write(os.path.join(output_dir, "adversarial_24k.wav"), adv_np, ENCODEC_SAMPLE_RATE)

    # Save 16k version
    adv_16k = AF.resample(result.adversarial_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    sf.write(os.path.join(output_dir, "adversarial_16k.wav"),
             adv_16k.squeeze().cpu().numpy(), TARGET_SAMPLE_RATE)

    # Save summary
    import json
    summary = {
        "music": args.music,
        "target": args.target,
        "phase1_success": result.phase1_result.success,
        "phase1_snr_db": result.phase1_result.snr_db,
        "phase1_output": result.phase1_result.adversarial_output,
        "phase2_direct_output": result.adversarial_output,
        "phase2_ota_output": result.ota_output,
        "direct_success": args.target.lower() in result.adversarial_output.lower(),
        "ota_success": result.success,
        "total_snr_db": result.snr_db,
        "total_steps": result.steps_taken,
        "duration_s": result.attack_duration_s,
        "config": {
            "latent_eps": args.latent_eps,
            "latent_steps": args.latent_steps,
            "waveform_eps": args.waveform_eps,
            "waveform_steps": args.waveform_steps,
            "bandpass_hz": [args.bandpass_low_hz, args.bandpass_high_hz],
            "noise_snr_db": args.noise_snr_db,
            "n_eot_samples": args.n_eot_samples,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Phase 1 success: {result.phase1_result.success}")
    print(f"Direct success:  {summary['direct_success']}")
    print(f"OTA success:     {result.success}")
    print(f"SNR: {result.snr_db:.1f} dB")


if __name__ == "__main__":
    main()
