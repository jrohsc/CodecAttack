"""
Joint latent + waveform attack for simultaneous codec and OTA robustness.

Optimizes TWO perturbations simultaneously:
  - delta_latent: perturbation in EnCodec latent space (codec robustness)
  - delta_waveform: perturbation in waveform space with band-pass (OTA robustness)

Gradient flow:
    z_orig + delta_latent
    → EnCodec.decoder(z_adv) → audio_24k
    → audio_24k + bandpass(delta_waveform)
    → OTA channel (RIR + noise, EoT)
    → resample 24k → 16k
    → model.compute_loss(audio_16k, target_text)
    → loss.backward() → gradients to BOTH delta_latent AND delta_waveform

Two stages:
  Stage 1 (warmup): no OTA channel — both deltas learn the adversarial target
  Stage 2 (harden): OTA channel on — latent delta maintains codec path,
                     waveform delta shapes frequencies for air survival

Usage:
    # Quick test
    python joint_attack.py --music jazz_1 --target "Turn left at the next intersection"

    # Full config
    python joint_attack.py --music jazz_1 --target "Turn left" \
        --latent-eps 10.0 --waveform-eps 0.5 --steps 500 \
        --n-eot-samples 4 --rir-dir data/rirs
"""

import os
import sys
import time
import argparse

import torch
import torchaudio
import torchaudio.functional as AF

from config import (
    WHISPER_INJECT_ROOT, TARGET_MODEL, MUSIC_FILES, DEFAULT_MUSIC,
    ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, PROMPT_MODES, DEFAULT_PROMPT_MODE,
    RESULTS_DIR, compute_wer,
)
from latent_attack import LatentCodecAttacker, AttackResult
from robust_latent_attack import (
    DifferentiableBandPassFilter,
    DifferentiableIRConv,
)
from channel_augmentation import AdditiveColoredNoise
from music_carrier import load_music_by_name, mel_distance

sys.path.insert(0, WHISPER_INJECT_ROOT)


class JointAttacker:
    """
    Joint latent + waveform adversarial attack.

    Simultaneously optimizes delta_latent (in EnCodec latent space) and
    delta_waveform (in audio waveform space, band-pass filtered) through
    the full pipeline including OTA channel simulation.
    """

    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        encodec_bandwidth: float = ENCODEC_BANDWIDTH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        # Latent perturbation
        latent_eps: float = LATENT_EPS,
        latent_alpha: float = LATENT_ALPHA,
        # Waveform perturbation
        waveform_eps: float = 0.5,
        waveform_alpha: float = 1e-3,
        # Shared
        perceptual_weight: float = PERCEPTUAL_WEIGHT,
        warmup_ratio: float = 0.3,
        # OTA channel
        rir_dir: str = None,
        bandpass_low_hz: float = 1000.0,
        bandpass_high_hz: float = 4000.0,
        noise_snr_db: float = 20.0,
        noise_snr_std: float = 5.0,
        n_eot_samples: int = 4,
    ):
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.latent_eps = latent_eps
        self.waveform_eps = waveform_eps
        self.latent_alpha = latent_alpha
        self.waveform_alpha = waveform_alpha
        self.perceptual_weight = perceptual_weight
        self.warmup_ratio = warmup_ratio
        self.n_eot_samples = n_eot_samples

        # Base attacker (for codec, model, encode/decode)
        self.log("Initializing model and codec...")
        self.base = LatentCodecAttacker(
            target_model=target_model,
            encodec_bandwidth=encodec_bandwidth,
            eps=latent_eps,
            alpha=latent_alpha,
            perceptual_weight=perceptual_weight,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )

        # OTA channel components
        self.log("Initializing OTA channel...")

        # Band-pass filter for waveform perturbation
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

        # Noise
        self.noise = AdditiveColoredNoise(
            sample_rate=ENCODEC_SAMPLE_RATE,
            snr_db_mean=noise_snr_db,
            snr_db_std=noise_snr_std,
        ).to(device)
        self.log(f"  Noise: SNR={noise_snr_db:.0f}±{noise_snr_std:.0f} dB")
        self.log(f"  EoT samples: {n_eot_samples}")

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _get_severity(self, step: int, total_steps: int) -> float:
        """Curriculum severity: ramp from 0.1 to 1.0 over the OTA hardening phase."""
        warmup_steps = int(total_steps * self.warmup_ratio)
        harden_steps = total_steps - warmup_steps
        if harden_steps <= 0:
            return 1.0
        progress = min((step - warmup_steps) / harden_steps, 1.0)
        return 0.1 + progress * 0.9

    def _apply_ota_channel(self, audio_24k: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """Apply OTA channel: RIR + noise, EoT averaged. Severity scales noise."""
        self.noise.train()
        acc = torch.zeros_like(audio_24k)
        for _ in range(self.n_eot_samples):
            x = audio_24k
            if self.ir_conv is not None:
                x = self.ir_conv(x)
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
        steps: int = 500,
        check_every: int = CHECK_EVERY,
        music_name: str = "",
        prompt: str = None,
    ) -> AttackResult:
        """
        Run joint latent + waveform attack.

        Both delta_latent and delta_waveform are optimized simultaneously.
        Stage 1 (warmup): no channel — learn the adversarial target
        Stage 2 (harden): OTA channel — harden for physical robustness
        """
        start_time = time.time()
        music_wav = music_wav.to(self.device)

        if hasattr(self.base.target_model, 'reset_cache'):
            self.base.target_model.reset_cache()

        # Encode music to latent space
        z_original = self.base.encode_music(music_wav)

        # Get original model output
        with torch.no_grad():
            original_audio_24k = self.base.codec.decode_from_continuous(z_original)
            original_audio_16k = AF.resample(
                original_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            original_output = self.base.target_model.generate(original_audio_16k, prompt=prompt)

        self.log(f"Original output: {original_output}")
        self.log(f"Target: {target_text}")

        # Initialize TWO learnable perturbations
        delta_latent = torch.zeros_like(z_original, requires_grad=True)
        delta_waveform = torch.zeros(
            1, 1, original_audio_24k.shape[-1],
            device=self.device, dtype=original_audio_24k.dtype,
            requires_grad=True,
        )

        # Joint optimizer with separate learning rates
        optimizer = torch.optim.Adam([
            {"params": [delta_latent], "lr": self.latent_alpha},
            {"params": [delta_waveform], "lr": self.waveform_alpha},
        ])

        # Stage boundaries
        warmup_steps = int(steps * self.warmup_ratio)

        # Enable gradients through decoder
        self.base.codec.model.decoder.train()

        # Tracking
        history = {"loss": [], "behavior_loss": [], "perceptual_loss": [], "outputs": [], "steps": []}
        best_loss = float('inf')
        best_delta_latent = delta_latent.data.clone()
        best_delta_waveform = delta_waveform.data.clone()

        self.log(f"\nStarting joint attack: latent_eps={self.latent_eps}, "
                 f"waveform_eps={self.waveform_eps}, steps={steps}")
        self.log(f"Stage 1 (steps 1-{warmup_steps}): warmup (no OTA channel)")
        self.log(f"Stage 2 (steps {warmup_steps + 1}-{steps}): "
                 f"OTA hardening with curriculum (RIR + noise, EoT x{self.n_eot_samples})")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        self.log("-" * 60)

        for step in range(steps):
            optimizer.zero_grad()

            use_channel = step >= warmup_steps

            # === Forward pass ===

            # 1. Latent perturbation → decode
            z_adv = z_original + delta_latent
            audio_decoded = self.base.codec.decode_from_continuous(z_adv)
            # audio_decoded: [1, 1, T] at 24kHz

            # 2. Add waveform perturbation (band-pass filtered)
            if self.bandpass is not None:
                waveform_pert = self.bandpass(delta_waveform)
            else:
                waveform_pert = delta_waveform

            audio_adv = audio_decoded + waveform_pert

            # 3. Apply OTA channel with curriculum severity (stage 2 only)
            if use_channel:
                severity = self._get_severity(step, steps)
                audio_for_loss = self._apply_ota_channel(audio_adv, severity=severity)
            else:
                audio_for_loss = audio_adv

            # 4. Resample to 16kHz for model
            audio_16k = AF.resample(
                audio_for_loss.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # === Loss computation ===
            behavior_loss = self.base.target_model.compute_loss(
                audio_16k, target_text, prompt=prompt
            )

            # Perceptual loss (on pre-channel audio vs original)
            p_loss_val = 0.0
            if self.perceptual_weight > 0:
                p_loss = mel_distance(
                    audio_adv.squeeze(0),
                    original_audio_24k.squeeze(0).detach(),
                    sample_rate=ENCODEC_SAMPLE_RATE,
                )
                total_loss = behavior_loss + self.perceptual_weight * p_loss
                p_loss_val = p_loss.item()
            else:
                total_loss = behavior_loss

            # === Backward + step ===
            total_loss.backward()
            optimizer.step()

            # Project both deltas to their L-inf balls
            with torch.no_grad():
                delta_latent.data = torch.clamp(
                    delta_latent.data, -self.latent_eps, self.latent_eps
                )
                delta_waveform.data = torch.clamp(
                    delta_waveform.data, -self.waveform_eps, self.waveform_eps
                )

            # Track
            loss_val = total_loss.item()
            b_loss_val = behavior_loss.item()
            history["loss"].append(loss_val)
            history["behavior_loss"].append(b_loss_val)
            history["perceptual_loss"].append(p_loss_val)

            # Only track best during OTA hardening phase (stage 2)
            # Warmup loss is artificially low (no channel) and those deltas
            # won't survive OTA, so we don't want them as "best"
            if use_channel and b_loss_val < best_loss:
                best_loss = b_loss_val
                best_delta_latent = delta_latent.data.clone()
                best_delta_waveform = delta_waveform.data.clone()

            # Progress logging
            if (step + 1) % 10 == 0:
                sev = self._get_severity(step, steps) if use_channel else 0.0
                stage = f"OTA s={sev:.2f}" if use_channel else "warm"
                # Compute waveform delta norm for monitoring
                wf_norm = delta_waveform.data.abs().max().item()
                lt_norm = delta_latent.data.abs().max().item()
                self.log(
                    f"Step {step+1:4d}/{steps} [{stage}] | "
                    f"Loss: {loss_val:.4f} | Beh: {b_loss_val:.4f} | "
                    f"Perc: {p_loss_val:.4f} | "
                    f"lat_max: {lt_norm:.2f} | wf_max: {wf_norm:.4f}"
                )

            # Periodic eval
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta_latent
                    audio_test = self.base.codec.decode_from_continuous(z_test)
                    if self.bandpass is not None:
                        audio_test = audio_test + self.bandpass(delta_waveform)
                    else:
                        audio_test = audio_test + delta_waveform

                    # Direct eval
                    test_16k = AF.resample(
                        audio_test.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output_direct = self.base.target_model.generate(test_16k, prompt=prompt)

                    # OTA eval (always full severity for honest evaluation)
                    test_ota = self._apply_ota_channel(audio_test, severity=1.0)
                    test_ota_16k = AF.resample(
                        test_ota.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output_ota = self.base.target_model.generate(test_ota_16k, prompt=prompt)

                history["outputs"].append({"direct": output_direct, "ota": output_ota})
                history["steps"].append(step + 1)

                direct_ok = target_text.lower() in output_direct.lower()
                ota_ok = target_text.lower() in output_ota.lower()
                self.log(f"  Direct: {'OK' if direct_ok else 'FAIL'} | {output_direct[:80]}")
                self.log(f"  OTA:    {'OK' if ota_ok else 'FAIL'} | {output_ota[:80]}")
                if ota_ok:
                    self.log(f"  >>> OTA TARGET ACHIEVED at step {step + 1}! <<<")
                    # Save deltas that actually passed OTA eval
                    best_delta_latent = delta_latent.data.clone()
                    best_delta_waveform = delta_waveform.data.clone()
                    best_loss = b_loss_val

        # Restore eval mode
        self.base.codec.model.decoder.eval()

        # === Final evaluation with best deltas ===
        with torch.no_grad():
            z_final = z_original + best_delta_latent
            audio_final = self.base.codec.decode_from_continuous(z_final)
            if self.bandpass is not None:
                audio_final = audio_final + self.bandpass(best_delta_waveform)
            else:
                audio_final = audio_final + best_delta_waveform

            # Direct eval
            final_16k = AF.resample(
                audio_final.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            output_direct = self.base.target_model.generate(final_16k, prompt=prompt)

            # OTA eval
            final_ota = self._apply_ota_channel(audio_final)
            final_ota_16k = AF.resample(
                final_ota.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            output_ota = self.base.target_model.generate(final_ota_16k, prompt=prompt)

        # SNR
        snr_db = self._compute_snr(
            original_audio_24k.squeeze().detach(),
            audio_final.squeeze().detach(),
        )
        latent_snr_db = self._compute_snr(
            z_original.detach(), (z_original + best_delta_latent).detach()
        )

        duration = time.time() - start_time
        direct_ok = target_text.lower() in output_direct.lower()
        ota_ok = target_text.lower() in output_ota.lower()

        self.log("\n" + "=" * 60)
        self.log("JOINT ATTACK COMPLETE")
        self.log(f"  Direct: {'OK' if direct_ok else 'FAIL'} | {output_direct[:100]}")
        self.log(f"  OTA:    {'OK' if ota_ok else 'FAIL'} | {output_ota[:100]}")
        self.log(f"  SNR: {snr_db:.1f} dB (audio), {latent_snr_db:.1f} dB (latent)")
        self.log(f"  Latent delta max: {best_delta_latent.abs().max().item():.2f}")
        self.log(f"  Waveform delta max: {best_delta_waveform.abs().max().item():.4f}")
        self.log(f"  Duration: {duration:.1f}s")
        self.log("=" * 60)

        result = AttackResult(
            original_wav=original_audio_24k.detach().cpu().squeeze(0),
            adversarial_wav=audio_final.squeeze(0).detach().cpu(),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=(z_original + best_delta_latent).detach().cpu(),
            latent_perturbation=best_delta_latent.detach().cpu(),
            original_output=original_output,
            adversarial_output=output_direct,
            target_text=target_text,
            success=ota_ok,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            latent_snr_db=latent_snr_db,
            history=history,
            music_name=music_name,
            attack_duration_s=duration,
        )
        result.ota_output = output_ota
        result.direct_success = direct_ok
        return result


def main():
    parser = argparse.ArgumentParser(description="Joint latent + waveform attack")

    parser.add_argument("--music", type=str, default=DEFAULT_MUSIC)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)

    # Perturbation budgets
    parser.add_argument("--latent-eps", type=float, default=LATENT_EPS)
    parser.add_argument("--latent-alpha", type=float, default=LATENT_ALPHA)
    parser.add_argument("--waveform-eps", type=float, default=0.5)
    parser.add_argument("--waveform-alpha", type=float, default=1e-3)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)

    # Steps
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-ratio", type=float, default=0.3)
    parser.add_argument("--check-every", type=int, default=100)

    # OTA channel
    parser.add_argument("--rir-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "rirs"))
    parser.add_argument("--bandpass-low-hz", type=float, default=1000.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=4000.0)
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.add_argument("--noise-snr-std", type=float, default=5.0)
    parser.add_argument("--n-eot-samples", type=int, default=4)

    # Output
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    rir_dir = args.rir_dir
    if not os.path.isdir(rir_dir):
        print(f"RIR directory not found: {rir_dir}")
        print("Run: python generate_rirs.py --output-dir data/rirs --n-rirs 615")
        sys.exit(1)

    music_wav = load_music_by_name(args.music)

    attacker = JointAttacker(
        target_model=args.target_model,
        latent_eps=args.latent_eps,
        latent_alpha=args.latent_alpha,
        waveform_eps=args.waveform_eps,
        waveform_alpha=args.waveform_alpha,
        perceptual_weight=args.perceptual_weight,
        warmup_ratio=args.warmup_ratio,
        rir_dir=rir_dir,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        noise_snr_db=args.noise_snr_db,
        noise_snr_std=args.noise_snr_std,
        n_eot_samples=args.n_eot_samples,
    )

    result = attacker.attack(
        music_wav,
        target_text=args.target,
        steps=args.steps,
        check_every=args.check_every,
        music_name=args.music,
        prompt=prompt_text,
    )

    # Save results
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(RESULTS_DIR), "results_joint",
        f"joint_leps{args.latent_eps}_weps{args.waveform_eps}_{args.music}"
    )
    os.makedirs(output_dir, exist_ok=True)

    import soundfile as sf
    import json

    adv_np = result.adversarial_wav.squeeze().cpu().numpy()
    sf.write(os.path.join(output_dir, "adversarial_24k.wav"), adv_np, ENCODEC_SAMPLE_RATE)

    adv_16k = AF.resample(result.adversarial_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    sf.write(os.path.join(output_dir, "adversarial_16k.wav"),
             adv_16k.squeeze().cpu().numpy(), TARGET_SAMPLE_RATE)

    summary = {
        "music": args.music,
        "target": args.target,
        "direct_output": result.adversarial_output,
        "ota_output": result.ota_output,
        "direct_success": result.direct_success,
        "ota_success": result.success,
        "snr_db": result.snr_db,
        "latent_snr_db": result.latent_snr_db,
        "steps": result.steps_taken,
        "duration_s": result.attack_duration_s,
        "config": {
            "latent_eps": args.latent_eps,
            "waveform_eps": args.waveform_eps,
            "latent_alpha": args.latent_alpha,
            "waveform_alpha": args.waveform_alpha,
            "bandpass_hz": [args.bandpass_low_hz, args.bandpass_high_hz],
            "noise_snr_db": args.noise_snr_db,
            "n_eot_samples": args.n_eot_samples,
            "warmup_ratio": args.warmup_ratio,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Direct: {'OK' if result.direct_success else 'FAIL'}")
    print(f"OTA:    {'OK' if result.success else 'FAIL'}")
    print(f"SNR: {result.snr_db:.1f} dB")


if __name__ == "__main__":
    main()
