"""
Yakura-style waveform OTA attack on Audio LLMs.

Pure waveform-domain attack with band-pass filtered perturbation + diverse
RIR convolution + Gaussian noise EoT — the exact Yakura & Sakuma (IJCAI 2019)
recipe, applied to Audio LLMs instead of DeepSpeech.

Key difference from joint_attack.py: NO latent space involved. The perturbation
delta is directly in 16kHz waveform space, band-pass filtered to 1-4 kHz.

Gradient flow:
    audio_16k + bandpass(delta)
    → upsample to 24kHz
    → OTA channel (random RIR + Gaussian noise, EoT)
    → downsample to 16kHz
    → model.compute_loss(audio_16k, target_text)
    → loss.backward() → gradient to delta

Two stages:
  Stage 1 (warmup): no OTA channel — learn the adversarial target
  Stage 2 (harden): OTA channel with curriculum severity

Usage:
    python yakura_waveform_attack.py --music jazz_1 \
        --target "Turn left at the next intersection" --eps 0.5 --steps 1000
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
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, PROMPT_MODES, DEFAULT_PROMPT_MODE,
    RESULTS_DIR, compute_wer, load_saycan_commands,
    AGENT_COMMANDS,
)
from latent_attack import AttackResult, UNTARGETED_WER_THRESHOLD
from robust_latent_attack import (
    DifferentiableBandPassFilter,
    DifferentiableIRConv,
)
from channel_augmentation import AdditiveColoredNoise
from music_carrier import load_music_by_name, mel_distance

sys.path.insert(0, WHISPER_INJECT_ROOT)


class YakuraWaveformAttacker:
    """
    Yakura-style waveform OTA attack on Audio LLMs.

    Perturbation is in 16kHz waveform space, band-pass filtered to
    constrain energy to frequencies that survive speaker→air→mic.
    OTA channel (RIR + noise) is applied at 24kHz with EoT averaging.
    """

    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        # Perturbation
        eps: float = 0.5,
        alpha: float = 1e-3,
        perceptual_weight: float = PERCEPTUAL_WEIGHT,
        warmup_ratio: float = 0.2,
        # Band-pass
        bandpass_low_hz: float = 1000.0,
        bandpass_high_hz: float = 4000.0,
        # OTA channel
        rir_dir: str = None,
        noise_snr_db: float = 20.0,
        noise_snr_std: float = 5.0,
        n_eot_samples: int = 4,
    ):
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.eps = eps
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight
        self.warmup_ratio = warmup_ratio
        self.n_eot_samples = n_eot_samples

        # Load target model
        self.log("Loading target model...")
        self._init_target_model(target_model, device, dtype)

        # Band-pass filter at 16kHz (perturbation's native rate)
        self.bandpass = None
        if bandpass_low_hz > 0 and bandpass_high_hz > bandpass_low_hz:
            self.bandpass = DifferentiableBandPassFilter(
                low_hz=bandpass_low_hz,
                high_hz=bandpass_high_hz,
                sample_rate=TARGET_SAMPLE_RATE,
                device=device,
            ).to(device)
            self.log(f"Band-pass: {bandpass_low_hz:.0f}-{bandpass_high_hz:.0f} Hz @ {TARGET_SAMPLE_RATE}Hz")

        # RIR convolution at 24kHz (OTA channel rate)
        self.ir_conv = None
        if rir_dir and os.path.isdir(rir_dir):
            self.ir_conv = DifferentiableIRConv(
                ir_dir=rir_dir,
                sample_rate=ENCODEC_SAMPLE_RATE,
                device=device,
            )
            self.ir_conv.train()  # Enable random RIR selection
            self.log(f"RIRs: {self.ir_conv._n_irs} from {rir_dir}")

        # Noise at 24kHz
        self.noise = AdditiveColoredNoise(
            sample_rate=ENCODEC_SAMPLE_RATE,
            snr_db_mean=noise_snr_db,
            snr_db_std=noise_snr_std,
        ).to(device)
        self.noise.train()
        self.log(f"Noise: SNR={noise_snr_db:.0f}±{noise_snr_std:.0f} dB")
        self.log(f"EoT samples: {n_eot_samples}")

    def _init_target_model(self, model_name, device, dtype):
        from config import MODEL_PATHS
        if model_name == "qwen2_audio":
            from models.qwen2_audio import Qwen2AudioModel
            self.target_model = Qwen2AudioModel(
                model_path=MODEL_PATHS.get(model_name), device=device, dtype=dtype)
        elif model_name == "qwen25_omni":
            from models.qwen25_omni import Qwen25OmniModel
            self.target_model = Qwen25OmniModel(
                model_path=MODEL_PATHS.get(model_name), device=device, dtype=dtype)
        elif model_name == "audio_flamingo":
            from models.audio_flamingo import AudioFlamingoModel
            self.target_model = AudioFlamingoModel(
                model_path=MODEL_PATHS.get(model_name), device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.log(f"Model loaded: {model_name}")

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _get_severity(self, step, total_steps):
        warmup_steps = int(total_steps * self.warmup_ratio)
        harden_steps = total_steps - warmup_steps
        if harden_steps <= 0:
            return 1.0
        progress = min((step - warmup_steps) / harden_steps, 1.0)
        return 0.1 + progress * 0.9

    def _apply_ota_channel(self, audio_24k, severity=1.0):
        """Apply OTA at 24kHz: RIR + noise, EoT averaged."""
        acc = torch.zeros_like(audio_24k)
        for _ in range(self.n_eot_samples):
            x = audio_24k
            if self.ir_conv is not None:
                x = self.ir_conv(x)
            x = self.noise(x, severity=severity)
            acc = acc + x
        return acc / self.n_eot_samples

    def _compute_snr(self, original, adversarial):
        noise = adversarial - original
        sig_pow = (original ** 2).mean()
        noise_pow = (noise ** 2).mean()
        if noise_pow < 1e-12:
            return 100.0
        return 10 * torch.log10(sig_pow / noise_pow).item()

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps: int = 1000,
        check_every: int = 200,
        music_name: str = "",
        prompt: str = None,
    ) -> AttackResult:
        """
        Run Yakura-style waveform OTA attack.

        Args:
            music_wav: [1, 1, T] at 24kHz
            target_text: target transcription
            steps: optimization steps
            prompt: model prompt
        """
        start_time = time.time()
        music_wav = music_wav.to(self.device)

        if hasattr(self.target_model, 'reset_cache'):
            self.target_model.reset_cache()

        # Resample to 16kHz for model
        music_16k = AF.resample(
            music_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )  # [1, T_16k]

        # Get original output
        with torch.no_grad():
            original_output = self.target_model.generate(music_16k, prompt=prompt)

        self.log(f"Original: {original_output}")
        self.log(f"Target: {target_text}")

        # Initialize perturbation at 16kHz
        delta = torch.zeros_like(music_16k, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        original_16k = music_16k.detach()
        warmup_steps = int(steps * self.warmup_ratio)

        # Tracking
        history = {"loss": [], "behavior_loss": [], "perceptual_loss": [], "outputs": [], "steps": []}
        best_loss = float('inf')
        best_delta = delta.data.clone()
        best_from_ota_eval = False

        self.log(f"\nYakura waveform attack: eps={self.eps}, steps={steps}")
        self.log(f"Stage 1 (1-{warmup_steps}): warmup (no channel)")
        self.log(f"Stage 2 ({warmup_steps+1}-{steps}): OTA harden (RIR+noise, EoT x{self.n_eot_samples})")
        self.log("-" * 60)

        for step in range(steps):
            optimizer.zero_grad()

            use_channel = step >= warmup_steps

            # Band-pass filter the perturbation
            if self.bandpass is not None:
                delta_filtered = self.bandpass(delta.unsqueeze(0)).squeeze(0)
            else:
                delta_filtered = delta

            # Adversarial audio at 16kHz
            audio_adv_16k = original_16k + delta_filtered

            if use_channel:
                severity = self._get_severity(step, steps)
                # Upsample to 24kHz for OTA channel
                audio_adv_24k = AF.resample(
                    audio_adv_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE
                )
                # Apply OTA channel at 24kHz
                audio_ota_24k = self._apply_ota_channel(audio_adv_24k, severity=severity)
                # Downsample back to 16kHz
                audio_for_loss = AF.resample(
                    audio_ota_24k, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                # Match length
                if audio_for_loss.shape[-1] > original_16k.shape[-1]:
                    audio_for_loss = audio_for_loss[..., :original_16k.shape[-1]]
            else:
                audio_for_loss = audio_adv_16k

            # Loss
            behavior_loss = self.target_model.compute_loss(
                audio_for_loss, target_text, prompt=prompt
            )

            p_loss_val = 0.0
            if self.perceptual_weight > 0:
                p_loss = mel_distance(
                    audio_adv_16k, original_16k.detach(),
                    sample_rate=TARGET_SAMPLE_RATE,
                )
                total_loss = behavior_loss + self.perceptual_weight * p_loss
                p_loss_val = p_loss.item()
            else:
                total_loss = behavior_loss

            total_loss.backward()
            optimizer.step()

            # Project to L-inf ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                # Keep audio in [-1, 1]
                delta.data = torch.clamp(original_16k + delta.data, -1.0, 1.0) - original_16k

            loss_val = total_loss.item()
            b_loss_val = behavior_loss.item()
            history["loss"].append(loss_val)
            history["behavior_loss"].append(b_loss_val)
            history["perceptual_loss"].append(p_loss_val)

            # Track best during OTA hardening
            if use_channel and b_loss_val < best_loss:
                best_loss = b_loss_val
                best_delta = delta.data.clone()

            if (step + 1) % 10 == 0:
                sev = self._get_severity(step, steps) if use_channel else 0.0
                stage = f"OTA s={sev:.2f}" if use_channel else "warm"
                dn = delta.data.abs().max().item()
                self.log(
                    f"Step {step+1:4d}/{steps} [{stage}] | "
                    f"Loss: {loss_val:.4f} | Beh: {b_loss_val:.4f} | "
                    f"Perc: {p_loss_val:.4f} | delta_max: {dn:.4f}"
                )

            # Periodic eval
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    if self.bandpass is not None:
                        df = self.bandpass(delta.unsqueeze(0)).squeeze(0)
                    else:
                        df = delta
                    test_16k = original_16k + df

                    # Direct
                    out_direct = self.target_model.generate(test_16k, prompt=prompt)

                    # OTA
                    test_24k = AF.resample(test_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
                    test_ota_24k = self._apply_ota_channel(test_24k, severity=1.0)
                    test_ota_16k = AF.resample(test_ota_24k, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                    if test_ota_16k.shape[-1] > original_16k.shape[-1]:
                        test_ota_16k = test_ota_16k[..., :original_16k.shape[-1]]
                    out_ota = self.target_model.generate(test_ota_16k, prompt=prompt)

                history["outputs"].append({"direct": out_direct, "ota": out_ota})
                history["steps"].append(step + 1)

                d_ok = target_text.lower() in out_direct.lower()
                o_ok = target_text.lower() in out_ota.lower()
                self.log(f"  Direct: {'OK' if d_ok else 'FAIL'} | {out_direct[:80]}")
                self.log(f"  OTA:    {'OK' if o_ok else 'FAIL'} | {out_ota[:80]}")
                if o_ok:
                    self.log(f"  >>> OTA TARGET at step {step+1} <<<")
                    best_delta = delta.data.clone()
                    best_loss = b_loss_val
                    best_from_ota_eval = True

        # Final eval with best delta
        with torch.no_grad():
            if self.bandpass is not None:
                df = self.bandpass(best_delta.unsqueeze(0)).squeeze(0)
            else:
                df = best_delta
            final_16k = original_16k + df
            out_direct = self.target_model.generate(final_16k, prompt=prompt)

            final_24k = AF.resample(final_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
            final_ota_24k = self._apply_ota_channel(final_24k, severity=1.0)
            final_ota_16k = AF.resample(final_ota_24k, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            if final_ota_16k.shape[-1] > original_16k.shape[-1]:
                final_ota_16k = final_ota_16k[..., :original_16k.shape[-1]]
            out_ota = self.target_model.generate(final_ota_16k, prompt=prompt)

        snr_db = self._compute_snr(original_16k.detach(), final_16k.detach())
        duration = time.time() - start_time
        d_ok = target_text.lower() in out_direct.lower()
        o_ok = target_text.lower() in out_ota.lower()

        self.log("\n" + "=" * 60)
        self.log("YAKURA WAVEFORM ATTACK COMPLETE")
        self.log(f"  Direct: {'OK' if d_ok else 'FAIL'} | {out_direct[:100]}")
        self.log(f"  OTA:    {'OK' if o_ok else 'FAIL'} | {out_ota[:100]}")
        self.log(f"  SNR: {snr_db:.1f} dB")
        self.log(f"  Delta max: {best_delta.abs().max().item():.4f}")
        self.log(f"  Duration: {duration:.1f}s")
        self.log("=" * 60)

        # Build result (reuse AttackResult, store waveform as adversarial_wav)
        result = AttackResult(
            original_wav=music_wav.squeeze(0).cpu(),
            adversarial_wav=final_24k.unsqueeze(0).cpu(),  # 24kHz for consistency
            original_latents=None,
            adversarial_latents=None,
            latent_perturbation=None,
            original_output=original_output,
            adversarial_output=out_direct,
            target_text=target_text,
            success=o_ok,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            latent_snr_db=0.0,
            history=history,
            music_name=music_name,
            attack_duration_s=duration,
        )
        result.ota_output = out_ota
        result.direct_success = d_ok
        result.audio_adv_16k = final_16k.detach().cpu()
        return result


def main():
    parser = argparse.ArgumentParser(description="Yakura waveform OTA attack")
    parser.add_argument("--music", default=DEFAULT_MUSIC)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--prompt-mode", default=DEFAULT_PROMPT_MODE)

    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--check-every", type=int, default=200)

    parser.add_argument("--bandpass-low-hz", type=float, default=1000.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=4000.0)
    parser.add_argument("--rir-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "rirs"))
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.add_argument("--noise-snr-std", type=float, default=5.0)
    parser.add_argument("--n-eot-samples", type=int, default=4)

    # Benchmark mode
    parser.add_argument("--dataset", default=None, choices=["ours", "saycan"])
    parser.add_argument("--n-commands", type=int, default=None)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    rir_dir = args.rir_dir
    if not os.path.isdir(rir_dir):
        print(f"RIR directory not found: {rir_dir}")
        print("Run: python generate_rirs.py --output-dir data/rirs --n-rirs 615")
        sys.exit(1)

    music_wav = load_music_by_name(args.music)

    attacker = YakuraWaveformAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        warmup_ratio=args.warmup_ratio,
        rir_dir=rir_dir,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        noise_snr_db=args.noise_snr_db,
        noise_snr_std=args.noise_snr_std,
        n_eot_samples=args.n_eot_samples,
    )

    # Single target or benchmark
    if args.dataset:
        if args.dataset == "saycan":
            all_commands = load_saycan_commands()
        else:
            all_commands = AGENT_COMMANDS
        cmd_keys = list(all_commands.keys())
        if args.n_commands:
            cmd_keys = cmd_keys[:args.n_commands]

        output_dir = args.output_dir or os.path.join(
            os.path.dirname(RESULTS_DIR), "results_yakura",
            f"yakura_wf_eps{args.eps}_{args.music}"
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"Yakura Waveform Benchmark: {len(cmd_keys)} commands, music={args.music}")
        print(f"  eps={args.eps}, steps={args.steps}, EoT={args.n_eot_samples}")
        print(f"  Output: {output_dir}\n")

        results = []
        for i, key in enumerate(cmd_keys):
            target = all_commands[key][args.prompt_mode]
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(cmd_keys)}] {key}: \"{target[:60]}...\"")
            print(f"{'='*70}")

            result = attacker.attack(
                music_wav, target_text=target, steps=args.steps,
                check_every=args.check_every, music_name=args.music, prompt=prompt_text,
            )

            entry = {
                "command": key,
                "target": target,
                "direct_output": result.adversarial_output,
                "ota_output": result.ota_output,
                "direct_success": result.direct_success,
                "ota_success": result.success,
                "snr_db": result.snr_db,
                "duration_s": result.attack_duration_s,
            }
            results.append(entry)
            d = "OK" if entry["direct_success"] else "FAIL"
            o = "OK" if entry["ota_success"] else "FAIL"
            print(f"  Direct={d} | OTA={o} | SNR={result.snr_db:.1f}dB | {result.attack_duration_s:.0f}s")

            # Save audio
            import soundfile as sf
            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            sf.write(
                os.path.join(audio_dir, f"{key}_16k.wav"),
                result.audio_adv_16k.squeeze().numpy(), TARGET_SAMPLE_RATE,
            )

        # Summary
        n = len(results)
        d_ok = sum(1 for r in results if r["direct_success"])
        o_ok = sum(1 for r in results if r["ota_success"])
        avg_snr = sum(r["snr_db"] for r in results) / n if n else 0

        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        print(f"Direct: {d_ok}/{n} ({100*d_ok/n:.0f}%)")
        print(f"OTA:    {o_ok}/{n} ({100*o_ok/n:.0f}%)")
        print(f"Avg SNR: {avg_snr:.1f} dB")

        import json
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_dir}")

    else:
        target = args.target or "Turn left at the next intersection"
        result = attacker.attack(
            music_wav, target_text=target, steps=args.steps,
            check_every=args.check_every, music_name=args.music, prompt=prompt_text,
        )

        output_dir = args.output_dir or os.path.join(
            os.path.dirname(RESULTS_DIR), "results_yakura",
            f"single_eps{args.eps}_{args.music}"
        )
        os.makedirs(output_dir, exist_ok=True)

        import soundfile as sf
        import json
        sf.write(
            os.path.join(output_dir, "adversarial_16k.wav"),
            result.audio_adv_16k.squeeze().numpy(), TARGET_SAMPLE_RATE,
        )
        summary = {
            "music": args.music, "target": target,
            "direct_output": result.adversarial_output,
            "ota_output": result.ota_output,
            "direct_success": result.direct_success,
            "ota_success": result.success,
            "snr_db": result.snr_db,
            "steps": result.steps_taken,
            "duration_s": result.attack_duration_s,
            "config": {
                "eps": args.eps, "alpha": args.alpha,
                "bandpass_hz": [args.bandpass_low_hz, args.bandpass_high_hz],
                "noise_snr_db": args.noise_snr_db,
                "n_eot_samples": args.n_eot_samples,
                "warmup_ratio": args.warmup_ratio,
            },
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
