"""
Two-stage encoder-embedding-matching attack for OTA robustness.

Stage 1: Run normal (non-OTA) latent attack to find adversarial audio that works.
          Extract target encoder embeddings from the working adversarial audio.

Stage 2: Hybrid optimization — initialize from Stage 1's waveform delta, then
          optimize with two losses through the OTA channel:
          - Encoder cosine-similarity loss every step (cheap, ~680M Whisper encoder)
          - LLM cross-entropy loss every N steps (expensive, full 7B backprop)
          The encoder loss provides consistent gradient signal; the LLM loss
          corrects course to ensure the decoder actually produces the target text.

Gradient path (Stage 2):
    delta → BPF(1-4kHz) → OTA(RIR+noise, EoT) → mel → WhisperEncoder → cos loss  (every step)
    delta → BPF(1-4kHz) → OTA(RIR+noise, EoT) → mel → WhisperEncoder → LLM → CE  (every N steps)

Usage:
    python encoder_match_attack.py --music jazz_1 --dataset saycan --n-commands 2
"""

import os
import sys
import time
import argparse

import torch
import torchaudio.functional as AF

from config import (
    WHISPER_INJECT_ROOT, TARGET_MODEL, DEFAULT_MUSIC,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, PERCEPTUAL_WEIGHT,
    PROMPT_MODES, DEFAULT_PROMPT_MODE,
    RESULTS_DIR, compute_wer, load_saycan_commands, AGENT_COMMANDS,
)
from latent_attack import LatentCodecAttacker, AttackResult
from robust_latent_attack import DifferentiableBandPassFilter, DifferentiableIRConv
from channel_augmentation import AdditiveColoredNoise
from music_carrier import load_music_by_name, mel_distance

sys.path.insert(0, WHISPER_INJECT_ROOT)


class EncoderMatchAttacker:
    """
    Two-stage attack: (1) find target embeddings, (2) survive OTA to reproduce them.
    """

    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        # Stage 1: latent attack params
        latent_eps: float = 0.5,
        latent_steps: int = 150,
        # Stage 2: waveform OTA params
        waveform_eps: float = 5.0,
        waveform_alpha: float = 0.01,
        waveform_steps: int = 500,
        # Band-pass
        bandpass_low_hz: float = 1000.0,
        bandpass_high_hz: float = 4000.0,
        # OTA channel
        rir_dir: str = None,
        noise_snr_db: float = 20.0,
        noise_snr_std: float = 5.0,
        n_eot_samples: int = 4,
        # Hybrid loss: mix in LLM CE loss every N steps
        llm_loss_every: int = 10,
        llm_loss_weight: float = 1.0,
    ):
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        # Stage 1 params
        self.latent_eps = latent_eps
        self.latent_steps = latent_steps

        # Stage 2 params
        self.wf_eps = waveform_eps
        self.wf_alpha = waveform_alpha
        self.wf_steps = waveform_steps
        self.n_eot_samples = n_eot_samples
        self.llm_loss_every = llm_loss_every
        self.llm_loss_weight = llm_loss_weight

        # Stage 1: latent attacker (for finding target embeddings)
        self.log("Loading latent attacker (Stage 1)...")
        self.latent_attacker = LatentCodecAttacker(
            target_model=target_model,
            eps=latent_eps,
            alpha=LATENT_ALPHA,
            perceptual_weight=PERCEPTUAL_WEIGHT,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )

        # Get reference to Qwen2Audio model and its audio encoder
        self.qwen_model = self.latent_attacker.target_model
        self.audio_tower = self.qwen_model.model.audio_tower

        # Stage 2: OTA channel components
        self.log("Initializing OTA channel (Stage 2)...")

        self.bandpass = None
        if bandpass_low_hz > 0 and bandpass_high_hz > bandpass_low_hz:
            self.bandpass = DifferentiableBandPassFilter(
                low_hz=bandpass_low_hz, high_hz=bandpass_high_hz,
                sample_rate=TARGET_SAMPLE_RATE, device=device,
            ).to(device)
            self.log(f"  BPF: {bandpass_low_hz:.0f}-{bandpass_high_hz:.0f} Hz")

        self.ir_conv = None
        if rir_dir and os.path.isdir(rir_dir):
            self.ir_conv = DifferentiableIRConv(
                ir_dir=rir_dir, sample_rate=ENCODEC_SAMPLE_RATE, device=device,
            )
            self.ir_conv.train()
            self.log(f"  RIRs: {self.ir_conv._n_irs} files")

        self.noise = AdditiveColoredNoise(
            sample_rate=ENCODEC_SAMPLE_RATE,
            snr_db_mean=noise_snr_db, snr_db_std=noise_snr_std,
        ).to(device)
        self.noise.train()
        self.log(f"  Noise: SNR={noise_snr_db:.0f}±{noise_snr_std:.0f} dB, EoT x{n_eot_samples}")

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _extract_encoder_embeddings(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract Whisper encoder embeddings from 16kHz audio. Differentiable."""
        mel = self.qwen_model.wav_to_mel(audio_16k)  # [1, 128, 3000]
        mel = mel.to(self.dtype)
        # The audio encoder (Whisper) internally downsamples 3000 → 1500 via conv layers.
        # Pass no attention_mask to let it process the full sequence.
        outputs = self.audio_tower(mel)
        return outputs.last_hidden_state  # [1, 1500, hidden_dim]

    def _apply_ota(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Apply OTA channel at 24kHz with EoT."""
        acc = torch.zeros_like(audio_24k)
        for _ in range(self.n_eot_samples):
            x = audio_24k
            if self.ir_conv is not None:
                x = self.ir_conv(x)
            x = self.noise(x, severity=1.0)
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
        prompt: str = None,
        music_name: str = "",
        check_every: int = 100,
    ) -> AttackResult:
        start_time = time.time()
        music_wav = music_wav.to(self.device)

        if hasattr(self.qwen_model, 'reset_cache'):
            self.qwen_model.reset_cache()

        # ==================== STAGE 1: Find target embeddings ====================
        self.log("\n" + "=" * 60)
        self.log("STAGE 1: Latent attack (find target embeddings)")
        self.log("=" * 60)

        result_s1 = self.latent_attacker.attack(
            music_wav, target_text=target_text,
            steps=self.latent_steps, music_name=music_name, prompt=prompt,
        )

        if not (target_text.lower() in result_s1.adversarial_output.lower()):
            self.log(f"WARNING: Stage 1 did not converge: {result_s1.adversarial_output[:80]}")
            self.log("Proceeding anyway — target embeddings may be suboptimal.")

        # Extract target encoder embeddings from Stage 1 adversarial audio
        adv_24k = result_s1.adversarial_wav.to(self.device)  # [1, T] at 24kHz
        adv_16k = AF.resample(adv_24k, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)

        with torch.no_grad():
            target_embeddings = self._extract_encoder_embeddings(adv_16k)
            self.log(f"Target embeddings: {target_embeddings.shape}")  # [1, T_enc, D]

        # ==================== STAGE 2: OTA-robust waveform optimization ====================
        self.log("\n" + "=" * 60)
        self.log("STAGE 2: Hybrid encoder-matching + LLM loss through OTA channel")
        self.log(f"  waveform_eps={self.wf_eps}, steps={self.wf_steps}")
        self.log(f"  Encoder embedding loss every step, LLM CE loss every {self.llm_loss_every} steps (weight={self.llm_loss_weight})")
        self.log("=" * 60)

        # Get clean 16kHz audio
        music_16k = AF.resample(
            music_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )  # [1, T_16k]
        original_16k = music_16k.detach()

        # Fix A: Initialize delta from Stage 1 adversarial result
        # Compute the waveform difference Stage 1 found, clamped to Stage 2's eps
        # Align lengths (resampling can shift by a few samples)
        min_len = min(adv_16k.shape[-1], original_16k.shape[-1])
        adv_16k = adv_16k[..., :min_len]
        original_16k = original_16k[..., :min_len]
        s1_delta_16k = (adv_16k - original_16k).detach()
        s1_delta_16k = torch.clamp(s1_delta_16k, -self.wf_eps, self.wf_eps)
        delta = s1_delta_16k.clone().requires_grad_(True)
        self.log(f"  Initialized delta from Stage 1 (max={s1_delta_16k.abs().max().item():.4f})")

        optimizer = torch.optim.Adam([delta], lr=self.wf_alpha)

        best_loss = float('inf')
        best_delta = delta.data.clone()

        # Enable grad for audio tower
        self.audio_tower.train()

        for step in range(self.wf_steps):
            optimizer.zero_grad()

            # Band-pass filter perturbation
            if self.bandpass is not None:
                delta_bp = self.bandpass(delta.unsqueeze(0)).squeeze(0)
            else:
                delta_bp = delta

            audio_adv_16k = original_16k + delta_bp

            # Upsample to 24kHz → OTA → downsample to 16kHz
            audio_24k = AF.resample(audio_adv_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
            audio_ota_24k = self._apply_ota(audio_24k.unsqueeze(0))
            audio_ota_16k = AF.resample(
                audio_ota_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            if audio_ota_16k.shape[-1] > original_16k.shape[-1]:
                audio_ota_16k = audio_ota_16k[..., :original_16k.shape[-1]]

            # Extract encoder embeddings after OTA
            current_embeddings = self._extract_encoder_embeddings(audio_ota_16k)

            # Encoder embedding loss (cosine similarity per token, averaged)
            emb_loss = 1.0 - torch.nn.functional.cosine_similarity(
                current_embeddings, target_embeddings.detach(), dim=-1,
            ).mean()

            emb_loss.backward()
            optimizer.step()

            # Fix B: Hybrid loss — LLM CE loss on separate forward (saves memory)
            # We do a second backward pass with a fresh, lightweight graph:
            # delta -> BPF -> single OTA sample -> LLM CE (no EoT accumulation)
            use_llm = (self.llm_loss_every > 0 and (step + 1) % self.llm_loss_every == 0)
            if use_llm:
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                # Recompute with single OTA sample (no EoT) to save memory
                if self.bandpass is not None:
                    delta_bp2 = self.bandpass(delta.unsqueeze(0)).squeeze(0)
                else:
                    delta_bp2 = delta
                adv2_16k = original_16k + delta_bp2
                adv2_24k = AF.resample(adv2_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
                # Single OTA sample
                x_ota = adv2_24k.unsqueeze(0)
                if self.ir_conv is not None:
                    x_ota = self.ir_conv(x_ota)
                x_ota = self.noise(x_ota, severity=1.0)
                x_ota_16k = AF.resample(x_ota.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
                if x_ota_16k.shape[-1] > original_16k.shape[-1]:
                    x_ota_16k = x_ota_16k[..., :original_16k.shape[-1]]

                llm_loss = self.qwen_model.compute_loss(
                    x_ota_16k, target_text=target_text, prompt=prompt,
                )
                (self.llm_loss_weight * llm_loss).backward()
                optimizer.step()
                torch.cuda.empty_cache()
            else:
                llm_loss = None

            # Project to L-inf ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.wf_eps, self.wf_eps)
                delta.data = torch.clamp(
                    original_16k + delta.data, -1.0, 1.0
                ) - original_16k

            emb_val = emb_loss.item()
            if emb_val < best_loss:
                best_loss = emb_val
                best_delta = delta.data.clone()

            if (step + 1) % 10 == 0:
                dn = delta.data.abs().max().item()
                llm_str = f" | LLM CE: {llm_loss.item():.4f}" if llm_loss is not None else ""
                self.log(
                    f"Step {step+1:4d}/{self.wf_steps} | "
                    f"Emb cos: {1.0 - emb_val:.4f}{llm_str} | delta_max: {dn:.4f}"
                )

            # Periodic LLM eval
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    if self.bandpass is not None:
                        df = self.bandpass(best_delta.unsqueeze(0)).squeeze(0)
                    else:
                        df = best_delta
                    test_16k = original_16k + df

                    # Direct eval
                    out_direct = self.qwen_model.generate(test_16k, prompt=prompt)

                    # OTA eval
                    test_24k = AF.resample(test_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
                    test_ota_24k = self._apply_ota(test_24k.unsqueeze(0))
                    test_ota_16k = AF.resample(
                        test_ota_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    if test_ota_16k.shape[-1] > original_16k.shape[-1]:
                        test_ota_16k = test_ota_16k[..., :original_16k.shape[-1]]
                    out_ota = self.qwen_model.generate(test_ota_16k, prompt=prompt)

                d_ok = target_text.lower() in out_direct.lower()
                o_ok = target_text.lower() in out_ota.lower()
                self.log(f"  Direct: {'OK' if d_ok else 'FAIL'} | {out_direct[:80]}")
                self.log(f"  OTA:    {'OK' if o_ok else 'FAIL'} | {out_ota[:80]}")
                if o_ok:
                    self.log(f"  >>> OTA TARGET at step {step+1} <<<")
                    best_delta = delta.data.clone()

        # Restore eval mode
        self.audio_tower.eval()

        # ==================== FINAL EVALUATION ====================
        with torch.no_grad():
            if self.bandpass is not None:
                df = self.bandpass(best_delta.unsqueeze(0)).squeeze(0)
            else:
                df = best_delta
            final_16k = original_16k + df

            out_direct = self.qwen_model.generate(final_16k, prompt=prompt)

            final_24k = AF.resample(final_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
            final_ota_24k = self._apply_ota(final_24k.unsqueeze(0))
            final_ota_16k = AF.resample(
                final_ota_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            if final_ota_16k.shape[-1] > original_16k.shape[-1]:
                final_ota_16k = final_ota_16k[..., :original_16k.shape[-1]]
            out_ota = self.qwen_model.generate(final_ota_16k, prompt=prompt)

            # Also eval original Stage 1 output for comparison
            original_output = self.qwen_model.generate(
                AF.resample(music_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE),
                prompt=prompt,
            )

        snr = self._compute_snr(original_16k.detach(), final_16k.detach())
        duration = time.time() - start_time
        d_ok = target_text.lower() in out_direct.lower()
        o_ok = target_text.lower() in out_ota.lower()

        self.log("\n" + "=" * 60)
        self.log("ENCODER MATCH ATTACK COMPLETE")
        self.log(f"  Stage 1: {'OK' if target_text.lower() in result_s1.adversarial_output.lower() else 'FAIL'} (latent, eps={self.latent_eps})")
        self.log(f"  Stage 2 Direct: {'OK' if d_ok else 'FAIL'} | {out_direct[:100]}")
        self.log(f"  Stage 2 OTA:    {'OK' if o_ok else 'FAIL'} | {out_ota[:100]}")
        self.log(f"  SNR: {snr:.1f} dB | Emb L2: {best_loss:.6f}")
        self.log(f"  Delta max: {best_delta.abs().max().item():.4f}")
        self.log(f"  Duration: {duration:.1f}s")
        self.log("=" * 60)

        result = AttackResult(
            original_wav=music_wav.squeeze(0).cpu(),
            adversarial_wav=final_24k.unsqueeze(0).cpu(),
            original_latents=None,
            adversarial_latents=None,
            latent_perturbation=None,
            original_output=original_output,
            adversarial_output=out_direct,
            target_text=target_text,
            success=o_ok,
            final_loss=best_loss,
            steps_taken=self.latent_steps + self.wf_steps,
            snr_db=snr,
            latent_snr_db=0.0,
            history={},
            music_name=music_name,
            attack_duration_s=duration,
        )
        result.ota_output = out_ota
        result.direct_success = d_ok
        result.stage1_success = target_text.lower() in result_s1.adversarial_output.lower()
        result.audio_adv_16k = final_16k.detach().cpu()
        return result


def main():
    parser = argparse.ArgumentParser(description="Encoder-matching OTA attack")
    parser.add_argument("--music", default=DEFAULT_MUSIC)
    parser.add_argument("--prompt-mode", default=DEFAULT_PROMPT_MODE)

    # Stage 1
    parser.add_argument("--latent-eps", type=float, default=0.5)
    parser.add_argument("--latent-steps", type=int, default=150)

    # Stage 2
    parser.add_argument("--waveform-eps", type=float, default=5.0)
    parser.add_argument("--waveform-alpha", type=float, default=0.01)
    parser.add_argument("--waveform-steps", type=int, default=500)

    # OTA
    parser.add_argument("--bandpass-low-hz", type=float, default=1000.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=4000.0)
    parser.add_argument("--rir-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "rirs"))
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.add_argument("--noise-snr-std", type=float, default=5.0)
    parser.add_argument("--n-eot-samples", type=int, default=4)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--llm-loss-every", type=int, default=10,
                        help="Mix in LLM CE loss every N steps (0=disable)")
    parser.add_argument("--llm-loss-weight", type=float, default=1.0)

    # Benchmark
    parser.add_argument("--dataset", default=None, choices=["ours", "saycan"])
    parser.add_argument("--n-commands", type=int, default=None)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    music_wav = load_music_by_name(args.music)

    attacker = EncoderMatchAttacker(
        latent_eps=args.latent_eps,
        latent_steps=args.latent_steps,
        waveform_eps=args.waveform_eps,
        waveform_alpha=args.waveform_alpha,
        waveform_steps=args.waveform_steps,
        rir_dir=args.rir_dir,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        noise_snr_db=args.noise_snr_db,
        noise_snr_std=args.noise_snr_std,
        n_eot_samples=args.n_eot_samples,
        llm_loss_every=args.llm_loss_every,
        llm_loss_weight=args.llm_loss_weight,
    )

    if args.dataset:
        all_cmds = load_saycan_commands() if args.dataset == "saycan" else AGENT_COMMANDS
        keys = list(all_cmds.keys())
        if args.n_commands:
            keys = keys[:args.n_commands]

        output_dir = args.output_dir or os.path.join(
            os.path.dirname(RESULTS_DIR), "results_encoder_match",
            f"enc_match_leps{args.latent_eps}_weps{args.waveform_eps}_{args.music}"
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nEncoder-Match Benchmark: {len(keys)} commands, music={args.music}")
        print(f"  Stage 1: latent eps={args.latent_eps}, {args.latent_steps} steps")
        print(f"  Stage 2: waveform eps={args.waveform_eps}, {args.waveform_steps} steps, EoT x{args.n_eot_samples}")
        print(f"  Output: {output_dir}\n")

        results = []
        for i, key in enumerate(keys):
            target = all_cmds[key][args.prompt_mode]
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(keys)}] {key}: \"{target[:60]}\"")
            print(f"{'='*70}")

            result = attacker.attack(
                music_wav, target_text=target, prompt=prompt_text,
                music_name=args.music, check_every=args.check_every,
            )

            entry = {
                "command": key, "target": target,
                "stage1_success": result.stage1_success,
                "direct_success": result.direct_success,
                "ota_success": result.success,
                "snr_db": result.snr_db,
                "emb_l2": result.final_loss,
                "duration_s": result.attack_duration_s,
                "direct_output": result.adversarial_output,
                "ota_output": result.ota_output,
            }
            results.append(entry)
            s1 = "OK" if entry["stage1_success"] else "FAIL"
            d = "OK" if entry["direct_success"] else "FAIL"
            o = "OK" if entry["ota_success"] else "FAIL"
            print(f"  S1={s1} | Direct={d} | OTA={o} | SNR={result.snr_db:.1f}dB")

            import soundfile as sf
            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            sf.write(
                os.path.join(audio_dir, f"{key}_16k.wav"),
                result.audio_adv_16k.squeeze().numpy(), TARGET_SAMPLE_RATE,
            )

        n = len(results)
        s1_ok = sum(1 for r in results if r["stage1_success"])
        d_ok = sum(1 for r in results if r["direct_success"])
        o_ok = sum(1 for r in results if r["ota_success"])
        avg_snr = sum(r["snr_db"] for r in results) / n if n else 0

        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        print(f"Stage 1: {s1_ok}/{n} ({100*s1_ok/n:.0f}%)")
        print(f"Direct:  {d_ok}/{n} ({100*d_ok/n:.0f}%)")
        print(f"OTA:     {o_ok}/{n} ({100*o_ok/n:.0f}%)")
        print(f"Avg SNR: {avg_snr:.1f} dB")

        import json
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_dir}")

    else:
        target = "Turn left at the next intersection"
        result = attacker.attack(
            music_wav, target_text=target, prompt=prompt_text,
            music_name=args.music, check_every=args.check_every,
        )


if __name__ == "__main__":
    main()
