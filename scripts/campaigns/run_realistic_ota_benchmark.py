"""
Realistic OTA benchmark: latent attacks optimized through measured channel.

Uses the empirical transfer function from real SoundCloud+airgap recordings
to build a more accurate channel simulation. Includes loudness normalization,
measured frequency shaping, speaker nonlinearity, AGC, and ambient noise.

Results saved to results_realistic_ota/.

Usage:
    python run_realistic_ota_benchmark.py --music calm_1 --eps 5.0 --steps 300
    python run_realistic_ota_benchmark.py --quick --music calm_1 --eps 5.0
"""

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
import torchaudio

from config import (
    MUSIC_FILES, DEFAULT_MUSIC, AGENT_COMMANDS, RESULTS_DIR,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
    OPUS_EVAL_BITRATES, PROMPT_MODES, DEFAULT_PROMPT_MODE, TARGET_MODEL,
    DATASET_CHOICES, load_saycan_commands,
    compute_wer, save_results_summary, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
)
from music_carrier import load_music_by_name, resolve_music_path
from robust_latent_attack import RobustLatentCodecAttacker, AttackResult
from realistic_channel import RealisticOTAChannel, FrequencyShapedPerturbation
from aac_channel import apply_aac_compression

from run_benchmark import (
    _music_short_name, CATEGORY_PREFIXES, get_category,
    filter_commands, load_completed, eval_opus,
    _save_summary, _print_summary,
)
from run_robust_benchmark import eval_ota


# Paths to empirical channel data
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TF_DIR = os.path.join(_SCRIPT_DIR, "soundcloud_recorded", "ir_trained_qwen2_audio_calm_1_eps_5.0")
_TF_PATH = os.path.join(_TF_DIR, "real_channel_tf.npy")
_TF_FREQS_PATH = os.path.join(_TF_DIR, "real_channel_tf_freqs.npy")
_BG_NOISE_PATH = os.path.join(_SCRIPT_DIR, "background_empty_noise.m4a")


class RealisticOTAAttacker(RobustLatentCodecAttacker):
    """
    Extends RobustLatentCodecAttacker with realistic OTA channel.

    Replaces the simple IR convolution with a full measured channel:
      loudness norm -> freq shaping -> speaker clip -> AGC -> noise

    Optionally applies frequency-shaped perturbation to concentrate
    adversarial energy in the channel's passband (500Hz-8kHz).
    """

    def __init__(
        self,
        freq_shape_perturbation: bool = True,
        n_eot_channel: int = 4,
        channel_ratio: float = 0.8,
        tf_path: str = _TF_PATH,
        tf_freqs_path: str = _TF_FREQS_PATH,
        bg_noise_path: str = _BG_NOISE_PATH,
        noise_snr_db: float = 25.0,
        **kwargs,
    ):
        # Use IR mode as base (we override _apply_channel anyway)
        kwargs.setdefault("channel_mode", "ir")
        super().__init__(**kwargs)

        self.realistic_channel = RealisticOTAChannel(
            sample_rate=ENCODEC_SAMPLE_RATE,
            tf_path=tf_path,
            tf_freqs_path=tf_freqs_path,
            bg_noise_path=bg_noise_path,
            noise_snr_db=noise_snr_db,
        ).to(self.device)
        self.realistic_channel.train()

        self.n_eot_channel = n_eot_channel
        self.channel_ratio = channel_ratio

        self.freq_shaper = None
        if freq_shape_perturbation:
            self.freq_shaper = FrequencyShapedPerturbation(
                sample_rate=ENCODEC_SAMPLE_RATE,
                low_hz=400.0, high_hz=8000.0,
            ).to(self.device)
            self.log("Frequency-shaped perturbation: 400-8000 Hz passband")

        self.log(f"Realistic OTA channel loaded (tf={os.path.basename(tf_path)})")
        self.log(f"  EoT samples: {n_eot_channel} per channel step")
        self.log(f"  Channel ratio: {channel_ratio:.0%} channel / {1-channel_ratio:.0%} direct (post-warmup)")
        self.log(f"  Loudness norm: -14 LUFS +/- 2dB")
        self.log(f"  Freq shaping: empirical MacBook->iPhone transfer function")
        self.log(f"  Speaker clip: tanh soft-clip (drive=2.0, mix=0.3)")
        self.log(f"  Mic AGC: +/- 3dB random gain")
        self.log(f"  Noise: {bg_noise_path} (SNR ~{noise_snr_db} dB)")

    def _apply_channel(self, audio_24k: torch.Tensor,
                       step: int = 0, total_steps: int = 0,
                       severity: float = 1.0) -> torch.Tensor:
        """Apply realistic OTA channel simulation with optional severity."""
        return self.realistic_channel(audio_24k, severity=severity)

    def _apply_freq_shaping(self, delta: torch.Tensor) -> torch.Tensor:
        """Shape perturbation to channel-survivable frequencies."""
        if self.freq_shaper is not None:
            return self.freq_shaper(delta)
        return delta

    def attack(self, music_wav, target_text, steps=ATTACK_STEPS,
               check_every=50, music_name="", use_multi_scale_loss=False,
               prompt=None, untargeted=False):
        """
        Override to add frequency shaping of perturbation.

        After each PGD step, filter delta to concentrate energy in
        the channel passband before projecting back to epsilon ball.
        """
        if self.freq_shaper is None:
            return super().attack(
                music_wav, target_text, steps=steps, check_every=check_every,
                music_name=music_name, use_multi_scale_loss=use_multi_scale_loss,
                prompt=prompt, untargeted=untargeted,
            )

        # Custom attack loop with freq shaping
        start_time = time.time()
        music_wav = music_wav.to(self.device)

        if hasattr(self.target_model, 'reset_cache'):
            self.target_model.reset_cache()

        z_original = self.encode_music(music_wav)

        with torch.no_grad():
            original_audio_24k = self.codec.decode_from_continuous(z_original)
            original_audio_for_model = torchaudio.functional.resample(
                original_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            original_output = self.target_model.generate(
                original_audio_for_model, prompt=prompt
            )

        self.log(f"Original output: {original_output}")
        self.log(f"Target: {target_text}")

        delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        warmup_steps = int(steps * self.warmup_ratio)
        best_loss = float('inf')
        best_delta = delta.data.clone()
        best_gen_delta = None  # best delta verified by generate()
        best_gen_loss = float('inf')
        best_gen_has_channel = False  # whether best_gen passed channel check
        success = False
        history = {"loss": [], "outputs": [], "steps": [], "stage": []}
        check_every = min(check_every, 25)  # check more frequently

        self.log(f"\nStarting realistic OTA attack: eps={self.eps}, steps={steps}")
        self.log(f"Stage 1 (1-{warmup_steps}): vanilla warmup (direct only)")
        self.log(f"Stage 2 ({warmup_steps+1}-{steps}): mixed loss (direct + channel), "
                 f"EoT={self.n_eot_channel}, curriculum severity 0→1")
        self.log(f"Frequency shaping: ON (400-8000 Hz)")
        self.log("-" * 60)

        if hasattr(self.codec, 'model') and hasattr(self.codec.model, 'decoder'):
            self.codec.model.decoder.train()

        # Pre-compute original audio for freq shaping (detached, constant)
        with torch.no_grad():
            orig_audio_for_shaping = self.codec.decode_from_continuous(z_original)

        for step_i in range(steps):
            optimizer.zero_grad()

            in_warmup = step_i < warmup_steps

            # Standard latent perturbation
            z_adv = z_original + torch.clamp(delta, -self.eps, self.eps)

            # Decode
            adv_audio_24k = self.codec.decode_from_continuous(z_adv)

            # Apply frequency shaping to decoded audio perturbation (only after warmup)
            if self.freq_shaper is not None and not in_warmup:
                audio_delta = adv_audio_24k - orig_audio_for_shaping.detach()
                shaped_audio_delta = self._apply_freq_shaping(audio_delta)
                adv_audio_24k = orig_audio_for_shaping.detach() + shaped_audio_delta

            if in_warmup:
                # Stage 1: direct loss only
                audio_16k = torchaudio.functional.resample(
                    adv_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                loss = self.target_model.compute_loss(audio_16k, target_text, prompt=prompt)
                stage_label = "W/di"
            else:
                # Stage 2: mixed loss = direct + channel with curriculum severity
                # Curriculum: severity ramps 0→1 over post-warmup steps
                post_warmup_progress = (step_i - warmup_steps) / max(steps - warmup_steps - 1, 1)
                severity = min(post_warmup_progress, 1.0)
                # Channel weight ramps with severity
                channel_weight = severity * self.channel_ratio

                # Direct loss (always computed)
                audio_16k = torchaudio.functional.resample(
                    adv_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                direct_loss = self.target_model.compute_loss(audio_16k, target_text, prompt=prompt)

                # Channel loss with EoT (severity-scaled channel)
                eot_loss = torch.zeros(1, device=self.device)
                for _ in range(self.n_eot_channel):
                    ch_audio = self._apply_channel(
                        adv_audio_24k, step_i, steps, severity=severity
                    )
                    ch_16k = torchaudio.functional.resample(
                        ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    eot_loss = eot_loss + self.target_model.compute_loss(
                        ch_16k, target_text, prompt=prompt
                    )
                channel_loss = eot_loss / self.n_eot_channel

                # Mixed loss
                loss = (1 - channel_weight) * direct_loss + channel_weight * channel_loss
                stage_label = f"H/mix(s={severity:.2f})"

            if untargeted:
                loss = -loss

            loss.backward()
            optimizer.step()

            # Project delta back to L-inf ball
            with torch.no_grad():
                delta.data.clamp_(-self.eps, self.eps)

            loss_val = loss.item()
            history["loss"].append(loss_val)
            history["stage"].append("warmup" if in_warmup else f"mix_s{severity:.2f}")

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            # Check progress — save best delta that passes generate()
            if (step_i + 1) % check_every == 0 or step_i == steps - 1:
                with torch.no_grad():
                    z_check = z_original + torch.clamp(delta.data, -self.eps, self.eps)
                    check_audio = self.codec.decode_from_continuous(z_check)
                    check_16k = torchaudio.functional.resample(
                        check_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    # Direct check
                    output = self.target_model.generate(check_16k, prompt=prompt)
                    direct_match = target_text.lower() in output.lower()

                    # Channel check (deterministic eval mode)
                    ch_match = False
                    if not in_warmup:
                        self.realistic_channel.eval()
                        ch_audio = self.realistic_channel(check_audio)
                        self.realistic_channel.train()
                        ch_16k = torchaudio.functional.resample(
                            ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                        )
                        ch_output = self.target_model.generate(ch_16k, prompt=prompt)
                        ch_match = target_text.lower() in ch_output.lower()

                match = direct_match or ch_match
                match_str = ("D" if direct_match else "") + ("C" if ch_match else "")
                self.log(f"  [{step_i+1}/{steps}] {stage_label} loss={loss_val:.4f} "
                         f"best={best_loss:.4f} match={match_str or 'no'}")

                if match:
                    success = True
                    # Prioritize channel matches over direct-only matches
                    upgrade = (
                        best_gen_delta is None
                        or (ch_match and not best_gen_has_channel)
                        or (ch_match == best_gen_has_channel and loss_val < best_gen_loss)
                    )
                    if upgrade:
                        best_gen_delta = delta.data.clone()
                        best_gen_loss = loss_val
                        best_gen_has_channel = ch_match

                history["outputs"].append(output)
                history["steps"].append(step_i + 1)

        # Use generate-verified delta if available, else fall back to lowest-loss
        final_delta = best_gen_delta if best_gen_delta is not None else best_delta

        # Final result
        with torch.no_grad():
            z_final = z_original + torch.clamp(final_delta, -self.eps, self.eps)
            final_audio = self.codec.decode_from_continuous(z_final)
            final_16k = torchaudio.functional.resample(
                final_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            final_output = self.target_model.generate(final_16k, prompt=prompt)

            # IR output
            ir_audio = self._apply_channel(final_audio)
            ir_16k = torchaudio.functional.resample(
                ir_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            ir_output = self.target_model.generate(ir_16k, prompt=prompt)

        snr = self._compute_snr(
            original_audio_24k.squeeze(), final_audio.squeeze()
        )
        latent_snr = self._compute_snr(z_original.squeeze(), z_final.squeeze())

        elapsed = time.time() - start_time
        self.log(f"\nDone in {elapsed:.1f}s | SNR={snr:.1f}dB | "
                 f"Direct={'PASS' if target_text.lower() in final_output.lower() else 'FAIL'} | "
                 f"Channel={'PASS' if target_text.lower() in ir_output.lower() else 'FAIL'}")

        result = AttackResult(
            adversarial_wav=final_audio,
            adversarial_output=final_output,
            original_output=original_output,
            target_text=target_text,
            success=target_text.lower() in final_output.lower(),
            snr_db=snr,
            latent_snr_db=latent_snr,
            final_loss=best_loss,
            history=history,
            steps_taken=steps,
        )
        result.ir_output = ir_output  # extra field for realistic OTA eval
        return result


def eval_realistic_ota(attacker, result, prompt=None):
    """Evaluate through the realistic channel (not the simple IR)."""
    adv_audio = result.adversarial_wav.to(attacker.device)

    with torch.no_grad():
        attacker.realistic_channel.eval()
        channel_audio = attacker.realistic_channel(adv_audio)
        attacker.realistic_channel.train()

        ch_16k = torchaudio.functional.resample(
            channel_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        output = attacker.target_model.generate(ch_16k, prompt=prompt)

    matches = result.target_text.lower() in output.lower()
    return {"success": matches, "output": output}


def run_benchmark(args):
    """Run the realistic OTA benchmark."""
    if args.music:
        music_load_keys = [m.strip() for m in args.music.split(",")]
        music_names = [_music_short_name(m) for m in music_load_keys]
    else:
        music_load_keys = list(MUSIC_FILES.keys())
        music_names = list(MUSIC_FILES.keys())

    commands = filter_commands(args.category, mode=args.prompt_mode, dataset=args.dataset)
    if args.commands:
        commands = {k: v for k, v in commands.items() if k in args.commands}

    opus_bitrates = [int(b) for b in args.opus_bitrates.split(",")]
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    print(f"REALISTIC OTA Benchmark: {len(music_names)} music x {len(commands)} commands")
    print(f"eps={args.eps}, steps={args.steps}, freq_shape={not args.no_freq_shape}")

    # Output directory
    music_tag = "_".join(music_names)
    dataset_tag = f"_{args.dataset}" if args.dataset != "ours" else ""
    fs_tag = "" if not args.no_freq_shape else "_nofs"
    dir_name = f"benchmark_{args.target_model}_{args.prompt_mode}_eps{args.eps}_{music_tag}_realOTA{fs_tag}{dataset_tag}"
    results_dir = os.path.join(_SCRIPT_DIR, "results_realistic_ota")
    output_dir = args.output_dir or os.path.join(results_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    audio_dir = os.path.join(output_dir, "audio")
    model_dir = os.path.join(output_dir, args.target_model)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    completed = load_completed(output_dir, model_subdir=args.target_model)
    if completed:
        print(f"Skipping {len(completed)} completed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attacker = RealisticOTAAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        device=device,
        verbose=not args.quiet,
        warmup_ratio=args.warmup_ratio,
        freq_shape_perturbation=not args.no_freq_shape,
        n_eot_channel=args.n_eot,
        channel_ratio=args.channel_ratio,
        noise_snr_db=args.noise_snr_db,
    )

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "target_model": args.target_model, "eps": args.eps,
                "steps": args.steps, "warmup_ratio": args.warmup_ratio,
                "freq_shape": not args.no_freq_shape,
                "n_eot": args.n_eot, "channel_ratio": args.channel_ratio,
                "noise_snr_db": args.noise_snr_db,
                "dataset": args.dataset, "prompt_mode": args.prompt_mode,
            }, f, indent=2)

    all_results = []
    # Load existing
    for entry in sorted(os.listdir(model_dir)):
        rp = os.path.join(model_dir, entry, "record.json")
        if os.path.isfile(rp):
            try:
                with open(rp) as f:
                    all_results.append(json.load(f))
            except:
                pass

    total = len(music_names) * len(commands)
    done = len(completed)

    for mi, music_name in enumerate(music_names):
        music_wav = load_music_by_name(music_load_keys[mi])

        for cmd_name, cmd_info in commands.items():
            target_text = cmd_info["target"]
            if (music_name, cmd_name) in completed:
                continue

            done += 1
            print(f"\n[{done}/{total}] {music_name} / {cmd_name}", flush=True)

            result = attacker.attack(
                music_wav, target_text=target_text,
                steps=args.steps, music_name=music_name, prompt=prompt_text,
            )

            opus_results = eval_opus(attacker, result, opus_bitrates, prompt=prompt_text)
            ota_result = eval_ota(attacker, result, prompt=prompt_text)
            real_ch = eval_realistic_ota(attacker, result, prompt=prompt_text)

            ir_output = getattr(result, 'ir_output', None)
            ir_success = ir_output and target_text.lower() in ir_output.lower()

            record = {
                "music": music_name, "command_name": cmd_name,
                "target_text": target_text, "category": get_category(cmd_name),
                "direct": {
                    "success": target_text.lower() in result.adversarial_output.lower(),
                    "output": result.adversarial_output,
                    "snr_db": result.snr_db,
                    "latent_snr_db": result.latent_snr_db,
                    "final_loss": result.final_loss,
                },
                "ir": {"success": ir_success, "output": ir_output or ""},
                "opus": {},
                "sim_ota": {
                    "success": ota_result["success"], "output": ota_result["output"],
                },
                "realistic_ota": {
                    "success": real_ch["success"], "output": real_ch["output"],
                },
                "steps": result.steps_taken,
                "original_output": result.original_output,
            }
            for bw, r in opus_results.items():
                record["opus"][str(bw)] = {"success": r["matches_target"], "output": r["output"]}

            all_results.append(record)

            exp_key = f"{music_name}__{cmd_name}"
            attacker.save_adversarial_wav(result, audio_dir, exp_key)
            exp_dir = os.path.join(model_dir, exp_key)
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "record.json"), "w") as f:
                json.dump(record, f, indent=2)
            save_results_summary(output_dir, all_results, total=total)

            d = "OK" if record["direct"]["success"] else "--"
            ir_s = "OK" if ir_success else "--"
            rota = "OK" if real_ch["success"] else "--"
            print(f"  D={d} IR={ir_s} RealCh={rota} SNR={result.snr_db:.1f}dB", flush=True)

    # Summary
    n = len(all_results)
    if n > 0:
        print(f"\n{'='*60}")
        print(f"REALISTIC OTA SUMMARY (n={n})")
        print(f"{'='*60}")
        for key, label in [("direct", "Direct"), ("ir", "IR"),
                           ("sim_ota", "Simulated OTA"), ("realistic_ota", "Realistic OTA")]:
            ok = sum(1 for r in all_results if r.get(key, {}).get("success"))
            print(f"  {label:>16}: {ok}/{n} ({100*ok/n:.1f}%)")
        for bw in opus_bitrates:
            ok = sum(1 for r in all_results if r.get("opus",{}).get(str(bw),{}).get("success"))
            print(f"  {'Opus'+str(bw):>16}: {ok}/{n} ({100*ok/n:.1f}%)")

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Realistic OTA benchmark")
    parser.add_argument("--music", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--commands", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=LATENT_ALPHA)
    parser.add_argument("--perceptual-weight", type=float, default=PERCEPTUAL_WEIGHT)
    parser.add_argument("--warmup-ratio", type=float, default=0.3)
    parser.add_argument("--opus-bitrates", type=str, default="64,128")
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--dataset", type=str, default="saycan", choices=DATASET_CHOICES)
    parser.add_argument("--no-freq-shape", action="store_true")
    parser.add_argument("--n-eot", type=int, default=4, help="EoT samples per channel step")
    parser.add_argument("--channel-ratio", type=float, default=0.8, help="Fraction of post-warmup steps using channel")
    parser.add_argument("--noise-snr-db", type=float, default=25.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quick:
        if not args.music:
            args.music = "calm_1"
        args.steps = 150
        cmd_source = load_saycan_commands() if args.dataset == "saycan" else AGENT_COMMANDS
        args.commands = args.commands or list(cmd_source.keys())[:3]
        args.quiet = True

    run_benchmark(args)


if __name__ == "__main__":
    main()
