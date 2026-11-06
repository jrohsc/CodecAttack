"""
Neural OTA benchmark: latent attacks optimized through a learned nonlinear channel.

Uses a trained 1D U-Net that models the full physical OTA chain
(MacBook speaker → air → iPhone mic) including nonlinearities, AGC,
and phase distortion that the FIR-based RealisticOTAChannel cannot capture.

Usage:
    # Quick test (3 commands)
    python run_neural_ota_benchmark.py --quick --music calm_1 --eps 5.0

    # Full benchmark
    python run_neural_ota_benchmark.py --music calm_1 --eps 10.0 --steps 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torchaudio

from config import (
    MUSIC_FILES, AGENT_COMMANDS, LATENT_ALPHA, ATTACK_STEPS, PERCEPTUAL_WEIGHT,
    OPUS_EVAL_BITRATES, PROMPT_MODES, DEFAULT_PROMPT_MODE, TARGET_MODEL,
    DATASET_CHOICES, load_saycan_commands,
    save_results_summary, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
)
from music_carrier import load_music_by_name
from robust_latent_attack import RobustLatentCodecAttacker, AttackResult
from neural_channel import NeuralOTAChannel, NeuralChannelUNet, load_trained_channel
from realistic_channel import RealisticOTAChannel

from run_benchmark import (
    _music_short_name, get_category,
    filter_commands, load_completed, eval_opus,
    _save_summary, _print_summary,
)
from run_robust_benchmark import eval_ota

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CKPT = os.path.join(_SCRIPT_DIR, "checkpoints", "neural_channel_best.pt")

# Empirical channel for comparison eval
_TF_DIR = os.path.join(_SCRIPT_DIR, "soundcloud_recorded", "ir_trained_qwen2_audio_calm_1_eps_5.0")
_TF_PATH = os.path.join(_TF_DIR, "real_channel_tf.npy")
_TF_FREQS_PATH = os.path.join(_TF_DIR, "real_channel_tf_freqs.npy")
_BG_NOISE_PATH = os.path.join(_SCRIPT_DIR, "background_empty_noise.m4a")


class NeuralOTAAttacker(RobustLatentCodecAttacker):
    """
    Extends RobustLatentCodecAttacker with a trained neural channel model.

    The neural channel captures nonlinear distortion, phase smearing,
    and time-varying AGC from real OTA recordings.
    """

    def __init__(
        self,
        neural_channel_ckpt: str = _DEFAULT_CKPT,
        n_eot_channel: int = 4,
        channel_ratio: float = 0.8,
        neural_dropout: float = 0.15,
        also_eval_fir: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("channel_mode", "ir")
        super().__init__(**kwargs)

        # Load trained neural channel
        self.neural_channel = load_trained_channel(
            neural_channel_ckpt, device=self.device, dropout=neural_dropout,
        )

        self.n_eot_channel = n_eot_channel
        self.channel_ratio = channel_ratio

        # Optionally load FIR channel for comparison eval
        self.fir_channel = None
        if also_eval_fir and os.path.isfile(_TF_PATH):
            self.fir_channel = RealisticOTAChannel(
                sample_rate=ENCODEC_SAMPLE_RATE,
                tf_path=_TF_PATH,
                tf_freqs_path=_TF_FREQS_PATH,
                bg_noise_path=_BG_NOISE_PATH if os.path.isfile(_BG_NOISE_PATH) else None,
            ).to(self.device)
            self.fir_channel.eval()

        self.log(f"Neural OTA channel loaded from {neural_channel_ckpt}")
        self.log(f"  EoT samples: {n_eot_channel}, channel ratio: {channel_ratio:.0%}")
        self.log(f"  Neural dropout: {neural_dropout:.0%} (EoT stochasticity)")

    def _apply_channel(self, audio_24k, step=0, total_steps=0, severity=1.0):
        """
        Apply bandpass + neural channel model with STE (Straight-Through Estimator).

        1. Bandpass filter the perturbation (keeps energy in target band)
        2. Neural channel (STE: forward = nonlinear channel, backward = identity)
        """
        # Apply bandpass to perturbation (same as parent class)
        if self.bandpass is not None and hasattr(self, '_original_audio_24k'):
            perturbation = audio_24k - self._original_audio_24k
            perturbation_filtered = self.bandpass(perturbation)
            audio_24k = self._original_audio_24k + perturbation_filtered

        with torch.no_grad():
            degraded = self.neural_channel(audio_24k, severity=severity)
        # STE: forward uses degraded, backward treats as identity through audio_24k
        return audio_24k + (degraded - audio_24k).detach()

    def attack(self, music_wav, target_text, steps=ATTACK_STEPS,
               check_every=50, music_name="", use_multi_scale_loss=False,
               prompt=None, untargeted=False):
        """
        PGD attack with neural channel in the loop.

        Two-stage curriculum:
          Stage 1 (warmup): direct loss only
          Stage 2: mixed loss = (1-w)*direct + w*E[channel_loss]
                   with severity ramping 0→1 and EoT over dropout variations
        """
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
        best_gen_delta = None
        best_gen_loss = float('inf')
        best_gen_has_channel = False
        success = False
        history = {"loss": [], "outputs": [], "steps": [], "stage": []}
        check_every = min(check_every, 25)

        self.log(f"\nStarting neural OTA attack: eps={self.eps}, steps={steps}")
        self.log(f"Stage 1 (1-{warmup_steps}): direct loss only")
        self.log(f"Stage 2 ({warmup_steps+1}-{steps}): mixed (direct + neural channel)")
        self.log("-" * 60)

        if hasattr(self.codec, 'model') and hasattr(self.codec.model, 'decoder'):
            self.codec.model.decoder.train()

        for step_i in range(steps):
            optimizer.zero_grad()
            in_warmup = step_i < warmup_steps

            z_adv = z_original + torch.clamp(delta, -self.eps, self.eps)
            adv_audio_24k = self.codec.decode_from_continuous(z_adv)

            if in_warmup:
                audio_16k = torchaudio.functional.resample(
                    adv_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                loss = self.target_model.compute_loss(audio_16k, target_text, prompt=prompt)
                if untargeted:
                    loss = -loss
                loss.backward()
                stage_label = "W/di"
            else:
                progress = (step_i - warmup_steps) / max(steps - warmup_steps - 1, 1)
                severity = min(progress, 1.0)
                channel_weight = severity * self.channel_ratio

                # Alternate between direct and channel steps to avoid OOM
                # (two full Qwen2-Audio forward passes won't fit in memory simultaneously)
                use_channel = (step_i % 2 == 1) and channel_weight > 0.01

                if not use_channel:
                    # Direct loss step
                    audio_16k = torchaudio.functional.resample(
                        adv_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    loss = self.target_model.compute_loss(audio_16k, target_text, prompt=prompt)
                    stage_label = f"N/di(s={severity:.2f})"
                else:
                    # Channel loss step (STE)
                    self.neural_channel.train()
                    ch_audio = self._apply_channel(
                        adv_audio_24k, step_i, steps, severity=severity
                    )
                    ch_16k = torchaudio.functional.resample(
                        ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    loss = self.target_model.compute_loss(
                        ch_16k, target_text, prompt=prompt
                    )
                    stage_label = f"N/ch(s={severity:.2f})"

                if untargeted:
                    loss = -loss
                loss.backward()

            optimizer.step()

            with torch.no_grad():
                delta.data.clamp_(-self.eps, self.eps)

            loss_val = loss.item()
            history["loss"].append(loss_val)
            history["stage"].append("warmup" if in_warmup else f"neural_s{severity:.2f}")

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            if (step_i + 1) % check_every == 0 or step_i == steps - 1:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    z_check = z_original + torch.clamp(delta.data, -self.eps, self.eps)
                    check_audio = self.codec.decode_from_continuous(z_check)
                    check_16k = torchaudio.functional.resample(
                        check_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output = self.target_model.generate(check_16k, prompt=prompt)
                    direct_match = target_text.lower() in output.lower()

                    # Neural channel check
                    ch_match = False
                    if not in_warmup:
                        self.neural_channel.eval()
                        ch_audio = self.neural_channel(check_audio)
                        self.neural_channel.train()
                        ch_16k = torchaudio.functional.resample(
                            ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                        )
                        ch_output = self.target_model.generate(ch_16k, prompt=prompt)
                        ch_match = target_text.lower() in ch_output.lower()

                    match = direct_match or ch_match
                    match_str = ("D" if direct_match else "") + ("N" if ch_match else "")
                    self.log(f"  [{step_i+1}/{steps}] {stage_label} loss={loss_val:.4f} "
                             f"best={best_loss:.4f} match={match_str or 'no'}")

                    if match:
                        success = True
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

        final_delta = best_gen_delta if best_gen_delta is not None else best_delta

        with torch.no_grad():
            z_final = z_original + torch.clamp(final_delta, -self.eps, self.eps)
            final_audio = self.codec.decode_from_continuous(z_final)
            final_16k = torchaudio.functional.resample(
                final_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            final_output = self.target_model.generate(final_16k, prompt=prompt)

            # Neural channel output
            self.neural_channel.eval()
            neural_ch_audio = self.neural_channel(final_audio)
            self.neural_channel.train()
            neural_16k = torchaudio.functional.resample(
                neural_ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            neural_output = self.target_model.generate(neural_16k, prompt=prompt)

        snr = self._compute_snr(original_audio_24k.squeeze(), final_audio.squeeze())
        latent_snr = self._compute_snr(z_original.squeeze(), z_final.squeeze())

        elapsed = time.time() - start_time
        self.log(f"\nDone in {elapsed:.1f}s | SNR={snr:.1f}dB | "
                 f"Direct={'PASS' if target_text.lower() in final_output.lower() else 'FAIL'} | "
                 f"Neural={'PASS' if target_text.lower() in neural_output.lower() else 'FAIL'}")

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
        result.neural_output = neural_output
        return result


def eval_neural_channel(attacker, result, prompt=None):
    """Evaluate through the neural channel."""
    adv_audio = result.adversarial_wav.to(attacker.device)
    with torch.no_grad():
        attacker.neural_channel.eval()
        ch_audio = attacker.neural_channel(adv_audio)
        attacker.neural_channel.train()
        ch_16k = torchaudio.functional.resample(
            ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        output = attacker.target_model.generate(ch_16k, prompt=prompt)
    matches = result.target_text.lower() in output.lower()
    return {"success": matches, "output": output}


def eval_fir_channel(attacker, result, prompt=None):
    """Evaluate through the FIR-based realistic channel for comparison."""
    if attacker.fir_channel is None:
        return {"success": False, "output": "FIR channel not loaded"}
    adv_audio = result.adversarial_wav.to(attacker.device)
    with torch.no_grad():
        attacker.fir_channel.eval()
        ch_audio = attacker.fir_channel(adv_audio)
        ch_16k = torchaudio.functional.resample(
            ch_audio.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        output = attacker.target_model.generate(ch_16k, prompt=prompt)
    matches = result.target_text.lower() in output.lower()
    return {"success": matches, "output": output}


def run_benchmark(args):
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

    print(f"NEURAL OTA Benchmark: {len(music_names)} music x {len(commands)} commands")
    print(f"eps={args.eps}, steps={args.steps}, channel_ckpt={args.channel_ckpt}")

    music_tag = "_".join(music_names)
    dataset_tag = f"_{args.dataset}" if args.dataset != "ours" else ""
    dir_name = f"benchmark_{args.target_model}_{args.prompt_mode}_eps{args.eps}_{music_tag}_neuralOTA{dataset_tag}"
    results_dir = os.path.join(_SCRIPT_DIR, "results_neural_ota")
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
    attacker = NeuralOTAAttacker(
        target_model=args.target_model,
        eps=args.eps,
        alpha=args.alpha,
        perceptual_weight=args.perceptual_weight,
        device=device,
        verbose=not args.quiet,
        warmup_ratio=args.warmup_ratio,
        neural_channel_ckpt=args.channel_ckpt,
        n_eot_channel=args.n_eot,
        channel_ratio=args.channel_ratio,
        neural_dropout=args.neural_dropout,
    )

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "target_model": args.target_model, "eps": args.eps,
                "steps": args.steps, "warmup_ratio": args.warmup_ratio,
                "n_eot": args.n_eot, "channel_ratio": args.channel_ratio,
                "neural_dropout": args.neural_dropout,
                "channel_ckpt": args.channel_ckpt,
                "dataset": args.dataset, "prompt_mode": args.prompt_mode,
            }, f, indent=2)

    all_results = []
    for entry in sorted(os.listdir(model_dir)):
        rp = os.path.join(model_dir, entry, "record.json")
        if os.path.isfile(rp):
            try:
                with open(rp) as f:
                    all_results.append(json.load(f))
            except Exception:
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
            neural_ch = eval_neural_channel(attacker, result, prompt=prompt_text)
            fir_ch = eval_fir_channel(attacker, result, prompt=prompt_text)

            neural_output = getattr(result, 'neural_output', None)
            neural_inline = neural_output and target_text.lower() in neural_output.lower()

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
                "opus": {},
                "sim_ota": {
                    "success": ota_result["success"], "output": ota_result["output"],
                },
                "neural_ota": {
                    "success": neural_ch["success"], "output": neural_ch["output"],
                },
                "fir_ota": {
                    "success": fir_ch["success"], "output": fir_ch["output"],
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
            n = "OK" if neural_ch["success"] else "--"
            fir = "OK" if fir_ch["success"] else "--"
            print(f"  D={d} Neural={n} FIR={fir} SNR={result.snr_db:.1f}dB", flush=True)

    n = len(all_results)
    if n > 0:
        print(f"\n{'='*60}")
        print(f"NEURAL OTA SUMMARY (n={n})")
        print(f"{'='*60}")
        for key, label in [("direct", "Direct"), ("sim_ota", "Simulated OTA"),
                           ("neural_ota", "Neural OTA"), ("fir_ota", "FIR OTA")]:
            ok = sum(1 for r in all_results if r.get(key, {}).get("success"))
            print(f"  {label:>16}: {ok}/{n} ({100*ok/n:.1f}%)")
        for bw in opus_bitrates:
            ok = sum(1 for r in all_results if r.get("opus", {}).get(str(bw), {}).get("success"))
            print(f"  {'Opus'+str(bw):>16}: {ok}/{n} ({100*ok/n:.1f}%)")

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Neural OTA benchmark")
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
    parser.add_argument("--channel-ckpt", type=str, default=_DEFAULT_CKPT)
    parser.add_argument("--n-eot", type=int, default=4)
    parser.add_argument("--channel-ratio", type=float, default=0.8)
    parser.add_argument("--neural-dropout", type=float, default=0.15)
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
