"""Two-model targeted ensemble attack in the 16 kHz waveform domain.

The optimization and checkpoint objective match the clean-channel latent
ensemble: a 50-step primary warmup, then equally weighted, unit-normalized
input gradients from Qwen2-Audio and Omni-7B. The optimized variable is a
bounded additive waveform perturbation, with no EnCodec encoder or decoder.
"""

from __future__ import annotations

import os
import time

import torch
import torchaudio

from config import (ATTACK_STEPS, CHECK_EVERY, ENCODEC_SAMPLE_RATE,
                    TARGET_SAMPLE_RATE, compute_wer)
from latent_attack import AttackResult
from music_carrier import mel_distance
from model_registry import load_model
from waveform_attack import WaveformAttacker


class EnsembleWaveformAttacker(WaveformAttacker):
    """Optimize one waveform against Qwen2-Audio and Omni-7B together."""

    MODELS = ("qwen2_audio", "qwen25_omni")

    def __init__(self, *, ensemble_models, primary_model="qwen2_audio",
                 eps=0.075, alpha=0.015, perceptual_weight=0.1,
                 warmup_steps=50, grad_normalize=True,
                 best_after_warmup=True, device="cuda",
                 dtype=torch.bfloat16, verbose=True):
        if tuple(ensemble_models) != self.MODELS:
            raise ValueError(f"waveform ensemble requires {self.MODELS}, got {ensemble_models}")
        if primary_model != self.MODELS[0]:
            raise ValueError("waveform ensemble primary must be qwen2_audio")
        if not best_after_warmup:
            raise ValueError("waveform ensemble must select checkpoints after both models are active")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.perceptual_weight = float(perceptual_weight)
        self.warmup_steps = int(warmup_steps)
        self.grad_normalize = bool(grad_normalize)
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.ensemble_models = self.MODELS
        self.models = {}

        for name in self.MODELS:
            self.log(f"Loading waveform ensemble member: {name}")
            wrapper = load_model(name, device=device, dtype=dtype)
            hf = getattr(wrapper, "model", None)
            if hf is not None:
                for parameter in hf.parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
                if os.environ.get("ENSEMBLE_GRAD_CKPT", "0") == "1":
                    if hasattr(hf, "config"):
                        hf.config.use_cache = False
                    hf.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                    hf.train()
                    for module in hf.modules():
                        if isinstance(module, torch.nn.Dropout):
                            module.p = 0.0
                    if not any(getattr(m, "gradient_checkpointing", False)
                               for m in hf.modules()):
                        raise RuntimeError(f"gradient checkpointing was a no-op for {name}")
            self.models[name] = wrapper
        self.target_model = self.models[primary_model]

    def _model_gradient(self, name, audio_16k, target_text, prompt):
        wrapper = self.models[name]
        if hasattr(wrapper, "reset_cache"):
            wrapper.reset_cache()
        if os.environ.get("ENSEMBLE_GRAD_CKPT", "0") == "1":
            hf = getattr(wrapper, "model", None)
            if hf is not None:
                hf.train()
        model_input = audio_16k.detach().requires_grad_(True)
        loss = wrapper.compute_loss(model_input, target_text, prompt=prompt)
        loss.backward()
        if model_input.grad is None:
            raise RuntimeError(f"{name} returned no waveform gradient")
        grad = model_input.grad.detach().clone()
        value = float(loss.item())
        del loss, model_input
        if self.grad_normalize:
            norm = grad.norm()
            if norm > 1e-8:
                grad = grad / norm
        return grad, value

    def attack(self, music_wav, target_text, steps=ATTACK_STEPS,
               check_every=CHECK_EVERY, music_name="", prompt=None,
               untargeted=False, **kwargs) -> AttackResult:
        if untargeted:
            raise ValueError("this experiment is targeted transcription only")
        if steps <= self.warmup_steps:
            raise ValueError("steps must exceed the primary-only warmup")
        start = time.time()
        music_wav = music_wav.to(self.device)
        original_16k = torchaudio.functional.resample(
            music_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE).detach()
        with torch.no_grad():
            original_output = self.target_model.generate(original_16k, prompt=prompt)

        delta = torch.zeros_like(original_16k, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)
        best_loss = float("inf")
        best_delta = None
        best_step = None
        history = {"loss": [], "behavior_loss": [], "perceptual_loss": [],
                   "per_model_loss": {name: [] for name in self.MODELS},
                   "outputs": [], "steps": []}
        self.log(f"Waveform ensemble: eps={self.eps}, alpha={self.alpha}, "
                 f"steps={steps}, warmup={self.warmup_steps}, models={self.MODELS}")

        for step in range(steps):
            optimizer.zero_grad()
            audio_16k = original_16k + delta
            active = self.MODELS[:1] if step < self.warmup_steps else self.MODELS
            combined = torch.zeros_like(audio_16k)
            behavior = 0.0
            for name in active:
                grad, value = self._model_gradient(name, audio_16k, target_text, prompt)
                combined += grad / len(active)
                behavior += value / len(active)
                history["per_model_loss"][name].append(value)

            perceptual = 0.0
            if self.perceptual_weight > 0:
                p_loss = mel_distance(audio_16k, original_16k,
                                      sample_rate=TARGET_SAMPLE_RATE)
                perceptual = float(p_loss.item())
                (self.perceptual_weight * p_loss).backward(retain_graph=True)
            audio_16k.backward(combined)
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-self.eps, self.eps)
                delta.copy_((original_16k + delta).clamp(-1.0, 1.0) - original_16k)

            total = behavior + self.perceptual_weight * perceptual
            history["loss"].append(total)
            history["behavior_loss"].append(behavior)
            history["perceptual_loss"].append(perceptual)
            if step >= self.warmup_steps and total < best_loss:
                best_loss = total
                best_delta = delta.detach().clone()
                best_step = step + 1
            if (step + 1) % 10 == 0:
                self.log(f"Step {step+1}/{steps}: total={total:.4f}, "
                         f"Qwen2={history['per_model_loss']['qwen2_audio'][-1]:.4f}, "
                         f"Omni={history['per_model_loss']['qwen25_omni'][-1]:.4f}"
                         if step >= self.warmup_steps else
                         f"Step {step+1}/{steps}: warmup total={total:.4f}")
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    output = self.target_model.generate(original_16k + delta, prompt=prompt)
                history["outputs"].append(output)
                history["steps"].append(step + 1)
                self.log(f"Primary output at step {step+1}: "
                         f"match={target_text.lower() in output.lower()} "
                         f"text={output[:120]!r}")

        if best_delta is None:
            raise RuntimeError("no two-model checkpoint was selected")
        with torch.no_grad():
            final_16k = original_16k + best_delta
            final_output = self.target_model.generate(final_16k, prompt=prompt)
            final_24k = torchaudio.functional.resample(
                final_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
            original_24k = torchaudio.functional.resample(
                original_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE)
        success = target_text.lower() in final_output.lower()
        snr_db = self._compute_snr(original_16k.squeeze(), final_16k.squeeze())
        history["best_step"] = best_step
        self.log(f"Best waveform checkpoint: step {best_step}, "
                 f"loss={best_loss:.4f}, SNR={snr_db:.2f} dB, primary={success}")
        return AttackResult(
            original_wav=original_24k.detach().cpu(),
            adversarial_wav=final_24k.detach().cpu(),
            original_latents=None, adversarial_latents=None,
            latent_perturbation=None,
            original_output=original_output, adversarial_output=final_output,
            target_text=target_text, success=success, final_loss=best_loss,
            steps_taken=steps, snr_db=snr_db, latent_snr_db=0.0,
            history=history, music_name=music_name,
            attack_duration_s=time.time() - start,
        )
