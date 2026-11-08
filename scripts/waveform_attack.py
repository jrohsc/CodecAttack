"""
Waveform-domain adversarial attacks on Audio LLMs (baseline comparison).

Unlike LatentCodecAttacker which perturbs EnCodec's latent space, this
attacker perturbs the raw 16kHz waveform directly — the standard adversarial
audio approach. This serves as a baseline to demonstrate that waveform-domain
perturbations do NOT survive Opus compression, even when they succeed on
direct (uncompressed) evaluation.

Gradient flow:
    delta (learnable) -> audio_adv = audio_original + delta
    -> clamp to L-inf ball
    -> Qwen2AudioModel.compute_loss(audio_adv, target_text)
    -> loss.backward() -> gradients flow to delta
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from typing import Optional, Dict, List

import torch
import torchaudio
import numpy as np

from config import (
    WHISPER_INJECT_ROOT, MODEL_PATHS, TARGET_MODEL,
    ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    WAVEFORM_EPS, WAVEFORM_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, OPUS_TEST_BITRATES,
    RESULTS_DIR, compute_wer,
)
from latent_attack import AttackResult, UNTARGETED_WER_THRESHOLD
from music_carrier import mel_distance, multi_scale_mel_distance

# Add codec_wrappers to path for reusable components
sys.path.insert(0, WHISPER_INJECT_ROOT)


class WaveformAttacker:
    """
    Waveform-domain adversarial attack on Audio LLMs (baseline).

    Directly perturbs the 16kHz waveform that the target model sees.
    No EnCodec encoder/decoder is involved — the perturbation is added
    to the raw audio signal.
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
    ):
        self.eps = eps
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        self._init_target_model(target_model, device, dtype)

    def _init_target_model(self, model_name: str, device: str, dtype: torch.dtype):
        """Initialize target Audio LLM."""
        if model_name == "qwen2_audio":
            from models.qwen2_audio import Qwen2AudioModel
            model_path = MODEL_PATHS.get(model_name)
            self.target_model = Qwen2AudioModel(
                model_path=model_path,
                device=device,
                dtype=dtype,
            )
        elif model_name == "audio_flamingo":
            from models.audio_flamingo import AudioFlamingoModel
            model_path = MODEL_PATHS.get(model_name)
            self.target_model = AudioFlamingoModel(
                model_path=model_path,
                device=device,
                dtype=dtype,
            )
        elif model_name == "kimi_audio":
            from models.kimi_audio import KimiAudioModel
            model_path = MODEL_PATHS.get(model_name)
            self.target_model = KimiAudioModel(
                model_path=model_path,
                device=device,
                dtype=dtype,
            )
        elif model_name == "qwen25_omni":
            from models.qwen25_omni import Qwen25OmniModel
            model_path = MODEL_PATHS.get(model_name)
            self.target_model = Qwen25OmniModel(
                model_path=model_path,
                device=device,
                dtype=dtype,
            )
        elif model_name == "phi4":
            from model_registry import load_model
            self.target_model = load_model(model_name, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown target model: {model_name}")

        self.log(f"Target model loaded: {model_name}")

    def log(self, msg: str):
        if self.verbose:
            print(msg)

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
        Run waveform-domain adversarial attack.

        Args:
            music_wav: Music carrier [1, 1, T] at 24kHz (same input format as latent attack)
            target_text: Text to force the Audio LLM to output
            steps: Number of optimization steps
            check_every: Check model output every N steps
            music_name: Name of the music file (for logging)
            use_multi_scale_loss: Use multi-scale mel loss
            prompt: Text prompt for the model
            untargeted: If True, maximize loss on original output

        Returns:
            AttackResult with adversarial audio and metrics
        """
        start_time = time.time()

        music_wav = music_wav.to(self.device)

        # Reset model caches
        if hasattr(self.target_model, 'reset_cache'):
            self.target_model.reset_cache()

        # Resample from 24kHz to 16kHz (model's native rate)
        # music_wav is [1, 1, T] at 24kHz
        music_16k = torchaudio.functional.resample(
            music_wav.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )  # [1, T_16k]

        # Get original model output
        with torch.no_grad():
            original_output = self.target_model.generate(music_16k, prompt=prompt)

        self.log(f"Original model output: {original_output}")

        # Untargeted mode
        if untargeted:
            target_text = original_output
            self.log("UNTARGETED mode: maximizing loss on original output")
        self.log(f"Target: {target_text}")
        if prompt:
            self.log(f"Prompt: {prompt}")

        # Initialize learnable perturbation in waveform domain
        delta = torch.zeros_like(music_16k, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        # Cache original audio for perceptual loss
        original_16k = music_16k.detach()

        # History tracking
        history = {"loss": [], "behavior_loss": [], "perceptual_loss": [], "outputs": [], "steps": []}
        best_delta = delta.data.clone()
        best_loss = float('inf')
        success = False

        self.log(f"\nStarting waveform attack: eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log(f"Mode: {'untargeted' if untargeted else 'targeted'}")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        self.log(f"Waveform shape: {music_16k.shape} at {TARGET_SAMPLE_RATE}Hz")
        self.log("-" * 60)

        for step in range(steps):
            optimizer.zero_grad()

            # Apply perturbation in waveform domain
            audio_adv_16k = original_16k + delta

            # Behavior loss
            behavior_loss = self.target_model.compute_loss(audio_adv_16k, target_text, prompt=prompt)
            if untargeted:
                behavior_loss = -behavior_loss

            # Perceptual loss
            if self.perceptual_weight > 0:
                if use_multi_scale_loss:
                    p_loss = multi_scale_mel_distance(
                        audio_adv_16k, original_16k.detach(),
                        sample_rate=TARGET_SAMPLE_RATE
                    )
                else:
                    p_loss = mel_distance(
                        audio_adv_16k, original_16k.detach(),
                        sample_rate=TARGET_SAMPLE_RATE
                    )
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            total_loss = behavior_loss + self.perceptual_weight * p_loss
            total_loss.backward()
            optimizer.step()

            # Project to L-inf epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                # Also ensure the adversarial audio stays in valid range [-1, 1]
                delta.data = torch.clamp(original_16k + delta.data, -1.0, 1.0) - original_16k

            # Track
            loss_val = total_loss.item()
            b_loss_val = behavior_loss.item()
            p_loss_val = p_loss.item()

            history["loss"].append(loss_val)
            history["behavior_loss"].append(b_loss_val)
            history["perceptual_loss"].append(p_loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            # Progress logging
            if (step + 1) % 10 == 0:
                self.log(
                    f"Step {step+1:4d}/{steps} | "
                    f"Total: {loss_val:.4f} | "
                    f"Behavior: {b_loss_val:.4f} | "
                    f"Perceptual: {p_loss_val:.4f}"
                )

            # Check model output periodically
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    audio_test_16k = original_16k + delta
                    output = self.target_model.generate(audio_test_16k, prompt=prompt)

                history["outputs"].append(output)
                history["steps"].append(step + 1)
                self.log(f"  -> Output: {output[:100]}")

                if untargeted:
                    wer = compute_wer(target_text, output)
                    if wer > UNTARGETED_WER_THRESHOLD:
                        self.log(f"  -> DISRUPTED at step {step+1}! (WER={wer:.3f} > {UNTARGETED_WER_THRESHOLD})")
                        success = True
                    else:
                        self.log(f"  -> Not disrupted (WER={wer:.3f})")
                else:
                    if target_text.lower() in output.lower():
                        self.log(f"  -> TARGET ACHIEVED at step {step+1}!")
                        success = True

        # Final evaluation with best delta
        with torch.no_grad():
            audio_final_16k = original_16k + best_delta
            final_output = self.target_model.generate(audio_final_16k, prompt=prompt)

        # Definitive success check
        success = False
        if untargeted:
            final_wer = compute_wer(target_text, final_output)
            if final_wer > UNTARGETED_WER_THRESHOLD:
                success = True
            self.log(f"Final WER vs original: {final_wer:.3f} (threshold={UNTARGETED_WER_THRESHOLD})")
        else:
            if target_text.lower() in final_output.lower():
                success = True

        # Compute SNR (in waveform domain)
        snr_db = self._compute_snr(original_16k.squeeze().detach(), audio_final_16k.squeeze().detach())

        # Upsample adversarial audio back to 24kHz for saving/Opus testing
        audio_final_24k = torchaudio.functional.resample(
            audio_final_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE
        )
        original_24k = torchaudio.functional.resample(
            original_16k, TARGET_SAMPLE_RATE, ENCODEC_SAMPLE_RATE
        )

        duration = time.time() - start_time

        self.log("\n" + "=" * 60)
        self.log("Waveform Attack Complete")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Success: {success}")
        self.log(f"Final Output: {final_output}")
        self.log(f"Duration: {duration:.1f}s")
        self.log("=" * 60)

        return AttackResult(
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

    def test_opus_robustness(
        self,
        result: AttackResult,
        bitrates: List[int] = None,
        prompt: str = None,
    ) -> Dict[int, dict]:
        """
        Test if adversarial audio survives Opus compression.

        Uses ffmpeg to encode/decode Opus.

        Args:
            result: AttackResult from attack()
            bitrates: List of Opus bitrates to test (kbps)
            prompt: Text prompt to use for evaluation

        Returns:
            Dict mapping bitrate -> {output, matches_target, snr_db}
        """
        if bitrates is None:
            bitrates = OPUS_TEST_BITRATES

        # Find ffmpeg
        import shutil
        ffmpeg_candidates = []
        which_ffmpeg = shutil.which("ffmpeg")
        if which_ffmpeg:
            ffmpeg_candidates.append(which_ffmpeg)
        env_bin = os.path.dirname(sys.executable)
        env_candidate = os.path.join(env_bin, "ffmpeg")
        if os.path.isfile(env_candidate) and env_candidate not in ffmpeg_candidates:
            ffmpeg_candidates.append(env_candidate)
        conda_root = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "..")
        if not os.path.isdir(os.path.join(conda_root, "envs")):
            conda_root = os.path.expanduser("~/miniconda3")
        conda_root = os.path.join(conda_root, "envs")
        for env_name in ["codec-attack", "qwen-omni", "base"]:
            fallback = os.path.join(conda_root, env_name, "bin", "ffmpeg")
            if os.path.isfile(fallback) and fallback not in ffmpeg_candidates:
                ffmpeg_candidates.append(fallback)
        ffmpeg_bin = None
        for candidate in ffmpeg_candidates:
            try:
                subprocess.run([candidate, "-version"], capture_output=True, check=True)
                ffmpeg_bin = candidate
                break
            except (subprocess.CalledProcessError, OSError):
                continue
        if ffmpeg_bin is None:
            self.log("\nSkipping Opus robustness test: ffmpeg not found or broken")
            return {}

        adv_audio = result.adversarial_wav.squeeze().cpu()
        robustness = {}

        for bw in bitrates:
            self.log(f"\nTesting Opus compression at {bw} kbps...")

            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = os.path.join(tmpdir, "input.wav")
                opus_path = os.path.join(tmpdir, "compressed.opus")
                out_path = os.path.join(tmpdir, "output.wav")

                # Save adversarial audio as WAV at 24kHz
                import soundfile as sf
                wav_np = adv_audio.numpy() if adv_audio.dim() == 1 else adv_audio.squeeze().numpy()
                sf.write(wav_path, wav_np, ENCODEC_SAMPLE_RATE)

                # Encode to Opus
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", wav_path, "-c:a", "opus",
                     "-strict", "-2",
                     "-b:a", f"{bw}k", opus_path],
                    capture_output=True, check=True
                )

                # Decode back from Opus
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", opus_path, "-ar", str(ENCODEC_SAMPLE_RATE), out_path],
                    capture_output=True, check=True
                )

                # Load decoded audio
                decoded_np, sr = sf.read(out_path)
                decoded_audio = torch.FloatTensor(decoded_np).unsqueeze(0)  # [1, T]
                if sr != ENCODEC_SAMPLE_RATE:
                    decoded_audio = torchaudio.functional.resample(decoded_audio, sr, ENCODEC_SAMPLE_RATE)

            # Test with target model
            with torch.no_grad():
                decoded_16k = torchaudio.functional.resample(
                    decoded_audio.to(self.device), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                output = self.target_model.generate(decoded_16k, prompt=prompt)

            # SNR
            min_len = min(adv_audio.shape[-1], decoded_audio.shape[-1])
            snr = self._compute_snr(
                adv_audio[..., :min_len],
                decoded_audio.squeeze()[..., :min_len]
            )

            matches = result.target_text.lower() in output.lower()
            robustness[bw] = {
                "output": output,
                "matches_target": matches,
                "snr_db": snr,
            }

            self.log(f"  Opus {bw} kbps | Match: {matches} | SNR: {snr:.2f} dB")
            self.log(f"  Output: {output[:100]}")

        result.opus_robustness = robustness
        return robustness

    def save_adversarial_wav(
        self,
        result: AttackResult,
        audio_dir: str,
        filename: str,
    ) -> str:
        """Save adversarial WAV to a shared audio directory. Skip if exists."""
        os.makedirs(audio_dir, exist_ok=True)
        wav_path = os.path.join(audio_dir, f"{filename}.wav")

        if os.path.isfile(wav_path):
            self.log(f"Audio already exists, skipping: {wav_path}")
            return wav_path

        import soundfile as sf
        adv_wav = result.adversarial_wav.squeeze().cpu().numpy()
        sf.write(wav_path, adv_wav, ENCODEC_SAMPLE_RATE)
        self.log(f"Saved adversarial audio: {wav_path}")
        return wav_path

    @staticmethod
    def _compute_snr(original: torch.Tensor, modified: torch.Tensor) -> float:
        """Compute SNR in dB."""
        noise = modified - original
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)
        if noise_power < 1e-10:
            return float('inf')
        return (10 * torch.log10(signal_power / noise_power)).item()
