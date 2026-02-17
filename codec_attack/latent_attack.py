"""
Latent-space adversarial attacks on Audio LLMs via neural codec pipelines.

Core attacker class that:
1. Encodes music to EnCodec latent space
2. Optimizes perturbations in latent space (PGD / Adam)
3. Decodes adversarial latents to audio
4. Evaluates against Qwen2-Audio target model
5. Tests robustness to re-encoding and Opus compression

Gradient flow:
    delta (learnable) -> z_adv = z_original + delta
    -> EnCodec.decoder(z_adv) -> audio_24kHz
    -> resample(24k -> 16k) -> audio_16kHz
    -> Qwen2AudioModel.compute_loss(audio_16kHz, target_text)
    -> loss.backward() -> gradients flow to delta
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

import torch
import torchaudio
import numpy as np

from config import (
    FRAMEWORK_ROOT, MODEL_PATHS, TARGET_MODEL,
    ENCODEC_BANDWIDTH, ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE,
    LATENT_EPS, LATENT_ALPHA, ATTACK_STEPS, CHECK_EVERY,
    PERCEPTUAL_WEIGHT, ENCODEC_TEST_BITRATES, OPUS_TEST_BITRATES,
    RESULTS_DIR,
)
from music_carrier import mel_distance, multi_scale_mel_distance

# Add external framework to path for reusable components
sys.path.insert(0, FRAMEWORK_ROOT)


@dataclass
class AttackResult:
    """Result of a latent-space attack on an Audio LLM."""

    # Audio
    original_wav: Optional[torch.Tensor] = None      # [1, T] at 24kHz
    adversarial_wav: Optional[torch.Tensor] = None    # [1, T] at 24kHz

    # Latents
    original_latents: Optional[torch.Tensor] = None
    adversarial_latents: Optional[torch.Tensor] = None
    latent_perturbation: Optional[torch.Tensor] = None

    # Model outputs
    original_output: str = ""
    adversarial_output: str = ""
    target_text: str = ""

    # Metrics
    success: bool = False
    final_loss: float = float('inf')
    steps_taken: int = 0
    snr_db: float = 0.0
    latent_snr_db: float = 0.0

    # Robustness
    encodec_robustness: Dict[float, dict] = field(default_factory=dict)
    opus_robustness: Dict[int, dict] = field(default_factory=dict)

    # Training history
    history: Dict[str, list] = field(default_factory=dict)

    # Metadata
    music_name: str = ""
    attack_duration_s: float = 0.0

    def to_json(self) -> dict:
        """Serialize to JSON-safe dict (excludes tensors)."""
        return {
            "original_output": self.original_output,
            "adversarial_output": self.adversarial_output,
            "target_text": self.target_text,
            "success": self.success,
            "final_loss": self.final_loss,
            "steps_taken": self.steps_taken,
            "snr_db": self.snr_db,
            "latent_snr_db": self.latent_snr_db,
            "encodec_robustness": self.encodec_robustness,
            "opus_robustness": self.opus_robustness,
            "music_name": self.music_name,
            "attack_duration_s": self.attack_duration_s,
        }


class LatentCodecAttacker:
    """
    Adversarial attack on Audio LLMs via EnCodec latent-space perturbations.

    Composes:
    - EnCodec encoder/decoder
    - Qwen2AudioModel
    - Perceptual loss (mel-spectrogram distance)
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
    ):
        self.eps = eps
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        self._init_encodec(encodec_bandwidth)
        self._init_target_model(target_model, device, dtype)

    def _init_encodec(self, bandwidth: float):
        """Initialize EnCodec model."""
        from attacks.latent_codec import EnCodecWrapper
        self.codec = EnCodecWrapper(bandwidth=bandwidth, device=self.device)
        self.log(f"EnCodec loaded (bandwidth={bandwidth} kbps, sr={self.codec.sample_rate})")

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
        else:
            raise ValueError(f"Unknown target model: {model_name}")

        self.log(f"Target model loaded: {model_name}")

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def encode_music(self, music_wav: torch.Tensor) -> torch.Tensor:
        """
        Encode music to EnCodec continuous latent space.

        Args:
            music_wav: Music audio [1, 1, T] at 24kHz

        Returns:
            Continuous latent tensor z_original
        """
        music_wav = music_wav.to(self.device)
        with torch.no_grad():
            z = self.codec.encode_to_continuous(music_wav)
        self.log(f"Encoded to latent space: {z.shape}")
        return z

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps: int = ATTACK_STEPS,
        check_every: int = CHECK_EVERY,
        music_name: str = "",
        use_multi_scale_loss: bool = False,
        prompt: str = None,
    ) -> AttackResult:
        """
        Run latent-space adversarial attack.

        Args:
            music_wav: Music carrier [1, 1, T] at 24kHz
            target_text: Text to force the Audio LLM to output
            steps: Number of optimization steps
            check_every: Check model output every N steps
            music_name: Name of the music file (for logging)
            use_multi_scale_loss: Use multi-scale mel loss
            prompt: Text prompt for the model (default: transcription)

        Returns:
            AttackResult with adversarial audio and metrics
        """
        start_time = time.time()

        music_wav = music_wav.to(self.device)

        # Reset model caches (e.g. Kimi Audio's cached prompt structure)
        if hasattr(self.target_model, 'reset_cache'):
            self.target_model.reset_cache()

        # Encode to latent space
        z_original = self.encode_music(music_wav)

        # Cache original decoded audio for perceptual loss
        with torch.no_grad():
            original_audio_24k = self.codec.decode_from_continuous(z_original)
            original_audio_16k = torchaudio.functional.resample(
                original_audio_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            original_output = self.target_model.generate(original_audio_16k, prompt=prompt)

        self.log(f"Original model output: {original_output}")
        self.log(f"Target: {target_text}")
        if prompt:
            self.log(f"Prompt: {prompt}")

        # Initialize learnable perturbation
        delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        # History tracking
        history = {"loss": [], "behavior_loss": [], "perceptual_loss": [], "outputs": [], "steps": []}
        best_delta = delta.data.clone()
        best_loss = float('inf')
        success = False

        self.log(f"\nStarting latent attack: eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log(f"Perceptual weight: {self.perceptual_weight}")
        self.log("-" * 60)

        # Enable gradients through EnCodec decoder
        self.codec.model.decoder.train()

        for step in range(steps):
            optimizer.zero_grad()

            # Apply perturbation in latent space
            z_adv = z_original + delta

            # Decode to audio (24kHz)
            audio_adv_24k = self.codec.decode_from_continuous(z_adv)

            # Resample to target model's sample rate (16kHz)
            audio_adv_16k = torchaudio.functional.resample(
                audio_adv_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # Behavior loss: force target output
            behavior_loss = self.target_model.compute_loss(audio_adv_16k, target_text, prompt=prompt)

            # Perceptual loss: keep audio sounding like original music
            if self.perceptual_weight > 0:
                if use_multi_scale_loss:
                    p_loss = multi_scale_mel_distance(
                        audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE
                    )
                else:
                    p_loss = mel_distance(
                        audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach(),
                        sample_rate=ENCODEC_SAMPLE_RATE
                    )
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            total_loss = behavior_loss + self.perceptual_weight * p_loss
            total_loss.backward()
            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)

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
                    z_test = z_original + delta
                    audio_test_24k = self.codec.decode_from_continuous(z_test)
                    audio_test_16k = torchaudio.functional.resample(
                        audio_test_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                    )
                    output = self.target_model.generate(audio_test_16k, prompt=prompt)

                history["outputs"].append(output)
                history["steps"].append(step + 1)
                self.log(f"  -> Output: {output[:100]}")

                if target_text.lower() in output.lower():
                    self.log(f"  -> TARGET ACHIEVED at step {step+1}!")
                    success = True

        # Restore eval mode
        self.codec.model.decoder.eval()

        # Final evaluation with best delta
        with torch.no_grad():
            z_final = z_original + best_delta
            audio_final_24k = self.codec.decode_from_continuous(z_final)
            audio_final_16k = torchaudio.functional.resample(
                audio_final_24k.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            final_output = self.target_model.generate(audio_final_16k, prompt=prompt)

        if target_text.lower() in final_output.lower():
            success = True

        # Compute SNR
        snr_db = self._compute_snr(
            original_audio_24k.squeeze().detach(),
            audio_final_24k.squeeze().detach()
        )
        latent_snr_db = self._compute_snr(z_original.detach(), z_final.detach())

        duration = time.time() - start_time

        self.log("\n" + "=" * 60)
        self.log("Attack Complete")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Latent SNR: {latent_snr_db:.2f} dB")
        self.log(f"Success: {success}")
        self.log(f"Final Output: {final_output}")
        self.log(f"Duration: {duration:.1f}s")
        self.log("=" * 60)

        return AttackResult(
            original_wav=original_audio_24k.detach().cpu().squeeze(0),
            adversarial_wav=audio_final_24k.detach().cpu().squeeze(0),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=z_final.detach().cpu(),
            latent_perturbation=best_delta.detach().cpu(),
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

    def test_encodec_robustness(
        self,
        result: AttackResult,
        bitrates: List[float] = None,
        prompt: str = None,
    ) -> Dict[float, dict]:
        """
        Test if adversarial audio survives EnCodec re-encoding at various bitrates.

        Args:
            result: AttackResult from attack()
            bitrates: List of bitrates to test (kbps)
            prompt: Text prompt to use for evaluation (default: transcription)

        Returns:
            Dict mapping bitrate -> {output, matches_target, snr_db}
        """
        if bitrates is None:
            bitrates = ENCODEC_TEST_BITRATES

        adv_audio = result.adversarial_wav.to(self.device)
        if adv_audio.dim() == 2:
            adv_audio = adv_audio.unsqueeze(0)

        robustness = {}

        for bw in bitrates:
            self.log(f"\nTesting EnCodec re-encoding at {bw} kbps...")
            original_bw = self.codec._bandwidth
            self.codec.set_bandwidth(bw)

            with torch.no_grad():
                # Full encode-decode cycle using model directly
                # (bypasses wrapper's encode/decode which has shape issues)
                frames = self.codec.model.encode(adv_audio)
                reencoded = self.codec.model.decode(frames)

                # Resample and test
                reencoded_16k = torchaudio.functional.resample(
                    reencoded.squeeze(0), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )
                output = self.target_model.generate(reencoded_16k, prompt=prompt)

                # SNR between adversarial and re-encoded
                snr = self._compute_snr(
                    adv_audio.squeeze().detach(),
                    reencoded.squeeze().detach()
                )

            matches = result.target_text.lower() in output.lower()
            robustness[bw] = {
                "output": output,
                "matches_target": matches,
                "snr_db": snr,
            }

            self.log(f"  Bitrate {bw} kbps | Match: {matches} | SNR: {snr:.2f} dB")
            self.log(f"  Output: {output[:100]}")

            self.codec.set_bandwidth(original_bw)

        result.encodec_robustness = robustness
        return robustness

    def test_opus_robustness(
        self,
        result: AttackResult,
        bitrates: List[int] = None,
        prompt: str = None,
    ) -> Dict[int, dict]:
        """
        Test if adversarial audio survives Opus compression.

        Uses ffmpeg to encode/decode Opus, maintaining gradient-free pipeline.

        Args:
            result: AttackResult from attack()
            bitrates: List of Opus bitrates to test (kbps)
            prompt: Text prompt to use for evaluation (default: transcription)

        Returns:
            Dict mapping bitrate -> {output, matches_target, snr_db}
        """
        if bitrates is None:
            bitrates = OPUS_TEST_BITRATES

        # Find ffmpeg: check PATH, then look next to the current Python binary,
        # then try well-known conda envs. Validate that the binary actually runs.
        import shutil
        ffmpeg_candidates = []
        which_ffmpeg = shutil.which("ffmpeg")
        if which_ffmpeg:
            ffmpeg_candidates.append(which_ffmpeg)
        env_bin = os.path.dirname(sys.executable)
        env_candidate = os.path.join(env_bin, "ffmpeg")
        if os.path.isfile(env_candidate) and env_candidate not in ffmpeg_candidates:
            ffmpeg_candidates.append(env_candidate)
        # Fallback: try whisperinject-v2 env which is known to have a working ffmpeg
        conda_root = os.path.expanduser("~/miniconda3/envs")
        for env_name in ["whisperinject-v2", "base"]:
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

                # Save adversarial audio as WAV (use soundfile to avoid torchcodec dep)
                import soundfile as sf
                wav_np = adv_audio.numpy() if adv_audio.dim() == 1 else adv_audio.squeeze().numpy()
                sf.write(wav_path, wav_np, ENCODEC_SAMPLE_RATE)

                # Encode to Opus using ffmpeg (opus encoder resamples to 48kHz internally)
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

                # Load decoded audio (use soundfile to avoid torchcodec dep)
                import soundfile as sf
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

    def save_result(
        self,
        result: AttackResult,
        output_dir: str = None,
        prefix: str = "attack",
    ):
        """
        Save attack result: adversarial audio + JSON metrics.

        Args:
            result: AttackResult to save
            output_dir: Directory to save to (default: RESULTS_DIR)
            prefix: Filename prefix
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{result.music_name}_{timestamp}"

        # Save adversarial audio (use soundfile to avoid torchcodec dep)
        import soundfile as sf
        audio_path = os.path.join(output_dir, f"{base_name}_adversarial.wav")
        adv_wav = result.adversarial_wav.squeeze().cpu().numpy()
        sf.write(audio_path, adv_wav, ENCODEC_SAMPLE_RATE)
        self.log(f"Saved adversarial audio: {audio_path}")

        # Save original audio
        orig_path = os.path.join(output_dir, f"{base_name}_original.wav")
        orig_wav = result.original_wav.squeeze().cpu().numpy()
        sf.write(orig_path, orig_wav, ENCODEC_SAMPLE_RATE)

        # Save JSON metrics
        json_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(result.to_json(), f, indent=2)
        self.log(f"Saved metrics: {json_path}")

    def save_adversarial_wav(
        self,
        result: AttackResult,
        audio_dir: str,
        filename: str,
    ) -> str:
        """
        Save adversarial WAV to a shared audio directory. Skip if already exists.

        Args:
            result: AttackResult containing adversarial_wav
            audio_dir: Directory for shared adversarial audio files
            filename: Filename without extension (e.g., "jazz_1__nav_turn_left")

        Returns:
            Path to the saved WAV file
        """
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
