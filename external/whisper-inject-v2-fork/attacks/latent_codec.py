"""
Neural Audio Codec Latent Space Attacks.

Novel attack that perturbs the latent representation of neural audio codecs
(EnCodec, DAC) rather than raw audio waveforms.

Key advantages:
1. Perturbations survive re-encoding/compression
2. More robust to lossy transmission
3. Attacks the native representation of modern audio pipelines

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

from attacks.base import AttackResult


@dataclass
class LatentAttackResult:
    """Result of latent space attack."""

    # Audio tensors
    original_wav: torch.Tensor          # Original audio [1, T]
    adversarial_wav: torch.Tensor       # Perturbed audio [1, T]

    # Latent tensors
    original_latents: torch.Tensor      # Original latent representation
    adversarial_latents: torch.Tensor   # Perturbed latents
    latent_perturbation: torch.Tensor   # Delta in latent space

    # Outputs
    original_output: str                # Model output on original
    adversarial_output: str             # Model output on adversarial
    target_text: str                    # Target we were trying to achieve

    # Metrics
    success: bool                       # Whether target was achieved
    final_loss: float                   # Final loss value
    steps_taken: int                    # Number of attack steps

    snr_db: float                       # SNR in audio domain
    latent_snr_db: float               # SNR in latent domain

    # Robustness metrics
    survives_reencoding: bool          # Attack survives codec re-encoding
    survives_low_bandwidth: bool       # Attack survives low bandwidth encoding

    history: Dict[str, list]           # Training history


class BaseCodec(ABC):
    """Abstract base class for neural audio codecs."""

    @abstractmethod
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent representation."""
        pass

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to audio."""
        pass

    @abstractmethod
    def encode_to_continuous(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode to continuous latents (before quantization)."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Codec's native sample rate."""
        pass


class EnCodecWrapper(BaseCodec):
    """Wrapper for Meta's EnCodec model."""

    def __init__(
        self,
        bandwidth: float = 6.0,
        device: str = "cuda"
    ):
        from encodec import EncodecModel

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self._bandwidth = bandwidth

    @property
    def sample_rate(self) -> int:
        return 24000

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete codes."""
        with torch.no_grad():
            frames = self.model.encode(audio)
        return frames[0][0]  # [n_q, T]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from discrete codes."""
        with torch.no_grad():
            # Reconstruct frame format
            frames = [(codes.unsqueeze(0), None)]
            audio = self.model.decode(frames)
        return audio

    def encode_to_continuous(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode to continuous latents before quantization."""
        return self.model.encoder(audio)

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from continuous latents."""
        return self.model.decoder(z)

    def set_bandwidth(self, bandwidth: float):
        """Change target bandwidth."""
        self.model.set_target_bandwidth(bandwidth)
        self._bandwidth = bandwidth


class DACWrapper(BaseCodec):
    """Wrapper for Descript Audio Codec (DAC)."""

    def __init__(
        self,
        model_type: str = "44khz",
        device: str = "cuda"
    ):
        import dac

        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self._sample_rate = 44100 if model_type == "44khz" else 16000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z, codes, latents, _, _ = self.model.encode(audio)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.model.quantizer.from_codes(codes)
            audio = self.model.decode(z)
        return audio

    def encode_to_continuous(self, audio: torch.Tensor) -> torch.Tensor:
        z, _, _, _, _ = self.model.encode(audio)
        return z

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


class LatentSpaceAttacker:
    """
    Adversarial attack on neural audio codec latent space.

    Instead of perturbing raw audio waveforms (like traditional attacks),
    we perturb the continuous latent representation of neural codecs.
    This makes perturbations more robust to compression/transmission.
    """

    def __init__(
        self,
        codec: BaseCodec,
        target_model,
        eps: float = 1.0,
        alpha: float = 0.05,
        loss_type: Literal["ce", "ctc", "semantic"] = "ce",
        use_adam: bool = True,
        verbose: bool = True
    ):
        """
        Initialize latent space attacker.

        Args:
            codec: Neural audio codec (EnCodec, DAC, etc.)
            target_model: Target ASR/audio model to attack
            eps: Maximum perturbation in latent space (L-inf)
            alpha: Step size / learning rate
            loss_type: Type of loss function
            use_adam: Use Adam optimizer (else SGD)
            verbose: Print progress
        """
        self.codec = codec
        self.target_model = target_model
        self.eps = eps
        self.alpha = alpha
        self.loss_type = loss_type
        self.use_adam = use_adam
        self.verbose = verbose
        self.device = codec.device

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def compute_loss(
        self,
        audio: torch.Tensor,
        target_text: str,
        use_margin: bool = True
    ) -> torch.Tensor:
        """
        Compute loss for target ASR model.
        Uses the target model's compute_loss or compute_margin_loss method.

        Args:
            audio: Audio tensor [1, T] or [1, 1, T]
            target_text: Target text
            use_margin: Use margin loss (better for adversarial attacks)
        """
        # Squeeze to [1, T] format expected by models
        if audio.dim() == 3:
            audio = audio.squeeze(0)

        if use_margin and hasattr(self.target_model, 'compute_margin_loss'):
            return self.target_model.compute_margin_loss(audio, target_text)
        else:
            return self.target_model.compute_loss(audio, target_text)

    def attack(
        self,
        audio: torch.Tensor,
        target_text: str,
        steps: int = 200,
        check_every: int = 20,
        test_robustness: bool = True
    ) -> LatentAttackResult:
        """
        Perform latent space adversarial attack.

        Args:
            audio: Input audio tensor [1, C, T] or [1, T]
            target_text: Target text to force ASR to output
            steps: Number of optimization steps
            check_every: Check transcription every N steps
            test_robustness: Test if attack survives re-encoding

        Returns:
            LatentAttackResult with adversarial audio and metrics
        """
        # Ensure correct shape [1, 1, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # Step 1: Encode to continuous latents
        self.log("Encoding audio to latent space...")
        with torch.no_grad():
            z_original = self.codec.encode_to_continuous(audio)

        self.log(f"Latent shape: {z_original.shape}")

        # Get original output
        with torch.no_grad():
            original_audio_decoded = self.codec.decode_from_continuous(z_original)
            original_output = self.target_model.generate(
                original_audio_decoded.squeeze(0)
            )

        self.log(f"Original output: {original_output}")
        self.log(f"Target: {target_text}")

        # Step 2: Initialize perturbation
        delta = torch.zeros_like(z_original, requires_grad=True)

        if self.use_adam:
            optimizer = torch.optim.Adam([delta], lr=self.alpha)
        else:
            optimizer = torch.optim.SGD([delta], lr=self.alpha)

        # History tracking
        history = {
            "loss": [],
            "outputs": [],
            "steps": []
        }

        best_delta = delta.data.clone()
        best_loss = float('inf')
        success = False

        self.log(f"\nStarting latent space attack")
        self.log(f"eps={self.eps}, alpha={self.alpha}, steps={steps}")
        self.log("-" * 50)

        # Enable training mode for decoder gradients
        self.codec.model.decoder.train()

        for step in range(steps):
            optimizer.zero_grad()

            # Apply perturbation
            z_adv = z_original + delta

            # Decode to audio
            audio_adv = self.codec.decode_from_continuous(z_adv)

            # Compute loss
            loss = self.compute_loss(audio_adv.squeeze(0), target_text)

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            # Track best
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            history["loss"].append(loss_val)

            # Progress
            if (step + 1) % 10 == 0:
                self.log(f"Step {step+1}/{steps} | Loss: {loss_val:.4f}")

            # Check output
            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta
                    audio_test = self.codec.decode_from_continuous(z_test)
                    output = self.target_model.generate(audio_test.squeeze(0))

                history["outputs"].append(output)
                history["steps"].append(step + 1)

                self.log(f"  → Output: {output[:80]}...")

                if target_text.lower() in output.lower():
                    self.log(f"\n✓ Target achieved at step {step+1}!")
                    success = True

        # Final results
        self.codec.model.decoder.eval()

        with torch.no_grad():
            z_final = z_original + best_delta
            audio_final = self.codec.decode_from_continuous(z_final)
            final_output = self.target_model.generate(audio_final.squeeze(0))

        if target_text.lower() in final_output.lower():
            success = True

        # Compute SNRs
        snr_db = self._compute_snr(
            original_audio_decoded.squeeze(),
            audio_final.squeeze()
        )
        latent_snr_db = self._compute_snr(z_original, z_final)

        # Test robustness to re-encoding
        survives_reencoding = False
        survives_low_bandwidth = False

        if test_robustness:
            survives_reencoding, survives_low_bandwidth = self._test_robustness(
                audio_final, target_text
            )

        self.log("\n" + "=" * 50)
        self.log("Attack Complete")
        self.log(f"Final Loss: {best_loss:.4f}")
        self.log(f"Audio SNR: {snr_db:.2f} dB")
        self.log(f"Latent SNR: {latent_snr_db:.2f} dB")
        self.log(f"Success: {success}")
        self.log(f"Survives re-encoding: {survives_reencoding}")
        self.log(f"Survives low bandwidth: {survives_low_bandwidth}")
        self.log(f"Final Output: {final_output}")
        self.log("=" * 50)

        return LatentAttackResult(
            original_wav=original_audio_decoded.detach().cpu().squeeze(0),
            adversarial_wav=audio_final.detach().cpu().squeeze(0),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=z_final.detach().cpu(),
            latent_perturbation=best_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=step + 1,
            snr_db=snr_db,
            latent_snr_db=latent_snr_db,
            survives_reencoding=survives_reencoding,
            survives_low_bandwidth=survives_low_bandwidth,
            history=history
        )

    def _compute_snr(
        self,
        original: torch.Tensor,
        modified: torch.Tensor
    ) -> float:
        """Compute SNR in dB."""
        noise = modified - original
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        return (10 * torch.log10(signal_power / noise_power)).item()

    def _test_robustness(
        self,
        adversarial_audio: torch.Tensor,
        target_text: str
    ) -> Tuple[bool, bool]:
        """
        Test if attack survives re-encoding.

        Returns:
            (survives_same_bandwidth, survives_low_bandwidth)
        """
        survives_same = False
        survives_low = False

        # Test same bandwidth re-encoding
        with torch.no_grad():
            # Full encode-decode cycle
            codes = self.codec.encode(adversarial_audio)
            reencoded = self.codec.decode(codes)
            output = self.target_model.generate(reencoded.squeeze(0))

            if target_text.lower() in output.lower():
                survives_same = True

        # Test low bandwidth (if supported)
        if hasattr(self.codec, 'set_bandwidth'):
            original_bw = self.codec._bandwidth
            self.codec.set_bandwidth(1.5)  # Very low

            with torch.no_grad():
                codes = self.codec.encode(adversarial_audio)
                reencoded = self.codec.decode(codes)
                output = self.target_model.generate(reencoded.squeeze(0))

                if target_text.lower() in output.lower():
                    survives_low = True

            self.codec.set_bandwidth(original_bw)

        return survives_same, survives_low


class LatentCodecPGDAttacker(LatentSpaceAttacker):
    """
    PGD-style attack in latent space (no optimizer, pure gradient steps).
    """

    def attack(
        self,
        audio: torch.Tensor,
        target_text: str,
        steps: int = 200,
        check_every: int = 20,
        random_init: bool = True,
        test_robustness: bool = True
    ) -> LatentAttackResult:
        """PGD attack in latent space."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # Encode
        with torch.no_grad():
            z_original = self.codec.encode_to_continuous(audio)
            original_audio_decoded = self.codec.decode_from_continuous(z_original)
            original_output = self.target_model.generate(
                original_audio_decoded.squeeze(0)
            )

        self.log(f"Original output: {original_output}")
        self.log(f"Target: {target_text}")

        # Initialize perturbation
        if random_init:
            delta = torch.empty_like(z_original).uniform_(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(z_original)
        delta.requires_grad_(True)

        history = {"loss": [], "outputs": [], "steps": []}
        best_delta = delta.data.clone()
        best_loss = float('inf')
        success = False

        self.codec.model.decoder.train()

        for step in range(steps):
            z_adv = z_original + delta
            audio_adv = self.codec.decode_from_continuous(z_adv)

            loss = self.compute_loss(audio_adv.squeeze(0), target_text)
            loss.backward()

            # PGD step
            grad = delta.grad.data
            delta.data = delta.data - self.alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.grad.zero_()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            history["loss"].append(loss_val)

            if (step + 1) % 10 == 0:
                self.log(f"Step {step+1}/{steps} | Loss: {loss_val:.4f}")

            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta
                    audio_test = self.codec.decode_from_continuous(z_test)
                    output = self.target_model.generate(audio_test.squeeze(0))

                history["outputs"].append(output)
                history["steps"].append(step + 1)
                self.log(f"  → Output: {output[:80]}...")

                if target_text.lower() in output.lower():
                    success = True

        # Final
        self.codec.model.decoder.eval()

        with torch.no_grad():
            z_final = z_original + best_delta
            audio_final = self.codec.decode_from_continuous(z_final)
            final_output = self.target_model.generate(audio_final.squeeze(0))

        if target_text.lower() in final_output.lower():
            success = True

        snr_db = self._compute_snr(original_audio_decoded.squeeze(), audio_final.squeeze())
        latent_snr_db = self._compute_snr(z_original, z_final)

        survives_reencoding = False
        survives_low_bandwidth = False
        if test_robustness:
            survives_reencoding, survives_low_bandwidth = self._test_robustness(
                audio_final, target_text
            )

        return LatentAttackResult(
            original_wav=original_audio_decoded.detach().cpu().squeeze(0),
            adversarial_wav=audio_final.detach().cpu().squeeze(0),
            original_latents=z_original.detach().cpu(),
            adversarial_latents=z_final.detach().cpu(),
            latent_perturbation=best_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=step + 1,
            snr_db=snr_db,
            latent_snr_db=latent_snr_db,
            survives_reencoding=survives_reencoding,
            survives_low_bandwidth=survives_low_bandwidth,
            history=history
        )
