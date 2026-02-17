"""
Latent-space adversarial attack on Audio LLMs via EnCodec.

Gradient flow:
    delta (learnable) -> z_adv = z + delta
    -> EnCodec.decoder(z_adv) -> audio_24kHz
    -> resample(24k -> 16k)
    -> AudioLLM.compute_loss(audio_16kHz, target_text)
    -> loss.backward() -> gradients flow to delta
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torchaudio

from attacks.latent_codec import EnCodecWrapper

ENCODEC_SR = 24000
TARGET_SR = 16000


@dataclass
class AttackResult:
    """Result of a latent-space attack."""
    original_wav: Optional[torch.Tensor] = None
    adversarial_wav: Optional[torch.Tensor] = None
    original_output: str = ""
    adversarial_output: str = ""
    target_text: str = ""
    success: bool = False
    final_loss: float = float('inf')
    steps_taken: int = 0
    snr_db: float = 0.0
    history: Dict[str, list] = field(default_factory=dict)
    duration_s: float = 0.0


def mel_distance(audio_a, audio_b, sample_rate=ENCODEC_SR, n_mels=80, n_fft=1024, hop_length=256):
    """Differentiable mel-spectrogram L2 distance (perceptual loss)."""
    if audio_a.dim() == 3:
        audio_a = audio_a.squeeze(0)
    if audio_b.dim() == 3:
        audio_b = audio_b.squeeze(0)
    min_len = min(audio_a.shape[-1], audio_b.shape[-1])
    audio_a, audio_b = audio_a[..., :min_len], audio_b[..., :min_len]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0,
    ).to(audio_a.device)

    mel_a = torch.log1p(mel_transform(audio_a))
    mel_b = torch.log1p(mel_transform(audio_b))
    return torch.nn.functional.mse_loss(mel_a, mel_b)


class LatentCodecAttacker:
    """
    Adversarial attack on Audio LLMs via EnCodec latent-space perturbations.

    1. Encode music carrier to EnCodec continuous latent space
    2. Optimize perturbation delta in latent space (Adam + projection)
    3. Decode perturbed latents back to audio
    4. Evaluate against target Audio LLM
    """

    def __init__(
        self,
        target_model,
        eps: float = 0.5,
        alpha: float = 0.2,
        perceptual_weight: float = 0.1,
        device: str = "cuda",
    ):
        self.eps = eps
        self.alpha = alpha
        self.perceptual_weight = perceptual_weight
        self.device = device

        self.codec = EnCodecWrapper(bandwidth=6.0, device=device)
        self.target_model = target_model

    def attack(
        self,
        music_wav: torch.Tensor,
        target_text: str,
        steps: int = 150,
        check_every: int = 25,
        prompt: str = None,
    ) -> AttackResult:
        """
        Run latent-space adversarial attack.

        Args:
            music_wav: Music carrier [1, 1, T] at 24kHz
            target_text: Text to force the model to output
            steps: Optimization steps
            check_every: Check model output every N steps
            prompt: Text prompt for the model

        Returns:
            AttackResult with adversarial audio and metrics
        """
        start_time = time.time()
        music_wav = music_wav.to(self.device)

        # Encode music to latent space
        with torch.no_grad():
            z_original = self.codec.encode_to_continuous(music_wav)
            original_audio_24k = self.codec.decode_from_continuous(z_original)
            original_audio_16k = torchaudio.functional.resample(
                original_audio_24k.squeeze(0), ENCODEC_SR, TARGET_SR
            )
            original_output = self.target_model.generate(original_audio_16k, prompt=prompt)

        print(f"Original output: {original_output}")
        print(f"Target: {target_text}")
        print(f"Latent shape: {z_original.shape}")
        print(f"eps={self.eps}, lr={self.alpha}, steps={steps}")
        print("-" * 60)

        # Learnable perturbation
        delta = torch.zeros_like(z_original, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)

        history = {"loss": [], "behavior_loss": [], "perceptual_loss": [], "outputs": []}
        best_delta = delta.data.clone()
        best_loss = float('inf')

        # Enable gradients through EnCodec decoder
        self.codec.model.decoder.train()

        for step in range(steps):
            optimizer.zero_grad()

            z_adv = z_original + delta
            audio_adv_24k = self.codec.decode_from_continuous(z_adv)
            audio_adv_16k = torchaudio.functional.resample(
                audio_adv_24k.squeeze(0), ENCODEC_SR, TARGET_SR
            )

            behavior_loss = self.target_model.compute_loss(audio_adv_16k, target_text, prompt=prompt)

            if self.perceptual_weight > 0:
                p_loss = mel_distance(audio_adv_24k.squeeze(0), original_audio_24k.squeeze(0).detach())
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            total_loss = behavior_loss + self.perceptual_weight * p_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)

            loss_val = total_loss.item()
            history["loss"].append(loss_val)
            history["behavior_loss"].append(behavior_loss.item())
            history["perceptual_loss"].append(p_loss.item())

            if loss_val < best_loss:
                best_loss = loss_val
                best_delta = delta.data.clone()

            if (step + 1) % 10 == 0:
                print(f"Step {step+1:4d}/{steps} | Loss: {loss_val:.4f} | "
                      f"Behavior: {behavior_loss.item():.4f} | Perceptual: {p_loss.item():.4f}")

            if check_every > 0 and (step + 1) % check_every == 0:
                with torch.no_grad():
                    z_test = z_original + delta
                    audio_test = self.codec.decode_from_continuous(z_test)
                    audio_test_16k = torchaudio.functional.resample(audio_test.squeeze(0), ENCODEC_SR, TARGET_SR)
                    output = self.target_model.generate(audio_test_16k, prompt=prompt)
                history["outputs"].append(output)
                print(f"  -> Output: {output[:120]}")
                if target_text.lower() in output.lower():
                    print(f"  -> TARGET ACHIEVED at step {step+1}!")

        self.codec.model.decoder.eval()

        # Final evaluation with best delta
        with torch.no_grad():
            z_final = z_original + best_delta
            audio_final_24k = self.codec.decode_from_continuous(z_final)
            audio_final_16k = torchaudio.functional.resample(
                audio_final_24k.squeeze(0), ENCODEC_SR, TARGET_SR
            )
            final_output = self.target_model.generate(audio_final_16k, prompt=prompt)

        success = target_text.lower() in final_output.lower()

        # SNR
        noise = audio_final_24k.squeeze() - original_audio_24k.squeeze()
        signal_power = torch.mean(original_audio_24k.squeeze() ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_db = (10 * torch.log10(signal_power / (noise_power + 1e-10))).item()

        duration = time.time() - start_time

        print("\n" + "=" * 60)
        print(f"Attack {'SUCCEEDED' if success else 'FAILED'}")
        print(f"Final output: {final_output}")
        print(f"SNR: {snr_db:.1f} dB | Duration: {duration:.1f}s")
        print("=" * 60)

        return AttackResult(
            original_wav=original_audio_24k.detach().cpu().squeeze(0),
            adversarial_wav=audio_final_24k.detach().cpu().squeeze(0),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=best_loss,
            steps_taken=steps,
            snr_db=snr_db,
            history=history,
            duration_s=duration,
        )
