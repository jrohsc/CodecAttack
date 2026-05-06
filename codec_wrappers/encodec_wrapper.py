"""
EnCodec wrapper for latent-space adversarial attacks.

Provides encode_to_continuous / decode_from_continuous methods that
operate on the continuous latent representation (before quantization),
enabling gradient-based optimization in latent space.
"""

import torch
import torch.nn as nn
from encodec import EncodecModel


class EnCodecWrapper:
    """
    Wrapper around Meta's EnCodec model that exposes the continuous
    latent space (pre-quantization) for gradient-based attacks.

    Key insight: EnCodec's encoder produces continuous embeddings that are
    then quantized via RVQ. By operating on these continuous embeddings
    directly, we can compute gradients through the decoder.
    """

    def __init__(self, bandwidth: float = 6.0, device: str = "cuda"):
        self.device = device
        self._bandwidth = bandwidth
        self.sample_rate = 24000

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(device)
        self.model.eval()

    def encode_to_continuous(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to continuous latent space (pre-quantization).

        Args:
            wav: Audio tensor [B, 1, T] at 24kHz

        Returns:
            Continuous latent tensor z [B, C, T'] where C=128
        """
        with torch.no_grad():
            # Run encoder to get continuous embeddings
            z = self.model.encoder(wav)
        return z

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from continuous latent space back to audio.

        This path is differentiable when decoder is in train mode.

        Args:
            z: Continuous latent tensor [B, C, T']

        Returns:
            Audio tensor [B, 1, T] at 24kHz
        """
        audio = self.model.decoder(z)
        return audio

    def set_bandwidth(self, bandwidth: float):
        """Change target bandwidth for encoding."""
        self._bandwidth = bandwidth
        self.model.set_target_bandwidth(bandwidth)
