"""
EnCodec wrapper for latent-space adversarial attacks.

Exposes the continuous latent representation (pre-quantization) for
gradient-based optimization.
"""

import torch
from encodec import EncodecModel


class EnCodecWrapper:
    """
    Wrapper around EnCodec that exposes continuous latent space
    (pre-quantization) for gradient-based attacks.
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
        """Encode audio to continuous latent space [B, 128, T']."""
        with torch.no_grad():
            z = self.model.encoder(wav)
        return z

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from continuous latents to audio [B, 1, T]. Differentiable when decoder.train()."""
        return self.model.decoder(z)

    def set_bandwidth(self, bandwidth: float):
        self._bandwidth = bandwidth
        self.model.set_target_bandwidth(bandwidth)
