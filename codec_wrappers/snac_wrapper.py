"""
SNAC (Multi-Scale Neural Audio Codec) wrapper for latent-space adversarial attacks.

Mirrors EnCodecWrapper / DACWrapper. SNAC's RVQ produces codes at three temporal
scales, but the encoder produces a single continuous tensor (pre-VQ). The continuous
bypass skips the quantizer entirely, so the wrapper interface is identical to DAC.

Reference: external/codecattack_lib/attacks/latent_codec_dac.py
"""

import torch
from snac import SNAC


class SNACWrapper:
    """
    Wrapper around hubertsiuzdak/snac_24khz exposing the continuous pre-VQ
    latent space for gradient-based attacks.
    """

    def __init__(self, repo_id: str = "hubertsiuzdak/snac_24khz", device: str = "cuda"):
        self.device = device
        self.model = SNAC.from_pretrained(repo_id).to(device)
        self.model.train(False)
        self.sample_rate = self.model.sampling_rate  # 24000 for snac_24khz
        self.latent_dim = self.model.latent_dim       # 768 for snac_24khz
        self.hop_length = int(self.model.hop_length)  # 512 for snac_24khz

    def encode_to_continuous(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to continuous pre-VQ latent.

        Args:
            wav: [B, 1, T] at self.sample_rate

        Returns:
            z: [B, D, T_z] where D=latent_dim, T_z = T_padded / hop_length
        """
        wav_pad = self.model.preprocess(wav)
        with torch.no_grad():
            z = self.model.encoder(wav_pad)
        return z

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode audio from continuous latent. Differentiable when self.model.decoder
        is in train() mode.

        Args:
            z: [B, D, T_z]

        Returns:
            wav: [B, 1, T_padded] at self.sample_rate
        """
        return self.model.decoder(z)
