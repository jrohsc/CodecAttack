"""
Mimi codec wrapper for latent-space adversarial attacks on Moshi-based models.

Provides encode_to_continuous / decode_from_continuous methods that operate
on the continuous latent representation (pre-quantization) in Mimi's VQ space,
enabling gradient-based optimization in latent space.

Mimi architecture (encoding path):
    audio [B, 1, T] @ 24kHz
    → encoder (Conv1d) → [B, 512, T']
    → encoder_transformer → [B, 512, T']
    → downsample → [B, 512, T'']
    → input_proj (Conv1d 512→256) → [B, 256, T''] (VQ space)
    → RVQ quantize → codes [B, K, T'']

Decoding path:
    codes → RVQ decode → [B, 256, T'']
    → output_proj (Conv1d 256→512) → [B, 512, T'']
    → upsample → [B, 512, T']
    → decoder_transformer → [B, 512, T']
    → decoder (ConvTranspose1d) → [B, 1, T] @ 24kHz
"""

import torch
import torch.nn as nn
from transformers import MimiModel


class MimiCodecWrapper:
    """
    Wrapper around HuggingFace's MimiModel that exposes the continuous
    latent space (pre-quantization, post-input_proj) for gradient-based attacks.

    The VQ space has dimension 256, operating at 12.5 fps frame rate.
    """

    def __init__(
        self,
        model: MimiModel = None,
        model_path: str = "kyutai/mimi",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model: Pre-loaded MimiModel instance (e.g. from MoshiForConditionalGeneration.audio_encoder).
                   If provided, model_path is ignored.
            model_path: HuggingFace model path to load Mimi from (default: kyutai/mimi).
            device: Device to use.
            dtype: Data type for the model.
        """
        self.device = device
        self._dtype = dtype
        self.sample_rate = 24000
        self.frame_rate = 12.5

        if model is not None:
            self.model = model
        else:
            self.model = MimiModel.from_pretrained(model_path).to(device=device, dtype=dtype)
        self.model.eval()

        # Detect model weight dtype (may differ from self._dtype if model was loaded in bf16)
        self._model_dtype = next(self.model.parameters()).dtype

        # Key dimensions
        self.hidden_size = self.model.config.hidden_size  # 512
        self.vq_dim = self.model.config.vector_quantization_hidden_dimension  # 256
        self.codebook_size = self.model.config.codebook_size  # 2048
        self.num_quantizers = self.model.config.num_quantizers  # 32

    def encode_to_continuous(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to continuous latent space (pre-downsampled encoder output).

        Returns the encoder output BEFORE quantization. The input_proj
        (512→256) that maps into VQ space lives inside each sub-RVQ and is
        applied during soft-RVQ in PersonaPLEXModel.compute_loss_from_latents.

        Args:
            wav: Audio tensor [B, 1, T] at 24kHz

        Returns:
            Continuous latent tensor z [B, 512, T'] (encoder hidden space)
        """
        with torch.no_grad():
            # Cast to model weight dtype (e.g. bf16 if loaded as part of bf16 Moshi)
            wav = wav.to(self.device, dtype=self._model_dtype)

            # 1. Conv encoder
            embeddings = self.model.encoder(wav)  # [B, 512, T']

            # 2. Encoder transformer
            encoder_out = self.model.encoder_transformer(
                embeddings.transpose(1, 2)  # [B, T', 512]
            )
            embeddings = encoder_out[0].transpose(1, 2)  # [B, 512, T']

            # 3. Downsample
            embeddings = self.model.downsample(embeddings)  # [B, 512, T'']

        return embeddings

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from continuous encoder latent space back to audio.

        Applies upsample → decoder_transformer → decoder.
        Mirrors the Mimi _decode_frame path (without quantization).
        This path is differentiable when decoder components are in train mode.

        Args:
            z: Continuous latent tensor [B, 512, T''] (encoder hidden space)

        Returns:
            Audio tensor [B, 1, T] at 24kHz
        """
        # Cast to model dtype for decoder path
        z = z.to(self._model_dtype)

        # 1. Upsample
        embeddings = self.model.upsample(z)  # [B, 512, T']

        # 2. Decoder transformer
        decoder_out = self.model.decoder_transformer(
            embeddings.transpose(1, 2)  # [B, T', 512]
        )
        embeddings = decoder_out[0].transpose(1, 2)  # [B, 512, T']

        # 3. Conv decoder
        audio = self.model.decoder(embeddings)  # [B, 1, T]

        return audio

    def encode_to_codes(
        self, wav: torch.Tensor, num_quantizers: int = 8
    ) -> torch.Tensor:
        """
        Standard encode: audio → discrete codes.

        Args:
            wav: Audio tensor [B, 1, T] at 24kHz
            num_quantizers: Number of codebooks to use (default: 8 for Moshi)

        Returns:
            Codes tensor [B, num_quantizers, T']
        """
        with torch.no_grad():
            wav = wav.to(self.device, dtype=self._model_dtype)
            output = self.model.encode(wav, num_quantizers=num_quantizers)
            # output is MimiEncoderOutput or tuple; codes are first element
            if hasattr(output, 'audio_codes'):
                codes = output.audio_codes
            else:
                codes = output[0]
        return codes

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Standard decode: codes → audio.

        Args:
            codes: Codes tensor [B, K, T']

        Returns:
            Audio tensor [B, 1, T] at 24kHz
        """
        with torch.no_grad():
            output = self.model.decode(codes)
            if hasattr(output, 'audio_values'):
                audio = output.audio_values
            else:
                audio = output[0]
        return audio

    def get_codebook_vectors(self, num_codebooks: int = 8):
        """
        Extract codebook embedding vectors for soft-RVQ approximation.

        Returns list of codebook tensors, one per codebook layer.
        Codebook 0 is semantic, codebooks 1..K-1 are acoustic.

        Args:
            num_codebooks: Number of codebooks to extract (default: 8 for Moshi)

        Returns:
            List of tensors, each [codebook_size, codebook_dim] = [2048, 256]
        """
        codebooks = []
        quantizer = self.model.quantizer

        # Codebook 0: semantic
        semantic_rvq = quantizer.semantic_residual_vector_quantizer
        for layer in semantic_rvq.layers:
            codebooks.append(layer.codebook.embed.detach())
            if len(codebooks) >= num_codebooks:
                return codebooks

        # Codebooks 1+: acoustic
        acoustic_rvq = quantizer.acoustic_residual_vector_quantizer
        for layer in acoustic_rvq.layers:
            codebooks.append(layer.codebook.embed.detach())
            if len(codebooks) >= num_codebooks:
                return codebooks

        return codebooks

    def set_decode_train_mode(self, train: bool = True):
        """Enable/disable train mode on decoder components for gradient flow."""
        if train:
            self.model.upsample.train()
            self.model.decoder_transformer.train()
            self.model.decoder.train()
        else:
            self.model.upsample.eval()
            self.model.decoder_transformer.eval()
            self.model.decoder.eval()
