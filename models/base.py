"""Abstract base class for audio models."""

from abc import ABC, abstractmethod
import torch


class BaseAudioModel(ABC):
    """
    Abstract interface for audio-to-text models.

    Required methods:
    - compute_loss(): Differentiable loss for target text given audio
    - generate(): Generate text from audio
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        pass

    @abstractmethod
    def generate(self, wav: torch.Tensor, **kwargs) -> str:
        pass

    @abstractmethod
    def compute_loss(self, wav: torch.Tensor, target_text: str, **kwargs) -> torch.Tensor:
        pass
