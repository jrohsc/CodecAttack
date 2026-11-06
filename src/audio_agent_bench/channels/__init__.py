"""Eval-time channel wrappers. Non-differentiable — inference only.

All wrappers accept a wav file path and return a path to the transformed wav.
Composable via `chain.apply_chain([...])`.
"""
from .base import Channel, apply_channel

__all__ = ["Channel", "apply_channel", "get_channel"]


def get_channel(kind: str, **params) -> Channel:
    if kind == "clean":
        from .clean import CleanChannel
        return CleanChannel()
    if kind == "opus":
        from .opus import OpusChannel
        return OpusChannel(**params)
    if kind == "g711_mulaw":
        from .g711 import G711MulawChannel
        return G711MulawChannel(**params)
    if kind == "g722":
        from .g722 import G722Channel
        return G722Channel(**params)
    if kind == "aac":
        from .aac import AACChannel
        return AACChannel(**params)
    if kind == "mp3":
        from .mp3 import MP3Channel
        return MP3Channel(**params)
    if kind == "packet_loss":
        from .packet_loss import PacketLossChannel
        return PacketLossChannel(**params)
    if kind == "physical_sim":
        from .physical_sim import PhysicalSimChannel
        return PhysicalSimChannel(**params)
    if kind == "chain":
        from .chain import ChainChannel
        return ChainChannel(**params)
    raise ValueError(f"unknown channel: {kind}")
