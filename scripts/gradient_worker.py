#!/usr/bin/env python3
"""
Subprocess gradient worker for ensemble adversarial attacks.

Loads a single Audio LLM model, computes loss.backward() on an input
audio tensor, and saves the gradient + loss value to disk.

This script is designed to be invoked in any conda environment,
allowing ensemble attacks to span models with incompatible dependencies
(e.g. Audio Flamingo 3 requires transformers >= 5.0).

Usage:
    conda activate flamingo3
    python gradient_worker.py --input /tmp/audio_16k.pt --output /tmp/gradient.pt

Input file format (torch.save):
    {
        "audio_16k": Tensor [1, T] at 16kHz,
        "target_text": str,
        "prompt": str,
        "model_name": str,
    }

Output file format (torch.save):
    {
        "gradient": Tensor [1, T] (same shape as audio_16k),
        "loss": float,
    }
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Compute model gradient for ensemble attack")
    parser.add_argument("--input", required=True, help="Path to input .pt file")
    parser.add_argument("--output", required=True, help="Path to write output .pt file")
    args = parser.parse_args()

    # Load input
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    audio_16k = data["audio_16k"]
    target_text = data["target_text"]
    prompt = data["prompt"] if data["prompt"] else None
    model_name = data["model_name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    from model_registry import load_model as _registry_load
    model = _registry_load(model_name, device=device, dtype=dtype)

    # Freeze parameters: this worker returns d(loss)/d(audio) only, so the
    # .grad buffers backward() would accumulate on every weight are dead memory
    # the size of the model itself. See the note in ensemble_attack.py.
    _hf = getattr(model, "model", None)
    if _hf is not None:
        for _prm in _hf.parameters():
            _prm.requires_grad_(False)
            _prm.grad = None
    # Compute gradient
    audio_input = audio_16k.to(device).requires_grad_(True)

    if hasattr(model, 'reset_cache'):
        model.reset_cache()

    loss = model.compute_loss(audio_input, target_text, prompt=prompt)
    loss.backward()

    gradient = audio_input.grad.detach().cpu()
    loss_val = loss.item()

    # Save output
    torch.save({
        "gradient": gradient,
        "loss": loss_val,
    }, args.output)


if __name__ == "__main__":
    main()
