#!/usr/bin/env python3
"""
PERSISTENT subprocess gradient worker for ensemble adversarial attacks.

Why this exists: `gradient_worker.py` is invoked once per optimization step, so it
reloads the model from disk every step. Audio Flamingo 3 takes ~46 s to load, which
makes a 300-step x 180-clip ensemble run ~600 GPU-hours -- infeasible. Worse, the
caller's `subprocess.run(..., timeout=120)` returns (None, 0.0) on timeout and the
attack loop silently rescales the remaining weights, so the held-out model would
contribute NOTHING while the log looked healthy.

This version loads the model ONCE and then serves gradient requests over a simple
line protocol, so the per-step cost is a forward+backward (~0.3-0.5 s).

Protocol (line-based on stdin/stdout, tensors go via files to avoid pipe framing):
    worker -> "READY\\n"                        once the model is resident
    parent -> "<path-to-input.pt>\\n"
    worker -> "OK <loss>\\n"                    gradient written to "<path>.out"
           or "ERR <message>\\n"                parent decides what to do
    parent -> "QUIT\\n"                         clean shutdown

Input .pt : {"audio_16k": Tensor[1,T], "target_text": str, "prompt": str}
Output .pt: {"gradient": Tensor[1,T] (cpu), "loss": float}

Usage (run inside the model's own conda env):
    conda activate flamingo3
    python -u gradient_worker_persistent.py --model audio_flamingo
"""

import argparse
import os
import sys

import torch


def load_model(model_name, device, dtype):
    """Delegates to model_registry (single source of truth for name -> wrapper)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_registry import load_model as _registry_load
    return _registry_load(model_name, device=device, dtype=dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device if torch.cuda.is_available() else "cpu"

    # The model wrappers print to stdout ("Loading Audio Flamingo 3 model from:
    # ..."), which corrupts the line protocol -- the parent's first readline()
    # gets that banner instead of READY. Claim the real stdout for protocol use
    # only and send every other write to stderr.
    proto = sys.stdout
    sys.stdout = sys.stderr

    def reply(msg):
        proto.write(msg + "\n")
        proto.flush()

    print(f"[worker] loading {args.model} on {device}...", file=sys.stderr, flush=True)
    model = load_model(args.model, device, dtype)
    # Freeze parameters: this worker returns d(loss)/d(audio) only, so the
    # .grad buffers backward() would accumulate on every weight are dead memory
    # the size of the model itself. See the note in ensemble_attack.py.
    _hf = getattr(model, "model", None)
    if _hf is not None:
        for _prm in _hf.parameters():
            _prm.requires_grad_(False)
            _prm.grad = None
    print(f"[worker] {args.model} resident", file=sys.stderr, flush=True)

    reply("READY")

    for line in sys.stdin:
        req = line.strip()
        if not req:
            continue
        if req == "QUIT":
            print("[worker] shutting down", file=sys.stderr, flush=True)
            break
        try:
            # weights_only=True: payload is only tensors + str, so there is no
            # reason to allow arbitrary unpickling here.
            data = torch.load(req, map_location="cpu", weights_only=True)
            audio_16k = data["audio_16k"]
            target_text = data["target_text"]
            prompt = data["prompt"] or None

            audio_input = audio_16k.to(device).detach().requires_grad_(True)
            if hasattr(model, "reset_cache"):
                model.reset_cache()

            loss = model.compute_loss(audio_input, target_text, prompt=prompt)
            loss.backward()

            loss_val = float(loss.item())
            torch.save({"gradient": audio_input.grad.detach().cpu(),
                        "loss": loss_val}, req + ".out")
            # Drop the graph before the next request.
            del loss, audio_input
            reply(f"OK {loss_val}")
        except Exception as e:  # noqa: BLE001 - report, never die on one bad request
            reply(f"ERR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
