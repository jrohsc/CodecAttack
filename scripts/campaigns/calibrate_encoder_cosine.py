"""
Calibrate the Whisper-encoder per-frame cosine scale.

The encoder-transfer attack drives mean per-frame cosine between (carrier+delta)
and the spoken target. A rising cosine is only evidence of a *content* match if
we know the scale. First smoke run showed cos(unrelated_carrier, target) = 0.68
before a single optimization step, so the floor is high and "cos improved" may
mean nothing.

This measures the reference points that make a number readable:

  self      cos(target_i, target_i)            = 1.0 by construction (sanity)
  cross     cos(target_i, target_j), i != j    <- DIFFERENT spoken commands
  carrier   cos(carrier_k, target_i)           <- unrelated benign speech
  noise     cos(target_i + noise, target_i)    <- same content, degraded

If the attack's final cosine lands inside the `cross` distribution, it has not
matched content -- it has only matched the generic structure that any two pieces
of speech share, and the transfer cells will be zero for a reason that has
nothing to do with transfer.

Writes calibration.json next to the target audio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import MODEL_PATHS, WHISPER_INJECT_ROOT, TARGET_SAMPLE_RATE  # noqa: E402
from encoder_transfer_attack import (  # noqa: E402
    CARRIER_DIR, PAIRS_JSON, TARGET_AUDIO_DIR, load_wav_16k,
)

sys.path.insert(0, WHISPER_INJECT_ROOT)

N = 12
DEVICE = "cuda"


def main():
    from models.qwen2_audio import Qwen2AudioModel
    model = Qwen2AudioModel(model_path=MODEL_PATHS["qwen2_audio"],
                            device=DEVICE, dtype=torch.bfloat16)
    tower = model.model.audio_tower

    def embed(wav_16k):
        mel = model.wav_to_mel(wav_16k).to(torch.bfloat16)
        with torch.no_grad():
            return tower(mel).last_hidden_state.float()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()

    pairs = json.load(open(PAIRS_JSON))["pairs"][:N]
    tidx = json.load(open(TARGET_AUDIO_DIR / "index.json"))["targets"]

    targets, carriers, texts = [], [], []
    for p in pairs:
        texts.append(p["target"])
        targets.append(embed(load_wav_16k(Path(tidx[p["target"]]["wav"]), DEVICE)))
        carriers.append(embed(load_wav_16k(CARRIER_DIR / p["carrier"], DEVICE)))
    print(f"embedded {len(targets)} targets + carriers, shape={tuple(targets[0].shape)}")

    self_c = [cos(t, t) for t in targets]
    cross_c = [cos(targets[i], targets[j])
               for i in range(len(targets)) for j in range(len(targets)) if i != j]
    carrier_c = [cos(carriers[i], targets[i]) for i in range(len(targets))]

    noise_c = []
    for p, t_emb in zip(pairs, targets):
        w = load_wav_16k(Path(tidx[p["target"]]["wav"]), DEVICE)
        for snr in (20.0, 10.0):
            sig = (w ** 2).mean()
            n = torch.randn_like(w) * torch.sqrt(sig / (10 ** (snr / 10)))
            noise_c.append((snr, cos(embed(w + n), t_emb)))

    def stats(xs):
        a = np.asarray(xs, dtype=float)
        return {"n": int(a.size), "mean": round(float(a.mean()), 4),
                "std": round(float(a.std()), 4), "min": round(float(a.min()), 4),
                "max": round(float(a.max()), 4)}

    out = {
        "n_clips": len(targets),
        "encoder_frames": int(targets[0].shape[1]),
        "self_identical": stats(self_c),
        "cross_different_targets": stats(cross_c),
        "carrier_vs_target": stats(carrier_c),
        "target_plus_noise_snr20": stats([c for s, c in noise_c if s == 20.0]),
        "target_plus_noise_snr10": stats([c for s, c in noise_c if s == 10.0]),
    }
    print(json.dumps(out, indent=1))

    hi = out["cross_different_targets"]["max"]
    print(f"\nREADABILITY GATE: an attack cosine must clear {hi:.4f} "
          f"(max over unrelated spoken commands) before it is evidence of "
          f"content match rather than generic speech structure.")

    json.dump(out, open(TARGET_AUDIO_DIR / "calibration.json", "w"), indent=1)
    print(f"wrote {TARGET_AUDIO_DIR/'calibration.json'}")


if __name__ == "__main__":
    main()
