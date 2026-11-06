"""SNAC EoT attack — single pair, subprocess-friendly.

Mirrors run_snac_v05.py / _snac_v05_attack_one_pair.py but inserts a
DifferentiableOpusProxy (STE/BPDA) between SNAC.decode_from_continuous and
the 16 kHz resample, so each PGD step trains against a randomly-sampled
Opus bitrate from EOT_BITRATES — directly comparable to MIMI eot_full.

Usage: _snac_eot_attack_one_pair.py <pair_idx> <out_subdir>
  out_subdir is appended under .../results_SNAC/03b_copyright_bypass/qwen25_omni/
"""
import sys
import time
import traceback
from pathlib import Path

SCRIPTS_DIR = "${REPO_ROOT}/0_all_combined/scripts"
sys.path.insert(0, SCRIPTS_DIR)

from run_snac_v05 import (
    EPS_SNAC, ATTACK_STEPS, ALPHA, DURATION_S, SNAC_SR, TARGET_SR, DEVICE,
    MODEL_PATH,
    lookup_pair, load_carrier,
)

import torch
import torchaudio
import torch.nn as nn
import soundfile as sf

# Resolves via run_snac_v05.py's PROJ_FORK insertion
from attacks.latent_codec_snac import SNACWrapper
from models.qwen25_omni import Qwen25OmniModel

# Local imports (codec_attack on path via run_snac_v05's parent inserts? No —
# need to add codec_attack ourselves)
sys.path.insert(0, "${REPO_ROOT}")
from channel_augmentation import DifferentiableOpusProxy

OUT_BASE = Path(
    "${REPO_ROOT}/"
    "0_all_combined/results_OTHER_NEURAL_CODECS/results_SNAC/"
    "03b_copyright_bypass/qwen25_omni"
)

EOT_BITRATES = [16, 24, 32, 64, 128, 192]


def pgd_attack_eot(codec, model, opus_proxy, audio_24k, target_text,
                   eps, steps, alpha):
    """SNAC PGD with Opus EoT in the loop.

    z_adv = z0 + δ  →  audio_adv (24 kHz)  →  opus_proxy@random_bitrate
    →  resample(16 kHz)  →  Qwen2.5-Omni compute_loss
    """
    n_orig = audio_24k.shape[-1]
    z0 = codec.encode_to_continuous(audio_24k).detach()
    delta = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=alpha)

    losses = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        z_adv = z0 + delta
        audio_adv = codec.decode_from_continuous(z_adv)[..., :n_orig]   # [1,1,T] @ 24k
        audio_eot = opus_proxy(audio_adv)                               # STE Opus
        audio_eot_16k = torchaudio.functional.resample(
            audio_eot.squeeze(0), SNAC_SR, TARGET_SR
        )
        loss = model.compute_loss(audio_eot_16k, target_text, prompt=None)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        z_final = z0 + delta
        audio_final_24k = codec.decode_from_continuous(z_final)[..., :n_orig]
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"(min={min(losses):.4f})  total={time.time()-t0:.1f}s", flush=True)
    return audio_final_24k.detach(), losses


def main(pair_idx: int, out_subdir: str) -> int:
    out_dir = OUT_BASE / out_subdir
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    summary = lookup_pair(pair_idx)
    carrier_path = Path(summary["wav_path"])
    target_text = summary["target_text"]
    carrier_stem = carrier_path.stem
    attacked_wav = audio_dir / f"music_{carrier_stem}__pair{pair_idx:03d}_attacked.wav"

    if attacked_wav.exists():
        print(f"  already done: {attacked_wav.name}", flush=True)
        return 0

    print(f"[attack-eot] pair{pair_idx:03d}  carrier={carrier_stem}", flush=True)
    print(f"  target:    {target_text!r}", flush=True)
    print(f"  eps:       {EPS_SNAC}", flush=True)
    print(f"  steps:     {ATTACK_STEPS}", flush=True)
    print(f"  EoT brates: {EOT_BITRATES}", flush=True)
    print(f"  out:       {out_dir}", flush=True)

    print("[init] SNAC...", flush=True)
    codec = SNACWrapper(device=DEVICE)
    codec.model.decoder.train()

    print("[init] Qwen2.5-Omni-7B...", flush=True)
    model = Qwen25OmniModel(model_path=MODEL_PATH, device=DEVICE, dtype=torch.bfloat16)

    # Memory-saver part 1: HF gradient checkpointing only fires when
    # training=True. Wrapper sits in inference mode, so flip the thinker into
    # train mode and zero any nn.Dropout (deterministic forward).
    model.thinker.train()
    n_drops = 0
    for m in model.thinker.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            n_drops += 1
    print(f"  thinker -> train mode (dropouts zeroed: {n_drops})", flush=True)

    # Memory-saver part 2: Qwen2.5-Omni's audio_tower (Whisper-style encoder)
    # has _supports_gradient_checkpointing=None, so HF's flag doesn't reach it
    # — and its activations are the dominant 30+ GiB hog during backward.
    # Manually wrap each transformer layer's forward with
    # torch.utils.checkpoint.checkpoint to recompute activations on backward.
    import torch.utils.checkpoint as ckpt
    audio_tower = model.thinker.audio_tower
    layers = None
    # Try a few common nesting patterns for the encoder layer list
    for path in ("encoder.layers", "layers", "encoder.encoder.layers"):
        obj = audio_tower
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            layers = obj
            print(f"  audio_tower layers found at .{path}: {len(layers)}", flush=True)
            break
        except AttributeError:
            continue
    if layers is None:
        print(f"  WARN: could not find audio_tower layer list — "
              f"checkpointing the whole audio_tower instead", flush=True)
        orig_at_fwd = audio_tower.forward
        def at_ckpt_forward(*args, **kwargs):
            def f(*a, **kw):
                return orig_at_fwd(*a, **kw)
            return ckpt.checkpoint(f, *args, use_reentrant=False, **kwargs)
        audio_tower.forward = at_ckpt_forward
    else:
        for i, layer in enumerate(layers):
            orig = layer.forward
            def make_ckpt(orig_fn):
                def ckpt_fwd(*args, **kwargs):
                    def f(*a, **kw):
                        return orig_fn(*a, **kw)
                    return ckpt.checkpoint(f, *args, use_reentrant=False, **kwargs)
                return ckpt_fwd
            layer.forward = make_ckpt(orig)
        print(f"  audio_tower: wrapped {len(layers)} layers with checkpointing",
              flush=True)

    print("[init] DifferentiableOpusProxy (STE)...", flush=True)
    opus_proxy = DifferentiableOpusProxy(
        sample_rate=SNAC_SR, bitrates_kbps=EOT_BITRATES
    )

    audio_24k = load_carrier(carrier_path, sr=SNAC_SR, duration_s=DURATION_S)
    print(f"  carrier shape: {tuple(audio_24k.shape)}", flush=True)

    try:
        audio_final, losses = pgd_attack_eot(
            codec, model, opus_proxy, audio_24k, target_text,
            eps=EPS_SNAC, steps=ATTACK_STEPS, alpha=ALPHA,
        )
    except Exception as e:
        print(f"  ERROR pgd: {e}", flush=True)
        traceback.print_exc()
        return 2

    sf.write(str(attacked_wav), audio_final.cpu().squeeze().numpy(), SNAC_SR)
    print(f"  saved: {attacked_wav.name}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: _snac_eot_attack_one_pair.py <pair_idx> <out_subdir>",
              file=sys.stderr)
        sys.exit(64)
    sys.exit(main(int(sys.argv[1]), sys.argv[2]))
