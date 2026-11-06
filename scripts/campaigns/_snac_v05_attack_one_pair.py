"""Attack a single SNAC v0.5 pair. Invoked as a subprocess so the parent
sheds all GPU memory between pairs — works around the 39.5/40 GB headroom
issue we hit when running pairs in a single Python process."""
import sys
import time
import traceback
from pathlib import Path

SCRIPTS_DIR = "${REPO_ROOT}/0_all_combined/scripts"
sys.path.insert(0, SCRIPTS_DIR)

from run_snac_v05 import (
    EPS_SNAC, ATTACK_STEPS, ALPHA, DURATION_S, SNAC_SR, DEVICE, MODEL_PATH, OUT_DIR,
    lookup_pair, load_carrier, pgd_attack,
)

import torch
import soundfile as sf

# These resolve because run_snac_v05.py inserted the whisper-inject-v2-fork path
from attacks.latent_codec_snac import SNACWrapper
from models.qwen25_omni import Qwen25OmniModel


def main(pair_idx: int) -> int:
    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    summary = lookup_pair(pair_idx)
    carrier_path = Path(summary["wav_path"])
    target_text = summary["target_text"]
    carrier_stem = carrier_path.stem
    attacked_wav = audio_dir / f"music_{carrier_stem}__pair{pair_idx:03d}_attacked.wav"

    if attacked_wav.exists():
        print(f"  already done: {attacked_wav.name}", flush=True)
        return 0

    print(f"[attack] pair{pair_idx:03d}  carrier={carrier_stem}", flush=True)
    print(f"  target: {target_text!r}", flush=True)

    print("[init] SNAC...", flush=True)
    codec = SNACWrapper(device=DEVICE)
    codec.model.decoder.train()
    print("[init] Qwen2.5-Omni-7B...", flush=True)
    model = Qwen25OmniModel(model_path=MODEL_PATH, device=DEVICE, dtype=torch.bfloat16)

    # The wrapper enables gradient_checkpointing but leaves the model in
    # inference mode. HuggingFace's checkpointed forward only fires when
    # training=True, so on a 40 GB A100 the activation graph still OOMs at
    # 15 s audio with the 7B thinker. Force training mode on the thinker so
    # checkpointing triggers, then zero every Dropout's p so the forward
    # stays deterministic across PGD steps.
    import torch.nn as nn
    model.thinker.train()
    n_drops = 0
    for m in model.thinker.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            n_drops += 1
    print(f"  thinker -> train mode  (dropouts zeroed: {n_drops})", flush=True)

    audio_24k = load_carrier(carrier_path, sr=SNAC_SR, duration_s=DURATION_S)
    print(f"  carrier shape: {tuple(audio_24k.shape)}", flush=True)

    t0 = time.time()
    try:
        audio_final = pgd_attack(
            codec, model, audio_24k, target_text,
            eps=EPS_SNAC, steps=ATTACK_STEPS, alpha=ALPHA,
        )
    except Exception as e:
        print(f"  ERROR pgd: {e}", flush=True)
        traceback.print_exc()
        return 2

    sf.write(str(attacked_wav), audio_final.cpu().squeeze().numpy(), SNAC_SR)
    print(f"  saved: {attacked_wav.name}  pgd_total={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: _snac_v05_attack_one_pair.py <pair_idx>", file=sys.stderr)
        sys.exit(64)
    sys.exit(main(int(sys.argv[1])))
