"""SNAC v0.5 — attack-only resume.

Reuses constants + helpers from run_snac_v05.py. For every pair index in
PAIRS whose attacked WAV does NOT already exist in OUT_DIR/audio/, runs the
PGD attack and saves it. No routing/eval here — that's done separately by
_snac_v05_eval_only.py over the 15-channel grid.
"""
import sys
SCRIPTS_DIR = "${REPO_ROOT}/0_all_combined/scripts"
sys.path.insert(0, SCRIPTS_DIR)

from run_snac_v05 import (
    PAIRS, EPS_SNAC, ATTACK_STEPS, ALPHA, DURATION_S, SNAC_SR, DEVICE,
    MODEL_PATH, OUT_DIR,
    lookup_pair, load_carrier, pgd_attack,
)

import time
import traceback
from pathlib import Path

import torch
import soundfile as sf

# Imports below resolve because run_snac_v05.py inserted PROJ_FORK on import
from attacks.latent_codec_snac import SNACWrapper
from models.qwen25_omni import Qwen25OmniModel


def main() -> None:
    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: figure out which pairs are missing
    missing = []
    targets = {}
    for pair_idx in PAIRS:
        try:
            summary = lookup_pair(pair_idx)
        except Exception as e:
            print(f"  ! pair{pair_idx:03d} lookup failed: {e}", flush=True)
            continue
        carrier_path = Path(summary["wav_path"])
        carrier_stem = carrier_path.stem
        attacked_wav = audio_dir / f"music_{carrier_stem}__pair{pair_idx:03d}_attacked.wav"
        if attacked_wav.exists():
            continue
        missing.append(pair_idx)
        targets[pair_idx] = (carrier_path, carrier_stem, summary["target_text"], attacked_wav)

    print("=" * 70, flush=True)
    print(f"SNAC v0.5 — attack resume", flush=True)
    print(f"  total pairs:    {len(PAIRS)}", flush=True)
    print(f"  already done:   {len(PAIRS) - len(missing)}", flush=True)
    print(f"  to attack:      {len(missing)}  -> {missing}", flush=True)
    print(f"  eps:            {EPS_SNAC}", flush=True)
    print(f"  steps:          {ATTACK_STEPS}", flush=True)
    print(f"  out:            {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    if not missing:
        print("[resume] nothing to do — all 45 pairs already attacked.", flush=True)
        return

    print("\n[init] Loading SNAC codec...", flush=True)
    codec = SNACWrapper(device=DEVICE)
    codec.model.decoder.train()
    print(f"  SNAC: sr={codec.sample_rate}  latent_dim={codec.latent_dim}  "
          f"hop={codec.hop_length}", flush=True)

    print("[init] Loading Qwen2.5-Omni-7B...", flush=True)
    model = Qwen25OmniModel(model_path=MODEL_PATH, device=DEVICE, dtype=torch.bfloat16)

    t_total0 = time.time()
    for pair_idx in missing:
        carrier_path, carrier_stem, target_text, attacked_wav = targets[pair_idx]
        print(f"\n{'-'*70}", flush=True)
        print(f"[attack] pair {pair_idx:03d}", flush=True)
        print(f"  carrier:     {carrier_stem}", flush=True)
        print(f"  target_text: {target_text!r}", flush=True)
        try:
            audio_24k = load_carrier(carrier_path, sr=SNAC_SR, duration_s=DURATION_S)
            print(f"  carrier shape: {tuple(audio_24k.shape)}", flush=True)
            audio_final = pgd_attack(
                codec, model, audio_24k, target_text,
                eps=EPS_SNAC, steps=ATTACK_STEPS, alpha=ALPHA,
            )
            sf.write(str(attacked_wav), audio_final.cpu().squeeze().numpy(), SNAC_SR)
            print(f"  saved: {attacked_wav.name}", flush=True)
        except Exception as e:
            print(f"  ERROR pair{pair_idx:03d}: {e}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    elapsed = time.time() - t_total0
    print(f"\n[resume done] {len(missing)} pairs attacked in {elapsed:.0f}s "
          f"(~{elapsed/max(1,len(missing)):.0f}s/pair)", flush=True)


if __name__ == "__main__":
    main()
