"""
D1 step 1 — build identical (clean, adv) 16 kHz waveform pairs for the
encoder-displacement audit.

Both members of a pair are produced here, once, and frozen into a single .npz
so that all three model environments (codec-attack / qwen-omni / flamingo3)
see byte-identical inputs. Any difference in measured displacement is then
attributable to the encoder, not to loading.

Clean reference is rebuilt exactly as the attack built it:
    load_music(wav_path, target_sr=24000, duration=carrier_duration,
               normalize=True)   # peak-norm to 0.95
which is what dataset/interview/run_attack.py feeds the attacker.

Usage:
    python scripts/d1_prepare_pairs.py \
        --bundle-dir results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0_seed1 \
        --n 30 --out /path/to/d1_pairs.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # 0_all_combined
CODEC = ROOT.parent         # codec_attack
sys.path.insert(0, str(CODEC))

from music_carrier import load_music  # noqa: E402

ENCODEC_SR = 24000
TARGET_SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True,
                    help="bundle containing audio/*_attacked.wav + *_summary.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    audio_dir = bundle / "audio"

    advs = sorted(audio_dir.glob("*_attacked.wav"))
    if not advs:
        raise SystemExit(f"no *_attacked.wav under {audio_dir}")

    resamp = torchaudio.transforms.Resample(ENCODEC_SR, TARGET_SR)

    clips, clean_list, adv_list, meta = [], [], [], []
    skipped = []

    for adv_path in advs:
        if len(clips) >= args.n:
            break
        stem = adv_path.name[: -len("_attacked.wav")]
        summary_path = audio_dir / f"{stem}_summary.json"
        if not summary_path.exists():
            skipped.append((stem, "no summary.json"))
            continue
        s = json.loads(summary_path.read_text())
        src_wav = s.get("wav_path")
        dur = s.get("carrier_duration_s")
        if not src_wav or not Path(src_wav).exists() or dur is None:
            skipped.append((stem, "missing carrier path/duration"))
            continue

        # clean, exactly as the attacker loaded it
        clean_24k = load_music(src_wav, target_sr=ENCODEC_SR,
                               duration=float(dur), normalize=True)
        clean_24k = clean_24k.squeeze()  # [T]

        adv_np, sr = sf.read(str(adv_path), dtype="float32", always_2d=False)
        if sr != ENCODEC_SR:
            skipped.append((stem, f"adv sr={sr}"))
            continue
        adv_24k = torch.from_numpy(np.asarray(adv_np, dtype=np.float32)).squeeze()

        n = min(clean_24k.shape[-1], adv_24k.shape[-1])
        clean_24k, adv_24k = clean_24k[:n], adv_24k[:n]

        clean_16k = resamp(clean_24k).numpy().astype(np.float32)
        adv_16k = resamp(adv_24k).numpy().astype(np.float32)

        # sanity: perturbation SNR at 24 k, to confirm the pair is really a pair
        delta = (adv_24k - clean_24k).numpy()
        p_sig = float(np.mean(clean_24k.numpy() ** 2))
        p_del = float(np.mean(delta ** 2))
        snr_db = 10.0 * np.log10(p_sig / p_del) if p_del > 0 else float("inf")

        clips.append(stem)
        clean_list.append(clean_16k)
        adv_list.append(adv_16k)
        meta.append({
            "clip": stem,
            "carrier_wav": src_wav,
            "carrier_duration_s": float(dur),
            "target_text": s.get("target_text"),
            "delta_snr_db_24k": snr_db,
        })

    if not clips:
        raise SystemExit("no usable pairs built")

    lens = {len(c) for c in clean_list} | {len(a) for a in adv_list}
    L = min(lens)
    clean_arr = np.stack([c[:L] for c in clean_list])
    adv_arr = np.stack([a[:L] for a in adv_list])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        clean=clean_arr, adv=adv_arr,
        clips=np.array(clips),
        meta=np.array([json.dumps(m) for m in meta]),
        sample_rate=np.array(TARGET_SR),
        bundle=np.array(str(bundle)),
    )

    snrs = [m["delta_snr_db_24k"] for m in meta]
    print(f"[d1-prep] wrote {out}")
    print(f"[d1-prep] n_pairs={len(clips)}  samples={L} @ {TARGET_SR} Hz "
          f"({L / TARGET_SR:.2f} s)")
    print(f"[d1-prep] delta SNR (24 kHz): min {min(snrs):.2f} / "
          f"median {float(np.median(snrs)):.2f} / max {max(snrs):.2f} dB")
    if skipped:
        print(f"[d1-prep] skipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    main()
