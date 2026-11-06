"""
Why the encoder-cosine arms score 0/50 white-box despite cos ~0.87.

Both the spoken target (4.9s) and the adversarial audio (10s) are zero-padded to
30s before the STFT, so the Whisper tower always returns 1500 frames. Only the
first ~16% of the target's frames carry content; the rest is padding. If the
mean per-frame cosine is dominated by pad-vs-pad frames, then "cos 0.87" is
mostly free points and says nothing about content -- and the optimizer's cheapest
remaining move is to *silence* the carrier in the frames where the target is
already silent, which is exactly the transcript degradation we observe.

This script measures the profile instead of assuming it:

  region A  frames [0, n_tgt)          target speech   vs adversarial speech
  region B  frames [n_tgt, n_adv)      target PADDING  vs adversarial speech
  region C  frames [n_adv, 1500)       pad vs pad      (free points)

and reports the masked (region A only) cosine beside the reported global one, for
clean carrier, the attacked wav, and the true target audio round-tripped through
EnCodec (the ceiling this parameterization can reach at all).

Usage: python diagnose_encoder_objective.py [--arm enc_targetinit] [--n 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio.functional as AF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import MODEL_PATHS, WHISPER_INJECT_ROOT, TARGET_SAMPLE_RATE  # noqa: E402
from encoder_transfer_attack import load_wav_16k  # noqa: E402

sys.path.insert(0, WHISPER_INJECT_ROOT)

ENCODEC_SR = 24000
WINDOW_S = 30.0        # every input is zero-padded to this before the STFT

# Frames per window is model-specific and must be measured, not assumed: bare
# Whisper emits 1500 for its 30s window, Qwen2-Audio pools that 2x to 750. An
# assumed 1500 here doubles every region boundary.
FRAMES = None          # set from the first embedding


def frames_for(seconds: float) -> int:
    return int(round(seconds / WINDOW_S * FRAMES))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="enc_targetinit")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ceiling", action="store_true",
                    help="also round-trip the true target through EnCodec "
                         "(loads a second model copy; needs a free card)")
    args = ap.parse_args()

    root = HERE / "0_all_combined/results_encoder_transfer" / args.arm
    summaries = sorted((root / "audio").glob("*_summary.json"))[: args.n]
    if not summaries:
        sys.exit(f"no summaries under {root}")

    from models.qwen2_audio import Qwen2AudioModel
    model = Qwen2AudioModel(model_path=MODEL_PATHS["qwen2_audio"],
                            device=args.device, dtype=torch.bfloat16)
    tower = model.model.audio_tower

    def embed(wav_16k):
        mel = model.wav_to_mel(wav_16k).to(torch.bfloat16)
        with torch.no_grad():
            return tower(mel).last_hidden_state.float()

    def cos_profile(a, b):
        """Per-frame cosine [T]."""
        return torch.nn.functional.cosine_similarity(a, b, dim=-1).squeeze(0)

    # EnCodec round-trip, to measure the ceiling of the latent parameterization.
    # Loaded lazily: LatentCodecAttacker pulls in its own Qwen2-Audio copy, which
    # will OOM if anything else is sharing the card. --ceiling opts in.
    codec = None
    if args.ceiling:
        from latent_attack import LatentCodecAttacker  # noqa
        codec = LatentCodecAttacker(target_model="qwen2_audio", device=args.device,
                                    dtype=torch.bfloat16, verbose=False).codec

    def encodec_roundtrip_16k(wav_16k):
        w24 = AF.resample(wav_16k, TARGET_SAMPLE_RATE, ENCODEC_SR)
        with torch.no_grad():
            z = codec.encode_to_continuous(w24.unsqueeze(0) if w24.dim() == 2 else w24)
            out24 = codec.decode_from_continuous(z)
        return AF.resample(out24.squeeze(0), ENCODEC_SR, TARGET_SAMPLE_RATE)

    print(f"\n{'clip':30s}{'nA':>5s}{'nB':>6s}{'nC':>6s}"
          f"{'globalA':>9s}{'A(spch)':>9s}{'B(pad)':>9s}{'C(pad)':>9s}")
    print("-" * 84)

    global FRAMES
    rows = []
    sil_emb = None
    for sp in summaries:
        s = json.load(open(sp))
        tgt = load_wav_16k(Path(s["target_audio"]), args.device)
        adv = load_wav_16k(HERE / s["attacked_wav"], args.device)

        e_t, e_a = embed(tgt), embed(adv)
        if FRAMES is None:
            FRAMES = e_t.shape[1]
            # Where does target content actually end? Compare the target's frames
            # against an all-silence input: padding frames match it almost
            # exactly. This checks the duration-derived boundary empirically
            # instead of trusting the frame-rate arithmetic.
            sil_emb = embed(torch.zeros_like(tgt))
            sil_prof = cos_profile(e_t, sil_emb)
            thr = 0.99
            beyond = (sil_prof > thr).nonzero().flatten()
            first_pad = int(beyond[0]) if len(beyond) else FRAMES
            print(f"[frames] tower emits {FRAMES} frames for a {WINDOW_S:.0f}s "
                  f"window -> {WINDOW_S / FRAMES * 1000:.0f} ms/frame")
            print(f"[frames] duration-derived target content ends at frame "
                  f"{frames_for(s['target_speech_duration_s'])}; "
                  f"first frame matching silence (cos>{thr}) is {first_pad}")

        n_tgt = frames_for(s["target_speech_duration_s"])
        n_adv = frames_for(s["carrier_duration_s"])
        prof = cos_profile(e_a, e_t)

        A = prof[:n_tgt].mean().item()
        B = prof[n_tgt:n_adv].mean().item()
        C = prof[n_adv:].mean().item()
        G = prof.mean().item()

        print(f"{s['clip'][:29]:30s}{n_tgt:5d}{n_adv - n_tgt:6d}"
              f"{max(0, FRAMES - n_adv):6d}{G:9.4f}{A:9.4f}{B:9.4f}{C:9.4f}")
        rows.append((G, A, B, C, n_tgt, n_adv))

    n = len(rows)
    print("-" * 84)
    print(f"{'MEAN':30s}{'':17s}"
          + "".join(f"{sum(r[i] for r in rows) / n:9.4f}" for i in range(4)))
    fA = rows[0][4] / FRAMES
    print(f"\nfraction of frames that carry target content: {fA:.1%}  "
          f"-> {1 - fA:.1%} of the mean-per-frame objective is pad-dominated")
    print(f"(regions measured over {FRAMES} tower frames)")

    # Ceiling: can this parameterization carry the target at all?
    if not args.ceiling:
        return
    print("\n--- ceiling check: true target speech through the EnCodec round-trip ---")
    prompt = "What does the person say in this audio?"
    for sp in summaries:
        s = json.load(open(sp))
        tgt = load_wav_16k(Path(s["target_audio"]), args.device)
        rt = encodec_roundtrip_16k(tgt)
        with torch.no_grad():
            out_raw = model.generate(tgt, prompt=prompt)
            out_rt = model.generate(rt, prompt=prompt)
        print(f"  target : {s['target_text']}")
        print(f"    raw      -> {str(out_raw)[:110]}")
        print(f"    encodec  -> {str(out_rt)[:110]}")
        break


if __name__ == "__main__":
    main()
