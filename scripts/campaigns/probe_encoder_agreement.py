"""
Does the encoder alignment we build in the source model exist in the others?

The transfer premise is that Qwen2-Audio, Qwen2.5-Omni and Audio-Flamingo-3 all
carry Whisper-large-v3-derived audio towers, so an attack that makes the SOURCE
encoder see the target should make THEIR encoders see it too, and only the
LM readout should differ. That premise has never been measured here -- the
transfer cells only ever measured the end-to-end text, which conflates "the
encoder didn't move" with "the encoder moved but the LM didn't follow".

This measures the encoder half directly. For each attacked clip:

    delta_cos = cos(adv, target) - cos(carrier, target)

computed inside ONE model's own audio tower, over the frames the spoken target
occupies. Run it once per evaluator and compare:

  - delta_cos large in the source AND in an evaluator -> encoder geometry is
    shared; the attack reaches their encoder and the failure is downstream, in
    the adapter/LM. Worth pushing on the LM readout.
  - delta_cos large in the source and ~0 elsewhere -> the direction we optimize
    is specific to the source's fine-tuned encoder. Encoder-space transfer is
    structurally dead for these victims, and the move is ARO's trick: align to a
    frozen external representation (WavLM/HuBERT) that no victim was trained on.

Each model needs its own conda env, so this writes one json per model and
--summarize prints the joint table.

Usage:
    conda activate codec-attack && python probe_encoder_agreement.py \
        --bundle 0_all_combined/results_encoder_transfer/enc_mask_match \
        --model qwen2_audio
    conda activate qwen-omni   && python probe_encoder_agreement.py ... --model qwen25_omni
    python probe_encoder_agreement.py --bundle <same> --summarize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import MODEL_PATHS, WHISPER_INJECT_ROOT, TARGET_SAMPLE_RATE  # noqa: E402
from encoder_transfer_attack import load_wav_16k  # noqa: E402

sys.path.insert(0, WHISPER_INJECT_ROOT)

MODELS = ["qwen2_audio", "qwen25_omni", "qwen25_omni_3b", "audio_flamingo"]


def build(model_name, device):
    """Return (embed_fn, padded_len_or_None)."""
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        m = Qwen2AudioModel(model_path=MODEL_PATHS[model_name],
                            device=device, dtype=torch.bfloat16)
        tower = m.model.audio_tower

        def embed(w):
            mel = m.wav_to_mel(w).to(torch.bfloat16)
            with torch.no_grad():
                return tower(mel).last_hidden_state.float()
        # wav_to_mel zero-pads to a fixed 30s window.
        return embed, m.WAV_LENGTH

    if model_name in ("qwen25_omni", "qwen25_omni_3b"):
        from models.qwen25_omni import Qwen25OmniModel
        m = Qwen25OmniModel(model_path=MODEL_PATHS[model_name],
                            device=device, dtype=torch.bfloat16)
        tower = m.thinker.audio_tower

        # Qwen2.5-Omni's tower does NOT take a batched 3-D mel. It chunks the
        # time axis itself (input_features.split(chunk_lengths, dim=1)), so it
        # wants the 2-D [n_mels, T] that wav_to_mel already returns, plus the
        # two length tensors -- without feature_lens it dies on
        #   torch.ceil(feature_lens / (self.n_window * 2))  ->  NoneType / int
        # which is what killed array tasks 1,2,5,6,9,10 of job 62874687.
        # Contract copied from the model's own call site
        # (modeling_qwen2_5_omni.py ~L1743): aftercnn_lens is the FIRST return
        # of _get_feat_extract_output_lengths, not the second.
        def embed(w):
            mel = m.wav_to_mel(w.squeeze(0)).to(torch.bfloat16)   # [n_mels, T]
            flen = torch.tensor([mel.shape[-1]], device=mel.device,
                                dtype=torch.long)
            aftercnn_lens, _ = tower._get_feat_extract_output_lengths(flen)
            with torch.no_grad():
                out = tower(mel, feature_lens=flen,
                            aftercnn_lens=aftercnn_lens)
                h = getattr(out, "last_hidden_state", out)
            # This tower returns 2-D [T_out, D] (it cats per-audio states on
            # dim 0), so it needs the batch axis put back.
            return h.float().unsqueeze(0) if h.dim() == 2 else h.float()
        # No fixed padding: the mel tracks the true input length.
        return embed, None

    if model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        m = AudioFlamingoModel(model_path=MODEL_PATHS[model_name],
                               device=device, dtype=torch.bfloat16)
        tower = m.model.audio_tower if hasattr(m.model, "audio_tower") else None
        if tower is None:
            raise SystemExit("audio_flamingo: no audio_tower attribute; "
                             "inspect the wrapper before trusting a number here")

        # AF3's tower requires input_features_mask -- without it it dies on
        #   input_features_mask.sum(-1)  ->  'NoneType' has no attribute 'sum'
        # (array tasks 3,7,11 of job 62874687). The mask must mark the REAL
        # frames: wav_to_mel zero-pads the waveform to a fixed 30 s window, and
        # an all-ones mask would make the encoder attend across ~25 s of
        # silence, changing every embedding we then take cosines of.
        AF3_HOP = 160  # hard-coded in the wrapper's wav_to_mel STFT

        def embed(w):
            mel = m.wav_to_mel(w).to(torch.bfloat16)   # [1, n_mels, T']
            true_samples = min(int(w.shape[-1]), m.WAV_LENGTH)
            n_real = min(true_samples // AF3_HOP + 1, mel.shape[-1])
            mask = (torch.arange(mel.shape[-1], device=mel.device)
                    < n_real).unsqueeze(0)
            with torch.no_grad():
                return tower(mel,
                             input_features_mask=mask).last_hidden_state.float()
        return embed, getattr(m, "WAV_LENGTH", None)

    raise SystemExit(f"unknown model {model_name}")


def masked_cos(a, b, n_content):
    """Per-frame cosine over the target's content frames, on a common length."""
    L = min(a.shape[1], b.shape[1])
    a, b = a[:, :L], b[:, :L]
    n = min(n_content, L)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)[..., :n]
    return cos.mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    bundle = args.bundle if args.bundle.is_absolute() else HERE / args.bundle

    # One file per model. A single shared encoder_agreement.json would be a
    # read-modify-write race: the four models run as concurrent array tasks and
    # the last writer would silently drop the other three.
    def model_path(m):
        return bundle / f"encoder_agreement__{m}.json"

    if args.summarize:
        data = {}
        for m in MODELS:
            p = model_path(m)
            if p.is_file():
                data[m] = json.load(open(p))
        legacy = bundle / "encoder_agreement.json"
        if legacy.is_file():
            for m, rec in json.load(open(legacy)).items():
                data.setdefault(m, rec)
        if not data:
            sys.exit(f"no encoder_agreement__*.json under {bundle} yet")
        missing = [m for m in MODELS if m not in data]
        if missing:
            print(f"[warn] no result yet for: {', '.join(missing)}")
        print(f"\nENCODER AGREEMENT  {bundle.name}   "
              f"masked cos over the target's content frames\n")
        print(f"{'model':18s}{'n':>4s}{'carrier':>10s}{'adv':>9s}{'delta':>9s}")
        print("-" * 50)
        for mname, d in data.items():
            print(f"{mname:18s}{d['n']:4d}{d['mean_cos_carrier']:10.4f}"
                  f"{d['mean_cos_adv']:9.4f}{d['mean_delta']:+9.4f}")
        print("\ndelta = cos(adv,target) - cos(carrier,target), inside that "
              "model's own tower.\nLarge in the source and ~0 elsewhere means the "
              "optimized direction is\nsource-specific and encoder-space transfer "
              "cannot work for these victims.")
        return

    if not args.model:
        sys.exit("--model is required unless --summarize")

    summaries = sorted((bundle / "audio").glob("*_summary.json"))[: args.n]
    if not summaries:
        sys.exit(f"no summaries under {bundle}")

    embed, padded_len = build(args.model, args.device)

    d_adv, d_car = [], []
    for sp in summaries:
        s = json.load(open(sp))
        tgt = load_wav_16k(Path(s["target_audio"]), args.device)
        adv = load_wav_16k(HERE / s["attacked_wav"], args.device)
        car = load_wav_16k(Path(s["carrier"]), args.device)

        e_t = embed(tgt)
        n_total = e_t.shape[1]
        # Frames the target's speech occupies, in THIS tower's frame rate.
        ref_len = padded_len if padded_len else tgt.shape[-1]
        n_content = max(1, int(np.ceil(tgt.shape[-1] / (ref_len / n_total))))

        c_adv = masked_cos(embed(adv), e_t, n_content)
        c_car = masked_cos(embed(car), e_t, n_content)
        d_adv.append(c_adv)
        d_car.append(c_car)
        print(f"  {s['clip'][:44]:45s} carrier={c_car:.4f} adv={c_adv:.4f} "
              f"delta={c_adv - c_car:+.4f}")

    rec = {
        "n": len(d_adv),
        "mean_cos_carrier": round(float(np.mean(d_car)), 4),
        "mean_cos_adv": round(float(np.mean(d_adv)), 4),
        "mean_delta": round(float(np.mean(d_adv) - np.mean(d_car)), 4),
    }
    out_p = model_path(args.model)
    json.dump(rec, open(out_p, "w"), indent=1)
    print(f"\n{args.model}: carrier={rec['mean_cos_carrier']} "
          f"adv={rec['mean_cos_adv']} delta={rec['mean_delta']:+}")
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
