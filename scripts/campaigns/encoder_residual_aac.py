"""
Encoder-residual analysis under AAC-LC.

For each (model, scenario, channel) computes:
    R_M(channel) = || h_M(x_atk_chan) - h_M(x_atk_clean) || / || h_M(x_atk_clean) - h_M(x_clean) ||

where h_M is the model's audio-tower hidden state (Whisper-style encoder output,
post-pooling/projection where available; raw last_hidden_state otherwise) and
x_atk_chan = AAC(x_atk_clean, bitrate).

Interpretation:
    R_M small  -> model's encoder is invariant to AAC's distortion of the
                  perturbation (perturbation features survive in the embedding).
    R_M >> 1   -> AAC-induced mel changes blow up in the encoder; perturbation
                  is effectively erased.

Useful denominator: || h_M(x_atk) - h_M(x_clean) || is the "useful" perturbation
displacement in encoder space, i.e. what the LLM tower sees as the attack signal.
R = AAC-induced drift / useful drift. R<1 means encoder absorbs codec noise more
than it absorbs the attack -> attack survives.

Run one model at a time. AF3 needs transformers>=5.0 (flamingo3 env).

Outputs:
    0_all_combined/mechanistic_analysis/encoder_residual_aac/<model>/<scenario>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

REPO_ROOT = Path("${REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT))
from aac_channel import apply_aac_compression, apply_mp3_compression  # noqa: E402

SCENARIO_DIRS = {
    "S1":  REPO_ROOT / "0_all_combined/results/01_finance_voice_agent/qwen2_audio/eps_1.0/audio",
    "S3a": REPO_ROOT / "0_all_combined/results/03a_ai_detection_bypass/qwen2_audio/eps_1.0/audio",
    "S3b": REPO_ROOT / "0_all_combined/results/03b_copyright_bypass/qwen2_audio/eps_1.0/audio",
}
AAC_BITRATES = [64, 96, 128, 192]
MP3_BITRATES = [64, 96, 128, 192]
# Codec channels analyzed for R. Add-only: AAC keys are unchanged; MP3 keys are
# added so the SAME carriers/embeddings give R_MP3 alongside R_AAC. This is the
# mechanistic answer to AC Q5 "why is MP3 transfer >> AAC-LC": predicted R_MP3
# low everywhere (encoder absorbs MP3 distortion) vs R_AAC high on speech.
CODECS = {
    "aac": (apply_aac_compression, AAC_BITRATES),
    "mp3": (apply_mp3_compression, MP3_BITRATES),
}
OUT_ROOT = REPO_ROOT / "0_all_combined/mechanistic_analysis/encoder_residual_aac"


def load_wav_pair(summary_path: Path):
    with summary_path.open() as fh:
        meta = json.load(fh)
    clean_path = Path(meta["wav_path"])
    adv_path = summary_path.parent / summary_path.name.replace("_summary.json", "_attacked.wav")
    if not adv_path.exists():
        return None
    adv, sr_a = sf.read(str(adv_path), dtype="float32", always_2d=False)
    clean, sr_c = sf.read(str(clean_path), dtype="float32", always_2d=False)
    if adv.ndim > 1:
        adv = adv.mean(axis=1)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if sr_c != sr_a:
        from math import gcd
        g = gcd(sr_a, sr_c)
        clean = resample_poly(clean, sr_a // g, sr_c // g).astype(np.float32)
    n = min(len(adv), len(clean))
    return adv[:n], clean[:n], sr_a


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x
    from math import gcd
    g = gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


class Qwen2AudioEncoder:
    name = "qwen2_audio"
    sr = 16000
    model_path = "${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=device, local_files_only=True
        )
        self.model.train(False)
        self.device = next(self.model.parameters()).device
        self.dtype = dtype

    @torch.no_grad()
    def embed(self, x16k: np.ndarray) -> np.ndarray:
        feats = self.proc.feature_extractor(
            x16k, sampling_rate=self.sr, return_attention_mask=True, return_tensors="pt"
        )
        inp = feats["input_features"].to(self.device).to(self.dtype)
        mask = feats["attention_mask"].to(self.device).bool()
        out = self.model.audio_tower(input_features=inp)
        h = out.last_hidden_state
        # audio_tower halves the time dim (conv stride 2)
        T_h = h.shape[1]
        valid = mask[0]
        if valid.shape[0] == 2 * T_h:
            valid = valid[::2]
        if valid.shape[0] == T_h:
            h = h[:, valid, :]
        return h.mean(dim=1).squeeze(0).float().cpu().numpy()


class Qwen25OmniEncoder:
    name = "qwen25_omni"
    sr = 16000
    model_path = "${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=device, local_files_only=True
        )
        self.model.train(False)
        self.device = next(self.model.thinker.parameters()).device
        self.dtype = dtype

    @torch.no_grad()
    def embed(self, x16k: np.ndarray) -> np.ndarray:
        fe = self.proc.feature_extractor
        feats = fe(x16k, sampling_rate=self.sr, return_attention_mask=True, return_tensors="pt")
        inp = feats["input_features"].to(self.device).to(self.dtype)
        mask = feats["attention_mask"].to(self.device)
        feat_lens = mask.sum(dim=-1)
        af = self.model.thinker.get_audio_features(
            input_features=inp,
            feature_attention_mask=mask,
            audio_feature_lengths=feat_lens,
        )
        if af.dim() == 3:
            af = af.squeeze(0)
        return af.mean(dim=0).float().cpu().numpy()


class AudioFlamingoEncoder:
    name = "audio_flamingo"
    sr = 16000
    model_path = "${MODEL_ROOT}/nvidia/hub/models--nvidia--audio-flamingo-3-hf/snapshots/1b7715c1cbdfcaa5042e79cc3c814f6625681cc7"

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.proc = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=device, local_files_only=True, trust_remote_code=True
        )
        self.model.train(False)
        self.device = next(self.model.parameters()).device
        self.dtype = dtype

    @torch.no_grad()
    def embed(self, x16k: np.ndarray) -> np.ndarray:
        fe = self.proc.feature_extractor
        feats = fe(x16k, sampling_rate=self.sr, return_attention_mask=True, return_tensors="pt")
        inp = feats["input_features"].to(self.device).to(self.dtype)
        mask = feats["attention_mask"].to(self.device)
        out = self.model.audio_tower(input_features=inp, input_features_mask=mask)
        h = getattr(out, "last_hidden_state", out)
        if h.dim() == 3:
            h = h.mean(dim=1).squeeze(0)
        return h.float().cpu().numpy()


MODELS = {
    "qwen2_audio": Qwen2AudioEncoder,
    "qwen25_omni": Qwen25OmniEncoder,
    "audio_flamingo": AudioFlamingoEncoder,
}


def run(model_key: str, scenario: str, max_carriers: int):
    audio_dir = SCENARIO_DIRS[scenario]
    summaries = sorted(audio_dir.glob("*_summary.json"))
    if max_carriers > 0:
        summaries = summaries[:max_carriers]

    print(f"[{model_key} / {scenario}] loading model...", flush=True)
    enc = MODELS[model_key]()

    out_dir = OUT_ROOT / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    per_carrier = []
    for i, s in enumerate(summaries):
        pair = load_wav_pair(s)
        if pair is None:
            continue
        adv, clean, sr = pair
        x_atk = to_16k(adv, sr)
        x_clean = to_16k(clean, sr)

        h_atk = enc.embed(x_atk)
        h_clean = enc.embed(x_clean)
        useful = float(np.linalg.norm(h_atk - h_clean) + 1e-9)

        row = {"name": s.stem, "useful": useful, "channels": {}}
        for codec, (apply_fn, bitrates) in CODECS.items():
            for br in bitrates:
                x_cod = apply_fn(x_atk, 16000, bitrate_kbps=br)
                x_cod = to_16k(np.asarray(x_cod, dtype=np.float32), 16000)
                n = min(len(x_cod), len(x_atk))
                h_cod = enc.embed(x_cod[:n])
                drift = float(np.linalg.norm(h_cod - h_atk))
                row["channels"][f"{codec}_{br}k"] = {"drift": drift, "ratio": drift / useful}
        per_carrier.append(row)
        print(f"  [{i+1}/{len(summaries)}] {s.stem[:60]:60s} useful={useful:7.2f}", flush=True)

    mean_ratio = {
        codec: {br: float(np.mean([c["channels"][f"{codec}_{br}k"]["ratio"] for c in per_carrier]))
                for br in bitrates}
        for codec, (_, bitrates) in CODECS.items()
    }
    mean_useful = float(np.mean([c["useful"] for c in per_carrier]))
    out = {
        "model": model_key, "scenario": scenario,
        "n": len(per_carrier),
        "mean_useful_drift": mean_useful,
        "mean_ratio_aac": mean_ratio["aac"],          # back-compat (unchanged schema)
        "mean_ratio_mp3": mean_ratio["mp3"],          # new: MP3 mechanism (Q5)
        "mean_ratio": mean_ratio,                      # nested {codec: {br: R}}
        "per_carrier": per_carrier,
    }
    out_path = out_dir / f"{scenario}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{model_key} / {scenario}] saved -> {out_path}")
    for codec, (_, bitrates) in CODECS.items():
        rs = "  ".join(f"{br}k={mean_ratio[codec][br]:.3f}" for br in bitrates)
        print(f"  mean R({codec}): {rs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--scenario", required=True, choices=list(SCENARIO_DIRS))
    ap.add_argument("--max-carriers", type=int, default=-1, help="cap N for smoke testing")
    args = ap.parse_args()
    run(args.model, args.scenario, args.max_carriers)


if __name__ == "__main__":
    main()
