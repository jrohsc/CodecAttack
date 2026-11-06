import argparse
import json
import os
import re
from typing import List, Tuple

import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import jiwer
except ImportError:
    jiwer = None

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def extract_single_quoted_text(text: str) -> str:
    """Match ALMGuard-like extraction logic (same as evaluate_qwen2.py)."""
    if not isinstance(text, str):
        return ""
    if "'" not in text:
        return text

    match = re.search(r"'(.*?)'", text)
    if match:
        start = match.start(0)
        last_single_quote_index = text.rfind("'")
        if last_single_quote_index > start:
            return text[start + 1 : last_single_quote_index]
    return text


def normalize_text(text: str) -> str:
    """Simple normalization to keep evaluation stable and comparable."""
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_batches(arr: List[Tuple[str, str]], bs: int):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def _detect_feature_key(inputs: dict) -> str:
    """Copied from Inference/qwen2.5_omni.py for processor output compatibility."""
    candidate_keys = ["input_features", "audio_values", "audio_features"]
    for k in candidate_keys:
        v = inputs.get(k, None)
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    raise KeyError("No audio feature tensor found in processor outputs.")


def load_tile_from_noise(noise_path: str, device: torch.device) -> torch.Tensor:
    path = noise_path
    if os.path.isdir(path):
        cand = os.path.join(path, "best.pt")
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(f"Noise checkpoint not found: {path}")

    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)

    tile = obj.get("tile") if isinstance(obj, dict) else obj
    if not isinstance(tile, torch.Tensor):
        raise ValueError(f"Unsupported checkpoint format at {path}")

    tile = tile.to(device)
    if tile.dim() != 3 or tile.size(0) != 1:
        raise ValueError(f"Loaded tile has wrong shape {tuple(tile.shape)}; expect [1, F, L]")

    return tile.detach().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser("Evaluate Qwen2.5-Omni ASR with optional adversarial noise")
    parser.add_argument("--model_path", default="../data/models/Qwen2.5-Omni-7B")
    parser.add_argument("--eval_file", default=".${DATASETS_ROOT}/libri_500.json")
    parser.add_argument("--noise_path", default="./Qwen2.5/logs/128_noise/xx.pt")
    parser.add_argument("--no_noise", action="store_true", help="Disable noise even if --noise_path provided")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--output_log", default="evaluation_omni_log.jsonl")
    args = parser.parse_args()

    if jiwer is None:
        print("Please `pip install jiwer` for WER computation.")
        return

    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    noise_tile = None
    if args.noise_path and (not args.no_noise):
        print(f"Loading noise from {args.noise_path}")
        noise_tile = load_tile_from_noise(args.noise_path, device=torch.device(args.device)).to(dtype=dtype)
    elif args.no_noise:
        print("--no_noise specified: skipping noise application.")

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(args.eval_file)

    # ASR prompt style (same as evaluate_qwen2.py)
    question = "Transcribe the audio into text word-for-word, preserving all words accurately."
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "_"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    sr = processor.feature_extractor.sampling_rate

    gts, preds = [], []
    with open(args.output_log, "w", encoding="utf-8") as lf:
        for batch in tqdm(create_batches(data, args.batch_size), desc="ASR(Omni)"):
            wavs, gt_batch = [], []
            for rel, gt in batch:
                p = os.path.join(base, rel)
                if os.path.exists(p):
                    y, _ = librosa.load(p, sr=sr, mono=True)
                    wavs.append(y)
                    gt_batch.append(gt)
            if not wavs:
                continue

            inputs = processor(
                text=[prompt] * len(wavs),
                audio=wavs,
                sampling_rate=sr,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            if noise_tile is not None:
                feature_key = _detect_feature_key(inputs)
                feats = inputs[feature_key]
                T = feats.shape[-1]
                if T == noise_tile.shape[-1]:
                    tile_use = noise_tile
                else:
                    tile_use = F.interpolate(noise_tile, size=T, mode="linear", align_corners=False)
                tile_use = tile_use.to(dtype=feats.dtype, device=feats.device)
                inputs[feature_key] = feats + tile_use

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.gen_max_new_tokens,
                    return_audio=False,
                )

            pl = inputs["input_ids"].shape[1]
            dec = processor.batch_decode(out[:, pl:], skip_special_tokens=True)

            for gt, pr in zip(gt_batch, dec):
                pr = extract_single_quoted_text(pr)
                ngt, npr = normalize_text(gt), normalize_text(pr)
                gts.append(ngt)
                preds.append(npr)
                lf.write(json.dumps({"gt": ngt, "pred": npr}, ensure_ascii=False) + "\n")

    res = jiwer.process_words(gts, preds)
    print(f"\nWER={res.wer:.4f} | Subs={res.substitutions} Del={res.deletions} Ins={res.insertions}")
    print(f"Log saved -> {args.output_log}")


if __name__ == "__main__":
    main()

