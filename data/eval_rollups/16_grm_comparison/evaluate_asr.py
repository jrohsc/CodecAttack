import argparse
import json
import os
import re

import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import jiwer
except ImportError:
    jiwer = None

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from gradient import load_model_and_processor, _detect_feature_key


def extract_single_quoted_text(text: str) -> str:
    """Match ALMGuard's extraction logic."""
    if not isinstance(text, str):
        return ""
    if "'" not in text:
        return text

    match = re.search(r"'(.*?)'", text)
    if match:
        start = match.start(0)
        last_single_quote_index = text.rfind("'")
        if last_single_quote_index > start:
            return text[start + 1:last_single_quote_index]
    return text


def normalize_text(text: str) -> str:
    """Simple normalization to keep evaluation stable and comparable."""
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def load_tile_from_noise(noise_path: str, device: torch.device) -> torch.Tensor:
    path = noise_path
    if os.path.isdir(path):
        cand = os.path.join(path, "best.pt")
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(f"Noise checkpoint not found: {path}")

    obj = torch.load(path, map_location=device)
    tile = obj.get("tile") if isinstance(obj, dict) else obj
    if not isinstance(tile, torch.Tensor):
        raise ValueError(f"Unsupported checkpoint format at {path}")

    tile = tile.to(device)
    if tile.dim() != 3 or tile.size(0) != 1:
        raise ValueError(f"Loaded tile has wrong shape {tuple(tile.shape)}; expect [1, F, L]")

    return tile.detach().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser("Evaluate Qwen2-Audio ASR with optional adversarial noise")
    parser.add_argument("--model_path", default="../data/models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--eval_file", default=".${DATASETS_ROOT}/libri_500.json")
    parser.add_argument("--noise_path", default="./logs/checkpoints/best.pt")
    parser.add_argument("--no_noise", action="store_true", help="Disable noise even if --noise_path provided")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--output_log", default="evaluation_log.jsonl")
    args = parser.parse_args()

    if jiwer is None:
        print("Please `pip install jiwer` for WER computation.")
        return

    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    processor, model, _ = load_model_and_processor(args.model_path, torch_dtype=dtype)
    model.to(args.device).eval()

    noise_tile = None
    if args.noise_path and (not args.no_noise):
        print(f"Loading noise from {args.noise_path}")
        noise_tile = load_tile_from_noise(args.noise_path, device=args.device).to(dtype)
    elif args.no_noise:
        print("--no_noise specified: skipping noise application.")

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(args.eval_file)

    # Match ALMGuard's ASR prompt style
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
        for batch in tqdm(create_batches(data, args.batch_size), desc="ASR"):
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
                audios=wavs,
                sampling_rate=sr,
                padding=True,
                return_tensors="pt",
            ).to(args.device)

            if noise_tile is not None:
                feature_key = _detect_feature_key(inputs)
                feats = inputs[feature_key]
                T = feats.shape[-1]
                tile_use = noise_tile if T == noise_tile.shape[-1] else F.interpolate(
                    noise_tile, size=T, mode="linear", align_corners=False
                )
                inputs[feature_key] = feats + tile_use

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.gen_max_new_tokens)

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
