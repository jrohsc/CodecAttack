import argparse
import json
import os
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import jiwer
except ImportError:
    jiwer = None

import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tile_from_noise(noise_path: str, device: torch.device) -> torch.Tensor:
    path = noise_path
    if os.path.isdir(path):
        cand = os.path.join(path, "best.pt")
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(f"noise checkpoint not found: {path}")

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


def tile_repeat_crop(tile_1xFxL: torch.Tensor, target_frames: int) -> torch.Tensor:
    L = int(tile_1xFxL.size(-1))
    T = int(target_frames)
    if T <= L:
        return tile_1xFxL[..., :T]
    reps = (T + L - 1) // L
    tiled = tile_1xFxL.repeat(1, 1, reps)
    return tiled[..., :T]


def _build_meralion_asr_prompt(processor: AutoProcessor) -> str:
    prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: {speech}"
    speech = getattr(processor, "speech_token", "<SpeechHere>")
    query = "Transcribe the audio into text word-for-word, preserving all words accurately."
    conv = [{"role": "user", "content": prompt_template.format(query=query, speech=speech)}]
    return processor.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


def create_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def main():
    parser = argparse.ArgumentParser("Evaluate MERaLioN ASR with optional mel tile")
    parser.add_argument("--model_path", type=str, default="../data/models/MERaLioN")
    parser.add_argument("--eval_file", type=str, default=".${DATASETS_ROOT}/libri_500.json")
    parser.add_argument("--noise_path", type=str, default="./Meralion/logs/64_noise/xx.pt")
    parser.add_argument("--no_noise", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_log", type=str, default="meralion_asr_eval_prefix.jsonl")
    args = parser.parse_args()

    if jiwer is None:
        print("Please `pip install jiwer` for WER computation.")
        return

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Keep evaluation simple: single-device load.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map="auto" if args.device.startswith("cuda") else None,
    )
    model.eval()

    device = torch.device(args.device)

    tile = None
    if args.noise_path and (not args.no_noise):
        print(f"Loading noise from {args.noise_path}")
        tile = load_tile_from_noise(args.noise_path, device=device).to(dtype=torch_dtype)
    elif args.no_noise:
        print("--no_noise specified: skipping noise application.")

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(args.eval_file)

    prompt_text = _build_meralion_asr_prompt(processor)

    gts, preds = [], []
    wrote_header = False

    with open(args.output_log, "w", encoding="utf-8") as lf:
        for batch in tqdm(create_batches(data, args.batch_size), desc="MERaLioN-ASR"):
            wavs, gt_batch, rel_batch = [], [], []
            for rel, gt in batch:
                p = os.path.join(base, rel)
                if not os.path.exists(p):
                    continue
                wav = whisper.load_audio(p)  # float32, 16k
                wavs.append(wav)
                gt_batch.append(gt)
                rel_batch.append(rel)
            if not wavs:
                continue

            inputs = processor(
                text=[prompt_text] * len(wavs),
                audios=wavs,
                padding=True,
                sampling_rate=16000,
            )

            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    if k == "input_features":
                        inputs[k] = v.to(device=device, dtype=torch_dtype)
                    else:
                        inputs[k] = v.to(device=device)

            if tile is not None:
                feats = inputs.get("input_features", None)
                if feats is None:
                    raise KeyError("processor did not return input_features")

                if tile.size(1) != feats.size(1):
                    raise ValueError(
                        f"Tile mel dim mismatch: tile.shape={tuple(tile.shape)} feats.shape={tuple(feats.shape)}"
                    )

                tile_use = tile_repeat_crop(tile, target_frames=int(feats.size(-1)))
                inputs["input_features"] = feats + tile_use.to(device=device, dtype=feats.dtype)

            if not wrote_header:
                wrote_header = True
                feats = inputs["input_features"]
                print(f"[INFO] input_features.shape={tuple(feats.shape)} dtype={feats.dtype}")
                if tile is not None:
                    print(f"[INFO] tile.shape={tuple(tile.shape)}")

            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            in_lens = inputs["attention_mask"].sum(dim=1)
            gen_seqs = []
            for i in range(out_ids.size(0)):
                gen_seqs.append(out_ids[i, int(in_lens[i].item()) :])

            decoded = processor.batch_decode(gen_seqs, skip_special_tokens=True)
            if isinstance(decoded, str):
                decoded = [decoded]

            for rel, gt, pr in zip(rel_batch, gt_batch, decoded):
                ngt, npr = normalize_text(gt), normalize_text(pr)
                gts.append(ngt)
                preds.append(npr)
                lf.write(json.dumps({"audio": rel, "gt": ngt, "pred": npr}, ensure_ascii=False) + "\n")

    res = jiwer.process_words(gts, preds)
    summary = {
        "summary": {
            "wer": float(res.wer),
            "substitutions": int(res.substitutions),
            "deletions": int(res.deletions),
            "insertions": int(res.insertions),
            "num_samples": int(len(gts)),
        }
    }

    with open(args.output_log, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\nWER={res.wer:.4f} | Subs={res.substitutions} Del={res.deletions} Ins={res.insertions}")
    print(f"Log saved -> {args.output_log}")


if __name__ == "__main__":
    main()
