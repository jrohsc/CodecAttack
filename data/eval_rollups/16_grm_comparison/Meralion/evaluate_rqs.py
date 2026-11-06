import argparse
import json
import os
import sys
import whisper
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def create_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def load_wav(audio_path: str):
    try:
        return whisper.load_audio(audio_path)
    except Exception as e:
        print(f"[WARN] Failed to load audio {audio_path}: {e}")
        return None


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


def tile_repeat_crop(tile_1xFxL: torch.Tensor, target_frames: int) -> torch.Tensor:
    L = int(tile_1xFxL.size(-1))
    T = int(target_frames)
    if T <= L:
        return tile_1xFxL[..., :T]
    reps = (T + L - 1) // L
    tiled = tile_1xFxL.repeat(1, 1, reps)
    return tiled[..., :T]


def _build_meralion_prompt(processor: AutoProcessor, question: str) -> str:
    prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: {speech}"
    speech = getattr(processor, "speech_token", "<SpeechHere>")
    conv = [{"role": "user", "content": prompt_template.format(query=question, speech=speech)}]
    return processor.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


def parse_args():
    p = argparse.ArgumentParser(description="Run AIR-Bench Omni inference using MERaLioN.")
    p.add_argument("--model_path", type=str, default="../data/models/MERaLioN")
    p.add_argument("--data_root", type=str, default=".${DATASETS_ROOT}/AIR-Bench-Dataset/Chat")
    p.add_argument("--meta_file", type=str, default="Chat_meta_filtered.json")
    p.add_argument("--output_dir", type=str, default="./Meralion/logs/Chat_result")
    p.add_argument("--noise_path", type=str, default="./Meralion/logs/64_noise/xx.pt")
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "Chat_result_prefix.jsonl")

    print(f"Loading MERaLioN model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()

    noise_tile = None
    if args.noise_path and (not args.no_noise):
        print(f"Loading noise from {args.noise_path}")
        noise_tile = load_tile_from_noise(args.noise_path, device=device).to(dtype=torch_dtype)

    input_file = os.path.join(args.data_root, args.meta_file)
    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    def _resolve_audio_path(item):
        wav_rel = item["path"]
        task_name = item["task_name"]
        dataset_name = item["dataset_name"]
        wav_fn = os.path.join(args.data_root, f"{task_name}_{dataset_name}", wav_rel)
        if os.path.exists(wav_fn):
            return wav_fn
        base, _ = os.path.splitext(wav_fn)
        for ext in [".mp3", ".flac", ".m4a"]:
            alt_path = base + ext
            if os.path.exists(alt_path):
                return alt_path
        return None

    def _load_one(item):
        wav_fn = _resolve_audio_path(item)
        if wav_fn is None:
            return None
        y = load_wav(wav_fn)
        if y is None or len(y) == 0:
            return None
        return (y, item)

    with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
        with open(output_file, "w", encoding="utf-8") as fout:
            for batch_items in tqdm(create_batches(all_data, args.batch_size), desc="AIR-Bench MERaLioN"):
                wavs, questions, valid_items = [], [], []
                results = list(ex.map(_load_one, batch_items))

                for res in results:
                    if res is None:
                        continue
                    y, item = res
                    wavs.append(y)
                    questions.append(item["question"])
                    valid_items.append(item)

                if not wavs:
                    continue

                prompts = [_build_meralion_prompt(processor, q) for q in questions]

                inputs = processor(
                    text=prompts,
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

                if noise_tile is not None:
                    feats = inputs.get("input_features")
                    if feats is None:
                        print("[WARN] 'input_features' not found in processor output, cannot apply noise.")
                    elif noise_tile.size(1) != feats.size(1):
                        print(f"[WARN] Tile mel dim mismatch, skipping noise. tile={noise_tile.size(1)} feats={feats.size(1)}")
                    else:
                        tile_use = tile_repeat_crop(noise_tile, target_frames=int(feats.size(-1)))
                        inputs["input_features"] = feats + tile_use

                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                }
                if int(getattr(args, "min_new_tokens", 0)) > 0:
                    gen_kwargs["min_new_tokens"] = int(args.min_new_tokens)

                with torch.inference_mode():
                    out_ids = model.generate(**inputs, **gen_kwargs)

                in_lens = inputs["attention_mask"].sum(dim=1)
                responses = []
                for i in range(out_ids.size(0)):
                    gen_seq = out_ids[i, int(in_lens[i].item()) :]
                    responses.append(processor.decode(gen_seq, skip_special_tokens=True))

                for item, response in zip(valid_items, responses):
                    item["response"] = response.strip()
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Inference complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
