import argparse
import json
import os
import sys
import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Ensure we can import from the MM project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if MM_ROOT not in sys.path:
    sys.path.append(MM_ROOT)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _detect_feature_key(inputs):
    """Detect the audio feature tensor key produced by the processor."""
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


def create_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def parse_args():
    p = argparse.ArgumentParser(description="Run AIR-Bench Omni inference using Qwen2.5-Omni in MM project's style.")
    p.add_argument("--model_path", type=str, default="../data/models/Qwen2.5-Omni-7B")
    p.add_argument("--data_root", type=str, default=".${DATASETS_ROOT}/AIR-Bench-Dataset/Chat")
    p.add_argument("--meta_file", type=str, default="Chat_meta_filtered.json")
    p.add_argument("--output_dir", type=str, default="./Qwen2.5/logs/128_noise/Chat_result")
    p.add_argument("--noise_path", type=str, default="./Qwen2.5/logs/128_noise/xx.pt")
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4, help="Thread workers for parallel audio loading")
    p.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--truncate_audio_features", type=int, default=0, help="If >0, truncate audio feature length to this value")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "Omni_result_prefix.jsonl")

    print(f"Loading Qwen2.5-Omni model from {args.model_path}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device).eval()

    noise_tile = None
    if args.noise_path and (not args.no_noise):
        print(f"Loading noise from {args.noise_path}")
        noise_tile = load_tile_from_noise(args.noise_path, device=device).to(dtype)

    input_file = os.path.join(args.data_root, args.meta_file)
    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    sr = processor.feature_extractor.sampling_rate

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
        y, _ = librosa.load(wav_fn, sr=sr, mono=True)
        if y is None or len(y) == 0:
            return None
        if len(y) < max(1, int(sr * 0.01)):
            return None
        return (y, item)

    with open(output_file, "w", encoding="utf-8") as fout:
        for batch_items in tqdm(create_batches(all_data, args.batch_size), desc="AIR-Bench Omni"):
            # Audio loading is CPU+IO bound; parallelize it to reduce GPU idle time.
            wavs, questions, valid_items = [], [], []
            with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as ex:
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

            prompts = []
            for q in questions:
                conv = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": "_"},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
                prompts.append(processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))

            # NOTE: Qwen2.5-Omni processor uses `audio=` rather than `audios=` in official examples.
            inputs = processor(text=prompts, audio=wavs, sampling_rate=sr, padding=True, return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            feature_key = _detect_feature_key(inputs)

            if args.truncate_audio_features and args.truncate_audio_features > 0:
                target_T = int(args.truncate_audio_features)
                feats = inputs[feature_key]
                if isinstance(feats, torch.Tensor) and feats.dim() == 3 and feats.shape[-1] > target_T:
                    inputs[feature_key] = feats[..., :target_T]
                    if "feature_attention_mask" in inputs and isinstance(inputs.get("feature_attention_mask"), torch.Tensor):
                        inputs["feature_attention_mask"] = inputs["feature_attention_mask"][:, :target_T]

            if noise_tile is not None:
                feats = inputs[feature_key]
                T = feats.shape[-1]
                tile_to_add = noise_tile if T == noise_tile.shape[-1] else F.interpolate(noise_tile, size=T, mode="linear", align_corners=False)
                inputs[feature_key] = feats + tile_to_add

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    return_audio=False,
                )

            prompt_len = inputs["input_ids"].shape[1]
            responses = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)

            for item, response in zip(valid_items, responses):
                item["response"] = response
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Inference complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()

