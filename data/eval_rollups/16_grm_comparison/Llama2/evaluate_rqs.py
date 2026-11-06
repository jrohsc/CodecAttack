import argparse
import json
import os
import sys
import whisper
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List


def create_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def load_wav(audio_path: str):
    wav = whisper.load_audio(audio_path)
    return torch.from_numpy(wav) if not isinstance(wav, torch.Tensor) else wav


def wav_to_mel(wav: torch.Tensor) -> torch.Tensor:
    wav = whisper.pad_or_trim(wav)
    return whisper.log_mel_spectrogram(wav, n_mels=128).permute(1, 0)


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
    if tile.dim() != 3 or tile.size(0) != 1 or tile.size(1) != 128:
        raise ValueError(f"Loaded tile has wrong shape {tuple(tile.shape)}; expect [1, 128, L]")

    return tile.detach().requires_grad_(False)


def tile_repeat_crop(tile_1x128xL: torch.Tensor, target_frames: int) -> torch.Tensor:
    L = int(tile_1x128xL.size(-1))
    T = int(target_frames)
    if T <= L:
        return tile_1x128xL[..., :T]
    reps = (T + L - 1) // L
    tiled = tile_1x128xL.repeat(1, 1, reps)
    return tiled[..., :T]


def parse_args():
    p = argparse.ArgumentParser(description="Run AIR-Bench Omni inference using LLaMA-Omni2.")
    p.add_argument("--llama_omni2_root", type=str, default="../LLaMA-Omni2")
    p.add_argument("--model_path", type=str, default="../data/models/Llama-Omni2")
    p.add_argument("--data_root", type=str, default=".${DATASETS_ROOT}/AIR-Bench-Dataset/Chat")
    p.add_argument("--meta_file", type=str, default="Chat_meta_filtered.json")
    p.add_argument("--output_dir", type=str, default="./Llama2/logs/32_noise/Chat_result")
    p.add_argument("--noise_path", type=str, default="./Llama2/logs/32_noise/checkpoints/xx.pt")
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Import from official repo
    root = os.path.expanduser(args.llama_omni2_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from transformers import AutoConfig, AutoTokenizer  # noqa: E402
    from llama_omni2.model import Omni2SpeechQwen2ForCausalLM  # noqa: E402
    from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN  # noqa: E402

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "Chat_result_prefix.jsonl")

    print(f"Loading LLaMA-Omni2 model from {args.model_path}...")
    model_path = os.path.expanduser(args.model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    noise_tile = None
    if args.noise_path and (not args.no_noise):
        print(f"Loading noise from {args.noise_path}")
        noise_tile = load_tile_from_noise(args.noise_path, device=device)

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
            for batch_items in tqdm(create_batches(all_data, args.batch_size), desc="AIR-Bench LLaMA-Omni2"):
                mels, valid_items = [], []
                results = list(ex.map(_load_one, batch_items))

                for res in results:
                    if res is None:
                        continue
                    y, item = res
                    mel = wav_to_mel(y)
                    mels.append(mel)
                    valid_items.append(item)

                if not mels:
                    continue

                for i in range(len(mels)):
                    item = valid_items[i]
                    mel = mels[i]
                    question = item["question"]

                    user_content = f"{DEFAULT_SPEECH_TOKEN} {question}"
                    messages = [{"role": "user", "content": user_content}]
                    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
                    ids[ids == tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX

                    input_ids = ids.unsqueeze(0).to(device)
                    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

                    mel_dev = mel.to(device=device, dtype=torch.float32)
                    if noise_tile is not None:
                        tile_use = tile_repeat_crop(noise_tile, target_frames=mel_dev.shape[0])  # [1,128,T]
                        tile_use = tile_use.squeeze(0).transpose(0, 1).contiguous()  # [T,128]
                        mel_dev = mel_dev + tile_use.to(mel_dev.dtype)

                    speech = mel_dev.to(dtype=torch.bfloat16).unsqueeze(0)  # [1,T,128]
                    speech_lengths = torch.LongTensor([speech.shape[1]]).to(device)

                    with torch.inference_mode():
                        out_ids = model.generate(
                            input_ids,
                            attention_mask=attn_mask,
                            speech=speech,
                            speech_lengths=speech_lengths,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    response = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

                    item["response"] = response
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Inference complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()

