import argparse
import json
import os
import re
from typing import List, Tuple

import torch
import whisper
from tqdm import tqdm

try:
    import jiwer
except ImportError:
    jiwer = None


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield arr[i : i + bs]


def load_wav(audio_path: str):
    wav = whisper.load_audio(audio_path)
    return torch.from_numpy(wav) if not isinstance(wav, torch.Tensor) else wav


def wav_to_mel_T128(wav: torch.Tensor) -> torch.Tensor:
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


def parse_eval_file(path: str) -> List[Tuple[str, str]]:
    base = os.path.dirname(os.path.expanduser(path))

    if path.endswith(".jsonl"):
        pairs: List[Tuple[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ap = rec.get("audio_path") or rec.get("audio")
                gt = rec.get("text") or rec.get("gt") or rec.get("transcript") or rec.get("reference") or rec.get("ref")
                if not ap:
                    continue
                if isinstance(ap, str) and not os.path.isabs(ap):
                    ap = os.path.join(base, ap)
                pairs.append((ap, gt or ""))
        return pairs

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs2: List[Tuple[str, str]] = []
    for r in data:
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            rel, gt = r[0], r[1]
            ap = rel
            if isinstance(ap, str) and not os.path.isabs(ap):
                ap = os.path.join(base, ap)
            pairs2.append((ap, gt if isinstance(gt, str) else ""))
        elif isinstance(r, dict):
            ap = r.get("audio_path") or r.get("audio")
            gt = r.get("text") or r.get("gt") or r.get("transcript") or r.get("reference") or r.get("ref")
            if not ap:
                continue
            if isinstance(ap, str) and not os.path.isabs(ap):
                ap = os.path.join(base, ap)
            pairs2.append((ap, gt if isinstance(gt, str) else ""))

    return pairs2


def main():
    parser = argparse.ArgumentParser("Evaluate LLaMA-Omni2 ASR (WER) with optional noise")
    parser.add_argument("--llama_omni2_root", type=str, default="../LLaMA-Omni2")
    parser.add_argument("--model_path", type=str, default="../data/models/Llama-Omni2")
    parser.add_argument("--eval_file", type=str, default=".${DATASETS_ROOT}/libri_500.json")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, default="Transcribe this audio verbatim:")
    parser.add_argument("--prefix", type=str, default="Sure, the transcription is: ")
    parser.add_argument("--noise_path", type=str, default="./Llama2/logs/32_noise/checkpoints/xx.pt")
    parser.add_argument("--no_noise", action="store_true")
    parser.add_argument("--output_log", type=str, default="evaluation_log.jsonl")
    args = parser.parse_args()

    if jiwer is None:
        print("Please `pip install jiwer` for WER computation.")
        return

    # Import official repo
    root = os.path.expanduser(args.llama_omni2_root)
    if root not in os.sys.path:
        os.sys.path.insert(0, root)

    from transformers import AutoConfig, AutoTokenizer  # noqa: E402
    from llama_omni2.model import Omni2SpeechQwen2ForCausalLM  # noqa: E402
    from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN  # noqa: E402

    device = torch.device(args.device)

    model_path = os.path.expanduser(args.model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    pairs = parse_eval_file(os.path.expanduser(args.eval_file))

    noise_tile = None
    if args.noise_path and (not args.no_noise):
        noise_tile = load_tile_from_noise(args.noise_path, device=device)  # [1,128,L]
    elif args.no_noise:
        print("--no_noise specified: skipping noise application.")

    # Build prompt input_ids
    user_content = DEFAULT_SPEECH_TOKEN
    if args.prompt:
        user_content = f"{DEFAULT_SPEECH_TOKEN} {args.prompt}"

    messages = [{"role": "user", "content": user_content}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
    ids[ids == tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX

    if args.prefix:
        prefix_ids = tokenizer(args.prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
        ids = torch.cat([ids, prefix_ids], dim=0)

    input_ids_once = ids.unsqueeze(0).to(device)
    attn_once = torch.ones_like(input_ids_once, dtype=torch.long, device=device)

    gts, preds = [], []
    gts_no_empty, preds_no_empty = [], []
    empty_pred_count = 0
    total_count = 0

    with open(args.output_log, "w", encoding="utf-8") as lf:
        for batch in tqdm(list(create_batches(pairs, args.batch_size)), desc="ASR"):
            mels: List[torch.Tensor] = []
            gtb: List[str] = []

            for ap, gt in batch:
                if not os.path.exists(ap):
                    continue
                wav = load_wav(ap)
                mel = wav_to_mel_T128(wav)  
                mels.append(mel)
                gtb.append(gt)

            if not mels:
                continue

            # Stack fixed-length Whisper mels into a true batch: [B,T,128]
            mel_batch = torch.stack([m.to(dtype=torch.float32) for m in mels], dim=0).to(device, non_blocking=True)
            B, T, _ = mel_batch.shape

            if noise_tile is not None:
                tile_use = tile_repeat_crop(noise_tile, target_frames=T)  # [1,128,T]
                tile_use = tile_use.squeeze(0).transpose(0, 1).contiguous()  # [T,128]
                mel_batch = mel_batch + tile_use.to(device=device, dtype=mel_batch.dtype).unsqueeze(0)

            speech = mel_batch.to(dtype=torch.bfloat16)
            speech_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            input_ids = input_ids_once.expand(B, -1)
            attn = attn_once.expand(B, -1)

            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    attention_mask=attn,
                    speech=speech,
                    speech_lengths=speech_lengths,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.gen_max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            pred_texts = tokenizer.batch_decode(out, skip_special_tokens=True)

            for gt, pred_raw in zip(gtb, pred_texts):
                pred_raw = (pred_raw or "").strip()
                if args.prefix:
                    idx = pred_raw.rfind(args.prefix)
                    if idx != -1:
                        pred_raw = pred_raw[idx + len(args.prefix):].strip()

                ngt, npr = normalize_text(gt), normalize_text(pred_raw)
                gts.append(ngt)
                preds.append(npr)

                total_count += 1
                if npr == "":
                    empty_pred_count += 1
                else:
                    gts_no_empty.append(ngt)
                    preds_no_empty.append(npr)

                lf.write(json.dumps({"gt": ngt, "pred": npr}, ensure_ascii=False) + "\n")

    res_all = jiwer.process_words(gts, preds)

    if preds_no_empty:
        res_no_empty = jiwer.process_words(gts_no_empty, preds_no_empty)
    else:
        res_no_empty = None

    print(
        f"\nSamples={total_count} | empty_pred={empty_pred_count} ({(empty_pred_count / max(1, total_count)) * 100:.2f}%)"
    )
    print(
        f"WER(all)={res_all.wer:.4f} | Subs={res_all.substitutions} Del={res_all.deletions} Ins={res_all.insertions}"
    )
    if res_no_empty is None:
        print("WER(non-empty)=N/A (no non-empty predictions)")
    else:
        print(
            f"WER(non-empty)={res_no_empty.wer:.4f} | Subs={res_no_empty.substitutions} Del={res_no_empty.deletions} Ins={res_no_empty.insertions}"
        )

    print(f"Log saved -> {args.output_log}")


if __name__ == "__main__":
    main()
