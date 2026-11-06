import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import whisper
from whisper.audio import load_audio as whisper_load_audio, log_mel_spectrogram, pad_or_trim, SAMPLE_RATE as WHISPER_SR


def load_meralion_model_and_processor(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: "45GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return processor, model


def build_chat_texts(processor: AutoProcessor, prompt: str, target: str) -> Tuple[str, str]:
    prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: {speech}"
    speech = getattr(processor, "speech_token", "<SpeechHere>")

    conv_user = [{"role": "user", "content": prompt_template.format(query=prompt, speech=speech)}]
    conv_full = conv_user + [{"role": "assistant", "content": target}]

    prompt_text = processor.tokenizer.apply_chat_template(conv_user, tokenize=False, add_generation_prompt=True)
    full_text = processor.tokenizer.apply_chat_template(conv_full, tokenize=False, add_generation_prompt=False)

    return prompt_text, full_text


def build_target_only_labels(processor: AutoProcessor, input_ids: Tensor, target: str) -> Tuple[Tensor, Tensor]:
    labels = torch.full_like(input_ids, -100)

    target_ids = processor.tokenizer(target, add_special_tokens=False)["input_ids"]
    if not isinstance(target_ids, list) or len(target_ids) == 0:
        raise ValueError("Tokenized target is empty; check --target")

    target_ids_t = torch.tensor(target_ids, device=input_ids.device, dtype=input_ids.dtype)

    def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
        h_len = int(haystack.numel())
        n_len = int(needle.numel())
        if n_len == 0 or n_len > h_len:
            return -1
        for s in range(h_len - n_len + 1):
            if torch.equal(haystack[s : s + n_len], needle):
                return s
        return -1

    start_pos = torch.full((input_ids.size(0),), -1, device=input_ids.device, dtype=torch.long)
    for i in range(input_ids.size(0)):
        seq = input_ids[i]
        pos = _find_subsequence(seq, target_ids_t)
        start_pos[i] = int(pos)
        if pos >= 0:
            labels[i, pos : pos + int(target_ids_t.numel())] = input_ids[i, pos : pos + int(target_ids_t.numel())]

    return labels, start_pos


def compute_band_importance(grad_features: Tensor, normalize_time: bool = True) -> Tensor:
    assert grad_features.dim() == 3 and grad_features.size(0) == 1, "Expected shape [1, F, T]"
    saliency = grad_features[0].abs()  # [F, T]
    band = saliency.sum(dim=-1)
    if normalize_time:
        T = int(saliency.size(-1))
        if T > 0:
            band = band / float(T)
    return band


def prepare_inputs_from_audio(
    processor: AutoProcessor,
    full_text: str,
    wav: np.ndarray,
    sampling_rate: int,
    device: str,
) -> Dict[str, Tensor]:
    inputs = processor(
        text=full_text,
        audios=[wav.astype(np.float32)],
        sampling_rate=sampling_rate,
        padding=True,
    )

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    if "input_features" not in inputs:
        raise KeyError(f"MERaLioN processor did not return input_features. keys={list(inputs.keys())}")

    # Make features leaf tensor so we can read .grad
    feats = inputs["input_features"].to(torch.float32).detach().clone().requires_grad_(True)
    inputs["input_features"] = feats

    return inputs


def forward_logprob_and_grad(
    model,
    inputs: Dict[str, Tensor],
    labels: Tensor,
) -> Tuple[Tensor, Tensor]:
    run_inputs = dict(inputs)
    run_inputs["labels"] = labels
    run_inputs["use_cache"] = False

    model.zero_grad(set_to_none=True)

    feats_fp32 = run_inputs["input_features"]
    model_dtype = next(model.parameters()).dtype
    run_inputs["input_features"] = feats_fp32.to(dtype=model_dtype)

    out = model(**run_inputs)
    loss = out.loss
    avg_logprob = -loss

    grad_features = torch.autograd.grad(
        avg_logprob,
        feats_fp32,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    return avg_logprob.detach(), grad_features.detach()


def compute_meralion_band_j(
    processor: AutoProcessor,
    model,
    audio_path: str,
    prompt: str,
    target: str,
    device: str,
) -> Tuple[Tensor, Tensor]:
    _p, full_text = build_chat_texts(processor, prompt, target)
    sr = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000))
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)

    inputs = prepare_inputs_from_audio(processor, full_text, wav, sr, device)
    input_ids = inputs["input_ids"]
    labels, start_pos = build_target_only_labels(processor, input_ids, target)

    avg_logprob, grad_features = forward_logprob_and_grad(model, inputs, labels)
    band_j = compute_band_importance(grad_features)
    return band_j, avg_logprob


def compute_whisper_openai_band_a(
    asr_model,
    audio_path: str,
    text_gt: str,
    device: str,
    pad_to_30s: bool = True,
) -> Tuple[Tensor, Tensor]:
    asr_model.eval()

    audio = whisper_load_audio(audio_path)
    if pad_to_30s:
        audio = pad_or_trim(audio)

    mel = log_mel_spectrogram(audio, asr_model.dims.n_mels)  # [128, T]
    mel = mel.unsqueeze(0).to(device)  # [1, 128, T]
    mel = mel.detach().clone().requires_grad_(True)

    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")

    prefix = [tok.sot]
    try:
        prefix.append(tok.lang_id("en"))
    except Exception:
        pass
    if hasattr(tok, "no_timestamps"):
        prefix.append(tok.no_timestamps)

    text_ids = tok.encode(text_gt or "")
    n_text_ctx = getattr(getattr(asr_model, "dims", None), "n_text_ctx", 448)
    max_text_len = max(0, int(n_text_ctx) - len(prefix) - 1)
    if len(text_ids) > max_text_len:
        text_ids = text_ids[:max_text_len]

    full = prefix + text_ids + [tok.eot]
    tokens = torch.tensor([full], dtype=torch.long, device=device)
    input_tokens = tokens[:, :-1].contiguous()
    target_tokens = tokens[:, 1:].contiguous()

    x = asr_model.encoder(mel)
    logits = asr_model.decoder(input_tokens, x)
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab), target_tokens.view(-1))

    asr_model.zero_grad(set_to_none=True)
    loss.backward()

    grad_mel = mel.grad.detach()
    band_a = compute_band_importance(grad_mel)
    return band_a, loss.detach()


def select_bins_from_bands(
    band_j: Tensor,
    band_a: Tensor,
    k: int = 48,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    assert band_j.ndim == 1 and band_a.ndim == 1 and band_j.numel() == band_a.numel(), "band shape mismatch"

    bj = band_j.float()
    ba = band_a.float()

    bj = bj / (bj.sum() + eps)
    ba = ba / (ba.sum() + eps)

    scores = bj / (ba + eps)
    k_eff = min(int(k), int(scores.numel()))
    vals, idx = torch.topk(scores, k=k_eff)

    mask = torch.zeros_like(scores)
    mask[idx] = 1.0

    return {"scores": scores, "topk_idx": idx, "topk_scores": vals, "mask": mask}


def compute_sample_bands_and_topk(
    meralion_processor: AutoProcessor,
    meralion_model,
    asr_model,
    audio_path: str,
    text_gt: str,
    prompt: str,
    target: str,
    k: int = 48,
    meralion_device: str = "cuda",
    whisper_device: str = "cuda",
) -> Dict[str, Tensor]:
    band_j, avg_logprob = compute_meralion_band_j(
        meralion_processor,
        meralion_model,
        audio_path=audio_path,
        prompt=prompt,
        target=target,
        device=meralion_device,
    )

    band_a, asr_loss = compute_whisper_openai_band_a(
        asr_model,
        audio_path=audio_path,
        text_gt=text_gt,
        device=whisper_device,
        pad_to_30s=True,
    )

    sel = select_bins_from_bands(band_j, band_a, k=k)

    return {
        "band_j": band_j.detach().cpu(),
        "band_a": band_a.detach().cpu(),
        "sel_scores": sel["scores"].detach().cpu(),
        "topk_idx": sel["topk_idx"].detach().cpu(),
        "topk_scores": sel["topk_scores"].detach().cpu(),
        "mask": sel["mask"].detach().cpu(),
        "asr_loss": asr_loss.detach().cpu(),
        "meralion_avg_logprob": avg_logprob.detach().cpu(),
    }


def batch_compute_global_mask_for_dataset(
    meralion_processor: AutoProcessor,
    meralion_model,
    asr_model,
    train_file: str,
    prompt: str,
    target: str,
    k: int = 48,
    out_dir: str = "./Meralion/logs",
    meralion_device: str = "cuda",
    whisper_device: str = "cuda",
) -> Dict[str, Tensor]:
    os.makedirs(out_dir, exist_ok=True)

    with open(train_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    Fbins = 128
    counts = torch.zeros(Fbins, dtype=torch.float32)
    weights = torch.zeros(Fbins, dtype=torch.float32)

    for rec in tqdm(records, desc="MERaLioN mel band selection"):
        ap = rec.get("audio_path")
        txt = rec.get("prompt") or rec.get("text") or ""
        if not ap:
            continue
        out = compute_sample_bands_and_topk(
            meralion_processor,
            meralion_model,
            asr_model,
            audio_path=ap,
            text_gt=txt,
            prompt=prompt,
            target=target,
            k=k,
            meralion_device=meralion_device,
            whisper_device=whisper_device,
        )

        topk_idx = out["topk_idx"].to(dtype=torch.long)
        topk_scores = out["topk_scores"].to(dtype=torch.float32)

        counts.index_add_(0, topk_idx, torch.ones_like(topk_scores))
        weights.index_add_(0, topk_idx, topk_scores)

    k_eff = min(int(k), Fbins)
    vals_c, idx_c = torch.topk(counts, k=k_eff)
    vals_w, idx_w = torch.topk(weights, k=k_eff)

    mask_c = torch.zeros(Fbins, dtype=torch.float32)
    mask_w = torch.zeros(Fbins, dtype=torch.float32)
    mask_c[idx_c] = 1.0
    mask_w[idx_w] = 1.0

    np.savez(
        os.path.join(out_dir, "topk_count.npz"),
        mask=mask_c.numpy(),
        top_idx=idx_c.numpy(),
        scores=vals_c.numpy(),
        counts=counts.numpy(),
    )
    np.savez(
        os.path.join(out_dir, "topk_weighted.npz"),
        mask=mask_w.numpy(),
        top_idx=idx_w.numpy(),
        scores=vals_w.numpy(),
        counts=counts.numpy(),
    )

    return {
        "counts": counts,
        "weights": weights,
        "top_idx_counts": idx_c,
        "top_idx_weights": idx_w,
        "mask_counts": mask_c,
        "mask_weights": mask_w,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERaLioN mel band selection (batch only)")

    parser.add_argument("--model_path", type=str, default="../data/models/MERaLioN")
    parser.add_argument("--asr_path", type=str, default="../data/models/Whisper-large-v3/large-v3.pt")
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--meralion_device_map", type=str, default="auto")
    parser.add_argument("--meralion_device", type=str, default="cuda")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--out_dir", type=str, default="./Meralion/logs")

    args = parser.parse_args()

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    meralion_processor, meralion_model = load_meralion_model_and_processor(
        args.model_path,
        device_map=args.meralion_device_map,
        torch_dtype=torch_dtype,
    )

    asr_model = whisper.load_model(args.asr_path, device=args.whisper_device)
    asr_model.eval()
    for p in asr_model.parameters():
        p.requires_grad_(False)

    out = batch_compute_global_mask_for_dataset(
        meralion_processor,
        meralion_model,
        asr_model,
        train_file=args.train_file,
        prompt=args.prompt,
        target=args.target,
        k=args.k,
        out_dir=args.out_dir,
        meralion_device=args.meralion_device,
        whisper_device=args.whisper_device,
    )
    print("Top-{} by counts:".format(args.k), out["top_idx_counts"].tolist())
    print("Top-{} by weighted:".format(args.k), out["top_idx_weights"].tolist())

