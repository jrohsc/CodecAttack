from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import whisper
import json
import os
import numpy as np
from tqdm import tqdm
from whisper.audio import load_audio as whisper_load_audio, log_mel_spectrogram, pad_or_trim, SAMPLE_RATE as WHISPER_SR


def load_model_and_tokenizer(
    llama_omni2_root: str,
    model_path: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    model_path = os.path.expanduser(model_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root = os.path.expanduser(llama_omni2_root)
    if root not in os.sys.path:
        os.sys.path.insert(0, root)

    from transformers import AutoConfig, AutoTokenizer  # noqa: E402
    from llama_omni2.model import Omni2SpeechQwen2ForCausalLM  # noqa: E402

    cfg = AutoConfig.from_pretrained(model_path)
    cfg.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(model_path, config=cfg, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return tok, model, device


def build_chat_ids_and_labels(
    tokenizer,
    prompt: str,
    target: str,
) -> Tuple[Tensor, Tensor]:
    from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN  # noqa: E402

    conv_user_only = [{"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN} {prompt}"}]
    conv_with_answer = conv_user_only + [{"role": "assistant", "content": target}]

    prompt_ids = tokenizer.apply_chat_template(
        conv_user_only, add_generation_prompt=True, return_tensors="pt"
    )[0]

    speech_placeholder_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    prompt_ids = prompt_ids.clone()
    prompt_ids[prompt_ids == speech_placeholder_id] = SPEECH_TOKEN_INDEX

    prefix_len = int(prompt_ids.numel())

    full_ids = tokenizer.apply_chat_template(
        conv_with_answer, add_generation_prompt=False, return_tensors="pt"
    )[0]

    speech_placeholder_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    full_ids = full_ids.clone()
    full_ids[full_ids == speech_placeholder_id] = SPEECH_TOKEN_INDEX

    labels = full_ids.clone()
    labels[:prefix_len] = -100

    return full_ids.unsqueeze(0), labels.unsqueeze(0)


def load_wav(audio_path: str) -> Tensor:
    wav = whisper_load_audio(audio_path)
    return torch.from_numpy(wav) if not isinstance(wav, torch.Tensor) else wav


def wav_to_omni2_speech(wav: Tensor, device: str, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    wav = pad_or_trim(wav)
    mel = log_mel_spectrogram(wav, n_mels=128).permute(1, 0)  # [T, 128]
    speech = mel.to(device=device, dtype=dtype).unsqueeze(0)  # [1, T, 128]
    speech_lengths = torch.LongTensor([speech.shape[1]]).to(device)
    return speech, speech_lengths


def forward_avg_logprob_and_grad(
    model,
    input_ids: Tensor,
    labels: Tensor,
    speech: Tensor,
    speech_lengths: Tensor,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    model.eval()

    speech = speech.contiguous().detach().requires_grad_(True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        speech=speech,
        speech_lengths=speech_lengths,
        use_cache=False,
    )

    loss = outputs.loss
    avg_logprob = -loss

    if speech.grad is not None:
        speech.grad.zero_()
    model.zero_grad(set_to_none=True)
    avg_logprob.backward()

    grad_speech = speech.grad.detach()
    extra = {"loss": loss.detach(), "logits": outputs.logits.detach()}
    return avg_logprob.detach(), grad_speech, extra


def compute_band_importance(
    grad_features: Tensor,
    input_features: Optional[Tensor] = None,
    mode: str = "abs",
    valid_T: Optional[int] = None,
    reduce: str = "sum",
    topk: int = 10,
) -> Dict[str, Tensor]:
    assert grad_features.dim() == 3 and grad_features.size(0) == 1

    g = grad_features[0]
    if g.dim() != 2:
        raise ValueError(f"Unexpected grad_features inner dim: {tuple(g.shape)}")

    if g.size(-1) == 128:
        layout = "T_F"  # Omni2 speech grad: [T, 128]
    elif g.size(0) == 128:
        layout = "F_T"  # Whisper mel grad: [128, T]
    else:
        raise ValueError(f"Cannot infer grad layout; expected one axis to be 128, got {tuple(g.shape)}")

    if mode == "grad_x_input":
        if input_features is None:
            raise ValueError("grad_x_input mode requires input_features")
        assert input_features.dim() == 3 and input_features.size(0) == 1
        x0 = input_features[0]
        if layout == "T_F":
            x = x0
        else:
            x = x0.transpose(0, 1)
        sal = (g * x).abs()
    elif mode == "l2":
        sal = g.pow(2)
    else:
        sal = g.abs()

    if valid_T is not None:
        if layout == "T_F":
            valid_T = max(1, min(int(valid_T), sal.size(0)))
            sal = sal[:valid_T]
        else:
            valid_T = max(1, min(int(valid_T), sal.size(1)))
            sal = sal[:, :valid_T]

    if layout == "T_F":
        if reduce == "l2":
            freq_importance = torch.linalg.vector_norm(sal, dim=0)
        else:
            freq_importance = sal.sum(dim=0)
    else:
        if reduce == "l2":
            freq_importance = torch.linalg.vector_norm(sal, dim=1)
        else:
            freq_importance = sal.sum(dim=1)

    k = min(int(topk), int(freq_importance.numel()))
    vals, idx = torch.topk(freq_importance, k=k)
    return {"freq_importance": freq_importance, "topk_bins": idx, "topk_scores": vals}


def load_mel_from_audio(
    asr_model,
    audio_path: str,
    device: Optional[str] = None,
    pad_to_30s: bool = True,
    requires_grad: bool = True,
):
    if device is None:
        try:
            device = next(asr_model.parameters()).device
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    audio = whisper_load_audio(audio_path)
    if pad_to_30s:
        audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio, asr_model.dims.n_mels)  # [128, T]
    mel = mel.unsqueeze(0).to(device)  # [1, 128, T]
    mel = mel.detach().requires_grad_(bool(requires_grad))

    info = {
        "sr": int(WHISPER_SR),
        "mel_shape": tuple(mel.shape),
    }
    return mel, info


def build_whisper_tokens_from_text(
    asr_model,
    text: str,
    language: str = "en",
    task: str = "transcribe",
    device: Optional[str] = None,
    add_special: bool = True,
):
    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language=language, task=task)

    prefix = []
    if add_special:
        if hasattr(tok, "sot"):
            prefix.append(tok.sot)
        try:
            lang_tok = tok.lang_id(language)
            prefix.append(lang_tok)
        except Exception:
            pass
        if hasattr(tok, "no_timestamps"):
            prefix.append(tok.no_timestamps)

    text_ids = tok.encode(text or "")

    n_text_ctx = getattr(getattr(asr_model, "dims", None), "n_text_ctx", 448)
    max_text_len = max(0, int(n_text_ctx) - len(prefix) - 1)
    if len(text_ids) > max_text_len:
        text_ids = text_ids[:max_text_len]

    eot_id = getattr(tok, "eot", None)
    if eot_id is None:
        raise RuntimeError("Tokenizer does not provide EOT token id")
    full = prefix + text_ids + [eot_id]

    if device is None:
        try:
            device = next(asr_model.parameters()).device
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = torch.tensor([full], dtype=torch.long, device=device)
    input_tokens = tokens[:, :-1].contiguous()
    target_tokens = tokens[:, 1:].contiguous()

    info = {
        "n_text_ctx": int(n_text_ctx),
        "prefix_len": len(prefix),
        "text_token_len": len(text_ids),
        "total_len": int(tokens.size(1)),
        "language": language,
        "task": task,
    }
    return tokens, input_tokens, target_tokens, info


def compute_asr_band_sensitivity_from_gt(
    asr_model,
    mel: Tensor,
    input_tokens: Tensor,
    target_tokens: Tensor,
    reduce: str = "sum",
    valid_T: Optional[int] = None,
):
    asr_model.eval()

    x = asr_model.encoder(mel)
    logits = asr_model.decoder(input_tokens, x)  # [B, S, vocab]

    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab), target_tokens.view(-1))

    if mel.grad is not None:
        mel.grad.zero_()
    for p in asr_model.parameters():
        if p.grad is not None:
            p.grad = None
    loss.backward()

    grad_mel = mel.grad.detach()  # [1, n_mels, T]
    sal = grad_mel.abs()

    if valid_T is not None:
        vT = max(1, min(int(valid_T), sal.size(-1)))
        sal = sal[:, :, :vT]

    if reduce == "l2":
        band = torch.linalg.vector_norm(sal, dim=-1)[0]
    elif reduce == "mean":
        band = sal.mean(dim=-1)[0]
    else:
        band = sal.sum(dim=-1)[0]

    return band, grad_mel, loss.detach()


def select_bins_from_bands(
    band_j: Tensor,
    band_a: Tensor,
    k: int = 48,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    assert band_j.ndim == 1 and band_a.ndim == 1 and band_j.numel() == band_a.numel(), "band shape mismatch"

    bj = band_j.to(torch.float32)
    ba = band_a.to(torch.float32)

    bj = bj / (bj.sum() + eps)
    ba = ba / (ba.sum() + eps)

    scores = bj / (ba + eps)
    k_eff = min(int(k), int(scores.numel()))
    vals, idx = torch.topk(scores, k=k_eff)

    mask = torch.zeros_like(scores)
    mask[idx] = 1.0

    return {"scores": scores, "topk_idx": idx, "topk_scores": vals, "mask": mask}


def compute_sample_bands_and_topk(
    tokenizer,
    model,
    asr_model,
    audio_path: str,
    text_gt: str,
    prompt: str,
    target: str,
    k: int = 48,
) -> Dict[str, Tensor]:
    device = next(model.parameters()).device

    input_ids, labels = build_chat_ids_and_labels(tokenizer, prompt, target)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    wav = load_wav(audio_path)
    speech, speech_lengths = wav_to_omni2_speech(wav, device=device, dtype=getattr(model, "dtype", torch.bfloat16))

    avg_logprob, grad_speech, _ = forward_avg_logprob_and_grad(model, input_ids, labels, speech, speech_lengths)
    band_j = compute_band_importance(grad_speech, speech, mode="abs", reduce="sum", topk=k)["freq_importance"]

    mel, _ = load_mel_from_audio(asr_model, audio_path, device=next(asr_model.parameters()).device, requires_grad=True)
    _, input_tokens, target_tokens, _ = build_whisper_tokens_from_text(asr_model, text_gt, device=next(asr_model.parameters()).device)
    band_a, grad_mel, loss_asr = compute_asr_band_sensitivity_from_gt(asr_model, mel, input_tokens, target_tokens, reduce="sum", valid_T=None)

    if band_j.numel() != 128 or band_a.numel() != 128:
        raise ValueError(f"Expected 128 mel bins; got band_j={int(band_j.numel())} band_a={int(band_a.numel())}")

    sel = select_bins_from_bands(band_j, band_a, k=k)

    return {
        "band_j": band_j.detach(),
        "band_a": band_a.detach(),
        "sel_scores": sel["scores"].detach(),
        "topk_idx": sel["topk_idx"].detach(),
        "topk_scores": sel["topk_scores"].detach(),
        "mask": sel["mask"].detach(),
        "asr_loss": loss_asr.detach(),
        "omni2_avg_logprob": avg_logprob.detach(),
    }


def batch_compute_global_mask_for_dataset(
    tokenizer,
    model,
    asr_path: str,
    train_file: str,
    prompt: str,
    target: str,
    k: int = 48,
) -> Dict[str, Tensor]:
    device = next(model.parameters()).device

    asr_model = whisper.load_model(asr_path)
    asr_model.to(device)
    asr_model.eval()
    for p in asr_model.parameters():
        p.requires_grad_(False)

    with open(train_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    Fbins = 128
    counts = torch.zeros(Fbins, dtype=torch.float32, device=device)
    weights = torch.zeros(Fbins, dtype=torch.float32, device=device)

    for record in tqdm(records, desc="LLaMA-Omni2 band selection"):
        ap = record.get("audio_path")
        txt = record.get("prompt")
        if not ap:
            continue
        if txt is None:
            txt = ""

        out = compute_sample_bands_and_topk(
            tokenizer,
            model,
            asr_model,
            audio_path=ap,
            text_gt=txt,
            prompt=prompt,
            target=target,
            k=k,
        )

        topk_idx = out["topk_idx"].to(device)
        topk_scores = out["topk_scores"].to(device)

        counts.index_add_(0, topk_idx, torch.ones_like(topk_scores))
        weights.index_add_(0, topk_idx, topk_scores)

    k_eff = min(int(k), Fbins)
    vals_c, idx_c = torch.topk(counts, k=k_eff)
    vals_w, idx_w = torch.topk(weights, k=k_eff)
    mask_c = torch.zeros(Fbins, dtype=torch.float32, device=device)
    mask_w = torch.zeros(Fbins, dtype=torch.float32, device=device)
    mask_c[idx_c] = 1.0
    mask_w[idx_w] = 1.0
    return {
        "counts": counts.detach().cpu(),
        "weights": weights.detach().cpu(),
        "top_idx_counts": idx_c.detach().cpu(),
        "top_idx_weights": idx_w.detach().cpu(),
        "mask_counts": mask_c.detach().cpu(),
        "mask_weights": mask_w.detach().cpu(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaMA-Omni2 mel band selection (batch only)")

    parser.add_argument("--llama_omni2_root", type=str, default="../LLaMA-Omni2")
    parser.add_argument("--model_path", type=str, default="../data/models/Llama-Omni2")
    parser.add_argument("--asr_path", type=str, default="../data/models/Whisper-large-v3/large-v3.pt")
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--out_dir", type=str, default="./Llama2/logs")

    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    tok, model, device = load_model_and_tokenizer(
        llama_omni2_root=args.llama_omni2_root,
        model_path=os.path.expanduser(args.model_path),
        device=args.device,
        torch_dtype=dtype_map[args.dtype],
    )

    os.makedirs(args.out_dir, exist_ok=True)

    masks = batch_compute_global_mask_for_dataset(
        tok,
        model,
        asr_path=args.asr_path,
        train_file=args.train_file,
        prompt=args.prompt,
        target=args.target,
        k=args.k,
    )

    np.savez(
        os.path.join(args.out_dir, "topk_count.npz"),
        mask=masks["mask_counts"].numpy(),
        top_idx=masks["top_idx_counts"].numpy(),
        scores=masks["counts"].numpy(),
        counts=masks["counts"].numpy(),
    )
    np.savez(
        os.path.join(args.out_dir, "topk_weighted.npz"),
        mask=masks["mask_weights"].numpy(),
        top_idx=masks["top_idx_weights"].numpy(),
        scores=masks["weights"].numpy(),
        weights=masks["weights"].numpy(),
    )
