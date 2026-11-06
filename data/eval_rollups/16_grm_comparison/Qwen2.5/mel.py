from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _import_openai_whisper():
    try:
        import whisper
    except Exception as e:
        raise ModuleNotFoundError(
            "Missing dependency: openai-whisper. Install with: pip install -U openai-whisper"
        ) from e

    if not hasattr(whisper, "load_model"):
        raise ImportError(
            "You installed the wrong package named 'whisper' (it has no whisper.load_model). "
            "Uninstall it and install OpenAI Whisper instead:\n"
            "  pip uninstall -y whisper\n"
            "  pip install -U openai-whisper"
        )

    return whisper


def load_model_and_processor(
    model_path: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    model.eval()
    if hasattr(model, "thinker"):
        try:
            model.thinker.eval()
        except Exception:
            pass

    for p in model.parameters():
        p.requires_grad_(False)

    model.to(device)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return processor, model, device


def build_chat_texts(processor, prompt: str, target: str) -> Tuple[str, str, int]:
    conv_user_only = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "_"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    conv_with_answer = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "_"},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]

    prompt_text = processor.apply_chat_template(
        conv_user_only, add_generation_prompt=True, tokenize=False
    )
    full_text = processor.apply_chat_template(
        conv_with_answer, add_generation_prompt=False, tokenize=False
    )

    tok = processor.tokenizer
    prefix_ids = tok(prompt_text, add_special_tokens=False, return_tensors=None)["input_ids"]
    prefix_len = int(len(prefix_ids))

    return prompt_text, full_text, prefix_len


def _detect_feature_key(inputs: Dict[str, Any]) -> str:
    candidate_keys = ["input_features", "audio_values", "audio_features"]
    for k in candidate_keys:
        v = inputs.get(k, None)
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    raise KeyError("No audio feature tensor found in processor outputs.")


def load_omni_features_from_dir(feature_dir: str, key: str) -> Tuple[Tensor, Optional[Tensor]]:
    path = f"{feature_dir}/{key}.pt"
    obj = torch.load(path, map_location="cpu")
    feats = obj["input_features"]
    fam = obj.get("feature_attention_mask", None)
    return feats, fam


def prepare_inputs_from_omni_features(
    processor,
    full_text: str,
    input_features: Tensor,
    feature_attention_mask: Optional[Tensor],
    device: str,
    truncate_audio_features: Optional[int] = None,
) -> Tuple[Dict[str, Any], str]:
    tok_out = processor.tokenizer(
        [full_text],
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    inputs: Dict[str, Any] = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tok_out.items()
    }

    feats = input_features
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    if feats.dim() != 3 or feats.size(0) != 1:
        raise ValueError("input_features must have shape [F,T] or [1,F,T]")

    fam = feature_attention_mask
    if fam is not None and isinstance(fam, torch.Tensor) and fam.dim() == 1:
        fam = fam.unsqueeze(0)

    if truncate_audio_features is not None and int(truncate_audio_features) > 0:
        target_T = int(truncate_audio_features)
        if feats.size(-1) > target_T:
            feats = feats[..., :target_T]
            if fam is not None and isinstance(fam, torch.Tensor):
                fam = fam[:, :target_T]

    inputs["input_features"] = feats.to(device)
    if fam is not None and isinstance(fam, torch.Tensor):
        inputs["feature_attention_mask"] = fam.to(device)

    feature_key = _detect_feature_key(inputs)

    feat = inputs[feature_key]
    if not isinstance(feat, torch.Tensor):
        raise TypeError("Detected feature is not a Tensor")
    inputs[feature_key] = feat.contiguous().detach().requires_grad_(True)

    return inputs, feature_key


def build_labels(
    input_ids: Tensor,
    prefix_len: int,
    eos_token_ids: Optional[List[int]] = None,
    drop_trailing_eos: bool = True,
) -> Tensor:
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100

    if drop_trailing_eos and eos_token_ids:
        bsz = int(labels.size(0))
        for b in range(bsz):
            mask = labels[b] != -100
            if mask.any():
                last_idx = torch.nonzero(mask, as_tuple=False)[-1].item()
                if int(labels[b, last_idx].item()) in eos_token_ids:
                    labels[b, last_idx] = -100

    return labels


def forward_avg_logprob_and_grad(
    model,
    inputs: Dict[str, Any],
    feature_key: str,
    labels: Tensor,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    model.eval()

    run_inputs = dict(inputs)
    run_inputs["labels"] = labels
    run_inputs["use_cache"] = False

    core = getattr(model, "thinker", None)
    if core is None:
        core = model

    outputs = core(**run_inputs)
    loss = outputs.loss
    avg_logprob = -loss

    feat = run_inputs[feature_key]
    if isinstance(feat, torch.Tensor) and feat.grad is not None:
        feat.grad.zero_()

    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()

    avg_logprob.backward()

    grad_features = run_inputs[feature_key].grad.detach()
    extra: Dict[str, Tensor] = {"loss": loss.detach()}
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        extra["logits"] = outputs.logits.detach()

    return avg_logprob.detach(), grad_features, extra


def compute_band_importance(
    grad_features: Tensor,
    input_features: Optional[Tensor] = None,
    mode: str = "abs",
    valid_T: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    if grad_features.dim() != 3 or grad_features.size(0) != 1:
        raise ValueError("grad_features must have shape [1, F, T]")

    g = grad_features[0]
    if mode == "grad_x_input":
        if input_features is None:
            raise ValueError("grad_x_input requires input_features")
        if input_features.dim() != 3 or input_features.size(0) != 1:
            raise ValueError("input_features must have shape [1, F, T]")
        sal = (g * input_features[0]).abs()
    elif mode == "l2":
        sal = g.pow(2)
    else:
        sal = g.abs()

    if valid_T is not None:
        valid_T = max(1, min(int(valid_T), int(sal.size(-1))))
        sal = sal[:, :valid_T]

    if reduce == "l2":
        return torch.linalg.vector_norm(sal, dim=-1)
    if reduce == "mean":
        return sal.mean(dim=-1)
    return sal.sum(dim=-1)


def estimate_active_span(
    features: Tensor,
    energy_thresh_ratio: float = 0.05,
    min_frames: int = 50,
    margin: int = 10,
) -> Tuple[int, int]:
    if features.dim() != 3 or features.size(0) != 1:
        raise ValueError("estimate_active_span expects [1, F, T]")

    x = features[0].abs().mean(dim=0)
    T = int(x.numel())
    if T == 0:
        return 0, 0

    m = float(x.max().item())
    if m <= 0:
        return 0, T

    thr = m * float(energy_thresh_ratio)
    mask = x > thr
    if not bool(mask.any()):
        return 0, T

    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    t0 = int(max(0, int(idx[0].item()) - int(margin)))
    t1 = int(min(T, int(idx[-1].item()) + 1 + int(margin)))

    if t1 - t0 < int(min_frames):
        center = (t0 + t1) // 2
        half = int(min_frames) // 2
        t0 = max(0, center - half)
        t1 = min(T, center + half)

    return t0, t1


def select_bins_from_bands(
    band_j: Tensor,
    band_a: Tensor,
    k: int = 48,
    mode: str = "ratio",
    alpha: float = 1.0,
    beta: float = 1.0,
    normalize: bool = True,
    floor_q: float = 0.1,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    if band_j.ndim != 1 or band_a.ndim != 1 or band_j.numel() != band_a.numel():
        raise ValueError("band shape mismatch")

    F_ = int(band_j.numel())
    k = int(max(1, min(int(k), F_)))

    bj = band_j.to(torch.float32)
    ba = band_a.to(torch.float32)

    if normalize:
        bj = bj / (bj.sum() + eps)
        ba = ba / (ba.sum() + eps)
    tau_val: float = 0.0
    if floor_q and 0.0 < float(floor_q) < 1.0:
        try:
            tau_val = float(torch.quantile(ba, float(floor_q)).item())
        except Exception:
            kk = max(1, int(F_ * float(floor_q)))
            tau_val = float(torch.topk(ba, kk, largest=False).values.max().item())
        ba = torch.clamp(ba, min=tau_val)

    if mode == "diff":
        scores = (bj.clamp_min(eps).pow(alpha)) - (ba.clamp_min(eps).pow(beta))
    else:
        scores = (bj.clamp_min(eps).pow(alpha)) / (ba.clamp_min(eps).pow(beta))

    vals, idx = torch.topk(scores, k=k)
    mask = torch.zeros(F_, dtype=torch.float32, device=scores.device)
    mask[idx] = 1.0

    return {
        "scores": scores,
        "topk_idx": idx,
        "topk_scores": vals,
        "mask": mask,
        "band_jn": bj,
        "band_an": ba,
        "tau": torch.tensor(tau_val, device=scores.device),
    }


def load_mel_from_audio(
    asr_model,
    audio_path: str,
    device: Optional[str] = None,
    pad_to_30s: bool = True,
    requires_grad: bool = True,
):
    _import_openai_whisper()
    from whisper.audio import (
        load_audio as whisper_load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )

    if device is None:
        try:
            device = next(asr_model.parameters()).device
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    audio = whisper_load_audio(audio_path)
    if pad_to_30s:
        audio = pad_or_trim(audio)

    mel = log_mel_spectrogram(audio, asr_model.dims.n_mels)
    mel = mel.unsqueeze(0).to(device)
    mel = mel.detach().requires_grad_(bool(requires_grad))

    return mel


def build_whisper_tokens_from_text(
    asr_model,
    text: str,
    device: str,
):
    whisper = _import_openai_whisper()

    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")

    prefix: List[int] = []
    if hasattr(tok, "sot"):
        prefix.append(int(tok.sot))
    try:
        prefix.append(int(tok.lang_id("en")))
    except Exception:
        pass
    if hasattr(tok, "no_timestamps"):
        prefix.append(int(tok.no_timestamps))

    text_ids: List[int] = tok.encode(text or "")

    n_text_ctx = int(getattr(getattr(asr_model, "dims", None), "n_text_ctx", 448))
    max_text_len = max(0, int(n_text_ctx) - len(prefix) - 1)
    if len(text_ids) > max_text_len:
        text_ids = text_ids[:max_text_len]

    eot_id = getattr(tok, "eot", None)
    if eot_id is None:
        raise RuntimeError("Tokenizer does not provide EOT token id")

    full = prefix + text_ids + [int(eot_id)]

    tokens = torch.tensor([full], dtype=torch.long, device=device)
    input_tokens = tokens[:, :-1].contiguous()
    target_tokens = tokens[:, 1:].contiguous()

    return input_tokens, target_tokens


def compute_asr_band_sensitivity_from_gt(
    asr_model,
    mel: Tensor,
    input_tokens: Tensor,
    target_tokens: Tensor,
    valid_T: Optional[int] = None,
):
    asr_model.eval()

    x = asr_model.encoder(mel)
    logits = asr_model.decoder(input_tokens, x)

    vocab = int(logits.size(-1))
    loss = F.cross_entropy(logits.view(-1, vocab), target_tokens.view(-1))

    if mel.grad is not None:
        mel.grad.zero_()
    for p in asr_model.parameters():
        if p.grad is not None:
            p.grad = None

    loss.backward()

    grad_mel = mel.grad.detach()
    sal = grad_mel.abs()

    if valid_T is not None:
        vT = max(1, min(int(valid_T), int(sal.size(-1))))
        sal = sal[:, :, :vT]

    band = sal.sum(dim=-1)[0]

    return band, loss.detach()


def batch_compute_global_mask_for_dataset(
    processor,
    model,
    asr_path: str,
    train_file: str,
    prompt: str,
    target: str,
    k: int,
    floor_q: float,
    truncate_audio_features: Optional[int],
    omni_features_dir: str,
    asr_device: str,
    use_valid_t: bool,
    output_dir: str,
):
    import json
    import os
    import numpy as np

    whisper = _import_openai_whisper()

    qwen_device = next(model.parameters()).device

    asr_model = whisper.load_model(asr_path, device=asr_device)
    asr_model.eval()
    for p in asr_model.parameters():
        p.requires_grad_(False)

    with open(train_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    counts = torch.zeros(128, dtype=torch.float32, device=qwen_device)
    weights = torch.zeros(128, dtype=torch.float32, device=qwen_device)

    fail_cnt = 0
    first_err: Optional[str] = None

    for i, record in enumerate(records):
        key = record.get("key", "")
        ap = record.get("audio_path", "")
        txt = record.get("prompt", "")
        try:
            _, full_text, prefix_len = build_chat_texts(processor, prompt, target)

            feats, fam = load_omni_features_from_dir(omni_features_dir, key)
            inputs, feature_key = prepare_inputs_from_omni_features(
                processor,
                full_text,
                input_features=feats,
                feature_attention_mask=fam,
                device=str(qwen_device),
                truncate_audio_features=truncate_audio_features,
            )

            tok = processor.tokenizer
            eos_token_ids: List[int] = []
            if getattr(tok, "eos_token_id", None) is not None:
                eos_token_ids.append(int(tok.eos_token_id))
            extra_eos = getattr(tok, "eos_token_ids", None)
            if extra_eos is not None:
                if isinstance(extra_eos, (list, tuple, set)):
                    eos_token_ids.extend([int(x) for x in extra_eos])
                else:
                    eos_token_ids.append(int(extra_eos))

            labels = build_labels(inputs["input_ids"], prefix_len, eos_token_ids=eos_token_ids, drop_trailing_eos=True)

            avg_logprob, grad_features, _ = forward_avg_logprob_and_grad(model, inputs, feature_key, labels)

            valid_T_q = None
            if use_valid_t:
                _, t1 = estimate_active_span(inputs[feature_key])
                valid_T_q = t1
            band_j = compute_band_importance(grad_features, inputs[feature_key], mode="abs", reduce="sum", valid_T=valid_T_q)

            mel = load_mel_from_audio(asr_model, ap, device=asr_device, requires_grad=True)
            input_tokens, target_tokens = build_whisper_tokens_from_text(asr_model, txt, device=asr_device)

            valid_T_a = None
            if use_valid_t:
                _, t1a = estimate_active_span(mel)
                valid_T_a = t1a
            band_a, loss_asr = compute_asr_band_sensitivity_from_gt(asr_model, mel, input_tokens, target_tokens, valid_T=valid_T_a)
            band_a = band_a.to(device=qwen_device)

            sel = select_bins_from_bands(band_j, band_a, k=k, mode="ratio", alpha=1.0, beta=1.0, normalize=True, floor_q=floor_q)

            topk_idx = sel["topk_idx"].detach().to(qwen_device)
            topk_scores = sel["topk_scores"].detach().to(qwen_device)

            counts.index_add_(0, topk_idx, torch.ones_like(topk_scores))
            weights.index_add_(0, topk_idx, topk_scores)

        except Exception as e:
            fail_cnt += 1
            if first_err is None:
                first_err = f"sample[{i}] key={key} audio_path={ap} err={repr(e)}"

    if first_err is not None:
        print(f"[DEBUG] first_error: {first_err}")
    if fail_cnt > 0:
        print(f"[DEBUG] failed_samples={fail_cnt}/{len(records)}")

    if float(counts.sum().item()) == 0.0:
        raise RuntimeError("All counts are zero. See [DEBUG] first_error above.")

    k_eff = min(int(k), 128)
    vals_c, idx_c = torch.topk(counts, k=k_eff)
    vals_w, idx_w = torch.topk(weights, k=k_eff)

    mask_c = torch.zeros(128, dtype=torch.float32, device=qwen_device)
    mask_w = torch.zeros(128, dtype=torch.float32, device=qwen_device)
    mask_c[idx_c] = 1.0
    mask_w[idx_w] = 1.0

    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "topk_count.npz"), mask=mask_c.detach().cpu().numpy(), top_idx=idx_c.detach().cpu().numpy(), scores=vals_c.detach().cpu().numpy(), counts=counts.detach().cpu().numpy())
    np.savez(os.path.join(output_dir, "topk_weighted.npz"), mask=mask_w.detach().cpu().numpy(), top_idx=idx_w.detach().cpu().numpy(), scores=vals_w.detach().cpu().numpy(), weights=weights.detach().cpu().numpy())

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

    parser = argparse.ArgumentParser(description="Saliency and mask computation (Qwen2.5-Omni + Whisper, batch only)")

    parser.add_argument("--model_path", type=str, default="../data/models/Qwen2.5-Omni-7B")
    parser.add_argument("--asr_path", type=str, default="../data/models/Whisper-large-v3/large-v3.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--floor_q", type=float, default=0.1)
    parser.add_argument("--use_valid_t", action="store_true")
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--run_batch", action="store_true")
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./Qwen2.5")
    parser.add_argument("--truncate_audio_features", type=int, default=3000)
    parser.add_argument("--omni_features_dir", type=str, default="./Qwen2.5/omni_features")
    parser.add_argument("--asr_device", type=str, default="cpu")

    args = parser.parse_args()

    torch_dtype = None
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32

    processor, model, _ = load_model_and_processor(args.model_path, device=args.device, torch_dtype=torch_dtype)

    if args.run_batch:
        batch_compute_global_mask_for_dataset(
            processor,
            model,
            asr_path=args.asr_path,
            train_file=args.train_file,
            prompt=args.prompt,
            target=args.target,
            k=int(args.topk),
            floor_q=float(args.floor_q),
            truncate_audio_features=(int(args.truncate_audio_features) if int(args.truncate_audio_features) > 0 else None),
            omni_features_dir=args.omni_features_dir,
            asr_device=str(args.asr_device),
            use_valid_t=bool(args.use_valid_t),
            output_dir=args.output_dir,
        )
