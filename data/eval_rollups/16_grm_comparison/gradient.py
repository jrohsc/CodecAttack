from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import torch.nn.functional as F
import librosa
import whisper
import json
import inspect
import os
import numpy as np
# openai-whisper is needed ONLY by the mask-generation path below (the ASR
# utility-sensitivity gradient). We reproduce with the authors' RELEASED mask
# (logs/top48_weighted.npz), so main.py must import this module without it --
# and the codec-attack env has a `whisper.py` that shadows the package. Fail at
# use, not at import, so a missing dep can never be mistaken for a silent
# fallback to some other mask.
try:
    from whisper.audio import (load_audio as whisper_load_audio, log_mel_spectrogram,
                               pad_or_trim, SAMPLE_RATE as WHISPER_SR)
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    WHISPER_SR = 16000

    def _no_whisper(*a, **k):
        raise ImportError(
            "openai-whisper is required for GRM mask generation (gradient.py). "
            "Reproduction uses the released mask instead; install openai-whisper "
            "only if you intend to regenerate masks.")
    whisper_load_audio = log_mel_spectrogram = pad_or_trim = _no_whisper

def load_model_and_processor(
    model_path: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model.to(device)

    return processor, model, device

def build_chat_texts(processor: AutoProcessor, prompt: str, target: str) -> Tuple[str, str, int]:
    conv_user_only = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "DUMMY_AUDIO_PATH"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    conv_with_answer = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "DUMMY_AUDIO_PATH"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]
    prompt_text = processor.apply_chat_template(
        conv_user_only, add_generation_prompt=True, tokenize=False
    )
    full_text = processor.apply_chat_template(
        conv_with_answer, add_generation_prompt=False, tokenize=False
    )
    tok = processor.tokenizer
    prefix_ids = tok(prompt_text, add_special_tokens=False, return_tensors=None)["input_ids"]
    prefix_len = len(prefix_ids)
    return prompt_text, full_text, prefix_len

def _detect_feature_key(inputs: Dict[str, Tensor]) -> str:
    candidate_keys = ["input_features", "audio_values", "audio_features"]
    for k in candidate_keys:
        v = inputs.get(k, None)
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return k
    raise KeyError("No audio feature tensor found in processor outputs.")


def call_processor(processor, *, text, wavs, sampling_rate, **kw):
    """Invoke the Qwen2-Audio processor across the `audios`/`audio` rename.

    transformers <=4.4x took `audios=`; >=4.5x takes `audio=` and merely WARNS
    on the old name ("Keyword argument `audios` is not a valid argument for
    this processor and will be ignored") -- it then returns no `input_features`
    at all. That is a silent-failure shape: the attack would train against a
    batch with no audio rather than crash, so pick the name the installed
    processor actually accepts.
    """
    try:
        params = inspect.signature(processor.__call__).parameters
        key = "audios" if "audios" in params else "audio"
    except (TypeError, ValueError):
        key = "audio"
    return processor(text=text, sampling_rate=sampling_rate, **{key: wavs}, **kw)

def prepare_inputs_from_audio(
    processor: AutoProcessor,
    full_text: str,
    audio: Tensor,
    sampling_rate: int,
    device: str = "cpu",
) -> Tuple[Dict[str, Tensor], str]:
    audios = [audio.to(torch.float32).cpu().numpy()]
    inputs = processor(
        text=full_text,
        audios=audios,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    feature_key = _detect_feature_key(inputs)
    inputs[feature_key] = inputs[feature_key].contiguous().detach().requires_grad_(True)
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
        bsz = labels.size(0)
        for b in range(bsz):
            mask = labels[b] != -100
            if mask.any():
                last_idx = torch.nonzero(mask, as_tuple=False)[-1].item()
                if int(labels[b, last_idx].item()) in eos_token_ids:
                    labels[b, last_idx] = -100
    return labels


def build_labels_suffix(input_ids, suffix_len, eos_token_ids=None, drop_trailing_eos=True):
    """Supervise exactly the trailing target block, counting from the END.

    build_labels() masks a PREFIX of `prefix_len` tokens, where prefix_len is
    measured on chat text in which the audio is a single `<|AUDIO|>` token. On
    transformers >= 4.5x the processor expands that placeholder into ~130 audio
    tokens, so input_ids is far longer than the text-derived prefix and the
    original slice supervises audio placeholders instead of the target -- CE
    sits at ~17 and never descends. The tokenizer also pads LEFT here, so any
    fixed offset from the front is wrong per row anyway.

    The target block is the last `suffix_len` tokens of every row regardless of
    padding or audio expansion, so index from the end.
    """
    labels = torch.full_like(input_ids, -100)
    n = int(suffix_len)
    if n > 0:
        labels[:, -n:] = input_ids[:, -n:]
    if drop_trailing_eos and eos_token_ids:
        for b in range(labels.size(0)):
            mask = labels[b] != -100
            if mask.any():
                last_idx = torch.nonzero(mask, as_tuple=False)[-1].item()
                while last_idx >= 0 and int(input_ids[b, last_idx].item()) in eos_token_ids:
                    labels[b, last_idx] = -100
                    last_idx -= 1
    return labels

def forward_avg_logprob_and_grad(
    model: Qwen2AudioForConditionalGeneration,
    inputs: Dict[str, Tensor],
    feature_key: str,
    labels: Tensor,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    model.eval()
    run_inputs = {k: v for k, v in inputs.items()}
    run_inputs["labels"] = labels
    run_inputs["use_cache"] = False

    outputs = model(**run_inputs)
    loss = outputs.loss
    avg_logprob = -loss

    if run_inputs[feature_key].grad is not None:
        run_inputs[feature_key].grad.zero_()
    model.zero_grad(set_to_none=True)
    avg_logprob.backward()

    grad_features = run_inputs[feature_key].grad.detach()
    extra = {"loss": loss.detach(), "logits": outputs.logits.detach()}
    return avg_logprob.detach(), grad_features, extra

def compute_band_importance(
    grad_features: Tensor,
    input_features: Optional[Tensor] = None,
    mode: str = "abs",
    valid_T: Optional[int] = None,
    reduce: str = "sum",
    topk: int = 10,
) -> Dict[str, Tensor]:
    """
    Compute per-frequency-band importance and return top-k bands only.
    Input/output frequency axis is Mel bins (e.g., 128). No file I/O.
    """
    assert grad_features.dim() == 3 and grad_features.size(0) == 1
    g = grad_features[0]
    if mode == "grad_x_input":
        if input_features is None:
            raise ValueError("input_features is required for grad_x_input mode")
        assert input_features.dim() == 3 and input_features.size(0) == 1
        x = input_features[0]
        sal = (g * x).abs()
    elif mode == "l2":
        sal = g.pow(2)
    else:
        sal = g.abs()

    if valid_T is not None:
        valid_T = max(1, min(int(valid_T), sal.size(-1)))
        sal = sal[:, :valid_T]

    if reduce == "l2":
        freq_importance = torch.linalg.vector_norm(sal, dim=-1)
    else:
        freq_importance = sal.sum(dim=-1)

    k = min(topk, freq_importance.numel())
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

    # Load audio and convert to Mel (Whisper frontend at fixed 16kHz)
    audio = whisper_load_audio(audio_path)  # auto-converts to 16kHz mono float32
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
    # Build tokenizer (multilingual, English transcription, no timestamps)
    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language=language, task=task)

    # Special prefix: SOT, language token, no_timestamps
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

    # Encode main text
    text_ids = tok.encode(text or "")

    # Sequence length upper bound (decoder context), default 448
    n_text_ctx = getattr(getattr(asr_model, "dims", None), "n_text_ctx", 448)
    max_text_len = max(0, int(n_text_ctx) - len(prefix) - 1)  # reserve EOT
    truncated = False
    if len(text_ids) > max_text_len:
        text_ids = text_ids[:max_text_len]
        truncated = True

    # Build full sequence: prefix + text + EOT
    eot_id = getattr(tok, "eot", None)
    if eot_id is None:
        # Compatibility fallback (should not happen in practice)
        raise RuntimeError("Tokenizer does not provide EOT token id")
    full = prefix + text_ids + [eot_id]

    # Move to device
    if device is None:
        try:
            device = next(asr_model.parameters()).device
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = torch.tensor([full], dtype=torch.long, device=device)

    # Teacher forcing: input/target shift
    input_tokens = tokens[:, :-1].contiguous()
    target_tokens = tokens[:, 1:].contiguous()

    info = {
        "n_text_ctx": int(n_text_ctx),
        "prefix_len": len(prefix),
        "text_token_len": len(text_ids),
        "total_len": int(tokens.size(1)),
        "truncated": bool(truncated),
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

    # Forward pass: encode audio, then decode text
    x = asr_model.encoder(mel)
    logits = asr_model.decoder(input_tokens, x)  # [B, S, vocab]

    # Cross-entropy loss (flattened over steps)
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab), target_tokens.view(-1))

    # Backpropagate to mel
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

def estimate_active_span(features: Tensor, energy_thresh_ratio: float = 0.05, min_frames: int = 50, margin: int = 10) -> Tuple[int, int]:
    assert features.dim() == 3 and features.size(0) == 1, "expect [1, F, T]"
    x = features[0].abs().mean(dim=0)  # [T]
    T = x.numel()
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
    t0 = int(max(0, idx[0].item() - margin))
    t1 = int(min(T, idx[-1].item() + 1 + margin))
    if t1 - t0 < min_frames:
        center = (t0 + t1) // 2
        half = min_frames // 2
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
    assert band_j.ndim == 1 and band_a.ndim == 1 and band_j.numel() == band_a.numel(), "band shape mismatch"
    F = band_j.numel()
    k = int(max(1, min(int(k), F)))

    bj = band_j.to(torch.float32)
    ba = band_a.to(torch.float32)

    if normalize:
        bj = bj / (bj.sum() + eps)
        ba = ba / (ba.sum() + eps)

    tau = None
    if floor_q and 0 < floor_q < 1:
        try:
            tau = torch.quantile(ba, float(floor_q)).item()
        except Exception:
            tau = float(torch.topk(ba, max(1, int(F * floor_q)), largest=False).values.max().item())
        ba = torch.clamp(ba, min=float(tau))

    if mode == "diff":
        scores = (bj.pow(alpha)) - (ba.pow(beta))
    else:  # "ratio"
        scores = (bj.clamp_min(eps).pow(alpha)) / (ba.clamp_min(eps).pow(beta))

    vals, idx = torch.topk(scores, k=k)
    mask = torch.zeros(F, dtype=torch.float32, device=scores.device)
    mask[idx] = 1.0

    return {
        "scores": scores,
        "topk_idx": idx,
        "topk_scores": vals,
        "mask": mask,
        "band_jn": bj,
        "band_an": ba,
        "tau": torch.tensor(tau if tau is not None else 0.0),
    }

def compute_sample_bands_and_topk(
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration,
    asr_model,
    audio_path: str,
    text_gt: str,
    prompt: str,
    target: str,
    k: int = 48,
    ratio_alpha: float = 1.0,
    ratio_beta: float = 1.0,
    floor_q: float = 0.1,
    use_valid_t: bool = True,
) -> Dict[str, Tensor]:
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Qwen inputs
    prompt_text, full_text, prefix_len = build_chat_texts(processor, prompt, target)
    # Load waveform for Qwen processor path (keep sample rate that processor expects)
    sr = processor.feature_extractor.sampling_rate
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio = torch.tensor(wav, dtype=torch.float32)
    inputs, feature_key = prepare_inputs_from_audio(processor, full_text, audio, sr, device)

    # Labels for Qwen (mask out prompt prefix)
    tok = processor.tokenizer
    eos_token_ids: List[int] = []
    if getattr(tok, "eos_token_id", None) is not None:
        eos_token_ids.append(int(tok.eos_token_id))
    if getattr(tok, "eos_token_ids", None):
        eos_token_ids.extend([int(x) for x in (tok.eos_token_ids or [])])
    labels = build_labels(inputs["input_ids"], prefix_len, eos_token_ids, drop_trailing_eos=True)

    # Qwen grads → band_j
    avg_logprob, grad_features, _ = forward_avg_logprob_and_grad(model, inputs, feature_key, labels)
    valid_T_q = None
    if use_valid_t:
        t0, t1 = estimate_active_span(inputs[feature_key])
        valid_T_q = t1
    band_qwen = compute_band_importance(grad_features, inputs[feature_key], mode="abs", reduce="sum", topk=k, valid_T=valid_T_q)["freq_importance"]

    # ASR mel + tokens -> band_a (using already loaded asr_model)
    mel, _ = load_mel_from_audio(asr_model, audio_path, device=device, requires_grad=True)
    _, input_tokens, target_tokens, _ = build_whisper_tokens_from_text(asr_model, text_gt, device=device)
    valid_T_a = None
    if use_valid_t:
        t0a, t1a = estimate_active_span(mel)
        valid_T_a = t1a
    band_asr, grad_mel, loss_asr = compute_asr_band_sensitivity_from_gt(asr_model, mel, input_tokens, target_tokens, reduce="sum", valid_T=valid_T_a)

    sel = select_bins_from_bands(band_qwen, band_asr, k=k, mode="ratio", alpha=ratio_alpha, beta=ratio_beta, normalize=True, floor_q=floor_q)

    return {
        "band_j": band_qwen.detach(),
        "band_a": band_asr.detach(),
        "sel_scores": sel["scores"].detach(),
        "topk_idx": sel["topk_idx"],
        "topk_scores": sel["topk_scores"].detach(),
        "mask": sel["mask"].detach(),
        "asr_loss": loss_asr,
        "qwen_avg_logprob": avg_logprob,
    }

def batch_compute_global_mask_for_dataset(
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration,
    asr_path: str,
    train_file: str,
    prompt: str,
    target: str,
    k: int = 48,
    mode: str = "ratio",
    alpha: float = 1.0,
    beta: float = 1.0,
    floor_q: float = 0.1,
    use_valid_t: bool = True,
) -> Dict[str, Tensor]:
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Preload Whisper model to avoid repeated loading in loop
    print("[INFO] Loading Whisper model...")
    asr_model = whisper.load_model(asr_path)
    asr_model.eval()
    for p in asr_model.parameters():
        p.requires_grad_(False)
    print("[INFO] Whisper model loaded.")

    with open(train_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    F = 128  
    counts = torch.zeros(F, dtype=torch.float32, device=device)
    weights = torch.zeros(F, dtype=torch.float32, device=device)

    print(f"[INFO] Processing {len(records)} samples...")
    for idx, record in enumerate(records):
        ap = record['audio_path']
        txt = record['prompt']
        print(f"[{idx+1}/{len(records)}] Processing: {ap}")
        try:
            out = compute_sample_bands_and_topk(
                processor, model, asr_model,
                audio_path=ap, text_gt=txt,
                prompt=prompt, target=target,
                k=k, ratio_alpha=alpha, ratio_beta=beta,
                floor_q=floor_q, use_valid_t=use_valid_t,
            )
            topk_idx = out["topk_idx"].detach().cpu()
            topk_scores = out["topk_scores"].detach().cpu()
            # Count-based aggregation
            counts.index_add_(0, topk_idx.to(counts.device), torch.ones_like(topk_scores, device=counts.device))
            # Weighted aggregation (accumulate top-k scores)
            weights.index_add_(0, topk_idx.to(weights.device), topk_scores.to(weights.device))
        except Exception as e:
            print(f"[WARN] sample {idx} failed: {e}")

    # Global selection
    k_eff = min(k, F)
    vals_c, idx_c = torch.topk(counts, k=k_eff)
    vals_w, idx_w = torch.topk(weights, k=k_eff)
    mask_c = torch.zeros(F, dtype=torch.float32, device=device); mask_c[idx_c] = 1.0
    mask_w = torch.zeros(F, dtype=torch.float32, device=device); mask_w[idx_w] = 1.0

    # Always save two NPZ files to fixed output directory
    out_dir = "./logs"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "topk_count.npz"),
        mask=mask_c.detach().cpu().numpy(),
        top_idx=idx_c.detach().cpu().numpy(),
        scores=vals_c.detach().cpu().numpy(),
        counts=counts.detach().cpu().numpy(),
    )
    np.savez(
        os.path.join(out_dir, "topk_weighted.npz"),
        mask=mask_w.detach().cpu().numpy(),
        top_idx=idx_w.detach().cpu().numpy(),
        scores=vals_w.detach().cpu().numpy(),
        weights=weights.detach().cpu().numpy(),
    )

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
    parser = argparse.ArgumentParser(description="Saliency and mask computation (single-sample or dataset batch)")
    # common
    parser.add_argument("--model_path", type=str, default="../data/models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--asr_path", type=str, default="../data/models/Whisper-large-v3/large-v3.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto","fp16","bf16","fp32"])
    parser.add_argument("--topk", type=int, default=48)
    parser.add_argument("--floor_q", type=float, default=0.1)
    parser.add_argument("--use_valid_t", action="store_true")
    parser.add_argument("--prompt", type=str, default="Provide a detailed answer that exceeds 500 words.")
    parser.add_argument("--target", type=str, default="Sure! Here is a step-by-step guide:")
    parser.add_argument("--run_batch", action="store_true")
    parser.add_argument("--train_file", type=str, default="./train.jsonl")
    args = parser.parse_args()

    # dtype
    torch_dtype = None
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32

    # Load Qwen
    processor, model, device = load_model_and_processor(args.model_path, device=args.device or None, torch_dtype=torch_dtype)

    # Batch mode: compute global mask over dataset with fixed settings (alpha=beta=1, ratio)
    if getattr(args, "run_batch", False):
        out = batch_compute_global_mask_for_dataset(
            processor, model, args.asr_path,
            train_file=args.train_file,
            prompt=args.prompt,
            target=args.target,
            k=args.topk,
            mode="ratio",
            alpha=1.0,
            beta=1.0,
            floor_q=0.1,
            use_valid_t=getattr(args, "use_valid_t", False),
        )
        print("Top-{} by counts:".format(args.topk), out["top_idx_counts"].tolist())
        print("Top-{} by weighted:".format(args.topk), out["top_idx_weights"].tolist())
        import sys
        sys.exit(0)
