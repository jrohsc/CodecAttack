#!/usr/bin/env python3
"""External-ASR anchor: a differentiable "make this audio actually SAY the
target" loss, for the black-box transfer arm.

Why this exists
---------------
The encoder-agreement probe (2026-08-14) showed that our working attack does not
route through the audio encoder at all. `ce_zeros` scores 50/50 white-box while
moving its OWN source encoder AWAY from the target -- delta cos = -0.0416
(probe, n=10) and -0.0919 (the attack driver's own logged cosine, n=50), two
independent instruments agreeing in sign. The converse holds too: the pure
encoder-cosine arms reach cos 0.86-0.88 and score 0/50 even white-box.

So encoder similarity is neither necessary nor sufficient. The CE attack finds a
model-specific direction that steers the victim's LM readout, with no
target-like audio content anywhere in the representation. A readout-steering
direction is an idiosyncrasy of one model's loss landscape, which is precisely
why transfer is 0/12426 and why the attack is prompt-bound (88 -> 0 on a prompt
swap). No amount of eps, ensembling or MI/DI makes an idiosyncrasy shared.

To transfer, the attack has to inject something the victims actually SHARE.
The only such thing is speech content. This module supplies the objective that
targets content directly.

Design choices, and why
-----------------------
* **Discrete criterion, not cosine.** ARO (arXiv 2503.19591) aligns to an
  external representation model via cosine. In our setting cosine is a measured
  dead end: the readability gate is 0.8384 (max over UNRELATED spoken commands,
  n=132) and our enc arms sit at 0.86 -- inside noise. So we anchor on
  transcription (teacher-forced CE over target tokens), which is verifiable:
  either Whisper emits the target string or it does not.
* **Whisper-large-v3 as the anchor.** No victim was fine-tuned as this
  checkpoint, so it is genuinely external in ARO's sense -- but all four victims
  carry Whisper-large-v3-DERIVED towers, so content this model resolves is
  content their front-ends are built to resolve. That lineage is the bet.
* **Keep the source CE term.** ARO never uses the representation loss alone, and
  our enc-alone arms confirm why (0/50 white-box). Intended use is `asr_ce`.

The mel here is a differentiable torch reimplementation of
WhisperFeatureExtractor -- HF's extractor runs in numpy and would cut the graph.
It follows the same recipe already used by the model wrappers in
whisper-inject-v2-fork (zero-pad the RAW waveform to 30 s before STFT, power
spectrogram, processor's own mel filters, log10 + 8 dB floor + /4 normalize,
drop the last frame).

Self-test (verifies the mel matches HF's and that the loss is differentiable):
    python asr_anchor.py --self-test --wav <some_speech>.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DEFAULT_WHISPER = ("${HOME}/.cache/huggingface/hub/"
                   "models--openai--whisper-large-v3/snapshots/"
                   "06f233fe06e710322aca913c1bc4249a0d71fce1")

SAMPLE_RATE = 16000
N_SAMPLES = 480000      # 30 s
N_FFT = 400
HOP = 160


class WhisperASRAnchor:
    """Differentiable 'this audio transcribes to TARGET' loss."""

    # bfloat16 by default: the anchor is frozen and only needs to supply a
    # gradient DIRECTION to the audio, and the rest of this pipeline (EnCodec
    # decoder, Qwen2-Audio) is already bf16. fp32 weights + fp32 activations
    # over the 1500-position encoder push a co-resident attack past 80 GB.
    def __init__(self, model_path: str = DEFAULT_WHISPER,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        self.device = device
        self.dtype = dtype
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype).to(device).eval()
        # The anchor is frozen -- gradients flow to the audio, never the weights.
        for p in self.model.parameters():
            p.requires_grad_(False)

        fe = self.processor.feature_extractor
        mel_np = np.array(fe.mel_filters)          # (n_freqs, n_mels)
        self.mel_filters = torch.from_numpy(mel_np.T).to(device, torch.float32)
        self.hann = torch.hann_window(N_FFT, device=device)
        self.n_mels = self.mel_filters.shape[0]
        # whisper-large-v3 is 128-mel; earlier checkpoints are 80. Take it from
        # the checkpoint rather than assuming, so a medium/small fallback works.
        assert self.n_mels == self.model.config.num_mel_bins, (
            f"mel filter bank has {self.n_mels} bins but the model expects "
            f"{self.model.config.num_mel_bins}")

        tok = self.processor.tokenizer
        if hasattr(tok, "set_prefix_tokens"):
            tok.set_prefix_tokens(language="en", task="transcribe")
        self.tok = tok
        self._label_cache: dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------- mel
    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """[T] or [1,T] @16k -> [1, n_mels, 3000], differentiable."""
        if wav.dim() > 1:
            wav = wav.squeeze(0)
        wav = wav.to(self.device, torch.float32)

        # Pad/trim the RAW waveform to 30 s, matching the HF extractor. Padding
        # the mel afterwards instead would put the boundary in the wrong place.
        if wav.shape[0] < N_SAMPLES:
            wav = torch.cat([wav, wav.new_zeros(N_SAMPLES - wav.shape[0])])
        elif wav.shape[0] > N_SAMPLES:
            wav = wav[:N_SAMPLES]

        stft = torch.stft(wav, n_fft=N_FFT, hop_length=HOP,
                          window=self.hann, return_complex=True)
        magnitudes = stft.abs() ** 2
        mel = self.mel_filters @ magnitudes

        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec[:, :-1]                  # drop last frame
        return log_spec.unsqueeze(0)

    # -------------------------------------------------------------- labels
    def _labels(self, text: str) -> torch.Tensor:
        if text not in self._label_cache:
            ids = self.tok(text, return_tensors="pt").input_ids
            # HF shifts labels right internally and prepends
            # decoder_start_token_id, so a leading <|startoftranscript|> here
            # would be duplicated.
            start = self.model.config.decoder_start_token_id
            if ids.shape[1] > 0 and ids[0, 0].item() == start:
                ids = ids[:, 1:]
            self._label_cache[text] = ids.to(self.device)
        return self._label_cache[text]

    # ---------------------------------------------------------------- loss
    def loss(self, wav_16k: torch.Tensor, target_text: str) -> torch.Tensor:
        """Teacher-forced CE for `target_text` under the frozen anchor."""
        mel = self.wav_to_mel(wav_16k).to(self.dtype)
        labels = self._labels(target_text)
        out = self.model(input_features=mel, labels=labels)
        return out.loss.float()

    @torch.no_grad()
    def transcribe(self, wav_16k: torch.Tensor, max_new_tokens: int = 96) -> str:
        """What the anchor actually hears -- the verification the CE loss can't give."""
        mel = self.wav_to_mel(wav_16k).to(self.dtype)
        ids = self.model.generate(input_features=mel,
                                  max_new_tokens=max_new_tokens,
                                  language="en", task="transcribe")
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _self_test(wav_path: str, model_path: str, device: str):
    import soundfile as sf
    import torchaudio.functional as AF

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    w = torch.from_numpy(audio).to(device)
    if sr != SAMPLE_RATE:
        w = AF.resample(w, sr, SAMPLE_RATE)

    anchor = WhisperASRAnchor(model_path, device=device)
    print(f"n_mels={anchor.n_mels}  dtype={anchor.dtype}")

    mel = anchor.wav_to_mel(w)
    print(f"our mel   : {tuple(mel.shape)}  mean={mel.mean():.4f} std={mel.std():.4f}")

    # Agreement with the real (numpy) extractor -- catches a wrong window,
    # wrong power, or an off-by-one in the frame drop.
    ref = anchor.processor.feature_extractor(
        w.detach().cpu().numpy(), sampling_rate=SAMPLE_RATE,
        return_tensors="pt").input_features.to(device)
    L = min(mel.shape[-1], ref.shape[-1])
    mad = (mel[..., :L].float() - ref[..., :L].float()).abs().mean().item()
    print(f"HF mel    : {tuple(ref.shape)}  mean abs diff vs ours = {mad:.6f}")
    print("  -> mel MATCHES HF" if mad < 5e-3 else
          "  -> !! MEL MISMATCH: the anchor would optimize the wrong features")

    heard = anchor.transcribe(w)
    print(f"transcribes as: {heard!r}")

    # Differentiability: gradient must reach the waveform.
    w2 = w.clone().requires_grad_(True)
    loss = anchor.loss(w2, "test target phrase")
    loss.backward()
    g = w2.grad
    ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"loss={loss.item():.4f}  grad_norm={g.abs().sum().item():.4e}")
    print("  -> gradient reaches the waveform" if ok else
          "  -> !! NO GRADIENT: loss is not differentiable w.r.t. audio")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--wav", required=True)
    p.add_argument("--model", default=DEFAULT_WHISPER)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.self_test:
        _self_test(a.wav, a.model, a.device)
