"""
Differentiable reimplementation of Phi4MMAudioFeatureExtractor.

Phi-4-multimodal does NOT use a Whisper front-end. It uses SpeechLib-style
80-bin log filterbank features feeding a Conformer encoder trained from
scratch, which is why it adds encoder diversity to a Whisper-heavy
ensemble.

Spec replicated from processing_phi4mm.py (_extract_spectrogram /
_extract_features):
    win_length=400, hop_length=160, n_fft=512, hamming(400) window
    per-frame preemphasis 0.97 with first column clamped to itself
    waveform scaled by 32768 AFTER preemphasis
    power spectrum -> speechlib mel (80 bins, fmax=7690) -> clip(.,1.0) -> ln

The mel matrix itself is a constant, so it is built once with the model's
own numpy `speechlib_mel` and reused; only the wav->spectrogram path needs
to be differentiable.
"""

import numpy as np
import torch


class Phi4DifferentiableFrontend:
    WIN_LENGTH = 400
    HOP_LENGTH = 160
    N_FFT = 512
    PREEMPHASIS = 0.97
    SCALE = 32768.0

    def __init__(self, mel_matrix: np.ndarray, device="cuda", dtype=torch.float32):
        # mel_matrix: (n_freqs=257, n_mels=80), i.e. speechlib_mel(...).T
        self.device = device
        self.mel = torch.from_numpy(np.ascontiguousarray(mel_matrix)).to(device, torch.float32)
        self.window = torch.from_numpy(np.hamming(self.WIN_LENGTH)).to(device, torch.float32)

    @classmethod
    def from_feature_extractor(cls, fe, device="cuda"):
        """Build from a live Phi4MMAudioFeatureExtractor, reusing its own mel."""
        return cls(np.asarray(fe._mel), device=device)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [T] or [1, T] float, 16kHz -> log-fbank [n_frames, 80] (differentiable)."""
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        wav = wav.to(self.device, torch.float32)

        # SpeechLib drops the tail that cannot fill a whole window.
        n_batch = (wav.shape[0] - self.WIN_LENGTH) // self.HOP_LENGTH + 1
        if n_batch < 1:
            raise ValueError(f"audio too short: {wav.shape[0]} samples")

        idx = (torch.arange(self.WIN_LENGTH, device=self.device)[None, :]
               + self.HOP_LENGTH * torch.arange(n_batch, device=self.device)[:, None])
        frames = wav[idx]  # [n_batch, 400]  (gather keeps the graph)

        # np.roll(y,1) then y_prev[:,0]=y_prev[:,1] == prepend y[:,0] to y[:,:-1]
        prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        frames = (frames - self.PREEMPHASIS * prev) * self.SCALE

        spec = torch.fft.rfft(self.window * frames, n=self.N_FFT, dim=1).abs()
        fbank_power = torch.clamp(spec.pow(2) @ self.mel, min=1.0)
        return torch.log(fbank_power)
