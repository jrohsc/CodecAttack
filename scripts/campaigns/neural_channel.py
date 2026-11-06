"""
Neural OTA channel model: learns the nonlinear mapping original → recorded.

Trained on real paired recordings (MacBook speaker → air → iPhone mic).
Captures nonlinearities, time-varying AGC, phase distortion, and frequency
reshaping that a linear FIR filter cannot model.

Architecture: 1D U-Net with residual blocks. Input = original waveform,
output = predicted recorded waveform (after alignment).

Usage:
    # Train
    python neural_channel.py train --epochs 200 --lr 1e-4

    # Plug into attack
    from neural_channel import NeuralOTAChannel, load_trained_channel
    channel = load_trained_channel("checkpoints/neural_channel_best.pt")
    recorded_pred = channel(original_wav)  # differentiable
"""

import os
import glob
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture: 1D U-Net with residual convolution blocks
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """Residual block: two conv layers with GroupNorm and GELU."""

    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.block = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super().__init__()
        self.res = ResBlock1d(in_ch, kernel_size)
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        h = self.res(x)
        return self.down(h), h  # downsampled, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=7):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        # After concat: out_ch + skip_ch
        cat_ch = out_ch + skip_ch
        self.proj = nn.Conv1d(cat_ch, out_ch, 1)
        self.res = ResBlock1d(out_ch, kernel_size)

    def forward(self, x, skip):
        h = self.up(x)
        # Match lengths
        diff = skip.shape[-1] - h.shape[-1]
        if diff > 0:
            h = F.pad(h, (0, diff))
        elif diff < 0:
            h = h[..., :skip.shape[-1]]
        h = torch.cat([h, skip], dim=1)
        h = self.proj(h)
        h = self.res(h)
        return h


class NeuralChannelUNet(nn.Module):
    """
    1D U-Net that models the nonlinear OTA channel.

    Input:  [B, 1, T] original waveform at 24kHz
    Output: [B, 1, T] predicted recorded waveform

    The model learns:
    - Frequency-dependent gain/attenuation (speaker+mic response)
    - Phase distortion and smearing (room IR, multipath)
    - Nonlinear distortion (speaker clipping, mic AGC)
    - Time-varying effects via the convolutional receptive field
    """

    def __init__(self, channels=32, depth=4, kernel_size=7):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)

        # Encoder — track channel sizes for skip connections
        self.encoders = nn.ModuleList()
        enc_channels = []  # channel count at each encoder output (before down)
        ch = channels
        for i in range(depth):
            out_ch = min(ch * 2, 256)
            self.encoders.append(DownBlock(ch, out_ch, kernel_size))
            enc_channels.append(ch)  # skip comes from *before* downsampling
            ch = out_ch

        # Bottleneck with dilated convolutions for large receptive field
        self.bottleneck = nn.Sequential(
            ResBlock1d(ch, kernel_size, dilation=1),
            ResBlock1d(ch, kernel_size, dilation=2),
            ResBlock1d(ch, kernel_size, dilation=4),
        )

        # Decoder — mirror the encoder, with correct skip channel sizes
        self.decoders = nn.ModuleList()
        for i in range(depth):
            skip_ch = enc_channels[depth - 1 - i]
            out_ch = skip_ch  # decode back to the skip's channel count
            self.decoders.append(UpBlock(ch, skip_ch, out_ch, kernel_size))
            ch = out_ch

        self.output_conv = nn.Sequential(
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv1d(ch, 1, kernel_size, padding=kernel_size // 2),
        )

        # Residual connection: output = input + learned_distortion
        # This helps the model focus on learning the *difference* the channel adds
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] original audio
        Returns:
            [B, 1, T] predicted recorded audio
        """
        h = self.input_conv(x)

        skips = []
        for enc in self.encoders:
            h, skip = enc(h)
            skips.append(skip)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(h, skip)

        distortion = self.output_conv(h)

        # Trim/pad to match input
        if distortion.shape[-1] > x.shape[-1]:
            distortion = distortion[..., :x.shape[-1]]
        elif distortion.shape[-1] < x.shape[-1]:
            distortion = F.pad(distortion, (0, x.shape[-1] - distortion.shape[-1]))

        # Blend: learned residual connection
        w = torch.sigmoid(self.residual_weight)
        return w * x + (1 - w) * (x + distortion)

    def receptive_field(self):
        """Estimate the receptive field in samples."""
        # Each encoder level doubles it, dilated bottleneck expands further
        rf = 7  # input conv
        stride = 1
        for _ in self.encoders:
            rf += 7 * stride * 2  # res block + downconv
            stride *= 2
        rf += (7 * 1 + 7 * 2 + 7 * 4) * stride  # bottleneck dilated
        return rf


class NeuralOTAChannel(nn.Module):
    """
    Wrapper that loads a trained NeuralChannelUNet and provides
    a clean interface compatible with the attack pipeline.

    Supports:
    - severity parameter (0=bypass, 1=full channel)
    - Dropout-based EoT for training robustness
    - Eval mode (deterministic)
    """

    def __init__(self, model: NeuralChannelUNet, dropout: float = 0.1):
        super().__init__()
        self.model = model
        self.dropout = dropout

    def forward(self, x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        if severity < 0.01:
            return x

        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        elif x.dim() == 1:
            x = x.view(1, 1, -1)
            squeeze = True

        # Apply dropout to intermediate features during training (EoT effect)
        if self.training and self.dropout > 0:
            # Stochastic depth: randomly bypass channel with small probability
            if torch.rand(1).item() < self.dropout * 0.5:
                result = x
            else:
                result = self.model(x)
        else:
            result = self.model(x)

        # Blend with severity
        if severity < 1.0:
            result = severity * result + (1.0 - severity) * x

        if squeeze:
            result = result.squeeze(0)

        return result


# ---------------------------------------------------------------------------
# Data loading: collect all paired (original, recorded) files
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SC_DIR = _SCRIPT_DIR / "soundcloud_recorded"

PAIRED_DATASETS = [
    {
        "name": "calm_1_eps1.0",
        "dir": "qwen2_audio_eps1.0_calm_1_saycan",
        "rec_ext": "wav",
    },
    {
        "name": "christmas_jazz_1_eps1.0",
        "dir": "qwen2-audio_eps_1.0_christmas_jazz_1_saycan",
        "rec_ext": "wav",
    },
    {
        "name": "ir_trained_calm_1_eps5.0",
        "dir": "ir_trained_qwen2_audio_calm_1_eps_5.0",
        "rec_ext": "m4a",
    },
    {
        "name": "ir_trained_jazz_1_eps1.0",
        "dir": "ir_trained_qwen2_audio_jazz_1_eps_1.0",
        "rec_ext": "m4a",
    },
    {
        "name": "direct_play_calm_1_eps5.0",
        "dir": "directly_played_from_mac_ir_trained_qwen2_audio_calm_1_eps_5.0",
        "rec_ext": "m4a",
    },
]

SR = 24000


def _load_m4a(path, target_sr=SR):
    """Convert m4a to wav via ffmpeg and load."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(path), '-ac', '1', '-ar', str(target_sr),
             '-acodec', 'pcm_f32le', tmp],
            capture_output=True, check=True,
        )
        import soundfile as sf
        data, _ = sf.read(tmp)
        return data
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _align_signals(original, recorded):
    """Cross-correlation alignment + gain normalization."""
    from scipy.signal import correlate
    # Use first 5s for alignment (faster)
    align_len = min(len(original), SR * 5)
    corr = correlate(recorded[:len(original) * 2], original[:align_len], mode='valid')
    lag = np.argmax(np.abs(corr))
    end = lag + len(original)
    if end > len(recorded):
        aligned = np.zeros_like(original)
        n = len(recorded) - lag
        aligned[:n] = recorded[lag:lag + n]
    else:
        aligned = recorded[lag:end]
    # Gain normalize to match original RMS
    orig_rms = np.sqrt(np.mean(original ** 2) + 1e-10)
    rec_rms = np.sqrt(np.mean(aligned ** 2) + 1e-10)
    aligned = aligned * (orig_rms / rec_rms)
    return aligned, lag


def load_all_pairs(val_ratio=0.15):
    """
    Load all paired (original, recorded) audio files.

    Returns:
        train_pairs: list of (original_np, recorded_np) at 24kHz
        val_pairs: list of (original_np, recorded_np) at 24kHz
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    all_pairs = []

    for ds in PAIRED_DATASETS:
        ds_dir = _SC_DIR / ds["dir"]
        orig_dir = ds_dir / "original"
        rec_dir = ds_dir / "recorded"

        if not orig_dir.exists() or not rec_dir.exists():
            continue

        for orig_path in sorted(orig_dir.glob("*.wav")):
            name = orig_path.stem
            rec_path = rec_dir / f"{name}.{ds['rec_ext']}"
            if not rec_path.exists():
                continue

            # Load original
            orig, orig_sr = sf.read(str(orig_path))
            if orig.ndim == 2:
                orig = orig.mean(axis=1)
            if orig_sr != SR:
                g = gcd(orig_sr, SR)
                orig = resample_poly(orig, SR // g, orig_sr // g)

            # Load recorded
            if ds["rec_ext"] == "m4a":
                rec = _load_m4a(str(rec_path), SR)
            else:
                rec, rec_sr = sf.read(str(rec_path))
                if rec.ndim == 2:
                    rec = rec.mean(axis=1)
                if rec_sr != SR:
                    g = gcd(rec_sr, SR)
                    rec = resample_poly(rec, SR // g, rec_sr // g)

            # Align
            aligned, lag = _align_signals(orig, rec)
            all_pairs.append((orig.astype(np.float32), aligned.astype(np.float32)))

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(all_pairs))
    n_val = max(1, int(len(all_pairs) * val_ratio))
    val_idx = set(indices[:n_val])

    train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in val_idx]
    val_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i in val_idx]

    return train_pairs, val_pairs


class PairedAudioDataset(torch.utils.data.Dataset):
    """
    Dataset of (original, recorded) audio pairs.
    Uses full-length clips (no cropping) to preserve temporal context.
    """

    def __init__(self, pairs, augment=True):
        """
        Args:
            pairs: list of (original_np, recorded_np)
            augment: apply training augmentation (random gain)
        """
        self.pairs = pairs
        self.augment = augment
        # Find max length for padding within collate
        self.max_len = max(len(o) for o, _ in pairs) if pairs else 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig, rec = self.pairs[idx]

        if self.augment:
            gain = np.random.uniform(0.8, 1.2)
            orig = orig * gain
            rec = rec * gain

        return (
            torch.FloatTensor(orig).unsqueeze(0),   # [1, T]
            torch.FloatTensor(rec).unsqueeze(0),     # [1, T]
        )


def _collate_pad(batch):
    """Collate with zero-padding to the longest clip in the batch."""
    origs, recs = zip(*batch)
    max_len = max(o.shape[-1] for o in origs)
    # Pad to multiple of 16 (for strided convolutions)
    max_len = ((max_len + 15) // 16) * 16
    origs_pad = torch.stack([F.pad(o, (0, max_len - o.shape[-1])) for o in origs])
    recs_pad = torch.stack([F.pad(r, (0, max_len - r.shape[-1])) for r in recs])
    return origs_pad, recs_pad


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss combining time-domain and frequency-domain objectives.
    Encourages the model to match both waveform shape and spectral envelope.
    """

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_ratio=0.25):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target)  # time-domain L1

        for n_fft in self.fft_sizes:
            hop = int(n_fft * self.hop_ratio)
            # Magnitude spectrogram
            pred_spec = torch.stft(
                pred.squeeze(1), n_fft=n_fft, hop_length=hop,
                window=torch.hann_window(n_fft, device=pred.device),
                return_complex=True,
            ).abs()
            target_spec = torch.stft(
                target.squeeze(1), n_fft=n_fft, hop_length=hop,
                window=torch.hann_window(n_fft, device=target.device),
                return_complex=True,
            ).abs()

            # Spectral convergence + log-magnitude L1
            loss = loss + torch.norm(target_spec - pred_spec, p='fro') / (torch.norm(target_spec, p='fro') + 1e-8)
            loss = loss + F.l1_loss(
                torch.log(pred_spec + 1e-7),
                torch.log(target_spec + 1e-7),
            )

        return loss / (1 + 2 * len(self.fft_sizes))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs=200,
    lr=1e-4,
    batch_size=4,
    channels=32,
    depth=4,
    device="cuda",
    checkpoint_dir=None,
):
    if checkpoint_dir is None:
        checkpoint_dir = str(_SCRIPT_DIR / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading paired data...")
    train_pairs, val_pairs = load_all_pairs(val_ratio=0.15)
    print(f"  Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")
    for i, (o, r) in enumerate(train_pairs[:5]):
        print(f"    train[{i}]: {len(o)/SR:.1f}s ({len(o)} samples)")

    if len(train_pairs) == 0:
        raise ValueError("No training pairs found!")

    train_ds = PairedAudioDataset(train_pairs, augment=True)
    val_ds = PairedAudioDataset(val_pairs, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, collate_fn=_collate_pad,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=1,
        collate_fn=_collate_pad,
    )

    model = NeuralChannelUNet(channels=channels, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, receptive field ~{model.receptive_field()} samples "
          f"({model.receptive_field()/SR*1000:.0f}ms)")

    criterion = MultiScaleSpectralLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val = float('inf')
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batch = 0
        for orig, rec in train_loader:
            orig, rec = orig.to(device), rec.to(device)
            pred = model(orig)
            loss = criterion(pred, rec)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batch += 1

        train_loss /= max(n_batch, 1)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for orig, rec in val_loader:
                orig, rec = orig.to(device), rec.to(device)
                pred = model(orig)
                loss = criterion(pred, rec)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {"channels": channels, "depth": depth},
            }, os.path.join(checkpoint_dir, "neural_channel_best.pt"))

        if (epoch + 1) % 10 == 0 or improved:
            mark = " *" if improved else ""
            print(f"  Epoch {epoch+1:3d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"best={best_val:.5f}  lr={scheduler.get_last_lr()[0]:.2e}{mark}")

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epochs,
        "val_loss": val_loss,
        "config": {"channels": channels, "depth": depth},
    }, os.path.join(checkpoint_dir, "neural_channel_final.pt"))

    with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Saved to: {checkpoint_dir}/neural_channel_best.pt")
    return model


def load_trained_channel(checkpoint_path, device="cuda", dropout=0.1):
    """Load a trained neural channel model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {"channels": 32, "depth": 4})
    model = NeuralChannelUNet(**config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    channel = NeuralOTAChannel(model, dropout=dropout).to(device)
    print(f"Loaded neural channel from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.5f})")
    return channel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train neural OTA channel model")
    sub = parser.add_subparsers(dest="command")

    # Train
    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--channels", type=int, default=32)
    p_train.add_argument("--depth", type=int, default=4)
    p_train.add_argument("--device", type=str, default="cuda")
    p_train.add_argument("--checkpoint-dir", type=str, default=None)

    # Eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "train":
        train(
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            channels=args.channels, depth=args.depth,
            device=args.device, checkpoint_dir=args.checkpoint_dir,
        )

    elif args.command == "eval":
        channel = load_trained_channel(args.checkpoint, device=args.device)
        _, val_pairs = load_all_pairs()
        print(f"\nEvaluating on {len(val_pairs)} validation pairs...")
        channel.eval()
        for i, (orig, rec) in enumerate(val_pairs):
            orig_t = torch.FloatTensor(orig).unsqueeze(0).unsqueeze(0).to(args.device)
            rec_t = torch.FloatTensor(rec).unsqueeze(0).unsqueeze(0).to(args.device)
            with torch.no_grad():
                pred = channel(orig_t)
            # SNR: how well does prediction match recording
            noise = pred.squeeze().cpu().numpy() - rec
            snr = 10 * np.log10(np.mean(rec ** 2) / (np.mean(noise ** 2) + 1e-10))
            corr = np.corrcoef(pred.squeeze().cpu().numpy(), rec)[0, 1]
            print(f"  Pair {i}: pred→rec SNR={snr:.1f}dB  corr={corr:.4f}")

    else:
        parser.print_help()
