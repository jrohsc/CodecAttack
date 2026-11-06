"""
Train neural OTA channel model on Marshall speaker -> iPhone mic paired recordings.

Adapts neural_channel.py's architecture for Marshall data:
- 57 paired recordings from data/empirical_ota_marshall/nonlinearity_pairs/
- Original: .wav files, Recorded: .m4a files
- Random 2s crop augmentation for small dataset
- Multi-scale spectral loss

Usage:
    python train_marshall_channel.py --epochs 300
    python train_marshall_channel.py --epochs 300 --crop-seconds 3 --batch-size 8
"""

import os
import sys
import glob
import json
import subprocess
import tempfile
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_channel import (
    NeuralChannelUNet, NeuralOTAChannel, MultiScaleSpectralLoss,
    _align_signals, _load_m4a, SR,
)

_SCRIPT_DIR = Path(__file__).parent
_MARSHALL_DIR = _SCRIPT_DIR / "data" / "empirical_ota_marshall" / "nonlinearity_pairs"
_CHECKPOINT_DIR = _SCRIPT_DIR / "results_empirical_analysis_noise_cancellation" / \
    "28_exp_neural_channel_marshall" / "checkpoints"


def load_marshall_pairs(data_dir=_MARSHALL_DIR, val_ratio=0.15):
    """
    Load all paired (original, recorded) audio from Marshall speaker data.

    Returns:
        train_pairs: list of (original_np, recorded_np) at 24kHz
        val_pairs: list of (original_np, recorded_np) at 24kHz
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    orig_dir = data_dir / "original"
    rec_dir = data_dir / "recorded"

    if not orig_dir.exists() or not rec_dir.exists():
        raise FileNotFoundError(f"Data dirs not found: {orig_dir} / {rec_dir}")

    all_pairs = []
    skipped = 0

    for orig_path in sorted(orig_dir.glob("*.wav")):
        name = orig_path.stem
        # Recorded files are .m4a
        rec_path = rec_dir / f"{name}.m4a"
        if not rec_path.exists():
            skipped += 1
            continue

        # Load original
        orig, orig_sr = sf.read(str(orig_path))
        if orig.ndim == 2:
            orig = orig.mean(axis=1)
        if orig_sr != SR:
            g = gcd(orig_sr, SR)
            orig = resample_poly(orig, SR // g, orig_sr // g)

        # Load recorded (m4a -> wav)
        rec = _load_m4a(str(rec_path), SR)
        if rec.ndim == 2:
            rec = rec.mean(axis=1)

        # Align
        aligned, lag = _align_signals(orig, rec)
        all_pairs.append((orig.astype(np.float32), aligned.astype(np.float32)))

    print(f"Loaded {len(all_pairs)} Marshall pairs (skipped {skipped})")

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(all_pairs))
    n_val = max(1, int(len(all_pairs) * val_ratio))
    val_idx = set(indices[:n_val])

    train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in val_idx]
    val_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i in val_idx]

    return train_pairs, val_pairs


class CroppedPairedDataset(torch.utils.data.Dataset):
    """
    Dataset with random cropping for data augmentation.
    Each epoch yields random crops from the paired recordings.
    """

    def __init__(self, pairs, crop_samples, augment=True, samples_per_epoch=None):
        self.pairs = pairs
        self.crop_samples = crop_samples
        self.augment = augment
        # More samples per epoch than raw pairs (each pair yields many crops)
        self.samples_per_epoch = samples_per_epoch or len(pairs) * 10

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Pick a random pair
        pair_idx = np.random.randint(len(self.pairs))
        orig, rec = self.pairs[pair_idx]

        # Random crop
        max_start = max(0, len(orig) - self.crop_samples)
        if max_start > 0:
            start = np.random.randint(max_start)
        else:
            start = 0
        end = start + self.crop_samples

        o = orig[start:end]
        r = rec[start:end]

        # Pad if shorter than crop length
        if len(o) < self.crop_samples:
            o = np.pad(o, (0, self.crop_samples - len(o)))
            r = np.pad(r, (0, self.crop_samples - len(r)))

        if self.augment:
            # Random gain
            gain = np.random.uniform(0.7, 1.3)
            o = o * gain
            r = r * gain

            # Random polarity flip (channel may invert)
            if np.random.random() < 0.5:
                o = -o
                r = -r

        return (
            torch.FloatTensor(o).unsqueeze(0),  # [1, crop_samples]
            torch.FloatTensor(r).unsqueeze(0),
        )


class FullLengthDataset(torch.utils.data.Dataset):
    """Full-length clips for validation (no cropping)."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig, rec = self.pairs[idx]
        return (
            torch.FloatTensor(orig).unsqueeze(0),
            torch.FloatTensor(rec).unsqueeze(0),
        )


def _collate_pad(batch):
    """Collate with zero-padding to longest in batch."""
    origs, recs = zip(*batch)
    max_len = max(o.shape[-1] for o in origs)
    max_len = ((max_len + 15) // 16) * 16
    origs_pad = torch.stack([F.pad(o, (0, max_len - o.shape[-1])) for o in origs])
    recs_pad = torch.stack([F.pad(r, (0, max_len - r.shape[-1])) for r in recs])
    return origs_pad, recs_pad


def evaluate_per_band(model, val_pairs, device="cuda"):
    """Evaluate per-band correlation between prediction and actual recording."""
    from numpy.fft import rfft, rfftfreq

    model.eval()
    bands = [(100, 300), (300, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    band_corrs = {b: [] for b in bands}
    overall_corrs = []

    with torch.no_grad():
        for orig, rec in val_pairs:
            orig_t = torch.FloatTensor(orig).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(orig_t).squeeze().cpu().numpy()

            minlen = min(len(pred), len(rec))
            pred = pred[:minlen]
            rec_trim = rec[:minlen]

            # Overall correlation
            c = np.corrcoef(pred, rec_trim)[0, 1]
            overall_corrs.append(c)

            # Per-band spectral correlation
            freqs = rfftfreq(minlen, 1 / SR)
            spec_pred = np.abs(rfft(pred))
            spec_rec = np.abs(rfft(rec_trim))

            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                if mask.sum() < 10:
                    continue
                bc = np.corrcoef(spec_pred[mask], spec_rec[mask])[0, 1]
                band_corrs[(lo, hi)].append(bc)

    results = {"overall_corr": float(np.mean(overall_corrs))}
    for (lo, hi), vals in band_corrs.items():
        if vals:
            results[f"corr_{lo}_{hi}"] = float(np.mean(vals))

    return results


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading Marshall paired data...")
    data_dir = Path(args.data_dir)
    train_pairs, val_pairs = load_marshall_pairs(data_dir, val_ratio=0.15)
    print(f"  Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")
    for i, (o, r) in enumerate(train_pairs[:3]):
        print(f"    train[{i}]: {len(o) / SR:.1f}s ({len(o)} samples)")

    if len(train_pairs) == 0:
        raise ValueError("No training pairs found!")

    crop_samples = int(args.crop_seconds * SR)
    train_ds = CroppedPairedDataset(
        train_pairs, crop_samples, augment=True,
        samples_per_epoch=len(train_pairs) * 10,
    )
    val_ds = FullLengthDataset(val_pairs)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=_collate_pad,
    )

    device = args.device
    model = NeuralChannelUNet(channels=args.channels, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, receptive field ~{model.receptive_field()} samples "
          f"({model.receptive_field() / SR * 1000:.0f}ms)")

    criterion = MultiScaleSpectralLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val = float('inf')
    history = {"train_loss": [], "val_loss": [], "band_corr": []}

    for epoch in range(args.epochs):
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
                "config": {"channels": args.channels, "depth": args.depth},
            }, os.path.join(args.checkpoint_dir, "neural_channel_marshall_best.pt"))

        # Per-band evaluation every 50 epochs or on improvement
        if (epoch + 1) % 50 == 0 or (improved and epoch > 10):
            band_results = evaluate_per_band(model, val_pairs, device)
            history["band_corr"].append({"epoch": epoch + 1, **band_results})
            band_str = " | ".join(f"{k}={v:.3f}" for k, v in band_results.items())
            mark = " *" if improved else ""
            print(f"  Epoch {epoch + 1:3d}/{args.epochs}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  best={best_val:.5f}{mark}")
            print(f"    Bands: {band_str}")
        elif (epoch + 1) % 10 == 0 or improved:
            mark = " *" if improved else ""
            print(f"  Epoch {epoch + 1:3d}/{args.epochs}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  best={best_val:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{mark}")

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "epoch": args.epochs,
        "val_loss": val_loss,
        "config": {"channels": args.channels, "depth": args.depth},
    }, os.path.join(args.checkpoint_dir, "neural_channel_marshall_final.pt"))

    # Final per-band evaluation
    band_results = evaluate_per_band(model, val_pairs, device)
    history["final_band_corr"] = band_results
    print(f"\nFinal per-band correlation:")
    for k, v in band_results.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Saved to: {args.checkpoint_dir}/neural_channel_marshall_best.pt")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train neural channel on Marshall data")
    parser.add_argument("--data-dir", type=str, default=str(_MARSHALL_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(_CHECKPOINT_DIR))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--crop-seconds", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
