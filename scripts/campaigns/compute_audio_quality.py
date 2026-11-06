"""
Compute perceptual audio quality metrics for adversarial wavs under
0_all_combined/results_codec_robust.

For every *_attacked.wav we read the sibling *_summary.json to find the
clean carrier path (`wav_path`), then compute on (clean_ref, attacked):

  * SNR (dB)            — energy-only, sanity check vs training-time SNR
  * SI-SDR (dB)         — gain-invariant SNR
  * PESQ-WB (1..4.5)    — ITU P.862 wideband perceptual quality (16 kHz)
  * STOI / ESTOI (0..1) — speech intelligibility
  * LSD (dB)            — log-spectral distance

Outputs:
  - <out_dir>/per_file.csv          : one row per attacked wav
  - <out_dir>/aggregate.json        : mean/std per (scenario, model, eps_dir)
  - <out_dir>/aggregate_table.md    : pretty-printed Markdown table

Usage:
  conda activate codec-attack
  python compute_audio_quality.py \
      --root /work/.../0_all_combined/results_codec_robust \
      --out  /work/.../0_all_combined/results_codec_robust/_quality \
      --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from glob import glob
from math import gcd
from multiprocessing import Pool, cpu_count

import numpy as np
import soundfile as sf

PESQ_SR = 16000
STOI_SR = 16000


def resample_to(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g)


def load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float64), sr


def snr_db(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    r, d = ref[:n], deg[:n]
    noise = d - r
    p_sig = float(np.mean(r ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    return 10.0 * float(np.log10(p_sig / p_noise))


def si_sdr_db(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    r, d = ref[:n].astype(np.float64), deg[:n].astype(np.float64)
    r = r - r.mean()
    d = d - d.mean()
    alpha = float(np.dot(d, r) / (np.dot(r, r) + 1e-12))
    proj = alpha * r
    err = d - proj
    p = float(np.sum(proj ** 2)) + 1e-12
    e = float(np.sum(err ** 2)) + 1e-12
    return 10.0 * float(np.log10(p / e))


def lsd_db(ref: np.ndarray, deg: np.ndarray, sr: int,
           n_fft: int = 1024, hop: int = 256,
           rel_floor_db: float = -60.0) -> float:
    """Log-spectral distance (dB) averaged over frames.

    Uses a *relative* magnitude floor (rel_floor_db below per-frame peak)
    to avoid silent-bin amplification — without this, near-zero spectral
    bins dominate the metric and LSD blows up to 40+ dB.
    """
    n = min(len(ref), len(deg))
    r, d = ref[:n], deg[:n]

    def stft_mag(x):
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        n_frames = 1 + (len(x) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = x[idx] * np.hanning(n_fft)
        return np.abs(np.fft.rfft(frames, axis=-1))

    R = stft_mag(r)
    D = stft_mag(d)
    # Per-frame floor: rel_floor_db below frame peak (taken across both signals
    # so the floor is consistent for ref/deg in each frame).
    floor_lin_R = R.max(axis=-1, keepdims=True) * (10 ** (rel_floor_db / 20.0))
    floor_lin_D = D.max(axis=-1, keepdims=True) * (10 ** (rel_floor_db / 20.0))
    R = np.maximum(R, floor_lin_R + 1e-12)
    D = np.maximum(D, floor_lin_D + 1e-12)
    log_R = 20.0 * np.log10(R)
    log_D = 20.0 * np.log10(D)
    diff_sq = (log_R - log_D) ** 2
    per_frame = np.sqrt(np.mean(diff_sq, axis=-1))
    return float(np.mean(per_frame))


def compute_metrics(ref_path: str, deg_path: str) -> dict:
    """Compute all quality metrics on (ref, deg). Returns dict with float values
    (NaN on individual metric failure)."""
    out = {
        "snr_db": float("nan"),
        "si_sdr_db": float("nan"),
        "pesq_wb": float("nan"),
        "stoi": float("nan"),
        "estoi": float("nan"),
        "lsd_db": float("nan"),
        "ref_sr": -1,
        "deg_sr": -1,
        "ref_len_s": float("nan"),
        "deg_len_s": float("nan"),
        "error": "",
    }
    try:
        ref, ref_sr = load_mono(ref_path)
        deg, deg_sr = load_mono(deg_path)
    except Exception as e:
        out["error"] = f"load_fail:{e}"
        return out
    out["ref_sr"] = int(ref_sr)
    out["deg_sr"] = int(deg_sr)
    out["ref_len_s"] = round(len(ref) / max(ref_sr, 1), 3)
    out["deg_len_s"] = round(len(deg) / max(deg_sr, 1), 3)

    # Resample both to a common rate for time-domain metrics. Use the deg
    # rate (adversarial) — that's the rate the attack actually wrote at.
    # Then trim to min length.
    common_sr = deg_sr
    try:
        if ref_sr != common_sr:
            ref_c = resample_to(ref, ref_sr, common_sr)
        else:
            ref_c = ref
    except Exception as e:
        out["error"] = f"resample_fail:{e}"
        return out
    n = min(len(ref_c), len(deg))
    ref_c = ref_c[:n]
    deg_c = deg[:n]
    if n < 2:
        out["error"] = "too_short"
        return out

    # Time-domain energy metrics
    try:
        out["snr_db"] = round(snr_db(ref_c, deg_c), 3)
    except Exception as e:
        out["error"] += f"|snr:{e}"
    try:
        out["si_sdr_db"] = round(si_sdr_db(ref_c, deg_c), 3)
    except Exception as e:
        out["error"] += f"|sisdr:{e}"
    try:
        out["lsd_db"] = round(lsd_db(ref_c, deg_c, common_sr), 3)
    except Exception as e:
        out["error"] += f"|lsd:{e}"

    # 16 kHz versions for PESQ / STOI
    try:
        ref_16 = resample_to(ref_c, common_sr, PESQ_SR)
        deg_16 = resample_to(deg_c, common_sr, PESQ_SR)
        m = min(len(ref_16), len(deg_16))
        ref_16 = ref_16[:m]
        deg_16 = deg_16[:m]
    except Exception as e:
        out["error"] += f"|resample16k:{e}"
        return out

    try:
        from pesq import pesq as _pesq
        out["pesq_wb"] = round(float(_pesq(PESQ_SR, ref_16, deg_16, "wb")), 3)
    except Exception as e:
        out["error"] += f"|pesq:{e}"

    try:
        from pystoi import stoi as _stoi
        out["stoi"] = round(float(_stoi(ref_16, deg_16, STOI_SR, extended=False)), 4)
        out["estoi"] = round(float(_stoi(ref_16, deg_16, STOI_SR, extended=True)), 4)
    except Exception as e:
        out["error"] += f"|stoi:{e}"

    return out


def parse_path(adv_wav: str, root: str) -> tuple[str, str, str]:
    """results_codec_robust/<scenario>/<model>/<eps_dir>/audio/<...>.wav
    -> (scenario, model, eps_dir)
    """
    rel = os.path.relpath(adv_wav, root)
    parts = rel.split(os.sep)
    # Expect at least: scenario/model/eps_dir/audio/file.wav  -> 5 parts
    if len(parts) < 5:
        return ("?", "?", "?")
    return (parts[0], parts[1], parts[2])


def find_summary(adv_wav: str) -> str | None:
    """Sibling *_summary.json for an *_attacked.wav."""
    if not adv_wav.endswith("_attacked.wav"):
        return None
    cand = adv_wav[: -len("_attacked.wav")] + "_summary.json"
    return cand if os.path.isfile(cand) else None


def find_ref_path(summary_path: str) -> str | None:
    try:
        with open(summary_path) as f:
            s = json.load(f)
    except Exception:
        return None
    p = s.get("wav_path")
    if p and os.path.isfile(p):
        return p
    return None


def _worker(item: tuple[str, str, str]) -> dict:
    adv_wav, ref_path, root = item
    scen, model, eps_dir = parse_path(adv_wav, root)
    m = compute_metrics(ref_path, adv_wav)
    m.update({
        "scenario": scen,
        "model": model,
        "eps_dir": eps_dir,
        "adv_wav": adv_wav,
        "ref_path": ref_path,
    })
    return m


def aggregate(rows: list[dict]) -> dict:
    """Mean/std/median per (scenario, model, eps_dir) across each metric."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["model"], r["eps_dir"])].append(r)
    out: dict = {}
    metric_keys = ["snr_db", "si_sdr_db", "pesq_wb", "stoi", "estoi", "lsd_db"]
    for (scen, model, eps_dir), grp in sorted(groups.items()):
        cell: dict = {"n": len(grp)}
        for k in metric_keys:
            vals = [r[k] for r in grp
                    if isinstance(r.get(k), (int, float)) and not np.isnan(r[k])]
            if vals:
                cell[k] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "std": round(float(np.std(vals)), 3),
                    "median": round(float(np.median(vals)), 3),
                    "min": round(float(np.min(vals)), 3),
                    "max": round(float(np.max(vals)), 3),
                    "n": len(vals),
                }
            else:
                cell[k] = None
        key = f"{scen}/{model}/{eps_dir}"
        out[key] = cell
    return out


def render_markdown(agg: dict) -> str:
    lines = []
    lines.append("| Scenario | Model | Eps dir | n | SNR (dB) | SI-SDR (dB) | "
                 "PESQ-WB ↑ | STOI ↑ | ESTOI ↑ | LSD (dB) ↓ |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    def fmt(cell, key, decimals=2):
        c = cell.get(key)
        if c is None:
            return "—"
        return f"{c['mean']:.{decimals}f} ± {c['std']:.{decimals}f}"
    for key, cell in agg.items():
        scen, model, eps_dir = key.split("/")
        lines.append("| " + " | ".join([
            scen, model, eps_dir, str(cell["n"]),
            fmt(cell, "snr_db"),
            fmt(cell, "si_sdr_db"),
            fmt(cell, "pesq_wb", 3),
            fmt(cell, "stoi", 3),
            fmt(cell, "estoi", 3),
            fmt(cell, "lsd_db"),
        ]) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root dir, e.g. .../results_codec_robust")
    ap.add_argument("--out", required=True,
                    help="Output dir for csv/json/md")
    ap.add_argument("--workers", type=int, default=min(12, cpu_count()))
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only N wavs (debug)")
    ap.add_argument("--filter", default="",
                    help="Substring filter on adv_wav path (debug)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    adv_wavs = sorted(glob(os.path.join(args.root, "**", "*_attacked.wav"),
                            recursive=True))
    if args.filter:
        adv_wavs = [w for w in adv_wavs if args.filter in w]
    if args.limit:
        adv_wavs = adv_wavs[: args.limit]
    print(f"Found {len(adv_wavs)} *_attacked.wav under {args.root}")

    items = []
    n_no_summary = 0
    n_no_ref = 0
    for w in adv_wavs:
        s = find_summary(w)
        if s is None:
            n_no_summary += 1
            continue
        r = find_ref_path(s)
        if r is None:
            n_no_ref += 1
            continue
        items.append((w, r, args.root))
    print(f"  paired with summary+clean-ref: {len(items)} "
          f"(skipped: {n_no_summary} no-summary, {n_no_ref} no-ref)")

    if not items:
        sys.exit(1)

    # Parallel
    print(f"Running on {args.workers} workers...")
    with Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_worker, items, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0 or (i + 1) == len(items):
                print(f"  {i+1}/{len(items)} done")

    # CSV
    csv_path = os.path.join(args.out, "per_file.csv")
    fieldnames = [
        "scenario", "model", "eps_dir",
        "snr_db", "si_sdr_db", "pesq_wb", "stoi", "estoi", "lsd_db",
        "ref_sr", "deg_sr", "ref_len_s", "deg_len_s",
        "error", "adv_wav", "ref_path",
    ]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {csv_path}")

    # Aggregate
    agg = aggregate(rows)
    json_path = os.path.join(args.out, "aggregate.json")
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Wrote {json_path}")

    md = render_markdown(agg)
    md_path = os.path.join(args.out, "aggregate_table.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
