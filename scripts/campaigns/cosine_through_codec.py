"""
Tier-A #2 mechanistic experiment: pre/post-codec delta cosine similarity.

For each adversarial wav x_adv and its clean reference x_clean:
  delta_pre  = x_adv - x_clean
  x_adv_op   = Opus(x_adv, bitrate=B)
  x_clean_op = Opus(x_clean, bitrate=B)
  delta_post = x_adv_op - x_clean_op

We compute cos(delta_pre, delta_post) full-band, plus restricted to
sub-400Hz (where latent concentrates) and >4kHz (where waveform concentrates).

Hypothesis: the latent perturbation direction survives Opus compression
(cos > ~0.5 across bitrates), while the waveform perturbation direction
collapses (cos drops sharply as bitrate decreases). The band breakdown
should show that latent's surviving direction is in the sub-400Hz region
and waveform's is in the >4kHz region (consistent with the gate plot).

CPU only. Uses demo_ota.apply_opus_compression (ffmpeg subprocess).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

REPO_ROOT = Path("${REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT))

from demo_ota import apply_opus_compression  # noqa: E402

LATENT_DIR = REPO_ROOT / "0_all_combined/results/01_finance_voice_agent/qwen2_audio/eps_1.0/audio"
WAVEFORM_DIR = REPO_ROOT / "0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075/audio"
OUT_DIR = REPO_ROOT / "0_all_combined/mechanistic_analysis/cosine_through_codec"

BITRATES = [16, 24, 32, 64, 128]


def load_pair(summary_path: Path):
    with summary_path.open() as fh:
        meta = json.load(fh)
    clean_path = Path(meta["wav_path"])
    adv_path = summary_path.parent / summary_path.name.replace("_summary.json", "_attacked.wav")
    if not adv_path.exists():
        return None
    adv, sr_adv = sf.read(str(adv_path), dtype="float32", always_2d=False)
    clean, sr_clean = sf.read(str(clean_path), dtype="float32", always_2d=False)
    if adv.ndim > 1:
        adv = adv.mean(axis=1)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if sr_adv != sr_clean:
        from math import gcd
        g = gcd(sr_adv, sr_clean)
        clean = resample_poly(clean, sr_adv // g, sr_clean // g).astype(np.float32)
    n = min(len(adv), len(clean))
    return adv[:n].astype(np.float32), clean[:n].astype(np.float32), sr_adv


def bandpass(x: np.ndarray, sr: int, lo: float | None, hi: float | None) -> np.ndarray:
    nyq = sr / 2.0
    if lo is None and hi is None:
        return x
    if lo is None:
        sos = butter(4, hi / nyq, btype="low", output="sos")
    elif hi is None:
        sos = butter(4, lo / nyq, btype="high", output="sos")
    else:
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def gain_ratio(pre: np.ndarray, post: np.ndarray) -> float:
    """||delta_post|| / ||delta_pre|| — magnitude survival."""
    n = min(len(pre), len(post))
    npre = np.linalg.norm(pre[:n])
    npost = np.linalg.norm(post[:n])
    if npre < 1e-12:
        return float("nan")
    return float(npost / npre)


def cosine_for_pair(adv: np.ndarray, clean: np.ndarray, sr: int,
                    bitrates: list[int]) -> dict:
    """For one (adv, clean) pair, compute cos(delta_pre, delta_post) at each bitrate.

    Returns dict keyed by bitrate -> {full, sub400, above4k}.
    """
    delta_pre = adv - clean
    out = {}
    for br in bitrates:
        adv_op = apply_opus_compression(adv, sr, br)
        clean_op = apply_opus_compression(clean, sr, br)
        n = min(len(adv_op), len(clean_op), len(delta_pre))
        delta_post = (adv_op[:n] - clean_op[:n]).astype(np.float32)
        delta_pre_n = delta_pre[:n]
        pre_sub = bandpass(delta_pre_n, sr, None, 400)
        post_sub = bandpass(delta_post, sr, None, 400)
        pre_hi = bandpass(delta_pre_n, sr, 4000, None)
        post_hi = bandpass(delta_post, sr, 4000, None)
        out[br] = {
            "cos_full":     cos_sim(delta_pre_n, delta_post),
            "cos_sub400":   cos_sim(pre_sub, post_sub),
            "cos_above4k":  cos_sim(pre_hi, post_hi),
            "gain_full":    gain_ratio(delta_pre_n, delta_post),
            "gain_sub400":  gain_ratio(pre_sub, post_sub),
            "gain_above4k": gain_ratio(pre_hi, post_hi),
        }
    return out


def aggregate_dir(audio_dir: Path, label: str, bitrates: list[int],
                  max_carriers: int = 0) -> dict:
    summaries = sorted(audio_dir.glob("*_summary.json"))
    if max_carriers > 0:
        summaries = summaries[:max_carriers]
    per_carrier = []
    skipped = []
    for i, s in enumerate(summaries):
        pair = load_pair(s)
        if pair is None:
            skipped.append(s.name)
            continue
        adv, clean, sr = pair
        delta_norm = float(np.linalg.norm(adv - clean))
        if delta_norm < 1e-6:
            skipped.append(s.name + " (zero delta)")
            continue
        cos = cosine_for_pair(adv, clean, sr, bitrates)
        per_carrier.append({"name": s.name, "cos": cos, "sr": sr})
        first_br = bitrates[0]
        last_br = bitrates[-1]
        print(f"  [{label}] {i + 1:2d}/{len(summaries)} {s.name[:60]:60s} "
              f"cos@{first_br}k={cos[first_br]['cos_full']:+.3f}  "
              f"gain@{first_br}k={cos[first_br]['gain_full']:.2f}", flush=True)
    if not per_carrier:
        raise RuntimeError(f"No usable carriers in {audio_dir}")
    agg = {}
    metric_keys = ("cos_full", "cos_sub400", "cos_above4k",
                   "gain_full", "gain_sub400", "gain_above4k")
    for br in bitrates:
        agg[br] = {}
        for k in metric_keys:
            vals = np.array([c["cos"][br][k] for c in per_carrier], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            agg[br][k] = {
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std":  float(vals.std())  if len(vals) else float("nan"),
                "n":    int(len(vals)),
            }
    return {"label": label, "n_carriers": len(per_carrier), "per_bitrate": agg, "skipped": skipped}


def make_plot(latent: dict, waveform: dict, bitrates: list[int], out_path: Path) -> None:
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    band_titles = {
        "full":    "Full band",
        "sub400":  "Sub-400 Hz (latent concentration)",
        "above4k": ">4 kHz (waveform concentration)",
    }
    rows = [
        ("cos",  r"$\cos(\delta_{\mathrm{pre}},\, \delta_{\mathrm{post}})$",
         (-0.2, 1.05), 0.5, "cos = 0.5"),
        ("gain", r"$\|\delta_{\mathrm{post}}\| \,/\, \|\delta_{\mathrm{pre}}\|$",
         (0.0, 1.6), 1.0, "gain = 1 (lossless)"),
    ]
    for row_idx, (metric_prefix, ylabel, ylim, ref_y, ref_label) in enumerate(rows):
        for col_idx, band in enumerate(("full", "sub400", "above4k")):
            ax = axes[row_idx, col_idx]
            key = f"{metric_prefix}_{band}"
            for series, color, marker in [(latent, "C0", "o"), (waveform, "C1", "s")]:
                mu = [series["per_bitrate"][b][key]["mean"] for b in bitrates]
                sd = [series["per_bitrate"][b][key]["std"]  for b in bitrates]
                ax.errorbar(bitrates, mu, yerr=sd, marker=marker, capsize=3,
                            label=f"{series['label']} (n={series['n_carriers']})",
                            color=color, lw=2)
            ax.axhline(ref_y, color="gray", linestyle=":", alpha=0.5, label=ref_label)
            if row_idx == 0:
                ax.set_title(band_titles[band], fontsize=12)
            ax.set_xlabel("Opus bitrate (kbps)")
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(mticker.FixedLocator(bitrates))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(b) for b in bitrates]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.3, which="major")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dir", default=str(LATENT_DIR))
    p.add_argument("--waveform-dir", default=str(WAVEFORM_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--bitrates", default=",".join(str(b) for b in BITRATES))
    p.add_argument("--max-carriers", type=int, default=0,
                   help="If >0, cap carriers per attack (smoke test).")
    args = p.parse_args()

    bitrates = [int(b) for b in args.bitrates.split(",") if b.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Latent dir:   {args.latent_dir}")
    print(f"Waveform dir: {args.waveform_dir}")
    print(f"Bitrates:     {bitrates}")

    print("\n--- Latent ---")
    latent = aggregate_dir(Path(args.latent_dir), "Latent (eps=1.0)",
                           bitrates, max_carriers=args.max_carriers)

    print("\n--- Waveform ---")
    waveform = aggregate_dir(Path(args.waveform_dir), "Waveform (eps=0.075)",
                             bitrates, max_carriers=args.max_carriers)

    print("\n=== Direction survival (cosine similarity) ===")
    print(f"  {'br':>5}  {'cos full L':>11}  {'cos full W':>11}    "
          f"{'cos sub L':>11}  {'cos sub W':>11}    "
          f"{'cos hi L':>11}  {'cos hi W':>11}")
    for br in bitrates:
        l = latent["per_bitrate"][br]; w = waveform["per_bitrate"][br]
        print(f"  {br:>5}  "
              f"{l['cos_full']['mean']:+.3f}±{l['cos_full']['std']:.2f}  "
              f"{w['cos_full']['mean']:+.3f}±{w['cos_full']['std']:.2f}    "
              f"{l['cos_sub400']['mean']:+.3f}±{l['cos_sub400']['std']:.2f}  "
              f"{w['cos_sub400']['mean']:+.3f}±{w['cos_sub400']['std']:.2f}    "
              f"{l['cos_above4k']['mean']:+.3f}±{l['cos_above4k']['std']:.2f}  "
              f"{w['cos_above4k']['mean']:+.3f}±{w['cos_above4k']['std']:.2f}")

    print("\n=== Magnitude survival (||delta_post|| / ||delta_pre||) ===")
    print(f"  {'br':>5}  {'gain full L':>11}  {'gain full W':>11}    "
          f"{'gain sub L':>11}  {'gain sub W':>11}    "
          f"{'gain hi L':>11}  {'gain hi W':>11}")
    for br in bitrates:
        l = latent["per_bitrate"][br]; w = waveform["per_bitrate"][br]
        print(f"  {br:>5}  "
              f"{l['gain_full']['mean']:.3f}±{l['gain_full']['std']:.2f}  "
              f"{w['gain_full']['mean']:.3f}±{w['gain_full']['std']:.2f}    "
              f"{l['gain_sub400']['mean']:.3f}±{l['gain_sub400']['std']:.2f}  "
              f"{w['gain_sub400']['mean']:.3f}±{w['gain_sub400']['std']:.2f}    "
              f"{l['gain_above4k']['mean']:.3f}±{l['gain_above4k']['std']:.2f}  "
              f"{w['gain_above4k']['mean']:.3f}±{w['gain_above4k']['std']:.2f}")

    out = {"latent": latent, "waveform": waveform, "bitrates": bitrates}
    json_path = out_dir / "cosine_through_codec_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved JSON: {json_path}")

    plot_path = out_dir / "cosine_through_codec.png"
    make_plot(latent, waveform, bitrates, plot_path)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
