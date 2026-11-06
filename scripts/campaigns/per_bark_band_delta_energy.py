"""
Gate experiment for the latent-vs-waveform mechanistic story.

Computes per-Bark-band energy distribution of delta = x_adv - x_clean
for the SNR-matched latent (eps=1.0) and waveform (eps=0.075) S1 finance
bundles on Qwen2-Audio.

Hypothesis: latent perturbation energy concentrates in low-mid Bark bands
(4-12, roughly 300-2000 Hz), where Opus/MP3/AAC all spend bits. Waveform
perturbation is broader / whiter.

Decision rule (the "gate"):
    if mean ratio (latent_band_frac / waveform_band_frac) over bands 4-12
       is >= 3.0 -> bit-allocation alignment story is empirically supported.
    if ratio is roughly flat across bands -> framing is wrong, retreat.

CPU only. Pure numpy + scipy + soundfile + matplotlib.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

REPO_ROOT = Path("${REPO_ROOT}")

LATENT_DIR = REPO_ROOT / "0_all_combined/results/01_finance_voice_agent/qwen2_audio/eps_1.0/audio"
WAVEFORM_DIR = REPO_ROOT / "0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075/audio"
OUT_DIR = REPO_ROOT / "0_all_combined/mechanistic_analysis/per_bark_band"

BARK_EDGES_HZ = np.array([
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
    2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500,
], dtype=np.float64)


def load_pair(summary_path: Path):
    """Load adversarial wav and its clean reference. Returns (adv, clean, sr)."""
    with summary_path.open() as fh:
        meta = json.load(fh)
    clean_path = Path(meta["wav_path"])
    adv_filename = summary_path.name.replace("_summary.json", "_attacked.wav")
    adv_path = summary_path.parent / adv_filename
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
    return adv[:n], clean[:n], sr_adv


def per_bark_band_energy(delta: np.ndarray, sr: int, n_fft: int = 1024) -> np.ndarray:
    """Returns per-Bark-band energy (sum of |X|^2 over band freqs)."""
    f, _, Z = stft(delta, fs=sr, nperseg=n_fft, noverlap=n_fft // 2,
                   window="hann", return_onesided=True)
    psd = (np.abs(Z) ** 2).sum(axis=1)  # sum over time -> per-freq energy
    edges = BARK_EDGES_HZ[BARK_EDGES_HZ <= sr / 2 + 1e-6]
    n_bands = len(edges) - 1
    band_energy = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        mask = (f >= edges[i]) & (f < edges[i + 1])
        band_energy[i] = psd[mask].sum()
    return band_energy


def aggregate_dir(audio_dir: Path) -> dict:
    summaries = sorted(audio_dir.glob("*_summary.json"))
    per_carrier_abs = []   # absolute per-band energy
    per_carrier_frac = []  # normalized (sums to 1)
    sr_seen = None
    skipped = []
    for s in summaries:
        pair = load_pair(s)
        if pair is None:
            skipped.append(s.name)
            continue
        adv, clean, sr = pair
        delta = adv - clean
        if np.allclose(delta, 0):
            skipped.append(s.name + " (zero delta)")
            continue
        band_energy = per_bark_band_energy(delta, sr)
        per_carrier_abs.append(band_energy)
        per_carrier_frac.append(band_energy / band_energy.sum())
        sr_seen = sr if sr_seen is None else sr_seen
    if not per_carrier_abs:
        raise RuntimeError(f"No usable carriers in {audio_dir}")
    abs_arr = np.stack(per_carrier_abs)         # (n_carriers, n_bands)
    frac_arr = np.stack(per_carrier_frac)
    return {
        "n_carriers": len(per_carrier_abs),
        "sr": sr_seen,
        "n_bands": abs_arr.shape[1],
        "abs_mean": abs_arr.mean(axis=0).tolist(),
        "abs_std": abs_arr.std(axis=0).tolist(),
        "frac_mean": frac_arr.mean(axis=0).tolist(),
        "frac_std": frac_arr.std(axis=0).tolist(),
        "skipped": skipped,
    }


def gate_decision(latent_frac: np.ndarray, waveform_frac: np.ndarray,
                  n_bands: int) -> dict:
    """Spectral confinement gate.

    Test:
      (a) Latent has >=2x more sub-400Hz energy than waveform (low-freq tilt)
      (b) Waveform has >=2x more >4kHz energy than latent (broadband spread)
      (c) Spearman correlation of per-band shapes is near 0 (disjoint spectra)
    """
    from scipy.stats import spearmanr
    eps = 1e-12
    edges = BARK_EDGES_HZ[:n_bands + 1]
    band_lo = edges[:-1]
    band_hi = edges[1:]
    band_ctr = (band_lo + band_hi) / 2.0

    def mass(arr, lo_hz, hi_hz):
        sel = (band_ctr >= lo_hz) & (band_ctr < hi_hz)
        return float(arr[sel].sum())

    sub400_lat = mass(latent_frac, 0, 400)
    sub400_wav = mass(waveform_frac, 0, 400)
    above4k_lat = mass(latent_frac, 4000, 12000)
    above4k_wav = mass(waveform_frac, 4000, 12000)

    sub400_ratio = sub400_lat / max(sub400_wav, eps)         # want >=2
    above4k_ratio = above4k_wav / max(above4k_lat, eps)       # want >=2
    rho = float(spearmanr(latent_frac, waveform_frac).statistic)

    pass_a = sub400_ratio >= 2.0
    pass_b = above4k_ratio >= 2.0
    pass_c = abs(rho) <= 0.3

    if pass_a and pass_b and pass_c:
        verdict = "PASS"
    elif pass_a or pass_b:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        "test_a_low_freq_tilt": {
            "latent_pct": sub400_lat * 100,
            "waveform_pct": sub400_wav * 100,
            "ratio": sub400_ratio,
            "pass": pass_a,
            "threshold": "L/W >= 2x in sub-400Hz",
        },
        "test_b_broadband_spread": {
            "latent_pct": above4k_lat * 100,
            "waveform_pct": above4k_wav * 100,
            "ratio": above4k_ratio,
            "pass": pass_b,
            "threshold": "W/L >= 2x above 4kHz",
        },
        "test_c_spectral_independence": {
            "spearman_rho": rho,
            "pass": pass_c,
            "threshold": "|rho| <= 0.3",
        },
        "predicted_opus_survival": {
            "opus16k_keeps_below_4kHz": {
                "latent_pct": mass(latent_frac, 0, 4000) * 100,
                "waveform_pct": mass(waveform_frac, 0, 4000) * 100,
            },
            "opus24k_keeps_below_8kHz": {
                "latent_pct": mass(latent_frac, 0, 8000) * 100,
                "waveform_pct": mass(waveform_frac, 0, 8000) * 100,
            },
        },
        "verdict": verdict,
    }


def make_plot(latent: dict, waveform: dict, gate: dict, out_path: Path) -> None:
    n_bands = min(latent["n_bands"], waveform["n_bands"])
    x = np.arange(n_bands)
    band_centers = [(BARK_EDGES_HZ[i] + BARK_EDGES_HZ[i + 1]) / 2 for i in range(n_bands)]
    band_los = BARK_EDGES_HZ[:n_bands]
    width = 0.4

    # Find band indices for the 4kHz cutoff (Opus 16k narrowband boundary)
    cutoff_idx_4k = int(np.searchsorted(band_los, 4000))
    cutoff_idx_400 = int(np.searchsorted(band_los, 400))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sub400 = gate["test_a_low_freq_tilt"]
    above4k = gate["test_b_broadband_spread"]
    rho = gate["test_c_spectral_independence"]["spearman_rho"]
    pred = gate["predicted_opus_survival"]["opus16k_keeps_below_4kHz"]

    ax1.bar(x - width / 2, latent["frac_mean"][:n_bands], width,
            yerr=latent["frac_std"][:n_bands], capsize=2,
            label=f"Latent (eps=1.0)", color="C0", alpha=0.85)
    ax1.bar(x + width / 2, waveform["frac_mean"][:n_bands], width,
            yerr=waveform["frac_std"][:n_bands], capsize=2,
            label=f"Waveform (eps=0.075)", color="C1", alpha=0.85)
    ax1.set_xlabel("Bark band index")
    ax1.set_ylabel("Fractional perturbation energy (sums to 1)")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([f"{i}\n{int(c)} Hz" for i, c in enumerate(band_centers) if i % 2 == 0],
                        fontsize=8)
    ax1.axvspan(-0.5, cutoff_idx_400 - 0.5, color="C0", alpha=0.08,
                label="sub-400 Hz (latent tilt)")
    ax1.axvspan(cutoff_idx_4k - 0.5, n_bands - 0.5, color="C1", alpha=0.08,
                label=">4 kHz (waveform spread)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    stats_left = (
        f"sub-400 Hz:  L {sub400['latent_pct']:.1f}%  vs  W {sub400['waveform_pct']:.1f}%  "
        f"(L/W = {sub400['ratio']:.2f}×)\n"
        f">4 kHz:        L {above4k['latent_pct']:.1f}%  vs  W {above4k['waveform_pct']:.1f}%  "
        f"(W/L = {above4k['ratio']:.2f}×)"
    )
    ax1.text(0.34, 0.97, stats_left, transform=ax1.transAxes, fontsize=9,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="gray", alpha=0.95))

    ratio = np.array(latent["frac_mean"][:n_bands]) / (np.array(waveform["frac_mean"][:n_bands]) + 1e-12)
    colors = ["C0" if r > 1 else "C1" for r in ratio]
    ax2.bar(x, ratio, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="ratio = 1 (parity)")
    ax2.axhline(2.0, color="C0", linestyle=":", alpha=0.6, label=r"$2\times$ latent")
    ax2.axhline(0.5, color="C1", linestyle=":", alpha=0.6, label=r"$2\times$ waveform")
    ax2.set_xlabel("Bark band index")
    ax2.set_ylabel("Latent / Waveform fractional energy (log)")
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([f"{i}\n{int(c)} Hz" for i, c in enumerate(band_centers) if i % 2 == 0],
                        fontsize=8)
    ax2.axvline(cutoff_idx_4k - 0.5, color="red", linestyle="--", alpha=0.5,
                label=r"Opus 16k cutoff ($\sim$4 kHz)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3, which="both")
    ax2.set_yscale("log")

    stats_right = (
        f"Predicted Opus 16k survival\n"
        f"(energy below 4 kHz cutoff):\n"
        f"  latent     {pred['latent_pct']:.1f}%\n"
        f"  waveform   {pred['waveform_pct']:.1f}%"
    )
    ax2.text(0.02, 0.03, stats_right, transform=ax2.transAxes, fontsize=9,
             va="bottom", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="gray", alpha=0.95))

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", default=str(LATENT_DIR))
    parser.add_argument("--waveform-dir", default=str(WAVEFORM_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Latent dir:   {args.latent_dir}")
    print(f"Waveform dir: {args.waveform_dir}")
    print(f"Out dir:      {out_dir}")

    print("\nAggregating latent attack...")
    latent = aggregate_dir(Path(args.latent_dir))
    print(f"  n_carriers={latent['n_carriers']}  sr={latent['sr']}  bands={latent['n_bands']}")
    if latent["skipped"]:
        print(f"  skipped: {latent['skipped']}")

    print("\nAggregating waveform attack...")
    waveform = aggregate_dir(Path(args.waveform_dir))
    print(f"  n_carriers={waveform['n_carriers']}  sr={waveform['sr']}  bands={waveform['n_bands']}")
    if waveform["skipped"]:
        print(f"  skipped: {waveform['skipped']}")

    if latent["sr"] != waveform["sr"]:
        print(f"WARNING: sample rates differ ({latent['sr']} vs {waveform['sr']}). "
              f"Spectral comparison may be confounded.")

    n_bands = min(latent["n_bands"], waveform["n_bands"])
    gate = gate_decision(np.array(latent["frac_mean"][:n_bands]),
                         np.array(waveform["frac_mean"][:n_bands]),
                         n_bands)
    print("\n=== SPECTRAL CONFINEMENT GATE ===")
    a = gate["test_a_low_freq_tilt"]
    b = gate["test_b_broadband_spread"]
    c = gate["test_c_spectral_independence"]
    pred = gate["predicted_opus_survival"]
    mark = lambda p: "PASS" if p else "FAIL"
    print(f"  (a) Sub-400Hz tilt:        latent {a['latent_pct']:.1f}%  vs  waveform {a['waveform_pct']:.1f}%   ratio L/W = {a['ratio']:.2f}x   [{mark(a['pass'])}]")
    print(f"  (b) >4kHz spread:          latent {b['latent_pct']:.1f}%  vs  waveform {b['waveform_pct']:.1f}%   ratio W/L = {b['ratio']:.2f}x   [{mark(b['pass'])}]")
    print(f"  (c) Spectral independence: Spearman rho = {c['spearman_rho']:.3f}   [{mark(c['pass'])}]")
    print()
    print(f"  Predicted Opus 16k (drops >4kHz):  latent retains {pred['opus16k_keeps_below_4kHz']['latent_pct']:.1f}%  "
          f"waveform retains {pred['opus16k_keeps_below_4kHz']['waveform_pct']:.1f}%")
    print(f"  Predicted Opus 24k (drops >8kHz):  latent retains {pred['opus24k_keeps_below_8kHz']['latent_pct']:.1f}%  "
          f"waveform retains {pred['opus24k_keeps_below_8kHz']['waveform_pct']:.1f}%")
    print()
    print(f"  OVERALL VERDICT: {gate['verdict']}")
    if gate["verdict"] == "PASS":
        print("  -> Spectral confinement is the mechanism. Write the section with this framing.")
    elif gate["verdict"] == "MARGINAL":
        print("  -> Partial mechanism; inspect per-band plot before deciding.")
    else:
        print("  -> Not spectral confinement. Fall back to manifold or gradient framing.")

    out = {
        "latent": latent,
        "waveform": waveform,
        "gate": gate,
        "bark_edges_hz": BARK_EDGES_HZ.tolist(),
    }
    json_path = out_dir / "per_bark_band_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved JSON: {json_path}")

    plot_path = out_dir / "per_bark_band.png"
    make_plot(latent, waveform, gate, plot_path)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
