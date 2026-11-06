"""
Decoder frequency response per latent dimension.

Goal: explain *why* latent-space PGD perturbation energy concentrates
sub-4 kHz. The per_bark_band gate experiment showed the phenomenon; this
script tests whether it is structural — built into EnCodec's decoder —
or driven by optimization.

Three measurements at a representative z_clean:

  (A) Per-dim impulse response (decoder Jacobian column)
        For each of the 128 latent dims, compute
            J·e_i  ≈  (decode(z_clean + ε·e_i) - decode(z_clean - ε·e_i)) / (2ε)
        via central finite differences with ε small. FFT each impulse
        response, aggregate into Bark bands. Marginalize across dims.
        Reading: the decoder's intrinsic spectral routing of latent dims.

  (B) Random-direction control
        Sample δz ~ N(0, diag(σ_i²)) where σ_i is measured from the
        actual S1 adversarial δz (encode adv − encode clean). Decode
        z_clean + δz, subtract baseline, FFT.
        Reading: the spectrum produced by a typical-magnitude latent
        perturbation, with no optimization. If this matches (A) and
        the actual adversarial spectrum, the bias is structural.

  (C) Actual adversarial δz (re-encoded)
        Encode the saved adv wavs and the matched clean wavs, decode
        from continuous-latent space (no quantization), take the
        difference, FFT. Reading: ground truth — what the optimizer
        actually produced after compositing through the codec.

If (A), (B), (C) all show the same low-frequency tilt → the
parameterization itself (decoder structure) creates the tilt;
optimization is not the cause. This is the mechanistic explanation
asked of Section 6.1.

Run inside the `codec-attack` conda env. ~10 min on one GPU.
"""

import argparse
import json
import sys
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

REPO_ROOT = Path("${REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT.parent / "external" / "whisper-inject-v2-fork"))
from attacks.latent_codec import EnCodecWrapper  # noqa: E402

ADV_DIR = REPO_ROOT / "0_all_combined/results/01_finance_voice_agent/qwen2_audio/eps_1.0/audio"
OUT_DIR = REPO_ROOT / "0_all_combined/mechanistic_analysis/decoder_freq"

SR = 24000
BARK_EDGES_HZ = np.array([
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
    2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000,
], dtype=np.float64)


def to_24k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x.astype(np.float32)
    g = gcd(sr_in, SR)
    return resample_poly(x, SR // g, sr_in // g).astype(np.float32)


def load_pair_24k(summary_path: Path):
    """Return (clean_24k, adv_24k) cropped to common length, or None."""
    meta = json.loads(summary_path.read_text())
    clean_path = Path(meta["wav_path"])
    adv_path = summary_path.parent / summary_path.name.replace("_summary.json", "_attacked.wav")
    if not (clean_path.exists() and adv_path.exists()):
        return None
    x_c, sr_c = sf.read(str(clean_path), dtype="float32", always_2d=False)
    x_a, sr_a = sf.read(str(adv_path), dtype="float32", always_2d=False)
    if x_c.ndim > 1:
        x_c = x_c.mean(axis=1)
    if x_a.ndim > 1:
        x_a = x_a.mean(axis=1)
    x_c = to_24k(x_c, sr_c)
    x_a = to_24k(x_a, sr_a)
    n = min(len(x_c), len(x_a))
    return x_c[:n], x_a[:n]


def encode_single(codec: EnCodecWrapper, wav_24k: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(wav_24k.astype(np.float32)).to(codec.device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        z = codec.encode_to_continuous(t)
    return z  # [1, 128, T']


def freq_axis(n_samples: int, sr: int) -> np.ndarray:
    return np.fft.rfftfreq(n_samples, 1.0 / sr)


def per_bark_band_from_psd(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    edges = BARK_EDGES_HZ[BARK_EDGES_HZ <= freqs.max() + 1e-6]
    n_bands = len(edges) - 1
    out = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = psd[m].sum()
    return out


# ----------------------------- (A) per-dim impulse response -----------------

def per_dim_jacobian_columns(
    codec: EnCodecWrapper,
    z_clean: torch.Tensor,
    eps: float = 0.05,
    batch: int = 16,
) -> np.ndarray:
    """Central-difference Jacobian columns at z_clean, impulse at center frame.

    Returns array (n_dims, T_audio) where row i is J(z_clean) · e_i.
    """
    n_dims = z_clean.shape[1]
    t_idx = z_clean.shape[-1] // 2
    irs = []
    with torch.no_grad():
        for start in range(0, n_dims, batch):
            end = min(start + batch, n_dims)
            B = end - start
            z_p = z_clean.expand(B, -1, -1).contiguous().clone()
            z_m = z_clean.expand(B, -1, -1).contiguous().clone()
            for k, i in enumerate(range(start, end)):
                z_p[k, i, t_idx] += eps
                z_m[k, i, t_idx] -= eps
            x_p = codec.decode_from_continuous(z_p)  # [B, 1, T_aud]
            x_m = codec.decode_from_continuous(z_m)
            ir = ((x_p - x_m) / (2.0 * eps))[:, 0].cpu().numpy()
            irs.append(ir)
    return np.concatenate(irs, axis=0)  # (n_dims, T_audio)


def localized_psd(ir: np.ndarray, sr: int, t_idx_frames: int,
                  frame_rate_hz: float = 75.0,
                  win_ms: float = 200.0) -> np.ndarray:
    """Take a window centered on the impulse and return PSD."""
    center_samp = int(t_idx_frames * (sr / frame_rate_hz))
    half = int(win_ms * 1e-3 * sr / 2)
    lo = max(0, center_samp - half)
    hi = min(len(ir), center_samp + half)
    seg = ir[lo:hi]
    n = len(seg)
    win = np.hanning(n)
    X = np.fft.rfft(seg * win)
    return (np.abs(X) ** 2)


# ----------------------------- (B) random latent control --------------------

def measure_adv_dz_sigma(codec: EnCodecWrapper, summaries, max_files: int = 25):
    """Per-dim std of (z_adv - z_clean) across multiple carriers."""
    diffs = []
    used = []
    for s in summaries[:max_files]:
        pair = load_pair_24k(s)
        if pair is None:
            continue
        x_c, x_a = pair
        z_c = encode_single(codec, x_c)
        z_a = encode_single(codec, x_a)
        diff = (z_a - z_c)[0].cpu().numpy()  # (128, T')
        diffs.append(diff)
        used.append(s.name)
    cat = np.concatenate(diffs, axis=1)  # (128, total_T)
    sigma = cat.std(axis=1)               # (128,)
    return sigma, cat, used


def random_dz_decoded_psd(
    codec: EnCodecWrapper,
    z_clean: torch.Tensor,
    sigma: np.ndarray,
    n_samples: int = 8,
    sr: int = SR,
):
    """Sample N latent perturbations with per-dim std sigma, decode, compute PSD of (decode(z+dz) - decode(z))."""
    psds = []
    sigma_t = torch.from_numpy(sigma.astype(np.float32)).to(z_clean.device)[None, :, None]
    with torch.no_grad():
        x_base = codec.decode_from_continuous(z_clean)[0, 0].cpu().numpy()
        for _ in range(n_samples):
            dz = torch.randn_like(z_clean) * sigma_t
            x_pert = codec.decode_from_continuous(z_clean + dz)[0, 0].cpu().numpy()
            delta = x_pert - x_base
            n = len(delta)
            X = np.fft.rfft(delta * np.hanning(n))
            psds.append(np.abs(X) ** 2)
    return np.stack(psds), x_base


# ----------------------------- (C) actual adversarial spectrum --------------

def actual_adv_psd_via_codec(
    codec: EnCodecWrapper,
    summaries,
    max_files: int = 25,
):
    """For each pair, decode-from-continuous(z_clean), decode-from-continuous(z_adv), diff, FFT.

    This isolates the *codec-internal* delta — what the decoder produced
    given the optimizer's latent — so it is on the same footing as (A) and (B).
    """
    psds = []
    used = []
    for s in summaries[:max_files]:
        pair = load_pair_24k(s)
        if pair is None:
            continue
        x_c, x_a = pair
        z_c = encode_single(codec, x_c)
        z_a = encode_single(codec, x_a)
        with torch.no_grad():
            x_c_dec = codec.decode_from_continuous(z_c)[0, 0].cpu().numpy()
            x_a_dec = codec.decode_from_continuous(z_a)[0, 0].cpu().numpy()
        n = min(len(x_c_dec), len(x_a_dec))
        delta = x_a_dec[:n] - x_c_dec[:n]
        X = np.fft.rfft(delta * np.hanning(n))
        psds.append(np.abs(X) ** 2)
        used.append(s.name)
    return psds, used


# ----------------------------- aggregation & plotting -----------------------

def normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 0 else v


def mean_band_distribution(psd_list_or_array, freqs: np.ndarray) -> np.ndarray:
    """Per-Bark-band fraction, averaged over items (each item normalized first)."""
    fracs = []
    if isinstance(psd_list_or_array, np.ndarray) and psd_list_or_array.ndim == 2:
        items = list(psd_list_or_array)
    else:
        items = list(psd_list_or_array)
    for psd in items:
        b = per_bark_band_from_psd(psd, freqs)
        if b.sum() > 0:
            fracs.append(b / b.sum())
    return np.stack(fracs).mean(axis=0)


def low_freq_share(frac: np.ndarray, edges: np.ndarray, hi_hz: float) -> float:
    centers = (edges[:-1] + edges[1:]) / 2
    return float(frac[centers < hi_hz].sum())


def make_plots(jacobian_psds, random_psds, actual_psds, freqs, sigma,
               out_path_main: Path, out_path_heatmap: Path, n_dims_used: int):
    edges = BARK_EDGES_HZ[BARK_EDGES_HZ <= freqs.max() + 1e-6]
    centers = (edges[:-1] + edges[1:]) / 2
    n_bands = len(centers)
    x = np.arange(n_bands)

    jac_frac = mean_band_distribution(jacobian_psds, freqs)
    rnd_frac = mean_band_distribution(random_psds, freqs)
    act_frac = mean_band_distribution(actual_psds, freqs)

    # ------- main 3-trace bar plot -------
    fig, ax = plt.subplots(figsize=(13, 5.5))
    width = 0.27
    ax.bar(x - width, jac_frac, width, label=f"(A) Decoder Jacobian (per-dim impulse, n={n_dims_used})",
           color="C2", alpha=0.85)
    ax.bar(x, rnd_frac, width, label=f"(B) Random latent dz (matched per-dim sigma)",
           color="C0", alpha=0.85)
    ax.bar(x + width, act_frac, width, label="(C) Actual adversarial dz (re-encoded)",
           color="C3", alpha=0.85)
    ax.set_xlabel("Bark band index (center freq, Hz)")
    ax.set_ylabel("Fractional energy (sums to 1)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{i}\n{int(centers[i])}" for i in range(n_bands)][::2], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    # callout: sub-4 kHz share for each
    sub4_jac = low_freq_share(jac_frac, edges, 4000) * 100
    sub4_rnd = low_freq_share(rnd_frac, edges, 4000) * 100
    sub4_act = low_freq_share(act_frac, edges, 4000) * 100
    sub400_jac = low_freq_share(jac_frac, edges, 400) * 100
    sub400_rnd = low_freq_share(rnd_frac, edges, 400) * 100
    sub400_act = low_freq_share(act_frac, edges, 400) * 100
    txt = (
        f"sub-4 kHz energy share:\n"
        f"  (A) Jacobian   {sub4_jac:5.1f}%\n"
        f"  (B) Random dz  {sub4_rnd:5.1f}%\n"
        f"  (C) Actual adv {sub4_act:5.1f}%\n"
        f"\nsub-400 Hz share:\n"
        f"  (A)            {sub400_jac:5.1f}%\n"
        f"  (B)            {sub400_rnd:5.1f}%\n"
        f"  (C)            {sub400_act:5.1f}%"
    )
    ax.text(0.62, 0.97, txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.95))
    ax.set_title("Decoder frequency response: structural vs optimization-driven")
    fig.tight_layout()
    fig.savefig(out_path_main, dpi=140)
    plt.close(fig)

    # ------- per-dim heatmap (Jacobian) -------
    # Build (n_dims, n_bands) matrix.
    rows = []
    for psd in jacobian_psds:
        b = per_bark_band_from_psd(psd, freqs)
        if b.sum() > 0:
            rows.append(b / b.sum())
        else:
            rows.append(np.zeros(len(centers)))
    mat = np.stack(rows)  # (n_dims, n_bands)

    # Sort dims by their peak band for readability.
    peak_band = mat.argmax(axis=1)
    order = np.argsort(peak_band)
    mat_sorted = mat[order]

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    im = ax2.imshow(mat_sorted, aspect="auto", cmap="magma", origin="lower",
                    interpolation="nearest")
    ax2.set_xlabel("Bark band index (center freq, Hz)")
    ax2.set_ylabel("Latent dim (sorted by peak band)")
    ax2.set_xticks(np.arange(n_bands)[::2])
    ax2.set_xticklabels([f"{int(centers[i])}" for i in range(n_bands)][::2], fontsize=8, rotation=0)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Fractional energy (per latent dim)")
    ax2.set_title("Per-dim decoder frequency response (sorted by peak band)")
    fig2.tight_layout()
    fig2.savefig(out_path_heatmap, dpi=140)
    plt.close(fig2)

    return {
        "sub4k_share_pct": {"jacobian": sub4_jac, "random": sub4_rnd, "actual": sub4_act},
        "sub400_share_pct": {"jacobian": sub400_jac, "random": sub400_rnd, "actual": sub400_act},
        "n_dims_used": n_dims_used,
        "peak_band_histogram": np.bincount(peak_band, minlength=n_bands).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adv-dir", default=str(ADV_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--bandwidth", type=float, default=6.0,
                   help="EnCodec target bandwidth (kbps). Matches attack pipeline default.")
    p.add_argument("--eps-fd", type=float, default=0.05,
                   help="Finite-difference step in latent space.")
    p.add_argument("--n-jac-carriers", type=int, default=2,
                   help="Number of clean carriers to average Jacobian columns over.")
    p.add_argument("--n-random-samples", type=int, default=8,
                   help="Random latent perturbations per carrier.")
    p.add_argument("--n-stat-carriers", type=int, default=21,
                   help="Carriers used to measure adv dz per-dim sigma.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out dir: {out_dir}")
    print(f"Adv dir: {args.adv_dir}")

    print("\nLoading EnCodec ...")
    codec = EnCodecWrapper(bandwidth=args.bandwidth, device=args.device)
    print(f"  device={codec.device}  bandwidth={args.bandwidth} kbps  sr={codec.sample_rate}")

    summaries = sorted(Path(args.adv_dir).glob("*_summary.json"))
    print(f"  {len(summaries)} summaries available.")

    # --------- (A) per-dim Jacobian columns at one or more clean carriers ---
    print(f"\n[A] Per-dim Jacobian columns (n_carriers={args.n_jac_carriers}, eps={args.eps_fd}) ...")
    jac_psds_all = []  # collect (n_dims,) PSDs across carriers
    n_dims_used = None
    for j, s in enumerate(summaries[:args.n_jac_carriers]):
        pair = load_pair_24k(s)
        if pair is None:
            print(f"  skip (missing files): {s.name}")
            continue
        x_c, _ = pair
        z_clean = encode_single(codec, x_c)
        n_dims_used = z_clean.shape[1]
        t_idx_frames = z_clean.shape[-1] // 2
        print(f"  carrier {j}: {s.name[:60]}  z_shape={tuple(z_clean.shape)}")
        irs = per_dim_jacobian_columns(codec, z_clean, eps=args.eps_fd)
        # compute localized PSD per dim
        for i in range(irs.shape[0]):
            psd = localized_psd(irs[i], SR, t_idx_frames)
            jac_psds_all.append(psd)
    print(f"  collected {len(jac_psds_all)} per-dim PSDs.")

    # Frequency axis for the localized window.
    win_n = len(jac_psds_all[0])
    # Reverse-engineer time-domain window length from PSD length for rfft
    n_time_jac = (win_n - 1) * 2
    freqs_jac = np.fft.rfftfreq(n_time_jac, 1.0 / SR)

    # --------- adv dz statistics ----
    print(f"\nMeasuring adv dz per-dim sigma over {args.n_stat_carriers} carriers ...")
    sigma, _diff_cat, used_stat = measure_adv_dz_sigma(codec, summaries, max_files=args.n_stat_carriers)
    print(f"  used {len(used_stat)} pairs.  sigma mean={sigma.mean():.4f}  median={np.median(sigma):.4f}  max={sigma.max():.4f}")

    # --------- (B) random latent dz, decoded ----
    print(f"\n[B] Random latent dz with measured sigma, n={args.n_random_samples} per carrier ...")
    random_psds = []
    for j, s in enumerate(summaries[:args.n_jac_carriers]):
        pair = load_pair_24k(s)
        if pair is None:
            continue
        x_c, _ = pair
        z_clean = encode_single(codec, x_c)
        psds, _ = random_dz_decoded_psd(codec, z_clean, sigma,
                                        n_samples=args.n_random_samples, sr=SR)
        random_psds.extend(list(psds))
    print(f"  collected {len(random_psds)} random-dz PSDs.")
    n_time_full = (len(random_psds[0]) - 1) * 2
    freqs_full = np.fft.rfftfreq(n_time_full, 1.0 / SR)

    # --------- (C) actual adversarial decoded delta ---
    print(f"\n[C] Actual adversarial dz spectrum (re-encoded) ...")
    actual_psds, used_act = actual_adv_psd_via_codec(codec, summaries, max_files=args.n_stat_carriers)
    print(f"  collected {len(actual_psds)} actual-adv PSDs.")
    n_time_act = (len(actual_psds[0]) - 1) * 2
    freqs_act = np.fft.rfftfreq(n_time_act, 1.0 / SR)

    # The three sets may have different time lengths (windowed Jacobian vs full carriers).
    # That's fine — we aggregate into Bark bands so absolute lengths cancel. Just
    # use the appropriate frequency axis per set when binning.

    # Aggregate band fractions using each set's own freq axis.
    out_path_main = out_dir / "decoder_freq_three_traces.png"
    out_path_heatmap = out_dir / "per_dim_heatmap.png"

    # The make_plots helper expects a single freqs vector per call. Patch by binning here.
    edges = BARK_EDGES_HZ[BARK_EDGES_HZ <= SR / 2 + 1e-6]
    centers = (edges[:-1] + edges[1:]) / 2

    def list_to_band_fracs(psd_list):
        """Each PSD has its own length; derive freq axis from rfft length."""
        fracs = []
        for psd in psd_list:
            n_time = (len(psd) - 1) * 2
            f = np.fft.rfftfreq(n_time, 1.0 / SR)
            b = per_bark_band_from_psd(psd, f)
            if b.sum() > 0:
                fracs.append(b / b.sum())
        return np.stack(fracs)  # (n_items, n_bands)

    jac_fracs = list_to_band_fracs(jac_psds_all)
    rnd_fracs = list_to_band_fracs(random_psds)
    act_fracs = list_to_band_fracs(actual_psds)

    jac_mean = jac_fracs.mean(axis=0); jac_std = jac_fracs.std(axis=0)
    rnd_mean = rnd_fracs.mean(axis=0); rnd_std = rnd_fracs.std(axis=0)
    act_mean = act_fracs.mean(axis=0); act_std = act_fracs.std(axis=0)

    n_bands = len(centers)
    x = np.arange(n_bands)
    width = 0.27
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width, jac_mean, width, yerr=jac_std, capsize=2,
           label=f"(A) Decoder Jacobian (per-dim impulse, n={len(jac_psds_all)})",
           color="C2", alpha=0.85)
    ax.bar(x, rnd_mean, width, yerr=rnd_std, capsize=2,
           label=f"(B) Random latent dz, matched sigma (n={len(random_psds)})",
           color="C0", alpha=0.85)
    ax.bar(x + width, act_mean, width, yerr=act_std, capsize=2,
           label=f"(C) Actual adversarial dz, re-encoded (n={len(actual_psds)})",
           color="C3", alpha=0.85)
    ax.set_xlabel("Bark band index (center freq, Hz)", fontsize=15)
    ax.set_ylabel("Fractional perturbation energy (sums to 1)", fontsize=15)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{i}\n{int(centers[i])}" for i in range(n_bands)][::2], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(top=max(jac_mean.max(), rnd_mean.max(), act_mean.max()) * 1.55)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=13, framealpha=0.95)

    sub4_jac  = float(jac_mean[centers < 4000].sum() * 100)
    sub4_rnd  = float(rnd_mean[centers < 4000].sum() * 100)
    sub4_act  = float(act_mean[centers < 4000].sum() * 100)
    sub400_jac = float(jac_mean[centers < 400].sum() * 100)
    sub400_rnd = float(rnd_mean[centers < 400].sum() * 100)
    sub400_act = float(act_mean[centers < 400].sum() * 100)
    txt = (
        f"sub-4 kHz energy share:\n"
        f"  (A) Jacobian    {sub4_jac:5.1f}%\n"
        f"  (B) Random dz   {sub4_rnd:5.1f}%\n"
        f"  (C) Actual adv  {sub4_act:5.1f}%\n"
        f"\nsub-400 Hz share:\n"
        f"  (A)             {sub400_jac:5.1f}%\n"
        f"  (B)             {sub400_rnd:5.1f}%\n"
        f"  (C)             {sub400_act:5.1f}%"
    )
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=13,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.95))
    ax.set_title("Decoder frequency response: structural (A) vs random control (B) vs actual adv (C)",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path_main, dpi=140)
    plt.close(fig)

    # Per-dim heatmap (Jacobian only).
    n_dims = z_clean.shape[1] if 'z_clean' in dir() else 128
    n_carriers_jac = len(jac_psds_all) // n_dims if n_dims else 1
    if n_carriers_jac >= 1:
        # average across carriers per dim
        jac_arr = np.stack(jac_psds_all).reshape(n_carriers_jac, n_dims, -1).mean(axis=0)
        rows = []
        for psd in jac_arr:
            b = per_bark_band_from_psd(psd, freqs_jac)
            if b.sum() > 0:
                rows.append(b / b.sum())
            else:
                rows.append(np.zeros(n_bands))
        mat = np.stack(rows)
        peak_band = mat.argmax(axis=1)
        order = np.argsort(peak_band)
        mat_sorted = mat[order]

        fig2, ax2 = plt.subplots(figsize=(11, 8))
        im = ax2.imshow(mat_sorted, aspect="auto", cmap="magma", origin="lower",
                        interpolation="nearest")
        ax2.set_xlabel("Bark band index (center freq, Hz)", fontsize=15)
        ax2.set_ylabel("Latent dim (sorted by peak band)", fontsize=15)
        ax2.set_xticks(np.arange(n_bands)[::2])
        ax2.set_xticklabels([f"{int(centers[i])}" for i in range(n_bands)][::2], fontsize=12)
        ax2.tick_params(axis="y", labelsize=12)
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Fractional energy (per latent dim)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        ax2.set_title("Per-dim decoder frequency response (sorted by peak band)", fontsize=15)
        fig2.tight_layout()
        fig2.savefig(out_path_heatmap, dpi=140)
        plt.close(fig2)
        peak_hist = np.bincount(peak_band, minlength=n_bands).tolist()
    else:
        peak_hist = []

    print("\n=== DECODER FREQ-RESPONSE SUMMARY ===")
    print(f"  sub-4 kHz energy share   A={sub4_jac:.1f}%  B={sub4_rnd:.1f}%  C={sub4_act:.1f}%")
    print(f"  sub-400 Hz energy share  A={sub400_jac:.1f}%  B={sub400_rnd:.1f}%  C={sub400_act:.1f}%")
    if abs(sub4_jac - sub4_act) < 10 and abs(sub4_rnd - sub4_act) < 10:
        verdict = "STRUCTURAL"
        print("  -> Jacobian and random control match the actual adv spectrum within 10%.")
        print("  -> Spectral bias is a property of the decoder parameterization, not optimization.")
    elif sub4_jac > 60 and sub4_rnd > 60:
        verdict = "STRUCTURAL_LOW_FREQ_BIASED"
        print("  -> Both the Jacobian and the random control are sub-4 kHz dominant.")
        print("  -> Decoder routes generic latent perturbations to low frequencies.")
    else:
        verdict = "MIXED"
        print("  -> Discrepancy: optimization may contribute beyond pure structure.")

    out = {
        "config": {
            "bandwidth_kbps": args.bandwidth,
            "eps_fd": args.eps_fd,
            "n_jac_carriers": args.n_jac_carriers,
            "n_random_samples": args.n_random_samples,
            "n_stat_carriers": args.n_stat_carriers,
            "n_dims": int(n_dims),
        },
        "sigma_per_dim": sigma.tolist(),
        "sigma_summary": {
            "mean": float(sigma.mean()),
            "median": float(np.median(sigma)),
            "min": float(sigma.min()),
            "max": float(sigma.max()),
        },
        "bark_centers_hz": centers.tolist(),
        "band_fractions": {
            "jacobian_mean": jac_mean.tolist(),
            "jacobian_std": jac_std.tolist(),
            "random_mean": rnd_mean.tolist(),
            "random_std": rnd_std.tolist(),
            "actual_mean": act_mean.tolist(),
            "actual_std": act_std.tolist(),
        },
        "summary_pct": {
            "sub_4khz": {"jacobian": sub4_jac, "random": sub4_rnd, "actual": sub4_act},
            "sub_400hz": {"jacobian": sub400_jac, "random": sub400_rnd, "actual": sub400_act},
        },
        "peak_band_histogram_jac": peak_hist,
        "verdict": verdict,
        "n_items": {
            "jacobian_psds": len(jac_psds_all),
            "random_psds": len(random_psds),
            "actual_psds": len(actual_psds),
        },
    }
    json_path = out_dir / "decoder_freq_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved main plot: {out_path_main}")
    print(f"Saved heatmap:   {out_path_heatmap}")


if __name__ == "__main__":
    main()
