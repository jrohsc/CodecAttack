"""
Per-Bark-band delta energy: speech (S1) vs music (S3a) carriers.

Reuses aggregate_dir() from per_bark_band_delta_energy.py to compute
mean per-band perturbation energy (frac of total) for the eps=1.0
attacked carriers in both scenarios. Overlays the two distributions
to test the Q2 hypothesis: music has more delta energy in mid bands
(300-4000 Hz, the AAC psychoacoustic-active region) and less above
4 kHz than speech.

Outputs:
    0_all_combined/mechanistic_analysis/per_bark_band_speech_vs_music/
      results.json
      plot.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from per_bark_band_delta_energy import aggregate_dir, BARK_EDGES_HZ  # noqa: E402

REPO_ROOT = Path("${REPO_ROOT}")

S1_DIR = REPO_ROOT / "0_all_combined/results/01_finance_voice_agent/qwen2_audio/eps_1.0/audio"
S3A_DIR = REPO_ROOT / "0_all_combined/results/03a_ai_detection_bypass/qwen2_audio/eps_1.0/audio"
OUT_DIR = REPO_ROOT / "0_all_combined/mechanistic_analysis/per_bark_band_speech_vs_music"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Aggregating S1 (speech) carriers from {S1_DIR}", flush=True)
    speech = aggregate_dir(S1_DIR)
    print(f"  n_carriers={speech['n_carriers']}, n_bands={speech['n_bands']}, sr={speech['sr']}", flush=True)
    print(f"Aggregating S3a (music) carriers from {S3A_DIR}", flush=True)
    music = aggregate_dir(S3A_DIR)
    print(f"  n_carriers={music['n_carriers']}, n_bands={music['n_bands']}, sr={music['sr']}", flush=True)

    n_bands = min(speech["n_bands"], music["n_bands"])
    speech_frac = np.array(speech["frac_mean"][:n_bands])
    music_frac = np.array(music["frac_mean"][:n_bands])
    band_lo = BARK_EDGES_HZ[:n_bands]
    band_hi = BARK_EDGES_HZ[1 : n_bands + 1]
    band_ctr = (band_lo + band_hi) / 2.0

    def mass(arr, lo, hi):
        sel = (band_ctr >= lo) & (band_ctr < hi)
        return float(arr[sel].sum())

    bands = {
        "sub_400Hz":     (0,    400),
        "speech_400_4k": (400,  4000),
        "above_4k":      (4000, 12000),
    }
    summary = {
        "n_speech": speech["n_carriers"],
        "n_music":  music["n_carriers"],
        "speech_frac": speech_frac.tolist(),
        "music_frac":  music_frac.tolist(),
        "band_edges":  BARK_EDGES_HZ[: n_bands + 1].tolist(),
        "band_mass": {
            label: {"speech_pct": mass(speech_frac, lo, hi) * 100,
                    "music_pct":  mass(music_frac,  lo, hi) * 100}
            for label, (lo, hi) in bands.items()
        },
    }
    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json}")

    # Plot
    x = np.arange(n_bands)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.4
    ax.bar(x - width / 2, music_frac * 100, width, color="#1f77b4", label=f"Music (S3a, n={music['n_carriers']})")
    ax.bar(x + width / 2, speech_frac * 100, width, color="#d62728", label=f"Speech (S1, n={speech['n_carriers']})")
    ax.axvline(np.searchsorted(band_lo, 4000) - 0.5, ls="--", c="grey", alpha=0.6, label="4 kHz")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in band_ctr], rotation=45, fontsize=7)
    ax.set_xlabel("Bark-band center (Hz)")
    ax.set_ylabel("Fraction of perturbation energy (%)")
    ax.set_title("Per-Bark δ-energy distribution: speech (S1) vs music (S3a)  —  eps=1.0, latent attack")
    ax.legend()
    fig.tight_layout()
    out_png = OUT_DIR / "plot.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")

    print()
    print("=== band-mass comparison ===")
    for label, m in summary["band_mass"].items():
        print(f"  {label:18s}  speech {m['speech_pct']:5.1f}%   music {m['music_pct']:5.1f}%   "
              f"music/speech = {m['music_pct']/max(m['speech_pct'],1e-6):.2f}")


if __name__ == "__main__":
    run()
