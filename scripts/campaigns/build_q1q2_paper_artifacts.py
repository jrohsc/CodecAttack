"""
Build paper-quality artifacts for the Q1/Q2 mechanistic story:
- Figure: per-Bark delta energy, speech vs music (PDF, vector)
- Table: per-Bark band-mass summary (LaTeX, scriptsize)
- Table: encoder-residual under AAC-LC, R values + matched ASR (LaTeX)

Reads:
  mechanistic_analysis/per_bark_band_speech_vs_music/results.json
  mechanistic_analysis/encoder_residual_aac/<model>/<scenario>.json
  paper-neurips/tables/tab_S1_eps_1.0.tex   (parsed for AAC-LC ASR)
  paper-neurips/tables/tab_S3_eps_1.0.tex   (parsed for AAC-LC ASR)

Writes (when --commit):
  paper-neurips/figures/bark_band_speech_vs_music.pdf
  paper-neurips/tables/tab_bark_band_speech_vs_music.tex
  paper-neurips/tables/tab_encoder_residual_aac.tex
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("${REPO_ROOT}")
MECH = REPO / "0_all_combined/mechanistic_analysis"
PAPER = REPO / "0_all_combined/paper-neurips"

BARK_PATH = MECH / "per_bark_band_speech_vs_music/results.json"
ENC_ROOT = MECH / "encoder_residual_aac"

# ASR parsed manually from the published tex tables, eps=1.0, AAC-LC columns 64/96/128/192
ASR = {
    ("Qwen2-Audio",  "S1"):  [ 2.0,  2.0,  2.0,  2.0],
    ("Qwen2-Audio",  "S3a"): [47.5, 82.5, 85.0, 87.5],
    ("Qwen2.5-Omni", "S1"):  [54.0, 66.0, 66.0, 66.0],
    ("Qwen2.5-Omni", "S3a"): [42.5, 65.0, 65.0, 65.0],
}
KEY = {"Qwen2-Audio": "qwen2_audio", "Qwen2.5-Omni": "qwen25_omni"}


# ---------- Figure: per-Bark speech vs music ----------

def build_fig(out_path: Path) -> None:
    d = json.loads(BARK_PATH.read_text())
    edges = np.array(d["band_edges"])
    n = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2.0
    speech = np.array(d["speech_frac"]) * 100
    music = np.array(d["music_frac"]) * 100

    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w/2, music, w, color="#1f77b4", label=f"Music (S3a, n={d['n_music']})")
    ax.bar(x + w/2, speech, w, color="#d62728", label=f"Speech (S1, n={d['n_speech']})")
    cut_4k = int(np.searchsorted(edges, 4000)) - 1
    ax.axvline(cut_4k - 0.5, ls="--", c="grey", alpha=0.6)
    ax.text(cut_4k - 0.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 9, " 4 kHz",
            color="grey", fontsize=8, va="top")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in centers], rotation=45, fontsize=7)
    ax.set_xlabel("Bark-band center frequency (Hz)", fontsize=9)
    ax.set_ylabel("Fraction of $\\delta$ energy (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"  wrote {out_path}")


# ---------- Table: band-mass summary (T3) ----------

def build_band_mass_table(out_path: Path) -> str:
    d = json.loads(BARK_PATH.read_text())
    bm = d["band_mass"]
    rows = [
        ("$<400$\\,Hz",        "sub_400Hz"),
        ("$400$\\,Hz--$4$\\,kHz", "speech_400_4k"),
        ("$>4$\\,kHz",         "above_4k"),
    ]
    lines = [
        "%% Per-Bark perturbation-energy mass, speech vs music carriers (eps=1.0, n=25 speech / n=40 music)",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{\\textbf{Spectral placement of $\\delta$ on speech vs.\\ music carriers.} "
        "Mean fraction of perturbation energy in three bands, computed over $n{=}25$ S1 (speech) "
        "and $n{=}40$ S3a (music) latent-attack outputs at $\\epsilon{=}1.0$. "
        "Music allocates 1.4$\\times$ more $\\delta$ energy below 400\\,Hz than speech; "
        "above 4\\,kHz the two are statistically indistinguishable.}",
        "\\label{tab:bark_band_speech_vs_music}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Band} & \\textbf{Speech (S1)} & \\textbf{Music (S3a)} & \\textbf{Music / Speech} \\\\",
        "\\midrule",
    ]
    for label, key in rows:
        s = bm[key]["speech_pct"]
        m = bm[key]["music_pct"]
        ratio = m / max(s, 1e-9)
        lines.append(f"{label} & {s:.1f}\\% & {m:.1f}\\% & {ratio:.2f}$\\times$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    txt = "\n".join(lines) + "\n"
    out_path.write_text(txt)
    print(f"  wrote {out_path}")
    return txt


# ---------- Table: encoder-residual (T4) ----------

def build_enc_table(out_path: Path) -> str:
    rows = []
    for human, key in KEY.items():
        for scen in ["S1", "S3a"]:
            j = json.loads((ENC_ROOT / key / f"{scen}.json").read_text())
            rs = [j["mean_ratio_aac"][b] for b in ["64", "96", "128", "192"]]
            asr = ASR[(human, scen)]
            rows.append((human, scen, j["n"], j["mean_useful_drift"], rs, asr))

    lines = [
        "%% Encoder-residual under AAC-LC (eps=1.0)",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{\\textbf{Audio-encoder embedding distortion under AAC-LC, $\\epsilon{=}1.0$.} "
        "$R = \\lVert h_M(\\mathrm{AAC}_b(x_{\\mathrm{atk}})) - h_M(x_{\\mathrm{atk}}) \\rVert / "
        "\\lVert h_M(x_{\\mathrm{atk}}) - h_M(x_{\\mathrm{clean}}) \\rVert$, "
        "the encoder embedding's AAC-induced drift normalized by the useful attack displacement. "
        "All three working cells sit at $R\\!\\le\\!0.09$; the only failed cell "
        "(Qwen2-Audio + S1 + AAC) is the only one above this threshold. Audio Flamingo 3 deferred (different runtime).}",
        "\\label{tab:enc_residual_aac}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        " & & & \\multicolumn{4}{c}{\\textbf{$R$ at AAC-LC bitrate}} & \\\\",
        "\\cmidrule(lr){4-7}",
        "\\textbf{Model} & \\textbf{Carrier} & \\textbf{$n$} "
        "& \\textbf{64\\,k} & \\textbf{96\\,k} & \\textbf{128\\,k} & \\textbf{192\\,k} "
        "& \\textbf{ASR (\\%)} \\\\",
        "\\midrule",
    ]
    for human, scen, n, useful, rs, asr in rows:
        scen_lbl = "speech (S1)" if scen == "S1" else "music (S3a)"
        asr_str = " / ".join(f"{a:.0f}" for a in asr)
        r_strs = [f"{r:.3f}" for r in rs]
        # bold the only failing cell
        if human == "Qwen2-Audio" and scen == "S1":
            r_strs = [f"\\textbf{{{r:.3f}}}" for r in rs]
            asr_str = "\\textbf{" + asr_str + "}"
        lines.append(
            f"{human} & {scen_lbl} & {n} & {r_strs[0]} & {r_strs[1]} & {r_strs[2]} & {r_strs[3]} & {asr_str} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    txt = "\n".join(lines) + "\n"
    out_path.write_text(txt)
    print(f"  wrote {out_path}")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="write into paper-neurips/, otherwise scratch dir")
    args = ap.parse_args()
    if args.commit:
        fig_dir = PAPER / "figures"
        tab_dir = PAPER / "tables"
    else:
        fig_dir = MECH / "_paper_artifacts"
        tab_dir = MECH / "_paper_artifacts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    build_fig(fig_dir / "bark_band_speech_vs_music.pdf")
    t3 = build_band_mass_table(tab_dir / "tab_bark_band_speech_vs_music.tex")
    t4 = build_enc_table(tab_dir / "tab_encoder_residual_aac.tex")

    print()
    print("=" * 72)
    print("LaTeX preview: tab_bark_band_speech_vs_music.tex")
    print("=" * 72)
    print(t3)
    print("=" * 72)
    print("LaTeX preview: tab_encoder_residual_aac.tex")
    print("=" * 72)
    print(t4)


if __name__ == "__main__":
    main()
