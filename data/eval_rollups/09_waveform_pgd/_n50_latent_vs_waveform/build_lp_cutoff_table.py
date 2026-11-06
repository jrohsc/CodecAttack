"""Table 4 extension: band-limited waveform control at several cutoffs.

Emits the same row schema as the published band-limited control table
(SNR | Clean | Opus 16/32/64/128 | MP3 64 | AAC 64), with one row per cutoff so
"low-frequency placement alone is not sufficient" can be read as a sweep rather
than as a single hand-picked 400 Hz point.

Three things this deliberately does NOT do:

  * It does not read the `success` field. Scoring is delegated to
    score_bundles.score_bundle (paper criterion: target substring OR WER<=0.5).
  * It does not mix SNR conventions. Every arm is measured by
    measure_bundle_snr.py under ONE loader (default `soundfile`, which is what
    the waveform driver actually attacked under and which reproduces the
    published 6.77 dB for the latent arm). The published 6.34/6.64 dB figures
    for the two waveform rows do not reproduce under any loader in the repo --
    see --snr-loader and the note this script prints.
  * It does not silently drop a short bundle. Any seed whose wav count is under
    --expect-n is reported in an INCOMPLETE line, because SLURM COMPLETED has
    repeatedly meant "partial bundle" in this tree.

Usage:
    python build_lp_cutoff_table.py                       # markdown
    python build_lp_cutoff_table.py --tex tab_lp_sweep.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REBUTTAL = Path("${PROJECT_ROOT}_rebuttal")
sys.path.insert(0, str(REBUTTAL))
from score_bundles import score_bundle  # noqa: E402

CODEC = Path("${REPO_ROOT}")
ROOT = CODEC / "0_all_combined"
PY = "${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python"
SNR_TOOL = ROOT / "scripts/measure_bundle_snr.py"
BAND_TOOL = ROOT / "scripts/measure_delta_band_energy.py"

WAVE_BASE = ROOT / "results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni"
LAT_BASE = ROOT / "results_codec_robust/01_finance_voice_agent/qwen25_omni"
SEEDS = [1, 7, 42]

# (label, bundle-dir template). Order = row order in the table.
ARMS: list[tuple[str, str]] = []

COLS = [("Clean", "clean"), ("Opus 16k", "opus16"), ("Opus 32k", "opus32"),
        ("Opus 64k", "opus64"), ("Opus 128k", "opus128"),
        ("MP3 64k", "mp364"), ("AAC 64k", "aac64")]


def run_json(cmd: list[str]) -> dict | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return json.loads(out.stdout.strip())
    except Exception:
        return None


def measure_snr(path: Path, loader: str) -> float | None:
    d = run_json([PY, str(SNR_TOOL), "--bundle", str(path), "--loader", loader])
    return None if not d or "snr_mean" not in d else d["snr_mean"]


def measure_band(path: Path, cutoffs, loader: str) -> dict | None:
    # --force-wav so a projected arm and an unprojected one are computed the
    # same way; the npy path would give the projected arm an unfair rolloff.
    return run_json([PY, str(BAND_TOOL), "--bundle", str(path),
                     "--loader", loader, "--force-wav",
                     "--cutoffs", *[str(c) for c in cutoffs]])


def pct(kn) -> float | None:
    k, n = kn
    return None if not n else 100.0 * k / n


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt(m, s, nd=0):
    if m is None:
        return "--"
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}" if nd else f"{m:.0f} $\\pm$ {s:.0f}"


def collect(label: str, tmpl: str, args) -> dict:
    row = {"label": label, "seeds": {}, "incomplete": []}
    per_col = {c: [] for _, c in COLS}
    snrs, bands = [], []
    for seed in SEEDS:
        path = Path(tmpl.format(seed=seed))
        if not path.is_dir():
            row["incomplete"].append(f"seed{seed}: MISSING {path}")
            continue
        n_wav = len(glob.glob(str(path / "**" / "*_attacked.wav"),
                              recursive=True))
        neural = glob.glob(str(path / "neural_delivery_eval_*.json"))
        sc = score_bundle(str(path),
                          os.path.basename(neural[0]) if neural else None)
        row["seeds"][seed] = {"n_wav": n_wav,
                              "scores": {c: sc.get(c, (0, 0)) for _, c in COLS}}
        if n_wav < args.expect_n:
            row["incomplete"].append(f"seed{seed}: {n_wav}/{args.expect_n} wavs")
        for _, c in COLS:
            per_col[c].append(pct(sc.get(c, (0, 0))))
        s = measure_snr(path, args.snr_loader)
        if s is not None:
            snrs.append(s)
        if args.band:
            b = measure_band(path, args.cutoffs, args.snr_loader)
            if b:
                bands.append(b)
    row["cells"] = {c: mean_sd(per_col[c]) for _, c in COLS}
    row["snr"] = mean_sd(snrs)
    if bands:
        row["band"] = {f"le_{int(c)}": mean_sd(
            [b.get(f"pct_energy_le_{int(c)}hz") for b in bands])
            for c in args.cutoffs}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cutoffs", nargs="+", type=float, default=[400, 800, 1500],
                   help="Band-limited arms to include, in Hz.")
    p.add_argument("--eps", nargs="+", required=True,
                   help="eps' per cutoff, same order (e.g. 0.3 0.4 0.38).")
    p.add_argument("--modes", nargs="+", default=None,
                   choices=["lp", "fft"],
                   help="Projection per cutoff, same order. lp -> bundle dir "
                        "eps_<E>_lp<HZ>_seed<S> (causal sosfilt, --lowpass-hz); "
                        "fft -> eps_<E>_fftlp<HZ>_seed<S> (rFFT subspace, "
                        "--spectral-cutoff-hz). Default: all lp, which "
                        "reproduces the published 400 Hz row. Use fft for the "
                        "800/1500 Hz rows: an lp arm above 400 Hz is not one "
                        "-- lp800 keeps 99.3%% of delta below 400 Hz, so an "
                        "all-lp sweep tabulates the 400 Hz arm three times.")
    p.add_argument("--wave-eps", default="0.075",
                   help="eps for the unconstrained waveform baseline row.")
    p.add_argument("--expect-n", type=int, default=50)
    p.add_argument("--snr-loader", default="soundfile",
                   choices=["soundfile", "load_music"])
    p.add_argument("--band", action="store_true",
                   help="Also compute per-arm perturbation energy fractions.")
    p.add_argument("--tex", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    if len(args.eps) != len(args.cutoffs):
        p.error("--eps must have one value per --cutoff")
    modes = args.modes if args.modes else ["lp"] * len(args.cutoffs)
    if len(modes) != len(args.cutoffs):
        p.error("--modes must have one value per --cutoff")

    ARMS.append((f"Waveform, $\\epsilon'{{=}}{args.wave_eps}$",
                 str(WAVE_BASE / f"eps_{args.wave_eps}_seed{{seed}}")))
    for hz, e, mode in zip(args.cutoffs, args.eps, modes):
        tag = "lp" if mode == "lp" else "fftlp"
        ARMS.append((f"Waveform $\\le${int(hz)}\\,Hz, $\\epsilon'{{=}}{e}$",
                     str(WAVE_BASE / f"eps_{e}_{tag}{int(hz)}_seed{{seed}}")))
    ARMS.append(("\\textbf{Latent (ours)}, $\\epsilon{=}1.0$",
                 str(LAT_BASE / "eps_1.0_seed{seed}")))

    rows = [collect(lbl, tmpl, args) for lbl, tmpl in ARMS]

    print(f"# Band-limited cutoff sweep (Qwen2.5-Omni, S1 finance, "
          f"seeds {SEEDS})\n")
    print(f"SNR convention: measure_bundle_snr.py --loader {args.snr_loader}. "
          f"All arms measured identically.\n")
    print("Projection per cutoff: "
          + ", ".join(f"{int(hz)} Hz={m}" for hz, m in zip(args.cutoffs, modes))
          + ".  lp=causal sosfilt re-applied each step (NOT a projection above "
            "400 Hz); fft=exact rFFT subspace.\n")
    hdr = "| Method | SNR (dB) | " + " | ".join(n for n, _ in COLS) + " |"
    print(hdr)
    print("|" + "---|" * (len(COLS) + 2))
    for r in rows:
        m, s = r["snr"]
        cells = [fmt(*r["cells"][c]) for _, c in COLS]
        snr = "--" if m is None else f"{m:.2f} $\\pm$ {s:.2f}"
        print(f"| {r['label']} | {snr} | " + " | ".join(cells) + " |")

    if args.band:
        print("\n## Perturbation energy below cutoff (%, wav-reconstructed)\n")
        cols = [f"$\\le${int(c)} Hz" for c in args.cutoffs]
        print("| Method | " + " | ".join(cols) + " |")
        print("|" + "---|" * (len(cols) + 1))
        for r in rows:
            if "band" not in r:
                continue
            vals = [fmt(*r["band"][f"le_{int(c)}"], nd=1) for c in args.cutoffs]
            print(f"| {r['label']} | " + " | ".join(vals) + " |")

    print("\n## Completeness\n")
    any_bad = False
    for r in rows:
        for msg in r["incomplete"]:
            print(f"- INCOMPLETE  {r['label']}  {msg}")
            any_bad = True
    if not any_bad:
        print(f"- all arms {args.expect_n}/{args.expect_n} on every seed")

    print("\n## Per-seed values\n")
    print("| Method | seed | n | " + " | ".join(n for n, _ in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 3))
    for r in rows:
        for seed, d in r["seeds"].items():
            cs = []
            for _, c in COLS:
                k, n = d["scores"][c]
                cs.append(f"{100.0*k/n:.0f} ({k}/{n})" if n else "--")
            print(f"| {r['label']} | {seed} | {d['n_wav']} | "
                  + " | ".join(cs) + " |")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {args.json}")

    if args.tex:
        lines = [
            r"\begin{tabular}{@{}l c rrrrr rr@{}}", r"\toprule",
            r"\textbf{Method} & \textbf{SNR (dB)} & "
            + " & ".join(rf"\textbf{{{n}}}" for n, _ in COLS) + r" \\",
            r"\midrule",
        ]
        for r in rows[:-1]:
            m, s = r["snr"]
            snr = "--" if m is None else f"{m:.2f} $\\pm$ {s:.2f}"
            lines.append(f"{r['label']} & {snr} & "
                         + " & ".join(fmt(*r["cells"][c]) for _, c in COLS)
                         + r" \\")
        lines.append(r"\midrule")
        last = rows[-1]
        m, s = last["snr"]
        snr = "--" if m is None else f"{m:.2f} $\\pm$ {s:.2f}"
        lines.append(f"{last['label']} & {snr} & "
                     + " & ".join(r"\cellcolor{red!10}" + fmt(*last["cells"][c])
                                  for _, c in COLS) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.tex.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
