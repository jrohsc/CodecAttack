"""
Build the encoder-transfer results table.

Reports, per arm and per evaluator, both scoring columns:

  strict      the published criterion, unchanged -- directly comparable to the
              0/1566 cross-architecture cells
  normalized  after number verbalization / preamble stripping (rescore_normalized)

The white-box diagonal (qwen2_audio, the attacked model) is printed beside the
transfer cells and never merged into them. Positive-control ceilings are printed
in the same units so a transfer cell can be read against what the gate can
actually register on perfect audio.

Usage:
    python build_encoder_transfer_table.py [--root <results_encoder_transfer>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "0_all_combined/results_encoder_transfer"

ARMS = [
    ("ce_zeros", "CE + zeros (published baseline)"),
    ("ce_targetinit", "CE + ZQ target-init"),
    ("enc_zeros", "encoder-cosine + zeros"),
    ("enc_targetinit", "encoder-cosine + ZQ target-init"),
    ("encce_targetinit", "encoder+CE hybrid + target-init"),
    ("enc_mask_match", "encoder-cosine + mask + match"),
    ("encce_mask_match", "encoder+CE + mask + match"),
]
EVALS = ["qwen2_audio", "qwen25_omni", "qwen25_omni_3b", "audio_flamingo"]
CHANNELS = ["clean", "opus64", "opus32", "mp364", "aac64"]


def cell(bundle: Path, model: str, ch: str):
    """Return (strict, normalized, n, min_wer) or None if not evaluated yet.

    min_wer is carried because a bare 0/50 cannot distinguish "every clip just
    missed the 0.5 gate" from "nothing moved at all". The success criterion is
    already lowercased WER <= 0.5 (cross_model_transfer_eval.py: matches_target
    = exact OR wer <= 0.5), so the closest clip's WER is the margin to success.
    """
    tj = bundle / f"transfer_eval_{model}.json"
    rj = bundle / f"rescored_{model}.json"
    if not tj.is_file():
        return None
    rows = json.load(open(tj)).get("results", [])
    n = len(rows)
    strict = sum(1 for r in rows if (r.get(ch) or {}).get("matches_target"))
    wers = [w for w in ((r.get(ch) or {}).get("wer") for r in rows) if w is not None]
    min_wer = min(wers) if wers else None
    norm = None
    if rj.is_file():
        norm = json.load(open(rj)).get("channels", {}).get(ch, {}).get("normalized")
    return strict, norm, n, min_wer


def fmt(c):
    if c is None:
        return "   --   "
    s, nm, n, mw = c
    out = f"{s}/{n}"
    if nm is not None:
        out += f" ({nm})"
    if mw is not None:
        out += f" w{mw:.2f}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--channel", default="clean")
    args = ap.parse_args()
    ch = args.channel

    calib_p = HERE / "0_all_combined/data/speech/01_finance/target_audio/calibration.json"
    calib = json.load(open(calib_p)) if calib_p.is_file() else None

    print(f"\nENCODER-TRANSFER RESULTS  channel={ch}   cells = strict/n (normalized)")
    print("white-box diagonal is qwen2_audio; the other three are transfer.\n")

    hdr = f"{'arm':34s}" + "".join(f"{e:>24s}" for e in EVALS) + f"{'cos':>9s}{'wb%':>7s}"
    print(hdr)
    print("-" * len(hdr))

    for arm, label in ARMS:
        bundle = args.root / arm
        if not bundle.is_dir():
            print(f"{label:34s}" + "".join(f"{'(not run)':>24s}" for _ in EVALS))
            continue
        man_p = bundle / "manifest.json"
        man = json.load(open(man_p)) if man_p.is_file() else {}
        cos = man.get("mean_cos_final_to_target")
        wb = man.get("whitebox_exact_pct")
        row = f"{label:34s}" + "".join(f"{fmt(cell(bundle, e, ch)):>24s}" for e in EVALS)
        row += f"{(f'{cos:.3f}' if cos is not None else '--'):>9s}"
        row += f"{(f'{wb:.0f}' if wb is not None else '--'):>7s}"
        print(row)

    print("\n--- positive control ceiling (unperturbed spoken target) ---")
    for name in ("POSCTRL_transcribe", "POSCTRL_target_speech"):
        b = args.root / name
        if not b.is_dir():
            continue
        print(f"{name:34s}" + "".join(f"{fmt(cell(b, e, ch)):>24s}" for e in EVALS))

    if calib:
        g = calib["cross_different_targets"]["max"]
        print(f"\ncosine readability gate: {g:.4f} "
              f"(max over unrelated spoken commands, n="
              f"{calib['cross_different_targets']['n']}). "
              f"same-content+20dB-noise = {calib['target_plus_noise_snr20']['mean']:.4f}, "
              f"which sits BELOW that gate -- within-content variance exceeds "
              f"between-content separation, so cosine alone is weak evidence.")


if __name__ == "__main__":
    main()
