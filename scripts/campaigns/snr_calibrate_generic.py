"""Scenario- and model-generic SNR calibration for the waveform arm.

Generalizes snr_calibrate_s1_waveform.py (S1 + qwen2_audio only) to any
(scenario, target_model) cell in the multi-seed grid. Picks the waveform eps
whose clean-channel SNR matches the paired *latent* bundle, so that the
latent-vs-waveform contrast is a test of attack space rather than of
perturbation budget.

Two things differ from the original, both deliberate:

1. **Scenario config is imported, not re-declared.** The carrier map, carrier
   dir, prompt, loader and duration cap all come from the SCENARIOS dict in
   run_waveform_bundleA_multibitrate.py, so calibration can never drift from
   what the real run does.

2. **The recommended eps is interpolated, not snapped to a grid point.**
   SNR(dB) is close to linear in log10(eps) (doubling eps costs ~6 dB), so a
   coarse grid plus a fit lands much nearer the reference than nearest-of-N.
   This matters: the S3b waveform eps was picked by extrapolating a 20-step
   probe, predicted 9.26 dB, and measured 9.84/9.90 dB against a latent arm at
   8.82 dB -- outside the 1 dB gate. Both values are reported so the snap and
   the fit can be compared.

SNR convention is exactly measure_bundle_snr.py's: signal = the carrier as the
attack actually saw it (same loader), noise = attacked - carrier. The reference
is measured from the latent bundle with that same code path, so calibration and
the downstream admission gate agree by construction.

Usage:
    python 0_all_combined/scripts/snr_calibrate_generic.py \\
        --scenario s1 --target-model audio_flamingo \\
        --ref-bundle 0_all_combined/results_codec_robust/01_finance_voice_agent/audio_flamingo/eps_1.0 \\
        --eps 0.03 0.05 0.075 0.12 --steps 300 --k 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODEC = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CODEC))
sys.path.insert(0, str(HERE))

from audio_agent_bench.probe import select_probe_pairs        # noqa: E402
from robust_waveform_attack import RobustWaveformAttacker     # noqa: E402
from config import ENCODEC_SAMPLE_RATE                        # noqa: E402

# Single source of truth for scenario wiring + the one SNR convention.
from run_waveform_bundleA_multibitrate import (               # noqa: E402
    SCENARIOS, BANK_PROMPT, OPUS_TRAIN_BITRATES,
    _carrier_duration_s, _load_carrier_24k, _load_carrier_music_24k,
)
from measure_bundle_snr import LOADERS as SNR_LOADERS         # noqa: E402

# The two imports above each run `sys.path.insert(0, CODEC)` at module load,
# which puts codec_attack/ ahead of the fork root that waveform_attack.py
# installed. codec_attack/models/ is a stub (base.py only), so it shadows the
# fork's real models/ package and the attacker's lazy
# `from models.qwen2_audio import ...` dies with ModuleNotFoundError. Re-assert
# the fork root last so it wins.
from config import WHISPER_INJECT_ROOT                        # noqa: E402
sys.path.insert(0, WHISPER_INJECT_ROOT)


def _snr_db(adv: np.ndarray, car: np.ndarray) -> float:
    """measure_bundle_snr.py's convention, verbatim."""
    n = min(len(adv), len(car))
    adv, car = adv[:n], car[:n]
    d = adv - car
    return float(10 * np.log10((np.mean(car ** 2) + 1e-12) /
                               (np.mean(d ** 2) + 1e-12)))


def _reference_snr(bundle: Path, loader: str) -> tuple[float, int]:
    """Mean clean SNR of the paired latent bundle, same convention as the gate."""
    load = SNR_LOADERS[loader]
    snrs: list[float] = []
    for w in sorted(bundle.rglob("*_attacked.wav")):
        js = Path(str(w).replace("_attacked.wav", "_summary.json"))
        if not js.is_file():
            continue
        j = json.loads(js.read_text())
        adv, sr = sf.read(str(w))
        if adv.ndim > 1:
            adv = adv.mean(axis=1)
        car = load(j["wav_path"], sr, j["carrier_duration_s"])
        snrs.append(_snr_db(adv, car))
    if not snrs:
        raise SystemExit(f"No scorable clips under {bundle}")
    return float(np.mean(snrs)), len(snrs)


def _fit_eps_for_snr(eps_list: list[float], snr_list: list[float],
                     target: float) -> float | None:
    """Solve SNR = a*log10(eps) + b for the eps hitting `target`.

    Returns None if the fit is degenerate (fewer than 2 usable points, or a
    non-negative slope, which would mean SNR rising with eps -- physically
    wrong and a sign the probe is too short to have saturated the ball).
    """
    pts = [(math.log10(e), s) for e, s in zip(eps_list, snr_list)
           if e > 0 and s is not None and math.isfinite(s)]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    a, b = np.polyfit(xs, ys, 1)
    if a >= 0:
        return None
    return float(10 ** ((target - b) / a))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--target-model", required=True,
                   choices=["qwen2_audio", "qwen25_omni", "audio_flamingo"])
    p.add_argument("--ref-bundle", type=Path, required=True,
                   help="Paired LATENT bundle whose mean clean SNR is matched.")
    p.add_argument("--eps", nargs="+", type=float,
                   default=[0.03, 0.05, 0.075, 0.12])
    p.add_argument("--steps", type=int, default=300,
                   help="Probe steps. Long enough for the L-inf ball to "
                        "saturate; the full run uses 1000.")
    p.add_argument("--k", type=int, default=3, help="Probe carriers per eps.")
    p.add_argument("--loader", default=None, choices=["soundfile", "load_music"],
                   help="Carrier loader; default = load_music, matching the "
                        "latent arm (see --loader on the waveform driver).")
    p.add_argument("--ref-snr-override", type=float, default=None,
                   help="Match this SNR (dB) instead of --ref-bundle's own mean. "
                        "Use the mean over all three latent seeds: seed 1 runs "
                        "lowest, so matching it alone biases eps' high.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = SCENARIOS[args.scenario]
    # Default to load_music: that is what every latent driver uses, so it is
    # what an SNR-matched waveform arm must use too.
    loader = args.loader or "load_music"

    # prompt_inline must win: S2 optimizes under run_s2_attack.py's QA_PROMPT and
    # has no prompt_file, so the old two-way choice would have silently
    # calibrated S2 against the banking prompt.
    if cfg.get("prompt_inline") is not None:
        prompt = cfg["prompt_inline"]
    elif cfg["prompt_file"] is None:
        prompt = BANK_PROMPT
    else:
        prompt = cfg["prompt_file"].read_text(encoding="utf-8").strip()

    ref_bundle = args.ref_bundle if args.ref_bundle.is_absolute() else CODEC / args.ref_bundle
    ref_snr, ref_n = _reference_snr(ref_bundle, loader)
    if args.ref_snr_override is not None:
        # Seed 1 is systematically the LOWEST-SNR latent seed in every cell
        # measured so far, so matching it alone biases eps' high and pushes the
        # other two seeds toward the 1 dB gate -- how Omni S3b came to fail on
        # seeds 7 and 42. Pass the mean over all seeds instead.
        print(f"[calib] reference SNR overridden: {ref_snr:.3f} -> "
              f"{args.ref_snr_override:.3f} dB (multi-seed mean)")
        ref_snr = args.ref_snr_override
    print(f"[calib] cell: {args.scenario} x {args.target_model}  loader={loader}")
    print(f"[calib] reference latent SNR = {ref_snr:.3f} dB  (n={ref_n}) "
          f"from {ref_bundle.name}")
    print(f"[calib] prompt: {prompt[:80]}...")

    pairs = json.loads(cfg["carrier_map"].read_text())["pairs"]
    probes = select_probe_pairs(pairs, k=args.k)
    print(f"[calib] {len(probes)} probes x {len(args.eps)} eps = "
          f"{len(probes)*len(args.eps)} runs @ {args.steps} steps")

    out_dir = ROOT / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = out_dir / f"probe_{args.target_model}_{args.scenario}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    per_eps: dict[str, dict] = {}
    for eps in args.eps:
        alpha = eps / 5.0
        print(f"\n[calib] eps={eps:g} alpha={alpha:g} -- loading attacker...")
        attacker = RobustWaveformAttacker(
            target_model=args.target_model, eps=eps, alpha=alpha,
            perceptual_weight=0.0,
            channel_mode="multi_bitrate",
            multi_bitrate_kbps=OPUS_TRAIN_BITRATES,
            n_eot_samples=1, warmup_ratio=0.2,
            channel_severity=1.0, channel_curriculum=False,
            verbose=False,
        )
        attacker.rand_init = True

        per_carrier: dict[str, float] = {}
        for pp in probes:
            carrier = cfg["carrier_dir"] / pp["carrier"]
            if not carrier.is_file():
                print(f"  MISSING: {carrier}", file=sys.stderr)
                continue
            dur = _carrier_duration_s(carrier)
            # Same cap order as the real driver. Without the all-models cap the
            # probe would measure SNR on up to 25 s of audio while the run it is
            # calibrating uses 15 s, and SNR is duration-dependent.
            if cfg.get("duration_cap_all_models") is not None:
                dur = min(dur, cfg["duration_cap_all_models"])
            if cfg["max_duration"] is not None and args.target_model == "qwen25_omni":
                dur = min(dur, cfg["max_duration"])
            wav = (_load_carrier_music_24k(carrier, dur) if loader == "load_music"
                   else _load_carrier_24k(carrier, dur))

            t0 = time.time()
            res = attacker.attack(wav, target_text=pp["target"],
                                  steps=args.steps, music_name=carrier.stem,
                                  prompt=prompt)
            adv = res.adversarial_wav.detach().squeeze().cpu().numpy()
            out_wav = probe_dir / f"{carrier.stem}_eps{eps:g}_probe.wav"
            sf.write(str(out_wav), adv, ENCODEC_SAMPLE_RATE, subtype="PCM_16")

            car = SNR_LOADERS[loader](str(carrier), ENCODEC_SAMPLE_RATE, dur)
            snr = _snr_db(adv, car)
            per_carrier[pp["carrier"]] = snr
            print(f"  eps={eps:g}  {pp['carrier']}: SNR={snr:.2f} dB "
                  f"({time.time()-t0:.1f}s)")

        del attacker
        torch.cuda.empty_cache()
        vals = list(per_carrier.values())
        per_eps[f"{eps:g}"] = {
            "mean_snr_db": round(float(np.mean(vals)), 3) if vals else None,
            "sd_snr_db": (round(float(np.std(vals, ddof=1)), 3)
                          if len(vals) > 1 else 0.0),
            "per_carrier": {k: round(v, 3) for k, v in per_carrier.items()},
        }

    grid_eps = [float(k) for k in per_eps]
    grid_snr = [per_eps[k]["mean_snr_db"] for k in per_eps]
    usable = [(e, s) for e, s in zip(grid_eps, grid_snr) if s is not None]
    if not usable:
        raise SystemExit("[calib] every eps failed -- nothing to recommend.")

    snapped = min(usable, key=lambda t: abs(t[1] - ref_snr))[0]
    interp = _fit_eps_for_snr([e for e, _ in usable], [s for _, s in usable],
                              ref_snr)

    payload = {
        "scenario": cfg["scen_name"],
        "scenario_key": args.scenario,
        "target_model": args.target_model,
        "loader": loader,
        "channel_mode": "multi_bitrate",
        "opus_train_bitrates": OPUS_TRAIN_BITRATES,
        "probe_steps": args.steps,
        "probe_k": len(probes),
        "alpha_rule": "eps/5",
        "snr_convention": "measure_bundle_snr.py: signal=carrier via loader, "
                          "noise=attacked-carrier",
        "latent_reference": {
            "bundle": str(ref_bundle),
            "mean_snr_db": round(ref_snr, 3),
            "n": ref_n,
        },
        "per_eps": per_eps,
        "recommended_eps_snapped": snapped,
        "recommended_eps_interp": (round(interp, 5) if interp else None),
        "recommended_eps": round(interp, 5) if interp else snapped,
    }
    out_path = args.out or (out_dir / f"snr_{args.target_model}_{args.scenario}.json")
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\n[calib] reference {ref_snr:.2f} dB")
    for k in per_eps:
        m = per_eps[k]["mean_snr_db"]
        print(f"    eps={k:>7}: {m if m is None else f'{m:7.2f}'} dB")
    print(f"[calib] snapped eps = {snapped:g}")
    print(f"[calib] interp  eps = {interp:.5f}" if interp
          else "[calib] interp  eps = (degenerate fit, using snapped)")
    print(f"[calib] -> {payload['recommended_eps']}   wrote {out_path}")


if __name__ == "__main__":
    main()
