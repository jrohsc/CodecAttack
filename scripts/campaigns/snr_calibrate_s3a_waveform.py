"""Phase A — SNR calibration for the S3a waveform-vs-latent comparison.

For each eps in a sweep, run WaveformAttacker on 5 probe pairs (one per
unique carrier), save adv wavs, and compute SNR(adv, carrier). Pick the eps
whose mean SNR is closest to the latent reference (7.71 dB) and write the
decision + supporting data to `snr_sweep.json`.

Usage:
    conda activate codec-attack
    python 0_all_combined/scripts/snr_calibrate_s3a_waveform.py
    # or:
    python 0_all_combined/scripts/snr_calibrate_s3a_waveform.py \\
        --eps 0.05 0.1 0.2 0.3 0.5 --steps 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                             # 0_all_combined/
CODEC = ROOT.parent                            # codec_attack/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CODEC))

from audio_agent_bench.probe import select_probe_pairs  # noqa: E402
from audio_agent_bench.snr import snr_db                # noqa: E402
from waveform_attack import WaveformAttacker            # noqa: E402
from music_carrier import load_music_by_name            # noqa: E402
from config import ENCODEC_SAMPLE_RATE                  # noqa: E402

SCENARIO = "03a_ai_detection_bypass"
CARRIER_MAP = ROOT / "data/music/03a_ai_detection_bypass/carrier_to_target.json"
CARRIER_DIR = ROOT / "data/music/ai_generated"
SCEN_DIR = ROOT / f"scenarios/{SCENARIO}"
CALIB_ROOT = ROOT / f"results/{SCENARIO}/waveform_comparison/calibration"

DEFAULT_EPS = [0.05, 0.1, 0.2, 0.3, 0.5]

LATENT_REFERENCE_SNR_BY_MODEL = {
    "qwen2_audio": 7.71,
    "qwen25_omni": 7.55,
    "audio_flamingo": 7.71,  # placeholder until measured
}


def _carrier_duration_s(path: Path) -> float:
    try:
        info = sf.info(str(path))
        src_s = info.frames / info.samplerate
    except Exception:
        src_s = 25.0
    return max(10.0, min(25.0, src_s))


def _load_prompt() -> str:
    return (SCEN_DIR / "system_prompt.txt").read_text().strip()


def _run_one_probe(attacker: WaveformAttacker, carrier_path: Path,
                   target_text: str, prompt: str, steps: int,
                   out_wav: Path) -> float:
    dur = _carrier_duration_s(carrier_path)
    music_wav = load_music_by_name(str(carrier_path), duration=dur)
    result = attacker.attack(
        music_wav, target_text=target_text, steps=steps,
        music_name=carrier_path.stem, prompt=prompt,
    )
    adv = result.adversarial_wav.detach().squeeze().cpu().numpy()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), adv, ENCODEC_SAMPLE_RATE, subtype="PCM_16")
    return snr_db(out_wav, carrier_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--k", type=int, default=5, help="Number of probe pairs")
    p.add_argument("--target-model", default="qwen2_audio")
    args = p.parse_args()

    out_dir = CALIB_ROOT / args.target_model
    probe_audio_dir = out_dir / "probe_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_audio_dir.mkdir(parents=True, exist_ok=True)
    ref_snr = LATENT_REFERENCE_SNR_BY_MODEL.get(args.target_model, 7.71)
    print(f"[calibrate] target_model={args.target_model}  "
          f"latent reference SNR={ref_snr:.2f} dB")

    pairs_all = json.loads(CARRIER_MAP.read_text())["pairs"]
    probes = select_probe_pairs(pairs_all, k=args.k)
    prompt = _load_prompt()
    print(f"[calibrate] {len(probes)} probes x {len(args.eps)} eps = "
          f"{len(probes) * len(args.eps)} runs  steps={args.steps}")
    for pp in probes:
        print(f"  probe: {pp['carrier']}  -> {pp['target'][:60]}...")

    per_eps: dict[str, dict] = {}
    for eps in args.eps:
        alpha = eps / 5.0
        print(f"\n[calibrate] eps={eps} alpha={alpha:g}  loading model...")
        attacker = WaveformAttacker(
            target_model=args.target_model, eps=eps, alpha=alpha,
            verbose=False,
        )
        per_carrier_snr: dict[str, float] = {}
        for pp in probes:
            carrier_path = CARRIER_DIR / pp["carrier"]
            if not carrier_path.is_file():
                print(f"  MISSING: {carrier_path}", file=sys.stderr)
                continue
            out_wav = probe_audio_dir / (
                f"{carrier_path.stem}_eps{eps}_probe_attacked.wav")
            t0 = time.time()
            snr = _run_one_probe(attacker, carrier_path, pp["target"],
                                 prompt, args.steps, out_wav)
            print(f"  eps={eps}  {pp['carrier']}: SNR={snr:.2f} dB "
                  f"({time.time()-t0:.1f}s)")
            per_carrier_snr[pp["carrier"]] = snr
        del attacker
        torch.cuda.empty_cache()

        snrs = list(per_carrier_snr.values())
        per_eps[str(eps)] = {
            "mean_snr_db": round(sum(snrs) / len(snrs), 3) if snrs else None,
            "per_carrier": {k: round(v, 3) for k, v in per_carrier_snr.items()},
        }

    def _score(eps_s: str) -> float:
        m = per_eps[eps_s]["mean_snr_db"]
        return float("inf") if m is None else abs(m - ref_snr)

    recommended = min(per_eps, key=_score)

    payload = {
        "scenario": SCENARIO,
        "target_model": args.target_model,
        "steps": args.steps,
        "alpha_rule": "eps/5",
        "latent_reference_snr_db": ref_snr,
        "per_eps": per_eps,
        "recommended_eps": float(recommended),
    }
    (out_dir / "snr_sweep.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[calibrate] recommended eps: {recommended} "
          f"(mean SNR {per_eps[recommended]['mean_snr_db']:.2f} dB vs target "
          f"{ref_snr:.2f} dB)")
    print(f"[calibrate] wrote {out_dir / 'snr_sweep.json'}")


if __name__ == "__main__":
    main()
