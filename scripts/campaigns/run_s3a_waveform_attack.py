"""Phase B — full waveform attack on S3a carriers at the SNR-matched eps.

Reads `recommended_eps` from `calibration/snr_sweep.json` (or accepts
--eps). Runs WaveformAttacker on all 40 S3a pairs and writes the bundle
under `results/03a_ai_detection_bypass/waveform_comparison/qwen2_audio/
eps_{eps}/`. Emits the same `results.jsonl` attack-row schema as
run_s3_attack.py so the eval runner works unchanged.

Usage:
    conda activate codec-attack
    python 0_all_combined/scripts/run_s3a_waveform_attack.py
    python 0_all_combined/scripts/run_s3a_waveform_attack.py --eps 0.05
    python 0_all_combined/scripts/run_s3a_waveform_attack.py --dry-run --eps 0.05
    python 0_all_combined/scripts/run_s3a_waveform_attack.py --eps 0.05 --max-pairs 2 --steps 50
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
ROOT = HERE.parent
CODEC = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CODEC))

from waveform_attack import WaveformAttacker              # noqa: E402
from music_carrier import load_music_by_name              # noqa: E402
from config import ENCODEC_SAMPLE_RATE                    # noqa: E402

SCENARIO = "03a_ai_detection_bypass"
CARRIER_MAP = ROOT / f"data/music/{SCENARIO}/carrier_to_target.json"
CARRIER_DIR = ROOT / "data/music/ai_generated"
SCEN_DIR = ROOT / f"scenarios/{SCENARIO}"
BUNDLE_ROOT = ROOT / f"results/{SCENARIO}/waveform_comparison"
CALIB_ROOT = BUNDLE_ROOT / "calibration"


def _carrier_duration_s(path: Path) -> float:
    try:
        info = sf.info(str(path))
        src_s = info.frames / info.samplerate
    except Exception:
        src_s = 25.0
    return max(10.0, min(25.0, src_s))


def _load_prompt() -> str:
    return (SCEN_DIR / "system_prompt.txt").read_text().strip()


def _resolve_eps(cli_eps: float | None, target_model: str) -> float:
    if cli_eps is not None:
        return cli_eps
    cand = [
        CALIB_ROOT / target_model / "snr_sweep.json",
        CALIB_ROOT / "snr_sweep.json",  # legacy Qwen2-only location
    ]
    path = next((p for p in cand if p.is_file()), None)
    if path is None:
        raise SystemExit(
            f"No --eps given and no calibration file at {cand[0]}. "
            "Run snr_calibrate_s3a_waveform.py first or pass --eps explicitly."
        )
    d = json.loads(path.read_text())
    eps = d.get("recommended_eps")
    if eps is None:
        raise SystemExit(f"calibration JSON missing recommended_eps: {path}")
    return float(eps)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _adv_filename(carrier_stem: str, eps: float, pair_idx: int, category: str) -> str:
    eps_tag = f"eps{eps:g}".replace(".", "p")
    return (
        f"music_{carrier_stem}__target_american_f_q1_edgetts_{eps_tag}"
        f"__{category}_pair{pair_idx:03d}_attacked.wav"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--target-model", default="qwen2_audio")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Smoke-test cap on number of pairs to attack")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    eps = _resolve_eps(args.eps, args.target_model)
    alpha = eps / 5.0

    pairs = json.loads(CARRIER_MAP.read_text())["pairs"]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    bundle = BUNDLE_ROOT / args.target_model / f"eps_{eps}"
    audio_out = bundle / "audio"
    jsonl = bundle / "results.jsonl"

    print(f"[s3a-waveform] eps={eps}  alpha={alpha:g}  steps={args.steps}")
    print(f"[s3a-waveform] {len(pairs)} pairs  model={args.target_model}")
    print(f"[s3a-waveform] bundle: {bundle.relative_to(ROOT)}")

    if args.dry_run:
        for i, pp in enumerate(pairs):
            print(f"  [{i:03d}] {pp['carrier']}  cat={pp['category']}  "
                  f"target={pp['target'][:50]}...")
        return

    bundle.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    prompt = _load_prompt()
    print("[s3a-waveform] loading attacker...")
    attacker = WaveformAttacker(
        target_model=args.target_model, eps=eps, alpha=alpha, verbose=False,
    )

    for idx, pp in enumerate(pairs):
        carrier_path = CARRIER_DIR / pp["carrier"]
        if not carrier_path.is_file():
            print(f"  MISSING: {carrier_path}", file=sys.stderr)
            continue
        carrier_stem = Path(pp["carrier"]).stem
        category = pp.get("category", "x")
        out_name = _adv_filename(carrier_stem, eps, idx, category)
        out_wav = audio_out / out_name
        if out_wav.is_file():
            print(f"  [{idx+1}/{len(pairs)}] SKIP (exists): {out_wav.name}")
            continue

        dur = _carrier_duration_s(carrier_path)
        print(f"\n  [{idx+1}/{len(pairs)}] {pp['carrier']} "
              f"-> {pp['target'][:50]}...")
        t0 = time.time()
        music_wav = load_music_by_name(str(carrier_path), duration=dur)
        result = attacker.attack(
            music_wav, target_text=pp["target"], steps=args.steps,
            music_name=carrier_stem, prompt=prompt,
        )
        elapsed = time.time() - t0
        adv = result.adversarial_wav.detach().squeeze().cpu().numpy()
        sf.write(str(out_wav), adv, ENCODEC_SAMPLE_RATE, subtype="PCM_16")

        _append_jsonl(jsonl, {
            "event": "attack",
            "scenario": SCENARIO,
            "target_model": args.target_model,
            "attack_space": "waveform",
            "eps": float(eps),
            "carrier": pp["carrier"],
            "carrier_clean_path": str(carrier_path),
            "category": category,
            "target_command": pp["target"],
            "clean_text": pp.get("clean_text", ""),
            "adv_audio_path": str(out_wav.relative_to(ROOT)),
            "status": "ok",
            "elapsed_s": round(elapsed, 2),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"    -> saved {out_wav.name}  ({elapsed:.1f}s)")

    del attacker
    torch.cuda.empty_cache()
    print(f"\n[s3a-waveform] done. Bundle at: {bundle}")


if __name__ == "__main__":
    main()
