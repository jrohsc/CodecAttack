#!/usr/bin/env python3
"""Run one immutable waveform-control cell and emit a manifest JSONL.

The tuned arms use the latent attack's 30% clean warmup and then alternate
direct/Opus-BPDA updates.  The historical arm intentionally retains its
scenario-specific behavior for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for module_dir in (ROOT / 'scripts/campaigns', ROOT / 'scripts', ROOT / 'codec_wrappers'):
    sys.path.insert(0, str(module_dir))

from config import ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE  # noqa: E402
from music_carrier import load_music_by_name  # noqa: E402
from robust_waveform_attack import RobustWaveformAttacker  # noqa: E402
from waveform_attack import WaveformAttacker  # noqa: E402
from reviewer_controls.common import (  # noqa: E402
    TRAIN_OPUS_KBPS, above_cutoff_energy_fraction, append_jsonl, write_pcm16_pair,
)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    for scenario in cfg["scenarios"].values():
        if "prompt_file" in scenario:
            scenario["prompt"] = (ROOT / scenario["prompt_file"]).read_text().strip()
    return cfg


def partition_pairs(cfg: dict, scenario_key: str, split: str, pilot: bool) -> list[tuple[int, dict]]:
    sc = cfg["scenarios"][scenario_key]
    pairs = json.loads((ROOT / sc["carrier_map"]).read_text())["pairs"]
    if pilot:
        indices = sc[f"pilot_{split}_pair_indices"]
        return [(i, pairs[i]) for i in indices]
    dev = set(sc["development_carriers"])
    want_dev = split == "development"
    return [(i, p) for i, p in enumerate(pairs) if (p["carrier"] in dev) == want_dev]


def duration_s(path: Path) -> float:
    return max(10.0, min(25.0, sf.info(path).duration))


def stable_id(scenario: str, pair_index: int, carrier: str, target: str) -> str:
    digest = hashlib.sha256(f"{scenario}\0{pair_index}\0{carrier}\0{target}".encode()).hexdigest()[:12]
    return f"{scenario}_p{pair_index:03d}_{digest}"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--scenario", choices=["s1", "s3a"], required=True)
    p.add_argument("--split", choices=["development", "evaluation"], required=True)
    p.add_argument("--method", choices=["original_waveform", "tuned_waveform", "bandlimited_4khz"], required=True)
    p.add_argument("--epsilon", type=float, required=True,
                   help="Budget used when --epsilon-map is absent, and the "
                        "fallback for examples the map does not cover.")
    p.add_argument("--epsilon-map", type=Path, default=None,
                   help="calibrate_epsilon.py output. Its per-example budgets "
                        "override --epsilon so each example can be matched to "
                        "its own latent SNR rather than only the group mean.")
    p.add_argument("--alpha-ratio", type=float, required=True)
    p.add_argument("--perceptual-weight", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rand-init", action="store_true",
                   help="Uniform L-inf-ball init instead of zeros. Off by "
                        "default so the arms match the headline latent bundles.")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Run only the first N examples of the selected split. "
                        "The deterministic prefix is identical across probe and "
                        "calibrated runs, making bounded control runs resumable.")
    p.add_argument("--shard-id", type=int, default=None,
                   help="Zero-based deterministic shard of the selected split. "
                        "Must be supplied with --num-shards.")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Number of deterministic shards. Sharding is applied "
                        "after the development/evaluation split and pilot filter.")
    p.add_argument("--example-ids-file", type=Path, default=None,
                   help="Optional newline-delimited stable IDs to run. Used for "
                        "distortion-only correction of out-of-tolerance cases.")
    p.add_argument("--output-root", type=Path, default=ROOT / "reviewer_control_outputs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-tag", default="",
                   help="Suffix appended to the run directory so a calibrated "
                        "rerun never collides with the probe it was solved from.")
    args = p.parse_args()

    if not np.isfinite(args.perceptual_weight) or args.perceptual_weight < 0:
        raise ValueError("Invalid perceptual weight")
    cfg = load_config(args.config)
    steps = int(args.steps if args.steps is not None else cfg["steps"])
    eps_map = {}
    if args.epsilon_map is not None:
        payload = json.loads(args.epsilon_map.read_text())
        if payload.get("status") != "frozen":
            raise SystemExit(f"epsilon map is not frozen: {args.epsilon_map}")
        eps_map = {k: float(v) for k, v in payload["epsilon_map"].items()}
    alpha = float(args.epsilon * args.alpha_ratio)
    sc = cfg["scenarios"][args.scenario]
    pairs = partition_pairs(cfg, args.scenario, args.split, args.pilot)
    if args.max_examples is not None:
        if args.max_examples < 1:
            raise SystemExit("--max-examples must be >= 1")
        pairs = pairs[:args.max_examples]
    if (args.shard_id is None) != (args.num_shards is None):
        raise SystemExit("--shard-id and --num-shards must be supplied together")
    if args.num_shards is not None:
        if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
            raise SystemExit("require --num-shards >= 1 and 0 <= --shard-id < --num-shards")
        pairs = [pair for position, pair in enumerate(pairs)
                 if position % args.num_shards == args.shard_id]
    if args.example_ids_file is not None:
        requested = {line.strip() for line in args.example_ids_file.read_text().splitlines()
                     if line.strip()}
        available = {stable_id(args.scenario, i, pair["carrier"], pair["target"])
                     for i, pair in pairs}
        missing = requested - available
        if missing:
            raise SystemExit(f"requested example IDs outside selected split/shard: {sorted(missing)}")
        pairs = [(i, pair) for i, pair in pairs
                 if stable_id(args.scenario, i, pair["carrier"], pair["target"]) in requested]
    run_name = (
        f"{args.scenario}/{args.split}/{args.method}/eps{args.epsilon:g}"
        f"_ar{args.alpha_ratio:g}_steps{steps}_seed{args.seed}"
        + ("_pilot" if args.pilot else "")
        + (f"_n{args.max_examples}" if args.max_examples is not None else "")
        + (f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards is not None else "")
        + (f"_pw{args.perceptual_weight:g}" if args.perceptual_weight else "")
        + (f"_{args.run_tag}" if args.run_tag else "")
    )
    out = args.output_root / run_name
    audio_dir = out / "audio"
    manifest = out / "manifest.jsonl"
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit(f"output already exists: {out}; use a new run tag or explicitly resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({
        "scenario": args.scenario, "split": args.split, "method": args.method, "model": cfg["model"],
        "epsilon": args.epsilon, "alpha": alpha, "alpha_over_epsilon": args.alpha_ratio,
        "optimizer": "Adam", "perceptual_weight": args.perceptual_weight,
        "initialization": "rand_linf_ball" if args.rand_init else "zeros",
        "steps": steps,
        "seed": args.seed, "pilot": args.pilot,
        "max_examples": args.max_examples,
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "example_ids_file": str(args.example_ids_file) if args.example_ids_file else None,
        "snr_reference": "load_music_by_name 24 kHz carrier (peak-normalized to 0.95), PCM_16 round-trip",
        "warmup_ratio": (
            0.0 if (args.method == "original_waveform" and args.scenario == "s3a")
            else (0.2 if args.method == "original_waveform" else 0.3)
        ),
        "post_warmup_schedule": (
            "alternate direct/Opus-BPDA" if args.method != "original_waveform"
            else ("all Opus-BPDA" if args.scenario == "s1" else "direct only")
        ),
        "opus_train_kbps": list(TRAIN_OPUS_KBPS) if not (args.method == "original_waveform" and args.scenario == "s3a") else [],
        "spectral_cutoff_hz": 4000 if args.method == "bandlimited_4khz" else None,
        "checkpoint_rule": "last iterate",
    }, indent=2))

    base_eps = float(args.epsilon)
    attacker = None
    current_eps = None

    def build_attacker(eps_value: float):
        """Construct once. ``eps`` and ``alpha`` are read inside ``attack()``
        (alpha becomes the Adam lr there, eps is read by the projections), so a
        per-example budget is applied by mutating them rather than by rebuilding
        and reloading the victim model for every example."""
        a = float(eps_value * args.alpha_ratio)
        if args.method == "original_waveform" and args.scenario == "s3a":
            att = WaveformAttacker(
                target_model=cfg["model"], eps=eps_value, alpha=a,
                perceptual_weight=args.perceptual_weight, verbose=False,
            )
        else:
            att = RobustWaveformAttacker(
                target_model=cfg["model"], eps=eps_value, alpha=a,
                perceptual_weight=args.perceptual_weight, channel_mode="multi_bitrate",
                multi_bitrate_kbps=list(TRAIN_OPUS_KBPS), n_eot_samples=1,
                # The historical S1 baseline warmed up for 20% of steps and then
                # ran the codec on every step. The tuned arms instead reproduce
                # the latent attack's 30% warmup and direct/codec alternation.
                warmup_ratio=0.2 if args.method == "original_waveform" else 0.3,
                alternate_after_warmup=args.method != "original_waveform",
                spectral_cutoff_hz=4000.0 if args.method == "bandlimited_4khz" else 0.0,
                verbose=False,
            )
        att.rand_init = bool(args.rand_init)
        # Optimize the input only. Freezing weights preserves input gradients
        # and avoids accumulating unused victim-model parameter gradients.
        att.target_model.model.requires_grad_(False)
        return att

    existing = set()
    if args.resume and manifest.is_file():
        existing = {json.loads(line)["example_id"] for line in manifest.read_text().splitlines() if line.strip()}
    for pair_index, pair in pairs:
        example_id = stable_id(args.scenario, pair_index, pair["carrier"], pair["target"])
        if example_id in existing:
            print(f"SKIP {example_id}", flush=True)
            continue
        example_eps = float(eps_map.get(example_id, base_eps))
        if attacker is None:
            attacker = build_attacker(example_eps)
        if example_eps != current_eps:
            attacker.eps = example_eps
            attacker.alpha = float(example_eps * args.alpha_ratio)
        current_eps = example_eps
        seed_all(args.seed)
        carrier_path = ROOT / sc["carrier_root"] / pair["carrier"]
        wav = load_music_by_name(str(carrier_path), duration=duration_s(carrier_path))
        # The SNR reference is the loaded 24 kHz carrier itself -- identical
        # bytes for every arm on this example. Using each attacker's own
        # `result.original_wav` instead would reference the waveform arms to a
        # 16 kHz-bandlimited copy of the carrier while referencing the latent
        # arm to the true 24 kHz carrier, which is not a matched comparison.
        reference = wav.detach().squeeze().float().cpu().numpy()
        t0 = time.time()
        result = attacker.attack(
            wav, target_text=pair["target"], steps=steps, check_every=0,
            music_name=carrier_path.stem, prompt=sc["prompt"],
        )
        adversarial = result.adversarial_wav.detach().squeeze().float().cpu().numpy()
        n = min(len(reference), len(adversarial))
        reference, adversarial = reference[:n], adversarial[:n]
        # How much of the measured distortion is the attacker's own 24k->16k->24k
        # resampling rather than the adversarial perturbation. Recorded, not corrected.
        attacker_ref = result.original_wav.detach().squeeze().float().cpu().numpy()[:n]
        metrics = write_pcm16_pair(audio_dir, example_id, reference, adversarial, ENCODEC_SAMPLE_RATE)
        import numpy as _np
        _den = float(_np.mean((attacker_ref.astype(_np.float64) - reference.astype(_np.float64)) ** 2))
        metrics["resample_only_snr_db"] = (
            float("inf") if _den == 0 else
            float(10.0 * _np.log10(float(_np.mean(reference.astype(_np.float64) ** 2)) / _den))
        )
        metrics["snr_db_vs_attacker_reference"] = float(
            10.0 * _np.log10(
                float(_np.mean(attacker_ref.astype(_np.float64) ** 2))
                / max(float(_np.mean((adversarial.astype(_np.float64) - attacker_ref.astype(_np.float64)) ** 2)), 1e-20)
            )
        )
        # Serialization leakage of the perturbation *itself*. Differencing the
        # delivered attack against the delivered true carrier also contains the
        # pipeline's 24k->16k->24k band loss, which is identical for every
        # waveform arm and is not attack leakage; differencing against the
        # attacker's own reference isolates what resampling plus PCM_16 added.
        _attref_path = audio_dir / f"{example_id}_attacker_ref.wav"
        sf.write(_attref_path, attacker_ref.astype("float32"), ENCODEC_SAMPLE_RATE, subtype="PCM_16")
        _attref_pcm, _ = sf.read(_attref_path, dtype="float32")
        _adv_pcm, _ = sf.read(metrics["adv_path"], dtype="float32")
        _m = min(len(_attref_pcm), len(_adv_pcm))
        metrics["attacker_ref_path"] = str(_attref_path)
        metrics["perturbation_above_4khz_fraction_post_serialization"] = (
            above_cutoff_energy_fraction(
                _adv_pcm[:_m].astype("float64") - _attref_pcm[:_m].astype("float64"),
                ENCODEC_SAMPLE_RATE, 4000.0,
            )
        )
        _d16 = getattr(attacker, "_last_delta_16k", None)
        metrics["delta16k_above_4khz_fraction"] = (
            None if _d16 is None else
            above_cutoff_energy_fraction(_d16.squeeze().numpy(), TARGET_SAMPLE_RATE, 4000.0)
        )
        metrics["delta16k_peak"] = None if _d16 is None else float(_d16.abs().max())
        # The <=4 kHz arm enforces its L-inf budget by a single global scale, so
        # a realized peak below epsilon means the valid-range projection bound
        # instead. SNR(eps) is then no longer log-linear in eps and the budget
        # solve does not apply to this example, so flag it explicitly.
        metrics["range_projection_bound"] = getattr(attacker, "_last_range_bound", None)
        metrics["range_scale"] = getattr(attacker, "_last_range_scale", None)
        metrics["eps_scale"] = getattr(attacker, "_last_eps_scale", None)
        metrics["peak_below_budget"] = (
            None if _d16 is None else bool(float(_d16.abs().max()) < example_eps - 1e-6)
        )
        row = {
            "example_id": example_id, "scenario": args.scenario, "split": args.split,
            "pair_index": pair_index, "method": args.method, "model": cfg["model"], "carrier": pair["carrier"],
            "source_carrier_path": str(carrier_path), "target": pair["target"],
            "prompt": sc["prompt"], "epsilon": example_eps,
            "alpha": example_eps * args.alpha_ratio,
            "perceptual_weight": args.perceptual_weight,
            "epsilon_source": "per_example_map" if example_id in eps_map else "cli",
            "alpha_over_epsilon": args.alpha_ratio, "steps": steps, "seed": args.seed,
            "elapsed_s": time.time() - t0, "in_memory_clean_output": str(result.original_output),
            "in_memory_attack_output": str(result.adversarial_output), **metrics,
        }
        append_jsonl(manifest, row)
        print(f"{example_id}: {row['snr_db']:.2f} dB, {row['elapsed_s']:.1f}s -> {metrics['adv_path']}", flush=True)


if __name__ == "__main__":
    main()
