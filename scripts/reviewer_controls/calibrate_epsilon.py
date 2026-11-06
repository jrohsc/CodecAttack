#!/usr/bin/env python3
"""Solve for the waveform budget that matches a reference arm's realized SNR.

The initial budget estimate assumes the realized perturbation scales with
``epsilon``, so carrier-referenced SNR would obey

    SNR(eps) = SNR(eps0) - 20 log10(eps / eps0)

if the optimized direction stayed fixed and amplitude-range projection did not
bind. The assumption can fail sharply for the 4 kHz arm: changing epsilon also
changes the Adam trajectory, and the global valid-range scale may bind. The
estimate is therefore only a probe; saved/reloaded WAV SNR must be verified.
Selection uses only distortion, with no delivery-codec outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path, required=True,
                   help="Manifest from a run at eps0 (the attack to calibrate).")
    p.add_argument("--reference", type=Path, required=True,
                   help="Manifest of the arm whose SNR is being matched (latent).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["global", "per_example"], default="per_example")
    p.add_argument("--verify", type=Path, default=None,
                   help="Optional second manifest at the solved eps; reports residual.")
    p.add_argument("--range-guard", type=float, default=0.999,
                   help="Flag examples whose probe adversarial peak already "
                        "exceeds this, where the valid-range projection may bind.")
    args = p.parse_args()

    probe = {r["example_id"]: r for r in read_jsonl(args.probe)}
    ref = {r["example_id"]: r for r in read_jsonl(args.reference)}
    shared = sorted(set(probe) & set(ref))
    if not shared:
        raise SystemExit("probe and reference share no example_id")
    missing = sorted((set(probe) | set(ref)) - set(shared))

    eps0 = {float(probe[e]["epsilon"]) for e in shared}
    if len(eps0) != 1:
        raise SystemExit(f"probe mixes epsilons: {sorted(eps0)}")
    eps0 = eps0.pop()

    per_example, gaps = {}, []
    for e in shared:
        gap = float(probe[e]["snr_db"]) - float(ref[e]["snr_db"])
        gaps.append(gap)
        per_example[e] = {
            "epsilon": eps0 * (10.0 ** (gap / 20.0)),
            "probe_snr_db": float(probe[e]["snr_db"]),
            "target_snr_db": float(ref[e]["snr_db"]),
            "probe_gap_db": gap,
            "probe_peak_near_range_limit": float(probe[e].get("post_peak", 0.0)) >= args.range_guard,
            "probe_range_projection_bound": bool(probe[e].get("range_projection_bound") or False),
            "probe_peak_below_budget": bool(probe[e].get("peak_below_budget") or False),
            "probe_clipped_samples": int(probe[e].get("clipped_samples", 0)),
        }
    gaps = np.asarray(gaps, dtype=float)
    eps_global = eps0 * (10.0 ** (float(gaps.mean()) / 20.0))
    # Examples whose probe already hit the valid-range projection violate the
    # log-linear assumption; the solved budget for them is an extrapolation and
    # the verify pass is what decides whether it landed.
    bound = sorted(e for e, v in per_example.items() if v["probe_range_projection_bound"])
    clipped = sorted(e for e, v in per_example.items() if v["probe_clipped_samples"] > 0)

    payload = {
        "status": "frozen",
        "mode": args.mode,
        "eps0": eps0,
        "selection_input": "probe SNR vs reference SNR only; no delivery-codec outcome used",
        "n_examples": len(shared),
        "range_projection_bound_in_probe": bound,
        "clipped_in_probe": clipped,
        "extrapolated_examples": len(bound),
        "n_unmatched": len(missing),
        "unmatched_example_ids": missing[:10],
        "unmatched_truncated": len(missing) > 10,
        "global_epsilon": eps_global,
        "probe_gap_db": {
            "mean": float(gaps.mean()), "median": float(np.median(gaps)),
            "min": float(gaps.min()), "max": float(gaps.max()),
            "sd": float(gaps.std(ddof=1)) if len(gaps) > 1 else None,
        },
        "per_example": per_example,
        "epsilon_map": ({e: per_example[e]["epsilon"] for e in shared}
                        if args.mode == "per_example"
                        else {e: eps_global for e in shared}),
    }

    if args.verify is not None:
        ver = {r["example_id"]: r for r in read_jsonl(args.verify)}
        rows = []
        for e in shared:
            if e not in ver:
                continue
            got = float(ver[e]["snr_db"])
            want = float(ref[e]["snr_db"])
            rows.append({
                "example_id": e, "realized_snr_db": got, "target_snr_db": want,
                "difference_db": got - want,
                "epsilon_used": float(ver[e]["epsilon"]),
            })
        d = np.asarray([r["difference_db"] for r in rows], dtype=float)
        payload["verification"] = {
            "n": len(rows),
            "per_example_difference_db": rows,
            "mean_db": float(d.mean()) if len(d) else None,
            "median_db": float(np.median(d)) if len(d) else None,
            "max_abs_db": float(np.abs(d).max()) if len(d) else None,
            "n_within_1db": int((np.abs(d) <= 1.0).sum()),
            "all_within_1db": bool(len(d) and np.all(np.abs(d) <= 1.0)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("per_example", "epsilon_map")}, indent=2))


if __name__ == "__main__":
    main()
