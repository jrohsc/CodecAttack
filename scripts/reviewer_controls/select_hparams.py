#!/usr/bin/env python3
"""Freeze the learning rate from development data and training codecs only.

Budget selection is *not* done here: the epsilon that matches the latent SNR is
solved in closed form by ``calibrate_epsilon.py`` from distortion alone, which
keeps it independent of any attack outcome.  This script therefore has one job
-- pick alpha/epsilon -- and it refuses to run if a held-out delivery codec or a
non-development example reaches its input.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TRAIN_CHANNELS = ("clean", "opus16", "opus24", "opus32", "opus64", "opus128")
HELDOUT_PREFIXES = ("mp3", "aac")


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def score(rows: list[dict]) -> tuple[int, int]:
    """Validation objective: total exact target matches over clean + the Opus
    training grid, with clean matches as the documented tie-break."""
    total = sum(int(bool(r["outcomes"][c]["match"]))
                for r in rows for c in TRAIN_CHANNELS if c in r["outcomes"])
    clean = sum(int(bool(r["outcomes"]["clean"]["match"])) for r in rows)
    return total, clean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--development-eval", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if any(c.startswith(HELDOUT_PREFIXES) for c in TRAIN_CHANNELS):
        raise AssertionError("held-out codec present in the selection objective")

    rows = [r for path in args.development_eval for r in load(path)]
    if not rows:
        raise SystemExit("no development rows")
    bad = [r["example_id"] for r in rows if r.get("split") != "development"]
    if bad:
        raise SystemExit(f"non-development examples in selection input: {sorted(set(bad))[:5]}")
    for r in rows:
        missing = [c for c in TRAIN_CHANNELS if c not in r.get("outcomes", {})]
        if missing:
            raise SystemExit(f"{r['example_id']} missing training channels {missing}")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scenario"], r["method"], float(r["alpha_over_epsilon"]))].append(r)

    counts = {k: len(v) for k, v in grouped.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"unbalanced learning-rate cells, refusing to select: {counts}")

    selected, table = {}, []
    by_arm = defaultdict(list)
    for (scenario, method, ratio), cell in grouped.items():
        total, clean = score(cell)
        row = {"scenario": scenario, "method": method, "alpha_over_epsilon": ratio,
               "n_examples": len(cell), "validation_successes": total,
               "clean_successes": clean,
               "max_possible": len(cell) * len(TRAIN_CHANNELS)}
        table.append(row)
        by_arm[(scenario, method)].append(row)

    for (scenario, method), cand in sorted(by_arm.items()):
        # Rank by the objective, then clean successes, then the smaller step.
        best = max(cand, key=lambda r: (r["validation_successes"], r["clean_successes"],
                                        -r["alpha_over_epsilon"]))
        ties = [c for c in cand
                if (c["validation_successes"], c["clean_successes"])
                == (best["validation_successes"], best["clean_successes"])]
        selected[f"{scenario}:{method}"] = {
            "alpha_over_epsilon": best["alpha_over_epsilon"],
            "validation_successes": best["validation_successes"],
            "clean_successes": best["clean_successes"],
            "n_development_examples": best["n_examples"],
            "tied_with": sorted(c["alpha_over_epsilon"] for c in ties
                                if c["alpha_over_epsilon"] != best["alpha_over_epsilon"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "frozen",
        "selection_channels": list(TRAIN_CHANNELS),
        "heldout_codecs_used": False,
        "objective": "total exact target-substring matches over clean + Opus "
                     "16/24/32/64/128 on development examples; ties broken by "
                     "clean matches, then by the smaller alpha/epsilon",
        "selected": selected,
        "grid": sorted(table, key=lambda r: (r["scenario"], r["method"], r["alpha_over_epsilon"])),
    }, indent=2))
    print(json.dumps({"selected": selected}, indent=2))


if __name__ == "__main__":
    main()
