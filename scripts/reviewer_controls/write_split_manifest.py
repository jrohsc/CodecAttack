#!/usr/bin/env python3
"""Write the frozen S1 development/evaluation IDs with source-recording keys."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reviewer_controls.run_attack import load_config, partition_pairs, stable_id


def main() -> None:
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument("--config", type=Path, default=here / "config.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    rows = []
    for split in ("development", "evaluation"):
        for pair_index, pair in partition_pairs(cfg, "s1", split, pilot=False):
            rows.append({
                "example_id": stable_id("s1", pair_index, pair["carrier"], pair["target"]),
                "split": split,
                "pair_index": pair_index,
                "source_recording": pair["carrier"],
                "target": pair["target"],
            })
    assert len(rows) == 50
    assert len({r["example_id"] for r in rows}) == 50
    assert len({r["source_recording"] for r in rows}) == 50
    assert sum(r["split"] == "development" for r in rows) == 10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
