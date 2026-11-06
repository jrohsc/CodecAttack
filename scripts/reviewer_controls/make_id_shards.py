#!/usr/bin/env python3
"""Create deterministic disjoint ID shards, optionally excluding completed IDs."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--split", default="evaluation")
    p.add_argument("--completed", type=Path, nargs="*", default=[])
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    a = p.parse_args()
    import csv
    rows = list(csv.DictReader(a.split_csv.open()))
    ids = [r["example_id"] for r in rows if r["split"] == a.split]
    done = set()
    for path in a.completed:
        if path.is_file():
            done.update(json.loads(x)["example_id"] for x in path.read_text().splitlines() if x.strip())
    ids = [x for x in ids if x not in done]
    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for shard in range(a.num_shards):
        selected = ids[shard::a.num_shards]
        Path(f"{a.output_prefix}{shard}.txt").write_text("".join(x + "\n" for x in selected))
    print(json.dumps({"total_selected": len(ids), "excluded_completed": len(done),
                      "shard_sizes": [len(ids[s::a.num_shards]) for s in range(a.num_shards)]}))


if __name__ == "__main__":
    main()

