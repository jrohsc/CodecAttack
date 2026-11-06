"""Probe-pair selector for SNR calibration sweeps."""
from __future__ import annotations


def select_probe_pairs(pairs: list[dict], k: int = 5) -> list[dict]:
    """Return the first pair for each unique `carrier` in input order, capped at k."""
    seen: set[str] = set()
    picked: list[dict] = []
    for p in pairs:
        c = p.get("carrier")
        if c is None or c in seen:
            continue
        seen.add(c)
        picked.append(p)
        if len(picked) >= k:
            break
    return picked
