"""Read a --grad-cache bundle and report whether the staleness assumption holds.

The cache combines g_m(t - k) for every surrogate m. That is only legitimate if a
surrogate's gradient is still pointing the same way k steps later. Two numbers
decide it, and they must be read TOGETHER:

  self-cosine(k)  cos(g_m(t), g_m(t+k))  -- how much a member's own gradient
                  rotates while it sits in the cache.
  cross-cosine    cos(g_m, g_n), m != n  -- the floor. If self-cosine(k) decays
                  to this, the cached direction carries no more information
                  about model m than another model's gradient does, and the
                  cache is buying nothing.

Usage:
    python analyze_grad_cache.py <bundle-dir> [<bundle-dir> ...]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def summarize(bundle: Path):
    tgt = bundle / "targets.json"
    man = bundle / "manifest.json"
    if not tgt.is_file():
        return None
    targets = json.loads(tgt.read_text())
    manifest = json.loads(man.read_text()) if man.is_file() else {}

    self_cos = defaultdict(list)      # k -> [cos]
    cross_cos = []
    losses = []
    for rec in targets.values():
        for s in rec.get("grad_staleness", []):
            self_cos[s["k"]].append(s["cosine"])
        for a in rec.get("gradient_alignment", []):
            cross_cos.extend(a.get("cosine", {}).values())
        if rec.get("final_loss") is not None:
            losses.append(float(rec["final_loss"]))
    return {
        "bundle": bundle,
        "manifest": manifest,
        "n_clips": len(targets),
        "self_cos": self_cos,
        "cross_cos": cross_cos,
        "losses": losses,
    }


def ms(xs):
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{st.mean(xs):.3f} ± {st.pstdev(xs):.3f}"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = [r for r in (summarize(Path(b)) for b in sys.argv[1:]) if r]
    if not rows:
        raise SystemExit("no bundle with a targets.json among the arguments")

    print("## Arms\n")
    print("| arm | rule | block | offload | n clips | final loss | peak VRAM |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["manifest"]
        rule = ("cache" if m.get("grad_cache") else
                "model_eot" if m.get("model_eot") else "average")
        vram = f"{m['peak_vram_gb']:.1f} GB" if m.get("peak_vram_gb") else "--"
        print(f"| `{r['bundle'].name}` | {rule} | {m.get('cache_block','--')} | "
              f"{m.get('cpu_offload', False)} | {r['n_clips']} | "
              f"{ms(r['losses'])} | {vram} |")

    print("\n## Gradient staleness — does a cached direction survive k steps?\n")
    print("`self` is cos(g_m(t), g_m(t+k)); `cross` is the same-step cosine "
          "between different members, i.e. the floor.\n")
    for r in rows:
        if not r["self_cos"]:
            continue
        print(f"**{r['bundle'].name}** (cross-model floor: {ms(r['cross_cos'])}, "
              f"n={len(r['cross_cos'])})\n")
        print("| k (steps stale) | self-cosine | n |")
        print("|---|---|---|")
        for k in sorted(r["self_cos"]):
            xs = r["self_cos"][k]
            print(f"| {k} | {ms(xs)} | {len(xs)} |")
        floor = st.mean(r["cross_cos"]) if r["cross_cos"] else 0.0
        worst_k = max(r["self_cos"])
        worst = st.mean(r["self_cos"][worst_k])
        verdict = ("cache is VALID: a member's own gradient at max staleness is "
                   "still far more aligned with itself than with another member"
                   if worst > floor + 0.15 else
                   "cache is NOT buying a real direction: at max staleness a "
                   "member's own gradient is no more aligned with itself than "
                   "with a different model's")
        print(f"\n> max k={worst_k}: self {worst:.3f} vs floor {floor:.3f} — {verdict}\n")


if __name__ == "__main__":
    main()
