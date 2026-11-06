"""
Aggregate the surrogate-pool ladder into the standing results table.

One row per (held-out victim, rung N). The measurement is the held-out victim's
success rate as the surrogate pool grows -- read as a DELTA against two controls,
never as a bare ASR%:

  * white-box ceiling  -- attack that victim directly. A transfer 0 under a
    ceiling of 0 says nothing about transfer; it says the attack never lands.
  * clean same-prompt  -- unattacked carriers under the identical bank prompt.
    The bank prompt alone floors natural speech near 0-10%, so the clean column
    is the floor every transfer number is measured from.

Bare counts, no confidence intervals (project rule).

Usage:
    python 0_all_combined/scripts/build_ladder_table.py [--out results/POOL_LADDER_TABLE.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LADDER = ROOT / "results_pool_ladder"
WHITEBOX = LADDER / "whitebox"

ALWAYS = ["kimi_audio", "qwen25_omni_3b"]


def _summary(bundle: Path, victim: str) -> dict:
    """Evaluator writes <bundle>/<victim>_EVAL/summary.json."""
    for name in (f"{victim}_EVAL", f"{victim}_HELDOUT", f"{victim}_MEMBER"):
        f = bundle / name / "summary.json"
        if f.is_file():
            try:
                return json.load(open(f))["stats"]
            except Exception:
                return {}
    return {}


def _cell(stats: dict, channel: str = "clean") -> str:
    if not stats:
        return "--"
    ch = stats.get("channels", {}).get(channel)
    if not ch:
        return "--"
    return f"{ch['exact']}/{stats['n']} ({ch['exact_pct']}%)"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "results_pool_ladder/POOL_LADDER_TABLE.md"))
    p.add_argument("--channel", default="clean",
                   help="Channel to report (clean | opus_64 | ...).")
    args = p.parse_args()

    manifests = sorted(LADDER.rglob("manifest.json"))
    if not manifests:
        raise SystemExit(f"no bundles under {LADDER}")

    rows, seen_victims = [], []
    for man in manifests:
        bundle = man.parent
        m = json.loads(man.read_text())
        victim, rung = m["victim"], m["rung"]
        if victim not in seen_victims:
            seen_victims.append(victim)
        n_wav = len(list((bundle / "audio").glob("*.wav")))
        row = {
            "victim": victim, "rung": rung,
            "surrogates": ",".join(m["surrogates"]),
            "n_wav": n_wav,
            "heldout": _cell(_summary(bundle, victim), args.channel),
        }
        for a in ALWAYS:
            row[a] = _cell(_summary(bundle, a), args.channel)
        rows.append(row)

    rows.sort(key=lambda r: (r["victim"], r["rung"]))

    lines = [
        "# Surrogate-pool scaling ladder — held-out transfer vs pool size",
        "",
        f"Channel: `{args.channel}`. Cells are exact-match counts (bare counts, no CIs).",
        "`n_wav` is the number of crafted clips actually on disk — a short count means",
        "the cell is INCOMPLETE (SLURM preemption logs COMPLETED on partial bundles),",
        "so backfill before reading the row.",
        "",
        "| victim (held out) | N | surrogates | n_wav | victim | kimi_audio | omni_3b |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['victim']} | {r['rung']} | `{r['surrogates']}` | {r['n_wav']} | "
            f"**{r['heldout']}** | {r['kimi_audio']} | {r['qwen25_omni_3b']} |")

    lines += ["", "## White-box ceilings (attack the victim directly)", "",
              "Without these every transfer 0 is unreadable.", "",
              "| victim | white-box |", "|---|---|"]
    for v in seen_victims:
        wb = WHITEBOX / f"wb_{v}"
        cells = [d for d in wb.glob("*/") if d.is_dir()] if wb.is_dir() else []
        got = "--"
        for d in cells:
            s = _summary(d, v)
            if s:
                got = _cell(s, args.channel); break
        lines.append(f"| {v} | {got} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[table] wrote {out}")


if __name__ == "__main__":
    main()
