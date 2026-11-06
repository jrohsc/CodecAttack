"""Aggregate `results/<scenario>/<model>/eps_<eps>/results.jsonl` bundles into
CSV + markdown + heatmap. Also tolerates legacy `scenarios/**/results/*.jsonl`
layouts (pre-2026-04-15).

Usage:
    python scripts/summarize.py                               # all scenarios -> results_summary/
    python scripts/summarize.py --scenario 01_finance_voice_agent
    python scripts/summarize.py --out results_summary/
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _strip_tag(stem: str, tag: str) -> str:
    suffix = f"__{tag}"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _iter_records(scenario_dirs: list[Path]):
    """Yields (scenario_name, record) from three supported layouts:
      (a) NEW: results/<scen>/<model>/eps_<eps>/results.jsonl
      (b) LEGACY-per-model: scenarios/<scen>/results/<model>/results.jsonl
      (c) LEGACY-flat:      scenarios/<scen>/results/results.jsonl
    """
    for sd in scenario_dirs:
        scen_name = sd.name
        candidates: list[Path] = []
        # (a) new layout: sd is results/<scen>, walk any-depth for results.jsonl
        candidates += sorted(sd.rglob("results.jsonl"))
        # (b, c) legacy: sd is scenarios/<scen>
        candidates += sorted((sd / "results").glob("*/results.jsonl"))
        flat = sd / "results" / "results.jsonl"
        if flat.is_file():
            candidates.append(flat)
        seen_paths = set()
        for jsonl in candidates:
            if jsonl in seen_paths:
                continue
            seen_paths.add(jsonl)
            with jsonl.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Skip attack-event rows (no channel, no metrics) so aggregators stay correct
                    if r.get("event") == "attack":
                        continue
                    yield scen_name, r


def aggregate(scenario_dirs: list[Path]):
    """Return {(scenario, model, channel, attack_kind_or_''): stats}."""
    seen: set[tuple[str, str, str, str]] = set()
    stats = defaultdict(lambda: {"n": 0, "sub": 0, "wer_sum": 0.0, "snr_sum": 0.0, "snr_n": 0})
    for sc, r in _iter_records(scenario_dirs):
        model = r.get("model") or r.get("eval_model", "")
        tag = r.get("channel_tag", "")
        stem = Path(r.get("audio_path") or r.get("adv_audio_path", "")).stem
        stem = _strip_tag(stem, tag)
        key = (sc, model, tag, stem)
        if key in seen:
            continue
        seen.add(key)
        kind = (r.get("extra", {}).get("audio_meta", {}) or {}).get("attack_kind", "")
        bucket = (sc, model, tag, kind)
        s = stats[bucket]
        s["n"] += 1
        # `metrics` may be a list[{name,value}] (legacy) or dict[name->value] (new bench)
        raw = r.get("metrics", [])
        items = raw.items() if isinstance(raw, dict) else (
            (m.get("name"), m.get("value")) for m in raw
        )
        for name, val in items:
            if name == "target_substring_match" and val:
                s["sub"] += 1
            elif name == "wer" and isinstance(val, (int, float)):
                s["wer_sum"] += val
            elif name == "snr_db" and isinstance(val, (int, float)):
                s["snr_sum"] += val
                s["snr_n"] += 1
    return stats


def write_csv(stats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "model", "channel", "attack_kind", "n", "sub_pct", "avg_wer", "avg_snr_db"])
        for (sc, model, ch, kind), s in sorted(stats.items()):
            n = s["n"]
            sub_pct = 100.0 * s["sub"] / n if n else 0.0
            wer = s["wer_sum"] / n if n else 0.0
            snr = s["snr_sum"] / s["snr_n"] if s["snr_n"] else float("nan")
            w.writerow([sc, model, ch, kind, n, f"{sub_pct:.2f}", f"{wer:.3f}", f"{snr:.2f}"])


def _channel_order() -> list[str]:
    return [
        "clean",
        "opus192k", "opus128k", "opus64k", "opus32k", "opus24k", "opus16k",
        "g711_ulaw", "g722_64k",
    ]


def write_markdown(stats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Overall pivot (collapse attack_kind): scenario -> channel -> model -> (sub_pct, n)
    overall: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for (sc, model, ch, _kind), s in stats.items():
        if s["n"] == 0:
            continue
        c = overall[sc][ch][model]
        c[0] += s["sub"]
        c[1] += s["n"]

    lines: list[str] = ["# Audio-Agent Benchmark — summary", ""]
    scenarios = sorted({sc for (sc, *_r) in stats.keys()})
    for sc in scenarios:
        lines.append(f"## {sc}")
        lines.append("")
        sc_data = overall.get(sc, {})
        if sc_data:
            models = sorted({m for ch in sc_data for m in sc_data[ch]})
            lines.append("| channel | " + " | ".join(models) + " |")
            lines.append("|" + "---|" * (1 + len(models)))
            chans = [c for c in _channel_order() if c in sc_data]
            chans += sorted(c for c in sc_data if c not in chans)
            for ch in chans:
                row = [ch]
                for m in models:
                    cell = sc_data[ch].get(m)
                    if cell and cell[1]:
                        row.append(f"{100.0 * cell[0] / cell[1]:.1f}% ({cell[1]})")
                    else:
                        row.append("—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        # Per-attack-kind breakdown if present
        kinds = sorted({k for (s_, _m, _c, k) in stats if s_ == sc and k})
        if kinds:
            lines.append(f"### {sc} — by attack_kind")
            lines.append("")
            lines.append("| channel | model | " + " | ".join(kinds) + " |")
            lines.append("|" + "---|" * (2 + len(kinds)))
            pivot: dict[tuple[str, str], dict[str, tuple[float, int]]] = defaultdict(dict)
            for (s_, mm, cc, k), s in stats.items():
                if s_ != sc or not k or s["n"] == 0:
                    continue
                pivot[(cc, mm)][k] = (100.0 * s["sub"] / s["n"], s["n"])
            chan_sort = {c: i for i, c in enumerate(_channel_order())}
            for (cc, mm) in sorted(pivot, key=lambda p: (chan_sort.get(p[0], 999), p[0], p[1])):
                row = [cc, mm]
                for k in kinds:
                    cell = pivot[(cc, mm)].get(k)
                    row.append(f"{cell[0]:.1f}% ({cell[1]})" if cell else "—")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    path.write_text("\n".join(lines))


def write_heatmap(stats, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping heatmap")
        return
    by_sc: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (sc, model, ch, kind), s in stats.items():
        if kind or s["n"] == 0:
            continue
        by_sc[sc][model][ch] = 100.0 * s["sub"] / s["n"]
    scenarios = sorted(by_sc)
    if not scenarios:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for sc in scenarios:
        models = sorted(by_sc[sc])
        chans = [c for c in _channel_order() if any(c in by_sc[sc][m] for m in models)]
        chans += sorted({c for m in models for c in by_sc[sc][m]} - set(chans))
        mat = np.full((len(models), len(chans)), np.nan)
        for i, m in enumerate(models):
            for j, c in enumerate(chans):
                if c in by_sc[sc][m]:
                    mat[i, j] = by_sc[sc][m][c]
        fig, ax = plt.subplots(figsize=(max(6, len(chans) * 0.9), 1.2 + 0.5 * len(models)))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels(chans, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(chans)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                            color="white" if mat[i, j] < 50 else "black", fontsize=8)
        ax.set_title(f"{sc} — target-substring success (%)")
        fig.colorbar(im, ax=ax, label="sub %")
        fig.tight_layout()
        out = path.parent / f"heatmap_{sc}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=None, help="only this scenario dir name")
    p.add_argument("--out", default=str(ROOT / "results_summary"))
    args = p.parse_args()

    results_root = ROOT / "results"
    scen_root = ROOT / "scenarios"
    if args.scenario:
        # Accept either new-layout (results/<scen>) or scenario-dir (scenarios/<scen>)
        candidates = [results_root / args.scenario, scen_root / args.scenario]
        dirs = [c for c in candidates if c.is_dir()]
    else:
        dirs = []
        if results_root.is_dir():
            dirs += sorted([d for d in results_root.iterdir() if d.is_dir()])
        # Legacy fallback
        dirs += sorted({r.parents[2] for r in scen_root.rglob("results/*/results.jsonl")})
        dirs += sorted({r.parent.parent for r in scen_root.rglob("results/results.jsonl")
                        if r.parent.name == "results"})
        dirs = sorted(set(dirs))

    stats = aggregate(dirs)
    out = Path(args.out)
    write_csv(stats, out / "summary.csv")
    write_markdown(stats, out / "summary.md")
    write_heatmap(stats, out / "heatmap.png")
    print(f"wrote: {out}/summary.csv, summary.md, heatmap_*.png")


if __name__ == "__main__":
    main()
