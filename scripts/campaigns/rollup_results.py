"""Aggregate `results/<scen>/<model>/<bundle>/results.jsonl` into nested rollups.

Writes, from leaf to root:
  - results/<scen>/<model>/<bundle>/summaries.json  (one row per carrier: concat of audio/*_summary.json)
  - results/<scen>/<model>/<bundle>/rollup.json     (per-bundle, by channel × eval_model)
  - results/<scen>/<model>/rollup.json              (pretty JSON, all rows for that model)
  - results/<scen>/rollup.json                      (pretty JSON, all rows for that scenario)
  - results/rollup.json                             (pretty JSON, everything)

Each row is one JSON object keyed by (scenario, attack_target_model, bundle,
channel_tag, eval_model) with success_rate_pct, exact_match_pct, avg_wer,
avg_snr_db, n.

Usage:
    python scripts/rollup_results.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _mirror_jsonl_as_json(jsonl: Path) -> None:
    """Write a pretty `<stem>.json` next to the streaming `<stem>.jsonl` so the
    same data is available in a human-readable form. The .jsonl remains the
    authoritative append-target."""
    rows = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out = jsonl.with_suffix(".json")
    out.write_text(json.dumps({"n_rows": len(rows), "rows": rows}, indent=2))


def _aggregate_bundle(jsonl: Path, scen: str, target_model: str, bundle: str) -> dict:
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "sub": 0, "exact": 0,
                 "wer_sum": 0.0, "wer_n": 0,
                 "snr_sum": 0.0, "snr_n": 0}
    )
    attack_carriers: set[str] = set()
    attack_ok = 0
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = r.get("event")
            if evt == "attack":
                attack_carriers.add(r.get("carrier", ""))
                if r.get("status") == "ok":
                    attack_ok += 1
                continue
            if evt and evt != "eval":
                continue
            tag = r.get("channel_tag", "")
            eval_model = r.get("eval_model") or r.get("model") or target_model
            m = r.get("metrics") or {}
            if not isinstance(m, dict):
                continue
            s = buckets[(tag, eval_model)]
            s["n"] += 1
            if m.get("target_substring_match"):
                s["sub"] += 1
            if m.get("exact_match"):
                s["exact"] += 1
            w = m.get("wer")
            if isinstance(w, (int, float)):
                s["wer_sum"] += w
                s["wer_n"] += 1
            snr = m.get("snr_db")
            if isinstance(snr, (int, float)):
                s["snr_sum"] += snr
                s["snr_n"] += 1

    by_channel = []
    for (tag, eval_model), s in sorted(buckets.items()):
        n = s["n"]
        by_channel.append({
            "scenario": scen,
            "attack_target_model": target_model,
            "bundle": bundle,
            "channel_tag": tag,
            "eval_model": eval_model,
            "n": n,
            "success_rate_pct": round(100.0 * s["sub"] / n, 2) if n else None,
            "exact_match_pct": round(100.0 * s["exact"] / n, 2) if n else None,
            "avg_wer": round(s["wer_sum"] / s["wer_n"], 4) if s["wer_n"] else None,
            "avg_snr_db": round(s["snr_sum"] / s["snr_n"], 2) if s["snr_n"] else None,
        })
    return {
        "scenario": scen,
        "attack_target_model": target_model,
        "bundle": bundle,
        "source": str(jsonl.relative_to(ROOT)),
        "n_attack_carriers": len(attack_carriers),
        "n_attack_ok": attack_ok,
        "by_channel": by_channel,
    }


def _concat_summaries(bundle_dir: Path) -> int:
    """Flatten per-carrier `audio/*_summary.json` files into `summaries.json`
    next to `results.jsonl`. Returns number of summaries written."""
    audio_dir = bundle_dir / "audio"
    if not audio_dir.is_dir():
        return 0
    rows = []
    for p in sorted(audio_dir.glob("*_summary.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    out = bundle_dir / "summaries.json"
    legacy = bundle_dir / "summaries.jsonl"
    if rows:
        out.write_text(json.dumps({"n_rows": len(rows), "rows": rows}, indent=2))
    elif out.exists():
        out.unlink()
    # Drop legacy .jsonl so the two don't drift apart
    legacy.unlink(missing_ok=True)
    return len(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_json(path: Path, rows: list[dict]) -> None:
    """Pretty-print JSON with a small wrapper so it's human-readable at a glance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_rows": len(rows), "rows": rows}
    path.write_text(json.dumps(payload, indent=2))


def _walk_model_dir(model_dir: Path, scen_name: str, target_model_label: str) -> list[dict]:
    """Walk `model_dir/*/` (bundles). `target_model_label` is what appears in
    rollup rows — distinct for latent ('m') vs waveform ('m__waveform')."""
    model_rows: list[dict] = []
    for bundle_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        n_sum = _concat_summaries(bundle_dir)
        jsonl = bundle_dir / "results.jsonl"
        if not jsonl.is_file():
            if n_sum:
                print(f"  [summaries-only] {bundle_dir.relative_to(ROOT)}: {n_sum} carriers")
            continue
        _mirror_jsonl_as_json(jsonl)
        roll = _aggregate_bundle(jsonl, scen_name, target_model_label, bundle_dir.name)
        roll["n_carrier_summaries"] = n_sum
        (bundle_dir / "rollup.json").write_text(json.dumps(roll, indent=2))
        model_rows.extend(roll["by_channel"])
    if model_rows:
        _write_json(model_dir / "rollup.json", model_rows)
        (model_dir / "rollup.jsonl").unlink(missing_ok=True)
    return model_rows


def main() -> None:
    if not RESULTS.is_dir():
        print(f"no results dir: {RESULTS}")
        return

    all_rows: list[dict] = []
    for scen_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        scen_rows: list[dict] = []
        for model_dir in sorted(p for p in scen_dir.iterdir() if p.is_dir()):
            if model_dir.name != "waveform_comparison":
                scen_rows.extend(_walk_model_dir(
                    model_dir, scen_dir.name, model_dir.name))
                continue
            for inner_model_dir in sorted(
                    p for p in model_dir.iterdir() if p.is_dir()):
                if inner_model_dir.name == "calibration":
                    continue
                scen_rows.extend(_walk_model_dir(
                    inner_model_dir, scen_dir.name,
                    f"{inner_model_dir.name}__waveform"))
        if scen_rows:
            _write_json(scen_dir / "rollup.json", scen_rows)
            (scen_dir / "rollup.jsonl").unlink(missing_ok=True)
            all_rows.extend(scen_rows)

    _write_json(RESULTS / "rollup.json", all_rows)
    (RESULTS / "rollup.jsonl").unlink(missing_ok=True)
    print(f"wrote {len(all_rows)} rows across {RESULTS}/rollup.json")


if __name__ == "__main__":
    main()
