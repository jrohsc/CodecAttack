"""Scenario runner — new layout (2026-04-15).

Per-(scenario × target_model × eps) results bundle:

    results/<scenario>/<target_model>/eps_<eps>/
      audio/                  # adversarial wavs (produced by attack driver)
      channels/<channel_tag>/ # per-channel routed wavs (produced here)
      results.jsonl           # realtime rich rows: one per (carrier, channel) eval

Rich row fields per `event=eval` record:
    scenario, attack_target_model, eval_model, eps, channel_tag,
    carrier, carrier_clean_path, category, target_command, clean_text,
    adv_audio_path, routed_audio_path,
    model_output, metrics: {name: value}, elapsed_s, ts

Scenarios/ holds configs + dataset pointers only; adv wavs and eval outputs
live exclusively under results/.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path

from .adapters import get_adapter
from .channels import get_channel
from .metrics import get_metric
from .models import ChannelSpec, MetricResult, RunResult, Scenario


def _apply_metric(name: str, output: str, scenario: Scenario, **kwargs) -> MetricResult:
    fn = get_metric(name)
    return fn(output=output, scenario=scenario, **kwargs)


_PAIR_IDX_RE = re.compile(r"_pair(\d+)_attacked")


def _load_carrier_map(path: Path) -> list[dict]:
    """Return the raw ordered list of pair dicts. Kept as a list (not a dict
    keyed by carrier) because multiple pairs may share the same carrier; the
    adv-wav name embeds `__pair<NNN>_attacked` to disambiguate."""
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("pairs", [])


def _find_carrier_record(pairs: list[dict], adv_wav: Path) -> dict:
    """Match adv wav to its pair record. Prefers the `pair<NNN>` index embedded
    in the adv-wav name; falls back to substring match on the carrier stem
    with any audio extension stripped (.wav, .mp3, .flac, etc.)."""
    name = adv_wav.name
    m = _PAIR_IDX_RE.search(name)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(pairs):
            return {**pairs[idx]}
    for rec in pairs:
        carrier_name = rec.get("carrier", "")
        if not carrier_name:
            continue
        carrier_stem = Path(carrier_name).stem
        if carrier_stem and carrier_stem in name:
            return {**rec, "carrier": carrier_name}
    return {}


def _load_done(jsonl_path: Path) -> set[tuple[str, str]]:
    """Resume key: (channel_tag, adv_stem). eval_model is implicit (one model per bundle)."""
    done: set[tuple[str, str]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "eval":
                continue
            tag = d.get("channel_tag", "")
            adv_path = d.get("adv_audio_path", "")
            stem = Path(adv_path).name if adv_path else ""
            if tag and stem:
                done.add((tag, stem))
    return done


def _route_one(
    scenario: Scenario,
    adv_path: Path,
    channel_spec: ChannelSpec,
    adapter_name: str,
    channels_dir: Path,
    carrier_rec: dict,
) -> dict:
    """Route one adv wav through one channel, run eval, return a rich row dict."""
    adapter = get_adapter(adapter_name)
    ch = get_channel(channel_spec.kind, **channel_spec.params)
    ch_tag = ch.tag()

    work_dir = channels_dir / ch_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    routed_path = work_dir / f"{adv_path.stem}__{ch_tag}.wav"
    ch.apply(adv_path, routed_path)

    history: list[dict] = []
    if scenario.first_turn_is_agent:
        opener = adapter.respond(scenario.system_prompt, audio_path=None, history=[], tools=scenario.tools)  # type: ignore[arg-type]
        history.append({"role": "assistant", "content": opener})

    t0 = time.time()
    output = adapter.respond(
        system_prompt=scenario.system_prompt,
        audio_path=str(routed_path),
        history=history,
        tools=scenario.tools,
    )
    elapsed = time.time() - t0

    # Use the carrier-specific target when computing metrics
    eff_scenario = scenario
    if carrier_rec.get("target"):
        eff_scenario = replace(scenario, target_text=carrier_rec["target"])

    metric_results = [
        _apply_metric(m, output=output, scenario=eff_scenario,
                      clean_wav=str(adv_path), attacked_wav=str(routed_path))
        for m in scenario.metrics
    ]

    return {
        "model_output": output,
        "metrics": {m.name: m.value for m in metric_results},
        "routed_audio_path": str(routed_path),
        "channel_tag": ch_tag,
        "elapsed_s": round(elapsed, 3),
    }


def run_bundle(
    scenario: Scenario,
    bundle_dir: Path,
    eval_model: str,
    attack_target_model: str | None = None,
    eps: float | None = None,
    carrier_map_path: Path | None = None,
) -> list[dict]:
    """Eval every wav in bundle_dir/audio/ through scenario.channels, on `eval_model`.

    Each row written to bundle_dir/results.jsonl is a full eval record carrying
    target, clean_text, adv_path, channel, model_output, and metrics.
    """
    audio_dir = bundle_dir / "audio"
    channels_dir = bundle_dir / "channels"
    jsonl = bundle_dir / "results.jsonl"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    adv_wavs = sorted(audio_dir.glob("*.wav"))
    if not adv_wavs:
        print(f"[skip] no adv wavs under {audio_dir}")
        return []

    cmap = _load_carrier_map(carrier_map_path) if carrier_map_path else []
    done = _load_done(jsonl)
    if done:
        print(f"[resume] {len(done)} (channel,adv_wav) pairs already in {jsonl.name}")

    rows: list[dict] = []
    for ch_spec in scenario.channels:
        ch_tag = get_channel(ch_spec.kind, **ch_spec.params).tag()
        for adv in adv_wavs:
            if (ch_tag, adv.name) in done:
                continue
            carrier_rec = _find_carrier_record(cmap, adv)
            one = _route_one(scenario, adv, ch_spec, eval_model, channels_dir, carrier_rec)
            row = {
                "event": "eval",
                "scenario": scenario.id,
                "attack_target_model": attack_target_model or eval_model,
                "eval_model": eval_model,
                "eps": eps,
                "channel_tag": one["channel_tag"],
                "carrier": carrier_rec.get("carrier", adv.name),
                "carrier_clean_path": str(carrier_rec.get("carrier_clean_path",
                                           adv.parent.parent / "carrier_unknown")),
                "category": carrier_rec.get("category"),
                "target_command": carrier_rec.get("target"),
                "clean_text": carrier_rec.get("clean_text"),
                "adv_audio_path": str(adv),
                "routed_audio_path": one["routed_audio_path"],
                "model_output": one["model_output"],
                "metrics": one["metrics"],
                "elapsed_s": one["elapsed_s"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            rows.append(row)
            with jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
    return rows


# Legacy alias used by cli.py — maps old-style invocation onto run_bundle for
# one (model, eps) bundle. Kept thin so existing `python -m audio_agent_bench
# run <scenario_dir>` still works if you point --out at a bundle_dir.
def run_scenario(scenario: Scenario, out_dir: Path) -> list[dict]:
    # Expect out_dir to be a per-(model, eps) bundle dir:
    #   results/<scen>/<model>/eps_<eps>/
    parts = out_dir.parts
    eval_model = None
    eps = None
    for m in ("qwen2_audio", "audio_flamingo", "qwen25_omni"):
        if m in parts:
            eval_model = m
            break
    for p in parts:
        if p.startswith("eps_"):
            try:
                eps = float(p[4:])
            except ValueError:
                pass
    if eval_model is None:
        eval_model = scenario.target_models[0] if scenario.target_models else "qwen2_audio"
    carrier_map_env = os.environ.get("AUDIO_AGENT_BENCH_CARRIER_MAP")
    carrier_map = Path(carrier_map_env) if carrier_map_env else None
    return run_bundle(scenario, out_dir, eval_model=eval_model,
                      attack_target_model=eval_model, eps=eps,
                      carrier_map_path=carrier_map)
