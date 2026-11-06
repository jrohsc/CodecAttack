"""Load a scenario folder into a `Scenario` dataclass.

Convention (one dir per scenario):

    scenarios/NN_<name>/
        scenario.json     # all structured fields
        system_prompt.txt # long system prompt (optional; inline also OK)
        target.txt        # target adversarial output string (optional; inline also OK)
        audio/            # directory or symlink with adversarial wavs
        README.md         # human-readable rationale
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import ChannelSpec, Scenario, ToolSchema


def _read_if_exists(path: Path) -> str | None:
    return path.read_text().strip() if path.is_file() else None


def load_scenario(scenario_dir: str | Path) -> Scenario:
    d = Path(scenario_dir)
    spec_path = d / "scenario.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"no scenario.json under {d}")

    raw = json.loads(spec_path.read_text())

    # Sibling-file overrides (keep scenario.json terse for humans)
    if (sp := _read_if_exists(d / "system_prompt.txt")) is not None:
        raw["system_prompt"] = sp
    if (tg := _read_if_exists(d / "target.txt")) is not None:
        raw["target_text"] = tg
    # Auto-generated per-file targets from build_target_maps.py
    tmap_file = d / "_target_map.json"
    if tmap_file.is_file():
        auto = json.loads(tmap_file.read_text())
        existing = raw.get("audio_target_map", {})
        # Inline scenario.json wins over auto-generated
        merged = {**auto, **existing}
        raw["audio_target_map"] = merged

    # audio/ dir -> collect .wav files
    audio_dir = d / "audio"
    audio_paths: list[Path] = []
    if audio_dir.exists():
        audio_paths = sorted(audio_dir.rglob("*.wav"))
        # Fall back to MP3 if no wavs (raw music carriers)
        if not audio_paths:
            audio_paths = sorted(audio_dir.rglob("*.mp3"))
    # Explicit paths in scenario.json override discovery
    if "audio_paths" in raw:
        audio_paths = [Path(p) for p in raw["audio_paths"]]

    tools = [ToolSchema(**t) for t in raw.get("tools", [])]
    channels = [ChannelSpec(kind=c["kind"], params=c.get("params", {}))
                for c in raw.get("channels", [{"kind": "clean"}])]

    return Scenario(
        id=raw["id"],
        category=raw["category"],
        title=raw.get("title", raw["id"]),
        description=raw.get("description", ""),
        system_prompt=raw.get("system_prompt", ""),
        tools=tools,
        target_text=raw.get("target_text", ""),
        audio_target_map=raw.get("audio_target_map", {}),
        audio_meta_map=raw.get("audio_meta_map", {}),
        adversarial_goal=raw.get("adversarial_goal", ""),
        expected_safe_behavior=raw.get("expected_safe_behavior", ""),
        attack_success_if=raw.get("attack_success_if", ""),
        audio_paths=audio_paths,
        music_carriers=raw.get("music_carriers", []),
        channels=channels,
        target_models=raw.get("target_models", ["qwen2_audio"]),
        metrics=raw.get("metrics", ["exact_match", "wer"]),
        turns=raw.get("turns", 1),
        first_turn_is_agent=raw.get("first_turn_is_agent", False),
        reused_from=raw.get("reused_from", ""),
    )


def list_scenarios(scenarios_root: str | Path) -> list[Path]:
    """Return sorted list of scenario directories."""
    root = Path(scenarios_root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "scenario.json").is_file())
