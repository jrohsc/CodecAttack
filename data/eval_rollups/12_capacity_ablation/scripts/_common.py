"""Shared helpers for capacity-cliff ablation drivers."""
from __future__ import annotations

import json
from pathlib import Path

ATTACK_TAG = "qwen2_audio_eps1.0_multibitrate"


def count_words(text: str) -> int:
    return len(text.split())


def max_tokens_for_target(word_count: int) -> int:
    return max(60, word_count + 30)


def bundle_path_expA(root: Path, *, carrier_id: str, target_id: str) -> Path:
    return root / "expA_words" / ATTACK_TAG / carrier_id / target_id


def bundle_path_expB(root: Path, *, carrier_id: str, duration_s: int,
                     target_id: str) -> Path:
    return root / "expB_duration" / ATTACK_TAG / carrier_id / \
           f"dur_{duration_s}s__{target_id}"


def load_targets_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)
