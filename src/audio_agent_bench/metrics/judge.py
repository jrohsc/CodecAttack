"""LLM judge for policy violations. Wraps `external/whisper-inject-v2-fork/core/judge.py`.

Rubric loaded from `configs/prompts/judge_<scenario_category>.yaml`.

TODO (Phase 4): implement. Default judge model = GPT-4o-mini (cheap); Sonnet for spot-checks.
"""
from __future__ import annotations

from pathlib import Path

from ..models import MetricResult, Scenario


def policy_violation(output: str, scenario: Scenario, *, rubric_path: str | Path | None = None,
                     judge_model: str = "gpt-4o-mini", **_) -> MetricResult:
    raise NotImplementedError("Phase 4: wire to whisper_inject_v2_fork.core.judge.LLMJudge")
