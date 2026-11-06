"""Data models for the audio-agent safety benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Category = Literal[
    "voice_agent_auth_bypass",     # 01
    "robot_command_hijack",        # 02
    "interview_verdict_override",  # 03
    "transcribe_injection",        # 04
    "method_comparison",           # 05 (latent vs waveform)
    "channel_robustness",          # 06 (simulated physical OTA)
]

ChannelKind = Literal[
    "clean",
    "opus",
    "g711_mulaw",
    "g722",
    "packet_loss",
    "physical_sim",
    "chain",
]


@dataclass
class ToolSchema:
    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class ChannelSpec:
    kind: ChannelKind
    params: dict[str, Any] = field(default_factory=dict)

    def tag(self) -> str:
        if not self.params:
            return self.kind
        parts = [self.kind] + [f"{k}{v}" for k, v in sorted(self.params.items())]
        return "_".join(str(p) for p in parts)


@dataclass
class Scenario:
    """Self-contained benchmark record loaded from scenarios/NN_*/scenario.json."""

    id: str
    category: Category
    title: str
    description: str

    # Agent config
    system_prompt: str
    tools: list[ToolSchema] = field(default_factory=list)

    # Attacker objective
    target_text: str = ""
    adversarial_goal: str = ""
    expected_safe_behavior: str = ""
    attack_success_if: str = ""

    # Data
    audio_paths: list[Path] = field(default_factory=list)
    music_carriers: list[str] = field(default_factory=list)

    # Per-file target overrides — keyed by wav basename.
    # Falls back to top-level `target_text` when a wav isn't listed.
    audio_target_map: dict[str, str] = field(default_factory=dict)
    # Per-file metadata (e.g. attack_kind=latent vs waveform) for analysis.
    audio_meta_map: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Eval config
    channels: list[ChannelSpec] = field(default_factory=lambda: [ChannelSpec("clean")])
    target_models: list[str] = field(default_factory=lambda: ["qwen2_audio"])
    metrics: list[str] = field(default_factory=lambda: ["exact_match", "wer"])
    turns: int = 1  # >1 triggers MultiTurnRunner
    first_turn_is_agent: bool = False  # model-initiated (e.g. "Please verify your PIN")

    # Provenance
    reused_from: str = ""  # e.g. "1_results_qa/benchmark_qwen2_audio_eps1.0_calm_1_saycan"


@dataclass
class MetricResult:
    name: str
    value: float | bool | str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    scenario_id: str
    model: str
    channel_tag: str
    audio_path: str
    output_text: str
    metrics: list[MetricResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
