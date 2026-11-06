#!/usr/bin/env bash
# Usage: bash scripts/run_scenario.sh <scenario_dir> [model] [channel]
set -euo pipefail
SCENARIO="${1:-scenarios/diagnostics/smoke_latent_vs_waveform}"
MODEL="${2:-qwen2_audio}"
CHANNEL="${3:-clean}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/..:${PYTHONPATH:-}"
python -m audio_agent_bench run "$SCENARIO" --model "$MODEL" --channel "$CHANNEL"
