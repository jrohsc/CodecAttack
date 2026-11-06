#!/usr/bin/env bash
# Run every scenario × every model × every channel configured in each scenario.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/..:${PYTHONPATH:-}"
for S in scenarios/*/; do
  [[ -f "$S/scenario.json" ]] || continue
  echo ">>> $S"
  python -m audio_agent_bench run "$S" || echo "FAILED: $S (continuing)"
done
