#!/usr/bin/env bash
# S1 evaluation pipeline — eval pre-generated adv wavs through the S1 channel
# grid on each model it was attacked against. Adv wavs live in:
#   results/01_finance_voice_agent/<model>/eps_<eps>/audio/
#
# Output per (model, eps) bundle:
#   results/01_finance_voice_agent/<model>/eps_<eps>/
#     channels/<channel_tag>/*.wav       ← routed per-channel wavs
#     results.jsonl                       ← streamed rich rows (one per eval)
#
# Scenarios/ holds config only; nothing under scenarios/ gets written to here.
set -euo pipefail
cd "$(dirname "$0")/.."

SCEN_DIR="scenarios/01_finance_voice_agent"
RESULTS_BASE="results/01_finance_voice_agent"
CARRIER_MAP="data/speech/01_finance/carrier_to_target.json"

export AUDIO_AGENT_BENCH_CARRIER_MAP="$CARRIER_MAP"  # runner reads from env if set
export PYTHONPATH="$PWD/src:$PWD/..:${PYTHONPATH:-}"

run_bundle() {
  local env_name="$1"; local model="$2"; local eps="$3"
  local bundle="$RESULTS_BASE/$model/eps_$eps"
  if [[ ! -d "$bundle/audio" ]]; then
    echo "[skip] no adv wavs at $bundle/audio — run run_s1_attack.py first"
    return 0
  fi
  echo "=== eval  model=$model  eps=$eps  (env=$env_name) ==="
  conda run -n "$env_name" --no-capture-output \
    python -m audio_agent_bench run "$SCEN_DIR" \
      --model "$model" \
      --out "$bundle"
}

# Qwen2-Audio (white-box-attacked bundles)
for eps in 0.5 1.0 0.3; do run_bundle codec-attack qwen2_audio "$eps"; done

# Audio Flamingo 3
for eps in 0.5 1.0 0.3; do run_bundle flamingo3 audio_flamingo "$eps"; done

# Qwen2.5-Omni
for eps in 0.5 1.0 0.3; do run_bundle qwen-omni qwen25_omni "$eps"; done

conda run -n codec-attack --no-capture-output \
  python scripts/summarize.py --scenario 01_finance_voice_agent || true

# Refresh rollup.json tree with new eval rows
conda run -n codec-attack --no-capture-output \
  python scripts/rollup_results.py || true

echo "=== S1 eval done. Bundles under $RESULTS_BASE/ ==="
