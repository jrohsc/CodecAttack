#!/bin/bash
# Re-run S2 Omni bench eval after 2026-04-17 adapter size-fix
# (Qwen25OmniAdapter default size 3B -> 7B, aligning eval with attack-time
# 7B target model; 7B/3B mismatch was causing 0% flat evals).
set -euo pipefail
cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

SCEN=scenarios/02_interview_screening
export AUDIO_AGENT_BENCH_CARRIER_MAP=data/speech/02_interview/carrier_to_target.json
export PYTHONPATH="src:${REPO_ROOT}:${PYTHONPATH:-}"
envbin=$(dirname "$OMNI")
export PATH="${envbin}:$PATH"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {
  local tag="$1"; shift
  local eps="$1"; shift
  local bundle="results/02_interview_screening/qwen25_omni/eps_${eps}"
  local log="scripts/re_eval_s2_omni7b_eps${eps}.log"
  echo "[sweep $(ts)] START ${tag}"
  "$OMNI" -m audio_agent_bench run "$SCEN" \
      --model qwen25_omni --out "$bundle" > "$log" 2>&1
  echo "[sweep $(ts)] DONE  ${tag}  -> $bundle/results.jsonl"
}

run omni7b_eps0.5 0.5
run omni7b_eps1.0 1.0

echo "[sweep $(ts)] SWEEP COMPLETE — refreshing rollups"
"$CA" scripts/rollup_results.py
