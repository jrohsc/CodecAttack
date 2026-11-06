#!/bin/bash
# S3b Omni eps=1.0 channel eval — 8 adv wavs already in place, only channel eval runs.
set -euo pipefail
cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

SCEN=scenarios/03b_copyright_bypass
export AUDIO_AGENT_BENCH_CARRIER_MAP=data/music/03b_copyright_bypass/carrier_to_target.json
export PYTHONPATH="src:${REPO_ROOT}:${PYTHONPATH:-}"
envbin=$(dirname "$OMNI")
export PATH="${envbin}:$PATH"

BUNDLE="results/03b_copyright_bypass/qwen25_omni/eps_1.0"
LOG="scripts/re_eval_s3b_omni_eps1.0.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[re_eval $(ts)] START s3b_omni_eps1.0"
"$OMNI" -m audio_agent_bench run "$SCEN" \
    --model qwen25_omni --out "$BUNDLE" > "$LOG" 2>&1
echo "[re_eval $(ts)] DONE  s3b_omni_eps1.0  -> $BUNDLE/results.jsonl"

echo "[re_eval $(ts)] refreshing rollups"
"$CA" scripts/rollup_results.py
echo "[re_eval $(ts)] DONE"
