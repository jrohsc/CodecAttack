#!/bin/bash
# Step 4 of run_stage1.sh, re-run standalone after the sentencepiece fix.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
R=./results
for F in "$R"/*.jsonl; do
  case "$F" in *_scored.jsonl) continue ;; esac
  [ -f "${F%.jsonl}_scored.jsonl" ] && { echo "skip $F (scored)"; continue; }
  echo "=== [score] $F $(date -Is) ==="
  "$PY" -u grm_delivery.py score --responses "$F" --batch-size 8 || echo "FAILED $F"
done
echo "SCOREDONE $(date -Is)"
