#!/bin/bash
# Omni-only S3 sweep. Carrier duration capped to 15s for Omni (OOM at 25s on L40S).
# Patched in scripts/run_s3_attack.py: if target_model == qwen25_omni, dur = min(dur, 15.0).
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {
  local tag="$1"; shift
  local log="scripts/run_${tag}.log"
  echo "[sweep $(ts)] START ${tag}"
  "$@" > "${log}" 2>&1 || echo "[sweep $(ts)] ${tag} EXIT=$?"
  echo "[sweep $(ts)] DONE  ${tag}"
}

# Order: smallest first (03b 20 pairs), then 03a (40 pairs).
run s3b_omni_eps0.5_15s env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03b --target-model qwen25_omni --eps 0.5
run s3b_omni_eps1.0_15s env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03b --target-model qwen25_omni --eps 1.0
run s3a_omni_eps0.5_15s env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model qwen25_omni --eps 0.5
run s3a_omni_eps1.0_15s env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model qwen25_omni --eps 1.0

echo "[sweep $(ts)] OMNI S3 COMPLETE"
