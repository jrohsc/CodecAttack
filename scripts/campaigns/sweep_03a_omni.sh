#!/bin/bash
# Fill the only missing 03a bundles: qwen25_omni × {0.5, 1.0}.
# qwen2_audio and audio_flamingo are already complete at both eps.
# run_s3_attack.py is skip-if-exists and Omni branch caps carrier to 15s.
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {
  local tag="$1"; shift
  local log="scripts/run_${tag}.log"
  echo "[03a-omni $(ts)] START ${tag}" | tee -a scripts/sweep_03a_omni.log
  "$@" > "${log}" 2>&1 || echo "[03a-omni $(ts)] ${tag} EXIT=$?" | tee -a scripts/sweep_03a_omni.log
  echo "[03a-omni $(ts)] DONE  ${tag}" | tee -a scripts/sweep_03a_omni.log
}

run s3a_omni_eps0.5 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model qwen25_omni --eps 0.5

run s3a_omni_eps1.0 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model qwen25_omni --eps 1.0

echo "[03a-omni $(ts)] ALL 03a OMNI BUNDLES COMPLETE" | tee -a scripts/sweep_03a_omni.log

# Refresh rollups so parent JSON picks up both new bundles + any lingering stale state.
"$CA" -u scripts/rollup_results.py >> scripts/sweep_03a_omni.log 2>&1 || true
echo "[03a-omni $(ts)] ROLLUP REFRESHED" | tee -a scripts/sweep_03a_omni.log
