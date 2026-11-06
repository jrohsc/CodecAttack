#!/bin/bash
# Resume remaining S3 sweep bundles after 2026-04-17 interruption.
# Order: finish 03a AF3 eps=1.0 first (resume), then try Omni with
# expandable_segments env to work around 25s carrier OOM on L40S.
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
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

# 1. Finish 03a AF3 eps=1.0 (6/40 done; resume guard skips existing)
run s3a_af3_eps1.0_resume env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model audio_flamingo --eps 1.0

# 2. 03b Omni eps=0.5 (was OOM — retry with expandable_segments)
run s3b_omni_eps0.5_retry env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03b --target-model qwen25_omni --eps 0.5

# 3. 03b Omni eps=1.0
run s3b_omni_eps1.0_retry env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03b --target-model qwen25_omni --eps 1.0

# 4. 03a Omni eps=0.5
run s3a_omni_eps0.5_retry env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model qwen25_omni --eps 0.5

# 5. 03a Omni eps=1.0
run s3a_omni_eps1.0_retry env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py \
  --sub-scenario 03a --target-model qwen25_omni --eps 1.0

echo "[sweep $(ts)] ALL REMAINING BUNDLES COMPLETE"
