#!/bin/bash
# Sequential sweep: 03a/03b × {audio_flamingo, qwen25_omni} × eps {0.5, 1.0}.
# qwen2_audio bundles are already complete. Runs smaller bundles (03b) first.
# Attack scripts skip pairs whose adv wav already exists — preemption-safe.
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {
  local tag="$1"; shift
  local log="scripts/run_${tag}.log"
  echo "[sweep $(ts)] START ${tag}"
  "$@" > "${log}" 2>&1
  echo "[sweep $(ts)] DONE  ${tag}"
}

# --- 03b (smaller: 20 pairs per bundle) ---

# 03b × AF3 × 0.5
run s3b_af3_eps0.5 env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s3_attack.py --sub-scenario 03b --target-model audio_flamingo --eps 0.5

# 03b × AF3 × 1.0
run s3b_af3_eps1.0 env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s3_attack.py --sub-scenario 03b --target-model audio_flamingo --eps 1.0

# 03b × Omni × 0.5
run s3b_omni_eps0.5 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py --sub-scenario 03b --target-model qwen25_omni --eps 0.5

# 03b × Omni × 1.0
run s3b_omni_eps1.0 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py --sub-scenario 03b --target-model qwen25_omni --eps 1.0

# --- 03a (larger: 40 pairs per bundle) ---

# 03a × AF3 × 0.5
run s3a_af3_eps0.5 env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s3_attack.py --sub-scenario 03a --target-model audio_flamingo --eps 0.5

# 03a × AF3 × 1.0
run s3a_af3_eps1.0 env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s3_attack.py --sub-scenario 03a --target-model audio_flamingo --eps 1.0

# 03a × Omni × 0.5
run s3a_omni_eps0.5 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py --sub-scenario 03a --target-model qwen25_omni --eps 0.5

# 03a × Omni × 1.0
run s3a_omni_eps1.0 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s3_attack.py --sub-scenario 03a --target-model qwen25_omni --eps 1.0

echo "[sweep $(ts)] SWEEP COMPLETE"
