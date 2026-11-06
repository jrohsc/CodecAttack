#!/bin/bash
# S2 attack+eval sweep extended from 6 -> 25 carriers (2026-04-17).
# run_s2_attack.py's resume guard skips already-attacked pairs, so only the
# 19 new carriers get PGD-optimized; the channel eval pass then re-routes
# all 25 adv wavs through the 8 S2 channels for each (model, eps).
#
# Estimated wall time (single A100):
#   qwen2 eps{0.5,1.0}  : 19 new x ~1.5 min x 2 = ~57 min
#   AF3   eps{0.5,1.0}  : 19 new x ~4.8 min x 2 = ~182 min
#   Omni  eps{0.5,1.0}  : 19 new x ~5.7 min x 2 = ~216 min
#   Total sequential    : ~7.6 h (plus ~20 min of bench eval passes)
set -euo pipefail
cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {
  local tag="$1"; shift
  local log="scripts/s2_25pair_${tag}.log"
  echo "[sweep $(ts)] START ${tag}"
  "$@" > "${log}" 2>&1
  echo "[sweep $(ts)] DONE  ${tag}"
}

# qwen2 first (fastest)
run qwen2_eps0.5 env CODEC_ATTACK_PY=$CA $CA -u scripts/run_s2_attack.py --target-model qwen2_audio  --eps 0.5
run qwen2_eps1.0 env CODEC_ATTACK_PY=$CA $CA -u scripts/run_s2_attack.py --target-model qwen2_audio  --eps 1.0

# AF3
run af3_eps0.5   env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s2_attack.py --target-model audio_flamingo --eps 0.5
run af3_eps1.0   env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s2_attack.py --target-model audio_flamingo --eps 1.0

# Omni (slowest)
run omni_eps0.5  env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s2_attack.py --target-model qwen25_omni --eps 0.5
run omni_eps1.0  env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s2_attack.py --target-model qwen25_omni --eps 1.0

echo "[sweep $(ts)] SWEEP COMPLETE — refreshing rollups"
"$CA" scripts/rollup_results.py
