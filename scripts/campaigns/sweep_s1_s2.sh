#!/bin/bash
# Sequential S1+S2 sweep (eps ∈ {0.5, 1.0} only; 0.3 dropped 2026-04-16).
# Remaining bundles as of 2026-04-16 23:30:
#   5. S1 Omni eps=1.0      ~142 min
#   7. S2 AF3  eps=0.5      ~29  min
#   8. S2 AF3  eps=1.0      ~29  min
#  10. S2 Omni eps=0.5      ~34  min
#  11. S2 Omni eps=1.0      ~34  min
# Total ~4.5 h on single L40S. Attack scripts now skip pairs whose
# adv wav already exists on disk, so preemption-safe — just re-run this
# script after a Slurm kill.
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

# 5. S1 Omni eps=1.0
run s1_attack_omni_eps1.0 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s1_attack.py --target-model qwen25_omni --eps 1.0

# 7. S2 AF3 eps=0.5
run s2_attack_af3_eps0.5 env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s2_attack.py --target-model audio_flamingo --eps 0.5
# 8. S2 AF3 eps=1.0
run s2_attack_af3_eps1.0 env CODEC_ATTACK_PY=$AF3 $CA -u scripts/run_s2_attack.py --target-model audio_flamingo --eps 1.0

# 10. S2 Omni eps=0.5
run s2_attack_omni_eps0.5 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s2_attack.py --target-model qwen25_omni --eps 0.5
# 11. S2 Omni eps=1.0
run s2_attack_omni_eps1.0 env CODEC_ATTACK_PY=$OMNI $CA -u scripts/run_s2_attack.py --target-model qwen25_omni --eps 1.0

echo "[sweep $(ts)] SWEEP COMPLETE"
