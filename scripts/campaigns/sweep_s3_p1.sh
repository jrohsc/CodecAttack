#!/bin/bash
# P1 sweep: fresh attacks + inline channel eval for remaining 03a/03b bundles.
#   order = smallest cost first (03b AF3 eps1.0 finish, then 03b Omni, then 03a AF3, then 03a Omni)
# run_s3_attack.py is now skip-if-exists (preemption-safe).
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

ts() { date '+%Y-%m-%d %H:%M:%S'; }

launch() {
  local tag="$1"; local py="$2"; local sub="$3"; local model="$4"; local eps="$5"
  local log="scripts/p1_${tag}.log"
  echo "[p1 $(ts)] START ${tag}"
  env CODEC_ATTACK_PY="$py" "$CA" -u scripts/run_s3_attack.py \
      --sub-scenario "$sub" --target-model "$model" --eps "$eps" \
      > "$log" 2>&1
  echo "[p1 $(ts)] DONE  ${tag}"
}

# 1) finish 03b AF3 eps_1.0 (7 pairs remaining + full channel eval)
launch 03b_af3_eps1.0_finish $AF3  03b audio_flamingo 1.0

# 2) 03b Omni (fresh 20 pairs x 2 eps)
launch 03b_omni_eps0.5      $OMNI 03b qwen25_omni    0.5
launch 03b_omni_eps1.0      $OMNI 03b qwen25_omni    1.0

# 3) 03a AF3 (fresh 40 pairs x 2 eps)
launch 03a_af3_eps0.5       $AF3  03a audio_flamingo 0.5
launch 03a_af3_eps1.0       $AF3  03a audio_flamingo 1.0

# 4) 03a Omni (fresh 40 pairs x 2 eps)
launch 03a_omni_eps0.5      $OMNI 03a qwen25_omni    0.5
launch 03a_omni_eps1.0      $OMNI 03a qwen25_omni    1.0

echo "[p1 $(ts)] P1 SWEEP COMPLETE"
