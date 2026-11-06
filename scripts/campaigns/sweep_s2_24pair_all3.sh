#!/usr/bin/env bash
# Full S2 sweep: qwen2_audio, audio_flamingo, qwen25_omni × eps {0.5, 1.0}.
# Resumes by skipping carriers whose adv wav already exists (logic in run_s2_attack.py).
#
# Env mapping:
#   qwen2_audio     -> codec-attack  (transformers 4.57)
#   audio_flamingo  -> flamingo3     (transformers >= 5.0)
#   qwen25_omni     -> codec-attack  (7B fits on 80 GB GPU; prior 40 GB OOM no longer applies)
set -u
cd ${REPO_ROOT}/0_all_combined

CA_BIN=${WORK_ROOT}/miniconda3/envs/codec-attack/bin
AF3_BIN=${WORK_ROOT}/miniconda3/envs/flamingo3/bin
DRIVER="$CA_BIN/python"

LOG_ROOT=scripts
STAMP=$(date +%Y%m%d_%H%M%S)

export PYTORCH_ALLOC_CONF=expandable_segments:True

run_tier() {
  local model="$1"; local eps="$2"; local py_bin="$3"
  local env_bin; env_bin="$(dirname "$py_bin")"
  local log="$LOG_ROOT/sweep_s2_all3_${model}_eps${eps}_${STAMP}.log"
  echo "=== [$(date)] model=$model eps=$eps env=$env_bin log=$log ==="
  PATH="$env_bin:$PATH" CODEC_ATTACK_PY="$py_bin" \
    "$DRIVER" -u scripts/run_s2_attack.py \
      --target-model "$model" --eps "$eps" \
      >"$log" 2>&1
  local rc=$?
  echo "=== [$(date)] model=$model eps=$eps  exit=$rc ==="
}

for eps in 0.5 1.0; do
  run_tier qwen2_audio    "$eps" "$CA_BIN/python"
  run_tier audio_flamingo "$eps" "$AF3_BIN/python"
  run_tier qwen25_omni    "$eps" "$CA_BIN/python"
done

echo "=== Sweep complete $(date) ==="
