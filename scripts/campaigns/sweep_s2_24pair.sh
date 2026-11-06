#!/usr/bin/env bash
# Runs S2 attack sweep across models × eps. Resumes by skipping carriers whose
# adv wav already exists. Uses the correct conda env per model.
#   - qwen2_audio     -> codec-attack  (transformers 4.57)
#   - audio_flamingo  -> flamingo3     (transformers >= 5.0; AF3ForConditionalGeneration)
#   - qwen25_omni     -> SKIPPED (tracked separately — 7B OOMs on 40 GB GPU)
#
# PYTORCH_ALLOC_CONF=expandable_segments:True mitigates memory fragmentation
# that caused qwen2 eps=1.0 OOM on the v2 carriers in the previous run.
set -u
cd ${REPO_ROOT}/0_all_combined

CA_BIN=${WORK_ROOT}/miniconda3/envs/codec-attack/bin
AF3_BIN=${WORK_ROOT}/miniconda3/envs/flamingo3/bin
DRIVER="$CA_BIN/python"   # driver always runs in codec-attack (for rollup.py)

LOG_ROOT=scripts
STAMP=$(date +%Y%m%d_%H%M%S)

export PYTORCH_ALLOC_CONF=expandable_segments:True

run_tier() {
  local model="$1"; local eps="$2"; local py_bin="$3"
  local env_bin; env_bin="$(dirname "$py_bin")"
  local log="$LOG_ROOT/run_s2_attack_${model}_eps${eps}_${STAMP}.log"
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
done

echo "=== Sweep complete $(date) (qwen25_omni deferred — see TODO) ==="
