#!/usr/bin/env bash
# Resume every partial codec-robust S3 bundle + run codec eval right after.
# Idempotent: run_s3_attack.py skips carrier pairs whose *_attacked.wav exists.
# eval_mp3_aac_bundle.py skips clips whose *_codecs.json exists.
#
# Grouped by env to avoid model reloads. Smallest delta first inside a group.
#
# Master log: 0_all_combined/scripts/chain_resume_partials_2026_04_22.log
# Stage logs: 0_all_combined/scripts/logs/<stage>.log

set -u
cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S3A_ROOT=0_all_combined/results_codec_robust/03a_ai_detection_bypass
S3B_ROOT=0_all_combined/results_codec_robust/03b_copyright_bypass
EVAL=0_all_combined/scripts/eval_mp3_aac_bundle.py
RUN_S3=0_all_combined/scripts/run_s3_attack.py
LOGDIR=0_all_combined/scripts/logs
MASTER=0_all_combined/scripts/chain_resume_partials_2026_04_22.log
mkdir -p "$LOGDIR"

S3A_PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"
S3B_PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"

stage() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo "=== [$(date '+%F %T')] STAGE: $name ===" | tee -a "$MASTER"
  echo "  log: $log" | tee -a "$MASTER"
  if "$@" > "$log" 2>&1; then
    echo "  OK" | tee -a "$MASTER"
    return 0
  else
    local rc=$?
    echo "  FAILED rc=$rc -- continuing chain (see $log)" | tee -a "$MASTER"
    return $rc
  fi
}

echo "[$(date '+%F %T')] chain start (pid $$)" | tee -a "$MASTER"

################################################################################
# Pass A: qwen-omni env (headline eps=1.0 row)
################################################################################

stage attack_s3a_qwen25_omni_eps1.0_mb \
  env CODEC_ATTACK_PY="$OMNI" \
  "$OMNI" -u "$RUN_S3" \
    --sub-scenario 03a --target-model qwen25_omni --eps 1.0 \
    --channel-mode multi_bitrate --skip-eval

stage eval_s3a_qwen25_omni_eps1.0_mb \
  "$OMNI" -u "$EVAL" \
  --bundle-dir "$S3A_ROOT/qwen25_omni/eps_1.0_multibitrate" \
  --target-model qwen25_omni \
  --prompt-text "$S3A_PROMPT"

################################################################################
# Pass B: flamingo3 env (headline eps=1.0 row)
################################################################################

stage attack_s3b_audio_flamingo_eps1.0_mb \
  env CODEC_ATTACK_PY="$AF3" \
  "$AF3" -u "$RUN_S3" \
    --sub-scenario 03b --target-model audio_flamingo --eps 1.0 \
    --channel-mode multi_bitrate --skip-eval

stage eval_s3b_audio_flamingo_eps1.0_mb \
  "$AF3" -u "$EVAL" \
  --bundle-dir "$S3B_ROOT/audio_flamingo/eps_1.0_multibitrate" \
  --target-model audio_flamingo \
  --prompt-text "$S3B_PROMPT"

stage attack_s3a_audio_flamingo_eps1.0_mb \
  env CODEC_ATTACK_PY="$AF3" \
  "$AF3" -u "$RUN_S3" \
    --sub-scenario 03a --target-model audio_flamingo --eps 1.0 \
    --channel-mode multi_bitrate --skip-eval

stage eval_s3a_audio_flamingo_eps1.0_mb \
  "$AF3" -u "$EVAL" \
  --bundle-dir "$S3A_ROOT/audio_flamingo/eps_1.0_multibitrate" \
  --target-model audio_flamingo \
  --prompt-text "$S3A_PROMPT"

################################################################################
# Pass C: codec-attack env (secondary eps rows for qwen2_audio)
################################################################################

stage attack_s3b_qwen2_audio_eps0.5_mb \
  env CODEC_ATTACK_PY="$CA" \
  "$CA" -u "$RUN_S3" \
    --sub-scenario 03b --target-model qwen2_audio --eps 0.5 \
    --channel-mode multi_bitrate --skip-eval

stage eval_s3b_qwen2_audio_eps0.5_mb \
  "$CA" -u "$EVAL" \
  --bundle-dir "$S3B_ROOT/qwen2_audio/eps_0.5_multibitrate" \
  --target-model qwen2_audio \
  --prompt-text "$S3B_PROMPT"

stage attack_s3a_qwen2_audio_eps1.5_mb \
  env CODEC_ATTACK_PY="$CA" \
  "$CA" -u "$RUN_S3" \
    --sub-scenario 03a --target-model qwen2_audio --eps 1.5 \
    --channel-mode multi_bitrate --skip-eval

stage eval_s3a_qwen2_audio_eps1.5_mb \
  "$CA" -u "$EVAL" \
  --bundle-dir "$S3A_ROOT/qwen2_audio/eps_1.5_multibitrate" \
  --target-model qwen2_audio \
  --prompt-text "$S3A_PROMPT"

stage attack_s3a_qwen2_audio_eps0.5_mb \
  env CODEC_ATTACK_PY="$CA" \
  "$CA" -u "$RUN_S3" \
    --sub-scenario 03a --target-model qwen2_audio --eps 0.5 \
    --channel-mode multi_bitrate --skip-eval

stage eval_s3a_qwen2_audio_eps0.5_mb \
  "$CA" -u "$EVAL" \
  --bundle-dir "$S3A_ROOT/qwen2_audio/eps_0.5_multibitrate" \
  --target-model qwen2_audio \
  --prompt-text "$S3A_PROMPT"

echo "[$(date '+%F %T')] chain done" | tee -a "$MASTER"
