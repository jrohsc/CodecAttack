#!/usr/bin/env bash
# Serial codec-sidecar eval across the four complete bundles still missing *_codecs.json.
# 1) S1 qwen2_audio eps=1.5           (codec-attack env)
# 2) S2 qwen2_audio eps=1.0_multibitrate (codec-attack env)
# 3) S2 qwen25_omni eps=1.0_multibitrate (qwen-omni env)
# 4) S3b qwen25_omni eps=1.0_multibitrate (qwen-omni env)
set -u
cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
EVAL=0_all_combined/scripts/eval_mp3_aac_bundle.py
LOGDIR=0_all_combined/scripts/logs
MASTER=0_all_combined/scripts/chain_codec_eval_2026_04_22.log
mkdir -p "$LOGDIR"

S1_PROMPT="$(cat 0_all_combined/scenarios/01_finance_voice_agent/system_prompt.txt)"
S2_PROMPT="$(cat 0_all_combined/scenarios/02_interview_screening/system_prompt.txt)"
S3B_PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"

stage() {
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] STAGE: $name ===" | tee -a "$MASTER"
  local log="$LOGDIR/${name}.log"
  echo "  log: $log" | tee -a "$MASTER"
  if "$@" > "$log" 2>&1; then
    echo "  OK" | tee -a "$MASTER"
  else
    local rc=$?
    echo "  FAILED rc=$rc -- continuing chain (see $log)" | tee -a "$MASTER"
  fi
}

echo "[$(date '+%F %T')] chain start (pid $$)" | tee -a "$MASTER"

stage codec_eval_s1_qwen2_audio_eps1.5 \
  "$CA" -u "$EVAL" \
  --bundle-dir 0_all_combined/results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.5 \
  --target-model qwen2_audio \
  --prompt-text "$S1_PROMPT"

stage codec_eval_s2_qwen2_audio_eps1.0_multibitrate \
  "$CA" -u "$EVAL" \
  --bundle-dir 0_all_combined/results_codec_robust/02_interview_screening/qwen2_audio/eps_1.0_multibitrate \
  --target-model qwen2_audio \
  --prompt-text "$S2_PROMPT"

stage codec_eval_s2_qwen25_omni_eps1.0_multibitrate \
  "$OMNI" -u "$EVAL" \
  --bundle-dir 0_all_combined/results_codec_robust/02_interview_screening/qwen25_omni/eps_1.0_multibitrate \
  --target-model qwen25_omni \
  --prompt-text "$S2_PROMPT"

stage codec_eval_s3b_qwen25_omni_eps1.0_multibitrate \
  "$OMNI" -u "$EVAL" \
  --bundle-dir 0_all_combined/results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate \
  --target-model qwen25_omni \
  --prompt-text "$S3B_PROMPT"

echo "[$(date '+%F %T')] chain done" | tee -a "$MASTER"
