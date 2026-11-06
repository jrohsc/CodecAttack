#!/bin/bash
# S3b copyright-bypass TRANSFER matrix: 3 attacked models x 3 eval models, on the
# wavs already on disk (no crafting). Music carriers, S3b Content-ID system prompt
# passed explicitly -- cross_model_transfer_eval.py defaults to the BANK prompt,
# which would break optimize-prompt == eval-prompt.
# Diagonal = white-box ceiling, off-diagonal = transfer.
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
ENVROOT=${WORK_ROOT}/miniconda3/envs
R=0_all_combined/results/03b_copyright_bypass
PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"
py_for() { case "$1" in
  qwen2_audio) echo "$ENVROOT/codec-attack/bin/python" ;;
  qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
  audio_flamingo) echo "$ENVROOT/flamingo3/bin/python" ;; esac; }

echo "=== [$(date -Is)] S3b transfer matrix START host=$(hostname)"
for EVAL in qwen2_audio qwen25_omni audio_flamingo; do
  for ATTACKED in qwen2_audio qwen25_omni audio_flamingo; do
    B=$R/$ATTACKED/eps_1.0
    echo "--- [$(date -Is)] attacked=$ATTACKED -> eval=$EVAL  (wavs=$(ls $B/audio/*_attacked.wav 2>/dev/null | wc -l))"
    $(py_for $EVAL) 0_all_combined/scripts/cross_model_transfer_eval.py \
        --bundle-dir "$B" --target-model "$EVAL" \
        --prompt-text "$PROMPT" --opus-bitrates 64,128 --out-tag s3b
  done
done
echo "=== [$(date -Is)] S3b TRANSFER MATRIX DONE"
