#!/bin/bash
# Run one Qwen2.5-Omni S1 codec-EoT seed on the GPU allocated to this shell.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_local_omni_codec_eot.sh <aac|mp3|dac> <seed>
set -uo pipefail

REPO=${REPO_ROOT}
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
cd "$REPO"

TRAIN_CODEC="${1:?usage: run_local_omni_codec_eot.sh <aac|mp3|dac> <seed>}"
SEED="${2:?usage: run_local_omni_codec_eot.sh <aac|mp3|dac> <seed>}"
case "$TRAIN_CODEC" in
  aac) RESULTS_DIR=results_codec_robust_aac ;;
  mp3) RESULTS_DIR=results_codec_robust_mp3 ;;
  dac) RESULTS_DIR=results_codec_robust_dac ;;
  *) echo "ERROR: TRAIN_CODEC must be aac|mp3|dac"; exit 1 ;;
esac

export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BUNDLE="0_all_combined/${RESULTS_DIR}/01_finance_voice_agent/qwen25_omni/eps_1.0_seed${SEED}"

echo "=== LOCAL OMNI codec-EoT: codec=$TRAIN_CODEC seed=$SEED gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
echo "=== bundle: $BUNDLE ==="
"$OMNI" -c 'import torch; print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))'

"$OMNI" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
  --target-model qwen25_omni --eps 1.0 --seed "$SEED" --rand-init \
  --train-codec "$TRAIN_CODEC"

"$OMNI" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
  --bundle-dir "$BUNDLE" --target-model qwen25_omni

"$OMNI" -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
  --bundle-dir "$BUNDLE" --target-model qwen25_omni --codecs dac,encodec

echo "=== DONE codec=$TRAIN_CODEC seed=$SEED bundle=$BUNDLE ==="
