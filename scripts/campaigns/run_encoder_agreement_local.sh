#!/bin/bash
# Encoder-agreement probe, all 4 models x the CE arms, run locally.
#
# Replaces the failed half of SLURM job 62874687, where only the qwen2_audio
# tasks (0,4,8) completed and all nine Omni / Omni-3B / AF3 tasks died on
# per-model tower calling conventions:
#   * Qwen2.5-Omni wants a 2-D mel plus feature_lens / aftercnn_lens
#     -> TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
#   * AF3 wants input_features_mask
#     -> AttributeError: 'NoneType' object has no attribute 'sum'
# Both are fixed in probe_encoder_agreement.py.
#
# WHY THIS IS THE MEASUREMENT THAT MATTERS
# The end-to-end transfer cells are 0/12426, but a 0 there conflates two very
# different mechanisms:
#     (a) the attack never moved the victim's ENCODER, or
#     (b) it moved the encoder and the victim's LM simply did not follow.
# Only (a) licenses "encoder-space transfer is structurally dead for these
# victims". This probe separates them by measuring, inside each victim's OWN
# tower, delta = cos(adv,target) - cos(carrier,target) over the target's
# content frames.
#
# NOTE: n=2 smoke files were written while validating the fix. This runs at
# n=10 and overwrites them -- do not read encoder_agreement__*.json until this
# finishes.
#
# Usage: nohup bash run_encoder_agreement_local.sh > logs/enc_agree_local.log 2>&1 &

set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
declare -A PY=(
  [qwen2_audio]="$ENVROOT/codec-attack/bin/python"
  [qwen25_omni]="$ENVROOT/qwen-omni/bin/python"
  [qwen25_omni_3b]="$ENVROOT/qwen-omni/bin/python"
  [audio_flamingo]="$ENVROOT/flamingo3/bin/python"
)

# ce_zeros is the PUBLISHED baseline attack (50/50 white-box, 0/50 transfer) --
# the arm whose null most needs explaining. ce_targetinit is its target-init
# variant. The pure-encoder arms are 0% even white-box, so their geometry says
# nothing about why a WORKING attack fails to transfer.
ARMS=(ce_zeros ce_targetinit)
MODELS=(qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo)
N=10

for arm in "${ARMS[@]}"; do
    BUNDLE="0_all_combined/results_encoder_transfer/$arm"
    if [[ ! -d "$BUNDLE/audio" ]]; then
        echo "SKIP $arm: no audio/ dir"; continue
    fi
    for mdl in "${MODELS[@]}"; do
        echo ""
        echo "=== [$(date +%H:%M:%S)] agreement: arm=$arm model=$mdl n=$N ==="
        if ! "${PY[$mdl]}" probe_encoder_agreement.py \
                --bundle "$BUNDLE" --model "$mdl" --n "$N" 2>&1 \
                | grep -av "it/s"; then
            echo "!!! FAILED: $arm / $mdl"
        fi
    done
    echo ""
    echo "=== [$(date +%H:%M:%S)] SUMMARY $arm ==="
    "${PY[qwen2_audio]}" probe_encoder_agreement.py --bundle "$BUNDLE" --summarize
done

echo ""
echo "=== [$(date +%H:%M:%S)] encoder-agreement sweep done ==="
