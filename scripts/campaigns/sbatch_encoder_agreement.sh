#!/bin/bash
# Encoder-agreement probe: does the direction we optimize in the source model's
# audio tower exist in the victims' towers at all?
#
# Every transfer cell so far measured end-to-end TEXT, which conflates "the
# encoder never moved" with "the encoder moved but the LM didn't follow".
# probe_encoder_agreement.py measures the encoder half directly:
#
#     delta = cos(adv, target) - cos(carrier, target)      inside each own tower
#
# Three bundles, chosen so the probe is falsifiable rather than just suggestive:
#
#   ce_targetinit   the attack that DEMONSTRABLY works -- 50/50 white-box.
#                   If its direction is absent from the victims' towers, that is
#                   the structural answer for why nothing transfers.
#   enc_zeros       pure encoder-cosine, zeros init, no target acoustics anywhere.
#                   Drives source cos to 0.8585 with 0/50 white-box. The clean
#                   test of whether encoder-space alignment is source-specific.
#   enc_targetinit  POSITIVE CONTROL FOR THE PROBE. This bundle carries faint
#                   audible target speech in the carrier's tail (verified
#                   2026-08-13: leaked spans are target suffixes, identical
#                   across all codecs, ~0.003 RMS where the clean carrier is
#                   0.000). Real speech moves EVERY tower, so this must show
#                   delta > 0 on all four models. If it does not, the probe
#                   itself is broken and the other two rows mean nothing.
#
# Deliberately excludes enc_mask_match / encce_mask_match: job 62870856 is
# rewriting those bundles right now and the summaries would be a mixed set.
#
# Usage: sbatch sbatch_encoder_agreement.sh
#   then: conda activate codec-attack && python probe_encoder_agreement.py \
#             --bundle 0_all_combined/results_encoder_transfer/<arm> --summarize
#
#SBATCH --job-name=enc_agree
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -o ${REPO_ROOT}/logs/enc_agree_%A_%a.log
#SBATCH --array=0-11

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

BUNDLES=(ce_targetinit enc_zeros enc_targetinit)
MODELS=(qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo)
ENVS=(  codec-attack qwen-omni  qwen-omni      flamingo3)

i=$SLURM_ARRAY_TASK_ID
ARM=${BUNDLES[$((i / 4))]}
j=$((i % 4))
MODEL=${MODELS[$j]}
ENV=${ENVS[$j]}

ROLE="victim tower"
[[ "$MODEL" == "qwen2_audio" ]] && ROLE="SOURCE tower (must show large delta)"

echo "=== encoder-agreement arm=$ARM model=$MODEL [$ROLE] env=$ENV ==="
conda activate "$ENV"

# n=50 = the whole bundle; this is encoder forwards only, no generation.
python probe_encoder_agreement.py \
    --bundle "0_all_combined/results_encoder_transfer/${ARM}" \
    --model "$MODEL" \
    --n 50

echo "=== done $ARM / $MODEL ==="
