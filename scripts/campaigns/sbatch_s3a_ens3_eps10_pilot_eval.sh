#!/bin/bash
# Score one ensemble member or held-out model after both S3a craft shards finish.
# Submit with MODEL set to qwen2_audio, qwen25_omni, audio_flamingo, or
# qwen25_omni_3b and an afterok dependency on the craft array.
#SBATCH --job-name=s3ap10eval
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
MODEL="${MODEL:?set MODEL}"
REPO=${REPO_ROOT}
ENVROOT=${WORK_ROOT}/miniconda3/envs
cd "$REPO"
case "$MODEL" in
  qwen2_audio) PY="$ENVROOT/codec-attack/bin/python"; ROLE=member ;;
  qwen25_omni) PY="$ENVROOT/qwen-omni/bin/python"; ROLE=member ;;
  audio_flamingo) PY="$ENVROOT/flamingo3/bin/python"; ROLE=member ;;
  qwen25_omni_3b) PY="$ENVROOT/qwen-omni/bin/python"; ROLE=heldout ;;
  *) echo "unsupported MODEL=$MODEL" >&2; exit 2 ;;
esac
BUNDLE=0_all_combined/results_s3a_ens3_eps10_pilot/03a_ai_detection_bypass/opus_eot/eps_10_seed1
PROMPT="$(<0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"
N=$(find "$BUNDLE/audio" -maxdepth 1 -name '*_summary.json' 2>/dev/null | wc -l)
[[ "$N" -eq 10 ]] || { echo "expected 10 completed attacks, found $N" >&2; exit 1; }
export HF_HUB_OFFLINE=1
"$PY" -u 0_all_combined/scripts/ensemble_heldout_eval.py \
  --bundle-dir "$BUNDLE" --target-model "$MODEL" \
  --out-name "${MODEL}_${ROLE}_native_prompt" --prompt-text "$PROMPT" \
  --channels opus64,opus128
