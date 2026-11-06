#!/bin/bash
# S3a threat-model pilot: 10 carrier/target examples, split across two GPUs.
# The subset covers all 8 native AI-generated music carriers, with a second
# target assignment for the first two carriers.
# Submit with SHARD_ID=0 and SHARD_ID=1.
#SBATCH --job-name=s3ae3p10
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -uo pipefail
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:?set SHARD_ID or use --array=0-1}}"
NUM_SHARDS=2
SEED=1

REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
MODELS=qwen2_audio,qwen25_omni,audio_flamingo
PROMPT="$(<0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"
OUTBASE=results_s3a_ens3_eps10_pilot/03a_ai_detection_bypass
BUNDLE="0_all_combined/$OUTBASE/opus_eot/eps_10_seed${SEED}"
LOG="$BUNDLE.craft_shard${SHARD_ID}of${NUM_SHARDS}.log"
# Stable mapping indices: all eight carriers at target 0, then target 1 for the
# first two carriers. This is ten examples but eight unique native carriers.
PAIR_INDICES=0,1,5,6,10,15,20,25,30,35

JOBTMP="$REPO/tmp_s3aens3/opus_eps10_s${SEED}_sh${SHARD_ID}_${SLURM_JOB_ID}"
mkdir -p "$JOBTMP" "$(dirname "$BUNDLE")" logs
export TMPDIR="$JOBTMP" TORCHINDUCTOR_CACHE_DIR="$JOBTMP/inductor"
export ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

echo "=== [$(date -u +%FT%TZ)] S3a ens3 eps=10 shard=$SHARD_ID/$NUM_SHARDS job=$SLURM_JOB_ID"
"$PY" -u 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
  --carrier-map data/music/03a_ai_detection_bypass/carrier_to_target.json \
  --carrier-dir data/music/ai_generated \
  --scenario-name 03a_ai_detection_bypass --pair-indices "$PAIR_INDICES" \
  --ensemble-models "$MODELS" --prompt-text "$PROMPT" --train-codec opus \
  --eps 10.0 --steps 1000 --seed "$SEED" --rand-init --best-after-warmup \
  --out-base "$OUTBASE" --bundle-name opus_eot \
  --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
  2>&1 | grep -v "WARNING:root:System prompt modified" | tee -a "$LOG"
RC=${PIPESTATUS[0]}
(( RC == 0 )) || { echo "ERROR: craft exited $RC" >&2; exit "$RC"; }
grep -q "subprocess gradient requests FAILED\|failed to start" "$LOG" && {
  echo "ERROR: ensemble worker failure in shard log" >&2; exit 1;
}
N=$(find "$BUNDLE/audio" -maxdepth 1 -name '*_summary.json' 2>/dev/null | wc -l)
echo "=== shard complete; bundle sidecars now $N/10"
