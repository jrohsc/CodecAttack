#!/bin/bash
# S1 epsilon=10 Opus-EOT source-pool scaling: N=1 or N=2.
# Required exports: POOL_LABEL={n1_qwen2,n2_qwen2_omni}, SHARD_ID={0,1}.
#SBATCH --job-name=s1pool10
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
POOL_LABEL="${POOL_LABEL:?set POOL_LABEL}"
SHARD_ID="${SHARD_ID:?set SHARD_ID to 0 or 1}"
case "$POOL_LABEL" in
  n1_qwen2)       MODELS=qwen2_audio ;;
  n2_qwen2_omni)  MODELS=qwen2_audio,qwen25_omni ;;
  *) echo "unsupported POOL_LABEL=$POOL_LABEL" >&2; exit 2 ;;
esac
NUM_SHARDS=2
SEED=1

REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUTBASE=results_s1_opuseot_eps10_pool_scaling/01_finance_voice_agent
BUNDLE="0_all_combined/$OUTBASE/$POOL_LABEL/eps_10_seed${SEED}"
LOG="$BUNDLE.craft_shard${SHARD_ID}of${NUM_SHARDS}.log"

JOBTMP="$REPO/tmp_s1pool10/${POOL_LABEL}_s${SEED}_sh${SHARD_ID}_${SLURM_JOB_ID}"
mkdir -p "$JOBTMP" "$(dirname "$BUNDLE")" logs
export TMPDIR="$JOBTMP" TORCHINDUCTOR_CACHE_DIR="$JOBTMP/inductor"
export ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date -u +%FT%TZ)] S1 pool=$POOL_LABEL models=$MODELS Opus-EOT eps=10 shard=$SHARD_ID/$NUM_SHARDS host=$(hostname) job=$SLURM_JOB_ID"
"$CA" -u 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
  --ensemble-models "$MODELS" --train-codec opus \
  --eps 10.0 --steps 1000 --seed "$SEED" --rand-init --best-after-warmup \
  --out-base "$OUTBASE" --bundle-name "$POOL_LABEL" \
  --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
  2>&1 | grep -v "WARNING:root:System prompt modified" | tee -a "$LOG"
RC=${PIPESTATUS[0]}
(( RC == 0 )) || { echo "ERROR: craft exited $RC" >&2; exit "$RC"; }
grep -q "subprocess gradient requests FAILED\|failed to start" "$LOG" && {
  echo "ERROR: source-model worker failure in shard log" >&2; exit 1;
}
N=$(find "$BUNDLE/audio" -maxdepth 1 -name '*_summary.json' 2>/dev/null | wc -l)
echo "=== shard complete; bundle sidecars now $N/50"
