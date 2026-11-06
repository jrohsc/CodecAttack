#!/bin/bash
# Disjoint craft-only shard for the S1 three-model codec-EOT comparison.
# All shards share the audio directory, but own disjoint carrier sets and
# shard-specific JSONL/log files, avoiding concurrent writes to one file.
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
CODEC="${CODEC:?set CODEC=opus or mp3}"
SHARD_ID="${SHARD_ID:?set SHARD_ID}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SEED="${SEED:-1}"
case "$CODEC" in opus|mp3) ;; *) echo "CODEC must be opus or mp3" >&2; exit 2;; esac

REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
MODELS=qwen2_audio,qwen25_omni,audio_flamingo
OUTBASE=results_s1_ens3_codec_compare/01_finance_voice_agent
BUNDLE="0_all_combined/$OUTBASE/${CODEC}_eot/eps_1_seed${SEED}"
LOG="$BUNDLE.craft_shard${SHARD_ID}of${NUM_SHARDS}.log"

JOBTMP="$REPO/tmp_s1ens3/${CODEC}_s${SEED}_sh${SHARD_ID}_${SLURM_JOB_ID:-local}"
mkdir -p "$JOBTMP" "$(dirname "$BUNDLE")" logs
export TMPDIR="$JOBTMP" TORCHINDUCTOR_CACHE_DIR="$JOBTMP/inductor"
export ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [$(date -u +%FT%TZ)] S1 ens3 codec=$CODEC shard=$SHARD_ID/$NUM_SHARDS seed=$SEED host=$(hostname) job=${SLURM_JOB_ID:-local}"
"$CA" -u 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
  --ensemble-models "$MODELS" --train-codec "$CODEC" \
  --eps 1.0 --steps 1000 --seed "$SEED" --rand-init --best-after-warmup \
  --out-base "$OUTBASE" --bundle-name "${CODEC}_eot" \
  --shard-id "$SHARD_ID" --num-shards "$NUM_SHARDS" \
  2>&1 | grep -v "WARNING:root:System prompt modified" | tee -a "$LOG"
RC=${PIPESTATUS[0]}
(( RC == 0 )) || { echo "ERROR: craft exited $RC" >&2; exit "$RC"; }
grep -q "subprocess gradient requests FAILED\|failed to start" "$LOG" && {
  echo "ERROR: AF3 worker failure in shard log" >&2; exit 1;
}
N=$(find "$BUNDLE/audio" -maxdepth 1 -name '*_summary.json' 2>/dev/null | wc -l)
echo "=== shard complete; bundle sidecars now $N/50"
