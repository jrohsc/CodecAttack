#!/bin/bash
# Parallel shard of the 224-command three-model Opus-EoT attack.
# Each shard owns a separate bundle and can resume from per-command records.
#
#   sbatch --export=ALL,SHARD_ID=1,NUM_SHARDS=4 \
#     --job-name=ens3o_s1 sbatch_ensemble3_opus_shard.sh
#SBATCH --partition=gpu
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

set -euo pipefail

SHARD_ID="${SHARD_ID:?set SHARD_ID}"
NUM_SHARDS="${NUM_SHARDS:?set NUM_SHARDS}"
[[ "$SHARD_ID" =~ ^[0-9]+$ && "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || exit 2
(( SHARD_ID < NUM_SHARDS )) || exit 2

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

mapfile -t ALL_COMMANDS < <(python - <<'PY'
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)

COMMANDS=()
for ((i=SHARD_ID; i<${#ALL_COMMANDS[@]}; i+=NUM_SHARDS)); do
    COMMANDS+=("${ALL_COMMANDS[$i]}")
done
(( ${#ALL_COMMANDS[@]} == 224 && ${#COMMANDS[@]} == 56 )) || exit 2

export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="results/benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_shard${SHARD_ID}of${NUM_SHARDS}"
echo "Opus-EoT ensemble shard ${SHARD_ID}/${NUM_SHARDS}; commands=${#COMMANDS[@]}; output=$OUT"

python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --ensemble-channel-mode multi_bitrate \
    --music jazz_1 --output-dir "$OUT" \
    --commands "${COMMANDS[@]}"

N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "crafted wavs: $N/${#COMMANDS[@]}"
(( N == ${#COMMANDS[@]} )) || exit 1
if grep -q "subprocess gradient requests FAILED" \
    "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"; then
    echo "ERROR: AF3 gradient failures detected; do not report this shard" >&2
    exit 1
fi
