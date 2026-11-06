#!/bin/bash
# Parallel shard of the 224-command three-model MP3-EoT attack.
# Each shard owns a separate bundle, so incremental summary writes cannot race.
#
#   sbatch --export=ALL,SHARD_ID=0,NUM_SHARDS=4 \
#     --job-name=ens3m_s0 sbatch_ensemble3_mp3_shard.sh
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-80g
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
(( ${#ALL_COMMANDS[@]} == 224 )) || {
    echo "ERROR: expected 224 commands, found ${#ALL_COMMANDS[@]}" >&2
    exit 2
}

export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=1

OUT="results/benchmark_ensemble3_mp3eot_transcribe_eps10.0_jazz_1_shard${SHARD_ID}of${NUM_SHARDS}"
echo "MP3-EoT ensemble shard ${SHARD_ID}/${NUM_SHARDS}; commands=${#COMMANDS[@]}; output=$OUT"

python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --ensemble-channel-mode mp3_multibitrate \
    --music jazz_1 --output-dir "$OUT" \
    --commands "${COMMANDS[@]}"

N=$(find "$OUT/audio" -name '*.wav' | wc -l)
echo "crafted wavs: $N/${#COMMANDS[@]}"
(( N == ${#COMMANDS[@]} )) || exit 1
if grep -q "subprocess gradient requests FAILED" \
    "0_all_combined/n50_latent_vs_waveform/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"; then
    echo "ERROR: AF3 gradient failures detected; do not report this shard" >&2
    exit 1
fi
