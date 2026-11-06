#!/bin/bash
# Three-member ensemble-victim attack with MP3 EoT.
# Same members, carrier bank, epsilon, and 300-step budget as ens3; only the
# attack-time delivery proxy changes from none to MP3 {32,64,96,128} kbps.
#
# SMOKE=1 runs two commands for 40 steps in a separate bundle.
#   sbatch --export=ALL,SMOKE=1 --job-name=ens3_mp3_smoke sbatch_ensemble3_mp3.sh
#   sbatch --job-name=ens3_mp3 sbatch_ensemble3_mp3.sh
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

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=1

OUT=results/benchmark_ensemble3_mp3eot_transcribe_eps10.0_jazz_1
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
    OUT="${OUT}_SMOKE"
    EXTRA="--commands comm_alert comm_eta --steps 40 --ensemble-warmup 10"
fi

echo "3-model ensemble with MP3 EoT; output=$OUT"
python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --ensemble-channel-mode mp3_multibitrate \
    --music jazz_1 --output-dir "$OUT" $EXTRA

echo "crafted wavs: $(find "$OUT/audio" -name '*.wav' | wc -l)"
if grep -q "subprocess gradient requests FAILED" \
    "0_all_combined/n50_latent_vs_waveform/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"; then
    echo "ERROR: AF3 gradient failures detected; do not report this bundle" >&2
    exit 1
fi
