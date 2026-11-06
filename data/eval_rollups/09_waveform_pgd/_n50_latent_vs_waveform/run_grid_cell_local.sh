#!/usr/bin/env bash
# Run one (MODEL, SCEN, ARM, SEED) cell on the local GPU instead of via SLURM.
#
# Deliberately a thin wrapper around sbatch_grid_cell.sh rather than a reimplem-
# entation: local and cluster runs then execute byte-identical work, including
# stage 3, which the older run_local_seed.sh omits (that is how the first S3b
# waveform rows shipped with empty DAC/EnCodec cells). The only thing this adds
# is stubs for the three SLURM variables the wrapper echoes -- under `set -u`
# those would otherwise abort the script on line 1 of its banner.
#
# Usage:
#   bash run_grid_cell_local.sh audio_flamingo s3a waveform 1 0.06211
#
# ALWAYS scancel the matching queued job first. Both paths write the same bundle
# directory, and a local run racing its own sbatch twin corrupts it.
set -uo pipefail

MODEL="${1:?usage: run_grid_cell_local.sh MODEL SCEN ARM SEED [EPS]}"
SCEN="${2:?}"
ARM="${3:?}"
SEED="${4:?}"
EPS="${5:-}"

HERE=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
cd "$HERE" || exit 1

export MODEL SCEN ARM SEED
[ -n "$EPS" ] && export EPS
export SLURM_JOB_ID="local-$$"
export SLURM_NODELIST="$(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== LOCAL cell $MODEL/$SCEN/$ARM/seed$SEED eps=${EPS:-n/a} on $SLURM_NODELIST ==="
bash sbatch_grid_cell.sh
echo "=== LOCAL cell exit=$? ==="
