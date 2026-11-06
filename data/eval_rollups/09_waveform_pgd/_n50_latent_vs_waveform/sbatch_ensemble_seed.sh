#!/bin/bash
# ============================================================================
# SLURM job: ENSEMBLE-VICTIM seed sweep (AC Q1 error bars).
#
# Same configuration as the original `ens_transfer` run (62142472) -- 2 victims
# {qwen2_audio + qwen25_omni}, audio_flamingo held out, music carriers,
# transcribe mode, eps=10.0 -- with ONE change: --rand-init --seed N.
#
# WHY THIS EXISTS: ensemble_attack.py initialised delta to ZEROS and had no
# seeding, so the attack was fully deterministic. Re-running it under different
# SEED values produced byte-identical bundles; a "3-seed" table built that way
# would have been one run reported three times. --rand-init (add-only, OFF by
# default so the original bundle stays reproducible) makes the sweep real.
#
# Each seed writes its own bundle:
#   results/benchmark_ensemble_transcribe_eps10.0_jazz_1_seed<N>/
# so the original unseeded bundle is never overwritten or resumed into.
#
# Needs an 80GB A100: 2x7B weights + backward OOM'd repeatedly on 40GB-class
# cards (incl. the local L40S at 46GB). --constraint=a100-80g is REQUIRED.
# For a 40GB card instead, set ENSEMBLE_GRAD_CKPT=1 (slower, fits).
#
#   sbatch --job-name=ens_seed1  --export=ALL,SEED=1  sbatch_ensemble_seed.sh
#   sbatch --job-name=ens_seed7  --export=ALL,SEED=7  sbatch_ensemble_seed.sh
#   sbatch --job-name=ens_seed42 --export=ALL,SEED=42 sbatch_ensemble_seed.sh
#
# AFTER each job: score the wavs on held-out + members with
#   0_all_combined/scripts/ensemble_heldout_eval.py --bundle-dir <seed bundle>
# (the attack itself only scores the primary member).
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

SEED="${SEED:?set SEED via --export=ALL,SEED=<n>}"

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

echo "=========================================="
echo "  ENSEMBLE-VICTIM seed sweep -- SEED=$SEED"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
python -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0),"| mem",round(torch.cuda.get_device_properties(0).total_memory/1e9,1),"GB")'
echo "  ensemble = qwen2_audio + qwen25_omni   held-out = audio_flamingo"
echo "=========================================="

python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni \
    --music jazz_1 \
    --rand-init --seed "$SEED"

echo "=== seed $SEED attack done ==="
find results -maxdepth 1 -name "*ensemble*seed${SEED}*" -type d
echo "NEXT: ensemble_heldout_eval.py on the held-out + member models."
echo "=== DONE ==="
