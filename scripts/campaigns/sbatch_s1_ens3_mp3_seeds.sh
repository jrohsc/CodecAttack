#!/bin/bash
#SBATCH --job-name=s1e3mp3s
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=${REPO_ROOT}/logs/s1e3mp3s_%a_%A.out
#SBATCH --error=${REPO_ROOT}/logs/s1e3mp3s_%a_%A.err
#
# Seeds 7 and 42 for the post-fix S1 ensemble-3 mp3-EoT cell (seed 1 already on disk).
# eps stays 1.0 -- the published S1 operating point, so the ensemble row is directly
# comparable to the single-victim cells beside it. mp3-EoT trains 32/64/96/128, i.e.
# it never had the suspect Opus 16/24 rungs, so this run cannot be invalidated by the
# EoT-width pilot. Full 50 pairs, 1000 steps, bank prompt, --best-after-warmup.
# Resumes: the driver skips any pair whose attacked wav already exists.
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python

SEEDL=(7 7 42 42)
SHARDL=(0 1 0 1)
SEED=${SEEDL[$SLURM_ARRAY_TASK_ID]}
SHARD=${SHARDL[$SLURM_ARRAY_TASK_ID]}

echo "=== [$(date -Is)] S1 ens3 mp3-EoT seed=$SEED shard=$SHARD/2 host=$(hostname) job=$SLURM_JOB_ID"
$PY 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
    --eps 1.0 --alpha 0.2 --steps 1000 --seed "$SEED" \
    --train-codec mp3 \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --num-shards 2 --shard-id "$SHARD" \
    --best-after-warmup \
    --out-base results_s1_ens3_codec_compare/01_finance_voice_agent \
    --bundle-name mp3_eot
rc=$?
B=0_all_combined/results_s1_ens3_codec_compare/01_finance_voice_agent/mp3_eot/eps_1_seed$SEED
echo "=== [$(date -Is)] rc=$rc wavs=$(ls $B/audio/*_attacked.wav 2>/dev/null | wc -l)/50"
