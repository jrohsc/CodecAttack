#!/bin/bash
#SBATCH --job-name=s1e3eps50
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --array=0-5
#SBATCH --output=${REPO_ROOT}/logs/s1e3eps50_%a_%A.out
#SBATCH --error=${REPO_ROOT}/logs/s1e3eps50_%a_%A.err
#
# S1 ens3 eps ladder -- BACKFILL the 16-pair pilot to the full 50-pair S1 cell.
# Identical to sbatch_s1_ens3_eps_ladder_craft.sh except --max-pairs is dropped.
#
# Safe to resume: run_s1_ensemble_bundleA.py SKIPs a pair whose _attacked.wav
# already exists (and `continue`s before the jsonl write, so no duplicate
# records). pairs[:max_pairs] is a prefix, so the pilot's 16 are pairs 0-15 and
# this only crafts 16-49 -- 34 new pairs per eps, ~12.0 min each.
#
# Shard i takes pairs where idx % 2 == i, same rule as the pilot, so each shard
# re-walks its own 8 done pairs (instant skip) and crafts ~17 new ones.
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python

EPSL=(1.5 1.5 2.0 2.0 3.0 3.0)
SHARDL=(0 1 0 1 0 1)
EPS=${EPSL[$SLURM_ARRAY_TASK_ID]}
SHARD=${SHARDL[$SLURM_ARRAY_TASK_ID]}

echo "=== [$(date -Is)] S1 ens3 eps-ladder FULL50 eps=$EPS shard=$SHARD/2 host=$(hostname) job=$SLURM_JOB_ID"
$PY 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
    --eps "$EPS" --alpha 0.2 --steps 1000 --seed 1 \
    --train-codec opus \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --num-shards 2 --shard-id "$SHARD" \
    --best-after-warmup \
    --out-base results_s1_ens3_eps_ladder/01_finance_voice_agent \
    --bundle-name opus_eot
rc=$?
B=0_all_combined/results_s1_ens3_eps_ladder/01_finance_voice_agent/opus_eot/eps_${EPS}_seed1
echo "=== [$(date -Is)] rc=$rc wavs=$(ls $B/audio/*_attacked.wav 2>/dev/null | wc -l)/50"
