#!/bin/bash
#SBATCH --job-name=s1e3eps
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=${REPO_ROOT}/logs/s1e3eps_%a_%A.out
#SBATCH --error=${REPO_ROOT}/logs/s1e3eps_%a_%A.err
#
# S1 ensemble-3 eps ladder PILOT (16 pairs, opus-EoT, BANK prompt, seed 1, 1000 steps).
# Everything matches the published S1 ens3 opus_eot cell except eps, so the existing
# eps=1.0 bundle is the anchor (subset it to the same first 16 pairs).
#
# Why: the 3-model ensemble reaches ~100% on jazz_1 at eps=10 (SNR +2.5 dB) but only
# 58% lenient on S1 at eps=1.0 (SNR +6.3 dB). eps=2.0 on S1 measures +2.18 dB, i.e.
# the SNR-matched cell. Ladder 1.5 / 2.0 / 3.0 brackets it; 3.0 is the project cap.
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python

EPSL=(1.5 1.5 2.0 2.0 3.0 3.0)
SHARDL=(0 1 0 1 0 1)
EPS=${EPSL[$SLURM_ARRAY_TASK_ID]}
SHARD=${SHARDL[$SLURM_ARRAY_TASK_ID]}

echo "=== [$(date -Is)] S1 ens3 eps-ladder eps=$EPS shard=$SHARD/2 host=$(hostname) job=$SLURM_JOB_ID"
$PY 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
    --eps "$EPS" --alpha 0.2 --steps 1000 --seed 1 \
    --train-codec opus \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --max-pairs 16 --num-shards 2 --shard-id "$SHARD" \
    --best-after-warmup \
    --out-base results_s1_ens3_eps_ladder/01_finance_voice_agent \
    --bundle-name opus_eot
rc=$?
B=0_all_combined/results_s1_ens3_eps_ladder/01_finance_voice_agent/opus_eot/eps_${EPS}_seed1
echo "=== [$(date -Is)] rc=$rc wavs=$(ls $B/audio/*_attacked.wav 2>/dev/null | wc -l)/16"
