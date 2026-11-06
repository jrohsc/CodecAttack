#!/bin/bash
# EoT{64,128} shard 0 -- run on the LOCAL idle A100 (umd-cscdr-gpu002) instead of
# SLURM, because 64571396_0 was stuck in launch_failed_requeued_held and was
# blocking the whole eot eval. Cancelled that task first, so nothing else is
# crafting these pairs (shard 0 = even indices; SLURM 64571396_1 has the odd ones).
# Flags are copied verbatim from sbatch_s1_ens3_eot64128_craft.sh with SHARD=0.
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python

echo "=== [$(date -Is)] LOCAL EoT{64,128} eps=1.0 shard=0/2 host=$(hostname) pid=$$"
$PY 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
    --eps 1.0 --alpha 0.2 --steps 1000 --seed 1 \
    --train-codec opus --train-bitrates 64,128 \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --max-pairs 16 --num-shards 2 --shard-id 0 \
    --best-after-warmup \
    --out-base results_s1_ens3_eot64128/01_finance_voice_agent \
    --bundle-name opus_eot
rc=$?
B=0_all_combined/results_s1_ens3_eot64128/01_finance_voice_agent/opus_eot/eps_1_seed1
echo "=== [$(date -Is)] rc=$rc wavs=$(ls $B/audio/*_attacked.wav 2>/dev/null | wc -l)/16"
