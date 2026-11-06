#!/bin/bash
# Local pilot: all three 2-model pairs x Opus-EoT x eps=1.0 x seed 1, 16 commands
# each, crafted and scored back to back on the one local A100.
# DO NOT EDIT WHILE RUNNING.
set -u
N50=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
export CUDA_VISIBLE_DEVICES=0
for p in qa qo oa; do
    echo "##### [$(date -u +%FT%TZ)] PAIR $p start"
    PAIR=$p EPS=1.0 SEED=1 PILOT=1 bash "$N50/sbatch_ens2_pair_opus_eps.sh"
    echo "##### [$(date -u +%FT%TZ)] PAIR $p rc=$?"
done
echo "##### [$(date -u +%FT%TZ)] PILOT DONE"
