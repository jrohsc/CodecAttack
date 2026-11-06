#!/bin/bash
# Guard-3 over whatever results_pp/ cells are COMPLETE right now, alongside the
# running transfer generation. Omni-3B holds ~12GB of the 40GB card and Guard-3
# needs ~17GB, so this fits; HarmBench's 13B (~26GB) would not. Incomplete cells
# are skipped by the 104-line guard and picked up by the chained full pass.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python -u score_guard3.py results_pp
echo "GUARD3PPEARLYDONE $(date -Is)"
