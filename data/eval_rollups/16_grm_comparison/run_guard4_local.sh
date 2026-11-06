#!/bin/bash
# Guard-4 over both arms, then backfill the 3 Guard-3 cells that a dead job left
# missing in results_pp (score_guard3.py skips whatever is already scored).
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
$PY -u score_guard4.py results results_pp
$PY -u score_guard3.py results_pp
echo BOTHDONE
