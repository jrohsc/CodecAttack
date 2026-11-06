#!/bin/bash
# Guard-3 over results_pp/, run on the local A100 instead of SLURM.
# Waits for the paper-prompt arm to release the GPU first -- Qwen2-Audio (~16GB)
# and Guard-3 (~17GB) both resident on one 40GB card would OOM the generation.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 480); do
  grep -q PAPERPROMPTDONE logs/paper_prompt.log 2>/dev/null && break
  sleep 60
done
if ! grep -q PAPERPROMPTDONE logs/paper_prompt.log 2>/dev/null; then
  echo "[guard3-pp] timed out waiting for the paper-prompt arm $(date -Is)"; exit 1
fi

# GPU must actually be free, not just the marker written
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 2000 ] && break
  sleep 10
done
echo "[guard3-pp] starting $(date -Is), GPU used=${USED}MiB"
${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python -u score_guard3.py results_pp
echo "GUARD3PPDONE $(date -Is)"
