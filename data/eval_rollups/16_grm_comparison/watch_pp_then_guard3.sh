#!/bin/bash
# Dependency-free supervisor: no GPU held while waiting. Submits the Guard-3
# pass over results_pp/ only once the paper-prompt arm has actually finished,
# and guards on the job name so a second session cannot double-submit it.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
for i in $(seq 1 480); do
  if grep -q PAPERPROMPTDONE logs/paper_prompt.log 2>/dev/null; then
    if squeue -u "$USER" -h -o "%j" | grep -qx "grm-guard3-pp"; then
      echo "[watch] already queued, nothing to do"; exit 0
    fi
    echo "[watch] paper-prompt arm done $(date -Is), submitting guard3-pp"
    sbatch submit_guard3_pp.sh
    exit 0
  fi
  sleep 60
done
echo "[watch] timed out waiting for PAPERPROMPTDONE $(date -Is)"; exit 1
