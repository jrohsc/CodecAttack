#!/bin/bash
# Resume and validate local evaluations, then score saved two-model seed bundles.
set -euo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
for PID in ${WAIT_PIDS:-}; do
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
done
echo "$(date -u +%FT%TZ) checking Phi-4 224-target evaluation"
bash 0_all_combined/n50_latent_vs_waveform/run_local_phi4_224_eval.sh
echo "$(date -u +%FT%TZ) checking Kimi 224-target evaluation"
bash 0_all_combined/n50_latent_vs_waveform/run_local_kimi_224_eval.sh
echo "$(date -u +%FT%TZ) evaluating saved two-model seeds"
bash 0_all_combined/n50_latent_vs_waveform/run_local_ens2_224_eval.sh
${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python \
  0_all_combined/n50_latent_vs_waveform/aggregate_ens2_224_seeds.py
echo "$(date -u +%FT%TZ) all local 224-target evaluations complete"
