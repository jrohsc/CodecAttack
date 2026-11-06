#!/usr/bin/env bash
# Codec-robust S1 bundleA (LATENT multi-bitrate Opus EOT) sweep for qwen25_omni
# across eps {0.5, 1.0}. Runs directly — no wait-for-GPU guard. Any concurrent
# S1 waveform job is independent work and shares GPU.
set -u
cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
LOG_ROOT=scripts
STAMP=$(date +%Y%m%d_%H%M%S)

run_eps() {
  local eps="$1"
  local log="$LOG_ROOT/run_s1_bundleA_omni_eps${eps}_${STAMP}.log"
  echo "=== [$(date)] Launching bundleA Omni eps=$eps log=$log ==="
  "$CA" -u scripts/run_s1_attack_bundleA.py \
      --target-model qwen25_omni --eps "$eps" \
      >"$log" 2>&1
  local rc=$?
  echo "=== [$(date)] bundleA Omni eps=$eps  exit=$rc ==="
}

run_eps 0.5
run_eps 1.0

echo "=== [$(date)] bundleA Omni sweep complete ==="
