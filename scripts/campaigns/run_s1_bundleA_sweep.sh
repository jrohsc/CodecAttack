#!/bin/bash
set -e
cd ${REPO_ROOT}
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
R=0_all_combined/scripts

echo "[queue] $(date) -- launching S1 codec-robust sweep (5 bundles)"

echo "[1/5] qwen2_audio eps=0.5 -- $(date)"
CODEC_ATTACK_PY=$CA $CA -u $R/run_s1_attack_bundleA.py --target-model qwen2_audio --eps 0.5 \
  > $R/run_s1_bundleA_qwen2_eps0.5.log 2>&1

echo "[2/5] qwen25_omni eps=0.5 -- $(date)"
CODEC_ATTACK_PY=$OMNI $CA -u $R/run_s1_attack_bundleA.py --target-model qwen25_omni --eps 0.5 \
  > $R/run_s1_bundleA_omni_eps0.5.log 2>&1

echo "[3/5] qwen25_omni eps=1.0 -- $(date)"
CODEC_ATTACK_PY=$OMNI $CA -u $R/run_s1_attack_bundleA.py --target-model qwen25_omni --eps 1.0 \
  > $R/run_s1_bundleA_omni_eps1.0.log 2>&1

echo "[4/5] audio_flamingo eps=0.5 -- $(date)"
CODEC_ATTACK_PY=$AF3 $CA -u $R/run_s1_attack_bundleA.py --target-model audio_flamingo --eps 0.5 \
  > $R/run_s1_bundleA_af3_eps0.5.log 2>&1

echo "[5/5] audio_flamingo eps=1.0 -- $(date)"
CODEC_ATTACK_PY=$AF3 $CA -u $R/run_s1_attack_bundleA.py --target-model audio_flamingo --eps 1.0 \
  > $R/run_s1_bundleA_af3_eps1.0.log 2>&1

echo "[queue] $(date) -- all 5 bundles complete"
