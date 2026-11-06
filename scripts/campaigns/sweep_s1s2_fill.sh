#!/bin/bash
# Fill S1 finance gaps (AF3 eps_0.5, AF3 eps_1.0, Omni eps_0.5) then run the
# same multi-bitrate sweep on S2 interview across the three target models.
# Drivers are idempotent: existing per-pair attacked .wavs are skipped.

set -u
cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
R=0_all_combined/scripts

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[queue] $(ts) -- starting fill sweep"

# -------- S1 finance: fill missing --------
echo "[1/7] $(ts) S1 AF3 eps=0.5 (from scratch, 25 pairs)"
CODEC_ATTACK_PY=$AF3 $CA -u $R/run_s1_attack_bundleA.py \
    --target-model audio_flamingo --eps 0.5 \
    > $R/fill_s1_af3_eps0.5.log 2>&1

echo "[2/7] $(ts) S1 AF3 eps=1.0 (continue, 16 remaining)"
CODEC_ATTACK_PY=$AF3 $CA -u $R/run_s1_attack_bundleA.py \
    --target-model audio_flamingo --eps 1.0 \
    > $R/fill_s1_af3_eps1.0.log 2>&1

echo "[3/7] $(ts) S1 Omni eps=0.5 (continue, fills q3 + michelle_q2 redo)"
CODEC_ATTACK_PY=$OMNI $CA -u $R/run_s1_attack_bundleA.py \
    --target-model qwen25_omni --eps 0.5 \
    > $R/fill_s1_omni_eps0.5.log 2>&1

# -------- S2 interview: fresh multi-bitrate sweep, 3 models x 2 eps --------
echo "[4/7] $(ts) S2 qwen2_audio multi_bitrate eps=0.5 + 1.0"
CODEC_ATTACK_PY=$CA $CA -u $R/run_s2_attack.py \
    --target-model qwen2_audio --eps 0.5 1.0 \
    --channel-mode multi_bitrate --skip-eval \
    > $R/fill_s2_qwen2_audio.log 2>&1

echo "[5/7] $(ts) S2 audio_flamingo multi_bitrate eps=0.5 + 1.0"
CODEC_ATTACK_PY=$AF3 $CA -u $R/run_s2_attack.py \
    --target-model audio_flamingo --eps 0.5 1.0 \
    --channel-mode multi_bitrate --skip-eval \
    > $R/fill_s2_af3.log 2>&1

echo "[6/7] $(ts) S2 qwen25_omni multi_bitrate eps=0.5 + 1.0"
CODEC_ATTACK_PY=$OMNI $CA -u $R/run_s2_attack.py \
    --target-model qwen25_omni --eps 0.5 1.0 \
    --channel-mode multi_bitrate --skip-eval \
    > $R/fill_s2_omni.log 2>&1

echo "[7/7] $(ts) fill sweep COMPLETE"
