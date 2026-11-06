#!/usr/bin/env bash
# S1 finance eps=1.5 sweep across {qwen2_audio, qwen25_omni, audio_flamingo}.
# Each stage = attack (run_s1_attack_bundleA.py) then codec sidecar eval
# (eval_mp3_aac_bundle.py). Serial -- all three models share one GPU.
# Waits for the current S2 qwen2_audio multibitrate attack (PID $BLOCKER_PID)
# to finish before starting, so the codec-attack env is free.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S1_ROOT=0_all_combined/results_codec_robust/01_finance_voice_agent
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

BLOCKER_PID="${BLOCKER_PID:-}"
EPS=1.5

run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] STAGE: $name ==="
    echo "  log: $log"
    "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED rc=$rc -- abort chain"
        exit $rc
    fi
    echo "  OK"
}

if [[ -n "$BLOCKER_PID" ]]; then
    echo "[$(date +%H:%M:%S)] waiting for PID $BLOCKER_PID (S2 qwen2_audio attack) to finish..."
    while kill -0 "$BLOCKER_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%H:%M:%S)] blocker exited; codec-attack env free, starting S1 eps=${EPS} chain"
fi

# 1. qwen2_audio (codec-attack env)
run_stage s1_qwen2_audio_eps${EPS}_attack \
    env CODEC_ATTACK_PY="$CA" \
    "$CA" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model qwen2_audio --eps "$EPS"

run_stage s1_qwen2_audio_eps${EPS}_codec_sidecars \
    "$CA" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/qwen2_audio/eps_${EPS}" \
        --target-model qwen2_audio

# 2. qwen25_omni (qwen-omni env)
run_stage s1_qwen25_omni_eps${EPS}_attack \
    env CODEC_ATTACK_PY="$OMNI" \
    "$OMNI" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model qwen25_omni --eps "$EPS"

run_stage s1_qwen25_omni_eps${EPS}_codec_sidecars \
    "$OMNI" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/qwen25_omni/eps_${EPS}" \
        --target-model qwen25_omni

# 3. audio_flamingo (flamingo3 env)
run_stage s1_audio_flamingo_eps${EPS}_attack \
    env CODEC_ATTACK_PY="$AF3" \
    "$AF3" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model audio_flamingo --eps "$EPS"

run_stage s1_audio_flamingo_eps${EPS}_codec_sidecars \
    "$AF3" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/audio_flamingo/eps_${EPS}" \
        --target-model audio_flamingo

echo ""
echo "######################################################"
echo "###  S1 eps=${EPS} FULL SWEEP COMPLETE             ###"
echo "######################################################"
