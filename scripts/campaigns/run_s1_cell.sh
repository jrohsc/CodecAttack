#!/usr/bin/env bash
# Non-interactive single-cell S1 launcher.
# Runs attack (run_s1_attack_bundleA.py, all 50 carriers, skip-existing) then
# codec sidecar eval (eval_mp3_aac_bundle.py) for one (model, eps) cell.
#
# Designed to be launched from N parallel terminals -- different (model, eps)
# cells write to disjoint bundle dirs, so concurrent runs don't collide.
# GPU caveat: only one A100-80GB visible; two concurrent 7B-model attacks
# usually fit, three is risky.
#
# Usage:
#   bash 0_all_combined/scripts/run_s1_cell.sh <model> <eps>
#     model: qwen2_audio | qwen25_omni | audio_flamingo
#     eps  : 0.5 | 1.0 | 1.5 (or any positive number)
#
# Examples:
#   bash 0_all_combined/scripts/run_s1_cell.sh qwen2_audio    0.5
#   bash 0_all_combined/scripts/run_s1_cell.sh qwen25_omni    1.0
#   bash 0_all_combined/scripts/run_s1_cell.sh audio_flamingo 1.5
set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <model> <eps>"
    echo "  model: qwen2_audio | qwen25_omni | audio_flamingo"
    echo "  eps  : 0.5 | 1.0 | 1.5"
    exit 2
fi

MODEL="$1"
EPS="$2"

if ! [[ "$EPS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Invalid eps '$EPS' (must be a positive number)"; exit 2
fi

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

case "$MODEL" in
    qwen2_audio)    PY="$CA";   ENV_NAME="codec-attack" ;;
    qwen25_omni)    PY="$OMNI"; ENV_NAME="qwen-omni" ;;
    audio_flamingo) PY="$AF3";  ENV_NAME="flamingo3" ;;
    *) echo "Unknown model '$MODEL' (qwen2_audio|qwen25_omni|audio_flamingo)"; exit 2 ;;
esac

S1_ROOT=0_all_combined/results_codec_robust/01_finance_voice_agent
BUNDLE="$S1_ROOT/$MODEL/eps_${EPS}"
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

TAG="s1_${MODEL}_eps${EPS}"
ATTACK_LOG="$LOGDIR/${TAG}_attack.log"
SIDECAR_LOG="$LOGDIR/${TAG}_sidecars.log"

echo "======================================================"
echo " S1 finance voice agent  /  $MODEL  /  eps=$EPS"
echo " env    : $ENV_NAME"
echo " bundle : $BUNDLE"
echo " attack log  : $ATTACK_LOG"
echo " sidecar log : $SIDECAR_LOG"
echo "======================================================"

echo "=== [$(date +%F\ %H:%M:%S)] ATTACK STAGE START -> $ATTACK_LOG ==="
env CODEC_ATTACK_PY="$PY" "$PY" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model "$MODEL" --eps "$EPS" \
    > "$ATTACK_LOG" 2>&1
RC=$?
if (( RC != 0 )); then
    echo "=== [$(date +%F\ %H:%M:%S)] ATTACK STAGE FAILED rc=$RC (see $ATTACK_LOG) ==="
    exit $RC
fi
echo "=== [$(date +%F\ %H:%M:%S)] ATTACK STAGE OK ==="

echo "=== [$(date +%F\ %H:%M:%S)] SIDECAR STAGE START -> $SIDECAR_LOG ==="
"$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model "$MODEL" \
    > "$SIDECAR_LOG" 2>&1
RC=$?
if (( RC != 0 )); then
    echo "=== [$(date +%F\ %H:%M:%S)] SIDECAR STAGE FAILED rc=$RC (see $SIDECAR_LOG) ==="
    exit $RC
fi
echo "=== [$(date +%F\ %H:%M:%S)] SIDECAR STAGE OK ==="

echo "######################################################"
echo "### DONE: S1 / $MODEL / eps=$EPS"
echo "######################################################"
