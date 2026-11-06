#!/usr/bin/env bash
# Queued cleanup pass for missing-codec-sidecar gaps surfaced 2026-05-02.
#
# Order (priority):
#   P0  AF3 S3a eps=0.5 — single missing pair004 sidecar
#   P1a S3b qwen2_audio eps=1.0  — orphaned, full attack + sidecar
#   P1b S3b audio_flamingo eps=0.5 — orphaned, full attack + sidecar
#
# Waits for the in-flight chain (PID args, plus user's earlier qwen25_omni
# eps=0.5 chain) to release the GPU before doing anything.
#
# Usage:
#   bash queue_sidecar_fixes.sh                # wait for all default PIDs
#   bash queue_sidecar_fixes.sh --skip-wait    # start immediately (DANGEROUS)
set -u

SKIP_WAIT=0
[[ "${1:-}" == "--skip-wait" ]] && SKIP_WAIT=1

PROJ=${REPO_ROOT}
cd "$PROJ"

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
S3A_ROOT=$PROJ/0_all_combined/results_codec_robust/03a_ai_detection_bypass
S3A_PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"

WATCH_PIDS=(1769517 1769196)

if [[ $SKIP_WAIT -eq 0 ]]; then
    echo "=== [$(date)] Waiting for in-flight chain PIDs: ${WATCH_PIDS[*]} ==="
    for pid in "${WATCH_PIDS[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 60
        done
        echo "  PID $pid done at $(date)"
    done
    # small grace period so any final GPU release settles
    sleep 30
fi

echo
echo "######################################################"
echo "###  P0: AF3 S3a eps=0.5 single sidecar fix        ###"
echo "######################################################"
echo "[$(date)] starting"
"$AF3" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$S3A_ROOT/audio_flamingo/eps_0.5_multibitrate" \
    --target-model audio_flamingo \
    --prompt-text "$S3A_PROMPT"
echo "[$(date)] P0 done"

echo
echo "######################################################"
echo "###  P1a: S3b qwen2_audio eps=1.0 chain            ###"
echo "######################################################"
bash 0_all_combined/scripts/chain_s3b_n45_codec_robust.sh \
    --models qwen2_audio --eps 1.0
echo "[$(date)] P1a done"

echo
echo "######################################################"
echo "###  P1b: S3b audio_flamingo eps=0.5 chain         ###"
echo "######################################################"
bash 0_all_combined/scripts/chain_s3b_n45_codec_robust.sh \
    --models audio_flamingo --eps 0.5
echo "[$(date)] P1b done"

echo
echo "######################################################"
echo "###  QUEUED SIDECAR FIXES COMPLETE                 ###"
echo "######################################################"
