#!/usr/bin/env bash
# Resume the three in-flight partial bundles needed to close the headline
# cross-scenario row at eps=1.0 (S2/S3a/S3b x 3 models). Order = smallest delta
# first so the cheapest rows land first:
#   1) S3b / qwen25_omni / eps=1.0_multibitrate  12/20 -> 20/20  (+8)
#   2) S3a / qwen25_omni / eps=1.0_multibitrate  14/40 -> 40/40  (+26)
#   3) S3a / audio_flamingo / eps=1.0_multibitrate  11/40 -> 40/40 (+29)
#
# All three drivers are idempotent (skip-existing via glob match on carrier
# stem + pair index), so no per-pair gating here.
#
# Per-stage log: 0_all_combined/scripts/logs/<stage>.log
# Master log (via nohup): 0_all_combined/scripts/chain_finish_partials.log

set -u
cd ${REPO_ROOT}

OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

EPS=1.0

run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    echo "=== [$(date +%F\ %H:%M:%S)] STAGE: $name ==="
    echo "  log: $log"
    "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED rc=$rc -- continuing (do not abort whole chain)"
    else
        echo "  OK"
    fi
}

# 1) S3b / qwen25_omni / eps=1.0  (+8)
run_stage "resume_s3b_qwen25_omni_eps${EPS}_attack" \
    env CODEC_ATTACK_PY="$OMNI" \
    "$OMNI" -u 0_all_combined/scripts/run_s3_attack.py \
        --sub-scenario 03b --target-model qwen25_omni --eps "$EPS" \
        --channel-mode multi_bitrate --skip-eval

# 2) S3a / qwen25_omni / eps=1.0  (+26)
run_stage "resume_s3a_qwen25_omni_eps${EPS}_attack" \
    env CODEC_ATTACK_PY="$OMNI" \
    "$OMNI" -u 0_all_combined/scripts/run_s3_attack.py \
        --sub-scenario 03a --target-model qwen25_omni --eps "$EPS" \
        --channel-mode multi_bitrate --skip-eval

# 3) S3a / audio_flamingo / eps=1.0  (+29)
run_stage "resume_s3a_audio_flamingo_eps${EPS}_attack" \
    env CODEC_ATTACK_PY="$AF3" \
    "$AF3" -u 0_all_combined/scripts/run_s3_attack.py \
        --sub-scenario 03a --target-model audio_flamingo --eps "$EPS" \
        --channel-mode multi_bitrate --skip-eval

echo ""
echo "######################################################"
echo "### PARTIAL-RESUME CHAIN DONE                      ###"
echo "######################################################"
