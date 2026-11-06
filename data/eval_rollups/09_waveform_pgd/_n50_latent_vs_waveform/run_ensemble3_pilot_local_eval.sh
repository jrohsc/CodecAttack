#!/bin/bash
# Score the corrected two-clip pilots on the local 40 GB GPU, one model at a time.
set -euo pipefail

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p logs/local_eval

ENVROOT=${WORK_ROOT}/miniconda3/envs
CHANNELS=opus16,opus32,opus64,opus128,mp3_64,mp3_128,aac_64,aac_128
for CODEC in opus mp3; do
    BUNDLE="results/benchmark_ensemble3_${CODEC}eot_transcribe_eps10.0_jazz_1_bestpostcodec_steps300_pilot2_shard0of4"
    for ENTRY in \
        "qwen2_audio:qwen2_MEMBER_LOCAL:codec-attack" \
        "qwen25_omni:omni7b_MEMBER_LOCAL:qwen-omni" \
        "audio_flamingo:af3_MEMBER_LOCAL:flamingo3" \
        "qwen25_omni_3b:omni3b_HELDOUT_LOCAL:qwen-omni"; do
        IFS=: read -r MODEL OUTNAME ENVNAME <<< "$ENTRY"
        LOG="logs/local_eval/${CODEC}_${OUTNAME}_allcodecs.log"
        echo "${CODEC}: ${MODEL} -> ${OUTNAME} (all codecs)"
        "$ENVROOT/$ENVNAME/bin/python" -u \
            0_all_combined/scripts/ensemble_heldout_eval.py \
            --bundle-dir "$BUNDLE" --target-model "$MODEL" \
            --out-name "$OUTNAME" --channels "$CHANNELS" \
            > "$LOG" 2>&1
        tail -n 9 "$LOG" | tr '\r' '\n' | tail -n 9
    done
done
