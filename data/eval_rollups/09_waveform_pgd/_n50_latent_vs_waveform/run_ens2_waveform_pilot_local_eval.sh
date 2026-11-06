#!/bin/bash
# Evaluate the saved 300-step waveform pilot on four models, one GPU load at a time.
set -euo pipefail
cd ${REPO_ROOT}
mkdir -p logs/local_eval
BUNDLE=results/benchmark_ens2_waveform_transcribe_eps0.075_jazz_1_LOCAL_PILOT300
ENVROOT=${WORK_ROOT}/miniconda3/envs
CHANNELS=opus64,opus128,mp3_64,mp3_128,aac_64,aac_128
for ENTRY in \
    "qwen2_audio:qwen2_MEMBER_LOCAL:codec-attack" \
    "qwen25_omni:omni7b_MEMBER_LOCAL:qwen-omni" \
    "audio_flamingo:af3_HELDOUT_LOCAL:flamingo3" \
    "qwen25_omni_3b:omni3b_HELDOUT_LOCAL:qwen-omni"; do
    IFS=: read -r MODEL OUTNAME ENVNAME <<< "$ENTRY"
    LOG="logs/local_eval/ens2_waveform_${OUTNAME}.log"
    echo "scoring $MODEL -> $OUTNAME"
    "$ENVROOT/$ENVNAME/bin/python" -u \
        0_all_combined/scripts/ensemble_heldout_eval.py \
        --bundle-dir "$BUNDLE" --target-model "$MODEL" \
        --out-name "$OUTNAME" --channels "$CHANNELS" \
        > "$LOG" 2>&1
    tail -n 8 "$LOG" | tr '\r' '\n' | tail -n 8
done
