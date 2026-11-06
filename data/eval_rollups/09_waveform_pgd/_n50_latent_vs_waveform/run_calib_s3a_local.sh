#!/bin/bash
# S3a SNR calibration on the local interactive A100, run sequentially.
#
# Same command the sbatch path would run -- only the scheduler differs. Kept as
# a script rather than an inline loop so a resumed/re-launched session runs
# byte-identical work.
#
# Ordering is deliberate: qwen2_audio first. It is the fastest env and shares
# the code path with the other two, so a wiring bug surfaces in ~35 min rather
# than after the whole 2 h chain.
#
# REFSNR values are the mean latent SNR over seeds {1,7,42}, measured
# 2026-08-09 with measure_bundle_snr.py --loader load_music. Matching seed 1
# alone biases eps' high (seed 1 is the lowest-SNR seed) -- that is how Omni
# S3b failed its own 1 dB gate on two of three seeds.
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1
LOGDIR=0_all_combined/n50_latent_vs_waveform/logs
mkdir -p "$LOGDIR"

ENVROOT=${WORK_ROOT}/miniconda3/envs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one () {
    local MODEL="$1" REFSNR="$2" PY="$3"
    local LOG="$LOGDIR/calib_s3a_local_${MODEL}.log"
    local JS="0_all_combined/calibration/snr_${MODEL}_s3a.json"

    if [ -f "$JS" ]; then
        echo "[local-calib] SKIP $MODEL -- $JS already exists"
        return 0
    fi
    echo "[local-calib] START $MODEL ref=${REFSNR} dB  -> $LOG"
    $PY -u 0_all_combined/scripts/snr_calibrate_generic.py \
        --scenario s3a --target-model "$MODEL" \
        --ref-bundle "0_all_combined/results_codec_robust/03a_ai_detection_bypass/${MODEL}/eps_1.0_multibitrate_seed1" \
        --ref-snr-override "$REFSNR" \
        --loader load_music \
        --eps 0.03 0.05 0.075 0.10 0.15 --steps 300 --k 3 \
        > "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$JS" ]; then
        echo "[local-calib] OK $MODEL"
    else
        echo "[local-calib] FAIL $MODEL rc=$rc (see $LOG)"
    fi
    return $rc
}

run_one qwen2_audio    7.823 "$ENVROOT/codec-attack/bin/python"
run_one qwen25_omni    8.330 "$ENVROOT/qwen-omni/bin/python"
run_one audio_flamingo 7.874 "$ENVROOT/flamingo3/bin/python"

echo "[local-calib] CHAIN DONE"
