#!/usr/bin/env bash
# ============================================================================
# Full 3-seed campaign for additional band-limited cutoffs (Table 4 extension).
#
# Deliberately a thin submitter over sbatch_omni_waveform_eps.sh rather than a
# new launcher: that wrapper already runs stage 1 (attack) AND stage 2 (MP3/AAC
# sidecar) and, critically, already carries the guard that stops a preempted
# stage 1 from falling through to a sidecar over a partial wav set. Re-writing
# it here is how the lp400 arm first shipped with zero *_codecs.json.
#
# Each cutoff needs its OWN eps', taken from the calibration probes
# (run_lp_cutoff_calib_local.sh -> lp_cutoff_calib.jsonl). Do NOT reuse 0.30.
#
# MODE=fft is the DEFAULT here, unlike in sbatch_omni_waveform_eps.sh (which
# defaults to lp so its existing callers are unchanged). Reason: the point of
# this campaign is to read as a cutoff sweep, and --lowpass-hz does not give
# one. Measured on this tree's own seed998 probes, delta energy below 400 Hz is
#     lp800   99.3%     <- an lp "800 Hz" arm IS the 400 Hz arm
#     fft800  43.7%     <- a genuine 800 Hz arm (99.2% below 800)
#     fft1500 25.6%     <- 52.5% below 800, 98.7% below 1500
# An lp-mode sweep would compare three copies of one arm and the reviewer
# question ("is low-frequency placement alone sufficient?") would go unanswered.
#
# Usage:
#   bash submit_lp_cutoff_campaign.sh 800:0.40 1500:0.38
#   MODE=lp bash submit_lp_cutoff_campaign.sh 400:0.30      # legacy sosfilt arm
#   DRY=1 bash submit_lp_cutoff_campaign.sh 800:0.40        # print only
#
# Two sessions have submitted into this tree before and raced on the same
# bundle dir, so every job is guarded on its own name being absent from squeue.
# ============================================================================
set -uo pipefail

HERE=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
cd "$HERE" || exit 1

SEEDS="${SEEDS:-1 7 42}"
STEPS="${STEPS:-1000}"
DRY="${DRY:-0}"
MODE="${MODE:-fft}"
case "$MODE" in
    lp)  TAGPFX="lp"    ;;
    fft) TAGPFX="fftlp" ;;
    *) echo "ERROR: MODE must be lp or fft (got '$MODE')"; exit 2 ;;
esac
BASE=${REPO_ROOT}/0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni

[ "$#" -eq 0 ] && { echo "usage: $0 CUTOFF:EPS [CUTOFF:EPS ...]"; exit 2; }

RUNNING=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)

for SPEC in "$@"; do
    HZ="${SPEC%%:*}"
    EPS="${SPEC##*:}"
    EPS_TAG=$(printf '%g' "$EPS")          # match the driver's f"{eps:g}"
    for SEED in $SEEDS; do
        NAME="${TAGPFX}${HZ}m_e${EPS_TAG}_s${SEED}"
        # Must track sbatch_omni_waveform_eps.sh's LP_TAG, which in turn tracks
        # the driver's bundle_suffix. Mismatch here does not fail loudly -- it
        # makes NWAV read a directory nothing writes, so "resuming from 0/50"
        # and "already complete" both silently become wrong.
        BUNDLE="$BASE/eps_${EPS_TAG}_${TAGPFX}${HZ}_seed${SEED}"
        NWAV=$(find "$BUNDLE/audio" -name '*_attacked.wav' 2>/dev/null | wc -l)

        if grep -qx "$NAME" <<<"$RUNNING"; then
            echo "SKIP  $NAME -- already in squeue"
            continue
        fi
        if [ "$NWAV" -ge 50 ]; then
            echo "SKIP  $NAME -- bundle already complete ($NWAV/50)"
            continue
        fi
        # A partial bundle is fine to resubmit: the driver skips-when-exists,
        # so the job resumes rather than restarting. SLURM logging COMPLETED is
        # never evidence a bundle is whole -- only this wav count is.
        [ "$NWAV" -gt 0 ] && echo "  ($NAME resuming from $NWAV/50)"

        CMD=(sbatch --export=ALL,EPS="$EPS_TAG",LP="$HZ",MODE="$MODE",SEED="$SEED",STEPS="$STEPS"
             --job-name="$NAME" sbatch_omni_waveform_eps.sh)
        if [ "$DRY" = "1" ]; then
            echo "DRY   ${CMD[*]}"
        else
            OUT=$("${CMD[@]}" 2>&1)
            echo "SUBMIT $NAME -> $OUT"
        fi
    done
done

echo
echo "=== queue after submit ==="
squeue -u "$USER" -o "%.10i %.26j %.9T %.10M %R" 2>/dev/null
