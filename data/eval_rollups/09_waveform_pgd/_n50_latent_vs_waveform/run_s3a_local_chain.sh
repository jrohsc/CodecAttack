#!/usr/bin/env bash
# Finish the S3a waveform grid on the local GPU, one cell at a time.
#
# Sequential, not concurrent: AF3 seed 1 already holds 40 GB of the 80 GB card
# at 60% utilisation, and a co-scheduled Omni PGD job would risk an OOM. An OOM
# mid-clip is not a clean failure here -- it leaves an attacked wav with no
# summary.json, and stage 1 resume keys on the wav alone, so that clip becomes
# invisible to every later resubmit (exactly the 39/40 bug this chain repairs).
#
# Order: the two Omni backfills first. They are short (39 clips skip, 1 attack,
# then the neural eval) and they close out cells that are otherwise stuck at
# 39/40, so the quick wins bank early. The AF3 seeds follow -- those are full
# 40-clip attacks and dominate the wall clock either way.
#
# Each cell cancels its own queued SLURM twin immediately before running. Both
# paths write the same bundle directory; two writers corrupt it.
#
#   nohup bash run_s3a_local_chain.sh > logs/s3a_local_chain.log 2>&1 &
set -uo pipefail

HERE=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
cd "$HERE" || exit 1
ts() { date '+%F %T'; }

# MODEL:SCEN:ARM:SEED:EPS:jobname
CELLS=(
  "qwen25_omni:s3a:waveform:1:0.04679:omni_s3a_wav_s1"
  "qwen25_omni:s3a:waveform:42:0.04679:omni_s3a_wav_s42"
  "audio_flamingo:s3a:waveform:7:0.06211:af3_s3a_wav_s7"
  "audio_flamingo:s3a:waveform:42:0.06211:af3_s3a_wav_s42"
)

# The chain owns the GPU, so it must not start until the seed-1 run that is
# already on it has exited.
echo "[$(ts)] waiting for the running local cell to finish..."
while pgrep -f "run_grid_cell_local.sh audio_flamingo s3a waveform 1" > /dev/null; do
    sleep 60
done
echo "[$(ts)] GPU free -- starting chain"

BASE=${REPO_ROOT}/0_all_combined/results_codec_robust_waveform/03a_ai_detection_bypass

# Count summaries and sidecars, never wavs. A cell preempted between the wav
# write and the summary write leaves a wav that stage-1 resume then skips
# forever, so a wav count of 40 can sit on top of a 39-clip bundle.
complete() {
    local d="$1"
    [ -d "$d" ] || return 1
    local s c
    s=$(ls "$d"/audio/*_summary.json 2>/dev/null | wc -l)
    c=$(ls "$d"/audio/*_codecs.json 2>/dev/null | wc -l)
    [ "$s" -eq 40 ] && [ "$c" -eq 40 ] \
        && ls "$d"/neural_delivery_eval_*.json > /dev/null 2>&1
}

for entry in "${CELLS[@]}"; do
    IFS=: read -r MODEL SCEN ARM SEED EPS JOB <<< "$entry"

    # A finished cell leaves no squeue row, which is indistinguishable from a
    # cell that never ran -- so check the bundle itself before anything else,
    # or the chain re-runs completed work.
    if complete "$BASE/$MODEL/eps_${EPS}_seed${SEED}"; then
        echo "[$(ts)] SKIP $JOB -- bundle already complete (40/40 + neural eval)"
        continue
    fi

    # If SLURM already started this cell, leave it alone: it is doing the same
    # work on another node and cancelling it would waste that progress.
    STATE=$(squeue -u "$USER" -h -o "%T" -n "$JOB" 2>/dev/null | head -1)
    if [ "$STATE" = "RUNNING" ]; then
        echo "[$(ts)] SKIP $JOB -- already RUNNING under SLURM"
        continue
    fi
    if [ -n "$STATE" ]; then
        echo "[$(ts)] cancelling queued $JOB before running it locally"
        scancel -n "$JOB"
        sleep 5
        if squeue -u "$USER" -h -o "%j" | grep -qx "$JOB"; then
            echo "[$(ts)] !! $JOB still in queue after scancel -- skipping to avoid a double writer"
            continue
        fi
    fi

    echo "[$(ts)] === START $JOB (eps=$EPS) ==="
    bash run_grid_cell_local.sh "$MODEL" "$SCEN" "$ARM" "$SEED" "$EPS" \
        > "logs/${JOB}_LOCAL.out" 2>&1
    echo "[$(ts)] === END $JOB rc=$? ==="
done

echo "[$(ts)] chain done"
