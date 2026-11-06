#!/bin/bash
# Drain the S2-English grid onto this node, one cell at a time.
#
#   bash run_s2en_chain_local.sh
#
# Policy per cell, in list order:
#   - a live SLURM job is RUNNING it   -> skip. It is already making progress on
#                                         a dedicated GPU; taking it locally
#                                         would throw that work away.
#   - a live SLURM job is PENDING      -> scancel, wait for it to leave the
#                                         queue, then run it here.
#   - no job owns it                   -> run it here.
#   - bundle already at full n         -> skip.
#
# Never runs two cells at once, and never starts a cell SLURM still owns: one
# writer per bundle dir is the invariant (TODO.md:300).
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 1
REPO=${REPO_ROOT}
LATENT="$REPO/0_all_combined/results_codec_robust/02_interview_screening"
EXPECT=25

declare -A TAG=( [qwen2_audio]=q2a [qwen25_omni]=omni [audio_flamingo]=af3 )
# Same-model cells first: the carrier set and env are already warm.
CELLS=(
    "qwen2_audio 7" "qwen2_audio 42"
    "qwen25_omni 1" "qwen25_omni 7" "qwen25_omni 42"
    "audio_flamingo 1" "audio_flamingo 7" "audio_flamingo 42"
)

# Wait out whatever cell is already running on this node before claiming more.
while pgrep -f "run_s2_attack.py" > /dev/null 2>&1; do
    echo "[chain] a local cell is still running; waiting 120s"
    sleep 120
done

for cell in "${CELLS[@]}"; do
    set -- $cell
    MODEL=$1; SEED=$2
    JOB="${TAG[$MODEL]}_s2en_lat_s${SEED}"
    BUNDLE="$LATENT/$MODEL/eps_1.0_multibitrate_en_seed${SEED}"

    n=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
    if [ "$n" -ge "$EXPECT" ]; then
        echo "[chain] $JOB already $n/$EXPECT -- skip"
        continue
    fi

    state=$(squeue -u "$USER" -h -o "%j %T" | awk -v j="$JOB" '$1==j {print $2}' | head -1)
    if [ "$state" = "RUNNING" ]; then
        echo "[chain] $JOB is RUNNING on SLURM -- leave it there"
        continue
    fi
    if [ -n "$state" ]; then
        echo "[chain] $JOB is $state on SLURM -- cancelling to run here"
        scancel -n "$JOB"
        for _ in $(seq 1 30); do
            squeue -u "$USER" -h -o "%j" | grep -qx "$JOB" || break
            sleep 2
        done
        if squeue -u "$USER" -h -o "%j" | grep -qx "$JOB"; then
            echo "[chain] $JOB did not leave the queue -- skipping, will not risk two writers"
            continue
        fi
    fi

    echo "[chain] === starting $JOB locally at $(date -Is) ==="
    bash run_s2en_local.sh "$MODEL" "$SEED" 2>&1 \
        | tee -a "logs/local_${JOB}.log"
    n=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
    echo "[chain] === $JOB finished at $n/$EXPECT wavs ($(date -Is)) ==="
done

echo "[chain] list exhausted at $(date -Is)"
