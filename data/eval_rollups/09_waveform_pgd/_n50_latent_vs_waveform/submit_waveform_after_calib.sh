#!/usr/bin/env bash
# Watch for calibration results and submit each waveform/lp400 cell as soon as
# its (MODEL, SCEN) eps is known. Cells are submitted independently -- one slow
# calibration never holds up the others.
#
# 21 cells total:
#   waveform  s1  x {qwen2_audio, qwen25_omni, audio_flamingo} x 3 seeds  = 9
#   waveform  s3b x {qwen2_audio, qwen25_omni, audio_flamingo} x 3 seeds  = 9
#   lp400     s1  x {qwen2_audio, qwen25_omni}                 x 3 seeds  = 6
#   minus the 3 s3b/omni + ... -- see GRID below for the exact list.
#
# Run detached:
#   nohup bash submit_waveform_after_calib.sh > logs/submit_waveform.log 2>&1 &
set -uo pipefail

HERE=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
CALIB=${REPO_ROOT}/0_all_combined/calibration
cd "$HERE" || exit 1

# MODEL:SCEN:ARM  -- seeds are expanded to 1/7/42 for each
GRID=(
  "qwen2_audio:s1:waveform"     "qwen2_audio:s1:lp400"
  "qwen25_omni:s1:waveform"     "qwen25_omni:s1:lp400"
  "audio_flamingo:s1:waveform"
  "qwen2_audio:s3b:waveform"
  "qwen25_omni:s3b:waveform"
  "audio_flamingo:s3b:waveform"
)

SHORT() { case "$1" in qwen2_audio) echo q2a;; qwen25_omni) echo omni;;
                       audio_flamingo) echo af3;; esac; }

declare -A DONE
TOTAL=${#GRID[@]}
# 24 h, not 6: gpu-preempt is contended (the 25865 campaign is also queued), so
# a calibration job can sit PENDING for many hours before it even starts.
DEADLINE=$(( $(date +%s) + 24*3600 ))

echo "[$(date '+%F %T')] watching for ${TOTAL} calibration results..."

while :; do
    pending=0
    for entry in "${GRID[@]}"; do
        IFS=: read -r MODEL SCEN ARM <<< "$entry"
        key="$entry"
        [ -n "${DONE[$key]:-}" ] && continue

        json="$CALIB/snr_${MODEL}_${SCEN}.json"
        if [ ! -s "$json" ]; then pending=$((pending+1)); continue; fi

        EPS=$(python3 -c "
import json,sys
try:
    d=json.load(open('$json'))
    e=d.get('recommended_eps')
    print('' if e is None else f'{float(e):g}')
except Exception:
    print('')
" 2>/dev/null)

        if [ -z "$EPS" ]; then
            echo "[$(date '+%F %T')] !! $MODEL/$SCEN calibration has no recommended_eps -- SKIPPING $ARM"
            DONE[$key]=skip
            continue
        fi

        S=$(SHORT "$MODEL")
        echo "[$(date '+%F %T')] $MODEL/$SCEN/$ARM  eps=$EPS  -> submitting 3 seeds"
        for SEED in 1 7 42; do
            sbatch --export=ALL,MODEL="$MODEL",SCEN="$SCEN",ARM="$ARM",SEED="$SEED",EPS="$EPS" \
                   --job-name="${S}_${SCEN}_${ARM}_s${SEED}" sbatch_grid_cell.sh
        done
        DONE[$key]=1
    done

    if [ "$pending" -eq 0 ]; then
        echo "[$(date '+%F %T')] all calibrations resolved -- submitter exiting."
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[$(date '+%F %T')] !! deadline reached with $pending calibration(s) missing:"
        for entry in "${GRID[@]}"; do
            [ -z "${DONE[$entry]:-}" ] && echo "     $entry"
        done
        exit 1
    fi
    sleep 120
done
