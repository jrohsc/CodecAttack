#!/usr/bin/env bash
# ============================================================================
# Idempotent submitter for the multi-seed + adaptive-baseline S1 bundles.
#
# Safe to run repeatedly: for each job it SKIPS if
#   (a) a SLURM job with that name is already queued/running for $USER, or
#   (b) the target bundle already has 50 codec sidecars (complete).
# Otherwise it submits. Prints a status line per job.
#
#   bash submit_multiseed.sh          # submit whatever is not already running/done
#   bash submit_multiseed.sh --dry    # just show what WOULD be submitted
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"          # n50_latent_vs_waveform/ (logs/ here)
REPO=${REPO_ROOT}
SBATCH_SCRIPT="$PWD/sbatch_multiseed.sh"
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
SCEN=01_finance_voice_agent

# job-name | SPACE | SEED | LP | bundle path (relative to REPO)
JOBS=(
  "ms_lat_s1|latent|1|0|results_codec_robust/${SCEN}/qwen2_audio/eps_1.0_seed1"
  "ms_lat_s7|latent|7|0|results_codec_robust/${SCEN}/qwen2_audio/eps_1.0_seed7"
  "ms_lat_s42|latent|42|0|results_codec_robust/${SCEN}/qwen2_audio/eps_1.0_seed42"
  "ms_wav_s1|waveform|1|0|results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_0.075_seed1"
  "ms_wav_s7|waveform|7|0|results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_0.075_seed7"
  "ms_wav_s42|waveform|42|0|results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_0.075_seed42"
  "ms_wavlp400_s42|waveform|42|400|results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_0.075_lp400_seed42"
)

RUNNING_NAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"

printf "%-18s %-10s %s\n" "JOB" "ACTION" "DETAIL"
printf "%-18s %-10s %s\n" "---" "------" "------"
for spec in "${JOBS[@]}"; do
  IFS='|' read -r NAME SPACE SEED LP BUNDLE <<< "$spec"
  full="$REPO/0_all_combined/$BUNDLE"
  ncodec=$(ls "$full/audio/"*_codecs.json 2>/dev/null | wc -l | tr -d ' ')

  if grep -qxF "$NAME" <<< "$RUNNING_NAMES"; then
    printf "%-18s %-10s %s\n" "$NAME" "SKIP-queued" "already in squeue"
    continue
  fi
  if [ "$ncodec" -ge 50 ]; then
    printf "%-18s %-10s %s\n" "$NAME" "SKIP-done" "$ncodec/50 codec sidecars"
    continue
  fi
  EXPORTS="SPACE=$SPACE,SEED=$SEED,LP=$LP"
  if [ "$DRY" = "1" ]; then
    printf "%-18s %-10s %s\n" "$NAME" "WOULD-SUB" "$EXPORTS (have $ncodec/50)"
    continue
  fi
  JID=$(sbatch --parsable --job-name="$NAME" --export="ALL,$EXPORTS" "$SBATCH_SCRIPT" 2>&1)
  if [[ "$JID" =~ ^[0-9]+$ ]]; then
    printf "%-18s %-10s %s\n" "$NAME" "SUBMITTED" "job $JID  ($EXPORTS, resume from $ncodec/50)"
  else
    printf "%-18s %-10s %s\n" "$NAME" "ERROR" "$JID"
  fi
done
