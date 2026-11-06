#!/bin/bash
# SNR-calibrate the waveform arm for one (MODEL, SCEN) cell.
#
# Picks the waveform eps whose clean SNR matches that cell's LATENT bundle, so
# the latent-vs-waveform contrast tests attack space and not budget. Needed per
# cell because the latent reference is not constant: measured across the six
# cells it spans 6.67 dB (qwen2/s1) to 8.82 dB (omni/s3b).
#
# The eps grid brackets low: eps=0.075 measures ~3.6 dB on s1, i.e. ~3 dB MORE
# perturbation than the latent arm, so the matched value is well below 0.075.
#
#   sbatch --export=ALL,MODEL=qwen2_audio,SCEN=s1,REF=<latent bundle> \
#          --job-name=cal_q2a_s1 sbatch_calibrate.sh
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

MODEL="${MODEL:?set MODEL}"
SCEN="${SCEN:?set SCEN=s1|s3b}"
REF="${REF:?set REF=<paired latent bundle>}"
EPS_GRID="${EPS_GRID:-0.02 0.03 0.05 0.075}"
STEPS="${STEPS:-300}"
K="${K:-3}"

ENVROOT=${WORK_ROOT}/miniconda3/envs
case "$MODEL" in
    qwen2_audio)    PY="$ENVROOT/codec-attack/bin/python" ;;
    qwen25_omni)    PY="$ENVROOT/qwen-omni/bin/python" ;;
    audio_flamingo) PY="$ENVROOT/flamingo3/bin/python" ;;
    *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== calibrate MODEL=$MODEL SCEN=$SCEN ==="
echo "    job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "    ref bundle: $REF"
echo "    eps grid:   $EPS_GRID   steps=$STEPS  k=$K"

# REF_SNR (optional): match this SNR instead of REF's own mean. Set it to the
# mean over all three latent seeds -- seed 1 measures lowest in every cell so
# far, and matching seed 1 alone is what pushed Omni S3b seeds 7/42 outside the
# 1 dB gate.
REF_SNR_FLAG=""
if [ -n "${REF_SNR:-}" ]; then
    REF_SNR_FLAG="--ref-snr-override $REF_SNR"
    echo "    ref SNR override: $REF_SNR dB (multi-seed mean)"
fi

$PY -u 0_all_combined/scripts/snr_calibrate_generic.py \
    --scenario "$SCEN" --target-model "$MODEL" \
    --ref-bundle "$REF" \
    --eps $EPS_GRID --steps "$STEPS" --k "$K" \
    --loader load_music $REF_SNR_FLAG

echo "=== calibration done: MODEL=$MODEL SCEN=$SCEN ==="
cat "0_all_combined/calibration/snr_${MODEL}_${SCEN}.json" 2>/dev/null \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('  recommended_eps =',d['recommended_eps'],' (snapped',d['recommended_eps_snapped'],'/ interp',d['recommended_eps_interp'],')')" \
    2>/dev/null || echo "  (no json written)"
