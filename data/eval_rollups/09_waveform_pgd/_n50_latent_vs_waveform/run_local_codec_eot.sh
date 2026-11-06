#!/bin/bash
# Local (non-SLURM) runner for the codec-EoT ablation cell. Same two stages as
# sbatch_codec_eot.sh. Usage: bash run_local_codec_eot.sh <aac|ensemble> [SEED]
set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
TRAIN_CODEC="${1:?usage: run_local_codec_eot.sh <aac|ensemble> [SEED]}"
SEED="${2:-1}"; EPS=1.0; SCEN=01_finance_voice_agent
case "$TRAIN_CODEC" in
  aac)      RESULTS_DIR=results_codec_robust_aac ;;
  ensemble) RESULTS_DIR=results_codec_robust_ensemble ;;
  *) echo "ERROR: TRAIN_CODEC must be aac|ensemble"; exit 1 ;;
esac
BUNDLE="0_all_combined/${RESULTS_DIR}/${SCEN}/qwen2_audio/eps_${EPS}_seed${SEED}"
ts(){ date "+%F %H:%M:%S"; }
echo "=== [$(ts)] LOCAL codec-EoT TRAIN_CODEC=$TRAIN_CODEC SEED=$SEED GPU=${CUDA_VISIBLE_DEVICES:-?} bundle=$BUNDLE ==="
echo "=== [$(ts)] STAGE 1: latent attack (EoT=$TRAIN_CODEC) ==="
python -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen2_audio --eps "$EPS" --seed "$SEED" --rand-init \
    --train-codec "$TRAIN_CODEC"
echo "=== [$(ts)] STAGE 2: MP3+AAC sidecar eval ==="
python -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio
echo "=== [$(ts)] DONE bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
