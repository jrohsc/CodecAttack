#!/bin/bash
# ============================================================================
# SLURM job: codec-EoT ablation cell (AC Q5) — one S1/Qwen2-Audio latent bundle
# whose EoT training codec is AAC or Opus+AAC ensemble, + MP3/AAC sidecar eval.
#
# Answers meta-review Q5 ("why Opus for EoT; would MP3/AAC/ensemble generalize
# better?"). Opus-only is the existing headline/seed-sweep bundle
# (results_codec_robust/.../eps_1.0_seed<SEED>); this job adds the AAC-only and
# ensemble rows, evaluated on the SAME held-out delivery codecs (Opus/MP3/AAC).
# MP3 is kept out of every training grid so it stays a genuine held-out codec.
#
# Parametrized via --export:
#   TRAIN_CODEC   aac | ensemble       (EoT training codec; opus already exists)
#   SEED          integer (default 1)  (match the Opus comparator seed)
#
# Writes to a SEPARATE result tree (results_codec_robust_aac / _ensemble) so it
# never collides with the Opus seed-sweep bundles or the paper's numbers.
# Resumable (attack + sidecar skip pairs already on disk). Safe to requeue.
#
# Examples:
#   sbatch --export=TRAIN_CODEC=aac,SEED=1      --job-name=ce_aac_s1      sbatch_codec_eot.sh
#   sbatch --export=TRAIN_CODEC=ensemble,SEED=1 --job-name=ce_ens_s1      sbatch_codec_eot.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=60G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
PY=$(which python)

TRAIN_CODEC="${TRAIN_CODEC:?set TRAIN_CODEC=aac|ensemble}"
SEED="${SEED:-1}"
EPS=1.0
SCEN=01_finance_voice_agent

case "$TRAIN_CODEC" in
    aac)      RESULTS_DIR=results_codec_robust_aac ;;
    mp3)      RESULTS_DIR=results_codec_robust_mp3 ;;
    ensemble) RESULTS_DIR=results_codec_robust_ensemble ;;
    *) echo "ERROR: TRAIN_CODEC must be aac|mp3|ensemble"; exit 1 ;;
esac
BUNDLE="0_all_combined/${RESULTS_DIR}/${SCEN}/qwen2_audio/eps_${EPS}_seed${SEED}"

echo "=========================================="
echo "  codec-EoT ablation:  TRAIN_CODEC=$TRAIN_CODEC SEED=$SEED"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
echo "  python: $PY"
python -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'
echo "  bundle: $BUNDLE"
echo "=========================================="

echo "=== STAGE 1: latent attack, EoT=$TRAIN_CODEC (eps=$EPS seed=$SEED) ==="
python -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen2_audio --eps "$EPS" --seed "$SEED" --rand-init \
    --train-codec "$TRAIN_CODEC"

echo "=== STAGE 2: MP3+AAC codec sidecar eval on $BUNDLE ==="
python -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio

echo "=== DONE  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
