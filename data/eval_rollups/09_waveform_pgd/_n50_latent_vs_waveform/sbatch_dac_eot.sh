#!/bin/bash
# ============================================================================
# SLURM job: DAC-EoT ADAPTIVE ATTACK (defenses / W3tz + neural-delivery follow-up).
#
# Context: neural-codec re-encoding (DAC/EnCodec round-trip) STRIPS the headline
# attack (clean 44/50 -> DAC 0/50), i.e. it is an effective *non-adaptive*
# resynthesis defense. This job runs the ADAPTIVE counter: train the perturbation
# WITH a DAC continuous round-trip in the EoT loss (channel_mode=dac_multibitrate),
# then test whether it now SURVIVES DAC delivery. Thesis prediction: it survives,
# i.e. even neural re-encoding is not a reliable defense vs an adaptive white-box
# adversary — consistent with (and extending) the paper's central claim.
#
# Single 7B model + DAC fits a 40GB A100 (no ensemble); a100 (any) is fine.
# Add-only: writes to results_codec_robust_dac/ (never collides with headline/
# seed-sweep/Q5 bundles). Resumable.
#
#   sbatch --export=SEED=1 --job-name=dac_eot_s1 sbatch_dac_eot.sh
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

SEED="${SEED:-1}"
EPS=1.0
SCEN=01_finance_voice_agent
BUNDLE="0_all_combined/results_codec_robust_dac/${SCEN}/qwen2_audio/eps_${EPS}_seed${SEED}"

echo "=========================================="
echo "  DAC-EoT adaptive attack  SEED=$SEED"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST"
python -c 'import torch;print("  gpu",torch.cuda.get_device_name(0),round(torch.cuda.get_device_properties(0).total_memory/1e9,1),"GB")'
echo "  bundle: $BUNDLE"
echo "=========================================="

echo "=== STAGE 1: latent attack with DAC neural-codec EoT (eps=$EPS seed=$SEED) ==="
python -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen2_audio --eps "$EPS" --seed "$SEED" --rand-init \
    --train-codec dac

echo "=== STAGE 2 (KEY): does DAC-EoT SURVIVE DAC delivery? (adaptive vs defense) ==="
python -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio --codecs dac,encodec

echo "=== STAGE 3: does it also still survive traditional codecs? (MP3/AAC sidecar) ==="
python -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio || echo "  (sidecar optional; skipped/failed)"

echo "=== DONE  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
