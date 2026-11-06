#!/bin/bash
# ============================================================================
# SLURM batch script for music duration ablation
#
# Tests how attack success varies with carrier audio length (5s, 10s, 20s, 30s).
# Longer carriers provide more EnCodec latent frames for perturbation.
#
# Usage:
#   sbatch sbatch_duration_ablation.sh
#   sbatch --export=EPS=0.5 sbatch_duration_ablation.sh
#   sbatch --export=EPS=1.0,MUSIC=jazz_1 sbatch_duration_ablation.sh
# ============================================================================

#SBATCH --job-name=dur-ablation
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- Defaults (override via --export) ----
MODEL="${MODEL:-qwen2_audio}"
MUSIC="${MUSIC:-empire_state_of_mind}"
EPS="${EPS:-1.0}"
STEPS="${STEPS:-300}"
REPEATS="${REPEATS:-1}"

# ---- Setup ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "============================================"
echo "  Duration Ablation"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Model:     $MODEL"
echo "  Music:     $MUSIC"
echo "  EPS:       $EPS"
echo "  Steps:     $STEPS"
echo "  Repeats:   $REPEATS"
echo "============================================"

# ---- Activate conda ----
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

echo "Python: $(which python)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# ---- Run ----
python run_duration_ablation.py \
    --music "$MUSIC" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --target-model "$MODEL" \
    --repeats "$REPEATS"

echo ""
echo "============================================"
echo "  FINISHED: Duration ablation $MODEL eps=$EPS $MUSIC"
echo "============================================"
