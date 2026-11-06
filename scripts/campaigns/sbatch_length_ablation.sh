#!/bin/bash
# ============================================================================
# SLURM batch script for target text length ablation
#
# Tests how attack success varies with target command length (word count).
# Uses a single music carrier and eps, varying only the target text.
#
# Usage:
#   sbatch sbatch_length_ablation.sh
#   sbatch --export=EPS=1.0 sbatch_length_ablation.sh
#   sbatch --export=EPS=0.3,MUSIC=christmas_jazz_1 sbatch_length_ablation.sh
# ============================================================================

#SBATCH --job-name=len-ablation
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
EPS="${EPS:-0.5}"
STEPS="${STEPS:-300}"
REPEATS="${REPEATS:-3}"

# ---- Setup ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "============================================"
echo "  Length Ablation"
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
python run_length_ablation.py \
    --music "$MUSIC" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --target-model "$MODEL" \
    --repeats "$REPEATS"

echo ""
echo "============================================"
echo "  FINISHED: Length ablation $MODEL eps=$EPS $MUSIC"
echo "============================================"
