#!/bin/bash
# ============================================================================
# Submit all missing P0 experiments for paper.
#
# Experiments:
#   Job 1: Watermark codec_roundtrip (eps=0, clean encode-decode) — lightweight, no attack
#   Job 2: Watermark attack_detection at eps=0.3 (latent, christmas_jazz_1, saycan)
#   Job 3: Watermark attack_detection at eps=0.5 (latent, christmas_jazz_1, saycan)
#   Job 4: Waveform baseline QA mode (empire_state, qwen2_audio, saycan)
#   Job 5: Waveform baseline transcribe (calm_1, qwen2_audio, saycan)
#   Job 6: Waveform baseline transcribe (classical_music_1, qwen2_audio, saycan)
#
# Usage:
#   bash sbatch_missing_exps.sh          # submit all 6 jobs
#   bash sbatch_missing_exps.sh --dry    # print commands without submitting
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

submit() {
    local name="$1"
    shift
    if $DRY_RUN; then
        echo "[DRY] sbatch $@"
    else
        echo "Submitting: $name"
        sbatch "$@"
    fi
}

# ---------- Job 1: Clean codec roundtrip watermark (no GPU attack, just encode-decode) ----------
cat > /tmp/sbatch_wm_roundtrip.sh << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=wm-roundtrip
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/wm_roundtrip_%j.out
#SBATCH --error=logs/wm_roundtrip_%j.err

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
cd ${REPO_ROOT}

python run_watermark_experiments.py \
    --experiment codec_roundtrip \
    --music christmas_jazz_1 calm_1 classical_music_1 empire_state_of_mind
HEREDOC
submit "Watermark codec roundtrip (eps=0)" /tmp/sbatch_wm_roundtrip.sh

# ---------- Job 2: Watermark attack_detection at eps=0.3 ----------
cat > /tmp/sbatch_wm_eps03.sh << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=wm-eps03
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/wm_eps03_%j.out
#SBATCH --error=logs/wm_eps03_%j.err

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
cd ${REPO_ROOT}

python run_watermark_experiments.py \
    --experiment attack_detection \
    --music christmas_jazz_1 \
    --dataset saycan \
    --eps 0.3 \
    --steps 150
HEREDOC
submit "Watermark latent eps=0.3" /tmp/sbatch_wm_eps03.sh

# ---------- Job 3: Watermark attack_detection at eps=0.5 ----------
cat > /tmp/sbatch_wm_eps05.sh << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=wm-eps05
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/wm_eps05_%j.out
#SBATCH --error=logs/wm_eps05_%j.err

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
cd ${REPO_ROOT}

python run_watermark_experiments.py \
    --experiment attack_detection \
    --music christmas_jazz_1 \
    --dataset saycan \
    --eps 0.5 \
    --steps 150
HEREDOC
submit "Watermark latent eps=0.5" /tmp/sbatch_wm_eps05.sh

# ---------- Job 4: Waveform QA baseline (empire_state, qwen2_audio, saycan) ----------
cat > /tmp/sbatch_wf_qa.sh << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=wf-qa
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/wf_qa_%j.out
#SBATCH --error=logs/wf_qa_%j.err

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
cd ${REPO_ROOT}

python run_waveform_baseline.py \
    --prompt-mode qa \
    --music empire_state_of_mind \
    --eps 0.01 \
    --steps 300 \
    --target-model qwen2_audio \
    --dataset saycan
HEREDOC
submit "Waveform QA baseline (empire_state)" /tmp/sbatch_wf_qa.sh

# ---------- Job 5: Waveform transcribe baseline (calm_1) ----------
cat > /tmp/sbatch_wf_calm.sh << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=wf-calm
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/wf_calm_%j.out
#SBATCH --error=logs/wf_calm_%j.err

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
cd ${REPO_ROOT}

python run_waveform_baseline.py \
    --music calm_1 \
    --eps 0.01 \
    --steps 300 \
    --target-model qwen2_audio \
    --dataset saycan
HEREDOC
submit "Waveform transcribe baseline (calm_1)" /tmp/sbatch_wf_calm.sh

# ---------- Job 6: Waveform transcribe baseline (classical_music_1) ----------
cat > /tmp/sbatch_wf_classical.sh << 'HEREDOC'
#!/bin/bash
#SBATCH --job-name=wf-classical
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/wf_classical_%j.out
#SBATCH --error=logs/wf_classical_%j.err

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
cd ${REPO_ROOT}

python run_waveform_baseline.py \
    --music classical_music_1 \
    --eps 0.01 \
    --steps 300 \
    --target-model qwen2_audio \
    --dataset saycan
HEREDOC
submit "Waveform transcribe baseline (classical_music_1)" /tmp/sbatch_wf_classical.sh

echo ""
echo "============================================"
echo "  Submitted 6 jobs"
echo "  Track: squeue -u \$USER"
echo "  Logs:  tail -f logs/wm_*.out logs/wf_*.out"
echo "============================================"
