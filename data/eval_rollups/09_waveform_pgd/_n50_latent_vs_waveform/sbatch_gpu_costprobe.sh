#!/bin/bash
# ============================================================================
# GPU cost probe (CvHX Q4) — does CodecAttack fit and run on a 48 GB L40S,
# and what does one carrier cost?
#
# Runs the HEADLINE config unchanged (Qwen2.5-Omni S1, eps=1.0, seed 1,
# rand-init, multi-bitrate Opus EoT, 1000 steps) on 3 fixed carriers, so the
# per-clip seconds are directly comparable to the 4,206 clips already on disk.
#
# Output is redirected via CODEC_ATTACK_RESULTS_OVERRIDE to a DEDICATED tree
#   0_all_combined/results_gpu_costprobe/
# so it can never resume into, clobber, or contaminate the paper bundles or
# the results_codec_robust* audit trees. Stage 2 (MP3/AAC sidecar) is skipped
# on purpose — this measures attack cost only.
#
# Submit:
#   sbatch --job-name=cost_l40s --export=ALL,GPUKIND=l40s sbatch_gpu_costprobe.sh
#   sbatch --job-name=cost_a100 --export=ALL,GPUKIND=a100 sbatch_gpu_costprobe.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

GPUKIND="${GPUKIND:?set GPUKIND=l40s|a100}"

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Dedicated output tree — never collides with paper/audit bundles.
export CODEC_ATTACK_RESULTS_OVERRIDE="$REPO/0_all_combined/results_gpu_costprobe/${GPUKIND}"

CARRIERS=(
  banking_christopher_q2_rates_edgetts.wav
  banking_michelle_q2_rates_edgetts.wav
  banking_eric_q2_rates_edgetts.wav
)

echo "=========================================="
echo "  GPU COST PROBE   GPUKIND=$GPUKIND"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "  out: $CODEC_ATTACK_RESULTS_OVERRIDE"
echo "=========================================="
$OMNI -c 'import torch;p=torch.cuda.get_device_properties(0);print(f"  torch {torch.__version__} | gpu {p.name} | vram {p.total_memory/1e9:.1f} GB | sm_{p.major}{p.minor}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== attack: Omni S1 latent eps=1.0 seed=1 rand-init multi_bitrate, 3 carriers, 1000 steps ==="
START=$(date +%s)
$OMNI -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen25_omni --eps 1.0 --seed 1 --rand-init \
    --channel-mode multi_bitrate \
    --carriers "${CARRIERS[@]}"
RC=$?
END=$(date +%s)

echo "=== DONE  rc=$RC  wall=$(( (END-START)/60 )) min ==="
echo "--- peak GPU memory actually used ---"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
B="$CODEC_ATTACK_RESULTS_OVERRIDE/01_finance_voice_agent/qwen25_omni/eps_1.0_seed1"
ls "$B/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
$OMNI - <<PY
import json,glob,statistics as st
fs=glob.glob("$B/audio/*_summary.json")
e=[json.load(open(f)).get("elapsed_s") for f in fs]
e=[x for x in e if x]
print(f"  per-clip elapsed_s: n={len(e)} " + (f"mean={st.mean(e):.1f} median={st.median(e):.1f} vals={[round(x,1) for x in e]}" if e else "NONE"))
PY
