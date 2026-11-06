#!/bin/bash
# SLURM job: one Qwen2.5-Omni S3b (copyright bypass) multi-seed bundle,
# eps=1.0, multi_bitrate EoT, random-init + MP3/AAC sidecar. Resumable.
#   sbatch --export=ALL,SEED=7  --job-name=omni_s3b_s7  sbatch_omni_s3b.sh
#   sbatch --export=ALL,SEED=42 --job-name=omni_s3b_s42 sbatch_omni_s3b.sh
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED="${SEED:?set SEED}"
BUNDLE=0_all_combined/results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed${SEED}
PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"
echo "=== omni S3b seed=$SEED  job $SLURM_JOB_ID on $SLURM_NODELIST GPU=$CUDA_VISIBLE_DEVICES ==="
$OMNI -c 'import torch;print("torch",torch.__version__,"gpu",torch.cuda.get_device_name(0))'
echo "=== ATTACK ==="
$OMNI -u 0_all_combined/scripts/run_s3_attack.py \
    --sub-scenario 03b --target-model qwen25_omni --eps 1.0 \
    --channel-mode multi_bitrate --seed "$SEED" --rand-init --skip-eval
echo "=== SIDECAR ==="
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni --prompt-text "$PROMPT"
echo "=== DONE seed=$SEED ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
