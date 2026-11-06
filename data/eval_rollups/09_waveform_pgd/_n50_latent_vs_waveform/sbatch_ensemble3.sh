#!/bin/bash
# ============================================================================
# SLURM job: THREE-MODEL ENSEMBLE-VICTIM attack (AC Q1 follow-up).
#
#   ensemble = qwen2_audio + qwen25_omni(7B) + audio_flamingo
#   HELD OUT = qwen25_omni_3b
#
# WHY THIS DESIGN. With all three headline victims in the loss there is no
# headline model left to hold out, so the held-out slot goes to Qwen2.5-Omni-3B.
# That is not a consolation prize: the 2-model ensemble already transfers to
# Omni-3B at 2/180 (1.1%) while transferring 0/180 to every cross-architecture
# model, so Omni-3B is the ONLY victim with a nonzero baseline to move. The
# question is therefore sharp: does adding a third, architecturally distinct
# victim push same-family transfer above 1.1%?
#
# AF3 runs OUT OF ENV (transformers>=5) via gradient_worker_persistent.py, which
# loads it once and serves gradients over a line protocol (~0.33 s/step measured).
# ENSEMBLE_PERSISTENT_WORKER=1 selects that path; without it ensemble_attack.py
# falls back to spawning + reloading AF3 every step (~46 s => ~600 GPU-h).
#
# Two bugs had to be fixed before this could work at all -- see
# gradient_worker_persistent.py for both:
#   1. codec_attack/models/ shadowed the whisper-inject `models` package, so
#      `models.audio_flamingo` never imported in the old worker;
#   2. the AF3 wrapper prints to stdout, corrupting the line protocol.
# ensemble_attack.py now also COUNTS failed subprocess gradients and warns
# loudly, so a run where AF3 silently contributed nothing cannot be mistaken
# for a real 3-model ensemble. CHECK THAT WARNING before reporting results.
#
# COST: ~240 s/clip (157 s for the 2 in-process victims + ~83 s for AF3)
#       x 180 clips ~= 12 h. Bundles are resumable, --requeue is set, and the
#       time limit is 24 h so one preemption does not lose the run.
# MEMORY: ~45 GB (2x7B + backward) + ~16 GB (AF3 worker) ~= 61 GB -> a100-80g.
#       Set ENSEMBLE_WORKER_GPU=1 with --gres=gpu:2 to give AF3 its own card.
#
# OUTPUT DIR IS EXPLICIT AND MUST STAY THAT WAY. run_benchmark.py derives the
# bundle name from {mode, eps, music} only -- NOT from ensemble membership -- so
# without --output-dir this job would resume INTO the existing 2-model bundle,
# find all 180 clips already done, and exit having computed nothing.
#
#   sbatch --job-name=ens3 sbatch_ensemble3.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-80g
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

OUT=results/benchmark_ensemble3_transcribe_eps10.0_jazz_1

echo "=========================================="
echo "  3-MODEL ENSEMBLE-VICTIM  (Q1 follow-up)"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
python -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0),"| mem",round(torch.cuda.get_device_properties(0).total_memory/1e9,1),"GB")'
echo "  ensemble = qwen2_audio + qwen25_omni + audio_flamingo"
echo "  HELD OUT = qwen25_omni_3b   (2-model baseline: 2/180 = 1.1%)"
echo "  output   = $OUT"
echo "=========================================="

export ENSEMBLE_PERSISTENT_WORKER=1
# AF3 gets its own card. Measured failure mode on a single 44 GB card: the parent
# (in-process victims + caching allocator after backward) held 38 GB and the AF3
# worker OOM'd during weight load, silently degrading the run to a 2-model attack.
# The attacker now pre-warms workers and hard-fails instead, but a dedicated card
# removes the contention entirely.
export ENSEMBLE_WORKER_GPU=1

# SMOKE=1 -> 2 commands, 40 steps: validates that audio_flamingo actually appears
# as a loss term in the stitched gradient before committing ~12 h.
EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="--commands comm_alert comm_eta --steps 40 --ensemble-warmup 10"
    OUT="${OUT}_SMOKE"
    echo "  *** SMOKE MODE: $EXTRA -> $OUT"
fi

python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --music jazz_1 \
    --output-dir "$OUT" $EXTRA

echo
echo "=== did AF3 actually participate? (step lines must show an audio_flamingo term) ==="
grep -E "^Step +[0-9]+/.*\[ensemble\]" "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -5

echo "=== attack done. crafted wavs: ==="
find "$OUT/audio" -name "*.wav" 2>/dev/null | wc -l
echo
echo "!!! BEFORE REPORTING: grep the log for 'subprocess gradient requests FAILED'."
echo "!!! A nonzero count means AF3 did not fully participate and this is NOT a"
echo "!!! 3-model ensemble result."
echo
echo "NEXT: score held-out + members with ensemble_heldout_eval.py, e.g."
echo "  qwen-omni/bin/python 0_all_combined/scripts/ensemble_heldout_eval.py \\"
echo "     --bundle-dir $OUT --target-model qwen25_omni_3b \\"
echo "     --out-name omni3b_HELDOUT --channels opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"
echo "=== DONE ==="
