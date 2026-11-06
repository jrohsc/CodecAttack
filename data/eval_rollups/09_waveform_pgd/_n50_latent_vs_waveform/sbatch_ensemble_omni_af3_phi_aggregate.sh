#!/bin/bash
#SBATCH --job-name=e3phi1agg
#SBATCH --partition=cpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd ${REPO_ROOT}
SEED="${SEED:-1}"
[[ "$SEED" =~ ^(1|7|42)$ ]] || exit 2
PREFIX="benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed${SEED}"
EXTRA=()
[[ "$SEED" == 1 ]] || EXTRA=(--require-seeded-codec)
for SPEC in 'omni7b_MEMBER qwen25_omni' 'af3_MEMBER audio_flamingo' \
            'phi4_MEMBER phi4' 'qwen2_HELDOUT qwen2_audio' \
            'kimi_HELDOUT_100tok kimi_audio' 'omni3b_HELDOUT qwen25_omni_3b'; do
  set -- $SPEC
  python 0_all_combined/n50_latent_vs_waveform/aggregate_extra_heldout.py \
    --prefix "$PREFIX" --out-name "$1" --model "$2" \
    --expected-seed "$SEED" --expected-init zero "${EXTRA[@]}"
done
python 0_all_combined/n50_latent_vs_waveform/summarize_ensemble_omni_af3_phi.py --seed "$SEED"
