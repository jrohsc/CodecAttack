#!/bin/bash
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
SCRIPT=0_all_combined/n50_latent_vs_waveform/aggregate_ensemble3_eot.py
BASE=benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600_zeroeot
for SEED in 7 42; do
  python "$SCRIPT" opus --prefix "${BASE}_seed${SEED}" --expected-steps 600 \
    --require-best-after-warmup --expected-seed "$SEED" --expected-init zero \
    --require-seeded-codec
  python 0_all_combined/n50_latent_vs_waveform/aggregate_extra_heldout.py \
    --prefix "${BASE}_seed${SEED}" --out-name kimi_HELDOUT_100tok \
    --model kimi_audio --expected-seed "$SEED" --expected-init zero \
    --require-seeded-codec
  python 0_all_combined/n50_latent_vs_waveform/aggregate_extra_heldout.py \
    --prefix "${BASE}_seed${SEED}" --out-name phi4_HELDOUT \
    --model phi4 --expected-seed "$SEED" --expected-init zero \
    --require-seeded-codec
done
python 0_all_combined/n50_latent_vs_waveform/aggregate_ensemble3_seed_replicates.py
