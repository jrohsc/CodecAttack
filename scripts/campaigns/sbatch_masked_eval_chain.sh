#!/bin/bash
# Chain job: transfer-eval the masked arms once their attack array finishes.
#
# Exists because the eval fan-out is a bash script that calls sbatch, and it has
# to run AFTER the attacks write manifest.json -- submit_encoder_transfer_evals.sh
# deliberately SKIPs an arm with no manifest, so submitting both up front would
# silently evaluate nothing. Submitted with
# --dependency=afterok:<attack array id> so it fires exactly once, after both
# array tasks succeed.
#
# Usage: sbatch --dependency=afterok:<jobid> sbatch_masked_eval_chain.sh
#
#SBATCH --job-name=enc_mask_chain
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o ${REPO_ROOT}/logs/enc_mask_chain_%j.log

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack
cd ${REPO_ROOT}

echo "=== $(date) fanning out transfer evals for the masked arms ==="
bash submit_encoder_transfer_evals.sh enc_mask_match encce_mask_match
echo "=== $(date) chain done ==="
