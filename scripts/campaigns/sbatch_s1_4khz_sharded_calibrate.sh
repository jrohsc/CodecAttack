#!/bin/bash
# Merge 6 preserved + 34 resumed probe rows and freeze 40 evaluation budgets.
#SBATCH --job-name=s1_4k_cal
#SBATCH --partition=cpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1"
F="$OUT/frozen"
BASE="$OUT/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_probe/manifest.jsonl"
COMB="$F/bandlimited_4khz.evaluation.sharded_probe.jsonl"
cd "$ROOT"; mkdir -p "$F" "$REPO/logs"; export PYTHONPATH="$ROOT"
cp "$BASE" "$COMB"
for s in 0 1 2 3; do
  cat "$OUT/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_sharded_probe_${s}/manifest.jsonl" >> "$COMB"
done
"$PY" - "$COMB" <<'PY'
import json,sys
r=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; ids=[x['example_id'] for x in r]
assert len(r)==len(set(ids))==40, (len(r),len(set(ids)))
PY
"$PY" -m reviewer_controls.calibrate_epsilon --probe "$COMB" \
  --reference reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl --mode per_example \
  --output "$F/epsilon_bandlimited_4khz.evaluation.sharded.json"

