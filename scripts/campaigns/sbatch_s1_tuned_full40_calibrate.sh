#!/bin/bash
# Merge the four arm-C probe manifests and freeze per-example distortion budgets.
# No attack outcome or held-out model output is read by this job.
#SBATCH --job-name=s1_tuned40_cal
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
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_tuned_full40_seed1"
COMBINED="$OUT/frozen/tuned_waveform.evaluation.probe.manifest.jsonl"
REF="$ROOT/reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl"
cd "$ROOT"
mkdir -p "$OUT/frozen" "$REPO/logs"
export PYTHONPATH="$ROOT"

: > "$COMBINED"
for shard in 0 1 2 3; do
  src="$OUT/s1/evaluation/tuned_waveform/eps0.06082_ar0.02_steps1000_seed1_shard${shard}of4_probe/manifest.jsonl"
  [[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }
  cat "$src" >> "$COMBINED"
done

"$CA" - "$COMBINED" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
ids = [r["example_id"] for r in rows]
assert len(rows) == 40, f"expected 40 probe rows, got {len(rows)}"
assert len(set(ids)) == 40, "duplicate probe example IDs"
assert {r["split"] for r in rows} == {"evaluation"}
assert {r["alpha_over_epsilon"] for r in rows} == {0.02}
PY

"$CA" -m reviewer_controls.calibrate_epsilon \
  --probe "$COMBINED" --reference "$REF" --mode per_example \
  --output "$OUT/frozen/epsilon_tuned_waveform.evaluation.json"
echo "=== [$(date -u +%FT%TZ)] frozen 40 per-example distortion-only budgets"

