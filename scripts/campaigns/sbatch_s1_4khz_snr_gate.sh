#!/bin/bash
# Distortion-only gate for arm D. Re-attack only examples outside +/-1 dB.
# No model output, match, or ASR is read when deciding corrections.
#SBATCH --job-name=s1_4k_gate
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1"
REF="$ROOT/reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl"
FROZEN="$OUT/frozen"
CANON="$FROZEN/bandlimited_4khz.evaluation.canonical.jsonl"
cd "$ROOT"
mkdir -p "$FROZEN" "$REPO/logs"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_4k_gate_${SLURM_JOB_ID:-local_$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
: > "$CANON"
for shard in 0 1 2 3; do
  cat "$OUT/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_sharded_matched_${shard}/manifest.jsonl" >> "$CANON"
done
"$CA" - "$CANON" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
ids = [r["example_id"] for r in rows]
assert len(rows) == len(set(ids)) == 40, (len(rows), len(set(ids)))
PY

for round in 2 3; do
  ids="$FROZEN/bandlimited_4khz.evaluation.bad_r${round}.txt"
  epsmap="$FROZEN/epsilon_bandlimited_4khz.evaluation.r${round}.json"
  "$CA" - "$CANON" "$REF" "$ids" "$epsmap" <<'PY'
import json, math, sys
rows = {r["example_id"]: r for r in (json.loads(x) for x in open(sys.argv[1]) if x.strip())}
ref = {r["example_id"]: r for r in (json.loads(x) for x in open(sys.argv[2]) if x.strip())}
assert len(rows) == 40 and len(set(rows) & set(ref)) == 40
bad, emap, residuals = [], {}, []
for key, row in rows.items():
    gap = float(row["snr_db"]) - float(ref[key]["snr_db"])
    residuals.append({"example_id": key, "difference_db": gap})
    if abs(gap) > 1.0:
        bad.append(key)
        emap[key] = float(row["epsilon"]) * 10.0 ** (gap / 20.0)
open(sys.argv[3], "w").write("".join(x + "\n" for x in sorted(bad)))
json.dump({"status": "frozen", "mode": "per_example_refinement",
           "selection_input": "realized SNR vs reference SNR only; no attack outcome used",
           "n_examples": len(bad), "epsilon_map": emap,
           "input_residuals": residuals}, open(sys.argv[4], "w"), indent=2)
print(f"round candidate: {len(bad)}/40 outside 1 dB")
PY
  nbad=$(wc -l < "$ids")
  if [[ "$nbad" -eq 0 ]]; then break; fi
  "$CA" -u -m reviewer_controls.run_attack \
    --scenario s1 --split evaluation --method bandlimited_4khz \
    --epsilon 0.166 --alpha-ratio 0.02 --steps 1000 --seed 1 \
    --epsilon-map "$epsmap" --example-ids-file "$ids" \
    --run-tag "snr_correction_r${round}" --resume --output-root "$OUT"
  corr="$OUT/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_snr_correction_r${round}/manifest.jsonl"
  "$CA" - "$CANON" "$corr" <<'PY'
import json, sys
base = {r["example_id"]: r for r in (json.loads(x) for x in open(sys.argv[1]) if x.strip())}
corr = {r["example_id"]: r for r in (json.loads(x) for x in open(sys.argv[2]) if x.strip())}
base.update(corr)
assert len(base) == 40
with open(sys.argv[1], "w") as f:
    for key in sorted(base): f.write(json.dumps(base[key]) + "\n")
PY
done

"$CA" - "$CANON" "$REF" "$FROZEN/bandlimited_4khz.evaluation.verified.json" <<'PY'
import json, sys
rows = {r["example_id"]: r for r in (json.loads(x) for x in open(sys.argv[1]) if x.strip())}
ref = {r["example_id"]: r for r in (json.loads(x) for x in open(sys.argv[2]) if x.strip())}
diffs = [{"example_id": k, "difference_db": float(v["snr_db"])-float(ref[k]["snr_db"]),
          "epsilon": float(v["epsilon"])} for k, v in rows.items()]
payload = {"n": len(diffs), "n_within_1db": sum(abs(x["difference_db"]) <= 1 for x in diffs),
           "max_abs_db": max(abs(x["difference_db"]) for x in diffs),
           "mean_db": sum(x["difference_db"] for x in diffs)/len(diffs),
           "per_example": sorted(diffs, key=lambda x: x["example_id"])}
json.dump(payload, open(sys.argv[3], "w"), indent=2)
print(json.dumps({k:v for k,v in payload.items() if k != "per_example"}, indent=2))
assert payload["n"] == payload["n_within_1db"] == 40, "SNR gate failed after two refinement rounds"
PY
