#!/bin/bash
# Score the frozen 40-example S1 held-out comparison after arms C and D finish.
# A/B reuse existing artifacts; C/D use their one fixed selected waveform for
# every channel. This job does not tune or select anything.
#SBATCH --job-name=s1_base40_eval
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
COUT="$ROOT/reviewer_control_outputs/s1_tuned_full40_seed1"
DOUT="$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1"
FINAL="$ROOT/reviewer_control_outputs/s1_baseline_full40_seed1"
REF="$ROOT/reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl"

cd "$ROOT"
mkdir -p "$FINAL/manifests" "$FINAL/eval" "$FINAL/frozen" "$REPO/logs"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_base40_eval_${SLURM_JOB_ID:-local_$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"

C_MAN="$COUT/frozen/tuned_waveform.evaluation.canonical.jsonl"
D_MAN="$DOUT/frozen/bandlimited_4khz.evaluation.canonical.jsonl"
[[ -f "$C_MAN" && -f "$D_MAN" ]] || { echo "SNR-gated C/D manifests are incomplete" >&2; exit 1; }

"$CA" - "$C_MAN" "$D_MAN" <<'PY'
import json, sys
sets = []
for path, method in zip(sys.argv[1:], ("tuned_waveform", "bandlimited_4khz")):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    ids = [r["example_id"] for r in rows]
    assert len(rows) == len(set(ids)) == 40, (path, len(rows), len(set(ids)))
    assert {r["split"] for r in rows} == {"evaluation"}
    assert {r["method"] for r in rows} == {method}
    assert {r["alpha_over_epsilon"] for r in rows} == {0.02}
    sets.append(set(ids))
assert sets[0] == sets[1], "C/D held-out example IDs differ"
PY

"$CA" -m reviewer_controls.measure_reference_snr --scenario s1 --split evaluation \
  --method latent_codecattack \
  --bundle results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0 \
  --output "$FINAL/manifests/armA_latent.jsonl"
"$CA" -m reviewer_controls.measure_reference_snr --scenario s1 --split evaluation \
  --method original_waveform_published \
  --bundle results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.06082_norm_seed1 \
  --output "$FINAL/manifests/armB_original_waveform.jsonl"

"$CA" - "$FINAL/manifests/armA_latent.jsonl" \
        "$FINAL/manifests/armB_original_waveform.jsonl" "$C_MAN" "$D_MAN" <<'PY'
import json, sys
sets = []
for path in sys.argv[1:]:
    rows = [json.loads(x) for x in open(path) if x.strip()]
    ids = [r["example_id"] for r in rows]
    assert len(rows) == len(set(ids)) == 40, (path, len(rows), len(set(ids)))
    assert {r["split"] for r in rows} == {"evaluation"}
    sets.append(set(ids))
assert all(ids == sets[0] for ids in sets[1:]), "A/B/C/D held-out IDs differ"
PY

echo "=== [$(date -u +%FT%TZ)] scoring A/B/C/D on clean, Opus 16/32/64/128, held-out MP3 64 and AAC-LC 64"
"$CA" -u -m reviewer_controls.evaluate \
  --manifest "$FINAL/manifests/armA_latent.jsonl" \
             "$FINAL/manifests/armB_original_waveform.jsonl" "$C_MAN" "$D_MAN" \
  --output "$FINAL/eval/arms_ABCD.jsonl" --resume

"$CA" -m reviewer_controls.build_report \
  --records "$FINAL/eval/arms_ABCD.jsonl" \
  --reference-method latent_codecattack \
  --output-json "$FINAL/baseline_report.json" \
  --output-markdown "$FINAL/baseline_report.md"
echo "=== [$(date -u +%FT%TZ)] complete -> $FINAL/baseline_report.md"
