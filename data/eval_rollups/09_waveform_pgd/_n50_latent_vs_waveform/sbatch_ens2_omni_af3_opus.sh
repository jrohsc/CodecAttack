#!/bin/bash
# ============================================================================
# 2-model ensemble {qwen25_omni (primary) + audio_flamingo}, multi-bitrate
# Opus EoT, eps=10, transcribe prompt, jazz_1 carrier, 224 robot commands.
# Craft -> targets.json -> score on all codecs for members AND held-out models.
#
#   members : qwen25_omni, audio_flamingo
#   held out: qwen2_audio (cross-arch), qwen25_omni_3b (same family as Omni-7B)
#   (no Kimi -- out of the project)
#
# --ensemble-best-after-warmup: saved wav may only come from steps where both
# members are in the loss AND Opus EoT is on (fixes the warmup latch).
#
#   sbatch --job-name=e2oa_s1  --export=ALL,SEED=1  sbatch_ens2_omni_af3_opus.sh
#   sbatch --job-name=e2oa_s7  --export=ALL,SEED=7  sbatch_ens2_omni_af3_opus.sh
#   sbatch --job-name=e2oa_s42 --export=ALL,SEED=42 sbatch_ens2_omni_af3_opus.sh
# Smoke (local): SMOKE=1 SEED=1 bash sbatch_ens2_omni_af3_opus.sh
# ============================================================================
#SBATCH --partition=gpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
SEED="${SEED:?set SEED}"
SMOKE="${SMOKE:-0}"

REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
EVAL=0_all_combined/scripts/ensemble_heldout_eval.py
LOSSY=opus64,opus128,mp3_64,mp3_128,aac_64,aac_128
ALL=$LOSSY,dac,encodec

export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The bundle sidecars are written with ensure_ascii=False, so a clip whose
# victim output contains non-ASCII text is raw UTF-8 on disk. Without this,
# a compute node with an ASCII default encoding makes every python here read
# those clips as UnicodeDecodeError -- which the eval used to swallow, silently
# dropping the clip from n (2026-09-17: n=47/49 instead of 50).
export PYTHONUTF8=1

if [ "$SMOKE" = 1 ]; then
    OUT="results/benchmark_ens2_omni_af3_opuseot_transcribe_eps10.0_jazz_1_seed${SEED}_SMOKE"
    EXTRA=(--commands comm_acknowledge nav_detour --steps 80 --ensemble-warmup 20)
    EXPECT=2
else
    OUT="results/benchmark_ens2_omni_af3_opuseot_transcribe_eps10.0_jazz_1_seed${SEED}"
    EXTRA=()
    EXPECT=224
fi
CRAFT_LOG="$OUT.craft.log"   # one appended log per bundle, survives requeue
STAGE="${STAGE:-all}"        # all | craft | eval  (craft ~16h + eval ~5h > safe in one 24h job)
echo "=== [$(date)] seed=$SEED smoke=$SMOKE stage=$STAGE out=$OUT host=$(hostname) job=${SLURM_JOB_ID:-local}"

# ---- 1. craft ------------------------------------------------------------
if [ "$STAGE" != eval ]; then
python -u run_benchmark.py --ensemble \
    --ensemble-models qwen25_omni,audio_flamingo \
    --target-model qwen25_omni \
    --ensemble-channel-mode multi_bitrate \
    --ensemble-best-after-warmup \
    --rand-init --seed "$SEED" \
    --music jazz_1 --output-dir "$OUT" "${EXTRA[@]}" 2>&1 | tee -a "$CRAFT_LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: craft exited non-zero" >&2; exit 1; }
fi

N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "crafted wavs: $N/$EXPECT"
if grep -q "subprocess gradient requests FAILED" "$CRAFT_LOG"; then
    echo "ERROR: AF3 gradient failures -- do not report this bundle" >&2; exit 1
fi
$CA -c "import json,sys; c=json.load(open('$OUT/config.json')); sys.exit(0 if c.get('ensemble_best_after_warmup') and c.get('ensemble_channel_mode')=='multi_bitrate' and c.get('ensemble_models')==['qwen25_omni','audio_flamingo'] else 1)" \
    || { echo "ERROR: bundle config is not the fixed-selection Omni+AF3 Opus-EoT arm" >&2; exit 1; }
(( N == EXPECT )) || { echo "incomplete craft ($N/$EXPECT); requeue resumes" >&2; exit 1; }
[ "$STAGE" = craft ] && { echo "=== [$(date)] craft stage DONE seed=$SEED"; exit 0; }

# ---- 2. targets.json (Omni is primary -> no qwen2_audio/ record dir) -------
$CA - "$OUT" <<'PY' || exit 1
import json, sys, pathlib
b = pathlib.Path(sys.argv[1]); cfg = json.load(open(b / "config.json"))
wavs = sorted(p.stem for p in (b / "audio").glob("*.wav"))
t = {}
for s in wavs:
    cmd = s.split("__", 1)[1]
    t[s] = {"target_text": cfg["targets"][cmd], "command_name": cmd}
json.dump(t, open(b / "targets.json", "w"), indent=1)
print(f"targets.json: {len(t)}/{len(wavs)} wavs, prompt={cfg['prompt_text']!r}")
PY

# ---- 3. score: members + held-out, all codecs ------------------------------
FAIL=0
ev() {  # python model out-name channels
    echo; echo "=== [$(date)] eval $2 -> $3 ($4)"
    "$1" -u "$EVAL" --bundle-dir "$OUT" --target-model "$2" --out-name "$3" \
        --channels "$4" 2>&1 | grep -vE "Loading weights|it/s\]|^Loading checkpoint"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && { echo "!!! FAILED: $2 (exit $rc)"; FAIL=1; }
    return 0
}
ev "$OMNI" qwen25_omni    omni7b_MEMBER_allcodecs  "$ALL"
ev "$AF3"  audio_flamingo af3_MEMBER_allcodecs     "$LOSSY"
ev "$CA"   qwen2_audio    qwen2_HELDOUT_allcodecs  "$ALL"
ev "$OMNI" qwen25_omni_3b omni3b_HELDOUT_allcodecs "$ALL"
echo "=== [$(date)] DONE seed=$SEED fail=$FAIL"
exit $FAIL
