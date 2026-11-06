#!/bin/bash
# ============================================================================
# 2-model ensemble PAIR x multi-bitrate Opus EoT x EPS, transcribe prompt,
# jazz_1 carrier. Craft -> targets.json -> score members AND held-out models.
#
#   PAIR=qo : qwen2_audio (primary) + qwen25_omni     held out: audio_flamingo, omni3b
#   PAIR=qa : qwen2_audio (primary) + audio_flamingo  held out: qwen25_omni,    omni3b
#   PAIR=oa : qwen25_omni (primary) + audio_flamingo  held out: qwen2_audio,    omni3b
#   (no Kimi -- out of the project)
#
# --ensemble-best-after-warmup: the saved wav may only come from steps where both
# members are in the loss AND Opus EoT is on (fixes the warmup latch).
#
# PILOT=1 -> 16 commands (every 14th of the 224, spans all categories), dir
# suffix _PILOT16. Otherwise the full 224.
#
#   local : PAIR=qa EPS=1.0 SEED=1 PILOT=1 bash sbatch_ens2_pair_opus_eps.sh
#   slurm : sbatch --job-name=e2p_qa_e1_s1 --export=ALL,PAIR=qa,EPS=1.0,SEED=1 sbatch_ens2_pair_opus_eps.sh
#
# Sibling of sbatch_ens2_omni_af3_opus.sh (eps=10, other session) -- different
# output dirs, never shares a bundle with it.
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
PAIR="${PAIR:?set PAIR=qo|qa|oa}"
EPS="${EPS:?set EPS}"
SEED="${SEED:?set SEED}"
PILOT="${PILOT:-0}"
STAGE="${STAGE:-all}"        # all | craft | eval

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

case "$PAIR" in
    qo) MODELS=qwen2_audio,qwen25_omni;    PRIMARY=qwen2_audio; NAME=qwen2_omni ;;
    qa) MODELS=qwen2_audio,audio_flamingo; PRIMARY=qwen2_audio; NAME=qwen2_af3 ;;
    oa) MODELS=qwen25_omni,audio_flamingo; PRIMARY=qwen25_omni; NAME=omni_af3 ;;
    *) echo "bad PAIR=$PAIR" >&2; exit 2 ;;
esac

export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The bundle sidecars are written with ensure_ascii=False, so a clip whose
# victim output contains non-ASCII text is raw UTF-8 on disk. Without this,
# a compute node with an ASCII default encoding makes every python here read
# those clips as UnicodeDecodeError -- which the eval used to swallow, silently
# dropping the clip from n (2026-09-17: n=47/49 instead of 50).
export PYTHONUTF8=1

OUT="results/benchmark_ens2_${NAME}_opuseot_transcribe_eps${EPS}_jazz_1_seed${SEED}"
if [ "$PILOT" = 1 ]; then
    OUT="${OUT}_PILOT16"
    CMDS=(nav_turn_left nav_merge_lane manip_press_btn manip_slide_panel loco_turn_180
          sense_scan sense_color comm_log_event safe_lower_alt safe_seal_hatch
          deliv_express insp_corrosion insp_fluid_level maint_replace_filter
          interact_follow_person interview_chinese_magicdata_f_q2_idx4)
    EXTRA=(--commands "${CMDS[@]}")
    EXPECT=16
else
    EXTRA=()
    EXPECT=224
fi
CRAFT_LOG="$OUT.craft.log"
echo "=== [$(date -u +%FT%TZ)] pair=$PAIR models=$MODELS eps=$EPS seed=$SEED pilot=$PILOT stage=$STAGE out=$OUT host=$(hostname) job=${SLURM_JOB_ID:-local}"

# ---- 1. craft ------------------------------------------------------------
if [ "$STAGE" != eval ]; then
python -u run_benchmark.py --ensemble \
    --ensemble-models "$MODELS" \
    --target-model "$PRIMARY" \
    --ensemble-channel-mode multi_bitrate \
    --ensemble-best-after-warmup \
    --eps "$EPS" \
    --rand-init --seed "$SEED" \
    --music jazz_1 --output-dir "$OUT" "${EXTRA[@]}" 2>&1 \
    | grep -v "WARNING:root:System prompt modified" | tee -a "$CRAFT_LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: craft exited non-zero" >&2; exit 1; }
fi

N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l)
echo "crafted wavs: $N/$EXPECT"
if grep -q "subprocess gradient requests FAILED" "$CRAFT_LOG" 2>/dev/null; then
    echo "ERROR: AF3 gradient failures -- do not report this bundle" >&2; exit 1
fi
$CA - "$OUT" "$MODELS" "$EPS" <<'PY' || { echo "ERROR: bundle config mismatch" >&2; exit 1; }
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
ok = (c.get("ensemble_best_after_warmup") and c.get("ensemble_channel_mode") == "multi_bitrate"
      and c.get("ensemble_models") == sys.argv[2].split(",")
      and abs(float(c.get("eps")) - float(sys.argv[3])) < 1e-9)
sys.exit(0 if ok else 1)
PY
(( N == EXPECT )) || { echo "incomplete craft ($N/$EXPECT); rerun resumes" >&2; exit 1; }
[ "$STAGE" = craft ] && { echo "=== [$(date -u +%FT%TZ)] craft stage DONE"; exit 0; }

# ---- 2. targets.json (primary-agnostic target resolver for the eval) ------
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
    echo; echo "=== [$(date -u +%FT%TZ)] eval $2 -> $3"
    "$1" -u "$EVAL" --bundle-dir "$OUT" --target-model "$2" --out-name "$3" \
        --channels "$4" 2>&1 | grep -vE "Loading weights|it/s\]|^Loading checkpoint|WARNING:root:System prompt"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && { echo "!!! FAILED: $2 (exit $rc)"; FAIL=1; }
    return 0
}
role() { [[ ",$MODELS," == *",$1,"* ]] && echo MEMBER || echo HELDOUT; }
ev "$CA"   qwen2_audio    "qwen2_$(role qwen2_audio)_allcodecs"    "$ALL"
ev "$OMNI" qwen25_omni    "omni7b_$(role qwen25_omni)_allcodecs"   "$ALL"
ev "$AF3"  audio_flamingo "af3_$(role audio_flamingo)_allcodecs"   "$LOSSY"
ev "$OMNI" qwen25_omni_3b "omni3b_HELDOUT_allcodecs"               "$ALL"
echo "=== [$(date -u +%FT%TZ)] DONE pair=$PAIR eps=$EPS seed=$SEED fail=$FAIL"
exit $FAIL
