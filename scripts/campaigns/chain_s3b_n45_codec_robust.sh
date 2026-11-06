#!/usr/bin/env bash
# S3b copyright-bypass codec-robust sweep for the expanded n=45 carrier set
# (9 carriers x 5 primary copyright_bypass targets).
#
# Runs models x eps cells sequentially on one GPU. The original 20 pairs
# already exist in each bundle and are skipped automatically by
# run_s3_attack.py (skip-existing on *_pair{idx:03d}_attacked.wav). Sidecars
# also skip when *_codecs.json is already present.
#
# Output: 0_all_combined/results_codec_robust/03b_copyright_bypass/{model}/eps_{eps}_multibitrate
# Logs:   0_all_combined/scripts/logs/s3b_n45_<cell>.log
#
# Usage:
#   chain_s3b_n45_codec_robust.sh                    # interactive picker if a TTY,
#                                                    # else all 3 models x all 3 eps
#   chain_s3b_n45_codec_robust.sh --models qwen2_audio --eps 1.0    # one cell
#   chain_s3b_n45_codec_robust.sh --models qwen2_audio,qwen25_omni  # subset of models
#   chain_s3b_n45_codec_robust.sh --eps 1.0,1.5                     # subset of eps
#   chain_s3b_n45_codec_robust.sh --interactive                     # force picker
#   chain_s3b_n45_codec_robust.sh --all                              # force full sweep (no picker)
#
# Valid models: qwen2_audio, qwen25_omni, audio_flamingo
# Valid eps:    0.5, 1.0, 1.5
#
# Do NOT launch while any other attack/sidecar job is using the GPU.
set -u

MODELS_ARG=""
EPS_ARG=""
FORCE_INTERACTIVE=0
FORCE_ALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)      MODELS_ARG="$2"; shift 2 ;;
        --eps)         EPS_ARG="$2";    shift 2 ;;
        --interactive) FORCE_INTERACTIVE=1; shift ;;
        --all)         FORCE_ALL=1;        shift ;;
        -h|--help)
            sed -n '2,24p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            echo "Use --help for usage." >&2
            exit 2
            ;;
    esac
done

# ---- interactive picker ----
ALL_MODELS=("qwen2_audio" "qwen25_omni" "audio_flamingo")
ALL_EPSES=("0.5" "1.0" "1.5")

pick_subset() {
    # $1 = label, $2 = name of array variable holding choices
    local label="$1"
    local -n choices="$2"
    local n=${#choices[@]}
    echo "Select $label:" >&2
    for i in "${!choices[@]}"; do
        printf "  [%d] %s\n" "$((i+1))" "${choices[$i]}" >&2
    done
    printf "  [a] all\n" >&2
    printf "Enter choices (e.g. '1,3' or 'a'): " >&2
    local raw; read -r raw
    raw="${raw// /}"
    if [[ -z "$raw" || "$raw" == "a" || "$raw" == "all" ]]; then
        ( IFS=,; echo "${choices[*]}" )
        return
    fi
    local out=()
    IFS=',' read -ra idxs <<< "$raw"
    for idx in "${idxs[@]}"; do
        if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx < 1 || idx > n )); then
            echo "Invalid choice: $idx (must be 1..$n or 'a')" >&2
            return 1
        fi
        out+=("${choices[$((idx-1))]}")
    done
    ( IFS=,; echo "${out[*]}" )
}

if [[ $FORCE_ALL -eq 1 ]]; then
    [[ -z "$MODELS_ARG" ]] && MODELS_ARG="qwen2_audio,qwen25_omni,audio_flamingo"
    [[ -z "$EPS_ARG"    ]] && EPS_ARG="0.5,1.0,1.5"
fi

# Picker only runs when no CLI subset was provided AND we're attached to a TTY
# (or --interactive forced). For nohup / pipes we silently default to "all".
WANT_PICKER=0
if [[ $FORCE_INTERACTIVE -eq 1 ]]; then
    WANT_PICKER=1
elif [[ -z "$MODELS_ARG" && -z "$EPS_ARG" && $FORCE_ALL -eq 0 && -t 0 && -t 2 ]]; then
    WANT_PICKER=1
fi

if [[ $WANT_PICKER -eq 1 ]]; then
    if [[ -z "$MODELS_ARG" ]]; then
        MODELS_ARG="$(pick_subset 'models' ALL_MODELS)" || exit 2
    fi
    if [[ -z "$EPS_ARG" ]]; then
        EPS_ARG="$(pick_subset 'eps values' ALL_EPSES)" || exit 2
    fi
fi

# Final defaults for any field still empty (non-TTY / nohup with no args).
[[ -z "$MODELS_ARG" ]] && MODELS_ARG="qwen2_audio,qwen25_omni,audio_flamingo"
[[ -z "$EPS_ARG"    ]] && EPS_ARG="0.5,1.0,1.5"

IFS=',' read -ra MODELS <<< "$MODELS_ARG"
IFS=',' read -ra EPSES  <<< "$EPS_ARG"

VALID_MODELS=" qwen2_audio qwen25_omni audio_flamingo "
VALID_EPSES=" 0.5 1.0 1.5 "
for m in "${MODELS[@]}"; do
    [[ "$VALID_MODELS" == *" $m "* ]] || { echo "Invalid model: $m (valid:$VALID_MODELS)" >&2; exit 2; }
done
for e in "${EPSES[@]}"; do
    [[ "$VALID_EPSES" == *" $e "* ]] || { echo "Invalid eps: $e (valid:$VALID_EPSES)" >&2; exit 2; }
done

CELLS=$(( ${#MODELS[@]} * ${#EPSES[@]} ))
# Per-cell cost: ~25 new pairs * ~6.5 min = ~2.7 GPU-hours. Round to nearest 0.1h.
EST_HOURS=$(awk -v c="$CELLS" 'BEGIN{printf "%.1f", c*2.7}')

echo "=== chain_s3b_n45 plan ==="
echo "  models:  ${MODELS[*]}"
echo "  eps:     ${EPSES[*]}"
echo "  cells:   $CELLS"
echo "  est GPU: ~${EST_HOURS} h (single A100, sequential)"
echo

if [[ $WANT_PICKER -eq 1 ]]; then
    printf "Proceed? [y/N]: "
    read -r confirm
    case "$confirm" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
    echo
fi

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S3B_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/03b_copyright_bypass
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

S3B_PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"

run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    echo "=== [$(date +%F\ %H:%M:%S)] STAGE: $name ==="
    echo "  log: $log"
    "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED rc=$rc -- abort chain"
        exit $rc
    fi
    echo "  OK"
}

s3b_attack() {
    local model="$1" eps="$2" py="$3"
    run_stage "s3b_n45_${model}_eps${eps}_attack" \
        env CODEC_ATTACK_PY="$py" \
        "$py" -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario 03b --target-model "$model" --eps "$eps" \
            --channel-mode multi_bitrate --skip-eval
}

s3b_codec_sidecars() {
    local model="$1" eps="$2" py="$3"
    run_stage "s3b_n45_${model}_eps${eps}_codec_sidecars" \
        "$py" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3B_ROOT/$model/eps_${eps}_multibitrate" \
            --target-model "$model" \
            --prompt-text "$S3B_PROMPT"
}

py_for() {
    case "$1" in
        qwen2_audio)    echo "$CA" ;;
        qwen25_omni)    echo "$OMNI" ;;
        audio_flamingo) echo "$AF3" ;;
    esac
}

for EPS in "${EPSES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        PY="$(py_for "$MODEL")"
        s3b_attack         "$MODEL" "$EPS" "$PY"
        s3b_codec_sidecars "$MODEL" "$EPS" "$PY"
    done
done

echo ""
echo "######################################################"
echo "###  S3b n=45 codec-robust sweep COMPLETE          ###"
echo "######################################################"
