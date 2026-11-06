#!/bin/bash
# ============================================================================
# Airgap-robust WAVEFORM baseline attack runner
#
# Quick test (2 commands, 150 steps):
#   ./run_robust_waveform.sh --model qwen2_audio --music jazz_1 --eps 0.01 --quick
#
# Full benchmark (SayCan, 35 commands):
#   ./run_robust_waveform.sh --model qwen2_audio --music jazz_1 --eps 0.01 --dataset saycan
#
# With curriculum:
#   ./run_robust_waveform.sh --model qwen2_audio --music jazz_1 --eps 0.01 --curriculum
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
MODEL="qwen2_audio"
MUSIC="jazz_1"
EPS="0.01"
STEPS="300"
MODE="qa"
ALPHA="0.001"
PERCEPTUAL_WEIGHT="0.1"
GPU=""
QUIET="--quiet"
RESUME=""
DATASET="saycan"

# Robust-specific defaults
N_EOT="2"
SEVERITY="1.0"
CURRICULUM=""
NO_CHANNEL=""
WARMUP_RATIO="0.3"

# OTA eval defaults
OTA_OPUS="128"
OTA_SNR="25"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)          MODEL="$2";              shift 2;;
        --music|-c)          MUSIC="$2";              shift 2;;
        --eps|-e)            EPS="$2";                shift 2;;
        --steps|-s)          STEPS="$2";              shift 2;;
        --mode)              MODE="$2";               shift 2;;
        --alpha)             ALPHA="$2";              shift 2;;
        --perceptual-weight) PERCEPTUAL_WEIGHT="$2";  shift 2;;
        --gpu|-g)            GPU="$2";                shift 2;;
        --dataset|-d)        DATASET="$2";            shift 2;;
        --verbose|-v)        QUIET="";                shift;;
        --resume|-r)         RESUME="1";              shift;;
        # Robust-specific
        --n-eot)             N_EOT="$2";              shift 2;;
        --severity)          SEVERITY="$2";           shift 2;;
        --curriculum)        CURRICULUM="--channel-curriculum"; shift;;
        --no-channel)        NO_CHANNEL="--no-channel"; shift;;
        --warmup-ratio)      WARMUP_RATIO="$2";       shift 2;;
        # OTA eval
        --ota-opus)          OTA_OPUS="$2";           shift 2;;
        --ota-snr)           OTA_SNR="$2";            shift 2;;
        # Quick
        --quick)
            STEPS="150"
            QUIET="--quiet"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_robust_waveform.sh [OPTIONS]"
            echo ""
            echo "  --model, -m      Target model: qwen2_audio | qwen25_omni | audio_flamingo"
            echo "  --music, -c      Music carrier name"
            echo "  --eps, -e        Waveform L-inf epsilon (default: 0.01)"
            echo "  --steps, -s      Attack steps (default: 300)"
            echo "  --n-eot N        EoT samples per step (default: 2)"
            echo "  --severity S     Channel severity 0-1 (default: 1.0)"
            echo "  --curriculum     Enable severity warmup"
            echo "  --no-channel     Disable channel (vanilla waveform attack)"
            echo "  --quick          Quick test: 150 steps, 2 commands"
            echo "  --dataset, -d    Dataset: ours | saycan"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ---- Map model -> conda env ----
declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["qwen25_omni"]="qwen-omni"
    ["audio_flamingo"]="flamingo3"
    ["kimi_audio"]="kimi-audio"
)

CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'."
    exit 1
fi

# ---- Output directory ----
DATASET_TAG=""
if [ "$DATASET" != "ours" ]; then
    DATASET_TAG="_${DATASET}"
fi

if [ -n "$NO_CHANNEL" ]; then
    CH_TAG="noCh"
else
    CH_TAG="ota_eot${N_EOT}"
fi

OUTPUT_DIR="${SCRIPT_DIR}/results_waveform_airgap/robust_waveform_${MODEL}_eps${EPS}_${MUSIC}_${CH_TAG}${DATASET_TAG}"

# ---- GPU selection ----
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# ---- Resume check ----
RESUME_FLAG=""
if [ -d "$OUTPUT_DIR" ]; then
    DONE=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
    if [ "$DONE" -gt 0 ]; then
        echo "Found $DONE completed experiments in $OUTPUT_DIR"
        echo "Auto-resuming..."
        RESUME_FLAG="--resume $OUTPUT_DIR"
    fi
fi

# ---- Print config ----
echo "============================================"
echo "  ROBUST WAVEFORM ATTACK"
echo "============================================"
echo "  Model:       $MODEL"
echo "  Conda:       $CONDA_ENV"
echo "  Music:       $MUSIC"
echo "  Dataset:     $DATASET"
echo "  EPS:         $EPS"
echo "  Steps:       $STEPS"
echo "  Mode:        $MODE"
echo "  GPU:         ${GPU:-auto}"
echo "  EoT samples: $N_EOT"
echo "  Severity:    $SEVERITY"
echo "  Curriculum:  ${CURRICULUM:-off}"
echo "  No channel:  ${NO_CHANNEL:-off}"
echo "  Output:      $OUTPUT_DIR"
echo "============================================"

# ---- Activate conda and run ----
CONDA_EXE="${CONDA_EXE:-${WORK_ROOT}/miniconda3/bin/conda}"
eval "$("$CONDA_EXE" shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV" || {
    echo "ERROR: Could not activate conda env '$CONDA_ENV'"
    exit 1
}

python run_robust_waveform_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --dataset "$DATASET" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --alpha "$ALPHA" \
    --perceptual-weight "$PERCEPTUAL_WEIGHT" \
    --prompt-mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --n-eot-samples "$N_EOT" \
    --channel-severity "$SEVERITY" \
    --warmup-ratio "$WARMUP_RATIO" \
    --ota-opus-bitrate "$OTA_OPUS" \
    --ota-snr-db "$OTA_SNR" \
    $CURRICULUM \
    $NO_CHANNEL \
    $QUIET \
    $RESUME_FLAG
