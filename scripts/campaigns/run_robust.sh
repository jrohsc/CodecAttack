#!/bin/bash
# ============================================================================
# Channel-robust latent-space attack benchmark runner
#
# Uses RobustLatentCodecAttacker with EoT channel augmentation.
#
# Single experiment:
#   ./run_robust.sh --model qwen2_audio --music jazz_1 --eps 0.4
#   ./run_robust.sh --model qwen2_audio --music jazz_1 --eps 0.4 --n-eot 4 --severity 1.0
#   ./run_robust.sh --model qwen2_audio --music jazz_1 --eps 0.4 --curriculum
#
# Ablation (no channel = vanilla attack):
#   ./run_robust.sh --model qwen2_audio --music jazz_1 --eps 0.4 --no-channel
#
# Quick test:
#   ./run_robust.sh --model qwen2_audio --music jazz_1 --eps 0.4 --quick
#
# Check status:
#   ./run_robust.sh --status results_qa/benchmark_robust_qwen2_audio_eps0.4_jazz_1_eot4
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
MODEL="qwen2_audio"
MUSIC="calm_1"
EPS="3.0"
STEPS="300"
MODE="qa"
ALPHA="0.2"
PERCEPTUAL_WEIGHT="0.1"
WARMUP_RATIO="0.3"
GPU=""
QUIET="--quiet"
RESUME=""
STATUS_DIR=""
DATASET="saycan"

# Robust-specific defaults
N_EOT="2"
SEVERITY="1.0"
CURRICULUM=""
FREQ_SHAPING=""
NO_CHANNEL=""
EMPIRICAL_DIR=""
QUICK=""
CHANNEL_MODE="ir"
PROXY_BANDWIDTHS="1.5,3.0,6.0"
N_EOT_CODEC="1"

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
        --status)            STATUS_DIR="$2";         shift 2;;
        # Robust-specific
        --n-eot)             N_EOT="$2";              shift 2;;
        --severity)          SEVERITY="$2";           shift 2;;
        --curriculum)        CURRICULUM="--channel-curriculum"; shift;;
        --freq-shaping)      FREQ_SHAPING="--freq-shaping"; shift;;
        --no-channel)        NO_CHANNEL="--no-channel"; shift;;
        --channel-mode)      CHANNEL_MODE="$2";       shift 2;;
        --proxy-bandwidths)  PROXY_BANDWIDTHS="$2";   shift 2;;
        --n-eot-codec)       N_EOT_CODEC="$2";        shift 2;;
        --empirical-dir)     EMPIRICAL_DIR="$2";      shift 2;;
        # OTA eval
        --ota-opus)          OTA_OPUS="$2";           shift 2;;
        --ota-snr)           OTA_SNR="$2";            shift 2;;
        # Quick
        --quick)
            STEPS="150"
            QUIET="--quiet"
            QUICK="--quick"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_robust.sh [OPTIONS]"
            echo ""
            echo "Standard options:"
            echo "  --model, -m      Target model: qwen2_audio | qwen25_omni | audio_flamingo | kimi_audio"
            echo "  --music, -c      Music carrier name or path"
            echo "  --eps, -e        Latent epsilon (default: 0.4)"
            echo "  --steps, -s      Attack steps (default: 300)"
            echo "  --mode           Prompt mode: qa | transcribe (default: qa)"
            echo "  --gpu, -g        CUDA device index"
            echo "  --dataset, -d    Dataset: ours | saycan (default: ours)"
            echo ""
            echo "Robust options:"
            echo "  --channel-mode M Channel mode: ir | codec | full (default: ir)"
            echo "  --proxy-bandwidths B  Comma-separated bandwidths for codec proxy (default: 1.5,3.0,6.0)"
            echo "  --n-eot-codec N  Number of EoT codec proxy passes per step (default: 1)"
            echo "  --n-eot N        Number of EoT samples per step (legacy, default: 4)"
            echo "  --severity S     Max channel severity 0-1 (default: 1.0)"
            echo "  --curriculum     Enable severity warmup schedule"
            echo "  --freq-shaping   Enable frequency band projection"
            echo "  --no-channel     Disable channel (vanilla attack, for ablation)"
            echo "  --empirical-dir  Path to empirical IR/noise data"
            echo ""
            echo "OTA eval options:"
            echo "  --ota-opus N     Opus bitrate for OTA eval (default: 128)"
            echo "  --ota-snr N      SNR for OTA eval (default: 25)"
            echo ""
            echo "Other:"
            echo "  --quick          Quick test: 150 steps, 2 commands"
            echo "  --resume, -r     Resume if output exists"
            echo "  --status DIR     Show progress"
            echo "  --verbose, -v    Full per-step logging"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ---- Status mode ----
if [ -n "$STATUS_DIR" ]; then
    if [ ! -d "$STATUS_DIR" ]; then
        echo "ERROR: Directory not found: $STATUS_DIR"
        exit 1
    fi
    echo "Status: $STATUS_DIR"
    python -c "
import json, os, sys
d = '$STATUS_DIR'
results = {}
for root, dirs, files in os.walk(d):
    for f in files:
        if f == 'record.json':
            try:
                with open(os.path.join(root, f)) as fh:
                    rec = json.load(fh)
                rel = os.path.relpath(root, d)
                model = rel.split(os.sep)[0] if os.sep in rel else 'default'
                if model not in results:
                    results[model] = []
                results[model].append(rec)
            except: pass
for model, recs in sorted(results.items()):
    n = len(recs)
    d_ok = sum(1 for r in recs if r.get('direct',{}).get('success'))
    o64 = sum(1 for r in recs if r.get('opus',{}).get('64',{}).get('success'))
    o128 = sum(1 for r in recs if r.get('opus',{}).get('128',{}).get('success'))
    ota = sum(1 for r in recs if r.get('ota',{}).get('success'))
    print(f'[{model}] {n} exps  Direct={d_ok}/{n} ({100*d_ok/n:.0f}%)  Opus64={o64}/{n} ({100*o64/n:.0f}%)  Opus128={o128}/{n} ({100*o128/n:.0f}%)  OTA={ota}/{n} ({100*ota/n:.0f}%)')
"
    exit 0
fi

# ---- Map model → conda env ----
declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["qwen25_omni"]="qwen-omni"
    ["audio_flamingo"]="flamingo3"
    ["kimi_audio"]="kimi-audio"
)

CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'. Choose: qwen2_audio | qwen25_omni | audio_flamingo | kimi_audio"
    exit 1
fi

# ---- Output directory ----
DATASET_TAG=""
if [ "$DATASET" != "ours" ]; then
    DATASET_TAG="_${DATASET}"
fi

if [ -n "$NO_CHANNEL" ]; then
    CH_TAG="noCh"
elif [ "$CHANNEL_MODE" = "ir" ]; then
    CH_TAG="ir"
elif [ "$CHANNEL_MODE" = "codec" ]; then
    CH_TAG="codec_eot${N_EOT_CODEC}"
elif [ "$CHANNEL_MODE" = "full" ]; then
    CH_TAG="full_eot${N_EOT_CODEC}"
elif [ "$CHANNEL_MODE" = "ota" ]; then
    CH_TAG="ota_eot${N_EOT}"
else
    CH_TAG="${CHANNEL_MODE}"
fi

OUTPUT_DIR="${SCRIPT_DIR}/results_airgap/benchmark_${MODEL}_eps${EPS}_${MUSIC}_${CH_TAG}${DATASET_TAG}"

# ---- GPU selection ----
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# ---- Resume check ----
RESUME_FLAG=""
if [ -d "$OUTPUT_DIR" ]; then
    if [ -n "$RESUME" ]; then
        RESUME_FLAG="--resume $OUTPUT_DIR"
    else
        DONE=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
        if [ "$DONE" -gt 0 ]; then
            echo "Found $DONE completed experiments in $OUTPUT_DIR"
            echo "Auto-resuming..."
            RESUME_FLAG="--resume $OUTPUT_DIR"
        fi
    fi
fi

# ---- Print config ----
echo "============================================"
echo "  ROBUST ATTACK BENCHMARK"
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
echo "  Freq shape:  ${FREQ_SHAPING:-off}"
echo "  Channel mode:$CHANNEL_MODE"
echo "  Proxy BWs:   $PROXY_BANDWIDTHS"
echo "  EoT codec:   $N_EOT_CODEC"
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

# Build empirical dir flag
EMPIRICAL_FLAG=""
if [ -n "$EMPIRICAL_DIR" ]; then
    EMPIRICAL_FLAG="--empirical-data-dir $EMPIRICAL_DIR"
fi

python run_robust_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --dataset "$DATASET" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --alpha "$ALPHA" \
    --perceptual-weight "$PERCEPTUAL_WEIGHT" \
    --prompt-mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --channel-mode "$CHANNEL_MODE" \
    --proxy-bandwidths "$PROXY_BANDWIDTHS" \
    --n-eot-codec "$N_EOT_CODEC" \
    --n-eot-samples "$N_EOT" \
    --channel-severity "$SEVERITY" \
    --warmup-ratio "$WARMUP_RATIO" \
    --ota-opus-bitrate "$OTA_OPUS" \
    --ota-snr-db "$OTA_SNR" \
    $CURRICULUM \
    $FREQ_SHAPING \
    $NO_CHANNEL \
    $EMPIRICAL_FLAG \
    $QUICK \
    $QUIET \
    $RESUME_FLAG
