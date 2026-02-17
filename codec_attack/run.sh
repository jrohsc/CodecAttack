#!/bin/bash
# ============================================================================
# Latent-space attack benchmark runner
#
# Single experiment:
#   ./run.sh --model qwen2_audio --music jazz_1 --eps 0.4
#   ./run.sh --model audio_flamingo --music empire_state_of_mind --eps 1.0 --gpu 1
#   ./run.sh --model kimi_audio --music calm_1 --eps 10.0 --steps 300
#
# Run in parallel across terminals:
#   Terminal 1: ./run.sh -m qwen2_audio -c jazz_1 -e 0.4 -g 0
#   Terminal 2: ./run.sh -m qwen2_audio -c calm_1 -e 0.4 -g 1
#   Terminal 3: ./run.sh -m audio_flamingo -c jazz_1 -e 1.0 -g 2
#
# Resume an interrupted run:
#   ./run.sh -m qwen2_audio -c jazz_1 -e 0.4 --resume
#
# Check status of a run:
#   ./run.sh --status results_qa/benchmark_qwen2_audio_eps0.4_jazz_1
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
MODEL="audio_flamingo"
MUSIC="christmas_jazz_1"
EPS="0.5"
STEPS="150"
MODE="qa"
ALPHA="0.2"
PERCEPTUAL_WEIGHT="0.1"
GPU=""
QUIET="--quiet"
RESUME=""
STATUS_DIR=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)     MODEL="$2";              shift 2;;
        --music|-c)     MUSIC="$2";              shift 2;;
        --eps|-e)       EPS="$2";                shift 2;;
        --steps|-s)     STEPS="$2";              shift 2;;
        --mode)         MODE="$2";               shift 2;;
        --alpha)        ALPHA="$2";              shift 2;;
        --perceptual-weight) PERCEPTUAL_WEIGHT="$2"; shift 2;;
        --gpu|-g)       GPU="$2";                shift 2;;
        --verbose|-v)   QUIET="";                shift;;
        --resume|-r)    RESUME="1";              shift;;
        --status)       STATUS_DIR="$2";         shift 2;;
        --help|-h)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m    Target model: qwen2_audio | audio_flamingo | kimi_audio"
            echo "  --music, -c    Music carrier: empire_state_of_mind | jazz_1 | calm_1 | classical_music_1 | ..."
            echo "  --eps, -e      Epsilon for latent perturbation (default: 0.4)"
            echo "  --steps, -s    Attack optimization steps (default: 150)"
            echo "  --mode         Prompt mode: qa | transcribe (default: qa)"
            echo "  --alpha        Adam learning rate (default: 0.2)"
            echo "  --gpu, -g      CUDA device index (default: auto)"
            echo "  --verbose, -v  Show per-step logging"
            echo "  --resume, -r   Resume if output dir already exists"
            echo "  --status DIR   Show progress of a benchmark directory"
            echo ""
            echo "Models → Conda envs:"
            echo "  qwen2_audio    → codec-attack"
            echo "  audio_flamingo → flamingo3"
            echo "  kimi_audio     → kimi-audio"
            echo ""
            echo "Music carriers:"
            echo "  empire_state_of_mind, jazz_1, jazz_2, calm_1, calm_2,"
            echo "  classical_music_1, classical_music_2, christmas_jazz_1, christmas_jazz_2"
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
# Find all record.json files in any subdirectory
results = {}
for root, dirs, files in os.walk(d):
    for f in files:
        if f == 'record.json':
            try:
                with open(os.path.join(root, f)) as fh:
                    rec = json.load(fh)
                key = (rec.get('music','?'), rec.get('command_name','?'))
                # Get the model subdir name
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
    print(f'[{model}] {n}/200  Direct={d_ok}/{n} ({100*d_ok/n:.0f}%)  Opus64={o64}/{n} ({100*o64/n:.0f}%)  Opus128={o128}/{n} ({100*o128/n:.0f}%)')
"
    exit 0
fi

# ---- Map model → conda env ----
declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["audio_flamingo"]="flamingo3"
    ["kimi_audio"]="kimi-audio"
)

CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'. Choose: qwen2_audio | audio_flamingo | kimi_audio"
    exit 1
fi

# ---- Output directory ----
OUTPUT_DIR="${SCRIPT_DIR}/results_${MODE}/benchmark_${MODEL}_eps${EPS}_${MUSIC}"

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
        # Auto-resume: check how many are done
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
echo "  Model:     $MODEL"
echo "  Conda:     $CONDA_ENV"
echo "  Music:     $MUSIC"
echo "  EPS:       $EPS"
echo "  Steps:     $STEPS"
echo "  Mode:      $MODE"
echo "  GPU:       ${GPU:-auto}"
echo "  Output:    $OUTPUT_DIR"
echo "============================================"

# ---- Activate conda and run ----
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "Warning: Could not activate conda env '$CONDA_ENV'"
}

python run_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --alpha "$ALPHA" \
    --perceptual-weight "$PERCEPTUAL_WEIGHT" \
    --prompt-mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    $QUIET \
    $RESUME_FLAG
