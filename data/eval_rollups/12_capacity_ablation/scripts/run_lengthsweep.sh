#!/usr/bin/env bash
# Length sweep (Exp-B): capacity vs. carrier duration.
#
# 3 Suno carriers (calm, classic, jazz) x 6 durations {5,10,15,20,30,60}s
# x 3 anchor word counts {5,15,40} = 54 attack cells, multi-bitrate Opus EOT,
# qwen2_audio target, eps=1.0, 1000 steps.
#
# Re-runs only the duration sweep + its codec sidecars + rollups + figure.
# Skips Exp-A (words) and the transcribe control. The capacity table
# (build_table.py) is Exp-A-only and is not regenerated here.
#
# Resumable: cells with an existing *_attacked.wav skip; codec sidecars
# skip if *_codecs.json exists.
#
# Usage:
#   bash 0_all_combined/results_ablation_capacity/scripts/run_lengthsweep.sh
#
# Optional env vars:
#   CODEC_ATTACK_PY  python interpreter (default: codec-attack env)
#   DURATIONS        space-separated subset, e.g. "5 10 15"
#   CARRIERS         space-separated subset, e.g. "calm jazz"
#   ANCHORS          space-separated subset, e.g. "5 15"
#   SKIP_STAGES      comma list from {expB,eval,rollups,figure}
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ABL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ABL/../.." && pwd)"   # codec_attack/

PY="${CODEC_ATTACK_PY:-${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python}"
SKIP="${SKIP_STAGES:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ABL/scripts/run_lengthsweep.$STAMP.log"

banner() { echo; echo "===== $1 ====="; }

should_run() {
    case ",$SKIP," in
        *",$1,"*) return 1 ;;
        *)        return 0 ;;
    esac
}

run_stage() {
    local name="$1"; shift
    if ! should_run "$name"; then
        banner "SKIPPED: $name (SKIP_STAGES)"
        return 0
    fi
    banner "[$name] $*"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "STAGE FAILED: $name (exit $rc)" >&2
        exit $rc
    fi
}

# Build optional flags for the expB driver from env.
expB_args=()
if [[ -n "${DURATIONS:-}" ]]; then expB_args+=(--durations $DURATIONS); fi
if [[ -n "${CARRIERS:-}"  ]]; then expB_args+=(--carriers  $CARRIERS);  fi
if [[ -n "${ANCHORS:-}"   ]]; then expB_args+=(--anchors   $ANCHORS);   fi

cd "$REPO"

{
    echo "Length-sweep (Exp-B) pipeline"
    echo "  repo:      $REPO"
    echo "  abl:       $ABL"
    echo "  py:        $PY"
    echo "  durations: ${DURATIONS:-default {5,10,15,20,30,60}}"
    echo "  carriers:  ${CARRIERS:-default {calm,classic,jazz}}"
    echo "  anchors:   ${ANCHORS:-default {5,15,40}}"
    echo "  skip:      $SKIP"
    echo "  log:       $LOG"
    echo "  stamp:     $STAMP"

    run_stage expB     "$PY" "$ABL/scripts/run_expB_duration.py" "${expB_args[@]}"
    run_stage eval     "$PY" "$ABL/scripts/eval_channels.py" --exp expB
    run_stage rollups  "$PY" "$ABL/scripts/build_rollups.py"
    run_stage figure   "$PY" "$ABL/scripts/plot_capacity.py"

    echo
    echo "===== DONE ($STAMP) ====="
    echo "  rollup: $ABL/rollup_expB.json"
    echo "  figure: $ABL/figure_capacity.pdf  (panel B updated; panel A uses existing rollup_expA.json)"
} 2>&1 | tee "$LOG"
