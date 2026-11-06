#!/usr/bin/env bash
# Capacity-cliff ablation — end-to-end pipeline.
#
# Runs: Exp-A (words) + Exp-B (duration) + transcribe control,
# then codec sidecars, then rollups, then table + figure.
#
# Every stage is resumable: cells with existing *_attacked.wav skip;
# codec-sidecar stage skips if *_codecs.json exists.
#
# Usage:
#     bash 0_all_combined/results_ablation_capacity/scripts/run_all.sh
#
# Env overrides:
#     CODEC_ATTACK_PY        path to codec-attack conda env python
#     SKIP_STAGES            comma-separated names to skip
#                            (expA,expB,transcribe,eval,rollups,table,figure)
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ABL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ABL/../.." && pwd)"      # codec_attack/

PY="${CODEC_ATTACK_PY:-${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python}"
SKIP="${SKIP_STAGES:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ABL/scripts/run_all.$STAMP.log"

# Colors for stage banners
banner() { echo; echo "===== $1 ====="; }

# Return 0 if stage should run, non-zero if it's in SKIP list.
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

cd "$REPO"

{
    echo "Capacity-cliff ablation pipeline"
    echo "  repo: $REPO"
    echo "  abl:  $ABL"
    echo "  py:   $PY"
    echo "  skip: $SKIP"
    echo "  log:  $LOG"
    echo "  stamp: $STAMP"

    run_stage expA       "$PY" "$ABL/scripts/run_expA_words.py"
    run_stage expB       "$PY" "$ABL/scripts/run_expB_duration.py"
    run_stage transcribe "$PY" "$ABL/scripts/run_transcribe_control.py"
    run_stage eval       "$PY" "$ABL/scripts/eval_channels.py" --exp all
    run_stage rollups    "$PY" "$ABL/scripts/build_rollups.py"
    run_stage table      "$PY" "$ABL/scripts/build_table.py"
    run_stage figure     "$PY" "$ABL/scripts/plot_capacity.py"

    echo
    echo "===== DONE ($STAMP) ====="
    echo "  table:  $ABL/table_capacity_v2.tex"
    echo "  figure: $ABL/figure_capacity.pdf"
    echo "  rollup: $ABL/rollup_expA.json  $ABL/rollup_expB.json"
} 2>&1 | tee "$LOG"
