#!/bin/bash
# The black-box transfer arm: external-ASR-anchored content injection.
#
# WHY THIS ARM EXISTS
# The encoder-agreement probe (2026-08-14) showed our working attack does not
# route through the audio encoder at all: ce_zeros is 50/50 white-box while
# moving its OWN source encoder AWAY from the target (delta cos -0.0416 probe,
# -0.0919 driver, two independent instruments agreeing). Conversely the
# encoder-cosine arms reach cos 0.86 at 0/50 white-box. So the attack finds a
# model-specific LM-readout direction carrying no target-like audio content --
# an idiosyncrasy of one model's loss landscape, which cannot transfer by
# construction. That single fact explains transfer 0/12426, the prompt-binding
# (88 -> 0 on a prompt swap), and the encoder arms' white-box failure.
#
# To transfer, the audio must carry something the victims SHARE. The only such
# thing is speech content. These arms optimize content directly, anchored on a
# frozen Whisper-large-v3 that no victim was fine-tuned as.
#
# TWO ARMS, DELIBERATELY
#   asr      pure content term. NOT a deployable attack -- it is the
#            feasibility control. If this cannot make an external ASR
#            transcribe the target, the whole approach is dead and asr_ce
#            cannot be rescued by tuning. Cheapest decisive result.
#   asr_ce   the actual transfer arm: content + source CE. ARO never uses its
#            anchor term alone, and our enc-alone arms (0/50 white-box) show
#            why.
#
# READ asr_exact_pct IN THE MANIFEST FIRST, NOT whitebox_exact_pct. A bundle
# with high white-box but low asr_exact is still a readout attack and will
# still transfer at 0 -- that combination means the CE term won and the arm
# collapsed back to ce_zeros.
#
# asr_weight=3.0: in the 150-step smoke the CE term fell 6.41 -> 1.09 while the
# ASR term only moved 4.00 -> 2.92, i.e. CE dominates the descent at equal
# weight. Upweighting content is the whole point of the arm.
#
# Waits for the eps-ladder runner to exit before starting -- both want the same
# A100 and a co-resident attack plus anchor OOMs at 80 GB.
#
# Usage: nohup bash run_asr_transfer_local.sh > logs/asr_transfer_local.log 2>&1 &

set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
LADDER_PID="${LADDER_PID:-}"
N=10
STEPS=1000
ASR_W=3.0

stamp() { date +'%Y-%m-%d %H:%M:%S'; }

if [[ -n "$LADDER_PID" ]]; then
    echo "[$(stamp)] waiting for eps-ladder PID $LADDER_PID to free the GPU..."
    while kill -0 "$LADDER_PID" 2>/dev/null; do sleep 60; done
    echo "[$(stamp)] ladder done; starting ASR arms"
fi

run_arm() {
    local arm="$1" loss="$2" extra_w="$3"
    local OUT="0_all_combined/results_encoder_transfer/$arm"
    echo ""
    echo "##########################################################"
    echo "### [$(stamp)] ARM $arm  loss=$loss asr_weight=$extra_w n=$N steps=$STEPS"
    echo "##########################################################"
    "$CA" -u encoder_transfer_attack.py \
        --loss "$loss" --asr-weight "$extra_w" \
        --init zeros --eps 1.0 --steps "$STEPS" --check-every 100 --n "$N" \
        --frame-mask target --carrier-mode match --prompt-mode transcribe \
        --out-dir "$OUT" 2>&1 | grep -av "it/s"
    if [[ -f "$OUT/manifest.json" ]]; then
        echo "--- [$(stamp)] $arm manifest ---"
        "$CA" -c "
import json
m=json.load(open('$OUT/manifest.json'))
print(f\"  white-box exact : {m.get('whitebox_exact')}/{m['n']} ({m.get('whitebox_exact_pct')}%)\")
print(f\"  ASR-heard exact : {m.get('asr_exact')}/{m['n']} ({m.get('asr_exact_pct')}%)  <-- transfer predictor\")
"
    else
        echo "!!! $arm produced no manifest.json"
    fi
}

# Feasibility control first: it is cheaper to learn the approach is dead here
# than after also paying for asr_ce.
run_arm ASR_pure   asr    "$ASR_W"
run_arm ASR_ce     asr_ce "$ASR_W"

echo ""
echo "=== [$(stamp)] cross-model transfer evals, BOTH prompts, locally ==="
# NOT submit_encoder_transfer_evals.sh: that dispatches via sbatch, and the
# queue is exactly what we moved off (last night's enc_mask_chain is still
# pending). This runs the arm x evaluator x prompt grid on the GPU we already
# hold, and scores under transcribe (where the positive-control ceiling has
# headroom) as well as bank (the deployed prompt, ceiling ~0 -- kept so the
# zero is documented rather than confusing).
bash run_encoder_transfer_evals_local.sh ASR_pure ASR_ce || \
    echo "!!! eval fanout returned nonzero"

echo ""
echo "=== [$(stamp)] ASR transfer arms done ==="
