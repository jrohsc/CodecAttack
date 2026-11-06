#!/bin/bash
# Local (non-SLURM) runner for the whole eps-ladder transfer probe.
#
# Replaces sbatch_s1_eps15_transfer.sh (Phase 0) + sbatch_s1_eps_ladder_attack.sh
# (Phase 1), which were cancelled as jobs 62877360 / 62877537 because the queue
# was backed up on (Priority) while this session's own interactive allocation
# (umd-cscdr-gpu002, A100-80GB) sat completely idle.
#
# ONLY ONE WRITER. Do not resubmit either sbatch script while this is running --
# both write the same transfer_eval_<model>.json paths and would race. The
# sbatch scripts are kept for provenance and for reproducing on a fresh node.
#
# Resumable by construction:
#   * cross_model_transfer_eval.py reuses cached clips/channels from an existing
#     JSON and only computes what is missing. It aborts (nonzero) if the cached
#     file was written under a different prompt, which is the guard we want --
#     see the S3b neural-codec prompt-blending bug it references.
#   * run_s1_attack_bundleA.py skips carriers whose adv wav already exists.
# So re-running this script after an interruption picks up where it stopped.
#
# Sequential on one GPU: the attack and the evals each want the whole card.
#
# Usage:
#   nohup bash run_eps_ladder_local.sh > logs/eps_ladder_local.log 2>&1 &

set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
CA="$ENVROOT/codec-attack/bin/python"
OMNI="$ENVROOT/qwen-omni/bin/python"
AF3="$ENVROOT/flamingo3/bin/python"

ROOT=0_all_combined/results_codec_robust/01_finance_voice_agent
ATTACKER=qwen2_audio

# evaluator -> interpreter
py_for() {
    case "$1" in
        qwen2_audio)                  echo "$CA" ;;
        qwen25_omni|qwen25_omni_3b)   echo "$OMNI" ;;
        audio_flamingo)               echo "$AF3" ;;
        *) echo "UNKNOWN_MODEL" ;;
    esac
}

FAILED=()

stamp() { date +'%Y-%m-%d %H:%M:%S'; }

run_eval() {
    local bundle="$1" ev="$2" label="$3"
    local py; py=$(py_for "$ev")
    echo ""
    echo "=== [$(stamp)] EVAL $label :: bundle=$bundle evaluator=$ev"
    if ! "$py" 0_all_combined/scripts/cross_model_transfer_eval.py \
            --bundle-dir "$bundle" \
            --target-model "$ev" \
            --opus-bitrates 32,64; then
        echo "!!! FAILED: $label $bundle -> $ev"
        FAILED+=("eval:$bundle->$ev")
    fi
}

############################################################
# PHASE 0 -- eps=1.5, bundles already exist, eval only.
# 12 cells = 3 attacker bundles x 4 evaluators.
# 3 already landed via SLURM tasks 0-2 (qwen2_audio attacker); those are
# reused from cache, not recomputed.
############################################################
echo "##########################################################"
echo "### [$(stamp)] PHASE 0: eps=1.5 transfer, 12 cells"
echo "##########################################################"

for atk in qwen2_audio qwen25_omni audio_flamingo; do
    BUNDLE="$ROOT/$atk/eps_1.5"
    NWAV=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
    if [[ "$NWAV" -ne 50 ]]; then
        echo "SKIP $BUNDLE: $NWAV wavs, expected 50 (refusing to score a partial bundle)"
        FAILED+=("partial:$BUNDLE:$NWAV")
        continue
    fi
    for ev in qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo; do
        role="transfer"; [[ "$atk" == "$ev" ]] && role="DIAGONAL"
        run_eval "$BUNDLE" "$ev" "eps1.5 $atk->$ev [$role]"
    done
done

echo ""
echo "=== [$(stamp)] PHASE 0 ladder so far ==="
"$CA" build_eps_ladder_table.py || true

############################################################
# PHASE 1 -- eps in {2.0, 3.0}: attack then eval.
# One attacker (qwen2_audio) x 15 carriers. Ladder caps at 3.0 because the
# perturbation is already at PESQ 1.34 / SNR 3.9 dB by eps=1.5 (PESQ floors at
# 1.0), so eps>=4 would not leave a usable carrier.
############################################################
CARRIERS=(
  banking_christopher_q2_rates_edgetts.wav
  banking_eric_q2_rates_edgetts.wav
  banking_roger_q2_rates_edgetts.wav
  banking_michelle_q4_branch_edgetts.wav
  banking_aria_q4_branch_edgetts.wav
  banking_christopher_q1_balance_edgetts.wav
  banking_eric_q1_balance_edgetts.wav
  banking_roger_q1_balance_edgetts.wav
  banking_michelle_q5_card_edgetts.wav
  banking_aria_q5_card_edgetts.wav
  banking_christopher_q3_transactions_edgetts.wav
  banking_eric_q3_transactions_edgetts.wav
  banking_roger_q3_transactions_edgetts.wav
  banking_steffan_q1_balance_edgetts.wav
  banking_emily_q1_balance_edgetts.wav
)
NEED=${#CARRIERS[@]}

for EPS in 2.0 3.0; do
    BUNDLE="$ROOT/$ATTACKER/eps_$EPS"
    echo ""
    echo "##########################################################"
    echo "### [$(stamp)] PHASE 1: attack $ATTACKER eps=$EPS n=$NEED"
    echo "##########################################################"

    if ! env CODEC_ATTACK_PY="$CA" "$CA" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
            --target-model "$ATTACKER" \
            --eps "$EPS" \
            --carriers "${CARRIERS[@]}"; then
        echo "!!! attack driver returned nonzero for eps=$EPS (continuing to wav count)"
    fi

    NWAV=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
    echo "=== [$(stamp)] eps=$EPS attack produced $NWAV/$NEED wavs"
    if [[ "$NWAV" -ne "$NEED" ]]; then
        echo "!!! PARTIAL BUNDLE eps=$EPS ($NWAV/$NEED) -- not scoring. Re-run this script to resume."
        FAILED+=("attack:eps_$EPS:$NWAV/$NEED")
        continue
    fi

    # Diagonal first: if it is not high, the transfer cells are unreadable.
    run_eval "$BUNDLE" qwen2_audio    "eps$EPS DIAGONAL"
    run_eval "$BUNDLE" audio_flamingo "eps$EPS transfer"
    run_eval "$BUNDLE" qwen25_omni_3b "eps$EPS transfer"
done

echo ""
echo "##########################################################"
echo "### [$(stamp)] FINAL LADDER"
echo "##########################################################"
"$CA" build_eps_ladder_table.py || true

if ((${#FAILED[@]})); then
    echo ""
    echo "!!! ${#FAILED[@]} problem(s):"
    printf '   %s\n' "${FAILED[@]}"
    exit 1
fi
echo ""
echo "=== [$(stamp)] ALL DONE, no failures ==="
