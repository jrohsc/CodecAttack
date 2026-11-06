#!/usr/bin/env python3
"""Build the eps-ladder cross-model transfer table for S1 finance.

Answers one question: is cross-model transfer monotone in the perturbation
budget? The working diagnosis is that transfer reads 0 everywhere because at
eps=1.0 the perturbation has only enough capacity to control one model under
one prompt. If capacity is the binding constraint, transfer should climb with
eps. If it stays pinned at 0 while the white-box diagonal stays high, the null
is mechanism-level, not budget-level -- a much stronger claim.

Reads the UNTAGGED transfer_eval_<model>.json files (bank prompt) so the rungs
are comparable cell-for-cell. Tagged variants (__transcribe, __bankprompt) are
a different experiment and are deliberately skipped.

Usage:
    python build_eps_ladder_table.py
    python build_eps_ladder_table.py --channel clean
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("0_all_combined/results_codec_robust/01_finance_voice_agent")
BANK_PROMPT_MARK = "First Federal Bank"
ATTACKERS = ["qwen2_audio", "qwen25_omni", "audio_flamingo"]
EVALUATORS = ["qwen2_audio", "qwen25_omni", "qwen25_omni_3b", "audio_flamingo"]
SHORT = {
    "qwen2_audio": "Qwen2-A",
    "qwen25_omni": "Omni-7B",
    "qwen25_omni_3b": "Omni-3B",
    "audio_flamingo": "AF3",
}


def load_cells(channel):
    """cells[eps][attacker][evaluator] = (exact, semantic, n) or None."""
    cells = defaultdict(lambda: defaultdict(dict))
    skipped = []
    for atk in ATTACKERS:
        for bundle in sorted((ROOT / atk).glob("eps_*")):
            # Only plain eps_<float> bundles -- no _seed/_mi_di/_tool variants,
            # which are different experiments with their own confounds.
            m = re.fullmatch(r"eps_(\d+(?:\.\d+)?)", bundle.name)
            if not m:
                continue
            eps = float(m.group(1))
            for ev in EVALUATORS:
                # Untagged is the convention for bank-prompt cells. At eps=1.0
                # the DIAGONAL was re-measured separately as __bankprompt (the
                # code-path control, job 62874698) -- same script, same bank
                # prompt, so it is the same measurement and is the only source
                # for those diagonal cells. Fall back to it.
                p = bundle / f"transfer_eval_{ev}.json"
                if not p.exists():
                    p = bundle / f"transfer_eval_{ev}__bankprompt.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                # Guard: a tagged-prompt file must never be counted here.
                if BANK_PROMPT_MARK not in d.get("prompt_text", ""):
                    skipped.append((str(p), d.get("prompt_text", "")[:40]))
                    continue
                res = d["results"]
                chans = d.get("channels") or [
                    k for k in res[0] if isinstance(res[0][k], dict)
                ]
                if channel not in chans:
                    continue
                ex = sum(1 for r in res if r.get(channel, {}).get("exact_match"))
                se = sum(1 for r in res if r.get(channel, {}).get("matches_target"))
                cells[eps][atk][ev] = (ex, se, len(res))
    return cells, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="clean",
                    help="Which delivery channel to tabulate (clean/opus32/opus64).")
    args = ap.parse_args()

    cells, skipped = load_cells(args.channel)
    if not cells:
        print("No untagged bank-prompt transfer_eval files found.")
        return

    print(f"\nS1 CROSS-MODEL TRANSFER vs PERTURBATION BUDGET   channel={args.channel}")
    print("cells are exact/semantic out of n clips. [D] = white-box diagonal (control).\n")

    width = 16
    header = f"{'eps':<6}{'attacker':<12}" + "".join(
        f"{SHORT[e]:<{width}}" for e in EVALUATORS)
    print(header)
    print("-" * len(header))

    for eps in sorted(cells):
        for atk in ATTACKERS:
            if atk not in cells[eps]:
                continue
            row = f"{eps:<6}{SHORT[atk]:<12}"
            for ev in EVALUATORS:
                c = cells[eps][atk].get(ev)
                if c is None:
                    row += f"{'--':<{width}}"
                else:
                    ex, se, n = c
                    tag = f"{ex}/{se} of {n}" + ("[D]" if ev == atk else "")
                    row += f"{tag:<{width}}"
            print(row)
        print()

    # The headline: transfer mass (all off-diagonal cells) per eps.
    print("TRANSFER MASS (all off-diagonal cells pooled, exact-match)")
    print(f"{'eps':<8}{'exact':<12}{'semantic':<12}{'cells':<8}{'diagonal (mean exact %)':<26}")
    print("-" * 66)
    for eps in sorted(cells):
        ex = se = n = 0
        diag = []
        for atk in cells[eps]:
            for ev, (e, s, cn) in cells[eps][atk].items():
                if ev == atk:
                    diag.append(100.0 * e / cn if cn else 0.0)
                else:
                    ex += e
                    se += s
                    n += cn
        dstr = f"{sum(diag)/len(diag):.1f}%  (k={len(diag)})" if diag else "--"
        print(f"{eps:<8}{ex:<12}{se:<12}{n:<8}{dstr:<26}")

    if skipped:
        print("\nSkipped (non-bank prompt, different experiment):")
        for p, pt in skipped:
            print(f"  {p}\n    prompt={pt!r}")
    print()


if __name__ == "__main__":
    main()
