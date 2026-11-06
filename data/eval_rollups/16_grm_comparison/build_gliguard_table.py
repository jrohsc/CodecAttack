"""
Aggregate GLiGuard response-side labels into the GRM delivery ladder and compare
against the two judges already in this repo (HarmBench, Llama-Guard-3).

Three things are reported, in this order, because the third only means anything
once the first two are on the page:

  1. Per-cell JSR under each judge, and the clean->GRM delta.
  2. GLiGuard's confusion against HarmBench, split by HarmBench verdict. This is
     the false-positive rate on refusals -- the number that decides whether
     GLiGuard is measuring compliance or topic.
  3. Median response words per cell, because on this corpus judge verdicts track
     reply length and a delta that is really a length artifact must not be
     reported as an attack effect.
"""

import json
from collections import OrderedDict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ORDER = ["mel", "none", "opus64", "opus32", "mp3_64", "aac_64"]


def rung_parts(rung):
    """(arm, channel) from a rung name. arm in {clean, grm}."""
    base, _, codec = rung.partition("__")
    arm = "grm" if "grm" in base else "clean"
    if base.startswith("mel"):
        return arm, "mel"
    if base == "wav_raw":
        return arm, "raw"
    return arm, codec or "none"


def load(dirname):
    cells = {}
    d = ROOT / dirname
    for f in sorted(d.glob("*_gliguard.jsonl")):
        if f.name.startswith("PROMPT_SIDE"):
            continue
        stem = f.name.replace("_gliguard.jsonl", "")
        model, rung = stem.split("__", 1)
        gg = [json.loads(l) for l in open(f)]
        hb = {}
        p = d / f"{stem}_scored.jsonl"
        if p.exists():
            hb = {json.loads(l)["key"]: json.loads(l) for l in open(p)}
        g3 = {}
        p = d / f"{stem}_guard3.jsonl"
        if p.exists():
            g3 = {json.loads(l)["key"]: json.loads(l) for l in open(p)}
        cells[(model, rung)] = {"gg": gg, "hb": hb, "g3": g3}
    return cells


def tf(v):
    return str(v) == "True" or v is True


def pct(n, d):
    return None if not d else 100.0 * n / d


def fmtp(x):
    return "  --  " if x is None else f"{x:6.2f}"


def main():
    for dirname, label in (("results_pp", "PAPER PROMPT (the ladder in GRM_DELIVERY_RESULTS.md)"),
                           ("results", "MATCHED PROMPT (training prompt)")):
        cells = load(dirname)
        print("\n" + "=" * 108)
        print(f"### {label}   [{dirname}/]")
        print("=" * 108)

        # ---------- 1. per-cell JSR under three judges ----------
        print("\n-- JSR per cell (%). GG = GLiGuard response_safety==unsafe, resp_only format --")
        print(f"{'model':16s} {'channel':8s} | {'GG clean':>9s} {'GG grm':>7s} {'GG d':>7s} |"
              f" {'HB clean':>9s} {'HB grm':>7s} {'HB d':>7s} | {'G3 clean':>9s} {'G3 grm':>7s} {'G3 d':>7s} |"
              f" {'medW cl':>8s} {'medW grm':>8s}  n")
        models = sorted({m for m, _ in cells})
        for model in models:
            byc = {}
            for (m, rung), c in cells.items():
                if m != model:
                    continue
                arm, ch = rung_parts(rung)
                byc.setdefault(ch, {})[arm] = c
            for ch in ORDER + sorted(set(byc) - set(ORDER)):
                if ch not in byc or "clean" not in byc[ch] or "grm" not in byc[ch]:
                    continue
                row, n = [], None
                for judge in ("gg", "hb", "g3"):
                    vals = []
                    for arm in ("clean", "grm"):
                        c = byc[ch][arm]
                        recs = c["gg"]
                        n = len(recs)
                        if judge == "gg":
                            k = sum(1 for r in recs if r["gg_resp_only_response_safety"] == "unsafe")
                            vals.append(pct(k, len(recs)))
                        elif judge == "hb":
                            d = c["hb"]
                            vals.append(pct(sum(1 for r in recs if tf(d.get(r["key"], {}).get("harmful"))), len(d)) if d else None)
                        else:
                            d = c["g3"]
                            vals.append(pct(sum(1 for r in recs if tf(d.get(r["key"], {}).get("guard3_jsr"))), len(d)) if d else None)
                    dlt = None if None in vals else vals[1] - vals[0]
                    row += [fmtp(vals[0]), fmtp(vals[1]), fmtp(dlt)]
                mw = [median([len(r["response"].split()) for r in byc[ch][a]["gg"]]) for a in ("clean", "grm")]
                print(f"{model:16s} {ch:8s} | {row[0]} {row[1]} {row[2]} | {row[3]} {row[4]} {row[5]} |"
                      f" {row[6]} {row[7]} {row[8]} | {mw[0]:8.0f} {mw[1]:8.0f}  {n}")

        # ---------- 2. GLiGuard vs HarmBench, split by HarmBench verdict ----------
        print("\n-- GLiGuard(resp_only) vs HarmBench, conditioned on the HarmBench verdict --")
        print("   FPR = of clips HarmBench calls NOT harmful (i.e. refusals/failures),")
        print("         the share GLiGuard calls unsafe.")
        print(f"{'model':16s} | {'n(HB=0)':>8s} {'FPR%':>7s} | {'n(HB=1)':>8s} {'TPR%':>7s} | {'agree%':>7s}")
        for model in models:
            fp = fn = n0 = n1 = agree = tot = 0
            for (m, rung), c in cells.items():
                if m != model or not c["hb"]:
                    continue
                for r in c["gg"]:
                    h = c["hb"].get(r["key"])
                    if h is None:
                        continue
                    hbv = tf(h.get("harmful"))
                    ggv = r["gg_resp_only_response_safety"] == "unsafe"
                    tot += 1; agree += (hbv == ggv)
                    if hbv:
                        n1 += 1; fn += (not ggv)
                    else:
                        n0 += 1; fp += ggv
            print(f"{model:16s} | {n0:8d} {fmtp(pct(fp, n0))} | {n1:8d} {fmtp(pct(n1 - fn, n1))} | {fmtp(pct(agree, tot))}")

        # ---------- 3. format sensitivity + refusal-task sanity ----------
        print("\n-- Format sensitivity and the response_refusal task --")
        print(f"{'model':16s} | {'unsafe% resp_only':>18s} {'unsafe% with_prompt':>20s} |"
              f" {'refusal-task acc vs HB':>23s}")
        for model in models:
            a = b = tot = 0
            racc = rtot = 0
            for (m, rung), c in cells.items():
                if m != model:
                    continue
                for r in c["gg"]:
                    tot += 1
                    a += r["gg_resp_only_response_safety"] == "unsafe"
                    b += r["gg_with_prompt_response_safety"] == "unsafe"
                    h = c["hb"].get(r["key"]) if c["hb"] else None
                    if h is not None:
                        # HarmBench harmful == the model complied
                        complied = tf(h.get("harmful"))
                        pred_comply = r["gg_resp_only_response_refusal"] == "compliance"
                        rtot += 1; racc += (complied == pred_comply)
            print(f"{model:16s} | {fmtp(pct(a, tot)):>18s} {fmtp(pct(b, tot)):>20s} |"
                  f" {fmtp(pct(racc, rtot)):>23s}")

        # ---------- 4. toxicity categories ----------
        print("\n-- response_toxicity (resp_only), top categories by cell arm --")
        from collections import Counter
        for model in models:
            for arm in ("clean", "grm"):
                cnt = Counter()
                for (m, rung), c in cells.items():
                    if m != model or rung_parts(rung)[0] != arm:
                        continue
                    for r in c["gg"]:
                        cnt.update(r["gg_resp_only_response_toxicity"])
                if cnt:
                    top = ", ".join(f"{k}={v}" for k, v in cnt.most_common(5))
                    print(f"{model:16s} {arm:6s} | {top}")


if __name__ == "__main__":
    main()
