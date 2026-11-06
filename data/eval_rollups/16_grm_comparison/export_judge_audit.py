"""
Export every clip with all three judge verdicts so the false positives can be
audited by hand. Nothing here is filtered or adjudicated -- the point is that
the reader can disagree with GLIGUARD_RESULTS.md, including with its choice of
HarmBench as the reference judge.

Writes to audit/:
  judge_audit.jsonl   every clip, full response text, all three verdicts
  judge_audit.csv     same, response truncated to 400 chars for spreadsheets
  FALSE_POSITIVES.md  the disagreement buckets with worked examples
"""
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "audit"; OUT.mkdir(exist_ok=True)
CAP = re.compile(r"^\s*(the audio|the speech|the speaker|this audio|the clip|the recording)", re.I)

def tf(v): return str(v) == "True" or v is True

rows = []
for d in ("results", "results_pp"):
    for f in sorted((ROOT / d).glob("*_gliguard.jsonl")):
        if f.name.startswith("PROMPT_SIDE"):
            continue
        stem = str(f).replace("_gliguard.jsonl", "")
        model, rung = f.name.replace("_gliguard.jsonl", "").split("__", 1)
        try:
            hb = {json.loads(l)["key"]: json.loads(l) for l in open(stem + "_scored.jsonl")}
        except FileNotFoundError:
            hb = {}
        try:
            g3 = {json.loads(l)["key"]: json.loads(l) for l in open(stem + "_guard3.jsonl")}
        except FileNotFoundError:
            g3 = {}
        for l in open(f):
            r = json.loads(l); k = r["key"]
            H = hb.get(k); G = g3.get(k)
            rows.append({
                "prompt_arm": "paper" if d == "results_pp" else "matched",
                "model": model, "rung": rung, "key": k, "behavior": r["behavior"],
                "words": len(r["response"].split()),
                "is_caption": bool(CAP.match(r["response"])),
                "HB_harmful":  None if H is None else tf(H["harmful"]),
                "G3_jsr":      None if G is None else tf(G["guard3_jsr"]),
                "G3_raw":      None if G is None else G.get("guard3_raw"),
                "GG_unsafe":   r["gg_resp_only_response_safety"] == "unsafe",
                "GG_conf":     r["gg_resp_only_response_safety_conf"],
                "GG_toxicity": r["gg_resp_only_response_toxicity"],
                "GG_unsafe_with_prompt": r["gg_with_prompt_response_safety"] == "unsafe",
                "GG_refusal_task": r["gg_resp_only_response_refusal"],
                "response": r["response"],
            })

def pattern(r):
    if None in (r["HB_harmful"], r["G3_jsr"]):
        return "incomplete"
    v = (r["HB_harmful"], r["G3_jsr"], r["GG_unsafe"])
    if len(set(v)) == 1:
        return "unanimous_harmful" if v[0] else "unanimous_benign"
    names = ["HB", "G3", "GG"]
    for i, n in enumerate(names):
        o = [v[j] for j in range(3) if j != i]
        if o[0] == o[1] and v[i] != o[0]:
            return f"{n}_alone_{'harmful' if v[i] else 'benign'}"
    return "split"

for r in rows:
    r["pattern"] = pattern(r)

with open(OUT / "judge_audit.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

cols = ["prompt_arm","model","rung","key","pattern","HB_harmful","G3_jsr","GG_unsafe",
        "GG_conf","GG_toxicity","is_caption","words","behavior","response"]
with open(OUT / "judge_audit.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({**r, "response": r["response"][:400],
                    "GG_toxicity": "|".join(r["GG_toxicity"])})

from collections import Counter
c = Counter(r["pattern"] for r in rows)
lines = ["# Judge disagreements — audit set\n",
         f"{len(rows)} clips. `judge_audit.jsonl` has full responses; `judge_audit.csv` truncates to 400 chars.\n",
         "HarmBench is used as the reference in `GLIGUARD_RESULTS.md`. It is included here on",
         "the same footing as the others so that choice can be checked rather than assumed.\n",
         "## Buckets\n", "| pattern | n | meaning |", "|---|---|---|"]
MEAN = {
 "unanimous_benign": "all three say not harmful",
 "unanimous_harmful": "all three say harmful",
 "GG_alone_harmful": "**GLiGuard false positive** — refusal vocabulary (§3)",
 "HB_alone_benign": "**Guard-3 + GLiGuard false positive** — mostly AF3 captions (§A.2)",
 "G3_alone_harmful": "**Guard-3 false positive**",
 "GG_alone_benign": "GLiGuard miss",
 "G3_alone_benign": "Guard-3 miss",
 "HB_alone_harmful": "**HarmBench alone says harmful** — check these to audit HarmBench itself",
 "incomplete": "a judge is missing for this cell",
}
for k, v in c.most_common():
    lines.append(f"| `{k}` | {v} | {MEAN.get(k,'')} |")

def sample(pat, n, title, note):
    sub = [r for r in rows if r["pattern"] == pat]
    lines.append(f"\n## {title}\n\n`pattern == \"{pat}\"` — n={len(sub)}. {note}\n")
    sub = sorted(sub, key=lambda r: -r["words"])[:n*3:3]
    for r in sub[:n]:
        lines.append(f"**{r['model']} / {r['rung']} / {r['key']}** ({r['words']}w) — "
                     f"HB={int(r['HB_harmful'])} G3={int(r['G3_jsr'])} GG={int(r['GG_unsafe'])}"
                     f"{' · caption' if r['is_caption'] else ''}")
        lines.append(f"> *Request:* {r['behavior']}")
        lines.append(f">\n> {r['response'][:600].strip()}\n")

sample("HB_alone_benign", 4, "Guard-3 + GLiGuard call it harmful, HarmBench does not",
       "Read these first: if these are compliance, HarmBench is wrong and §A.2 falls.")
sample("GG_alone_harmful", 4, "GLiGuard alone calls it harmful",
       "Expected to be refusals that explain themselves.")
sample("G3_alone_harmful", 3, "Guard-3 alone calls it harmful", "")
sample("HB_alone_harmful", 3, "HarmBench alone calls it harmful",
       "The audit of HarmBench: if these are NOT compliance, HarmBench over-fires too.")

(OUT / "FALSE_POSITIVES.md").write_text("\n".join(lines))
print(f"wrote {len(rows)} clips ->", OUT)
for k, v in c.most_common():
    print(f"  {k:24s} {v:5d}")
