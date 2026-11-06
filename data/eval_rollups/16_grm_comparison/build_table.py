"""
Render the GRM delivery ladder from results/*_scored.jsonl.

Retention, not raw JSR, is the comparable quantity: GRM optimises an UNTARGETED
affirmative-prefix jailbreak scored by HarmBench, while CodecAttack optimises a
TARGETED transcript. Their success rates are different measurements and must not
be differenced. What is comparable across the two is how much of each attack
survives a transformation:

    retention_inversion = JSR(wav)   / JSR(mel)   cost of becoming real audio
    retention_codec     = JSR(codec) / JSR(wav)   cost of the channel

The codec retention denominator is the WAVEFORM rung, never the Mel rung, so
Griffin-Lim's loss is never billed to the codec.

Clean-side rungs are printed beside the attack rungs: a codec that moves the
CLEAN JSR is changing the model's behaviour generally, not disarming an attack.
"""

import json
from collections import OrderedDict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
CODECS = ["opus64", "opus32", "mp3_64", "aac_64"]


def load():
    out = {}
    for f in sorted(R.glob("*_scored.jsonl")):
        recs = [json.loads(l) for l in open(f)]
        if not recs:
            continue
        model, rung = f.name.replace("_scored.jsonl", "").split("__", 1)
        n = len(recs)
        h = sum(1 for r in recs if r["harmful"])
        empty = sum(1 for r in recs if not r["response"].strip())
        out[(model, rung)] = {"n": n, "h": h, "jsr": 100.0 * h / n, "empty": empty}
    return out


def pct(x):
    return "  --  " if x is None else f"{x:6.2f}"


def ret(num, den):
    if num is None or den is None or den["jsr"] <= 0:
        return "   --"
    return f"{100.0 * num['jsr'] / den['jsr']:5.1f}"


def main():
    d = load()
    if not d:
        raise SystemExit(f"no scored files in {R}")
    models = sorted({m for m, _ in d}, key=lambda m: (m != "qwen2_audio", m))

    for m in models:
        g = lambda r: d.get((m, r))
        mel_grm, wav_grm = g("mel_grm"), g("wav_grm_gl")
        role = "WHITE-BOX (surrogate)" if m == "qwen2_audio" else "TRANSFER (held-out)"
        print("=" * 88)
        print(f"{m}   {role}")
        print("=" * 88)
        print(f"{'rung':22s} {'JSR%':>6s} {'n':>4s} {'empty':>6s} | "
              f"{'clean JSR%':>10s} | {'ret_codec':>9s}")
        rows = [("mel_clean", "mel_clean"), ("mel_grm", "mel_clean"),
                ("wav_raw", "wav_raw"), ("wav_clean_gl", "wav_clean_gl"),
                ("wav_grm_gl", "wav_clean_gl")]
        rows += [(f"wav_grm_gl__{c}", f"wav_clean_gl__{c}") for c in CODECS]
        for rung, clean_rung in rows:
            v = g(rung)
            if v is None:
                continue
            cv = g(clean_rung)
            r = ret(v, wav_grm) if rung.startswith("wav_grm_gl__") else "   --"
            print(f"{rung:22s} {pct(v['jsr'])} {v['n']:4d} {v['empty']:6d} | "
                  f"{pct(cv['jsr']) if cv else '  --  ':>10s} | {r:>9s}")
        if mel_grm and wav_grm:
            print(f"\n  retention_inversion (wav/mel) = {ret(wav_grm, mel_grm)}%   "
                  f"[{mel_grm['jsr']:.2f}% -> {wav_grm['jsr']:.2f}%]")
        print()

    print("=" * 88)
    print("2x2: does the attack survive delivery?   (JSR%, retention_codec in brackets)")
    print("=" * 88)
    print(f"{'':26s} {'clean (wav)':>12s} " + " ".join(f"{c:>14s}" for c in CODECS))
    for m in models:
        g = lambda r: d.get((m, r))
        base = g("wav_grm_gl")
        if base is None:
            continue
        role = "white-box" if m == "qwen2_audio" else "transfer"
        cells = []
        for c in CODECS:
            v = g(f"wav_grm_gl__{c}")
            cells.append("            --" if v is None
                         else f"{v['jsr']:6.2f} [{ret(v, base).strip():>4}]")
        print(f"{m + ' (' + role + ')':26s} {base['jsr']:12.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
