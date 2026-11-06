"""Build S3b carrier_to_target.json from existing attack summary.json files.

The runner resolves adv wavs to their per-pair target via an
`AUDIO_AGENT_BENCH_CARRIER_MAP` JSON with a `"pairs"` list of dicts. Each dict
needs at minimum `{carrier, target, category}`; the pair<NNN> index embedded
in the adv wav filename is used as the index into this list.

S3b has no such file (scenario audio/ is empty; attack wavs are only under
results/...). This script walks one target-model's bundle, reads each
sibling *_summary.json, and materializes the pairs list in pair-order.

Output: data/speech/03b_copyright/carrier_to_target.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 0_all_combined/
SRC_BUNDLE = ROOT / "results/03b_copyright_bypass/qwen2_audio/eps_1.0/audio"
OUT_MAP = ROOT / "data/speech/03b_copyright/carrier_to_target.json"

_PAIR_RE = re.compile(r"_pair(\d+)_attacked\.wav$")
_CARRIER_RE = re.compile(r"^music_(.+?)__target_")


def main() -> None:
    if not SRC_BUNDLE.is_dir():
        raise SystemExit(f"no bundle at {SRC_BUNDLE}")
    wavs = sorted(SRC_BUNDLE.glob("*_attacked.wav"))
    if not wavs:
        raise SystemExit("no adv wavs found")

    pairs_by_idx: dict[int, dict] = {}
    for w in wavs:
        m_pair = _PAIR_RE.search(w.name)
        if not m_pair:
            print(f"skip (no pair idx): {w.name}")
            continue
        idx = int(m_pair.group(1))
        summary_path = w.with_name(w.name.replace("_attacked.wav", "_summary.json"))
        if not summary_path.is_file():
            print(f"skip (no summary): {w.name}")
            continue
        sj = json.loads(summary_path.read_text())
        tgt = sj.get("target_text") or sj.get("target")
        if not tgt:
            print(f"skip (no target in summary): {summary_path.name}")
            continue
        carrier_stem = ""
        mc = _CARRIER_RE.search(w.name)
        if mc:
            carrier_stem = mc.group(1)
        src_wav = sj.get("wav_path", "")
        pairs_by_idx[idx] = {
            "carrier": f"{carrier_stem}.wav",
            "carrier_clean_path": src_wav,
            "target": tgt,
            "category": "copyright_bypass",
            "clean_text": sj.get("original_output", ""),
            "pair_idx": idx,
        }

    if not pairs_by_idx:
        raise SystemExit("no pairs extracted")

    pairs = [pairs_by_idx[i] for i in sorted(pairs_by_idx)]
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAP.write_text(json.dumps({"pairs": pairs}, indent=2, ensure_ascii=False))
    print(f"wrote {len(pairs)} pairs -> {OUT_MAP.relative_to(ROOT)}")
    for p in pairs:
        print(f"  pair{p['pair_idx']:03d}  carrier={p['carrier']:<40} target={p['target'][:60]}")


if __name__ == "__main__":
    main()
