"""SNAC v0.5 — attack-only resume via per-pair subprocess.

Spawns one fresh Python process per pair so the OS reclaims GPU memory
between attacks. Each child loads SNAC + Qwen2.5-Omni-7B, attacks one
pair, saves, exits. Trade ~30 s per-pair model-load overhead for
bulletproof memory hygiene on a 40 GB GPU running this close to the limit.
"""
import sys
import time
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path("${REPO_ROOT}/0_all_combined/scripts")
ONE_PAIR = SCRIPTS_DIR / "_snac_v05_attack_one_pair.py"

sys.path.insert(0, str(SCRIPTS_DIR))
from run_snac_v05 import PAIRS, OUT_DIR, lookup_pair


def main() -> int:
    audio_dir = OUT_DIR / "audio"
    missing = []
    for pair_idx in PAIRS:
        try:
            summary = lookup_pair(pair_idx)
        except Exception:
            continue
        carrier_stem = Path(summary["wav_path"]).stem
        wav = audio_dir / f"music_{carrier_stem}__pair{pair_idx:03d}_attacked.wav"
        if not wav.exists():
            missing.append(pair_idx)

    print("=" * 70, flush=True)
    print("SNAC v0.5 — resume (per-pair subprocess)", flush=True)
    print(f"  total pairs:    {len(PAIRS)}", flush=True)
    print(f"  already done:   {len(PAIRS) - len(missing)}", flush=True)
    print(f"  to attack:      {len(missing)}  -> {missing}", flush=True)
    print("=" * 70, flush=True)
    if not missing:
        print("[done] nothing to do.", flush=True)
        return 0

    t_total = time.time()
    failed = []
    for i, pair_idx in enumerate(missing, 1):
        print(f"\n{'-'*70}\n[{i}/{len(missing)}] pair {pair_idx:03d}", flush=True)
        t0 = time.time()
        try:
            r = subprocess.run(
                [sys.executable, "-u", str(ONE_PAIR), str(pair_idx)],
                check=False,
            )
            elapsed = time.time() - t0
            if r.returncode == 0:
                print(f"  -> ok ({elapsed:.0f}s)", flush=True)
            else:
                print(f"  -> FAIL rc={r.returncode} ({elapsed:.0f}s)", flush=True)
                failed.append(pair_idx)
        except Exception as e:
            print(f"  -> EXCEPTION launching child: {e}", flush=True)
            failed.append(pair_idx)

    print(f"\n{'='*70}", flush=True)
    print(f"[resume done] {len(missing) - len(failed)}/{len(missing)} pairs attacked "
          f"in {time.time()-t_total:.0f}s", flush=True)
    if failed:
        print(f"  failures: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
