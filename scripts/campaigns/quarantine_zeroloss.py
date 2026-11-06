"""
Quarantine every pool-ladder / white-box-ceiling clip whose optimizer never took
a real step, so a resume regenerates it instead of skipping it.

`final_loss == 0.0` is not a converged attack. A cross-entropy loss is never
exactly zero on real audio; that value only ever means best_loss latched at its
initialization because no model contributed a gradient on any step that could
update it (primary_model outside the pool during warmup, or a dead out-of-env
worker under --model-eot). The wav on disk is then the random init, i.e. the
clean carrier, and both drivers resume by wav existence -- so these files would
be silently kept forever.

Nothing is deleted. Each clip moves to _quarantine_zeroloss/ beside its bundle
with a manifest recording why, and its targets.json / summary entry is pruned.

    python 0_all_combined/scripts/quarantine_zeroloss.py --dry-run
    python 0_all_combined/scripts/quarantine_zeroloss.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # 0_all_combined/
ZERO_TOL = 1e-12


def _is_latched(loss) -> bool:
    return loss is not None and abs(float(loss)) < ZERO_TOL


def scan_whitebox(bundle: Path):
    """Ceiling bundles: one *_summary.json per clip carries final_loss."""
    bad = []
    audio = bundle / "audio"
    for wav in sorted(audio.glob("*_attacked.wav")):
        summ = audio / (wav.name.replace("_attacked.wav", "_summary.json"))
        if not summ.is_file():
            bad.append((wav, None, "no summary json (clip died mid-write)"))
            continue
        j = json.loads(summ.read_text())
        if _is_latched(j.get("final_loss")):
            bad.append((wav, summ, f"final_loss={j.get('final_loss')!r}"))
    return bad


def scan_ladder(bundle: Path):
    """Ladder bundles: one targets.json holds every clip's final_loss."""
    bad = []
    audio = bundle / "audio"
    tpath = bundle / "targets.json"
    targets = json.loads(tpath.read_text()) if tpath.is_file() else {}
    for wav in sorted(audio.glob("*_attacked.wav")):
        rec = targets.get(wav.stem)
        if not isinstance(rec, dict) or "final_loss" not in rec:
            bad.append((wav, None, "no final_loss recorded (pre-schema bundle)"))
            continue
        if _is_latched(rec.get("final_loss")):
            why = f"final_loss={rec.get('final_loss')!r}"
            if rec.get("subproc_grad_failures"):
                why += f", subproc_grad_failures={rec['subproc_grad_failures']}"
            bad.append((wav, None, why))
    return bad


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    p.add_argument("--base", default=None,
                   help="Limit to one results tree (default: every results_pool_ladder*)")
    args = p.parse_args()

    bases = ([Path(args.base)] if args.base
             else sorted(ROOT.glob("results_pool_ladder*")))
    total = 0
    for base in bases:
        cells = []
        for b in sorted(base.glob("whitebox/*/*")):
            if (b / "audio").is_dir():
                cells.append((b, scan_whitebox))
        for b in sorted(base.glob("*/*/*")):
            if (b / "manifest.json").is_file():
                cells.append((b, scan_ladder))
        for bundle, scan in cells:
            bad = scan(bundle)
            n_wav = len(list((bundle / "audio").glob("*_attacked.wav")))
            rel = bundle.relative_to(ROOT)
            if not bad:
                print(f"  OK        {rel}  ({n_wav} wavs)")
                continue
            total += len(bad)
            print(f"  QUARANTINE {rel}  {len(bad)}/{n_wav} clips")
            print(f"             reason: {bad[0][2]}")
            if not args.apply:
                continue
            qdir = bundle / "_quarantine_zeroloss"
            qdir.mkdir(exist_ok=True)
            moved = []
            for wav, summ, why in bad:
                shutil.move(str(wav), str(qdir / wav.name))
                if summ is not None and summ.is_file():
                    shutil.move(str(summ), str(qdir / summ.name))
                moved.append({"wav": wav.name, "reason": why})
            (qdir / "QUARANTINE.json").write_text(json.dumps({
                "why": "final_loss == 0.0: optimizer never took a contributing "
                       "step, so this wav is the initialization, not an attack. "
                       "See 0_all_combined/scripts/quarantine_zeroloss.py.",
                "n_quarantined": len(moved),
                "clips": moved,
            }, indent=2))
            # Prune the pruned clips out of targets.json so the resume rewrites them.
            tpath = bundle / "targets.json"
            if tpath.is_file():
                targets = json.loads(tpath.read_text())
                for wav, _s, _w in bad:
                    targets.pop(wav.stem, None)
                tpath.write_text(json.dumps(targets, indent=2))
    print(f"\n{'moved' if args.apply else 'would move'} {total} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
