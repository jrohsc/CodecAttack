import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from tqdm import tqdm

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None


def load_rel_paths(eval_file: str) -> List[str]:
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval_file must be a JSON list, e.g. [[rel_path, transcript], ...]")

    rel_paths = [x[0] for x in data if isinstance(x, (list, tuple)) and len(x) >= 1 and isinstance(x[0], str)]
    if not rel_paths:
        raise RuntimeError("No valid relative paths found in eval_file")
    return rel_paths


def load_wav(path: str, sr: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def align_pair(clean: np.ndarray, adv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = min(len(clean), len(adv))
    if m <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return clean[:m], adv[:m]


def compute_metrics(clean: np.ndarray, adv: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    try:
        p = float(pesq(sr, clean, adv, "wb"))
    except Exception:
        p = None

    try:
        s = float(stoi(clean, adv, sr, extended=False))
    except Exception:
        s = None

    return p, s


def resolve_adv_path(adv_root: str, rel: str) -> str | None:
    path_struct = os.path.join(adv_root, rel)
    if os.path.exists(path_struct):
        return path_struct

    # fallback for flat directory: adv_root/libri_000.wav
    basename = os.path.basename(rel)
    path_flat = os.path.join(adv_root, basename)
    if os.path.exists(path_flat):
        return path_flat

    # fallback for suffix/prefix variants:
    # - AMSE: libri_000_trump_S0.5_E10_Nmachine.wav (prefix match)
    # - BoN: bon_000.wav (prefix match by index)
    stem = os.path.splitext(basename)[0]
    candidates = []
    try:
        for name in os.listdir(adv_root):
            if not name.lower().endswith(".wav"):
                continue

            if name.startswith(stem):
                candidates.append(os.path.join(adv_root, name))
                continue

            # match by last numeric index: libri_000 -> bon_000
            m_rel = re.search(r"(\d+)$", stem)
            if m_rel:
                idx = m_rel.group(1)
                if name.startswith(f"bon_{idx}"):
                    candidates.append(os.path.join(adv_root, name))
    except FileNotFoundError:
        return None

    if candidates:
        return candidates[0]

    return None


def evaluate_one_root(rel_paths: List[str], clean_root: str, adv_root: str, sr: int) -> Dict:
    results = []
    pesq_vals, stoi_vals = [], []
    miss_clean, miss_adv = 0, 0

    for rel in tqdm(rel_paths, desc=f"PESQ/STOI [{os.path.basename(adv_root) or adv_root}]"):
        clean_path = os.path.join(clean_root, rel)
        if not os.path.exists(clean_path):
            miss_clean += 1
            continue

        adv_path = resolve_adv_path(adv_root, rel)
        if adv_path is None:
            miss_adv += 1
            continue

        clean, adv = align_pair(load_wav(clean_path, sr), load_wav(adv_path, sr))
        p, s = compute_metrics(clean, adv, sr)

        if p is not None:
            pesq_vals.append(p)
        if s is not None:
            stoi_vals.append(s)

        results.append(
            {
                "rel_path": rel,
                "adv_path": adv_path,
                "pesq": p,
                "stoi": s,
                "num_samples": int(clean.shape[0]),
            }
        )

    summary = {
        "adv_root": adv_root,
        "sr": sr,
        "pesq_mode": "wb",
        "num_total": len(rel_paths),
        "num_missing_clean": miss_clean,
        "num_missing_adv": miss_adv,
        "num_scored": len(results),
        "pesq_mean": float(np.mean(pesq_vals)) if pesq_vals else None,
        "pesq_std": float(np.std(pesq_vals)) if pesq_vals else None,
        "stoi_mean": float(np.mean(stoi_vals)) if stoi_vals else None,
        "stoi_std": float(np.std(stoi_vals)) if stoi_vals else None,
    }

    return {"summary": summary, "results": results}


def parse_adv_roots(args) -> Dict[str, str]:
    if args.adv_roots_json:
        try:
            obj = json.loads(args.adv_roots_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --adv_roots_json: {e}")
        if not isinstance(obj, dict) or not obj:
            raise ValueError("--adv_roots_json must be a non-empty JSON object: {name: path}")
        return {str(k): str(v) for k, v in obj.items()}

    return {"default": args.adv_root}


def main() -> None:
    parser = argparse.ArgumentParser("Compute PESQ/STOI between clean wav and adv wav")
    parser.add_argument("--eval_file", default=".${DATASETS_ROOT}/libri_500.json")
    parser.add_argument("--adv_root", default=".${DATASETS_ROOT}/libri_500_adv", help="Single adv wav root")
    parser.add_argument("--adv_roots_json",default='{"ours":".${DATASETS_ROOT}/libri_500_full"}',help="JSON map of {name: adv_root}. If set, overrides --adv_root.")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", default="pesq_stoi_results.json")
    args = parser.parse_args()

    if pesq is None or stoi is None:
        missing = [name for name, mod in [("pesq", pesq), ("pystoi", stoi)] if mod is None]
        raise ImportError(f"Missing packages: {', '.join(missing)}. Install with: pip install pesq pystoi")

    rel_paths = load_rel_paths(args.eval_file)
    if args.max_samples is not None:
        rel_paths = rel_paths[: args.max_samples]

    clean_root = os.path.dirname(args.eval_file)
    adv_roots = parse_adv_roots(args)

    all_runs = {}
    for name, root in adv_roots.items():
        payload = evaluate_one_root(rel_paths=rel_paths, clean_root=clean_root, adv_root=root, sr=args.sr)
        payload["summary"]["eval_file"] = args.eval_file
        payload["summary"]["name"] = name
        all_runs[name] = payload

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_runs}, f, ensure_ascii=False, indent=2)

    print("=== Summary ===")
    for name, payload in all_runs.items():
        s = payload["summary"]
        print(
            f"[{name}] scored={s['num_scored']}/{s['num_total']} | "
            f"PESQ={s['pesq_mean']} | STOI={s['stoi_mean']} | miss_adv={s['num_missing_adv']}"
        )
    print(f"Saved -> {args.output_json}")


if __name__ == "__main__":
    main()
