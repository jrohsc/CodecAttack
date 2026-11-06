#!/usr/bin/env python3
"""Measure speech intelligibility and PESQ on the completed paired S1 WAVs."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pystoi import stoi
from pesq import pesq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviewer_control_outputs/s1_baseline_full40_seed1"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    records = [json.loads(x) for x in (OUT / "evaluation_records.jsonl").read_text().splitlines()]
    refs = {r["example_id"]: r for r in records if r["method"] == "latent_codecattack"}
    assert len(refs) == 40 and len(records) == 160
    rows = []
    for r in records:
        ref, sr = sf.read(resolve(refs[r["example_id"]]["carrier_ref_path"]), dtype="float64")
        adv, sr2 = sf.read(resolve(r["adv_path"]), dtype="float64")
        assert sr == sr2 == 24000 and len(ref) == len(adv) == 240000
        ref16 = resample_poly(ref, 2, 3)
        adv16 = resample_poly(adv, 2, 3)
        rows.append({"example_id":r["example_id"],"method":r["method"],
                     "stoi":float(stoi(ref16,adv16,16000,extended=False)),
                     "pesq_wb":float(pesq(16000,ref16,adv16,"wb"))})
    with open(OUT / "speech_quality.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    grouped=defaultdict(list)
    for row in rows:grouped[row["method"]].append(row)
    summary={m:{metric:{"mean":float(np.mean([r[metric] for r in rr])),
                         "median":float(np.median([r[metric] for r in rr])),
                         "min":float(np.min([r[metric] for r in rr]))}
                for metric in ("stoi","pesq_wb")}
             for m,rr in grouped.items()}
    (OUT / "speech_quality_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
