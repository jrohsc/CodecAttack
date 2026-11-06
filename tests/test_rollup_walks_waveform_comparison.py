"""Regression: rollup_results.main() must descend into scen/waveform_comparison/<model>/<bundle>/."""
import json
import subprocess
import sys
from pathlib import Path


def _minimal_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event": "attack", "scenario": "sx", "target_model": "m1",
         "eps": 0.1, "carrier": "c.wav", "status": "ok"},
        {"event": "eval", "scenario": "sx", "attack_target_model": "m1",
         "eval_model": "m1", "eps": 0.1, "channel_tag": "clean",
         "carrier": "c.wav", "adv_audio_path": "x",
         "metrics": {"target_substring_match": True, "exact_match": True,
                     "wer": 0.0, "snr_db": 80.0}},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_rollup_walks_waveform_comparison_subtree(tmp_path):
    root = tmp_path / "all"
    (root / "results").mkdir(parents=True)
    scripts = root / "scripts"
    scripts.mkdir()

    here = Path(__file__).resolve().parent.parent
    src = here / "scripts" / "rollup_results.py"
    dst = scripts / "rollup_results.py"
    dst.write_text(src.read_text())

    _minimal_jsonl(root / "results/sx/m1/eps_0.1/results.jsonl")
    _minimal_jsonl(root / "results/sx/waveform_comparison/m1/eps_0.1/results.jsonl")

    r = subprocess.run([sys.executable, str(dst)], cwd=str(root))
    assert r.returncode == 0

    top = json.loads((root / "results/rollup.json").read_text())
    names = {(row["scenario"], row["attack_target_model"], row["bundle"])
             for row in top["rows"]}
    assert ("sx", "m1", "eps_0.1") in names, names
    assert ("sx", "m1__waveform", "eps_0.1") in names, names
