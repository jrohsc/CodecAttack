import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from rebuttal.run_quality_matched import CHANNELS, distance, next_epsilon


def metrics(pesq=2.0, stoi=.9):
    return {c: dict(pesq_wb=pesq, stoi=stoi, snr_db=8) for c in CHANNELS}


def test_each_metric_and_channel_must_match():
    target = metrics()
    assert distance(metrics(2.05, .91), target) < 1
    candidate = metrics()
    candidate['opus128']['stoi'] = .85
    assert distance(candidate, target) > 1
    candidate = metrics()
    candidate['clean']['pesq_wb'] = float('nan')
    assert distance(candidate, target) == float('inf')


def test_search_brackets_quality_without_success_fields():
    target = metrics()
    assert next_epsilon([], target, .064) == .001
    a = dict(epsilon=.001, quality=metrics(3.0), attack_success=False)
    assert next_epsilon([a], target, .064) == .128
    b = dict(epsilon=.128, quality=metrics(1.0), attack_success=True)
    result = next_epsilon([a, b], target, .064)
    a['attack_success'], b['attack_success'] = True, False
    assert next_epsilon([a, b], target, .064) == result
    assert .001 < result < .128
