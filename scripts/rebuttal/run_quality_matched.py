"""Fresh waveform controls selected by saved-audio quality, never attack outcomes.

Historical released controls are deliberately not an input. An unmatched case
remains in the ledger and cannot be described as a quality-matched result.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from rebuttal.run_followup import read, save, write_rows

CHANNELS = ('clean', 'opus64', 'opus128')
TOLERANCES = {'pesq_wb': 0.10, 'stoi': 0.02}
METHODS = ('tuned_waveform', 'bandlimited_4khz')
BASE_EPS = {'tuned_waveform': 0.06082, 'bandlimited_4khz': 0.166}
MAX_CANDIDATES = 9


def quality(row):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from pesq import pesq
    from pystoi import stoi
    from reviewer_controls.common import codec_roundtrip, snr_db
    ref, sr = sf.read(row['carrier_ref_path'], dtype='float64')
    adv, sr2 = sf.read(row['adv_path'], dtype='float64')
    if sr != sr2 or ref.shape != adv.shape or ref.ndim != 1:
        raise ValueError('Audio reference/rate/alignment mismatch')
    if not np.isfinite(ref).all() or not np.isfinite(adv).all():
        raise ValueError('Nonfinite audio')
    g = math.gcd(sr, 16000)
    rr = resample_poly(ref, 16000 // g, sr // g)
    result = {}
    for channel in CHANNELS:
        delivered = adv if channel == 'clean' else codec_roundtrip(adv, sr, 'opus', int(channel[4:]))
        aa = resample_poly(delivered, 16000 // g, sr // g)
        result[channel] = dict(pesq_wb=float(pesq(16000, rr, aa, 'wb')),
                               stoi=float(stoi(rr, aa, 16000)),
                               snr_db=snr_db(ref, delivered))
    return result


def distance(candidate, reference):
    """Worst normalized quality gap; <=1 requires EVERY metric/channel to pass."""
    gaps = [abs(candidate[c][m] - reference[c][m]) / tol
            for c in CHANNELS for m, tol in TOLERANCES.items()]
    return max(gaps) if all(math.isfinite(v) for v in gaps) else float('inf')


def next_epsilon(candidates, target, base):
    """Quality-only log bracketing, with largest unexplored interval fallback.

    Endpoints are always measured. PESQ need not be monotonic: all observations
    remain available and a failed bracket does not imply equivalence.
    """
    low, high = base / 64, base * 2
    if not candidates:
        return low
    if len(candidates) == 1:
        return high
    points = sorted((x['epsilon'], sum(x['quality'][c]['pesq_wb'] - target[c]['pesq_wb']
                                     for c in CHANNELS) / len(CHANNELS)) for x in candidates)
    intervals = [(a, b) for a, b in zip(points, points[1:])]
    crossings = [(a, b) for a, b in intervals if a[1] * b[1] <= 0]
    a, b = max(crossings or intervals, key=lambda ab: math.log(ab[1][0] / ab[0][0]))
    return math.sqrt(a[0] * b[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['match', 'gate', 'reference', 'evaluate'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--development', type=Path, required=True)
    ap.add_argument('--split', choices=['development', 'evaluation'], default='development')
    ap.add_argument('--seed', type=int, choices=[1, 7, 42], default=1)
    ap.add_argument('--method', choices=METHODS)
    ap.add_argument('--alpha-ratio', type=float, default=.02)
    ap.add_argument('--perceptual-weight', type=float, default=0.)
    ap.add_argument('--retain-unmatched', action='store_true',
                    help='Run the full evaluation cohort even if development matching fails; retain all quality flags.')
    a = ap.parse_args()
    if not math.isfinite(a.alpha_ratio) or a.alpha_ratio <= 0 or not math.isfinite(a.perceptual_weight) or a.perceptual_weight < 0:
        raise ValueError('Invalid optimization settings')
    out = a.output.resolve()
    work = out / a.split / f'seed{a.seed}'
    work.mkdir(parents=True, exist_ok=True)
    def run(label, script, *args):
        with (work / (label + '.log')).open('a') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, args)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if a.stage == 'gate':
        checks = {m: json.loads((out / 'development/seed1' / m / 'gate.json').read_text()) for m in METHODS}
        passed = all(v['passed'] for v in checks.values())
        save(out / 'development_gate.json', dict(passed=passed, methods=checks))
        if not passed:
            raise RuntimeError('Quality equivalence not achieved for every development case; held-out chain blocked')
        return
    if a.split == 'evaluation' and not a.retain_unmatched:
        gate_path = out / 'development_gate.json'
        if not gate_path.exists() or not json.loads(gate_path.read_text())['passed']:
            raise RuntimeError('Development quality gate has not passed')
    policy_path = work / 'cohort_policy.json'
    policy = dict(retain_unmatched=a.retain_unmatched, quality_tolerances=TOLERANCES,
                  selection_uses_attack_outcomes=False, exclude_unmatched_from_denominator=False)
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise RuntimeError('Cohort quality policy changed on resume')
    save(policy_path, policy)
    latent = (a.development.resolve() / 'cells/latent_final_iterate_ar0.05/manifest.jsonl'
              if a.split == 'development' else work / 'latent/manifest.jsonl')
    if a.stage == 'reference':
        if a.split != 'evaluation':
            raise ValueError('Development reference is already frozen')
        run('latent', 'rebuttal/run_decision_latent.py', '--split', a.split, '--alpha', .05,
            '--seed', a.seed, '--output', work / 'latent')
        return
    refs = read(latent)
    if len(refs) != (10 if a.split == 'development' else 40) or len({r['example_id'] for r in refs}) != len(refs):
        raise ValueError('Incomplete or duplicate latent reference cohort')
    if a.stage == 'evaluate':
        manifests = [latent] + [work / m / 'final_manifest.jsonl' for m in METHODS]
        controls = [{**r, 'method': 'raw_carrier_control', 'epsilon': 0, 'alpha': 0,
                     'seed': None, 'steps': 0, 'adv_path': r['carrier_ref_path'], 'snr_db': float('inf')} for r in refs]
        write_rows(work / 'clean_manifest.jsonl', controls)
        run('delivery', 'reviewer_controls/evaluate.py', '--manifest', *manifests,
            work / 'clean_manifest.jsonl', '--output', work / 'evaluation.jsonl', '--resume')
        run('quality', 'rebuttal/measure_quality.py', '--manifest', *manifests, '--output', work / 'quality.jsonl')
        save(work / 'completion.json', dict(complete=True, quality_gates={m: json.loads((work/m/'gate.json').read_text()) for m in METHODS},
             note='All cases retained. Unmatched cases are not quality-equivalent; do not filter to successful matches.'))
        return
    if not a.method:
        raise ValueError('--method required for matching')
    dest = work / a.method
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = []
    start = time.time()
    for ref in refs:
        eid = ref['example_id']
        folder = dest / eid
        folder.mkdir(parents=True, exist_ok=True)
        target = quality(ref)
        candidates = []
        for i in range(MAX_CANDIDATES):
            eps = next_epsilon(candidates, target, BASE_EPS[a.method])
            cell = folder / f'candidate{i}'
            cell.mkdir(parents=True, exist_ok=True)
            signature = dict(epsilon=eps, seed=a.seed, method=a.method,
                             alpha_ratio=a.alpha_ratio, perceptual_weight=a.perceptual_weight,
                             reference_sha256=hashlib.sha256(Path(ref['adv_path']).read_bytes()).hexdigest(),
                             tolerances=TOLERANCES, channels=CHANNELS, outcome_used=False)
            signature = json.loads(json.dumps(signature))
            spec = cell / 'selection.json'
            if spec.exists() and json.loads(spec.read_text()) != signature:
                raise RuntimeError('Candidate configuration changed on resume')
            save(spec, signature)
            ids = cell / 'ids.txt'
            ids.write_text(eid + '\n')
            run(f'{a.method}_{eid}_{i}', 'reviewer_controls/run_attack.py', '--scenario', 's1', '--split', a.split,
                '--method', a.method, '--epsilon', format(eps, '.17g'), '--alpha-ratio', a.alpha_ratio, '--perceptual-weight', a.perceptual_weight,
                '--seed', a.seed, '--steps', 1000, '--example-ids-file', ids, '--output-root', cell / 'attack', '--resume')
            files = list((cell / 'attack').rglob('manifest.jsonl'))
            if len(files) != 1 or len(read(files[0])) != 1:
                raise ValueError('Incomplete candidate')
            row = read(files[0])[0]
            if Path(row['carrier_ref_path']).read_bytes() != Path(ref['carrier_ref_path']).read_bytes():
                raise ValueError('Methods did not use identical reference PCM')
            q = quality(row)
            score = distance(q, target)
            candidates.append(dict(epsilon=eps, quality=q, distance=score, row=row))
            save(cell / 'quality.json', dict(reference=target, candidate=q, distance=score, passed=score <= 1))
            print(eid, a.method, i, eps, score, flush=True)
            # Same quality-only stopping rule for both waveform arms.
            if score <= 1:
                break
        best = min(candidates, key=lambda c: (c['distance'], c['epsilon']))
        chosen = {**best['row'], 'quality_match_passed': best['distance'] <= 1,
                  'quality_match_max_normalized_gap': best['distance'], 'quality_candidate_count': len(candidates),
                  'quality_selection_uses_attack_outcomes': False}
        selected.append(chosen)
        write_rows(dest / 'final_manifest.jsonl', selected)
        save(dest / 'status.json', dict(completed=len(selected), expected=len(refs), job=os.getenv('SLURM_JOB_ID'), elapsed_s=time.time()-start))
    save(dest / 'gate.json', dict(passed=all(r['quality_match_passed'] for r in selected), n=len(selected),
         matched=sum(r['quality_match_passed'] for r in selected), retained_all_examples=True,
         tolerances=TOLERANCES, channels=CHANNELS, historical_released_controls_included=False,
         candidate_optimizations=sum(r['quality_candidate_count'] for r in selected)))


if __name__ == '__main__':
    main()
