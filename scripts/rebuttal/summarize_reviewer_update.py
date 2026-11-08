"""Recompute completed reviewer results from the shipped metadata, without GPUs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from audit_release import hit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/reviewer_update'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize():
    manifest = json.loads((BASE/'export_manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((ROOT/entry['export']).read_bytes()).hexdigest() == entry['export_sha256'], entry['export']
    groups = defaultdict(list)
    cohorts = {}
    for seed in [1, 7, 42]:
        rows = read(BASE/f's1_tuned_control/seed{seed}/evaluation.jsonl')
        identities = [(r['method'], r['example_id']) for r in rows]
        assert len(set(identities)) == len(rows)
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['method'] == method]
            assert len(selected) == 40
            cohorts[(seed, method)] = {(r['example_id'], r['target']) for r in selected}
            for channel in selected[0]['outcomes']:
                hits = sum(hit(r['target'], r['outcomes'][channel]['output']) for r in selected)
                groups[(method, channel)].append(dict(seed=seed, hits=hits, n=40, asr_pct=100*hits/40))
    assert all(v == next(iter(cohorts.values())) for v in cohorts.values())
    controls = [dict(method=m, channel=c, per_seed=rs, hits=sum(r['hits'] for r in rs), n=120,
                     mean_pct=statistics.mean(r['asr_pct'] for r in rs),
                     population_sd_pp=statistics.pstdev(r['asr_pct'] for r in rs)) for (m,c),rs in sorted(groups.items())]
    transfer = []
    for model in ['qwen25_omni','audio_flamingo','phi4','qwen2_audio','kimi_audio']:
        rows = []
        for path in sorted((BASE/'ensemble_primary').glob(f'*/eps1_seed*/{model}.jsonl')):
            targets = {r['example_id']: r['target'] for r in read(path.with_name('records.jsonl'))}
            for row in read(path):
                for outcome in row['outcomes'].values():
                    actual = hit(targets[row['example_id']], outcome['attacked_output'])
                    assert actual == outcome['attacked_hit'], (path, row['example_id'])
                rows.append(row)
        assert len(rows) == 345
        assert len({(r['scenario'],r['seed'],r['example_id']) for r in rows}) == 345
        for channel in rows[0]['outcomes']:
            transfer.append(dict(model=model, channel=channel, n=len(rows),
                                 hits=sum(r['outcomes'][channel]['attacked_hit'] for r in rows)))
    return dict(s1_controls=controls, ensemble_epsilon1=transfer,
                notes=['S1: 40 recordings repeated across three seeds.',
                       'Ensemble epsilon1: 115 carrier-target pairs repeated across three seeds; task-string hits are not validated decision flips.',
                       'Audio is not distributed; quality fields are recorded measurements, not independently remeasured here.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/reviewer_update_summary.json')
    args=parser.parse_args();report=summarize();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(args.out)


if __name__=='__main__':main()
