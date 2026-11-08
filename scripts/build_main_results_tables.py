"""Rebuild primary tables from seeds 1, 7 and 42 (mean and population SD).

Uses strict complete-target substring matching with lowercase and whitespace
normalization, preserving punctuation. Missing records fail the audit.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def render(rows, scenario):
    lines = [r'\begin{tabular}{lllrr}', r'\toprule',
             r'Model & Method & Channel & Mean ASR (\%) & Seed SD (pp) \\', r'\midrule']
    for row in rows:
        if row['scenario'] != scenario:
            continue
        label = [row[k].replace('_', r'\_') for k in ('model', 'method', 'channel')]
        lines.append(' & '.join(label + [f"{float(row['mean']):.1f}", f"{float(row['population_sd']):.1f}"]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              '% Mean and population SD across seeds 1, 7, 42; seeds repeat the same carriers.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/tables')
    parser.add_argument('--scenario', choices=['all', 'S1', 'S2', 'S3', 'S3a', 'S3b'], default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/rebuttal/audit_release.py'),
                    '--out', str(args.out/'audit')], check=True)
    with (args.out/'audit/aggregated.csv').open() as f:
        rows = list(csv.DictReader(f))
    scenarios = ['S1', 'S2', 'S3a', 'S3b'] if args.scenario == 'all' else (['S3a', 'S3b'] if args.scenario == 'S3' else [args.scenario])
    for scenario in scenarios:
        path = args.out / f'tab_{scenario}_multiseed.tex'
        path.write_text(render(rows, scenario))
        print(path)


if __name__ == '__main__':
    main()
