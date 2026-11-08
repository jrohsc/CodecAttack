"""Bounded development sweep, selection and distortion-only verification.

Runs from a frozen source snapshot. No held-out examples or delivery codecs.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATIOS = (0.02, 0.05, 0.1, 0.2)
CHANNELS = ('clean','opus16','opus24','opus32','opus64','opus128')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    cells = []
    for method,eps in [('latent_final_iterate',1.0),('tuned_waveform',0.06082),('bandlimited_4khz',0.166)]:
        for ratio in RATIOS:
            cell = out/'cells'/f'{method}_ar{ratio:g}'
            if method == 'latent_final_iterate':
                cmd = [sys.executable,str(ROOT/'scripts/rebuttal/run_latent_development.py'),
                       '--alpha',str(ratio),'--output',str(cell)]
                manifest = cell/'manifest.jsonl'
            else:
                cmd = [sys.executable,str(ROOT/'scripts/reviewer_controls/run_attack.py'),
                       '--scenario','s1','--split','development','--method',method,'--epsilon',str(eps),
                       '--alpha-ratio',str(ratio),'--steps','1000','--seed','1','--resume',
                       '--output-root',str(cell)]
                manifest = cell/f's1/development/{method}/eps{eps:g}_ar{ratio:g}_steps1000_seed1/manifest.jsonl'
            cells.append(dict(method=method,epsilon=eps,ratio=ratio,command=cmd,manifest=str(manifest)))
    if args.dry_run:
        print(json.dumps(dict(cells=cells,optimization_runs=120,steps_per_run=1000,
                              calibration_runs=20,selection_channels=CHANNELS),indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec = dict(cells=cells,selection_channels=CHANNELS,source_root=str(ROOT),
                gate='All 10 development examples per waveform arm within 1 dB; otherwise stop for diagnosis',
                heldout_used=False)
    spec_path = out/'campaign_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != json.loads(json.dumps(spec)):
        raise RuntimeError('Campaign specification changed')
    spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    def status(phase, **extra):
        state=dict(phase=phase,updated=time.time(),slurm_job_id=os.getenv('SLURM_JOB_ID'),**extra)
        tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'status.json')
    def run(label,cmd):
        status('running',task=label,command=cmd)
        print(label,flush=True)
        with (out/f'{label}.log').open('a') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    def rows(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    try:
        common_ids=None
        for c in cells:
            run(f'{c["method"]}_ar{c["ratio"]:g}',c['command'])
            records=rows(c['manifest']);ids={r['example_id'] for r in records}
            if len(records)!=10 or len(ids)!=10 or any(r['split']!='development' for r in records):
                raise RuntimeError('Incomplete or invalid development cell')
            if common_ids is None:common_ids=ids
            if ids!=common_ids:raise RuntimeError('Carrier-target cohort mismatch')
        evaluation=out/'development_eval.jsonl'
        run('score_training_channels',[sys.executable,str(ROOT/'scripts/reviewer_controls/evaluate.py'),
            '--manifest',*[c['manifest'] for c in cells],'--channels',*CHANNELS,
            '--output',str(evaluation),'--resume'])
        if len(rows(evaluation))!=120:raise RuntimeError('Incomplete development scoring')
        frozen=out/'frozen_learning_rates.json'
        run('select_learning_rates',[sys.executable,str(ROOT/'scripts/reviewer_controls/select_hparams.py'),
            '--development-eval',str(evaluation),'--output',str(frozen)])
        chosen=json.loads(frozen.read_text())['selected']
        def selected_cell(method):
            ratio=chosen[f's1:{method}']['alpha_over_epsilon']
            return next(c for c in cells if c['method']==method and c['ratio']==ratio)
        reference=selected_cell('latent_final_iterate')['manifest']
        gates={}
        for method in ('tuned_waveform','bandlimited_4khz'):
            c=selected_cell(method);cal=out/f'epsilon_{method}.json'
            command=[sys.executable,str(ROOT/'scripts/reviewer_controls/calibrate_epsilon.py'),
                '--probe',c['manifest'],'--reference',reference,'--output',str(cal)]
            run(f'calibrate_{method}',command)
            if json.loads(cal.read_text())['n_unmatched']!=0:raise RuntimeError('Calibration cohort incomplete')
            verify_root=out/'calibrated'/method
            cmd=c['command'].copy()
            cmd[cmd.index('--output-root')+1]=str(verify_root)
            cmd+=['--epsilon-map',str(cal)]
            run(f'verify_attack_{method}',cmd)
            manifest=verify_root/f's1/development/{method}/eps{c["epsilon"]:g}_ar{c["ratio"]:g}_steps1000_seed1/manifest.jsonl'
            if len(rows(manifest))!=10:raise RuntimeError('Incomplete verification')
            verified=out/f'epsilon_{method}_verified.json'
            command[command.index('--output')+1]=str(verified)
            run(f'verify_snr_{method}',command+['--verify',str(manifest)])
            gates[method]=json.loads(verified.read_text())['verification']['all_within_1db']
        status('development_complete' if all(gates.values()) else 'calibration_needs_diagnosis',
               calibration_gates=gates,heldout_started=False)
    except BaseException as e:
        status('failed',error=repr(e));raise


if __name__=='__main__':main()
