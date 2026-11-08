"""Frozen, resumable calibration and held-out S1 follow-up stages."""
import argparse, csv, fcntl, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
METHODS = {'tuned_waveform': .06082, 'bandlimited_4khz': .166}

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['calibration','heldout'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--development', type=Path, required=True)
    ap.add_argument('--seed', type=int, choices=[1,7,42], default=1)
    ap.add_argument('--snr-tolerance-db', type=float, default=1.0)
    a = ap.parse_args(); out = a.output.resolve(); dev = a.development.resolve()
    if not math.isfinite(a.snr_tolerance_db) or a.snr_tolerance_db <= 0:
        ap.error('--snr-tolerance-db must be finite and positive')
    work = out/'calibration' if a.stage == 'calibration' else out/f'seed{a.seed}'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase, **kw): save(work/'status.json', dict(phase=phase, time=time.time(), job=os.getenv('SLURM_JOB_ID'), **kw))
    def run(label, script, *args):
        status('running', task=label)
        with (work/(label+'.log')).open('a') as log:
            subprocess.run([sys.executable, '-u', str(ROOT/'scripts'/script), *map(str,args)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    def wave(method, dest, split, ratio, seed, epsmap=None, ids=None):
        args = ['--scenario','s1','--split',split,'--method',method,'--epsilon',METHODS[method], '--alpha-ratio',ratio,'--seed',seed,'--steps',1000,'--output-root',dest,'--resume']
        if epsmap: args += ['--epsilon-map',epsmap]
        if ids: args += ['--example-ids-file',ids]
        run(dest.name+'_'+method, 'reviewer_controls/run_attack.py', *args)
        manifests = list(dest.rglob('manifest.jsonl'))
        if len(manifests) != 1: raise RuntimeError(f'Expected one manifest: {dest}')
        return read(manifests[0])
    def correct(method, current, reference, split, ratio, seed, dest):
        refs = {r['example_id']:r for r in reference}
        cur = {r['example_id']:r for r in current}
        if set(cur) != set(refs): raise RuntimeError('Calibration cohort mismatch')
        # Frozen before held-out scoring: at most three multiplicative distortion-only corrections.
        for iteration in range(1,4):
            bad = [e for e in cur if abs(cur[e]['snr_db']-refs[e]['snr_db'])>a.snr_tolerance_db]
            if not bad: break
            budgets = {e: float(cur[e]['epsilon']) * 10**((cur[e]['snr_db']-refs[e]['snr_db'])/20) for e in bad}
            if any(not math.isfinite(v) or v<=0 or v>10 for v in budgets.values()): raise RuntimeError('Invalid correction budget')
            folder = dest/f'correction{iteration}'; folder.mkdir(parents=True,exist_ok=True)
            epsmap=folder/'epsilon.json'; ids=folder/'ids.txt'
            payload=dict(status='frozen',epsilon_map=budgets,selection_input='SNR only; no model outcome')
            if epsmap.exists() and json.loads(epsmap.read_text())!=payload: raise RuntimeError('Correction changed on resume')
            save(epsmap,payload); ids.write_text('\n'.join(bad)+'\n')
            for r in wave(method,folder/'attack',split,ratio,seed,epsmap,ids): cur[r['example_id']]=r
        rows = list(cur.values())
        gaps = {e:cur[e]['snr_db']-refs[e]['snr_db'] for e in cur}
        write_rows(dest/'final_manifest.jsonl',rows)
        passed = all(math.isfinite(v) and abs(v)<=a.snr_tolerance_db for v in gaps.values())
        save(dest/'verification.json',dict(n=len(rows),gaps_db=gaps,all_within_1db=all(abs(v)<=1 for v in gaps.values()),snr_tolerance_db=a.snr_tolerance_db,passed=passed,retained_all_examples=True))
        return rows, passed
    try:
        chosen=json.loads((dev/'frozen_learning_rates.json').read_text())
        if chosen['heldout_codecs_used'] or chosen['status']!='frozen': raise RuntimeError('Invalid selection')
        ratios={m:v['alpha_over_epsilon'] for m,v in chosen['selected'].items()}
        if a.stage=='calibration':
            state=json.loads((dev/'status.json').read_text())
            if state['phase'] not in ['development_complete','calibration_needs_diagnosis']: raise RuntimeError(f'Development not complete: {state}')
            reference=read(dev/'cells'/f"latent_final_iterate_ar{ratios['s1:latent_final_iterate']:g}"/'manifest.jsonl')
            gates={}
            for method in METHODS:
                paths=list((dev/'calibrated'/method).rglob('manifest.jsonl'))
                if len(paths)!=1: raise RuntimeError('Missing development verification')
                _, gates[method]=correct(method,read(paths[0]),reference,'development',ratios['s1:'+method],1,work/method)
            save(work/'gate.json',dict(passed=all(gates.values()),methods=gates,snr_tolerance_db=a.snr_tolerance_db))
            status('complete' if all(gates.values()) else 'needs_diagnosis',gates=gates)
            if not all(gates.values()): raise RuntimeError(f'Development calibration remains outside {a.snr_tolerance_db:g} dB; held-out stage blocked')
        else:
            gate = json.loads((out/'calibration/gate.json').read_text())
            if gate.get('snr_tolerance_db', 1.0) != a.snr_tolerance_db:
                raise RuntimeError('Development and held-out SNR tolerances differ')
            if not gate['passed']: raise RuntimeError('Development gate did not pass')
            latent=work/'latent'
            run('latent','rebuttal/run_decision_latent.py','--split','evaluation','--alpha',ratios['s1:latent_final_iterate'],'--seed',a.seed,'--output',latent)
            reference=read(latent/'manifest.jsonl')
            if len(reference)!=40: raise RuntimeError('Incomplete held-out latent cohort')
            manifests=[latent/'manifest.jsonl']; gates={}
            for method in METHODS:
                ratio=ratios['s1:'+method]; dest=work/method
                probe=wave(method,dest/'probe','evaluation',ratio,a.seed)
                rows, gates[method]=correct(method,probe,reference,'evaluation',ratio,a.seed,dest)
                manifests.append(dest/'final_manifest.jsonl')
            # Score every retained case even if a held-out distortion match fails; report failures.
            controls=[{**r,'method':'raw_carrier_control','epsilon':0,'alpha':0,'seed':None,'steps':0,'adv_path':r['carrier_ref_path'],'snr_db':None} for r in reference]
            # Infinity is the actual raw-vs-raw SNR; quality tool reports a null value with zero-noise flag.
            for r in controls: r['snr_db']=float('inf')
            write_rows(work/'clean_manifest.jsonl',controls); manifests.append(work/'clean_manifest.jsonl')
            run('delivery','reviewer_controls/evaluate.py','--manifest',*manifests,'--output',work/'evaluation.jsonl','--resume')
            review=[]; keys=[]
            for r in read(work/'evaluation.jsonl'):
                rid=hashlib.sha256(r['record_id'].encode()).hexdigest()[:16]
                review.append(dict(review_id=rid,prompt=r['prompt'],response=r['outcomes']['opus64']['output'],decision='',rationale=''))
                keys.append(dict(review_id=rid,example_id=r['example_id'],seed=r['seed'],method=r['method'],target_match=r['outcomes']['opus64']['match']))
            sheet=work/'decision_annotations.csv'
            if not sheet.exists():
                with sheet.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(sorted(review,key=lambda r:r['review_id']))
            save(work/'annotation_key.json',keys)
            run('quality','rebuttal/measure_quality.py','--manifest',*manifests[:3],'--output',work/'quality.jsonl')
            status('complete',heldout_calibration_gates=gates,annotation='pending',primary_optimizations=120)
    except BaseException as e:
        status('failed',error=repr(e)); raise
if __name__=='__main__': main()
